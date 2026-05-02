"""Core anomaly diagnostic logic.

Given a parcel polygon and a target date, produces the JSON response that the
brief specifies:

  {
    "statut": "vert" | "orange" | "rouge",
    "anomaly_score": float,
    "ndvi_observe": [5 floats, oldest to newest],
    "ndvi_attendu": [5 floats, oldest to newest],
    "explication": "...",
    "recommandation": "..."
  }
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from .features import build_feature_frame, select_features
from .gee_client import fetch_lst_series, fetch_ndvi_series
from .normalize import centroid_lonlat
from .train import TrainedModel
from .weather import add_rolling_features, fetch_weather, merge_weather_to_dates


@dataclass
class DiagnosticResult:
    statut: str
    anomaly_score: float
    ndvi_observe: list[float]
    ndvi_attendu: list[float]
    dates: list[str]
    feature_contributions: dict[str, float]
    weather_summary: dict[str, float]
    tier: str  # 'cached' or 'global'
    parcel_id: str
    system: Optional[str]


# ----------------------------------------------------------------------------
# Status thresholds


def _status_from_score(
    score: float, monthly_quantiles: pd.DataFrame, month: int
) -> str:
    """Map z score to vert / orange / rouge.

    Stress is a NEGATIVE residual (NDVI below expected). A positive residual means the
    parcel is doing better than the model expected — that is not a stress signal and
    must not trigger an alert.

    Rule:
      score >  -q66  -> vert       (within normal band, including positive surprises)
      -q66 >= score > -q90  -> orange
      score <= -q90  -> rouge
    """
    row = monthly_quantiles.loc[monthly_quantiles["month"] == month]
    if row.empty:
        q66, q90 = 1.0, 2.0
    else:
        q66 = float(row["q66"].iloc[0])
        q90 = float(row["q90"].iloc[0])
    if score > -q66:
        return "vert"
    if score > -q90:
        return "orange"
    return "rouge"


def _normalize_thresholds(model: TrainedModel) -> pd.DataFrame:
    """Return monthly quantiles expressed in z-score units (|residual| / std)."""
    q = model.monthly_quantiles.copy()
    std = model.residual_std or 1e-3
    q["q66"] = q["q66"] / std
    q["q90"] = q["q90"] / std
    return q


# ----------------------------------------------------------------------------
# Resampling to the brief's expected 5-point output


def _resample_to_n_points(
    df: pd.DataFrame, *, n: int, value_col: str
) -> tuple[list[str], list[float]]:
    """Pick n weekly representative points by binning the trailing window into n equal slots."""
    if df.empty:
        return [], []
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) == 0:
        return [], []

    earliest = df["date"].min()
    latest = df["date"].max()
    span = (latest - earliest).days
    if span <= 0 or len(df) == 1:
        # Single-point fallback: repeat.
        d = df["date"].iloc[-1].strftime("%Y-%m-%d")
        v = float(df[value_col].iloc[-1])
        return [d] * n, [v] * n

    edges = pd.date_range(earliest, latest, periods=n + 1)
    dates_out: list[str] = []
    values_out: list[float] = []
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        mask = (df["date"] >= lo) & (df["date"] <= hi)
        sub = df[mask]
        if sub.empty:
            # Find nearest in time.
            nearest_idx = (df["date"] - (lo + (hi - lo) / 2)).abs().idxmin()
            row = df.loc[nearest_idx]
        else:
            row = sub.iloc[len(sub) // 2]
        dates_out.append(pd.Timestamp(row["date"]).strftime("%Y-%m-%d"))
        values_out.append(round(float(row[value_col]), 3))
    return dates_out, values_out


# ----------------------------------------------------------------------------
# Feature contribution decomposition (used for explanations)


def _compute_contributions(
    model: TrainedModel,
    X_obs_mean: np.ndarray,
    X_history_mean: np.ndarray,
) -> dict[str, float]:
    """For each feature, contribution = coef * (x_obs - x_history_mean).

    Positive contribution -> the feature pushed expected NDVI UP relative to baseline.
    A drop in NDVI that the model fails to explain is captured by the residual itself.
    """
    coefs = model.ridge.coef_
    delta = X_obs_mean - X_history_mean
    contribs = coefs * delta
    return {
        col: float(contribs[i])
        for i, col in enumerate(model.feature_spec.columns)
    }


# ----------------------------------------------------------------------------
# Main entry point


def diagnose_parcel(
    poly: Polygon,
    target_date: date,
    model: TrainedModel,
    *,
    parcel_id: str,
    system: Optional[str],
    tier: str,
    history_for_baseline: Optional[pd.DataFrame] = None,
    ndvi_override: Optional[pd.DataFrame] = None,
) -> DiagnosticResult:
    """Execute the full diagnostic for a parcel.

    Args:
      poly: shapely polygon (parcel geometry)
      target_date: diagnose 'as of' this date — fetches the 6 weeks ending here
      model: a TrainedModel instance (per-parcel or global)
      parcel_id, system: used in the response payload
      tier: 'cached' or 'global'
      history_for_baseline: optional pre-loaded history (used for explanation deltas).
        If None, defaults to a synthetic baseline of feature means = 0.
      ndvi_override: optional precomputed NDVI frame (date, ndvi). Used by the
        synthetic-stress demo to inject anomalies for testing.
    """
    target = pd.to_datetime(target_date).date()
    fetch_start = target - timedelta(days=42)  # 6 weeks
    weather_start = fetch_start - timedelta(days=90)
    lon, lat = centroid_lonlat(poly)

    if ndvi_override is not None and not ndvi_override.empty:
        ndvi = ndvi_override[
            (ndvi_override["date"] >= pd.Timestamp(fetch_start))
            & (ndvi_override["date"] <= pd.Timestamp(target))
        ].copy()
    else:
        ndvi = fetch_ndvi_series(poly, fetch_start, target)

    if ndvi.empty:
        raise ValueError(
            f"No Sentinel-2 NDVI observations in {fetch_start}..{target} "
            f"for parcel {parcel_id} (cloud cover or area too small)."
        )

    lst = fetch_lst_series(poly, fetch_start, target)
    weather = fetch_weather(lat, lon, weather_start, target)
    rolled = add_rolling_features(weather)
    aligned_weather = merge_weather_to_dates(rolled, ndvi["date"])

    # Build the inference feature frame using the model's saved LST climatology.
    # build_feature_frame computes climatology from the 'lst' arg unless we pre-attach.
    # Since we're at inference, we want to use the model's stored climatology.
    feature_df = build_feature_frame(
        ndvi=ndvi,
        weather_aligned=aligned_weather,
        lst=lst,
        system=system,
        lst_climatology=model.lst_climatology if not model.lst_climatology.empty else None,
    )
    if feature_df.empty:
        raise ValueError(f"Could not build feature frame for parcel {parcel_id}")

    X, y_obs = select_features(feature_df, model.feature_spec)
    y_exp = model.predict(X)
    residuals = y_obs - y_exp

    # 3-week rolling average residual (actually 'last half of the 6-week window').
    n = len(residuals)
    tail = residuals[-max(1, n // 2):]
    mean_residual = float(np.mean(tail))
    score = mean_residual / max(model.residual_std, 1e-6)

    # Status from monthly quantiles (in z units).
    z_thresholds = _normalize_thresholds(model)
    statut = _status_from_score(score, z_thresholds, month=target.month)

    # Resample to 5 points each.
    dates_obs, ndvi_obs5 = _resample_to_n_points(
        pd.DataFrame({"date": feature_df["date"], "ndvi": y_obs}),
        n=5,
        value_col="ndvi",
    )
    _, ndvi_exp5 = _resample_to_n_points(
        pd.DataFrame({"date": feature_df["date"], "ndvi": y_exp}),
        n=5,
        value_col="ndvi",
    )

    # Feature contributions for explanation.
    if history_for_baseline is not None and not history_for_baseline.empty:
        Xh, _ = select_features(history_for_baseline, model.feature_spec)
        history_mean = Xh.mean(axis=0)
    else:
        history_mean = np.zeros(model.feature_spec.n_features)
    obs_mean = X.mean(axis=0)
    contributions = _compute_contributions(model, obs_mean, history_mean)

    # Weather summary for explanations.
    weather_summary = {
        "rain_cum_30d": float(np.nanmean(feature_df["rain_cum_30d"])),
        "rain_cum_90d": float(np.nanmean(feature_df["rain_cum_90d"])),
        "tmean_c": float(np.nanmean(feature_df["tmean_c"])),
    }
    if "lst_c" in feature_df.columns and feature_df["lst_c"].notna().any():
        weather_summary["lst_c"] = float(np.nanmean(feature_df["lst_c"]))
        weather_summary["lst_anomaly_30d"] = float(np.nanmean(feature_df["lst_anomaly_30d"]))

    return DiagnosticResult(
        statut=statut,
        anomaly_score=round(float(score), 2),
        ndvi_observe=ndvi_obs5,
        ndvi_attendu=ndvi_exp5,
        dates=dates_obs,
        feature_contributions=contributions,
        weather_summary=weather_summary,
        tier=tier,
        parcel_id=parcel_id,
        system=system,
    )
