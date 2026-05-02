"""Feature engineering for the per-parcel and global Ridge models.

Each row corresponds to one Sentinel-2 NDVI observation for a parcel. The feature columns are:

  Phenology (always computed from the date):
    - sin_doy_1, cos_doy_1, sin_doy_2, cos_doy_2

  Weather (rolling stats up to and including the observation date):
    - rain_cum_30d, rain_cum_90d, gdd_cum_30d, tmean_c

  Thermal (optional, used only when LST is available):
    - lst_c, lst_anomaly_30d  (lst_anomaly_30d = lst_c - climatology_for_DOY)

  System (only used in the global pooled model):
    - system_intensif  (1 if intensif, 0 if extensif. Hyper-intensif absent in dataset.)

Target: ndvi
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


PHENOLOGY_COLS = ["sin_doy_1", "cos_doy_1", "sin_doy_2", "cos_doy_2"]
WEATHER_COLS = ["rain_cum_30d", "rain_cum_90d", "gdd_cum_30d", "tmean_c"]
THERMAL_COLS = ["lst_c", "lst_anomaly_30d"]
SYSTEM_COLS = ["system_intensif"]


@dataclass(frozen=True)
class FeatureSpec:
    """Describes which features a given model expects, in order."""

    columns: tuple[str, ...]
    has_thermal: bool
    has_system: bool

    @property
    def n_features(self) -> int:
        return len(self.columns)


def per_parcel_spec(has_thermal: bool = True) -> FeatureSpec:
    cols = list(PHENOLOGY_COLS) + list(WEATHER_COLS)
    if has_thermal:
        cols += list(THERMAL_COLS)
    return FeatureSpec(tuple(cols), has_thermal, has_system=False)


def global_spec(has_thermal: bool = True) -> FeatureSpec:
    cols = list(PHENOLOGY_COLS) + list(WEATHER_COLS)
    if has_thermal:
        cols += list(THERMAL_COLS)
    cols += list(SYSTEM_COLS)
    return FeatureSpec(tuple(cols), has_thermal, has_system=True)


def add_phenology(df: pd.DataFrame, *, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    doy = pd.to_datetime(out[date_col]).dt.dayofyear.astype(float)
    radians = 2.0 * math.pi * doy / 365.25
    out["sin_doy_1"] = np.sin(radians)
    out["cos_doy_1"] = np.cos(radians)
    out["sin_doy_2"] = np.sin(2.0 * radians)
    out["cos_doy_2"] = np.cos(2.0 * radians)
    return out


def compute_lst_climatology(lst: pd.DataFrame) -> pd.DataFrame:
    """Compute mean LST per day-of-year over the input series. Returns DataFrame[doy, lst_clim]."""
    if lst.empty:
        return pd.DataFrame(columns=["doy", "lst_clim"])
    df = lst.copy()
    df["doy"] = pd.to_datetime(df["date"]).dt.dayofyear
    clim = df.groupby("doy", as_index=False)["lst_c"].mean().rename(
        columns={"lst_c": "lst_clim"}
    )
    return clim


def attach_lst(
    target: pd.DataFrame,
    lst: pd.DataFrame,
    *,
    climatology: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Attach lst_c (nearest 8-day MODIS scene) and lst_anomaly_30d to target dates.

    If climatology is None, it is computed on-the-fly from the provided lst series
    (so for training, pass the full series; for inference, pass the saved climatology).
    """
    out = target.sort_values("date").reset_index(drop=True).copy()
    if lst is None or lst.empty:
        out["lst_c"] = np.nan
        out["lst_anomaly_30d"] = np.nan
        return out

    lst_sorted = lst.sort_values("date").reset_index(drop=True)
    out = pd.merge_asof(
        out,
        lst_sorted[["date", "lst_c"]],
        on="date",
        direction="nearest",
        tolerance=pd.Timedelta(days=12),
    )

    clim = climatology if climatology is not None else compute_lst_climatology(lst_sorted)
    if clim.empty:
        out["lst_anomaly_30d"] = 0.0
        return out

    out["doy"] = pd.to_datetime(out["date"]).dt.dayofyear
    out = out.merge(clim, on="doy", how="left")
    out["lst_anomaly_30d"] = out["lst_c"] - out["lst_clim"]
    out = out.drop(columns=["doy", "lst_clim"])
    return out


def build_feature_frame(
    ndvi: pd.DataFrame,
    weather_aligned: pd.DataFrame,
    lst: Optional[pd.DataFrame],
    *,
    system: Optional[str] = None,
    lst_climatology: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build the model-ready feature frame.

    Args:
      ndvi: DataFrame[date, ndvi] (target)
      weather_aligned: DataFrame[date, rain_cum_30d, rain_cum_90d, gdd_cum_30d, tmean_c]
      lst: DataFrame[date, lst_c] or None
      system: 'extensif' or 'intensif' (used to add the system_* one-hot column)
      lst_climatology: optional precomputed DOY-LST mean (for inference)

    Returns:
      DataFrame with columns: date, ndvi, plus all phenology/weather/thermal/system feature
      columns. Rows with missing critical features (any weather column NaN) are dropped.
    """
    if ndvi is None or ndvi.empty:
        return pd.DataFrame()

    df = ndvi[["date", "ndvi"]].copy()
    df = df.merge(weather_aligned, on="date", how="left")
    df = add_phenology(df)

    if lst is not None and not lst.empty:
        df = attach_lst(df, lst, climatology=lst_climatology)
    else:
        df["lst_c"] = np.nan
        df["lst_anomaly_30d"] = np.nan

    # System one-hot (only used by global model; per-parcel models ignore it).
    if system is not None:
        df["system_intensif"] = 1.0 if system.lower() == "intensif" else 0.0
    else:
        df["system_intensif"] = 0.0

    df = df.dropna(subset=WEATHER_COLS).reset_index(drop=True)
    return df


def select_features(df: pd.DataFrame, spec: FeatureSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) using the columns declared in spec, with NaN-safe imputation.

    Thermal columns may have NaNs (no MODIS scene nearby) — we fill with 0.0 which means
    'no anomaly information'. The intercept absorbs the resulting bias.
    """
    missing = [c for c in spec.columns if c not in df.columns]
    if missing:
        raise KeyError(f"Feature frame missing columns: {missing}")
    X = df[list(spec.columns)].astype(float).values
    if spec.has_thermal:
        X = np.nan_to_num(X, nan=0.0)
    y = df["ndvi"].astype(float).values
    return X, y
