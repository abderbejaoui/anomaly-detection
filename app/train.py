"""Train per-parcel and pooled global Ridge models on EZZAYRA history.

A trained model artifact (saved with joblib) holds:
  - ridge: sklearn Ridge regressor
  - feature_spec: FeatureSpec describing the column order
  - residual_std: float, scale used to compute z-scores
  - monthly_quantiles: DataFrame[month, q66, q90] of |residual| per month
  - lst_climatology: DataFrame[doy, lst_clim] used at inference time
  - meta: dict with parcel_id, system, n_train, trained_at, residuals_min/max etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .features import (
    FeatureSpec,
    build_feature_frame,
    compute_lst_climatology,
    global_spec,
    per_parcel_spec,
    select_features,
)


@dataclass
class TrainedModel:
    ridge: Ridge
    feature_spec: FeatureSpec
    residual_std: float
    monthly_quantiles: pd.DataFrame
    lst_climatology: pd.DataFrame
    meta: dict = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.ridge.predict(X)

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path | str) -> "TrainedModel":
        return joblib.load(path)


def _residual_quantiles_by_month(
    df: pd.DataFrame, residuals: np.ndarray
) -> pd.DataFrame:
    """Compute |residual| quantiles per calendar month — used for status thresholds."""
    months = pd.to_datetime(df["date"]).dt.month.values
    abs_res = np.abs(residuals)
    rows = []
    for m in range(1, 13):
        mask = months == m
        if mask.sum() < 4:
            # Not enough data in this month — fall back to global quantiles.
            q66 = float(np.quantile(abs_res, 0.66)) if len(abs_res) else 0.05
            q90 = float(np.quantile(abs_res, 0.90)) if len(abs_res) else 0.10
        else:
            q66 = float(np.quantile(abs_res[mask], 0.66))
            q90 = float(np.quantile(abs_res[mask], 0.90))
        rows.append({"month": m, "q66": q66, "q90": q90})
    return pd.DataFrame(rows)


def train_parcel_model(
    history: pd.DataFrame,
    *,
    system: Optional[str],
    parcel_id: str,
    alpha: float = 1.0,
) -> TrainedModel:
    """Fit a per-parcel Ridge on a history frame.

    Expected history columns (from build_feature_frame): date, ndvi, plus all features.
    """
    spec = per_parcel_spec(has_thermal="lst_c" in history.columns)
    X, y = select_features(history, spec)
    ridge = Ridge(alpha=alpha)
    ridge.fit(X, y)

    pred = ridge.predict(X)
    residuals = y - pred
    residual_std = float(np.std(residuals)) or 1e-3

    quantiles = _residual_quantiles_by_month(history, residuals)

    # We rebuild climatology from the LST in history.
    lst_subset = history[["date", "lst_c"]].dropna() if "lst_c" in history.columns else pd.DataFrame()
    clim = compute_lst_climatology(lst_subset.rename(columns={"lst_c": "lst_c"})) if not lst_subset.empty else pd.DataFrame()

    meta = {
        "parcel_id": parcel_id,
        "system": system,
        "n_train": int(len(y)),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_kind": "per_parcel",
        "alpha": alpha,
        "feature_columns": list(spec.columns),
        "ndvi_mean": float(y.mean()),
        "residual_min": float(residuals.min()),
        "residual_max": float(residuals.max()),
    }

    return TrainedModel(
        ridge=ridge,
        feature_spec=spec,
        residual_std=residual_std,
        monthly_quantiles=quantiles,
        lst_climatology=clim,
        meta=meta,
    )


def train_global_model(
    pooled_history: pd.DataFrame,
    *,
    alpha: float = 1.0,
) -> TrainedModel:
    """Fit the pooled global Ridge across all parcels.

    Expects pooled_history to already contain a system_intensif column AND an `lst_c` column
    (it can have NaNs; select_features will impute with 0).
    """
    spec = global_spec(has_thermal="lst_c" in pooled_history.columns)
    X, y = select_features(pooled_history, spec)
    ridge = Ridge(alpha=alpha)
    ridge.fit(X, y)

    pred = ridge.predict(X)
    residuals = y - pred
    residual_std = float(np.std(residuals)) or 1e-3

    quantiles = _residual_quantiles_by_month(pooled_history, residuals)

    lst_subset = pooled_history[["date", "lst_c"]].dropna() if "lst_c" in pooled_history.columns else pd.DataFrame()
    clim = compute_lst_climatology(lst_subset) if not lst_subset.empty else pd.DataFrame()

    meta = {
        "parcel_id": "_global",
        "system": "pooled",
        "n_train": int(len(y)),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_kind": "global",
        "alpha": alpha,
        "feature_columns": list(spec.columns),
        "ndvi_mean": float(y.mean()),
        "residual_min": float(residuals.min()),
        "residual_max": float(residuals.max()),
    }

    return TrainedModel(
        ridge=ridge,
        feature_spec=spec,
        residual_std=residual_std,
        monthly_quantiles=quantiles,
        lst_climatology=clim,
        meta=meta,
    )
