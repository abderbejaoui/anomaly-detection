"""Comprehensive validation of the anomaly detector.

We don't have parcel-level stress labels (the problem is unsupervised by design).
Instead we run 5 complementary validations that together build a defensible
picture of the system's accuracy:

  (1) NDVI regression accuracy on a temporal holdout (R^2, RMSE)
       Train each per-parcel Ridge on 2019-2022, predict 2023, measure fit.
  (2) Sensitivity curve via synthetic NDVI drop injection
       At what drop magnitude does the detector switch vert -> orange -> rouge?
  (3) Specificity on a calmer/wetter baseline year (2020-2021)
       False-positive rate on a period the system should NOT flag heavily.
  (4) Tunisia 2023 drought retrospective (THE CRITICAL TEST)
       Trained on pre-2023, run through 2023 weekly, see if alert rate rises
       seasonally during the documented Tunisian drought of 2023.
  (5) Naive baseline comparison
       Same workflow with "flag if NDVI < 0.7 * DOY climatology" — proves Ridge
       adds value over a trivial detector.

All histories are read from history/{id}.parquet (already cached). No GEE calls.
Results are written to evaluation/report.json + printed to stdout.

Run: python -m scripts.evaluate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import mean_squared_error, r2_score  # noqa: E402

from app.db import all_parcels_with_status  # noqa: E402
from app.features import (  # noqa: E402
    PHENOLOGY_COLS,
    THERMAL_COLS,
    WEATHER_COLS,
    add_phenology,
    select_features,
)
from app.train import _residual_quantiles_by_month  # noqa: E402


HISTORY_DIR = ROOT / "history"
EVAL_DIR = ROOT / "evaluation"
EVAL_DIR.mkdir(exist_ok=True)


PER_PARCEL_FEATURES = list(PHENOLOGY_COLS) + list(WEATHER_COLS) + list(THERMAL_COLS)


# ---------------------------------------------------------------------------
# Helpers


def _load_all_histories() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for p in HISTORY_DIR.glob("parcel_*.parquet"):
        if "_smoke" in p.stem:
            continue
        df = pd.read_parquet(p)
        if df.empty or "ndvi" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"])
        out[p.stem] = df
    return out


def _ensure_phenology(df: pd.DataFrame) -> pd.DataFrame:
    if "sin_doy_1" in df.columns:
        return df
    return add_phenology(df)


def _split_history(history: pd.DataFrame, *, holdout_start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(holdout_start)
    train = history[history["date"] < cutoff].copy()
    test = history[history["date"] >= cutoff].copy()
    return train, test


def _fit_ridge(train: pd.DataFrame) -> tuple[Ridge, np.ndarray, np.ndarray, float, pd.DataFrame]:
    """Returns (model, residuals_train, y_train, residual_std, monthly_quantiles)."""
    X = train[PER_PARCEL_FEATURES].astype(float).values
    X = np.nan_to_num(X, nan=0.0)
    y = train["ndvi"].astype(float).values
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    pred = model.predict(X)
    residuals = y - pred
    residual_std = float(np.std(residuals)) or 1e-3
    quantiles = _residual_quantiles_by_month(train, residuals)
    return model, residuals, y, residual_std, quantiles


def _predict(model: Ridge, df: pd.DataFrame) -> np.ndarray:
    X = df[PER_PARCEL_FEATURES].astype(float).values
    X = np.nan_to_num(X, nan=0.0)
    return model.predict(X)


def _classify_window(
    residuals_window: np.ndarray, residual_std: float, quantiles: pd.DataFrame, month: int
) -> tuple[str, float]:
    """Replicates app.diagnose._status_from_score with the directional fix."""
    if len(residuals_window) == 0:
        return "vert", 0.0
    mean_residual = float(np.mean(residuals_window))
    z = mean_residual / max(residual_std, 1e-6)
    row = quantiles.loc[quantiles["month"] == month]
    if row.empty:
        q66, q90 = 1.0, 2.0
    else:
        q66 = float(row["q66"].iloc[0]) / max(residual_std, 1e-6)
        q90 = float(row["q90"].iloc[0]) / max(residual_std, 1e-6)
    if z > -q66:
        statut = "vert"
    elif z > -q90:
        statut = "orange"
    else:
        statut = "rouge"
    return statut, round(z, 3)


# ---------------------------------------------------------------------------
# (1) Regression accuracy on a temporal holdout


def eval_regression_accuracy(histories: dict[str, pd.DataFrame]) -> dict:
    print("\n[1/5] Regression accuracy (train pre-2023, test 2023)...")
    rows = []
    for parcel_id, hist in histories.items():
        hist = _ensure_phenology(hist)
        train, test = _split_history(hist, holdout_start="2023-01-01")
        # Restrict test to 2023 only.
        test = test[test["date"] < pd.Timestamp("2024-01-01")]
        if len(train) < 30 or len(test) < 5:
            continue
        model, _, _, _, _ = _fit_ridge(train)
        y_test = test["ndvi"].astype(float).values
        pred_test = _predict(model, test)
        if len(y_test) < 2:
            continue
        rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))
        r2 = float(r2_score(y_test, pred_test))
        rows.append(
            {
                "parcel_id": parcel_id,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "rmse": rmse,
                "r2": r2,
                "ndvi_mean": float(y_test.mean()),
            }
        )
    df = pd.DataFrame(rows)
    summary = {
        "n_parcels_evaluated": int(len(df)),
        "rmse_mean": float(df["rmse"].mean()),
        "rmse_median": float(df["rmse"].median()),
        "r2_mean": float(df["r2"].mean()),
        "r2_median": float(df["r2"].median()),
        "r2_q25": float(df["r2"].quantile(0.25)),
        "r2_q75": float(df["r2"].quantile(0.75)),
        "r2_min": float(df["r2"].min()),
        "r2_max": float(df["r2"].max()),
        "n_parcels_r2_above_0_5": int((df["r2"] > 0.5).sum()),
        "n_parcels_r2_above_0_3": int((df["r2"] > 0.3).sum()),
    }
    print(f"  parcels={summary['n_parcels_evaluated']}  "
          f"R2 median={summary['r2_median']:.3f}  mean={summary['r2_mean']:.3f}")
    print(f"  RMSE median={summary['rmse_median']:.4f}  mean={summary['rmse_mean']:.4f}")
    print(f"  parcels with R2>0.5: {summary['n_parcels_r2_above_0_5']}/{summary['n_parcels_evaluated']}")
    return {"summary": summary, "per_parcel": rows}


# ---------------------------------------------------------------------------
# (2) Synthetic sensitivity curve


def eval_sensitivity(
    histories: dict[str, pd.DataFrame],
    parcels_meta: dict[str, dict],
    *,
    drops: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40),
) -> dict:
    print("\n[2/5] Sensitivity curve (synthetic NDVI drop injection)...")
    # Use one healthy intensive parcel — they have higher baselines so the curve is cleaner.
    candidates = [
        pid for pid, m in parcels_meta.items()
        if m.get("system") == "intensif"
        and pid in histories
        and len(histories[pid]) > 200
    ]
    if not candidates:
        return {}
    parcel_id = candidates[0]
    hist = _ensure_phenology(histories[parcel_id])
    train, test = _split_history(hist, holdout_start="2024-01-01")
    if len(train) < 30 or len(test) < 5:
        return {}
    model, _, _, residual_std, quantiles = _fit_ridge(train)

    last_window = test.tail(min(15, len(test))).copy()
    half = len(last_window) // 2 or 1
    target_idx = last_window.index[-half:]

    rows = []
    for drop in drops:
        injected = last_window.copy()
        injected.loc[target_idx, "ndvi"] = (
            injected.loc[target_idx, "ndvi"] - drop
        ).clip(lower=0.05)
        y_obs = injected["ndvi"].astype(float).values
        pred = _predict(model, injected)
        residuals = y_obs - pred
        # Use the trailing half of the window (matches diagnose.py logic).
        n = len(residuals)
        tail = residuals[-max(1, n // 2):]
        month = int(injected["date"].iloc[-1].month)
        statut, z = _classify_window(tail, residual_std, quantiles, month)
        rows.append(
            {
                "drop": float(drop),
                "z_score": z,
                "statut": statut,
                "ndvi_observed_mean": float(np.mean(y_obs)),
                "ndvi_expected_mean": float(np.mean(pred)),
            }
        )
        print(f"  drop={drop:+.2f}  z={z:+.3f}  -> {statut}")
    return {
        "parcel_id": parcel_id,
        "n_train": int(len(train)),
        "residual_std": residual_std,
        "curve": rows,
    }


# ---------------------------------------------------------------------------
# (3) Specificity on a calmer baseline year


def eval_specificity(histories: dict[str, pd.DataFrame], baseline_year: int = 2020) -> dict:
    print(f"\n[3/5] Specificity check (false-positive rate on {baseline_year})...")
    counts = {"vert": 0, "orange": 0, "rouge": 0}
    n_eval = 0
    for parcel_id, hist in histories.items():
        hist = _ensure_phenology(hist)
        # Train on everything except the test year.
        train = hist[
            (hist["date"].dt.year != baseline_year)
        ].copy()
        test = hist[hist["date"].dt.year == baseline_year].copy()
        if len(train) < 30 or len(test) < 5:
            continue
        model, _, _, residual_std, quantiles = _fit_ridge(train)
        test_sorted = test.sort_values("date").reset_index(drop=True)
        pred = _predict(model, test_sorted)
        residuals = test_sorted["ndvi"].astype(float).values - pred
        # Diagnose every 21-day rolling window over the test year.
        for i in range(0, len(test_sorted) - 4, 3):
            window_idx = list(range(max(0, i - 4), i + 1))
            tail = residuals[window_idx]
            month = int(test_sorted["date"].iloc[i].month)
            statut, _ = _classify_window(tail, residual_std, quantiles, month)
            counts[statut] += 1
            n_eval += 1
    summary = {
        "year": baseline_year,
        "n_windows_evaluated": n_eval,
        "vert": counts["vert"],
        "orange": counts["orange"],
        "rouge": counts["rouge"],
        "alert_rate_pct": round(100.0 * (counts["orange"] + counts["rouge"]) / max(n_eval, 1), 2),
        "rouge_rate_pct": round(100.0 * counts["rouge"] / max(n_eval, 1), 2),
    }
    print(f"  windows={n_eval}  vert={counts['vert']}  orange={counts['orange']}  rouge={counts['rouge']}")
    print(f"  alert rate (orange+rouge): {summary['alert_rate_pct']}%   rouge rate: {summary['rouge_rate_pct']}%")
    return summary


# ---------------------------------------------------------------------------
# (4) Tunisia 2023 drought retrospective — the critical test


def eval_drought_retrospective(
    histories: dict[str, pd.DataFrame],
    parcels_meta: dict[str, dict],
) -> dict:
    print("\n[4/5] Tunisia 2023 drought retrospective (training on PRE-2023 only)...")
    monthly: dict[int, dict[str, int]] = {m: {"vert": 0, "orange": 0, "rouge": 0} for m in range(1, 13)}
    by_gouv: dict[str, dict[str, int]] = {}
    by_system: dict[str, dict[str, int]] = {"extensif": {"vert": 0, "orange": 0, "rouge": 0},
                                            "intensif": {"vert": 0, "orange": 0, "rouge": 0}}
    n_total = 0

    for parcel_id, hist in histories.items():
        meta = parcels_meta.get(parcel_id, {})
        system = meta.get("system", "?")
        gouv = meta.get("gouvernorat", "?")
        hist = _ensure_phenology(hist)
        train, test = _split_history(hist, holdout_start="2023-01-01")
        test = test[test["date"] < pd.Timestamp("2024-01-01")]
        if len(train) < 30 or len(test) < 5:
            continue
        model, _, _, residual_std, quantiles = _fit_ridge(train)
        test_sorted = test.sort_values("date").reset_index(drop=True)
        pred = _predict(model, test_sorted)
        residuals = test_sorted["ndvi"].astype(float).values - pred
        for i in range(0, len(test_sorted) - 4, 3):
            window_idx = list(range(max(0, i - 4), i + 1))
            tail = residuals[window_idx]
            d = test_sorted["date"].iloc[i]
            statut, _ = _classify_window(tail, residual_std, quantiles, d.month)
            monthly[d.month][statut] += 1
            by_gouv.setdefault(gouv, {"vert": 0, "orange": 0, "rouge": 0})[statut] += 1
            if system in by_system:
                by_system[system][statut] += 1
            n_total += 1

    monthly_summary = []
    for m, c in monthly.items():
        n = sum(c.values())
        rate = round(100.0 * (c["orange"] + c["rouge"]) / max(n, 1), 1)
        monthly_summary.append(
            {"month": m, "n": n, **c, "alert_rate_pct": rate}
        )
    print(f"  total windows: {n_total}")
    print(f"  Alert rate by month (2023, holdout):")
    print(f"    {'mo':>2}  {'n':>4}  {'vert':>5}  {'orange':>6}  {'rouge':>5}  {'alert%':>6}")
    for r in monthly_summary:
        print(f"    {r['month']:>2}  {r['n']:>4}  {r['vert']:>5}  {r['orange']:>6}  "
              f"{r['rouge']:>5}  {r['alert_rate_pct']:>6.1f}")

    print(f"  By gouvernorat (alert%):")
    gouv_summary = []
    for gname, c in sorted(by_gouv.items()):
        n = sum(c.values())
        rate = round(100.0 * (c["orange"] + c["rouge"]) / max(n, 1), 1)
        gouv_summary.append({"gouvernorat": gname, "n": n, **c, "alert_rate_pct": rate})
        print(f"    {gname:<14s}  n={n:>4}  alert={rate:>5.1f}%  (orange={c['orange']}, rouge={c['rouge']})")

    sys_summary = []
    for sname, c in by_system.items():
        n = sum(c.values())
        rate = round(100.0 * (c["orange"] + c["rouge"]) / max(n, 1), 1)
        sys_summary.append({"system": sname, "n": n, **c, "alert_rate_pct": rate})
    print(f"  By system: extensif={sys_summary[0]['alert_rate_pct']}%  intensif={sys_summary[1]['alert_rate_pct']}%")

    return {
        "n_windows": n_total,
        "by_month": monthly_summary,
        "by_gouvernorat": gouv_summary,
        "by_system": sys_summary,
    }


# ---------------------------------------------------------------------------
# (5) Naive baseline comparison: "flag if NDVI < 0.7 * DOY climatology"


def eval_naive_baseline_2023(histories: dict[str, pd.DataFrame]) -> dict:
    print("\n[5/5] Naive baseline (flag if NDVI < 0.7 * DOY mean from pre-2023)...")
    counts = {"flag": 0, "ok": 0}
    n_total = 0
    for parcel_id, hist in histories.items():
        hist = _ensure_phenology(hist)
        train, test = _split_history(hist, holdout_start="2023-01-01")
        test = test[test["date"] < pd.Timestamp("2024-01-01")]
        if len(train) < 30 or len(test) < 5:
            continue
        train["doy"] = train["date"].dt.dayofyear
        clim = train.groupby("doy")["ndvi"].mean().to_dict()

        test_sorted = test.sort_values("date").reset_index(drop=True)
        for i in range(0, len(test_sorted) - 4, 3):
            window_idx = list(range(max(0, i - 4), i + 1))
            window_obs = test_sorted["ndvi"].iloc[window_idx].mean()
            window_doy = test_sorted["date"].iloc[i].dayofyear
            # Use nearest DOY in the climatology dict.
            nearest_doy = min(clim.keys(), key=lambda d: abs(d - window_doy)) if clim else None
            if nearest_doy is None:
                continue
            expected = clim[nearest_doy]
            flag = window_obs < 0.70 * expected
            counts["flag" if flag else "ok"] += 1
            n_total += 1
    rate = round(100.0 * counts["flag"] / max(n_total, 1), 2)
    print(f"  windows={n_total}  flag={counts['flag']}  rate={rate}%")
    return {"n_windows": n_total, **counts, "alert_rate_pct": rate}


# ---------------------------------------------------------------------------
# Main


def main() -> int:
    print("=" * 60)
    print("EZZAYRA Anomaly Detector — Comprehensive Evaluation")
    print("=" * 60)

    histories = _load_all_histories()
    print(f"\nLoaded {len(histories)} cached parcel histories from history/")
    if not histories:
        print("No history files. Run scripts/refresh_all.py first.")
        return 1

    parcels_meta_rows = all_parcels_with_status()
    parcels_meta = {p["id"]: p for p in parcels_meta_rows}

    report: dict = {}
    report["regression_accuracy"] = eval_regression_accuracy(histories)
    report["sensitivity"] = eval_sensitivity(histories, parcels_meta)
    report["specificity_2020"] = eval_specificity(histories, baseline_year=2020)
    report["specificity_2021"] = eval_specificity(histories, baseline_year=2021)
    report["drought_2023"] = eval_drought_retrospective(histories, parcels_meta)
    report["naive_baseline_2023"] = eval_naive_baseline_2023(histories)

    out_path = EVAL_DIR / "report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to {out_path.relative_to(ROOT)}")

    print("\n" + "=" * 60)
    print("HEADLINE NUMBERS FOR THE JURY")
    print("=" * 60)
    rs = report["regression_accuracy"]["summary"]
    sp20 = report["specificity_2020"]
    sp21 = report["specificity_2021"]
    dr = report["drought_2023"]
    nb = report["naive_baseline_2023"]
    print(f"  R^2 (median across parcels, 2023 holdout): {rs['r2_median']:.3f}")
    print(f"  RMSE (median): {rs['rmse_median']:.4f} NDVI units")
    print(f"  Parcels with R^2 > 0.5: {rs['n_parcels_r2_above_0_5']} / {rs['n_parcels_evaluated']}")
    print(f"  Specificity baseline alert rate (2020): {sp20['alert_rate_pct']}%")
    print(f"  Specificity baseline alert rate (2021): {sp21['alert_rate_pct']}%")
    print(f"  2023 drought retrospective alert rate, by month:")
    for r in dr.get("by_month", []):
        if r["n"] == 0:
            continue
        bar = "#" * int(r["alert_rate_pct"] / 5)
        print(f"    {r['month']:>2}  {r['alert_rate_pct']:>5.1f}%  {bar}")
    print(f"  Naive baseline alert rate (2023): {nb['alert_rate_pct']}% (compare to our tier rates above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
