"""Demo helper: run a diagnostic with synthetically-stressed NDVI for an EZZAYRA parcel.

Useful for the live demo: even if no real anomaly is happening today, this lets you
deterministically produce a 'rouge' case in front of the jury to show the system
flags real stress correctly.

Run:
  python -m scripts.inject_synthetic_stress [--parcel-id ID] [--drop 0.30] [--date YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from app.db import DEFAULT_DB_PATH, all_parcels_with_status, get_parcel  # noqa: E402
from app.diagnose import diagnose_parcel  # noqa: E402
from app.explain import to_response_payload  # noqa: E402
from app.gee_client import fetch_ndvi_series  # noqa: E402
from app.history import load_history  # noqa: E402
from app.normalize import to_polygon  # noqa: E402
from app.train import TrainedModel  # noqa: E402


def _pick_default_parcel() -> str:
    rows = all_parcels_with_status()
    if not rows:
        raise SystemExit("No parcels in DB. Run scripts/load_parcels.py first.")
    # Prefer an intensive parcel since they have higher baseline NDVI -> a drop is more dramatic.
    intensive = [r for r in rows if r["system"] == "intensif"]
    return (intensive[0] if intensive else rows[0])["id"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parcel-id", default=None)
    ap.add_argument("--drop", type=float, default=0.30,
                    help="fraction to subtract from NDVI in the last 3 weeks (default 0.30)")
    ap.add_argument("--date", default="2024-09-30")
    args = ap.parse_args()

    target = date.fromisoformat(args.date)
    parcel_id = args.parcel_id or _pick_default_parcel()
    p = get_parcel(DEFAULT_DB_PATH, parcel_id)
    if p is None:
        raise SystemExit(f"Parcel {parcel_id} not found in DB.")

    print(f"=== Synthetic stress demo on {parcel_id} ({p['system']}, {p['gouvernorat']}) ===")

    poly = to_polygon({"polygone": p["polygon_geojson"]})
    model_path = ROOT / "models" / f"{parcel_id}.joblib"
    if not model_path.exists():
        raise SystemExit(f"No trained model at {model_path}. Run scripts/refresh_all.py first.")
    model = TrainedModel.load(model_path)
    history = load_history(parcel_id, ROOT / "history")

    fetch_start = target - timedelta(days=42)
    print(f"Fetching real NDVI {fetch_start} -> {target}...")
    real_ndvi = fetch_ndvi_series(poly, fetch_start, target)
    if real_ndvi.empty:
        raise SystemExit("No NDVI observations in window.")

    # Inject the drop on the most recent half of the window.
    cutoff = real_ndvi["date"].iloc[len(real_ndvi) // 2]
    stressed = real_ndvi.copy()
    stressed.loc[stressed["date"] >= cutoff, "ndvi"] = (
        stressed.loc[stressed["date"] >= cutoff, "ndvi"] - args.drop
    ).clip(lower=0.05)
    print(f"Injected -{args.drop:.2f} NDVI on {(stressed['date'] >= cutoff).sum()} obs after {cutoff.date()}")

    print("\n--- Diagnostic on REAL data ---")
    real_result = diagnose_parcel(
        poly, target_date=target, model=model,
        parcel_id=parcel_id, system=p["system"], tier="cached",
        history_for_baseline=history, ndvi_override=real_ndvi,
    )
    print(json.dumps(to_response_payload(real_result), indent=2, ensure_ascii=False))

    print("\n--- Diagnostic on SYNTHETICALLY STRESSED data ---")
    stress_result = diagnose_parcel(
        poly, target_date=target, model=model,
        parcel_id=parcel_id, system=p["system"], tier="cached",
        history_for_baseline=history, ndvi_override=stressed,
    )
    print(json.dumps(to_response_payload(stress_result), indent=2, ensure_ascii=False))

    print(f"\nSummary: real={real_result.statut}/{real_result.anomaly_score} "
          f"stressed={stress_result.statut}/{stress_result.anomaly_score}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
