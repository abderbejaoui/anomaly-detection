"""End-to-end pipeline test on ONE parcel: fetch 5y, train, diagnose.

Run: python -m scripts.train_one
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.diagnose import diagnose_parcel  # noqa: E402
from app.explain import to_response_payload  # noqa: E402
from app.history import fetch_history, load_history, save_history  # noqa: E402
from app.normalize import to_polygon  # noqa: E402
from app.train import train_parcel_model  # noqa: E402


def _pick_first_parcel(file_name: str) -> dict:
    raw = json.loads((ROOT / "data" / file_name).read_text())
    return raw["parcels"][0]


def main() -> int:
    parcel = _pick_first_parcel("parcels_intensif.json")
    parcel_id = parcel["id"]
    system = "intensif"
    poly = to_polygon(parcel)
    print(f"=== Parcel: {parcel_id}  ({parcel.get('name')}) area={parcel['area_ha']:.0f} ha ===")

    end = date(2024, 9, 30)
    start = end - timedelta(days=365 * 5)

    cached = load_history(parcel_id)
    if cached is not None and len(cached) > 50:
        print(f"\n[1/3] Using cached history: {len(cached)} rows")
        history = cached
    else:
        print(f"\n[1/3] Fetching 5y history {start} -> {end}...")
        history = fetch_history(poly, start=start, end=end, system=system)
        print(f"  rows={len(history)}  ndvi_mean={history['ndvi'].mean():.3f}")
        save_history(parcel_id, history)
        print(f"  saved -> history/{parcel_id}.parquet")

    print(f"\n[2/3] Training Ridge per-parcel model...")
    model = train_parcel_model(history, system=system, parcel_id=parcel_id)
    print(f"  n_train={model.meta['n_train']}  residual_std={model.residual_std:.4f}")
    print(f"  coefs:")
    for name, c in zip(model.feature_spec.columns, model.ridge.coef_):
        print(f"    {name:<22s} {c:+.4f}")
    print(f"  intercept: {model.ridge.intercept_:+.4f}")
    print(f"  monthly thresholds (|residual| q66/q90):")
    print(model.monthly_quantiles.to_string(index=False))

    model_path = ROOT / "models" / f"{parcel_id}.joblib"
    model.save(model_path)
    print(f"  saved -> {model_path.relative_to(ROOT)}")

    print(f"\n[3/3] Running diagnostic 'as of' {end}...")
    result = diagnose_parcel(
        poly,
        target_date=end,
        model=model,
        parcel_id=parcel_id,
        system=system,
        tier="cached",
        history_for_baseline=history,
    )
    payload = to_response_payload(result)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
