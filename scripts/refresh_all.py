"""Train per-parcel models for all EZZAYRA parcels + the global pooled model,
then refresh cached statuses in SQLite.

Resumable: parcels with both a cached history file AND a trained model are skipped
unless --force is passed.

Run: python -m scripts.refresh_all [--force] [--target-date YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from app.db import (  # noqa: E402
    DEFAULT_DB_PATH,
    all_parcels_with_status,
    transaction,
    upsert_status,
)
from app.diagnose import diagnose_parcel  # noqa: E402
from app.explain import to_response_payload  # noqa: E402
from app.history import cache_path, fetch_history, load_history, save_history  # noqa: E402
from app.normalize import to_polygon  # noqa: E402
from app.train import TrainedModel, train_global_model, train_parcel_model  # noqa: E402


HISTORY_DIR = ROOT / "history"
MODELS_DIR = ROOT / "models"
HISTORY_YEARS = 5


def _parcel_polygon(parcel_row: dict):
    return to_polygon({"polygone": parcel_row["polygon_geojson"]})


def _ensure_history_for(parcel_row: dict, *, end: date, force: bool) -> Optional[pd.DataFrame]:
    parcel_id = parcel_row["id"]
    cached = load_history(parcel_id, HISTORY_DIR)
    if cached is not None and len(cached) > 50 and not force:
        return cached

    poly = _parcel_polygon(parcel_row)
    start = end - timedelta(days=365 * HISTORY_YEARS)
    print(f"  fetching 5y NDVI+LST+weather for {parcel_id} ({start} -> {end})...")
    t0 = time.time()
    try:
        history = fetch_history(poly, start=start, end=end, system=parcel_row["system"])
    except Exception as e:
        print(f"    !! fetch failed: {e}")
        return None
    elapsed = time.time() - t0
    if history.empty or len(history) < 30:
        print(f"    !! too few obs ({len(history)} rows) - skipping")
        return None
    save_history(parcel_id, history, HISTORY_DIR)
    print(f"    ok: {len(history)} rows in {elapsed:.1f}s")
    return history


def _train_per_parcel(parcel_row: dict, history: pd.DataFrame) -> Optional[TrainedModel]:
    parcel_id = parcel_row["id"]
    try:
        model = train_parcel_model(
            history, system=parcel_row["system"], parcel_id=parcel_id
        )
    except Exception as e:
        print(f"    !! train failed: {e}")
        return None
    model_path = MODELS_DIR / f"{parcel_id}.joblib"
    model.save(model_path)
    return model


def _diagnose_and_cache(
    parcel_row: dict,
    model: TrainedModel,
    history: pd.DataFrame,
    target: date,
) -> Optional[dict]:
    parcel_id = parcel_row["id"]
    poly = _parcel_polygon(parcel_row)
    try:
        result = diagnose_parcel(
            poly,
            target_date=target,
            model=model,
            parcel_id=parcel_id,
            system=parcel_row["system"],
            tier="cached",
            history_for_baseline=history,
        )
    except Exception as e:
        print(f"    !! diagnose failed: {e}")
        return None
    payload = to_response_payload(result)
    with transaction() as conn:
        upsert_status(
            conn,
            parcel_id=parcel_id,
            statut=result.statut,
            anomaly_score=result.anomaly_score,
            target_date=target.isoformat(),
            response_json=payload,
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-fetch and retrain everything")
    parser.add_argument(
        "--target-date",
        default=None,
        help="diagnostic 'as of' date (YYYY-MM-DD). Defaults to most recent practical date.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process at most N parcels (debug)",
    )
    args = parser.parse_args()

    target = (
        date.fromisoformat(args.target_date)
        if args.target_date
        else date(2024, 9, 30)
    )
    end_for_history = target

    parcels = all_parcels_with_status()
    if args.limit:
        parcels = parcels[: args.limit]
    print(f"Refreshing {len(parcels)} parcels (target_date={target})")

    pooled_records: list[pd.DataFrame] = []
    statuses_summary = {"vert": 0, "orange": 0, "rouge": 0, "skipped": 0}

    for i, p in enumerate(parcels, 1):
        print(f"\n[{i}/{len(parcels)}] {p['id']} ({p['system']}, {p['gouvernorat']})")
        history = _ensure_history_for(p, end=end_for_history, force=args.force)
        if history is None:
            statuses_summary["skipped"] += 1
            continue
        pooled_records.append(history.assign(_parcel_id=p["id"]))

        model_path = MODELS_DIR / f"{p['id']}.joblib"
        if model_path.exists() and not args.force:
            print(f"  loading cached model")
            model = TrainedModel.load(model_path)
        else:
            print(f"  training per-parcel Ridge")
            model = _train_per_parcel(p, history)
            if model is None:
                statuses_summary["skipped"] += 1
                continue

        print(f"  diagnosing as of {target}...")
        payload = _diagnose_and_cache(p, model, history, target)
        if payload is None:
            statuses_summary["skipped"] += 1
        else:
            statuses_summary[payload["statut"]] = statuses_summary.get(payload["statut"], 0) + 1
            print(f"    -> {payload['statut']}  score={payload['anomaly_score']}")

    if pooled_records:
        print(f"\n=== Training global pooled Ridge on {len(pooled_records)} parcels ===")
        pooled = pd.concat(pooled_records, ignore_index=True)
        global_model = train_global_model(pooled)
        global_path = MODELS_DIR / "_global.joblib"
        global_model.save(global_path)
        print(f"  n_train={global_model.meta['n_train']}  residual_std={global_model.residual_std:.4f}")
        print(f"  saved -> {global_path.relative_to(ROOT)}")

    print(f"\nDone. Statuses: {statuses_summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
