"""Smoke test: load ONE EZZAYRA parcel, fetch a short window of NDVI + LST + weather.

Run: python -m scripts.seed_dev
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.gee_client import fetch_lst_series, fetch_ndvi_series  # noqa: E402
from app.normalize import centroid_lonlat, to_polygon  # noqa: E402
from app.weather import add_rolling_features, fetch_weather, merge_weather_to_dates  # noqa: E402


def _pick_first_parcel() -> dict:
    raw = json.loads((ROOT / "data" / "parcels_intensif.json").read_text())
    return raw["parcels"][0]


def main() -> int:
    parcel = _pick_first_parcel()
    print(f"Parcel: {parcel['id']}  ({parcel.get('name')})  area_ha={parcel['area_ha']:.1f}")

    poly = to_polygon(parcel)
    lon, lat = centroid_lonlat(poly)
    print(f"Centroid: lon={lon:.4f}, lat={lat:.4f}")

    end = date(2024, 9, 30)
    start = end - timedelta(days=180)
    print(f"Window: {start} -> {end}")

    print("\n[1/3] Sentinel-2 NDVI...")
    ndvi = fetch_ndvi_series(poly, start, end)
    print(f"  rows={len(ndvi)}  ndvi_mean={ndvi['ndvi'].mean():.3f}")
    print(ndvi.tail(5).to_string(index=False))

    print("\n[2/3] MODIS LST...")
    lst = fetch_lst_series(poly, start, end)
    print(f"  rows={len(lst)}  lst_mean={lst['lst_c'].mean():.1f} C")
    print(lst.tail(5).to_string(index=False))

    print("\n[3/3] Open-Meteo weather + rolling features...")
    weather = fetch_weather(lat, lon, start - timedelta(days=90), end)
    rolled = add_rolling_features(weather)
    aligned = merge_weather_to_dates(rolled, ndvi["date"])
    print(f"  weather rows={len(weather)}  rolled rows={len(rolled)}  aligned to ndvi={len(aligned)}")
    print(aligned.tail(5).to_string(index=False))

    out = pd.DataFrame({"date": ndvi["date"], "ndvi": ndvi["ndvi"]})
    out = out.merge(aligned, on="date", how="left")
    out["lst_c"] = pd.merge_asof(
        out.sort_values("date"),
        lst.sort_values("date"),
        on="date",
        direction="nearest",
        tolerance=pd.Timedelta(days=8),
    )["lst_c"]

    history_dir = ROOT / "history"
    history_dir.mkdir(exist_ok=True)
    out_path = history_dir / f"{parcel['id']}_smoke.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\nSaved {len(out)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
