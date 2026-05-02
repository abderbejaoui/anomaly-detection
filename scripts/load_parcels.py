"""Ingest both EZZAYRA JSON files into SQLite.

Tags `system` from the file name (extensif vs intensif), derives `gouvernorat`
from the centroid via the embedded reverse-geocoder, and stores the polygon as
GeoJSON for use by the dashboard.

Run: python -m scripts.load_parcels
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.db import init_schema, transaction, upsert_parcel  # noqa: E402
from app.geocode import gouvernorat_for  # noqa: E402
from app.normalize import centroid_lonlat, to_geojson_polygon, to_polygon  # noqa: E402


_FILES: list[tuple[str, str]] = [
    ("parcels_extensif.json", "extensif"),
    ("parcels_intensif.json", "intensif"),
]


def main() -> int:
    init_schema()
    print("Schema OK.")

    total = 0
    with transaction() as conn:
        for fname, system in _FILES:
            path = ROOT / "data" / fname
            if not path.exists():
                print(f"  ! missing: {path}")
                continue
            raw = json.loads(path.read_text())
            parcels = raw.get("parcels", [])
            print(f"\n[{system}] {fname} -> {len(parcels)} parcels")

            for parcel in parcels:
                try:
                    poly = to_polygon(parcel)
                except Exception as e:
                    print(f"  ! skip {parcel.get('id')}: {e}")
                    continue
                lon, lat = centroid_lonlat(poly)
                gouv = gouvernorat_for(lat, lon)
                upsert_parcel(
                    conn,
                    parcel_id=parcel["id"],
                    name=parcel.get("name") or parcel["id"],
                    system=system,
                    area_ha=float(parcel.get("area_ha") or 0.0),
                    gouvernorat=gouv,
                    lat_centroid=lat,
                    lng_centroid=lon,
                    polygon_geojson=to_geojson_polygon(poly),
                    source_file=fname,
                    created_at=parcel.get("created_at"),
                    modified_at=parcel.get("modified_at"),
                )
                total += 1

    with transaction() as conn:
        rows = conn.execute(
            "SELECT system, gouvernorat, COUNT(*) AS n FROM parcels "
            "GROUP BY system, gouvernorat ORDER BY system, gouvernorat"
        ).fetchall()
    print(f"\nTotal upserted: {total}\nDistribution:")
    for r in rows:
        print(f"  {r['system']:<10s} {r['gouvernorat']:<15s} {r['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
