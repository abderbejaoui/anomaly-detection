"""Tiny SQLite layer for parcels and cached statuses.

Schemas:
  parcels         (id PK, name, system, area_ha, gouvernorat, lat_centroid, lng_centroid,
                   polygon_geojson, source_file)
  latest_status   (parcel_id PK, statut, anomaly_score, response_json, computed_at,
                   target_date)
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


DEFAULT_DB_PATH = Path("data/parcels.db")


def _connect(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def transaction(db_path: Path | str = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = _connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_schema(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    with transaction(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS parcels (
                id TEXT PRIMARY KEY,
                name TEXT,
                system TEXT NOT NULL,
                area_ha REAL,
                gouvernorat TEXT,
                lat_centroid REAL,
                lng_centroid REAL,
                polygon_geojson TEXT NOT NULL,
                source_file TEXT,
                created_at TEXT,
                modified_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_parcels_system ON parcels(system);
            CREATE INDEX IF NOT EXISTS idx_parcels_gouvernorat ON parcels(gouvernorat);

            CREATE TABLE IF NOT EXISTS latest_status (
                parcel_id TEXT PRIMARY KEY,
                statut TEXT NOT NULL,
                anomaly_score REAL,
                response_json TEXT NOT NULL,
                target_date TEXT NOT NULL,
                computed_at TEXT NOT NULL,
                FOREIGN KEY (parcel_id) REFERENCES parcels(id)
            );
            CREATE INDEX IF NOT EXISTS idx_status ON latest_status(statut);
            """
        )


def upsert_parcel(
    conn: sqlite3.Connection,
    *,
    parcel_id: str,
    name: str,
    system: str,
    area_ha: float,
    gouvernorat: str,
    lat_centroid: float,
    lng_centroid: float,
    polygon_geojson: dict,
    source_file: str,
    created_at: Optional[str] = None,
    modified_at: Optional[str] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO parcels (id, name, system, area_ha, gouvernorat,
                             lat_centroid, lng_centroid, polygon_geojson, source_file,
                             created_at, modified_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            name=excluded.name,
            system=excluded.system,
            area_ha=excluded.area_ha,
            gouvernorat=excluded.gouvernorat,
            lat_centroid=excluded.lat_centroid,
            lng_centroid=excluded.lng_centroid,
            polygon_geojson=excluded.polygon_geojson,
            source_file=excluded.source_file,
            created_at=excluded.created_at,
            modified_at=excluded.modified_at
        """,
        (
            parcel_id,
            name,
            system,
            float(area_ha),
            gouvernorat,
            float(lat_centroid),
            float(lng_centroid),
            json.dumps(polygon_geojson),
            source_file,
            created_at,
            modified_at,
        ),
    )


def upsert_status(
    conn: sqlite3.Connection,
    *,
    parcel_id: str,
    statut: str,
    anomaly_score: float,
    target_date: str,
    response_json: dict,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO latest_status (parcel_id, statut, anomaly_score, response_json,
                                   target_date, computed_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(parcel_id) DO UPDATE SET
            statut=excluded.statut,
            anomaly_score=excluded.anomaly_score,
            response_json=excluded.response_json,
            target_date=excluded.target_date,
            computed_at=excluded.computed_at
        """,
        (
            parcel_id,
            statut,
            float(anomaly_score),
            json.dumps(response_json, ensure_ascii=False),
            target_date,
            now,
        ),
    )


def all_parcels_with_status(db_path: Path | str = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    with transaction(db_path) as conn:
        rows = conn.execute(
            """
            SELECT p.*, s.statut, s.anomaly_score, s.target_date, s.computed_at
            FROM parcels p LEFT JOIN latest_status s ON p.id = s.parcel_id
            ORDER BY p.id
            """
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        d["polygon_geojson"] = json.loads(d["polygon_geojson"])
        out.append(d)
    return out


def get_parcel(db_path: Path | str, parcel_id: str) -> Optional[dict[str, Any]]:
    with transaction(db_path) as conn:
        row = conn.execute("SELECT * FROM parcels WHERE id = ?", (parcel_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["polygon_geojson"] = json.loads(d["polygon_geojson"])
    return d


def get_cached_status(db_path: Path | str, parcel_id: str) -> Optional[dict[str, Any]]:
    with transaction(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM latest_status WHERE parcel_id = ?", (parcel_id,)
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["response_json"] = json.loads(d["response_json"])
    return d
