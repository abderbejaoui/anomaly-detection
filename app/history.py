"""Build / load a full history frame (NDVI + LST + weather + features) for a parcel.

Used by:
  - training (5 years of history per parcel)
  - the on-demand diagnostic API (last ~6 weeks)
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from shapely.geometry import Polygon

from .features import build_feature_frame
from .gee_client import fetch_lst_series, fetch_ndvi_series
from .normalize import centroid_lonlat
from .weather import add_rolling_features, fetch_weather, merge_weather_to_dates


def fetch_history(
    poly: Polygon,
    *,
    start: str | date,
    end: str | date,
    system: Optional[str] = None,
) -> pd.DataFrame:
    """Pull NDVI + LST + weather for a window and return a feature-ready frame.

    The weather window is extended back 90 days so rolling features at `start` are valid.
    """
    start = pd.to_datetime(start).date()
    end = pd.to_datetime(end).date()
    weather_start = start - timedelta(days=90)

    lon, lat = centroid_lonlat(poly)

    ndvi = fetch_ndvi_series(poly, start, end)
    if ndvi.empty:
        return pd.DataFrame()

    lst = fetch_lst_series(poly, start, end)
    weather = fetch_weather(lat, lon, weather_start, end)
    rolled = add_rolling_features(weather)
    aligned_weather = merge_weather_to_dates(rolled, ndvi["date"])

    return build_feature_frame(
        ndvi=ndvi,
        weather_aligned=aligned_weather,
        lst=lst,
        system=system,
    )


def cache_path(parcel_id: str, history_dir: Path | str = "history") -> Path:
    return Path(history_dir) / f"{parcel_id}.parquet"


def save_history(parcel_id: str, df: pd.DataFrame, history_dir: Path | str = "history") -> Path:
    p = cache_path(parcel_id, history_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p


def load_history(parcel_id: str, history_dir: Path | str = "history") -> Optional[pd.DataFrame]:
    p = cache_path(parcel_id, history_dir)
    if not p.exists():
        return None
    return pd.read_parquet(p)
