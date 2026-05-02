"""Open-Meteo Archive client for daily rain + temperature, plus rolling features.

Open-Meteo Archive is free, no auth needed, ~80 years of historical reanalysis.
Docs: https://open-meteo.com/en/docs/historical-weather-api
"""

from __future__ import annotations

from datetime import date as _date_t
from typing import Optional

import pandas as pd
import requests

_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_GDD_BASE_C = 10.0  # standard base for olive growing degree days


def fetch_weather(
    lat: float,
    lon: float,
    start: str | _date_t,
    end: str | _date_t,
    *,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Daily rain + tmin + tmax. Returns DataFrame: date, rain_mm, tmin_c, tmax_c, tmean_c."""
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "start_date": str(start),
        "end_date": str(end),
        "daily": "precipitation_sum,temperature_2m_min,temperature_2m_max",
        "timezone": "UTC",
    }
    r = requests.get(_ARCHIVE_URL, params=params, timeout=timeout)
    r.raise_for_status()
    js = r.json()

    daily = js.get("daily") or {}
    dates = daily.get("time") or []
    rain = daily.get("precipitation_sum") or []
    tmin = daily.get("temperature_2m_min") or []
    tmax = daily.get("temperature_2m_max") or []

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "rain_mm": pd.to_numeric(rain, errors="coerce"),
            "tmin_c": pd.to_numeric(tmin, errors="coerce"),
            "tmax_c": pd.to_numeric(tmax, errors="coerce"),
        }
    )
    df["tmean_c"] = (df["tmin_c"] + df["tmax_c"]) / 2.0
    df = df.dropna(subset=["rain_mm", "tmean_c"]).reset_index(drop=True)
    return df


def add_rolling_features(weather: pd.DataFrame) -> pd.DataFrame:
    """Add rain_cum_30d, rain_cum_90d, gdd_cum_30d to a daily weather frame.

    Inputs: DataFrame from fetch_weather. Output: same frame with three new columns.
    """
    df = weather.sort_values("date").set_index("date").copy()
    df["rain_cum_30d"] = df["rain_mm"].rolling("30D", min_periods=1).sum()
    df["rain_cum_90d"] = df["rain_mm"].rolling("90D", min_periods=1).sum()
    gdd = (df["tmean_c"] - _GDD_BASE_C).clip(lower=0.0)
    df["gdd_cum_30d"] = gdd.rolling("30D", min_periods=1).sum()
    return df.reset_index()


def merge_weather_to_dates(
    weather_rolling: pd.DataFrame, target_dates: pd.Series
) -> pd.DataFrame:
    """For each target date, attach the latest available rolling weather features.

    Uses merge_asof so an NDVI observation on day X picks up rolling stats computed up to X.
    """
    target = (
        pd.DataFrame({"date": pd.to_datetime(target_dates)})
        .sort_values("date")
        .reset_index(drop=True)
    )
    src = weather_rolling.sort_values("date").reset_index(drop=True)
    merged = pd.merge_asof(
        target,
        src[["date", "rain_cum_30d", "rain_cum_90d", "gdd_cum_30d", "tmean_c"]],
        on="date",
        direction="backward",
        tolerance=pd.Timedelta(days=2),
    )
    return merged
