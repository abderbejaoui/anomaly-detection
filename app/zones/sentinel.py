"""Sentinel Hub (Copernicus) thin client.

Used by the /api/analyze flow. Credentials come from environment variables
CLIENT_ID / CLIENT_SECRET (set in `.env`). When credentials are missing or
the request fails, the caller falls back to mock imagery — this keeps the
hackathon flow runnable without network access.
"""

from __future__ import annotations

import logging
import os
from datetime import date as _date_t, timedelta
from typing import Any, Optional

import requests

log = logging.getLogger("anomaly.zones.sentinel")

_TOKEN_URL = "https://services.sentinel-hub.com/oauth/token"
_PROCESS_URL = "https://services.sentinel-hub.com/api/v1/process"

_TOKEN_CACHE: dict[str, Any] = {"token": None, "expires_at": 0.0}


def _credentials() -> tuple[Optional[str], Optional[str]]:
    return os.getenv("CLIENT_ID"), os.getenv("CLIENT_SECRET")


def has_credentials() -> bool:
    cid, csec = _credentials()
    return bool(cid and csec)


def _get_token(timeout: float = 10.0) -> Optional[str]:
    """Fetch an OAuth2 access token from Sentinel Hub.

    Returns None on failure so callers can gracefully degrade to mock imagery.
    """
    cid, csec = _credentials()
    if not cid or not csec:
        return None

    import time

    now = time.time()
    cached = _TOKEN_CACHE.get("token")
    if cached and _TOKEN_CACHE.get("expires_at", 0) > now + 30:
        return cached

    try:
        resp = requests.post(
            _TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": cid,
                "client_secret": csec,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        token = data.get("access_token")
        ttl = float(data.get("expires_in", 1800))
        if token:
            _TOKEN_CACHE["token"] = token
            _TOKEN_CACHE["expires_at"] = now + ttl
        return token
    except Exception as e:
        log.warning("Sentinel Hub token request failed: %s", e)
        return None


def fetch_sentinel2_preview(
    polygon_geojson: dict[str, Any],
    target_date: Optional[_date_t] = None,
    width: int = 256,
    height: int = 256,
    timeout: float = 30.0,
) -> Optional[bytes]:
    """Fetch a small Sentinel-2 true-color PNG over the polygon.

    Returns raw PNG bytes, or None if the request couldn't be served.
    The bytes are not used by the mock classifier; this function exists so
    the real CNN can be plugged in later without changing the API surface.
    """
    token = _get_token(timeout=timeout)
    if not token:
        return None

    if target_date is None:
        target_date = _date_t.today()
    date_from = (target_date - timedelta(days=15)).isoformat()
    date_to = target_date.isoformat()

    evalscript = """//VERSION=3
function setup() {
  return {
    input: ["B02","B03","B04"],
    output: { bands: 3 }
  };
}
function evaluatePixel(s) {
  return [2.5*s.B04, 2.5*s.B03, 2.5*s.B02];
}
"""

    body = {
        "input": {
            "bounds": {
                "geometry": polygon_geojson,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{date_from}T00:00:00Z",
                            "to": f"{date_to}T23:59:59Z",
                        },
                        "maxCloudCoverage": 30,
                    },
                }
            ],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [
                {"identifier": "default", "format": {"type": "image/png"}}
            ],
        },
        "evalscript": evalscript,
    }

    try:
        resp = requests.post(
            _PROCESS_URL,
            json=body,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )
        if resp.status_code != 200:
            log.warning(
                "Sentinel Hub process API returned %s: %s",
                resp.status_code,
                resp.text[:300],
            )
            return None
        return resp.content
    except Exception as e:
        log.warning("Sentinel Hub process API request failed: %s", e)
        return None
