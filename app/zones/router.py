"""FastAPI router for the olive zone classification feature.

Exposes:
  GET  /zones                Standalone Leaflet map UI
  POST /api/analyze          Classify a user-drawn polygon into colored zones
  GET  /api/analyze/health   Reports whether Sentinel Hub credentials and
                              the (future) HF model are configured.
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from . import sentinel
from .classifier import classify_polygon
from .schemas import AnalyzeRequest


log = logging.getLogger("anomaly.zones")

router = APIRouter(tags=["zones"])

_executor = ThreadPoolExecutor(max_workers=2)

_STATIC_DIR = Path(__file__).resolve().parent / "static"


@router.get("/zones", response_class=HTMLResponse, include_in_schema=False)
def zones_page() -> Any:
    index = _STATIC_DIR / "index.html"
    if not index.exists():
        return HTMLResponse(
            "<h1>Zones UI not found</h1>"
            "<p>app/zones/static/index.html missing.</p>",
            status_code=200,
        )
    return FileResponse(str(index))


@router.get("/api/analyze/health")
def analyze_health() -> dict[str, Any]:
    return {
        "status": "ok",
        "sentinel_hub_configured": sentinel.has_credentials(),
        "hf_repo": os.getenv("OLIVE_CNN_HF_REPO", ""),
        "hf_filename": os.getenv("OLIVE_CNN_FILENAME", ""),
        "classifier_mode": "mock",
    }


def _run_analyze_sync(req: AnalyzeRequest) -> dict[str, Any]:
    polygon = req.polygon
    if polygon.get("type") == "Feature":
        polygon = polygon.get("geometry") or {}

    if polygon.get("type") != "Polygon":
        raise ValueError(
            f"Expected GeoJSON Polygon, got type={polygon.get('type')!r}"
        )

    image_bytes = None
    if sentinel.has_credentials():
        try:
            image_bytes = sentinel.fetch_sentinel2_preview(
                polygon_geojson=polygon,
                target_date=req.date,
            )
        except Exception as e:  # never block the response on imagery issues
            log.warning("Sentinel-2 fetch failed; continuing with mock: %s", e)

    return classify_polygon(polygon_geojson=polygon, image_bytes=image_bytes)


@router.post("/api/analyze")
async def analyze(req: AnalyzeRequest) -> Any:
    loop = asyncio.get_event_loop()
    try:
        payload = await loop.run_in_executor(_executor, _run_analyze_sync, req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("/api/analyze failed")
        raise HTTPException(status_code=500, detail=f"Analyze failed: {e}")
    return JSONResponse(content=payload)
