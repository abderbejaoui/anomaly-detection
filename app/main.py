"""FastAPI application — anomaly detection service.

Endpoints:
  GET  /                            Dashboard HTML
  GET  /api/health                  Health + system status
  GET  /api/parcels                 All EZZAYRA parcels with their cached statuses
  GET  /api/parcels/{id}            Single parcel detail
  GET  /api/diagnostic/cached/{id}  Latest cached diagnostic for an EZZAYRA parcel
  POST /api/diagnostic-anomalie     Live diagnostic (jury endpoint)
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date as _date_t
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import db
from .diagnose import diagnose_parcel
from .explain import to_response_payload
from .history import load_history
from .normalize import to_polygon
from .schemas import (
    DiagnosticRequest,
    DiagnosticResponse,
    HealthResponse,
    ParcelListItem,
)
from .train import TrainedModel


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
STATIC_DIR = Path(__file__).resolve().parent / "static"
HISTORY_DIR = ROOT / "history"


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("anomaly")


app = FastAPI(
    title="EZZAYRA Olive Anomaly Detection",
    version="1.0.0",
    description="Détection précoce d'anomalies sur oliveraies (NDVI vs météo).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


_executor = ThreadPoolExecutor(max_workers=4)


def _load_model(parcel_id: str) -> Optional[TrainedModel]:
    p = MODELS_DIR / f"{parcel_id}.joblib"
    if not p.exists():
        return None
    try:
        return TrainedModel.load(p)
    except Exception as e:
        log.error("Failed to load model %s: %s", p, e)
        return None


def _load_global_model() -> Optional[TrainedModel]:
    return _load_model("_global")


# ----------------------------------------------------------------------------
# Static dashboard

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def root_dashboard() -> Any:
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return HTMLResponse(
            "<h1>Dashboard not built yet</h1><p>app/static/index.html missing.</p>",
            status_code=200,
        )
    return FileResponse(str(index))


# ----------------------------------------------------------------------------
# Health


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    parcels = db.all_parcels_with_status()
    n_parcels = len(parcels)
    n_status = sum(1 for p in parcels if p.get("statut"))
    n_models = len(list(MODELS_DIR.glob("*.joblib")))
    try:
        from .ee_init import init as ee_init

        ee_init()
        ee_ok = True
    except Exception as e:
        log.warning("EE init failed at /health: %s", e)
        ee_ok = False
    return HealthResponse(
        status="ok",
        gee_initialized=ee_ok,
        parcels_loaded=n_parcels,
        models_trained=n_models,
        statuses_cached=n_status,
    )


# ----------------------------------------------------------------------------
# Parcels listing


@app.get("/api/parcels", response_model=list[ParcelListItem])
def list_parcels() -> list[ParcelListItem]:
    rows = db.all_parcels_with_status()
    out: list[ParcelListItem] = []
    for r in rows:
        out.append(
            ParcelListItem(
                id=r["id"],
                name=r["name"],
                system=r["system"],
                area_ha=float(r.get("area_ha") or 0.0),
                gouvernorat=r.get("gouvernorat") or "",
                lat_centroid=float(r["lat_centroid"]),
                lng_centroid=float(r["lng_centroid"]),
                polygon_geojson=r["polygon_geojson"],
                statut=r.get("statut"),
                anomaly_score=r.get("anomaly_score"),
                target_date=r.get("target_date"),
            )
        )
    return out


@app.get("/api/parcels/{parcel_id}")
def get_parcel(parcel_id: str) -> dict[str, Any]:
    p = db.get_parcel(db.DEFAULT_DB_PATH, parcel_id)
    if p is None:
        raise HTTPException(status_code=404, detail=f"Parcel {parcel_id} not found")
    cached = db.get_cached_status(db.DEFAULT_DB_PATH, parcel_id)
    if cached:
        p["latest_status"] = cached
    return p


@app.get("/api/diagnostic/cached/{parcel_id}")
def get_cached_diagnostic(parcel_id: str) -> dict[str, Any]:
    cached = db.get_cached_status(db.DEFAULT_DB_PATH, parcel_id)
    if cached is None:
        raise HTTPException(
            status_code=404,
            detail=f"No cached diagnostic for {parcel_id}. Run scripts/refresh_all.py first.",
        )
    return cached["response_json"]


# ----------------------------------------------------------------------------
# Live diagnostic (jury endpoint)


def _run_diagnostic_sync(req: DiagnosticRequest) -> dict[str, Any]:
    parcel_id = req.oliveraie.id
    system = (req.oliveraie.systeme or "").lower() or None

    poly = to_polygon(req.oliveraie.as_normalize_input())

    cached_model = _load_model(parcel_id)
    history = load_history(parcel_id, HISTORY_DIR)
    if cached_model is not None:
        log.info("[Tier 1] Using cached model for %s", parcel_id)
        result = diagnose_parcel(
            poly,
            target_date=req.date,
            model=cached_model,
            parcel_id=parcel_id,
            system=system or cached_model.meta.get("system"),
            tier="cached",
            history_for_baseline=history,
        )
    else:
        global_model = _load_global_model()
        if global_model is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "No model available yet. Run `python -m scripts.refresh_all` first "
                    "to train per-parcel and global models."
                ),
            )
        log.info("[Tier 2] Using GLOBAL model for unknown parcel %s", parcel_id)
        result = diagnose_parcel(
            poly,
            target_date=req.date,
            model=global_model,
            parcel_id=parcel_id,
            system=system,
            tier="global",
            history_for_baseline=None,
        )

    return to_response_payload(result)


@app.post(
    "/api/diagnostic-anomalie",
    response_model=DiagnosticResponse,
    response_model_exclude_none=False,
)
async def diagnostic_anomalie(req: DiagnosticRequest) -> Any:
    loop = asyncio.get_event_loop()
    try:
        payload = await loop.run_in_executor(_executor, _run_diagnostic_sync, req)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("Diagnostic failed")
        raise HTTPException(status_code=500, detail=f"Diagnostic failed: {e}")
    return JSONResponse(content=payload)
