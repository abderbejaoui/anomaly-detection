"""Pydantic schemas for the FastAPI service.

The brief specifies a precise input/output shape for POST /api/diagnostic-anomalie:

  Request:
    {
      "oliveraie": {
        "id": "O_2026_307",
        "polygone": <GeoJSON>,
        "systeme": "intensif"
      },
      "date": "2026-07-15"
    }

  Response:
    {
      "statut": "orange",
      "anomaly_score": 2.4,
      "ndvi_observe": [0.42, 0.45, 0.41, 0.37, 0.34],
      "ndvi_attendu": [0.48, 0.50, 0.52, 0.51, 0.50],
      "explication": "...",
      "recommandation": "..."
    }
"""

from __future__ import annotations

from datetime import date as _date_t
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class OliveraieIn(BaseModel):
    """Input parcel description. Accepts both EZZAYRA and brief input shapes.

    At least one of (`polygone`, `polygon`, `geometry`, `coordinates`) must be present.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str = Field(..., description="Identifiant de l'oliveraie")
    name: Optional[str] = None
    systeme: Optional[str] = Field(None, description="extensif | intensif")

    # GeoJSON-style polygon (brief format).
    polygone: Optional[dict[str, Any]] = None
    polygon: Optional[dict[str, Any]] = None
    geometry: Optional[dict[str, Any]] = None
    # EZZAYRA-style coordinates list.
    coordinates: Optional[list[Any]] = None

    def as_normalize_input(self) -> dict[str, Any]:
        """Return a dict the normalize.to_polygon() function accepts."""
        for k in ("polygone", "polygon", "geometry"):
            v = getattr(self, k, None)
            if v:
                return {k: v}
        if self.coordinates:
            return {"coordinates": self.coordinates}
        raise ValueError(
            "Aucun champ de polygone trouvé "
            "(attendu: polygone, polygon, geometry ou coordinates)"
        )


class DiagnosticRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    oliveraie: OliveraieIn
    date: _date_t


class DiagnosticResponse(BaseModel):
    """The exact JSON shape required by the brief, plus a few diagnostic fields prefixed
    with `_` for the dashboard / debugging."""

    model_config = ConfigDict(extra="allow")

    statut: str
    anomaly_score: float
    ndvi_observe: list[float]
    ndvi_attendu: list[float]
    explication: str
    recommandation: str


class ParcelListItem(BaseModel):
    id: str
    name: str
    system: str
    area_ha: float
    gouvernorat: str
    lat_centroid: float
    lng_centroid: float
    polygon_geojson: dict[str, Any]
    statut: Optional[str] = None
    anomaly_score: Optional[float] = None
    target_date: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    gee_initialized: bool
    parcels_loaded: int
    models_trained: int
    statuses_cached: int
