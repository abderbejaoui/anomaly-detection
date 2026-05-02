"""Pydantic schemas for the /api/analyze (zone classification) endpoint."""

from __future__ import annotations

from datetime import date as _date_t
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class AnalyzeRequest(BaseModel):
    """Input to POST /api/analyze.

    Example:
        {
          "polygon": {
            "type": "Polygon",
            "coordinates": [[[10.70, 36.45], ...]]
          },
          "date": "2025-05-15"
        }
    """

    model_config = ConfigDict(extra="ignore")

    polygon: dict[str, Any] = Field(
        ...,
        description="GeoJSON Polygon (or Feature wrapping a Polygon) drawn by the user.",
    )
    date: Optional[_date_t] = Field(
        None,
        description="Acquisition date for Sentinel-2 imagery (defaults to today).",
    )


class ZoneProperties(BaseModel):
    model_config = ConfigDict(extra="allow")

    cls: str = Field(..., alias="class")
    color: str
    confidence: float
    surface_ha: float


class ZoneFeature(BaseModel):
    type: str = "Feature"
    geometry: dict[str, Any]
    properties: ZoneProperties


class AnalyzeStats(BaseModel):
    model_config = ConfigDict(extra="allow")

    total_patches: int
    extensif: int
    intensif: int
    not_olive: int
    surface_totale_ha: float


class AnalyzeResponse(BaseModel):
    """GeoJSON FeatureCollection + summary stats."""

    model_config = ConfigDict(extra="allow")

    type: str = "FeatureCollection"
    features: list[dict[str, Any]]
    stats: AnalyzeStats
