"""Mock olive zone classifier.

Splits a user-drawn polygon into a regular grid of patches, intersects each
patch with the polygon, and assigns a class (`extensif`, `intensif`, or
`not_olive`) using a deterministic pseudo-random scheme keyed on the patch
position. This keeps the hackathon flow demoable until the real CNN
weights (see OLIVE_CNN_HF_REPO in .env) are wired in.

When the real model is published, replace `classify_patch()` with a call
that uses the Sentinel-2 preview bytes from `zones.sentinel`.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Optional

from shapely.geometry import Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry


CLASS_COLORS: dict[str, str] = {
    "extensif": "#22c55e",
    "intensif": "#ef4444",
    "not_olive": "#6b7280",
}

# Grid resolution used to chop the polygon up. 4x4 = up to 16 patches; the
# polygon mask removes patches that fall outside the drawn area.
_GRID_N = 4

# Average olive class distribution used by the mock classifier.
# Ordered by descending probability; cumulative thresholds applied below.
_CLASS_WEIGHTS = [
    ("extensif", 0.45),
    ("intensif", 0.35),
    ("not_olive", 0.20),
]


def _polygon_from_geojson(geom: dict[str, Any]) -> BaseGeometry:
    if geom.get("type") == "Feature":
        geom = geom["geometry"]
    return shape(geom)


def _haversine_area_ha(poly: BaseGeometry) -> float:
    """Approximate area (hectares) of a lat/lng polygon.

    Uses an equirectangular projection at the polygon centroid — accurate
    enough for the small parcels handled by this endpoint and avoids a
    pyproj dependency.
    """
    if poly.is_empty:
        return 0.0
    c = poly.centroid
    lat_rad = math.radians(c.y)
    m_per_deg_lat = 111_320.0
    m_per_deg_lng = 111_320.0 * math.cos(lat_rad)

    def project(x: float, y: float) -> tuple[float, float]:
        return (x * m_per_deg_lng, y * m_per_deg_lat)

    # Re-project exterior + interiors and compute area in m^2 via shoelace.
    if poly.geom_type == "MultiPolygon":
        return sum(_haversine_area_ha(p) for p in poly.geoms)
    ext = list(poly.exterior.coords)

    def shoelace(coords: list[tuple[float, float]]) -> float:
        s = 0.0
        for i in range(len(coords) - 1):
            x1, y1 = project(*coords[i])
            x2, y2 = project(*coords[i + 1])
            s += x1 * y2 - x2 * y1
        return abs(s) / 2.0

    area_m2 = shoelace(ext)
    for ring in poly.interiors:
        area_m2 -= shoelace(list(ring.coords))
    return max(area_m2, 0.0) / 10_000.0  # m² -> ha


def _seed(*parts: Any) -> float:
    """Return a deterministic float in [0, 1) from arbitrary parts."""
    h = hashlib.md5("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    # Use first 8 hex chars as a 32-bit int.
    return int(h[:8], 16) / 0xFFFFFFFF


def _pick_class(rnd: float) -> str:
    cumulative = 0.0
    for cls, w in _CLASS_WEIGHTS:
        cumulative += w
        if rnd < cumulative:
            return cls
    return _CLASS_WEIGHTS[-1][0]


def classify_polygon(
    polygon_geojson: dict[str, Any],
    image_bytes: Optional[bytes] = None,
    grid_n: int = _GRID_N,
) -> dict[str, Any]:
    """Classify a polygon into colored zones.

    When the Hugging Face model is configured (`OLIVE_CNN_HF_REPO`,
    `OLIVE_CNN_FILENAME`, `OLIVE_CNN_CLASSES` set in env) AND a
    Sentinel-2 preview is available in `image_bytes`, each grid patch
    is scored by the CNN. Otherwise — or on any failure during
    download / load / inference — we fall back to the deterministic
    mock so the demo always produces a result.
    """
    # Late import to keep the mock path dependency-free.
    from . import model as _zone_model  # noqa: WPS433

    use_real_model = bool(image_bytes) and _zone_model.is_available()

    poly = _polygon_from_geojson(polygon_geojson)
    if poly.is_empty:
        return {
            "type": "FeatureCollection",
            "features": [],
            "stats": {
                "total_patches": 0,
                "extensif": 0,
                "intensif": 0,
                "not_olive": 0,
                "surface_totale_ha": 0.0,
            },
        }

    minx, miny, maxx, maxy = poly.bounds
    dx = (maxx - minx) / grid_n
    dy = (maxy - miny) / grid_n

    # Use polygon bounds as part of the seed so the same polygon yields the
    # same classification across calls (deterministic for demos).
    base_seed = (round(minx, 6), round(miny, 6), round(maxx, 6), round(maxy, 6))

    features: list[dict[str, Any]] = []
    counts = {"extensif": 0, "intensif": 0, "not_olive": 0}
    total_area = 0.0

    for i in range(grid_n):
        for j in range(grid_n):
            cell = Polygon(
                [
                    (minx + i * dx, miny + j * dy),
                    (minx + (i + 1) * dx, miny + j * dy),
                    (minx + (i + 1) * dx, miny + (j + 1) * dy),
                    (minx + i * dx, miny + (j + 1) * dy),
                    (minx + i * dx, miny + j * dy),
                ]
            )
            patch = cell.intersection(poly)
            if patch.is_empty or patch.area <= 0:
                continue

            cls: Optional[str] = None
            confidence: Optional[float] = None

            if use_real_model:
                # Map the grid cell into normalised image coords. The
                # Sentinel-2 preview was rendered for the polygon bbox,
                # so cell (i, j) sits at fractional bbox offset
                # (i/N, j/N) → ((i+1)/N, (j+1)/N). PIL uses top-left
                # origin so we flip the y axis.
                left = i / grid_n
                right = (i + 1) / grid_n
                upper = 1.0 - (j + 1) / grid_n
                lower = 1.0 - j / grid_n
                pred = _zone_model.predict_patch(
                    image_bytes,  # type: ignore[arg-type]
                    (left, upper, right, lower),
                )
                if pred is not None:
                    cls, confidence = pred

            if cls is None:
                r1 = _seed(*base_seed, "cls", i, j)
                cls = _pick_class(r1)
                r2 = _seed(*base_seed, "conf", i, j)
                confidence = round(0.70 + 0.29 * r2, 2)  # 0.70 - 0.99
            else:
                confidence = round(float(confidence or 0.0), 2)

            surface_ha = round(_haversine_area_ha(patch), 2)
            total_area += surface_ha
            counts[cls] += 1

            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(patch),
                    "properties": {
                        "class": cls,
                        "color": CLASS_COLORS[cls],
                        "confidence": confidence,
                        "surface_ha": surface_ha,
                    },
                }
            )

    return {
        "type": "FeatureCollection",
        "features": features,
        "stats": {
            "total_patches": len(features),
            "extensif": counts["extensif"],
            "intensif": counts["intensif"],
            "not_olive": counts["not_olive"],
            "surface_totale_ha": round(total_area, 2),
        },
    }
