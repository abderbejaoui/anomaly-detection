"""Normalize the two parcel polygon input formats into a single shapely Polygon.

EZZAYRA dataset format:
    {"coordinates": [{"lat": 35.29, "lng": 10.61}, ...]}

Brief / jury format (GeoJSON):
    {"polygone": {"type": "Polygon", "coordinates": [[[10.61, 35.29], ...]]}}

Both must produce the same shapely Polygon (lng/x, lat/y order).
"""

from __future__ import annotations

from typing import Any

from shapely.geometry import Polygon, shape


class PolygonParseError(ValueError):
    pass


def _from_latlng_objects(coords: list[dict[str, float]]) -> Polygon:
    if len(coords) < 3:
        raise PolygonParseError(
            f"Need at least 3 vertices, got {len(coords)}"
        )
    points = []
    for i, c in enumerate(coords):
        if "lat" not in c or "lng" not in c:
            raise PolygonParseError(
                f"Vertex {i} missing lat/lng: {c!r}"
            )
        points.append((float(c["lng"]), float(c["lat"])))
    return Polygon(points)


def _from_geojson(geom: dict[str, Any]) -> Polygon:
    if "type" not in geom:
        raise PolygonParseError("GeoJSON missing 'type'")
    geom_type = geom["type"]
    if geom_type == "Polygon":
        try:
            return shape(geom)  # shapely understands GeoJSON natively
        except Exception as e:
            raise PolygonParseError(f"Invalid GeoJSON Polygon: {e}") from e
    if geom_type == "Feature":
        return _from_geojson(geom["geometry"])
    if geom_type == "MultiPolygon":
        # Pick the largest polygon — common when olive groves span discontinuous fields.
        multi = shape(geom)
        biggest = max(multi.geoms, key=lambda p: p.area)
        return biggest
    raise PolygonParseError(f"Unsupported GeoJSON type: {geom_type}")


def to_polygon(parcel: dict[str, Any]) -> Polygon:
    """Accept any of the supported input shapes and return a valid shapely Polygon.

    Accepted shapes:
      - {"coordinates": [{"lat", "lng"}, ...]}            # EZZAYRA
      - {"polygone": {GeoJSON Polygon | Feature | ...}}   # jury / brief
      - {"polygon":  {GeoJSON ...}}                       # tolerant alias
      - {"geometry": {GeoJSON ...}}                       # tolerant alias
      - GeoJSON itself: {"type": "Polygon", "coordinates": ...}
    """
    if not isinstance(parcel, dict):
        raise PolygonParseError(
            f"Expected dict, got {type(parcel).__name__}"
        )

    # Direct GeoJSON object passed in.
    if parcel.get("type") in {"Polygon", "MultiPolygon", "Feature"}:
        poly = _from_geojson(parcel)
    elif "polygone" in parcel:
        poly = _from_geojson(parcel["polygone"])
    elif "polygon" in parcel:
        poly = _from_geojson(parcel["polygon"])
    elif "geometry" in parcel and isinstance(parcel["geometry"], dict):
        poly = _from_geojson(parcel["geometry"])
    elif "coordinates" in parcel and isinstance(parcel["coordinates"], list):
        coords = parcel["coordinates"]
        if coords and isinstance(coords[0], dict):
            poly = _from_latlng_objects(coords)
        else:
            # Bare coordinates array — treat as GeoJSON Polygon ring.
            poly = _from_geojson({"type": "Polygon", "coordinates": coords})
    else:
        raise PolygonParseError(
            "No recognized polygon field "
            "(expected one of: coordinates, polygone, polygon, geometry, "
            "or a top-level GeoJSON Polygon)"
        )

    if not poly.is_valid:
        # buffer(0) is the standard shapely trick to fix tiny self-intersections.
        poly = poly.buffer(0)
    if poly.is_empty or poly.area == 0:
        raise PolygonParseError("Resulting polygon is empty or zero-area")
    return poly


def to_geojson_polygon(poly: Polygon) -> dict[str, Any]:
    """Round-trip a shapely Polygon into a GeoJSON dict (lng, lat order)."""
    return {
        "type": "Polygon",
        "coordinates": [list(map(list, poly.exterior.coords))],
    }


def centroid_lonlat(poly: Polygon) -> tuple[float, float]:
    c = poly.centroid
    return (float(c.x), float(c.y))
