"""Both input formats must produce equivalent polygons."""

from __future__ import annotations

import pytest

from app.normalize import (
    PolygonParseError,
    centroid_lonlat,
    to_geojson_polygon,
    to_polygon,
)


SQUARE_LATLNG = [
    {"lat": 35.0, "lng": 10.0},
    {"lat": 35.0, "lng": 10.5},
    {"lat": 35.5, "lng": 10.5},
    {"lat": 35.5, "lng": 10.0},
]


SQUARE_GEOJSON = {
    "type": "Polygon",
    "coordinates": [
        [[10.0, 35.0], [10.5, 35.0], [10.5, 35.5], [10.0, 35.5], [10.0, 35.0]]
    ],
}


def test_ezzayra_format_parses():
    poly = to_polygon({"coordinates": SQUARE_LATLNG})
    assert poly.is_valid
    assert poly.area == pytest.approx(0.25)


def test_jury_geojson_format_parses():
    poly = to_polygon({"polygone": SQUARE_GEOJSON})
    assert poly.is_valid
    assert poly.area == pytest.approx(0.25)


def test_both_formats_are_equivalent():
    poly_a = to_polygon({"coordinates": SQUARE_LATLNG})
    poly_b = to_polygon({"polygone": SQUARE_GEOJSON})
    assert poly_a.equals(poly_b) or poly_a.symmetric_difference(poly_b).area < 1e-9


def test_alias_polygon_key():
    poly = to_polygon({"polygon": SQUARE_GEOJSON})
    assert poly.area == pytest.approx(0.25)


def test_direct_geojson():
    poly = to_polygon(SQUARE_GEOJSON)
    assert poly.area == pytest.approx(0.25)


def test_centroid_order_is_lng_lat():
    poly = to_polygon({"coordinates": SQUARE_LATLNG})
    lon, lat = centroid_lonlat(poly)
    assert 9.5 < lon < 11.0
    assert 34.5 < lat < 36.0


def test_roundtrip_to_geojson():
    poly = to_polygon({"coordinates": SQUARE_LATLNG})
    gj = to_geojson_polygon(poly)
    assert gj["type"] == "Polygon"
    poly_again = to_polygon(gj)
    assert poly.equals(poly_again) or poly.symmetric_difference(poly_again).area < 1e-9


def test_too_few_vertices_rejected():
    with pytest.raises(PolygonParseError):
        to_polygon({"coordinates": [{"lat": 0, "lng": 0}, {"lat": 1, "lng": 1}]})


def test_missing_polygon_field_rejected():
    with pytest.raises(PolygonParseError):
        to_polygon({"foo": "bar"})


def test_real_ezzayra_parcel_loads():
    """Spot check against an actual EZZAYRA parcel layout."""
    import json
    from pathlib import Path

    p = Path("data/parcels_extensif.json")
    if not p.exists():
        pytest.skip("EZZAYRA data file not present")
    raw = json.loads(p.read_text())
    first = raw["parcels"][0]
    poly = to_polygon(first)
    assert poly.is_valid
    lon, lat = centroid_lonlat(poly)
    # Tunisia bounding box sanity check.
    assert 7.0 < lon < 12.0
    assert 30.0 < lat < 38.0
