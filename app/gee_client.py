"""Earth Engine wrappers for Sentinel-2 NDVI and MODIS LST timeseries over a parcel polygon.

Design choices:
- Parcel polygon is buffered INWARD by 10 m to drop edge / soil-contaminated pixels.
- All reductions happen server-side; we only pull a small per-image dictionary back.
- Cloud masking: SCL classes 4 (vegetation), 5 (bare soil), 6 (water), 7 (unclass), 11 (snow)
  are kept; everything else (3 cloud_shadow, 8/9 cloud, 10 cirrus) is dropped.
- Results are returned as plain pandas DataFrames so downstream code doesn't depend on ee.
"""

from __future__ import annotations

from datetime import date as _date_t
from typing import Optional

import ee
import pandas as pd
from shapely.geometry import Polygon

from .ee_init import init as ee_init
from .normalize import to_geojson_polygon


_BUFFER_METERS_INWARD = -10.0
_S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
_MODIS_LST_COLLECTION = "MODIS/061/MOD11A2"
_SCL_KEEP = [4, 5, 6, 7, 11]
_S2_NDVI_SCALE = 20  # m, faster than 10 m and indistinguishable for parcel mean


def _polygon_to_ee(poly: Polygon) -> ee.Geometry:
    gj = to_geojson_polygon(poly)
    return ee.Geometry(gj)


def _buffered_geom(poly: Polygon) -> ee.Geometry:
    geom = _polygon_to_ee(poly)
    buffered = geom.buffer(_BUFFER_METERS_INWARD)
    # If negative buffer collapses the polygon (tiny parcels), fall back to centroid 30m buffer.
    return ee.Algorithms.If(
        buffered.area().gt(100),
        buffered,
        geom.centroid(maxError=1).buffer(30),
    )


def _mask_s2_clouds(image: ee.Image) -> ee.Image:
    scl = image.select("SCL")
    mask = scl.remap(_SCL_KEEP, [1] * len(_SCL_KEEP), 0)
    return image.updateMask(mask)


def fetch_ndvi_series(
    poly: Polygon,
    start: str | _date_t,
    end: str | _date_t,
) -> pd.DataFrame:
    """Pull cloud-masked Sentinel-2 NDVI for the polygon.

    Returns DataFrame with columns: date (datetime64), ndvi (float).
    Rows where the parcel was fully cloudy or had zero valid pixels are dropped.
    """
    ee_init()
    geom = ee.Geometry(_buffered_geom(poly))
    start_s = str(start)
    end_s = str(end)

    collection = (
        ee.ImageCollection(_S2_COLLECTION)
        .filterBounds(geom)
        .filterDate(start_s, end_s)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
        .map(_mask_s2_clouds)
    )

    def _per_image(image: ee.Image) -> ee.Feature:
        ndvi = image.normalizedDifference(["B8", "B4"]).rename("ndvi")
        stats = ndvi.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                reducer2=ee.Reducer.count(), sharedInputs=True
            ),
            geometry=geom,
            scale=_S2_NDVI_SCALE,
            maxPixels=1e9,
            bestEffort=True,
        )
        return ee.Feature(
            None,
            {
                "date": image.date().format("YYYY-MM-dd"),
                "ndvi": stats.get("ndvi_mean"),
                "n_pixels": stats.get("ndvi_count"),
            },
        )

    feats = ee.FeatureCollection(collection.map(_per_image))
    rows = feats.getInfo().get("features", [])

    records = []
    for f in rows:
        props = f["properties"]
        if props.get("ndvi") is None:
            continue
        if props.get("n_pixels") is None or props["n_pixels"] < 5:
            continue
        records.append(
            {
                "date": pd.to_datetime(props["date"]),
                "ndvi": float(props["ndvi"]),
                "n_pixels": int(props["n_pixels"]),
            }
        )

    df = pd.DataFrame.from_records(records, columns=["date", "ndvi", "n_pixels"])
    if df.empty:
        return df
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    # Clip pathological values (water, shadow leaks).
    df["ndvi"] = df["ndvi"].clip(lower=-0.2, upper=1.0)
    return df


def fetch_lst_series(
    poly: Polygon,
    start: str | _date_t,
    end: str | _date_t,
) -> pd.DataFrame:
    """Pull MODIS LST_Day_1km, scaled to degrees Celsius.

    MOD11A2 LST_Day_1km is in Kelvin * 50 (multiply by 0.02, subtract 273.15 for C).
    Returns DataFrame with columns: date, lst_c.
    """
    ee_init()
    geom = ee.Geometry(_buffered_geom(poly))
    start_s = str(start)
    end_s = str(end)

    collection = (
        ee.ImageCollection(_MODIS_LST_COLLECTION)
        .filterBounds(geom)
        .filterDate(start_s, end_s)
        .select(["LST_Day_1km", "QC_Day"])
    )

    def _per_image(image: ee.Image) -> ee.Feature:
        # QC_Day bits 0-1 == 0 means good quality.
        qc = image.select("QC_Day")
        good = qc.bitwiseAnd(3).eq(0)
        lst_c = (
            image.select("LST_Day_1km")
            .multiply(0.02)
            .subtract(273.15)
            .updateMask(good)
            .rename("lst_c")
        )
        stats = lst_c.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=1000,
            maxPixels=1e9,
            bestEffort=True,
        )
        return ee.Feature(
            None,
            {
                "date": image.date().format("YYYY-MM-dd"),
                "lst_c": stats.get("lst_c"),
            },
        )

    feats = ee.FeatureCollection(collection.map(_per_image))
    rows = feats.getInfo().get("features", [])

    records = []
    for f in rows:
        props = f["properties"]
        if props.get("lst_c") is None:
            continue
        records.append(
            {
                "date": pd.to_datetime(props["date"]),
                "lst_c": float(props["lst_c"]),
            }
        )

    df = pd.DataFrame.from_records(records, columns=["date", "lst_c"])
    if df.empty:
        return df
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    return df
