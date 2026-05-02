"""Smoke test for Google Earth Engine access using a service account."""

import os
import sys
from pathlib import Path

import ee
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

PROJECT_ID = os.environ.get("EE_PROJECT_ID", "anomalydetection-495112")
KEY_PATH = os.environ.get(
    "EE_SERVICE_ACCOUNT_KEY",
    "/Users/abderrahmenbejaoui/Downloads/anomalydetection-495112-7b41d9cac5b2.json",
)


def get_service_account_email(key_path: str) -> str:
    import json

    with open(key_path, "r") as f:
        return json.load(f)["client_email"]


def main() -> int:
    print(f"Project:         {PROJECT_ID}")
    print(f"Key file:        {KEY_PATH}")

    if not os.path.exists(KEY_PATH):
        print(f"ERROR: key file not found at {KEY_PATH}", file=sys.stderr)
        return 1

    sa_email = get_service_account_email(KEY_PATH)
    print(f"Service account: {sa_email}")

    print("\n[1/3] Authenticating with service account...")
    credentials = ee.ServiceAccountCredentials(sa_email, KEY_PATH)
    ee.Initialize(credentials, project=PROJECT_ID)
    print("      OK")

    print("\n[2/3] Trivial server-side compute (1 + 1)...")
    result = ee.Number(1).add(1).getInfo()
    print(f"      Server returned: {result}")
    assert result == 2, "Earth Engine returned an unexpected value"

    print("\n[3/3] Fetching a real image (SRTM elevation at a point)...")
    point = ee.Geometry.Point([10.18, 36.81])  # Tunis
    elevation = (
        ee.Image("USGS/SRTMGL1_003")
        .reduceRegion(reducer=ee.Reducer.first(), geometry=point, scale=30)
        .getInfo()
    )
    print(f"      Elevation at Tunis: {elevation}")

    print("\nAll checks passed. Earth Engine is working with this project + key.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
