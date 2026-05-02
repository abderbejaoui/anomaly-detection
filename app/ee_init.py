"""Lazy, idempotent Earth Engine initialization with service account credentials."""

from __future__ import annotations

import json
import os
from pathlib import Path

import ee
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_INITIALIZED = False


def _resolve_key_path() -> str:
    return os.environ.get(
        "EE_SERVICE_ACCOUNT_KEY",
        "/Users/abderrahmenbejaoui/Downloads/anomalydetection-495112-7b41d9cac5b2.json",
    )


def _resolve_project_id() -> str:
    return os.environ.get("EE_PROJECT_ID", "anomalydetection-495112")


def init() -> None:
    """Initialize Earth Engine once per process. Safe to call repeatedly."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    key_path = _resolve_key_path()
    project_id = _resolve_project_id()

    if not Path(key_path).exists():
        raise RuntimeError(
            f"GEE service account key not found at {key_path}. "
            f"Set EE_SERVICE_ACCOUNT_KEY env var or place the JSON there."
        )

    with open(key_path, "r") as f:
        service_account_email = json.load(f)["client_email"]

    credentials = ee.ServiceAccountCredentials(service_account_email, key_path)
    ee.Initialize(credentials, project=project_id)
    _INITIALIZED = True
