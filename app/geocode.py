"""Lightweight reverse-geocoder for Tunisian gouvernorats.

Approach: pick the nearest gouvernorat seat by great-circle distance from the parcel
centroid. Avoids any network dependency and is good enough for filtering / labelling.

The 24 official gouvernorats with their seat coordinates are embedded inline.
"""

from __future__ import annotations

import math
from typing import Iterable

# (name, latitude, longitude) for each gouvernorat seat.
_GOUVERNORATS: tuple[tuple[str, float, float], ...] = (
    ("Ariana", 36.8625, 10.1956),
    ("Beja", 36.7256, 9.1817),
    ("Ben Arous", 36.7533, 10.2247),
    ("Bizerte", 37.2744, 9.8739),
    ("Gabes", 33.8814, 10.0982),
    ("Gafsa", 34.4256, 8.7842),
    ("Jendouba", 36.5012, 8.7805),
    ("Kairouan", 35.6781, 10.0961),
    ("Kasserine", 35.1675, 8.8362),
    ("Kebili", 33.7058, 8.9714),
    ("Kef", 36.1828, 8.7148),
    ("Mahdia", 35.5047, 11.0623),
    ("Manouba", 36.8101, 9.9756),
    ("Medenine", 33.3548, 10.5055),
    ("Monastir", 35.7780, 10.8262),
    ("Nabeul", 36.4513, 10.7357),
    ("Sfax", 34.7406, 10.7603),
    ("Sidi Bouzid", 35.0381, 9.4859),
    ("Siliana", 36.0844, 9.3708),
    ("Sousse", 35.8254, 10.6411),
    ("Tataouine", 32.9297, 10.4516),
    ("Tozeur", 33.9197, 8.1335),
    ("Tunis", 36.8065, 10.1815),
    ("Zaghouan", 36.4028, 10.1428),
)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def gouvernorat_for(lat: float, lon: float) -> str:
    best_name = "Tunis"
    best_dist = float("inf")
    for name, glat, glon in _GOUVERNORATS:
        d = _haversine_km(lat, lon, glat, glon)
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


def all_gouvernorat_names() -> list[str]:
    return [g[0] for g in _GOUVERNORATS]
