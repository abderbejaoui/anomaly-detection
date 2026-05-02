"""Olive zone classification module.

Separate, self-contained feature added on top of the existing anomaly-detection
service. Exposes POST /api/analyze which takes a user-drawn polygon, fetches
Sentinel-2 imagery (when credentials are available), runs a classifier
(currently a deterministic mock — the real CNN weights will be downloaded
from Hugging Face once published) and returns a colorised GeoJSON
FeatureCollection of sub-zones tagged as `extensif`, `intensif`, or
`not_olive`.

The existing /api/diagnostic-anomalie pipeline is untouched.
"""

from .router import router  # noqa: F401
