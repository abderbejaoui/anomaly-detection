"""Lazy loader + inference for the olive CNN.

Behaviour:
  - On first call we try to download the model file from
    `OLIVE_CNN_HF_REPO` / `OLIVE_CNN_FILENAME`.
  - File extension decides the framework:
      .pt / .pth      -> PyTorch (VGG16-style)
      .h5 / .keras    -> Keras
  - If anything fails (no creds, no network, missing dep, bad weights)
    we set `_DISABLED = True` and the caller falls back to the mock.
  - The class list is read from `OLIVE_CNN_CLASSES` (comma-separated).
    Class names are normalised to one of {extensif, intensif, not_olive}
    via `_normalise_class()` so the colour palette stays consistent
    regardless of what the model was trained on.

This module never raises; `predict_patch()` returns None on failure
and the caller falls back to mock classification.
"""

from __future__ import annotations

import io
import logging
import os
import threading
from typing import Any, Optional

log = logging.getLogger("anomaly.zones.model")


_DEFAULT_CLASSES = ["extensif", "intensif", "not_olive"]

# Mapping from real model labels (e.g. the disease classes in the published
# bassem404x/cnn_olive repo) to the high-level zone classes the UI shows.
# Anything not listed falls through to "not_olive".
_LABEL_NORMALISE = {
    # canonical zone labels
    "extensif": "extensif",
    "intensif": "intensif",
    "not_olive": "not_olive",
    "non_olive": "not_olive",
    "background": "not_olive",
    # common olive-disease labels: a healthy / mild grove ~ extensif,
    # a stressed grove ~ intensif. Best-effort mapping; safe default.
    "healthy": "extensif",
    "rust_mite": "intensif",
    "peacock_spot": "intensif",
}


class _ModelState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.tried = False
        self.disabled = False
        self.framework: Optional[str] = None  # "torch" | "keras"
        self.module: Any = None  # loaded model object
        self.classes: list[str] = []
        self.input_size: tuple[int, int] = (224, 224)
        self.reason: str = ""


_STATE = _ModelState()


def _classes_from_env() -> list[str]:
    raw = os.getenv("OLIVE_CNN_CLASSES", "").strip()
    if not raw:
        return list(_DEFAULT_CLASSES)
    return [c.strip() for c in raw.split(",") if c.strip()]


def _normalise_class(label: str) -> str:
    key = label.strip().lower().replace(" ", "_")
    return _LABEL_NORMALISE.get(key, "not_olive")


def _download_weights() -> Optional[str]:
    repo = os.getenv("OLIVE_CNN_HF_REPO", "").strip()
    filename = os.getenv("OLIVE_CNN_FILENAME", "").strip()
    if not repo or not filename:
        _STATE.reason = "OLIVE_CNN_HF_REPO / OLIVE_CNN_FILENAME not set"
        return None

    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        _STATE.reason = f"huggingface_hub not installed: {e}"
        return None

    token = os.getenv("HF_TOKEN") or None
    try:
        path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            token=token,
        )
        return path
    except Exception as e:
        _STATE.reason = f"hf_hub_download failed: {e}"
        return None


def _load_torch(weights_path: str) -> bool:
    try:
        import torch
    except Exception as e:
        _STATE.reason = f"torch not installed: {e}"
        return False

    try:
        # Allow loading a full saved model OR a state_dict on top of VGG16.
        obj = torch.load(weights_path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older torch without weights_only kwarg
        import torch  # type: ignore[no-redef]

        obj = torch.load(weights_path, map_location="cpu")
    except Exception as e:
        _STATE.reason = f"torch.load failed: {e}"
        return False

    try:
        import torch
        from torchvision import models

        if isinstance(obj, dict):
            n_classes = len(_STATE.classes) or len(_DEFAULT_CLASSES)
            model = models.vgg16(weights=None)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_features, n_classes)
            try:
                model.load_state_dict(obj)
            except Exception:
                # state_dict could be wrapped (e.g. {"state_dict": ...})
                inner = obj.get("state_dict") or obj.get("model") or obj
                model.load_state_dict(inner, strict=False)
        else:
            model = obj  # already a full module

        model.eval()
        _STATE.module = model
        _STATE.framework = "torch"
        return True
    except Exception as e:
        _STATE.reason = f"torch model assembly failed: {e}"
        return False


def _load_keras(weights_path: str) -> bool:
    try:
        # Quiet TF logs.
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        from tensorflow import keras  # type: ignore
    except Exception as e:
        _STATE.reason = f"tensorflow/keras not installed: {e}"
        return False
    try:
        model = keras.models.load_model(weights_path, compile=False)
        _STATE.module = model
        _STATE.framework = "keras"
        return True
    except Exception as e:
        _STATE.reason = f"keras load_model failed: {e}"
        return False


def _ensure_loaded() -> bool:
    """Return True iff a real model is ready for inference."""
    if _STATE.tried:
        return not _STATE.disabled and _STATE.module is not None

    with _STATE.lock:
        if _STATE.tried:
            return not _STATE.disabled and _STATE.module is not None
        _STATE.tried = True
        _STATE.classes = _classes_from_env()

        weights_path = _download_weights()
        if weights_path is None:
            _STATE.disabled = True
            log.info("[zones.model] disabled: %s — using mock classifier", _STATE.reason)
            return False

        ext = os.path.splitext(weights_path)[1].lower()
        ok = False
        if ext in (".pt", ".pth"):
            ok = _load_torch(weights_path)
        elif ext in (".h5", ".keras"):
            ok = _load_keras(weights_path)
        else:
            _STATE.reason = f"unsupported weights extension: {ext}"

        if not ok:
            _STATE.disabled = True
            log.warning(
                "[zones.model] could not load weights (%s) — using mock classifier",
                _STATE.reason,
            )
            return False

        log.info(
            "[zones.model] loaded %s model with classes=%s",
            _STATE.framework,
            _STATE.classes,
        )
        return True


def is_available() -> bool:
    """Cheap check used by health endpoint. Forces a load attempt."""
    return _ensure_loaded()


def status() -> dict[str, Any]:
    return {
        "tried": _STATE.tried,
        "loaded": _STATE.module is not None and not _STATE.disabled,
        "framework": _STATE.framework,
        "classes": _STATE.classes,
        "reason": _STATE.reason,
    }


def _crop_pil_from_bytes(image_bytes: bytes, box: tuple[float, float, float, float]):
    """Crop a fractional bbox (left, upper, right, lower) from raw image bytes.

    Each coordinate is in [0, 1] (top-left origin). Returns a PIL.Image.
    """
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    l = max(0, int(box[0] * w))
    u = max(0, int(box[1] * h))
    r = min(w, int(box[2] * w))
    d = min(h, int(box[3] * h))
    if r <= l or d <= u:
        return img  # degenerate box → use whole image
    return img.crop((l, u, r, d))


def predict_patch(
    image_bytes: bytes,
    box: tuple[float, float, float, float],
) -> Optional[tuple[str, float]]:
    """Run inference on one fractional crop of the polygon preview.

    Returns (zone_class, confidence) or None when the real model is not
    available (caller falls back to mock).
    """
    if not _ensure_loaded():
        return None

    try:
        crop = _crop_pil_from_bytes(image_bytes, box)
    except Exception as e:
        log.debug("[zones.model] crop failed: %s", e)
        return None

    fw = _STATE.framework
    classes = _STATE.classes or _DEFAULT_CLASSES

    try:
        if fw == "torch":
            import torch
            from torchvision import transforms

            tfm = transforms.Compose(
                [
                    transforms.Resize(_STATE.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
            x = tfm(crop).unsqueeze(0)
            with torch.no_grad():
                logits = _STATE.module(x)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            idx = int(probs.argmax())
            label = classes[idx] if idx < len(classes) else "not_olive"
            return _normalise_class(label), float(probs[idx])

        if fw == "keras":
            import numpy as np

            arr = np.asarray(crop.resize(_STATE.input_size)).astype("float32") / 255.0
            arr = np.expand_dims(arr, 0)
            preds = _STATE.module.predict(arr, verbose=0)[0]
            idx = int(preds.argmax())
            label = classes[idx] if idx < len(classes) else "not_olive"
            return _normalise_class(label), float(preds[idx])
    except Exception as e:
        log.warning("[zones.model] inference failed, falling back to mock: %s", e)
        return None

    return None
