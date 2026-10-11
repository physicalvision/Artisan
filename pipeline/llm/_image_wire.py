
from __future__ import annotations

import io
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

JPEG_QUALITY_ENV = "ARTISAN_IMAGE_WIRE_JPEG_QUALITY"
MIN_TRANSCODE_BYTES_ENV = "ARTISAN_IMAGE_WIRE_MIN_BYTES"

DEFAULT_JPEG_QUALITY = 90
DEFAULT_MIN_TRANSCODE_BYTES = 128 * 1024

_TRANSCODABLE_SUFFIXES = frozenset({".png", ".tif", ".tiff", ".bmp"})


def _jpeg_quality() -> int:
    raw = (os.environ.get(JPEG_QUALITY_ENV) or "").strip()
    if not raw:
        return DEFAULT_JPEG_QUALITY
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring unsupported %s=%r; expected an integer", JPEG_QUALITY_ENV, raw)
        return DEFAULT_JPEG_QUALITY


def _min_transcode_bytes() -> int:
    raw = (os.environ.get(MIN_TRANSCODE_BYTES_ENV) or "").strip()
    if not raw:
        return DEFAULT_MIN_TRANSCODE_BYTES
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_MIN_TRANSCODE_BYTES


def image_bytes_for_wire(path: Path, raw: bytes, mime_type: str) -> tuple[bytes, str]:
    quality = _jpeg_quality()
    if quality <= 0:
        return raw, mime_type
    if path.suffix.lower() not in _TRANSCODABLE_SUFFIXES:
        return raw, mime_type
    if len(raw) < _min_transcode_bytes():
        return raw, mime_type
    try:
        from PIL import Image

        with Image.open(io.BytesIO(raw)) as im:
            if im.mode in {"RGBA", "LA", "PA"}:
                alpha = im.getchannel("A")
                if alpha.getextrema()[0] != 255:
                    return raw, mime_type
            buf = io.BytesIO()
            im.convert("RGB").save(buf, "JPEG", quality=quality, optimize=True)
    except Exception as exc:  # noqa: BLE001 - never fail a request over an optimisation
        logger.warning("image wire transcode failed for %s (%s); sending original", path.name, exc)
        return raw, mime_type

    encoded = buf.getvalue()
    if len(encoded) >= len(raw):
        return raw, mime_type
    return encoded, "image/jpeg"
