
from __future__ import annotations

import base64
import math
import struct

STD_IMAGE_CAPS = (1568, 1568)
HIRES_IMAGE_CAPS = (4784, 2576)


def image_caps_for_model(model_id: str) -> tuple[int, int]:
    model = (model_id or "").strip().lower().rsplit("/", 1)[-1].replace(".", "-")
    if model.startswith("claude-opus-") and not model.startswith(
        ("claude-opus-4-0", "claude-opus-4-1", "claude-opus-4-5", "claude-opus-4-6")
    ):
        return HIRES_IMAGE_CAPS
    return STD_IMAGE_CAPS


def estimate_image_tokens(width: int, height: int, caps: tuple[int, int]) -> int:
    max_tokens, max_edge = caps
    if width <= 0 or height <= 0:
        return max_tokens
    long_edge = max(width, height)
    if long_edge > max_edge:
        scale = max_edge / long_edge
        width = width * scale
        height = height * scale
    return min(max_tokens, max(1, math.ceil((width * height) / 750)))


def sniff_image_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        width, height = struct.unpack(">II", data[16:24])
        return int(width), int(height)
    if len(data) >= 10 and data[:6] in (b"GIF87a", b"GIF89a"):
        width, height = struct.unpack("<HH", data[6:10])
        return int(width), int(height)
    if len(data) >= 30 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return _sniff_webp(data)
    if len(data) >= 4 and data[:2] == b"\xff\xd8":
        return _sniff_jpeg(data)
    return None


def image_tokens_from_base64(
    encoded: str, caps: tuple[int, int], *, max_decode_bytes: int = 131_072
) -> int:
    prefix = encoded[: (max_decode_bytes // 3) * 4]
    prefix = prefix[: len(prefix) - len(prefix) % 4]
    try:
        dimensions = sniff_image_dimensions(base64.b64decode(prefix, validate=False))
    except (ValueError, TypeError):
        return caps[0]
    if dimensions is None:
        return caps[0]
    return estimate_image_tokens(*dimensions, caps)


def data_url_image_tokens(url: str, caps: tuple[int, int]) -> int:
    marker = "base64,"
    position = url.find(marker)
    if position < 0:
        return caps[0]
    return image_tokens_from_base64(url[position + len(marker) :], caps)


def _sniff_jpeg(data: bytes) -> tuple[int, int] | None:
    i, n = 2, len(data)
    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xD8 or marker == 0xD9 or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if i + 4 > n:
            break
        seg_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            if i + 9 <= n:
                height, width = struct.unpack(">HH", data[i + 5 : i + 9])
                return int(width), int(height)
            return None
        i += 2 + seg_len
    return None


def _sniff_webp(data: bytes) -> tuple[int, int] | None:
    fmt = data[12:16]
    if fmt == b"VP8 ":
        width = struct.unpack("<H", data[26:28])[0] & 0x3FFF
        height = struct.unpack("<H", data[28:30])[0] & 0x3FFF
        return int(width), int(height)
    if fmt == b"VP8L":
        bits = int.from_bytes(data[21:25], "little")
        return int((bits & 0x3FFF) + 1), int(((bits >> 14) & 0x3FFF) + 1)
    if fmt == b"VP8X":
        width = (int.from_bytes(data[24:27], "little") & 0xFFFFFF) + 1
        height = (int.from_bytes(data[27:30], "little") & 0xFFFFFF) + 1
        return int(width), int(height)
    return None
