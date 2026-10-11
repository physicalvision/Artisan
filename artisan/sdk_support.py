from __future__ import annotations


def normalize_sdk_package(value: str) -> str:
    if str(value or "sdk").strip().lower() not in {"sdk", "base"}:
        raise ValueError("Only the Artisan SDK package 'sdk' is supported")
    return "sdk"
