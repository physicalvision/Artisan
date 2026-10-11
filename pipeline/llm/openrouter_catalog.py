from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from artisan.config import cache_dir

logger = logging.getLogger(__name__)

CATALOG_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_CATALOG_TTL_S = 86_400.0
DEFAULT_CATALOG_TIMEOUT_S = 10.0

CacheMode = Literal["automatic", "explicit", "unknown"]


_EXPLICIT_CACHE_VENDORS = frozenset({"anthropic", "qwen"})
_AUTOMATIC_CACHE_VENDORS = frozenset(
    {"openai", "moonshotai", "x-ai", "deepseek", "groq", "z-ai", "google"}
)


@dataclass(frozen=True, slots=True)
class OpenRouterModelSpec:

    model_id: str
    context_length: int
    max_completion_tokens: int | None
    input_modalities: frozenset[str]
    supported_parameters: frozenset[str]
    prompt_usd_per_m: float | None
    completion_usd_per_m: float | None
    cache_read_usd_per_m: float | None
    cache_write_usd_per_m: float | None


@dataclass(slots=True)
class _CatalogState:

    resolved: bool = False
    entries: dict[str, OpenRouterModelSpec] | None = None


_STATE = _CatalogState()
_LOGGED: set[str] = set()


def _log_once(message: str) -> None:
    if message in _LOGGED:
        return
    _LOGGED.add(message)
    logger.warning("%s", message)


def reset_catalog_cache() -> None:
    _STATE.resolved = False
    _STATE.entries = None
    _LOGGED.clear()


def catalog_cache_path() -> Path:
    override = (os.environ.get("OPENROUTER_CATALOG_CACHE_PATH") or "").strip()
    if override:
        return Path(override).expanduser()

    return cache_dir() / "openrouter_catalog.json"


def model_spec(model_id: str) -> OpenRouterModelSpec | None:
    key = (model_id or "").strip()
    if not key:
        return None
    entries = _catalog()
    if entries is None:
        return None
    spec = entries.get(key)
    if spec is None and ":" in key:
        spec = entries.get(key.split(":", 1)[0])
    if spec is None:
        _log_once(f"openrouter catalog: no entry for model id {key!r}; using caller defaults")
    return spec


def cache_mode(model_id: str) -> CacheMode:
    key = (model_id or "").strip().lower()
    if "/" not in key:
        return "unknown"
    vendor = key.split("/", 1)[0]
    if vendor in _EXPLICIT_CACHE_VENDORS:
        return "explicit"
    if vendor in _AUTOMATIC_CACHE_VENDORS:
        return "automatic"
    return "unknown"


def _catalog() -> dict[str, OpenRouterModelSpec] | None:
    if _STATE.resolved:
        return _STATE.entries
    _STATE.resolved = True
    _STATE.entries = _load_catalog()
    return _STATE.entries


def _load_catalog() -> dict[str, OpenRouterModelSpec] | None:
    path = catalog_cache_path()
    cached = _read_cache_file(path)
    ttl = _env_float("OPENROUTER_CATALOG_TTL_S", DEFAULT_CATALOG_TTL_S)
    if cached is not None and _cache_age_s(path) <= ttl:
        return cached

    payload = _fetch_catalog()
    if payload is not None:
        entries = _parse_catalog(payload)
        if entries:
            _write_cache_file(path, payload)
            return entries
        _log_once("openrouter catalog: /api/v1/models returned no usable rows")

    if cached is not None:
        _log_once(f"openrouter catalog: refresh failed, using stale cache at {path}")
        return cached
    _log_once("openrouter catalog: unavailable (no cache, no network); using caller defaults")
    return None


def _fetch_catalog() -> Any | None:
    if _env_flag("OPENROUTER_CATALOG_OFFLINE"):
        return None
    timeout = _env_float("OPENROUTER_CATALOG_TIMEOUT_S", DEFAULT_CATALOG_TIMEOUT_S)
    request = urllib.request.Request(CATALOG_URL, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        _log_once(f"openrouter catalog: fetch failed ({type(exc).__name__}: {exc})")
        return None


def _read_cache_file(path: Path) -> dict[str, OpenRouterModelSpec] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    entries = _parse_catalog(payload)
    return entries or None


def _write_cache_file(path: Path, payload: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError as exc:
        _log_once(f"openrouter catalog: cache write failed at {path} ({exc})")


def _cache_age_s(path: Path) -> float:
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return float("inf")


def _parse_catalog(payload: Any) -> dict[str, OpenRouterModelSpec]:
    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return {}
    entries: dict[str, OpenRouterModelSpec] = {}
    for row in rows:
        spec = _parse_row(row)
        if spec is not None:
            entries[spec.model_id] = spec
    return entries


def _parse_row(row: Any) -> OpenRouterModelSpec | None:
    if not isinstance(row, dict):
        return None
    model_id = row.get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        return None
    context_length = _as_int(row.get("context_length"))
    top_provider = row.get("top_provider")
    if isinstance(top_provider, dict):
        if context_length is None:
            context_length = _as_int(top_provider.get("context_length"))
        max_completion = _as_int(top_provider.get("max_completion_tokens"))
    else:
        max_completion = None
    if context_length is None or context_length <= 0:
        return None

    architecture = row.get("architecture")
    modalities = architecture.get("input_modalities") if isinstance(architecture, dict) else None
    pricing = row.get("pricing") if isinstance(row.get("pricing"), dict) else {}
    return OpenRouterModelSpec(
        model_id=model_id.strip(),
        context_length=context_length,
        max_completion_tokens=max_completion,
        input_modalities=_as_str_set(modalities),
        supported_parameters=_as_str_set(row.get("supported_parameters")),
        prompt_usd_per_m=_as_rate_per_m(pricing.get("prompt")),
        completion_usd_per_m=_as_rate_per_m(pricing.get("completion")),
        cache_read_usd_per_m=_as_rate_per_m(pricing.get("input_cache_read")),
        cache_write_usd_per_m=_as_rate_per_m(pricing.get("input_cache_write")),
    )


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return None
    return None


def _as_rate_per_m(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return None
    if rate < 0:
        return None
    return rate * 1_000_000.0


def _as_str_set(value: Any) -> frozenset[str]:
    if not isinstance(value, list):
        return frozenset()
    return frozenset(item.strip() for item in value if isinstance(item, str) and item.strip())


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _env_flag(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}
