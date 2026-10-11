from __future__ import annotations

import os
from typing import Any


def build_provider_routing(model_id: str) -> dict[str, Any]:
    routing: dict[str, Any] = {}
    required = _env_flag_opt("OPENROUTER_REQUIRE_PARAMETERS")
    if required is not False:
        routing["require_parameters"] = True
    for field, name in (
        ("only", "OPENROUTER_PROVIDER_ONLY"),
        ("order", "OPENROUTER_PROVIDER_ORDER"),
    ):
        values = [value.strip() for value in (os.environ.get(name) or "").split(",")]
        selected = [value for value in values if value]
        if selected:
            routing[field] = selected
    fallbacks = _env_flag_opt("OPENROUTER_ALLOW_FALLBACKS")
    if fallbacks is not None:
        routing["allow_fallbacks"] = fallbacks
    elif "only" in routing:
        routing["allow_fallbacks"] = False
    return routing


def _env_flag_opt(name: str) -> bool | None:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return None
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")
