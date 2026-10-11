from __future__ import annotations

import os

from artisan.values import PROVIDER_VALUES, ThinkingLevel, normalize_thinking_level
from pipeline.llm.factory import ProviderConfig, default_model_id, normalize_provider_name

VALID = PROVIDER_VALUES
DEFAULT_PROVIDER = "openai"


def pipeline_provider(role: str | None = None, *, strict: bool = True) -> str:
    value = (
        ((os.environ.get(f"ARTISAN_PIPELINE_PROVIDER_{role.upper()}") or "").strip())
        if role
        else ""
    )
    value = (
        value or (os.environ.get("ARTISAN_PIPELINE_PROVIDER") or "").strip() or DEFAULT_PROVIDER
    )
    try:
        return normalize_provider_name(value)
    except ValueError:
        if strict:
            raise
        return DEFAULT_PROVIDER


def pipeline_model(role: str | None = None) -> str | None:
    if role:
        configured = (os.environ.get(f"ARTISAN_PIPELINE_MODEL_{role.upper()}") or "").strip()
        if configured:
            return configured
    return (os.environ.get("ARTISAN_PIPELINE_MODEL") or "").strip() or None


def resolve_pipeline_model(provider: str, model_id: str | None) -> str:
    return default_model_id(ProviderConfig(provider=provider, model_id=model_id))


def effective_pipeline_model(role: str | None = None) -> str:
    return resolve_pipeline_model(pipeline_provider(role), pipeline_model(role))


def pipeline_thinking(role: str | None = None) -> str:
    raw = (
        ((os.environ.get(f"ARTISAN_PIPELINE_THINKING_{role.upper()}") or "").strip())
        if role
        else ""
    )
    raw = (raw or os.environ.get("ARTISAN_PIPELINE_THINKING") or "high").strip().lower()
    if raw in ("mid", "medium"):
        raw = ThinkingLevel.MED.value
    if raw in {"xhigh", "max"}:
        if pipeline_provider(role) != "openai" or effective_pipeline_model(role) != "gpt-6-astra":
            raise ValueError(f"Thinking level {raw} requires the OpenAI gpt-6-astra model")
        return raw
    return normalize_thinking_level(raw, default=ThinkingLevel.HIGH).value


def set_pipeline_provider(provider: str | None, model: str | None) -> None:
    if provider:
        os.environ["ARTISAN_PIPELINE_PROVIDER"] = normalize_provider_name(provider)
    if model:
        os.environ["ARTISAN_PIPELINE_MODEL"] = resolve_pipeline_model(
            provider or pipeline_provider(), model
        )
