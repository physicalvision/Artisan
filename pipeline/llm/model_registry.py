from __future__ import annotations

import math
import os
from collections.abc import Mapping
from typing import Any, Literal

from artisan.values import normalize_provider_name
from pipeline.cost import pricing_for_provider_model
from pipeline.llm.credentials import (
    anthropic_api_keys_from_env,
    image_api_key_from_env,
    openai_api_keys_from_env,
    openrouter_api_keys_from_env,
)
from pipeline.llm.endpoints import provider_base_url
from pipeline.llm.errors import ProviderConfigurationError

ModelRole = Literal["text", "compaction", "image_envelope", "image_tool"]
DEFAULT_TEXT_MODEL = "gpt-6-astra"
DEFAULT_COMPACTION_MODEL = "gpt-5.6-sol"
DEFAULT_IMAGE_TOOL_MODEL = "gpt-image-2.5-sunburst"

_TEXT_ROLES = frozenset({"text", "compaction", "image_envelope"})
MODEL_CAPABILITIES: dict[str, frozenset[str]] = {
    "gpt-6-astra": _TEXT_ROLES,
    "gpt-5.6-sol": _TEXT_ROLES,
    "gpt-5.5": _TEXT_ROLES,
    "gpt-5.5-2026-04-23": _TEXT_ROLES,
    "gpt-image-2": frozenset({"image_tool"}),
    "gpt-image-2.5-sunburst": frozenset({"image_tool"}),
    "gpt-image-2.5-sunburst-2026-09-08": frozenset({"image_tool"}),
}

PROVIDER_DEFAULT_MODELS = {
    "openai": DEFAULT_TEXT_MODEL,
    "anthropic": "claude-opus-4-8",
    "gemini": "gemini-3.5-flash",
    "openrouter": "openai/gpt-5.5",
}


def default_model_for_provider(provider: str) -> str:
    provider = normalize_provider_name(provider).value
    try:
        return PROVIDER_DEFAULT_MODELS[provider]
    except KeyError as exc:
        raise ProviderConfigurationError(f"Unsupported provider: {provider}") from exc


def validate_provider_model(provider: str, model_id: str, role: ModelRole = "text") -> str:
    provider = normalize_provider_name(provider).value
    default_model_for_provider(provider)
    if provider == "openai":
        return validate_model(model_id, role)
    model = str(model_id).strip()
    if role not in {"text", "compaction"}:
        raise ProviderConfigurationError(f"Unsupported {provider} model role: {role}")
    if provider == "openrouter":
        if "/" not in model or any(character.isspace() for character in model):
            raise ProviderConfigurationError("OpenRouter requires a namespaced model ID")
        return model
    rates = pricing_for_provider_model(provider, model)
    required = ("input_uncached", "input_cached", "output")
    if rates is None or any(
        key not in rates or not math.isfinite(rates[key]) or rates[key] < 0 for key in required
    ):
        raise ProviderConfigurationError(
            f"Unsupported or unpriced {provider} {role} model: {model}"
        )
    return model


IMAGE_TOOL_PRICING: dict[str, dict[str, float]] = {
    "gpt-image-2": {"text_input": 5.0, "image_input": 8.0, "image_output": 30.0},
    "gpt-image-2.5-sunburst": {"text_input": 5.0, "image_input": 8.0, "image_output": 30.0},
    "gpt-image-2.5-sunburst-2026-09-08": {
        "text_input": 5.0,
        "image_input": 8.0,
        "image_output": 30.0,
    },
}


def validate_model(model_id: str, role: ModelRole) -> str:
    model = str(model_id).strip()
    if role not in MODEL_CAPABILITIES.get(model, frozenset()):
        raise ProviderConfigurationError(f"Unsupported or unpriced OpenAI {role} model: {model}")
    rates = (
        IMAGE_TOOL_PRICING.get(model)
        if role == "image_tool"
        else pricing_for_provider_model("openai", model)
    )
    required = (
        ("text_input", "image_input", "image_output")
        if role == "image_tool"
        else (
            "input_uncached",
            "input_cached",
            "output",
        )
    )
    if rates is None or any(
        key not in rates or not math.isfinite(rates[key]) or rates[key] < 0 for key in required
    ):
        raise ProviderConfigurationError(
            f"Missing or invalid OpenAI pricing for {role} model: {model}"
        )
    return model


def image_tool_pricing(model_id: str) -> dict[str, float]:
    return dict(IMAGE_TOOL_PRICING[validate_model(model_id, "image_tool")])


def _configured_value(values: Mapping[str, str], name: str, default: str) -> str:
    return (values.get(name) or "").strip() or default


def _configured_provider(values: Mapping[str, str], role: str | None = None) -> str:
    provider = _configured_value(values, "ARTISAN_PIPELINE_PROVIDER", "openai")
    if role:
        provider = _configured_value(values, f"ARTISAN_PIPELINE_PROVIDER_{role.upper()}", provider)
    return normalize_provider_name(provider).value


def _provider_endpoint(values: Mapping[str, str], provider: str) -> str:
    override = values.get(f"{provider.upper()}_BASE_URL")
    if provider == "gemini":
        override = override or values.get("GOOGLE_BASE_URL")
        if override:
            from pipeline.llm.gemini import gemini_client_config_from_env

            try:
                client_config = gemini_client_config_from_env(dict(values))
            except ValueError:
                client_config = None
            if client_config is not None and client_config.backend == "vertex":
                raise ProviderConfigurationError(
                    "Vertex AI chooses its own official endpoint; unset GEMINI_BASE_URL "
                    "and GOOGLE_BASE_URL rather than using the Gemini Developer API endpoint."
                )
    return provider_base_url(provider, override)


def _has_credentials(values: Mapping[str, str], provider: str) -> bool:
    if provider == "openai":
        return bool(openai_api_keys_from_env(values))
    if provider == "anthropic":
        return bool(anthropic_api_keys_from_env(values))
    if provider == "openrouter":
        return bool(openrouter_api_keys_from_env(values))
    if provider == "gemini":
        from pipeline.llm.gemini import gemini_client_config_from_env

        try:
            gemini_client_config_from_env(dict(values))
        except ValueError:
            return False
        return True
    return False


def compaction_model_from_env(env: Mapping[str, str] | None = None) -> str:
    values = os.environ if env is None else env
    return validate_model(
        _configured_value(values, "OPENAI_COMPACTION_MODEL", DEFAULT_COMPACTION_MODEL), "compaction"
    )


def image_models_from_env(env: Mapping[str, str] | None = None) -> tuple[str, str]:
    values = os.environ if env is None else env
    provider = _configured_provider(values, "design")
    pipeline_model = _configured_value(
        values, "ARTISAN_PIPELINE_MODEL", default_model_for_provider(provider)
    )
    design_model = _configured_value(values, "ARTISAN_PIPELINE_MODEL_DESIGN", pipeline_model)
    if provider != "openai":
        design_model = DEFAULT_TEXT_MODEL
    return (
        validate_model(
            _configured_value(values, "ARTISAN_IMAGE_MODEL", design_model), "image_envelope"
        ),
        validate_model(
            _configured_value(values, "ARTISAN_IMAGE_TOOL_MODEL", DEFAULT_IMAGE_TOOL_MODEL),
            "image_tool",
        ),
    )


def validate_runtime_configuration(
    *,
    env: Mapping[str, str] | None = None,
    require_credentials: bool = True,
) -> dict[str, Any]:
    values = os.environ if env is None else env
    provider = _configured_provider(values)
    default_model_for_provider(provider)
    text_providers = {
        role: _configured_provider(values, role) for role in ("design", "coding", "review")
    }
    text_models = {
        role: validate_provider_model(
            selected,
            _configured_value(
                values,
                f"ARTISAN_PIPELINE_MODEL_{role.upper()}",
                _configured_value(
                    values, "ARTISAN_PIPELINE_MODEL", default_model_for_provider(selected)
                ),
            ),
            "text",
        )
        for role, selected in text_providers.items()
    }
    compaction = compaction_model_from_env(values) if "openai" in text_providers.values() else None
    image_model, image_tool = image_models_from_env(values)
    endpoints = {
        "text": _provider_endpoint(values, provider),
        "image": provider_base_url(
            "openai", values.get("ARTISAN_IMAGE_BASE_URL") or values.get("OPENAI_BASE_URL")
        ),
    }
    for selected in text_providers.values():
        _provider_endpoint(values, selected)

    credentials = {
        "text": all(_has_credentials(values, selected) for selected in text_providers.values()),
        "image": bool(image_api_key_from_env(values)),
    }
    if require_credentials and not all(credentials.values()):
        raise ProviderConfigurationError(
            "Provider credentials are required for each selected text provider and OpenAI image generation. "
            "Configure the provider API key; use OPENAI_API_KEY, OPENAI_API_KEYS or ARTISAN_IMAGE_API_KEY for images."
        )
    return {
        "provider": provider,
        "text_models": text_models,
        "text_providers": text_providers,
        "compaction_model": compaction,
        "image_model": image_model,
        "image_tool_model": image_tool,
        "endpoints": endpoints,
        "credentials": credentials,
    }
