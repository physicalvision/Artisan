from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from artisan.values import PROVIDER_VALUE_SET
from artisan.values import infer_provider_from_model_id as _infer_provider
from artisan.values import normalize_provider_name as _normalize_provider
from pipeline.llm.base import ProviderClient
from pipeline.llm.credentials import (
    anthropic_api_key_from_env,
    openai_api_key_from_env,
    openrouter_api_key_from_env,
)
from pipeline.llm.model_registry import default_model_for_provider, validate_provider_model
from pipeline.llm.openai import OpenAILLM

SUPPORTED_PROVIDERS = PROVIDER_VALUE_SET


@dataclass(slots=True, frozen=True)
class ProviderConfig:
    provider: str = "openai"
    model_id: str | None = None
    thinking_level: str = "high"
    openai_transport: str | None = None
    openai_reasoning_summary: str | None = "auto"
    openai_prompt_cache_key: str | None = None
    openai_prompt_cache_retention: str | None = None
    openai_canonical_contract_fingerprint: str | None = None

    @property
    def normalized_provider(self) -> str:
        return normalize_provider_name(self.provider)


@dataclass(slots=True, frozen=True)
class ProviderConstructors:
    openai: Callable[..., ProviderClient] = OpenAILLM
    anthropic: Callable[..., ProviderClient] | None = None
    gemini: Callable[..., ProviderClient] | None = None
    openrouter: Callable[..., ProviderClient] | None = None


def normalize_provider_name(provider: str | None) -> str:
    return _normalize_provider(provider).value


def infer_provider_from_model_id(model_id: str | None) -> str | None:
    provider = _infer_provider(model_id)
    return provider.value if provider is not None else None


def default_model_id(config: ProviderConfig) -> str:
    provider = normalize_provider_name(config.provider)
    return validate_provider_model(
        provider, config.model_id or default_model_for_provider(provider), role="text"
    )


def create_provider_client(
    config: ProviderConfig,
    *,
    dry_run: bool = False,
    constructors: ProviderConstructors | None = None,
) -> ProviderClient:
    model_id = default_model_id(config)
    provider = config.normalized_provider
    selected = constructors or ProviderConstructors()
    if provider != "openai":
        constructor = getattr(selected, provider)
        if constructor is None:
            if provider == "anthropic":
                from pipeline.llm.anthropic import AnthropicLLM

                constructor = AnthropicLLM
            elif provider == "gemini":
                from pipeline.llm.gemini import GeminiLLM

                constructor = GeminiLLM
            else:
                from pipeline.llm.openrouter import OpenRouterLLM

                constructor = OpenRouterLLM
        return constructor(
            model_id=model_id,
            thinking_level=config.thinking_level,
            dry_run=dry_run,
        )
    return selected.openai(
        model_id=model_id,
        thinking_level=config.thinking_level,
        reasoning_summary=config.openai_reasoning_summary,
        transport=config.openai_transport,
        prompt_cache_key=config.openai_prompt_cache_key,
        prompt_cache_retention=config.openai_prompt_cache_retention,
        canonical_contract_fingerprint=config.openai_canonical_contract_fingerprint,
        dry_run=dry_run,
    )


def validate_provider_credentials(provider: str) -> None:
    name = normalize_provider_name(provider)
    if name == "gemini":
        from pipeline.llm.gemini import gemini_client_config_from_env

        gemini_client_config_from_env()
    elif name == "anthropic" and not anthropic_api_key_from_env():
        raise ValueError("Anthropic credentials are required. Set ANTHROPIC_API_KEY.")
    elif name == "openrouter" and not openrouter_api_key_from_env():
        raise ValueError("OpenRouter credentials are required. Set OPENROUTER_API_KEY.")
    elif name == "openai" and not openai_api_key_from_env():
        raise ValueError("OpenAI credentials are required. Set OPENAI_API_KEY.")
