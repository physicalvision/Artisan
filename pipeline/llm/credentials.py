from __future__ import annotations

import os
import random
from collections.abc import Mapping


def openai_api_keys_from_env(env: Mapping[str, str] | None = None) -> list[str]:
    values = os.environ if env is None else env
    primary = (values.get("OPENAI_API_KEY") or "").strip()
    pool = (values.get("OPENAI_API_KEYS") or "").replace("\n", ",").split(",")
    keys = [primary, *(value.strip() for value in pool)]
    return list(dict.fromkeys(key for key in keys if key))


def openai_api_key_from_env(env: Mapping[str, str] | None = None) -> str | None:
    keys = openai_api_keys_from_env(env)
    return random.choice(keys) if keys else None


def image_api_key_from_env(env: Mapping[str, str] | None = None) -> str | None:
    values = os.environ if env is None else env
    dedicated = (values.get("ARTISAN_IMAGE_API_KEY") or "").strip()
    return dedicated or openai_api_key_from_env(values)


def _provider_keys(values: Mapping[str, str], *names: str) -> list[str]:
    keys = [
        token.strip()
        for name in names
        for token in (values.get(name) or "").replace("\n", ",").split(",")
    ]
    return list(dict.fromkeys(key for key in keys if key))


def anthropic_api_keys_from_env(env: Mapping[str, str] | None = None) -> list[str]:
    return _provider_keys(
        os.environ if env is None else env, "ANTHROPIC_API_KEY", "ANTHROPIC_API_KEYS"
    )


def anthropic_api_key_from_env(env: Mapping[str, str] | None = None) -> str | None:
    keys = anthropic_api_keys_from_env(env)
    return random.choice(keys) if keys else None


def openrouter_api_keys_from_env(env: Mapping[str, str] | None = None) -> list[str]:
    return _provider_keys(
        os.environ if env is None else env, "OPENROUTER_API_KEY", "OPENROUTER_API_KEYS"
    )


def openrouter_api_key_from_env(env: Mapping[str, str] | None = None) -> str | None:
    keys = openrouter_api_keys_from_env(env)
    return random.choice(keys) if keys else None


def gemini_api_keys_from_env(env: Mapping[str, str] | None = None) -> list[str]:
    return _provider_keys(
        os.environ if env is None else env,
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEYS",
        "GOOGLE_API_KEYS",
    )
