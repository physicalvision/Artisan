from __future__ import annotations

from urllib.parse import urlsplit

OPENAI_BASE_URL = "https://api.openai.com/v1"
PROVIDER_BASE_URLS = {
    "openai": OPENAI_BASE_URL,
    "anthropic": "https://api.anthropic.com",
    "openrouter": "https://openrouter.ai/api/v1",
    "gemini": "https://generativelanguage.googleapis.com",
}


def official_openai_base_url(value: str | None = None) -> str:
    return provider_base_url("openai", value)


def provider_base_url(provider: str, value: str | None = None) -> str:
    provider = "gemini" if provider == "google" else provider
    if provider not in PROVIDER_BASE_URLS:
        raise ValueError(f"Unsupported provider: {provider}")
    expected = PROVIDER_BASE_URLS[provider]
    raw = (value or expected).strip().rstrip("/")
    parsed = urlsplit(raw)
    canonical = urlsplit(expected)
    if (
        parsed.scheme != "https"
        or parsed.hostname != canonical.hostname
        or parsed.port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", canonical.path)
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"Expected the official {provider} endpoint: {expected}")
    return expected
