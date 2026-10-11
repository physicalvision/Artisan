from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from typing import Any, Optional

from artisan.sdk_support import normalize_sdk_package as _normalize_sdk_package

logger = logging.getLogger(__name__)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _short_cache_digest(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _stable_tool_schema_payload(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _tool_identity(tool: dict[str, Any]) -> tuple[str, str]:
        tool_type = str(tool.get("type") or "")
        if tool_type == "function":
            function = tool.get("function")
            if isinstance(function, dict):
                return tool_type, str(function.get("name") or "")
        return tool_type, str(tool.get("name") or "")

    normalized = [tool for tool in tools if isinstance(tool, dict)]
    return sorted(normalized, key=_tool_identity)


def _prompt_cache_key_strategy_from_env() -> str:
    raw = os.environ.get("OPENAI_PROMPT_CACHE_KEY_STRATEGY")
    if raw is None:
        return "artisan-v1"
    value = raw.strip().lower()
    if not value:
        return "artisan-v1"
    if value in {"artisan-v1", "off"}:
        return value
    logger.warning(
        "Unknown OPENAI_PROMPT_CACHE_KEY_STRATEGY=%r; falling back to artisan-v1",
        raw,
    )
    return "artisan-v1"


def _prompt_cache_retention_from_env(*, model_id: str) -> Optional[str]:
    raw = os.environ.get("OPENAI_PROMPT_CACHE_RETENTION")
    if raw is None:
        normalized = model_id.strip().lower()
        if (
            normalized == "gpt-5"
            or normalized.startswith("gpt-5.1")
            or normalized.startswith("gpt-5-codex")
            or normalized.startswith("gpt-5.1-codex")
            or normalized.startswith("gpt-4.1")
        ):
            return "24h"
        return None
    value = raw.strip()
    if not value or value.lower() in {"0", "false", "none", "off"}:
        return None
    return value


def build_openai_prompt_cache_settings(
    *,
    model_id: str,
    sdk_package: str,
    system_prompt: str,
    sdk_docs_context: str,
    tools: list[dict[str, Any]],
) -> tuple[Optional[str], Optional[str], str]:
    strategy = _prompt_cache_key_strategy_from_env()
    retention = _prompt_cache_retention_from_env(model_id=model_id)

    normalized_sdk_package = _normalize_sdk_package(sdk_package)
    key_payload = {
        "provider": "openai",
        "model_id": model_id.strip(),
        "sdk_package": normalized_sdk_package,
        "system_prompt_sha256": _sha256_text(system_prompt),
        "sdk_docs_sha256": _sha256_text(sdk_docs_context),
        "tool_schema_sha256": _sha256_text(_canonical_json(_stable_tool_schema_payload(tools))),
    }
    canonical_contract_fingerprint = _sha256_text(_canonical_json(key_payload))

    if strategy == "off":
        logger.info(
            "OpenAI prompt caching disabled (strategy=off, retention=%s)",
            retention or "off",
        )
        return None, retention, canonical_contract_fingerprint

    prefix = (os.environ.get("OPENAI_PROMPT_CACHE_KEY_PREFIX") or "").strip()
    digest = _short_cache_digest(_canonical_json(key_payload))
    visible_prefix = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in prefix
    ).strip("-_")[:16]
    key = f"ac1:{digest}"
    if visible_prefix:
        key = f"ac1:{visible_prefix}:{digest}"
    if len(key) > 64:
        key = f"ac1:{digest}"

    logger.info(
        "OpenAI prompt caching enabled (strategy=%s, retention=%s, prefix=%s)",
        strategy,
        retention or "off",
        prefix or "<none>",
    )
    return key, retention, canonical_contract_fingerprint
