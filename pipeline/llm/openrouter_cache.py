from __future__ import annotations

import logging
import os
from typing import Any

from pipeline.llm.openrouter_catalog import cache_mode

logger = logging.getLogger(__name__)

OPENROUTER_PROMPT_CACHE_ENV = "OPENROUTER_PROMPT_CACHE"
OPENROUTER_PROMPT_CACHE_TTL_ENV = "OPENROUTER_PROMPT_CACHE_TTL"
OPENROUTER_PROMPT_CACHE_FIRST_MESSAGE_MIN_CHARS_ENV = (
    "OPENROUTER_PROMPT_CACHE_FIRST_MESSAGE_MIN_CHARS"
)


_TTL_FALLBACK_ENV = "ANTHROPIC_PROMPT_CACHE_TTL"

DEFAULT_FIRST_MESSAGE_MIN_CHARS = 2_000

_ONE_HOUR_TOKENS = frozenset({"1h", "1hr", "1hour", "hour"})
_FIVE_MINUTE_TOKENS = frozenset({"", "5m", "5min", "5mins", "5minute", "5minutes"})


def prompt_cache_control() -> dict[str, str] | None:
    enabled = (os.environ.get(OPENROUTER_PROMPT_CACHE_ENV, "1") or "").strip().lower()
    if enabled in {"0", "false", "no", "off", "disabled"}:
        return None

    cache_control = {"type": "ephemeral"}
    ttl = (os.environ.get(OPENROUTER_PROMPT_CACHE_TTL_ENV) or "").strip().lower()
    if not ttl:
        ttl = (os.environ.get(_TTL_FALLBACK_ENV) or "").strip().lower()
    if ttl in _ONE_HOUR_TOKENS:
        cache_control["ttl"] = "1h"
    elif ttl not in _FIVE_MINUTE_TOKENS:
        logger.warning(
            "Ignoring unsupported %s=%r; expected 5m or 1h",
            OPENROUTER_PROMPT_CACHE_TTL_ENV,
            ttl,
        )
    return cache_control


def apply_cache_breakpoints(payload: dict[str, Any], *, model_id: str) -> int:
    if cache_mode(model_id) != "explicit":
        return 0
    cache_control = prompt_cache_control()
    if cache_control is None:
        return 0

    messages = payload.get("messages")
    if not isinstance(messages, list):
        return 0

    marked = 0
    marked_ids: set[int] = set()

    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system":
            if _mark_message_content(message, cache_control):
                marked += 1
                marked_ids.add(id(message))
            break

    first = next(
        (m for m in messages if isinstance(m, dict) and m.get("role") != "system"),
        None,
    )
    if first is not None and _is_large_stable_first_message(first):
        if _mark_message_content(first, cache_control, text_only=True):
            marked += 1
            marked_ids.add(id(first))

    conversation = [m for m in messages if isinstance(m, dict) and m.get("role") != "system"]
    if len(conversation) >= 2:
        for message in reversed(conversation):
            if id(message) in marked_ids:
                break
            if _mark_message_content(message, cache_control):
                marked += 1
            break

    return marked


def _first_message_min_chars() -> int:
    raw = (os.environ.get(OPENROUTER_PROMPT_CACHE_FIRST_MESSAGE_MIN_CHARS_ENV) or "").strip()
    if not raw:
        return DEFAULT_FIRST_MESSAGE_MIN_CHARS
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(
            "Ignoring unsupported %s=%r; expected an integer",
            OPENROUTER_PROMPT_CACHE_FIRST_MESSAGE_MIN_CHARS_ENV,
            raw,
        )
        return DEFAULT_FIRST_MESSAGE_MIN_CHARS


def _is_large_stable_first_message(message: dict[str, Any]) -> bool:
    if message.get("role") != "user":
        return False
    content = message.get("content")
    if isinstance(content, str):
        text_len = len(content.strip())
    elif isinstance(content, list):
        text_len = sum(
            len(part["text"].strip())
            for part in content
            if isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
        )
    else:
        return False
    return text_len >= _first_message_min_chars()


def _mark_message_content(
    message: dict[str, Any],
    cache_control: dict[str, str],
    *,
    text_only: bool = False,
) -> bool:
    content = message.get("content")
    if isinstance(content, str):
        if not content.strip():
            return False
        message["content"] = [
            {"type": "text", "text": content, "cache_control": dict(cache_control)}
        ]
        return True
    if not isinstance(content, list):
        return False

    for part in reversed(content):
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and isinstance(part.get("text"), str) and part["text"]:
            part["cache_control"] = dict(cache_control)
            return True
        if not text_only and part.get("type") == "image_url":
            part["cache_control"] = dict(cache_control)
            return True
    return False
