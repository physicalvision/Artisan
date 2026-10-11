from __future__ import annotations

import asyncio
import base64
import copy
import json
import logging
import math
import mimetypes
import os
import random
import uuid
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from artisan.values import reasoning_level_alias
from pipeline.llm._image_tokens import data_url_image_tokens, image_caps_for_model
from pipeline.llm._image_wire import image_bytes_for_wire
from pipeline.llm.base import (
    ContextWindowPressure,
    PrepareRequestResult,
    build_context_window_pressure,
)
from pipeline.llm.credentials import openrouter_api_key_from_env as openrouter_api_key_from_env
from pipeline.llm.credentials import openrouter_api_keys_from_env
from pipeline.llm.endpoints import provider_base_url
from pipeline.llm.errors import CostAccountingError
from pipeline.llm.openrouter_cache import apply_cache_breakpoints
from pipeline.llm.openrouter_catalog import OpenRouterModelSpec, model_spec
from pipeline.llm.openrouter_routing import build_provider_routing
from pipeline.llm.usage import require_reported_cost

logger = logging.getLogger(__name__)

DEFAULT_OPENROUTER_MODEL = "openai/gpt-5.5"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


DEFAULT_OPENROUTER_CONTEXT_TOKENS = 262_144
DEFAULT_OPENROUTER_MAX_TOKENS = 32_000
DEFAULT_OPENROUTER_OUTPUT_SAFETY_TOKENS = 1_024


_OVER_BUDGET_MAX_TOKENS = 32_000


_MAX_BUFFERED_TRACE_EVENTS = 64


def openrouter_base_url(env: dict[str, str] | None = None) -> str:
    values = os.environ if env is None else env
    return provider_base_url("openrouter", values.get("OPENROUTER_BASE_URL"))


def _resolve_output_limit(spec: OpenRouterModelSpec | None) -> int:
    override = _env_int_opt("OPENROUTER_MAX_TOKENS")
    if override is None:
        override = _env_int_opt("OPENROUTER_MAX_COMPLETION_TOKENS")
    if override is not None:
        return override
    if spec is not None and spec.max_completion_tokens and spec.max_completion_tokens > 0:
        return spec.max_completion_tokens
    return DEFAULT_OPENROUTER_MAX_TOKENS


def _resolve_context_limit(spec: OpenRouterModelSpec | None) -> int:
    override = _env_int_opt("OPENROUTER_CONTEXT_TOKENS")
    if override is not None:
        return override
    if spec is not None and spec.context_length > 0:
        return spec.context_length
    return DEFAULT_OPENROUTER_CONTEXT_TOKENS


class OpenRouterLLM:

    supports_native_cost_compaction = False

    def __init__(
        self,
        model_id: str = DEFAULT_OPENROUTER_MODEL,
        *,
        thinking_level: str = "high",
        dry_run: bool = False,
        use_catalog: bool | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.base_url = provider_base_url(
            "openrouter", base_url or os.environ.get("OPENROUTER_BASE_URL")
        )
        self.thinking_level = thinking_level
        self.reasoning = _reasoning_config_from_thinking_level(thinking_level)

        allow_catalog = (not dry_run) if use_catalog is None else use_catalog
        spec = model_spec(model_id) if allow_catalog else None
        self.model_spec = spec
        self.max_tokens = _resolve_output_limit(spec)
        self.context_tokens = _resolve_context_limit(spec)
        self.output_safety_tokens = _env_int(
            "OPENROUTER_OUTPUT_SAFETY_TOKENS",
            DEFAULT_OPENROUTER_OUTPUT_SAFETY_TOKENS,
        )
        self.request_timeout_seconds = _env_float("OPENROUTER_REQUEST_TIMEOUT_SECONDS", 900.0)
        self.max_attempts = max(1, int(_env_float("OPENROUTER_MAX_ATTEMPTS", 4)))

        self.rate_limit_max_attempts = max(
            1, int(_env_float("OPENROUTER_RATE_LIMIT_MAX_ATTEMPTS", 12))
        )
        self.retry_base_seconds = _env_float("OPENROUTER_RETRY_BASE_SECONDS", 0.5)
        self.retry_max_seconds = _env_float("OPENROUTER_RETRY_MAX_SECONDS", 20.0)

        self.retry_after_max_seconds = _env_float("OPENROUTER_RETRY_AFTER_MAX_SECONDS", 120.0)

        self.stream = _env_flag("OPENROUTER_STREAM", True)

        self._trace_events: list[tuple[str, dict[str, Any]]] = []
        self._reported_route: str | None = None
        self._reported_cache_breakpoints: int | None = None
        if dry_run:
            self._client = None
            self._client_is_async = False
            self._api_keys: list[str] = []
            self._api_key_index = 0
        else:
            self._api_keys = openrouter_api_keys_from_env()
            base_url = self.base_url

            self._api_key_index = random.randrange(len(self._api_keys)) if self._api_keys else 0
            api_key = self._api_keys[self._api_key_index] if self._api_keys else None
            if not api_key:
                raise ValueError(
                    "OpenRouter credentials not found. "
                    "Set OPENROUTER_API_KEY or OPENROUTER_API_KEYS."
                )

            client_kwargs: dict[str, Any] = {
                "api_key": api_key,
                "base_url": base_url,
                "max_retries": 0,
            }
            default_headers = _openrouter_default_headers()
            if default_headers:
                client_kwargs["default_headers"] = default_headers
            if self.request_timeout_seconds and self.request_timeout_seconds > 0:
                client_kwargs["timeout"] = float(self.request_timeout_seconds)

            self._client: Any
            self._client_is_async: bool
            try:
                from openai import AsyncOpenAI  # type: ignore

                self._client = AsyncOpenAI(**client_kwargs)
                self._client_is_async = True
            except Exception:
                try:
                    from openai import OpenAI  # type: ignore
                except Exception as exc:  # pragma: no cover
                    raise RuntimeError(
                        "OpenRouter provider selected but the `openai` package is not installed. "
                        "Run ./install.sh to install the locked provider dependencies."
                    ) from exc

                self._client = OpenAI(**client_kwargs)
                self._client_is_async = False

    def build_request_preview(
        self,
        *,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
    ) -> dict[str, Any]:
        payload = self._build_chat_payload(
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
        )
        payload["base_url"] = self.base_url
        return payload

    def context_window_pressure(self, usage: dict[str, int]) -> ContextWindowPressure:
        return build_context_window_pressure(
            provider="openrouter",
            usage=usage,
            max_context_tokens=self.context_tokens,
        )

    async def prepare_next_request(
        self,
        *,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
        completed_turns: int,
        consecutive_compile_failure_count: int = 0,
        last_compile_failure_sig: str | None = None,
        entry_reason: str | None = None,
    ) -> PrepareRequestResult:
        return PrepareRequestResult()

    async def generate_with_tools(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
    ) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError("OpenRouter transport is unavailable in dry_run mode")

        request_payload = self._build_chat_payload(
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
        )

        async def _request_once() -> Any:
            request_coro = self._chat_completion(request_payload)
            if self.request_timeout_seconds and self.request_timeout_seconds > 0:
                return await asyncio.wait_for(
                    request_coro,
                    timeout=float(self.request_timeout_seconds),
                )
            return await request_coro

        response = await _async_retry(
            _request_once,
            max_attempts=self.max_attempts,
            should_retry=_should_retry_openrouter_exception,
            base_delay=self.retry_base_seconds,
            max_delay=self.retry_max_seconds,
            logger=logger,
            context="openrouter[chat.completions]",
            delay_hint=lambda exc: _retry_after_seconds(exc, cap=self.retry_after_max_seconds),
            on_retry=self._rotate_api_key,
            max_attempts_for=lambda exc: (
                self.rate_limit_max_attempts if _is_rate_limited(exc) else None
            ),
        )
        result = self._convert_response(response)
        require_reported_cost(result.get("usage"))
        return result

    def _rotate_api_key(self) -> None:
        keys = getattr(self, "_api_keys", None)
        if not keys or len(keys) < 2 or self._client is None:
            return
        self._api_key_index = (self._api_key_index + 1) % len(keys)
        try:
            self._client.api_key = keys[self._api_key_index]
        except Exception:  # noqa: BLE001 - key rotation must never fail the request
            logger.debug("openrouter: api key rotation not supported by this client")

    async def close(self) -> None:
        client = getattr(self, "_client", None)
        if client is None:
            return None
        close_method = getattr(client, "close", None)
        if close_method is None:
            return None
        result = close_method()
        if asyncio.iscoroutine(result):
            await result
        return None

    async def _chat_completion(self, request_payload: dict[str, Any]) -> Any:
        payload = dict(request_payload)
        extra_body = payload.pop("extra_body", None)

        if self.stream and self._client_is_async:
            return await self._chat_completion_streaming(payload, extra_body)
        if self._client_is_async:
            if extra_body is None:
                return await self._client.chat.completions.create(**payload)
            return await self._client.chat.completions.create(
                **payload,
                extra_body=extra_body,
            )
        if extra_body is None:
            return await asyncio.to_thread(self._client.chat.completions.create, **payload)
        return await asyncio.to_thread(
            self._client.chat.completions.create,
            **payload,
            extra_body=extra_body,
        )

    async def _chat_completion_streaming(
        self, payload: dict[str, Any], extra_body: dict[str, Any] | None
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = dict(payload)
        if extra_body is not None:
            kwargs["extra_body"] = extra_body
        stream = await self._client.chat.completions.create(**kwargs, stream=True)
        state = _new_stream_state()
        try:
            async for chunk in stream:
                _accumulate_stream_chunk(state, chunk)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                result = close()
                if asyncio.iscoroutine(result):
                    await result
        return _stream_state_to_response(state)

    def _build_chat_payload(
        self,
        *,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
    ) -> dict[str, Any]:
        chat_messages: list[dict[str, Any]] = []
        if system_prompt.strip():
            chat_messages.append({"role": "system", "content": system_prompt})
        chat_messages.extend(_convert_chat_messages(messages))

        extra_body: dict[str, Any] = {"reasoning": copy.deepcopy(self.reasoning)}
        routing = build_provider_routing(self.model_id)
        if routing:
            extra_body["provider"] = routing
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": chat_messages,
            "extra_body": extra_body,
        }
        converted_tools = _convert_tools(tools)
        if converted_tools:
            payload["tools"] = converted_tools

        self._note_cache_breakpoints(apply_cache_breakpoints(payload, model_id=self.model_id))
        max_tokens = self._request_max_tokens(payload)
        if max_tokens > 0:
            payload["max_tokens"] = max_tokens
        return payload

    def _request_max_tokens(self, payload: dict[str, Any]) -> int:
        if self.max_tokens <= 0:
            return 0
        if self.context_tokens <= 0:
            return self.max_tokens
        image_caps = image_caps_for_model(self.model_id)
        estimated_prompt_tokens = _estimate_prompt_tokens(payload, image_caps)
        available = (
            self.context_tokens - estimated_prompt_tokens - max(0, self.output_safety_tokens)
        )
        if available <= 0:
            clamped = min(self.max_tokens, _OVER_BUDGET_MAX_TOKENS)

            logger.warning(
                "openrouter: estimated prompt %s tokens >= context %s; clamping max_tokens to %s",
                estimated_prompt_tokens,
                self.context_tokens,
                clamped,
            )
            self._record_trace_event(
                "openrouter_output_clamped",
                {
                    "model_id": self.model_id,
                    "estimated_prompt_tokens": estimated_prompt_tokens,
                    "context_tokens": self.context_tokens,
                    "output_safety_tokens": max(0, self.output_safety_tokens),
                    "clamped_max_tokens": clamped,
                },
            )
            return clamped
        return min(self.max_tokens, available)

    def _record_trace_event(self, name: str, payload: dict[str, Any]) -> None:
        self._trace_events.append((name, payload))
        if len(self._trace_events) > _MAX_BUFFERED_TRACE_EVENTS:
            del self._trace_events[:-_MAX_BUFFERED_TRACE_EVENTS]

    def drain_trace_events(self) -> list[tuple[str, dict[str, Any]]]:
        events = self._trace_events
        self._trace_events = []
        return events

    def _note_cache_breakpoints(self, count: int) -> None:

        if count == self._reported_cache_breakpoints:
            return
        self._reported_cache_breakpoints = count
        self._record_trace_event(
            "openrouter_prompt_cache",
            {"model_id": self.model_id, "breakpoints": count},
        )

    def _note_served_by(self, response: Any) -> None:
        served_by = _get(response, "provider")
        if not isinstance(served_by, str) or not served_by.strip():
            return
        served_by = served_by.strip()
        if served_by == self._reported_route:
            return
        self._reported_route = served_by
        self._record_trace_event(
            "openrouter_route",
            {
                "model_id": self.model_id,
                "served_by": served_by,
                "requested": build_provider_routing(self.model_id),
            },
        )

    def _convert_response(self, response: Any) -> dict[str, Any]:
        self._note_served_by(response)
        choice = _first_choice(response)
        message = _get(choice, "message")
        if message is None:
            return {}

        content = _get(message, "content")
        tool_calls = _serialize_tool_calls(_get(message, "tool_calls"))
        reasoning = _get(message, "reasoning")
        reasoning_details = _serialize_json_value(_get(message, "reasoning_details"))
        usage = _extract_usage(response)

        result: dict[str, Any] = {
            "content": content if isinstance(content, str) else "",
            "tool_calls": tool_calls,
        }
        thought_summary = _reasoning_details_summary(reasoning_details)
        if not thought_summary and isinstance(reasoning, str):
            thought_summary = reasoning.strip()
        if thought_summary:
            result["thought_summary"] = thought_summary
        if reasoning_details or reasoning:
            result["extra_content"] = {
                "openrouter": {
                    "reasoning": reasoning,
                    "reasoning_details": reasoning_details,
                }
            }
        if usage:
            result["usage"] = usage
        return result


def _convert_chat_messages(messages: list[dict]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "user":
            converted.append({"role": "user", "content": _convert_message_content(message)})
            continue
        if role == "assistant":
            assistant: dict[str, Any] = {"role": "assistant"}
            content = message.get("content")
            tool_calls = message.get("tool_calls")
            if isinstance(content, str):
                assistant["content"] = content
            elif tool_calls:
                assistant["content"] = None
            else:
                assistant["content"] = ""
            if tool_calls:
                assistant["tool_calls"] = tool_calls
            extra_content = message.get("extra_content")
            if isinstance(extra_content, dict):
                openrouter = extra_content.get("openrouter")
                if isinstance(openrouter, dict):
                    reasoning = openrouter.get("reasoning")
                    reasoning_details = openrouter.get("reasoning_details")
                    if reasoning:
                        assistant["reasoning"] = reasoning
                    if reasoning_details:
                        assistant["reasoning_details"] = reasoning_details
            converted.append(assistant)
            continue
        if role == "tool":
            tool_message: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": message.get("tool_call_id") or message.get("call_id"),
                "content": _convert_message_content(message),
            }
            name = message.get("name")
            if isinstance(name, str) and name:
                tool_message["name"] = name
            converted.append(tool_message)
    return converted


def _convert_message_content(message: dict[str, Any]) -> Any:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in {"input_text", "text"}:
            text = part.get("text")
            if isinstance(text, str) and text:
                parts.append({"type": "text", "text": text})
            continue
        if part_type == "input_image":
            image = _convert_image_part(part)
            if image:
                parts.append(image)
    return parts or ""


def _convert_image_part(part: dict[str, Any]) -> dict[str, Any] | None:
    detail = part.get("detail")
    url = part.get("image_url")
    if not isinstance(url, str) or not url:
        image_path = part.get("image_path")
        if isinstance(image_path, str) and image_path:
            url = _image_path_to_data_url(image_path)
    if not isinstance(url, str) or not url:
        return None

    image_url: dict[str, Any] = {"url": url}
    if isinstance(detail, str) and detail:
        image_url["detail"] = detail
    return {"type": "image_url", "image_url": image_url}


def _convert_tools(tools: list[dict]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        func = tool.get("function") if isinstance(tool.get("function"), dict) else None
        if not func:
            continue
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters") or {"type": "object"},
                },
            }
        )
    return converted


class OpenRouterUpstreamError(Exception):

    def __init__(self, message: str, *, error_type: str | None, code: Any = None) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.code = code


TRUNCATED_STREAM_ERROR_TYPE = "truncated_stream"


_COMPLETE_FINISH_REASONS = frozenset({"stop", "tool_calls", "function_call", "end_turn"})


def _reject_truncated_tool_calls(tool_calls: list[dict[str, Any]], finish_reason: Any) -> None:
    if isinstance(finish_reason, str) and finish_reason in _COMPLETE_FINISH_REASONS:
        return
    for call in tool_calls:
        arguments = (call.get("function") or {}).get("arguments")
        if not isinstance(arguments, str) or not arguments.strip():
            continue
        try:
            json.loads(arguments)
        except ValueError as exc:
            name = (call.get("function") or {}).get("name") or "<unnamed>"
            raise OpenRouterUpstreamError(
                f"stream ended with unparseable arguments for tool {name!r} "
                f"({len(arguments)} chars, finish_reason={finish_reason!r}): {exc}",
                error_type=TRUNCATED_STREAM_ERROR_TYPE,
            ) from exc


def _new_stream_state() -> dict[str, Any]:
    return {
        "content": [],
        "reasoning": [],
        "reasoning_details": {},
        "reasoning_details_extra": [],
        "tool_calls": {},
        "usage": None,
        "provider": None,
        "finish_reason": None,
        "error": None,
    }


def _accumulate_stream_chunk(state: dict[str, Any], chunk: Any) -> None:
    provider = _get(chunk, "provider")
    if isinstance(provider, str) and provider:
        state["provider"] = provider

    error = _get(chunk, "error")
    if error is None:
        choices_for_error = _get(chunk, "choices")
        if isinstance(choices_for_error, list) and choices_for_error:
            error = _get(choices_for_error[0], "error")
    if error is not None and state.get("error") is None:
        state["error"] = error
    usage = _get(chunk, "usage")
    if usage is not None:
        state["usage"] = usage
    choices = _get(chunk, "choices")
    if not isinstance(choices, list) or not choices:
        return
    choice = choices[0]
    finish_reason = _get(choice, "finish_reason")
    if finish_reason:
        state["finish_reason"] = finish_reason
    delta = _get(choice, "delta")
    if delta is None:
        return

    content = _get(delta, "content")
    if isinstance(content, str):
        state["content"].append(content)
    reasoning = _get(delta, "reasoning")
    if isinstance(reasoning, str):
        state["reasoning"].append(reasoning)
    _accumulate_reasoning_details(state, _serialize_json_value(_get(delta, "reasoning_details")))

    for tool_call in _get(delta, "tool_calls") or []:
        index = _get(tool_call, "index")
        slot = state["tool_calls"].setdefault(
            int(index) if isinstance(index, int) else len(state["tool_calls"]),
            {"id": None, "type": "function", "name": "", "arguments": ""},
        )
        call_id = _get(tool_call, "id")
        if isinstance(call_id, str) and call_id:
            slot["id"] = call_id
        call_type = _get(tool_call, "type")
        if isinstance(call_type, str) and call_type:
            slot["type"] = call_type
        function = _get(tool_call, "function")
        if function is None:
            continue
        name = _get(function, "name")
        if isinstance(name, str) and name:
            slot["name"] += name
        arguments = _get(function, "arguments")
        if isinstance(arguments, str):
            slot["arguments"] += arguments


_WHOLE_VALUE_REASONING_FIELDS = frozenset({"type", "format", "id"})


def _accumulate_reasoning_details(state: dict[str, Any], details: Any) -> None:
    if not isinstance(details, list):
        return
    for item in details:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        if not isinstance(index, int):
            state["reasoning_details_extra"].append(item)
            continue
        merged = state["reasoning_details"].get(index)
        if merged is None:
            state["reasoning_details"][index] = dict(item)
            continue
        for key, value in item.items():
            previous = merged.get(key)
            if (
                isinstance(value, str)
                and isinstance(previous, str)
                and key not in _WHOLE_VALUE_REASONING_FIELDS
            ):
                merged[key] = previous + value
            else:
                merged[key] = value


def _error_type_of_payload(error: Any) -> str | None:
    if error is None:
        return None
    metadata = _get(error, "metadata")
    for source in (metadata, error):
        if source is None:
            continue
        value = _get(source, "error_type")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _stream_state_to_response(state: dict[str, Any]) -> dict[str, Any]:

    error = state.get("error")
    if error is not None or state.get("finish_reason") == "error":
        message = _get(error, "message") if error is not None else None
        raise OpenRouterUpstreamError(
            str(message or "stream terminated with an in-band error"),
            error_type=_error_type_of_payload(error),
            code=_get(error, "code") if error is not None else None,
        )
    tool_calls = [
        {
            "id": slot["id"],
            "type": slot["type"],
            "function": {"name": slot["name"], "arguments": slot["arguments"]},
        }
        for _, slot in sorted(state["tool_calls"].items())
    ]
    _reject_truncated_tool_calls(tool_calls, state.get("finish_reason"))
    reasoning_details = [
        state["reasoning_details"][index] for index in sorted(state["reasoning_details"])
    ]
    reasoning_details.extend(state["reasoning_details_extra"])
    message: dict[str, Any] = {
        "content": "".join(state["content"]),
        "tool_calls": tool_calls,
        "reasoning": "".join(state["reasoning"]) or None,
        "reasoning_details": reasoning_details or None,
    }
    return {
        "choices": [{"message": message, "finish_reason": state["finish_reason"]}],
        "usage": state["usage"],
        "provider": state["provider"],
    }


def _first_choice(response: Any) -> Any:
    choices = _get(response, "choices")
    if isinstance(choices, list) and choices:
        return choices[0]
    return None


def _serialize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not tool_calls:
        return []
    serialized: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        item = _serialize_json_value(tool_call)
        if not isinstance(item, dict):
            continue
        call_id = item.get("id") or f"call_{uuid.uuid4().hex}"
        function = item.get("function") if isinstance(item.get("function"), dict) else {}
        serialized.append(
            {
                "id": call_id,
                "type": item.get("type") or "function",
                "function": {
                    "name": str(function.get("name") or ""),
                    "arguments": str(function.get("arguments") or ""),
                },
            }
        )
    return serialized


def _extract_usage(response: Any) -> dict[str, int] | None:
    usage = _get(response, "usage")
    if usage is None:
        return None

    def get(field: str) -> Any:
        return _get(usage, field)

    def get_from(obj: Any, field: str) -> Any:
        return _get(obj, field) if obj is not None else None

    prompt_tokens = get("prompt_tokens")
    completion_tokens = get("completion_tokens")
    total_tokens = get("total_tokens")
    cached_tokens = get("cached_tokens")
    reasoning_tokens = None
    details = get("prompt_tokens_details")
    if not isinstance(cached_tokens, int):
        cached_tokens = get_from(details, "cached_tokens")
    cache_write_tokens = get_from(details, "cache_write_tokens")
    completion_details = get("completion_tokens_details")
    reasoning_tokens = get_from(completion_details, "reasoning_tokens")
    cost_details = get("cost_details")

    cleaned: dict[str, int] = {}
    if isinstance(prompt_tokens, int):
        cleaned["prompt_tokens"] = prompt_tokens
    if isinstance(completion_tokens, int):
        cleaned["candidates_tokens"] = completion_tokens
    if isinstance(total_tokens, int):
        cleaned["total_tokens"] = total_tokens
    if isinstance(cached_tokens, int):
        cleaned["cached_tokens"] = cached_tokens
    if isinstance(cache_write_tokens, int):
        cleaned["cache_write_tokens"] = cache_write_tokens
    if isinstance(reasoning_tokens, int):
        cleaned["reasoning_tokens"] = reasoning_tokens

    cost_micro_usd = _micro_usd(get("cost"))
    if cost_micro_usd is not None:
        cleaned["cost_micro_usd"] = cost_micro_usd
    upstream_micro_usd = _micro_usd(get_from(cost_details, "upstream_inference_cost"))
    if upstream_micro_usd is not None:
        cleaned["upstream_cost_micro_usd"] = upstream_micro_usd
    return cleaned or None


def _micro_usd(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise CostAccountingError("OpenRouter returned an invalid charge")
    try:
        dollars = float(value)
        if not math.isfinite(dollars) or dollars < 0:
            raise ValueError("charge must be finite and non-negative")
        return round(dollars * 1_000_000)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CostAccountingError("OpenRouter returned an invalid charge") from exc


def _reasoning_config_from_thinking_level(thinking_level: str) -> dict[str, Any]:
    level = reasoning_level_alias(thinking_level)
    if level in {"0", "false", "none", "off", "disabled"}:
        return {"enabled": False}
    reasoning: dict[str, Any] = {"enabled": True}
    if level in {"low", "high"}:
        reasoning["effort"] = level
    return reasoning


def _reasoning_details_summary(reasoning_details: Any) -> str:
    if not isinstance(reasoning_details, list):
        return ""
    fragments: list[str] = []
    for item in reasoning_details:
        if not isinstance(item, dict):
            continue
        text = item.get("summary") or item.get("text")
        if isinstance(text, str) and text.strip():
            fragments.append(text.strip())
    return "\n".join(fragments).strip()


def _serialize_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_serialize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize_json_value(child) for key, child in value.items()}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json", exclude_none=True, warnings="none")
    dump_json = getattr(value, "model_dump_json", None)
    if callable(dump_json):
        return json.loads(dump_json(exclude_none=True))
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict):
        return {
            str(key): _serialize_json_value(child)
            for key, child in attrs.items()
            if not key.startswith("_")
        }
    return str(value)


def _estimate_prompt_tokens(payload: dict[str, Any], image_caps: tuple[int, int]) -> int:

    messages = payload.get("messages")
    tools = payload.get("tools")
    msg_size, image_tokens = _text_and_image_tokens(messages, image_caps)
    tool_size, _ = _text_and_image_tokens(tools, image_caps)
    text_size = msg_size + tool_size

    structural_overhead = 128
    if isinstance(messages, list):
        structural_overhead += len(messages) * 16
    if isinstance(tools, list):
        structural_overhead += len(tools) * 128
    return max(1, (text_size + 2) // 3 + structural_overhead + image_tokens)


def _text_and_image_tokens(value: Any, image_caps: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, dict):
        if value.get("type") == "image_url" and isinstance(value.get("image_url"), dict):
            image_url = value["image_url"]
            size = len("type") + _json_text_size(value.get("type"))
            size += len("image_url")
            img_tokens = image_caps[0]
            for skey, sval in image_url.items():
                if skey == "url" and isinstance(sval, str) and "base64," in sval:
                    img_tokens = data_url_image_tokens(sval, image_caps)
                    continue
                size += len(str(skey)) + _json_text_size(sval)
            return size, img_tokens
        total = 0
        images = 0
        for key, child in value.items():
            child_size, child_images = _text_and_image_tokens(child, image_caps)
            total += len(str(key)) + child_size
            images += child_images
        return total, images
    if isinstance(value, list):
        total = 0
        images = 0
        for item in value:
            child_size, child_images = _text_and_image_tokens(item, image_caps)
            total += child_size
            images += child_images
        return total, images
    return _json_text_size(value), 0


def _json_text_size(value: Any) -> int:
    if value is None:
        return 0
    try:
        return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    except Exception:
        return len(str(value))


def _get(obj: Any, field: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(field)
    return getattr(obj, field, None)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    value = _env_int_opt(name)
    return default if value is None else value


def _env_int_opt(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        return int(raw.strip().replace("_", ""))
    except Exception:
        return None


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _openrouter_default_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    referer = os.environ.get("OPENROUTER_HTTP_REFERER") or os.environ.get("OPENROUTER_SITE_URL")
    title = os.environ.get("OPENROUTER_APP_TITLE") or os.environ.get("OPENROUTER_SITE_NAME")
    if referer and referer.strip():
        headers["HTTP-Referer"] = referer.strip()
    if title and title.strip():
        headers["X-OpenRouter-Title"] = title.strip()
    return headers


def _image_path_to_data_url(image_path: str) -> str:
    path = Path(image_path).expanduser().resolve()
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type:
        mime_type = "application/octet-stream"
    raw, mime_type = image_bytes_for_wire(path, path.read_bytes(), mime_type)
    data = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{data}"


def _extract_http_status(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int) and 100 <= status <= 599:
        return status
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None) or getattr(response, "status", None)
        if isinstance(status, int) and 100 <= status <= 599:
            return status
    return None


_BACKPRESSURE_ERROR_TYPES = frozenset(
    {"rate_limit_exceeded", "provider_overloaded", "provider_unavailable"}
)
_RETRYABLE_ERROR_TYPES = _BACKPRESSURE_ERROR_TYPES | {
    "server",
    "timeout",
    "unmapped",
    TRUNCATED_STREAM_ERROR_TYPE,
}

_TERMINAL_ERROR_TYPES = frozenset(
    {
        "authentication",
        "permission_denied",
        "payment_required",
        "invalid_request",
        "invalid_prompt",
        "not_found",
        "precondition_failed",
        "payload_too_large",
        "unprocessable",
        "context_length_exceeded",
        "max_tokens_exceeded",
        "token_limit_exceeded",
        "string_too_long",
        "content_policy_violation",
        "refusal",
        "invalid_image",
        "image_too_large",
        "image_too_small",
        "unsupported_image_format",
        "image_not_found",
        "image_download_failed",
    }
)
PAYLOAD_TOO_LARGE_ERROR_TYPE = "payload_too_large"


def is_payload_too_large(exc: BaseException) -> bool:
    if openrouter_error_type(exc) == PAYLOAD_TOO_LARGE_ERROR_TYPE:
        return True
    return _extract_http_status(exc) == 413


def openrouter_error_type(exc: BaseException) -> str | None:
    if isinstance(exc, OpenRouterUpstreamError):
        return exc.error_type
    body = getattr(exc, "body", None)
    for payload in (body, getattr(exc, "response", None)):
        if payload is None:
            continue
        error = _get(payload, "error") if not isinstance(payload, dict) else payload.get("error")
        found = _error_type_of_payload(error if error is not None else payload)
        if found:
            return found
    return None


def _is_rate_limited(exc: BaseException) -> bool:
    error_type = openrouter_error_type(exc)
    if error_type:
        return error_type in _BACKPRESSURE_ERROR_TYPES
    if _extract_http_status(exc) == 429:
        return True
    return "rate limit" in str(exc).lower() or "rate-limited" in str(exc).lower()


def _should_retry_openrouter_exception(exc: BaseException) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    if isinstance(exc, json.JSONDecodeError):
        return True

    if isinstance(exc, OpenRouterUpstreamError):
        return exc.error_type not in _TERMINAL_ERROR_TYPES

    error_type = openrouter_error_type(exc)
    if error_type:
        if error_type in _TERMINAL_ERROR_TYPES:
            return False
        if error_type in _RETRYABLE_ERROR_TYPES:
            return True
    status = _extract_http_status(exc)
    if status is not None:
        if status in {408, 409, 425, 429, 500, 502, 503, 504}:
            return True
        if 400 <= status < 500:
            return False
        if status >= 500:
            return True
    message = str(exc).lower()
    return any(
        needle in message
        for needle in (
            "timeout",
            "timed out",
            "connection error",
            "connection reset",
            "connection aborted",
            "server disconnected",
            "protocol error",
            "temporarily unavailable",
            "rate limit",
            "bad gateway",
            "service unavailable",
            "overloaded",
            "provider returned error",
            "inappropriate content",
        )
    )


def _retry_after_seconds(exc: BaseException, *, cap: float) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = None
    for key in ("retry-after", "Retry-After"):
        try:
            raw = headers.get(key)
        except Exception:  # noqa: BLE001 - a header mapping we do not recognize
            return None
        if raw:
            break
    if not raw:
        return None
    text = str(raw).strip()
    try:
        seconds = float(text)
    except ValueError:
        try:
            when = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
        if when is None:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = (when - datetime.now(timezone.utc)).total_seconds()
    if seconds <= 0:
        return None
    return min(seconds, max(0.0, cap))


def _format_retry_exception(exc: BaseException) -> str:
    status = _extract_http_status(exc)
    message = str(exc).strip()
    summary = type(exc).__name__
    if status is not None:
        summary += f" (HTTP {status})"
    if message:
        return f"{summary}: {message}"
    return f"{summary}: {repr(exc)}"


async def _async_retry(
    operation: Any,
    *,
    max_attempts: int,
    should_retry: Any,
    base_delay: float,
    max_delay: float,
    logger: logging.Logger,
    context: str,
    sleep_fn: Any = asyncio.sleep,
    rng: Any = random.random,
    delay_hint: Any = None,
    on_retry: Any = None,
    max_attempts_for: Any = None,
) -> Any:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be >= 1")

    attempt = 0
    while True:
        try:
            return await operation()
        except Exception as exc:
            attempt += 1

            budget = max_attempts
            if max_attempts_for is not None:
                override = max_attempts_for(exc)
                if override:
                    budget = max(budget, int(override))
            if attempt >= budget or not should_retry(exc):
                raise

            curve = max(0.0, float(rng()) * min(max_delay, base_delay * (2 ** (attempt - 1))))
            delay = curve
            if delay_hint is not None:
                hint = delay_hint(exc)
                if hint is not None:
                    delay = max(float(hint), curve)
            if on_retry is not None:
                on_retry()
            logger.warning(
                "%s failed (attempt %s/%s), retrying in %.2fs: %s",
                context,
                attempt,
                budget,
                delay,
                _format_retry_exception(exc),
            )
            await sleep_fn(delay)
