
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse

from artisan.values import ThinkingLevel, provider_reasoning_level
from pipeline.llm import openai_codec
from pipeline.llm.base import (
    CompactionEvent,
    ContextWindowPressure,
    PrepareRequestResult,
    ProviderTraceEvent,
    build_context_window_pressure,
)
from pipeline.llm.compaction_policy import decide_compaction
from pipeline.llm.credentials import openai_api_key_from_env as openai_api_key_from_env
from pipeline.llm.credentials import openai_api_keys_from_env as openai_api_keys_from_env
from pipeline.llm.endpoints import official_openai_base_url
from pipeline.llm.errors import ProviderConfigurationError
from pipeline.llm.model_registry import (
    DEFAULT_COMPACTION_MODEL,
    DEFAULT_TEXT_MODEL,
    compaction_model_from_env,
    validate_model,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CompactionResult:

    items: list[dict[str, Any]]
    usage: Optional[dict[str, int]]
    model_id: str


DEFAULT_OPENAI_MODEL = DEFAULT_TEXT_MODEL
DEFAULT_OPENAI_HARD_PRESSURE_TOKENS = 272_000
DEFAULT_OPENAI_CONTEXT_TOKENS = 1_050_000
DEFAULT_OPENAI_COMPACTION_MODEL = DEFAULT_COMPACTION_MODEL
DEFAULT_OPENAI_COST_TRIGGER_TOKENS = 110_000
OPENAI_COMPACTION_COST_TRIGGER_ENV = "OPENAI_COMPACTION_COST_TRIGGER_TOKENS"
OPENAI_CANONICAL_CHECKPOINT_FORMAT_VERSION = 1
USAGE_NEVER_OBSERVED = "never_observed"
USAGE_OBSERVED = "observed"
USAGE_INVALIDATED_BY_BOUNDARY_REBUILD = "invalidated_by_boundary_rebuild"
USAGE_UNAVAILABLE_AFTER_CHECKPOINT_RESTORE = "unavailable_after_checkpoint_restore"


class OpenAILLM:

    supports_native_cost_compaction = True

    def __init__(
        self,
        model_id: str = DEFAULT_OPENAI_MODEL,
        *,
        compaction_model_id: Optional[str] = None,
        thinking_level: str = "high",
        reasoning_summary: Optional[str] = "auto",
        transport: str | None = None,
        prompt_cache_key: Optional[str] = None,
        prompt_cache_retention: Optional[str] = None,
        canonical_contract_fingerprint: Optional[str] = None,
        store: Optional[bool] = None,
        dry_run: bool = False,
    ) -> None:
        self.model_id = validate_model(model_id, "text")
        configured_compaction_model = compaction_model_id or compaction_model_from_env()
        self.compaction_model_id = validate_model(configured_compaction_model, "compaction")
        self.reasoning_effort = _effort_from_thinking_level(thinking_level, self.model_id)
        self.reasoning_summary = _normalize_reasoning_summary(reasoning_summary)
        self.transport = _normalize_transport(
            transport or os.environ.get("OPENAI_TRANSPORT") or "http"
        )
        self.prompt_cache_key = _normalize_prompt_cache_key(prompt_cache_key)
        self.prompt_cache_retention = _normalize_prompt_cache_retention(prompt_cache_retention)
        if self.model_id == "gpt-6-astra" and self.prompt_cache_retention not in {None, "30m"}:
            raise ProviderConfigurationError(
                "GPT-6 Astra cache lifetime is 30m. Unset OPENAI_PROMPT_CACHE_RETENTION "
                "or set it to 30m; legacy in_memory/24h retention is unsupported."
            )
        self.canonical_contract_fingerprint = (
            str(canonical_contract_fingerprint).strip()
            if canonical_contract_fingerprint is not None
            else None
        )
        self.cost_trigger_tokens: int | None = None
        self.store = self.transport == "websocket" if store is None else bool(store)
        self.request_timeout_seconds = _env_float("OPENAI_REQUEST_TIMEOUT_SECONDS", 900.0)
        self.max_attempts = max(1, int(_env_float("OPENAI_MAX_ATTEMPTS", 6)))
        self.retry_base_seconds = _env_float("OPENAI_RETRY_BASE_SECONDS", 1.0)
        self.retry_max_seconds = _env_float("OPENAI_RETRY_MAX_SECONDS", 30.0)
        self.websocket_open_timeout_seconds = _env_float(
            "OPENAI_WEBSOCKET_OPEN_TIMEOUT_SECONDS", 20.0
        )
        self.websocket_max_connection_age_seconds = _env_float(
            "OPENAI_WEBSOCKET_MAX_CONNECTION_AGE_SECONDS", 3300.0
        )
        base_url = official_openai_base_url(os.environ.get("OPENAI_BASE_URL"))
        self._api_key: Optional[str] = None
        if dry_run:
            self._client = None
            self._client_is_async = False
        else:
            api_key = openai_api_key_from_env()
            if not api_key:
                raise ValueError(
                    "OpenAI credentials not found. Set OPENAI_API_KEY or OPENAI_API_KEYS."
                )
            self._api_key = api_key

            self._client: Any
            self._client_is_async: bool
            client_kwargs: dict[str, Any] = {
                "api_key": api_key,
                "base_url": base_url,
                "max_retries": 0,
            }
            if self.request_timeout_seconds and self.request_timeout_seconds > 0:
                client_kwargs["timeout"] = float(self.request_timeout_seconds)
            try:
                from openai import AsyncOpenAI  # type: ignore

                self._client = AsyncOpenAI(**client_kwargs)
                self._client_is_async = True
            except Exception:
                try:
                    from openai import OpenAI  # type: ignore
                except Exception as exc:  # pragma: no cover
                    raise RuntimeError(
                        "OpenAI provider selected but the `openai` package is not installed. "
                        "Install it (e.g. `uv add openai`), then try again."
                    ) from exc

                self._client = OpenAI(**client_kwargs)
                self._client_is_async = False

        self._responses_websocket_url = _responses_websocket_url(base_url)
        self.openai_wire_fingerprint = _openai_wire_fingerprint(base_url)

        self._input_items: list[dict[str, Any]] = []
        self._pending_incremental_input_items: list[dict[str, Any]] = []
        self._last_message_count: int = 0
        self._previous_response_id: Optional[str] = None
        self._last_response_start_index: Optional[int] = None
        self._last_usage: Optional[dict[str, int]] = None
        self._usage_validity = USAGE_NEVER_OBSERVED
        self._request_sequence = 0
        self._last_compaction_turn_number: Optional[int] = None
        self._last_soft_compaction_failure_sig: Optional[str] = None
        self._last_soft_compaction_turn_number: Optional[int] = None
        self._last_soft_compaction_prompt_tokens: Optional[int] = None
        self._unsupported_threshold_event_emitted = False
        self.persistent_canonical_active = False
        self._websocket: Any = None
        self._websocket_opened_at: Optional[float] = None

    def build_request_preview(
        self,
        *,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
    ) -> dict[str, Any]:

        converted_tools = self._convert_tools(tools)

        self.reset_canonical_state(usage_validity=USAGE_NEVER_OBSERVED)
        self._append_new_inputs(messages)

        return self._build_request_payload(
            system_prompt=system_prompt,
            tools=converted_tools,
            input_items=self._input_items,
        )

    def reset_canonical_state(
        self,
        *,
        usage_validity: str = USAGE_INVALIDATED_BY_BOUNDARY_REBUILD,
    ) -> None:
        self.reset_canonical_history(
            preserve_policy_state=False,
            usage_validity=usage_validity,
        )

    def reset_canonical_history(
        self,
        *,
        preserve_policy_state: bool = True,
        usage_validity: str = USAGE_INVALIDATED_BY_BOUNDARY_REBUILD,
    ) -> None:
        request_sequence = self._request_sequence
        last_compaction_sequence = self._last_compaction_turn_number
        last_soft_failure_sig = self._last_soft_compaction_failure_sig
        last_soft_sequence = self._last_soft_compaction_turn_number
        last_soft_prompt_tokens = self._last_soft_compaction_prompt_tokens
        unsupported_emitted = self._unsupported_threshold_event_emitted
        persistent_active = self.persistent_canonical_active

        self._input_items = []
        self._pending_incremental_input_items = []
        self._last_message_count = 0
        self._previous_response_id = None
        self._last_response_start_index = None
        self._last_usage = None
        self._usage_validity = usage_validity
        self._last_compaction_turn_number = None
        self._last_soft_compaction_failure_sig = None
        self._last_soft_compaction_turn_number = None
        self._last_soft_compaction_prompt_tokens = None
        self._unsupported_threshold_event_emitted = False
        self.persistent_canonical_active = False
        if preserve_policy_state:
            self._request_sequence = request_sequence
            self._last_compaction_turn_number = last_compaction_sequence
            self._last_soft_compaction_failure_sig = last_soft_failure_sig
            self._last_soft_compaction_turn_number = last_soft_sequence
            self._last_soft_compaction_prompt_tokens = last_soft_prompt_tokens
            self._unsupported_threshold_event_emitted = unsupported_emitted
            self.persistent_canonical_active = persistent_active
        else:
            self._request_sequence = 0

    def sync_canonical_state(self, conversation: list[dict[str, Any]]) -> None:
        self._append_new_inputs(conversation)
        self._last_message_count = len(conversation)

    def export_canonical_state(
        self,
        conversation: list[dict[str, Any]],
        *,
        conversation_digest: str,
    ) -> dict[str, Any]:
        if not self.canonical_contract_fingerprint:
            raise ValueError("canonical contract fingerprint is required for export")
        self.sync_canonical_state(conversation)
        self._pending_incremental_input_items = []
        self._previous_response_id = None
        assert not self._pending_incremental_input_items

        return {
            "header": {
                "format_version": OPENAI_CANONICAL_CHECKPOINT_FORMAT_VERSION,
                "provider": "openai",
                "model_id": self.model_id,
                "canonical_contract_fingerprint": self.canonical_contract_fingerprint,
                "prompt_cache_key": self.prompt_cache_key,
                "openai_wire_fingerprint": self.openai_wire_fingerprint,
                "conversation_digest": conversation_digest,
                "item_count": len(self._input_items),
                "message_count": self._last_message_count,
            },
            "state": {
                "input_items": self._input_items,
                "last_message_count": self._last_message_count,
                "last_response_start_index": self._last_response_start_index,
                "previous_response_id": None,
                "last_compaction_turn_number": self._last_compaction_turn_number,
                "last_soft_compaction_failure_sig": (self._last_soft_compaction_failure_sig),
                "last_soft_compaction_turn_number": (self._last_soft_compaction_turn_number),
                "last_soft_compaction_prompt_tokens": (self._last_soft_compaction_prompt_tokens),
                "request_sequence": self._request_sequence,
                "unsupported_threshold_event_emitted": (self._unsupported_threshold_event_emitted),
            },
        }

    def import_canonical_state(
        self,
        checkpoint: dict[str, Any],
        *,
        conversation_digest: str,
        conversation_message_count: int,
    ) -> bool:
        self._last_import_replaced_state = False
        if not isinstance(checkpoint, dict) or not self.canonical_contract_fingerprint:
            return False
        header = checkpoint.get("header")
        state = checkpoint.get("state")
        if not isinstance(header, dict) or not isinstance(state, dict):
            return False
        expected_header = {
            "format_version": OPENAI_CANONICAL_CHECKPOINT_FORMAT_VERSION,
            "provider": "openai",
            "model_id": self.model_id,
            "canonical_contract_fingerprint": self.canonical_contract_fingerprint,
            "openai_wire_fingerprint": self.openai_wire_fingerprint,
            "conversation_digest": conversation_digest,
        }
        if any(header.get(key) != value for key, value in expected_header.items()):
            return False
        stored_cache_key = header.get("prompt_cache_key")
        if (
            stored_cache_key is not None
            and self.prompt_cache_key is not None
            and stored_cache_key != self.prompt_cache_key
        ):
            return False

        input_items = state.get("input_items")
        last_message_count = state.get("last_message_count")
        last_response_start_index = state.get("last_response_start_index")
        request_sequence = state.get("request_sequence")
        if not isinstance(input_items, list) or not all(
            isinstance(item, dict) for item in input_items
        ):
            return False
        if not isinstance(last_message_count, int) or last_message_count < 0:
            return False
        if header.get("message_count") != last_message_count:
            return False
        if last_message_count != conversation_message_count:
            return False
        if header.get("item_count") != len(input_items):
            return False
        if last_response_start_index is not None and (
            not isinstance(last_response_start_index, int)
            or not 0 <= last_response_start_index <= len(input_items)
        ):
            return False
        if not isinstance(request_sequence, int) or request_sequence < 0:
            return False
        if state.get("previous_response_id") is not None:
            return False
        for key in (
            "last_compaction_turn_number",
            "last_soft_compaction_turn_number",
            "last_soft_compaction_prompt_tokens",
        ):
            value = state.get(key)
            if value is not None and (not isinstance(value, int) or value < 0):
                return False
        soft_sig = state.get("last_soft_compaction_failure_sig")
        if soft_sig is not None and not isinstance(soft_sig, str):
            return False
        if not isinstance(state.get("unsupported_threshold_event_emitted", False), bool):
            return False

        restored_items = json.loads(json.dumps(input_items))
        self._last_import_replaced_state = (
            self._input_items != restored_items
            or self._last_message_count != last_message_count
            or self._last_response_start_index != last_response_start_index
        )
        self._input_items = restored_items
        self._pending_incremental_input_items = []
        self._last_message_count = last_message_count
        self._previous_response_id = None
        self._last_response_start_index = last_response_start_index
        if self._last_import_replaced_state:
            self._last_usage = None
            self._usage_validity = USAGE_UNAVAILABLE_AFTER_CHECKPOINT_RESTORE
        self._last_compaction_turn_number = _optional_nonnegative_int(
            state.get("last_compaction_turn_number")
        )
        self._last_soft_compaction_failure_sig = soft_sig if isinstance(soft_sig, str) else None
        self._last_soft_compaction_turn_number = _optional_nonnegative_int(
            state.get("last_soft_compaction_turn_number")
        )
        self._last_soft_compaction_prompt_tokens = _optional_nonnegative_int(
            state.get("last_soft_compaction_prompt_tokens")
        )
        self._request_sequence = request_sequence
        self._unsupported_threshold_event_emitted = bool(
            state.get("unsupported_threshold_event_emitted", False)
        )
        self.persistent_canonical_active = True
        return True

    def enable_cost_compaction_for_session(self) -> None:
        raw = os.environ.get(OPENAI_COMPACTION_COST_TRIGGER_ENV)
        if raw is not None:
            self.cost_trigger_tokens = _env_int_or_none(OPENAI_COMPACTION_COST_TRIGGER_ENV)
            return
        self.cost_trigger_tokens = DEFAULT_OPENAI_COST_TRIGGER_TOKENS

    def user_message_fingerprint(self, message: dict[str, Any]) -> str | None:
        if not isinstance(message, dict) or message.get("role") != "user":
            return None
        content = self._convert_message_content(message.get("content"))
        if not content:
            return None
        return _user_item_fingerprint({"role": "user", "content": content})

    def context_window_pressure(self, usage: dict[str, int]) -> ContextWindowPressure:
        return build_context_window_pressure(
            provider="openai",
            usage=usage,
            max_context_tokens=_context_window_tokens_for_model(self.model_id),
            hard_pressure_tokens=_hard_pressure_threshold_for_model(self.model_id),
        )

    async def prepare_next_request(
        self,
        *,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
        completed_turns: int,
        consecutive_compile_failure_count: int = 0,
        last_compile_failure_sig: Optional[str] = None,
        entry_reason: Optional[str] = None,
    ) -> PrepareRequestResult:
        result = PrepareRequestResult()
        self._request_sequence += 1
        request_sequence = self._request_sequence
        prior_prompt_tokens = self._last_prompt_tokens()
        prior_usage_validity = self._usage_validity

        self._append_new_inputs(messages)
        if consecutive_compile_failure_count <= 0 or not last_compile_failure_sig:
            self._last_soft_compaction_failure_sig = None

        repair_entry = entry_reason == "review_repair"
        repair_guard_outcome = "not_evaluated"

        def _finish(*, compact_ran: bool = False) -> PrepareRequestResult:
            if repair_entry:
                result.trace_events.append(
                    ProviderTraceEvent(
                        event_type="compaction_repair_entry",
                        payload={
                            "provider": "openai",
                            "model_id": self.model_id,
                            "prior_prompt_tokens": prior_prompt_tokens,
                            "usage_validity": prior_usage_validity,
                            "raw_tail_guard": repair_guard_outcome,
                            "compact_ran": compact_ran,
                            "request_sequence": request_sequence,
                        },
                    )
                )
            return result

        if request_sequence <= 2:
            return _finish()

        if self._usage_validity != USAGE_OBSERVED:
            if repair_entry:
                result.trace_events.append(
                    ProviderTraceEvent(
                        event_type="compaction_skipped",
                        payload={
                            "reason": "usage_unavailable",
                            "usage_validity": self._usage_validity,
                            "provider": "openai",
                            "model_id": self.model_id,
                            "request_sequence": request_sequence,
                        },
                    )
                )
            return _finish()

        prompt_tokens = self._last_prompt_tokens()
        hard_threshold = _hard_pressure_threshold_for_model(self.model_id)
        if (
            hard_threshold is None
            and prompt_tokens is not None
            and not self._unsupported_threshold_event_emitted
        ):
            self._unsupported_threshold_event_emitted = True
            result.trace_events.append(
                ProviderTraceEvent(
                    event_type="compaction_skipped",
                    payload={
                        "reason": "unsupported_threshold_model",
                        "model_id": self.model_id,
                        "prompt_tokens": prompt_tokens,
                        "turn_before_request": completed_turns,
                        "request_sequence": request_sequence,
                    },
                )
            )

        immutable_prefix_items, compactable_items, raw_tail_items = (
            self._partition_compaction_items()
        )
        decision = decide_compaction(
            prompt_tokens=prompt_tokens,
            cached_tokens=self._last_cached_tokens(),
            hard_threshold=hard_threshold,
            consecutive_compile_failure_count=consecutive_compile_failure_count,
            last_compile_failure_sig=last_compile_failure_sig,
            last_soft_compaction_failure_sig=self._last_soft_compaction_failure_sig,
            compactable_item_count=len(compactable_items),
            turn_number=request_sequence,
            last_soft_compaction_turn_number=self._last_soft_compaction_turn_number,
            last_soft_compaction_prompt_tokens=self._last_soft_compaction_prompt_tokens,
            cost_trigger_tokens=self.cost_trigger_tokens,
        )
        trigger = decision.trigger
        if trigger is None:
            if consecutive_compile_failure_count > 0 and last_compile_failure_sig:
                result.trace_events.append(
                    ProviderTraceEvent(
                        event_type="compaction_skipped",
                        payload={
                            "reason": decision.reason,
                            "model_id": self.model_id,
                            "prompt_tokens": prompt_tokens,
                            "cached_tokens": self._last_cached_tokens(),
                            "turn_before_request": completed_turns,
                            "failure_streak": consecutive_compile_failure_count,
                            "pressure_ratio": decision.pressure_ratio,
                            "cache_ratio": decision.cache_ratio,
                            "soft_failure_threshold": decision.soft_failure_threshold,
                            "min_compactable_items": decision.min_compactable_items,
                            "hard_trigger_tokens": decision.hard_trigger_tokens,
                            "cost_trigger_tokens": decision.cost_trigger_tokens,
                            "request_sequence": request_sequence,
                        },
                    )
                )
            return _finish()

        if not compactable_items:
            result.trace_events.append(
                ProviderTraceEvent(
                    event_type="compaction_skipped",
                    payload={
                        "reason": "nothing_to_compact",
                        "trigger": trigger,
                        "model_id": self.model_id,
                        "turn_before_request": completed_turns,
                        "request_sequence": request_sequence,
                    },
                )
            )
            return _finish()

        immutable_fingerprints = _user_item_fingerprints(immutable_prefix_items)
        raw_tail_fingerprints = _user_item_fingerprints(raw_tail_items)
        if repair_entry:
            repair_message = next(
                (
                    message
                    for message in reversed(messages)
                    if isinstance(message, dict) and message.get("role") == "user"
                ),
                None,
            )
            repair_fingerprint = (
                self.user_message_fingerprint(repair_message)
                if isinstance(repair_message, dict)
                else None
            )
            if repair_fingerprint and repair_fingerprint in raw_tail_fingerprints:
                repair_guard_outcome = "present"
            else:
                repair_guard_outcome = "missing"
                result.trace_events.append(
                    ProviderTraceEvent(
                        event_type="compaction_skipped",
                        payload={
                            "reason": "repair_task_not_in_raw_tail",
                            "provider": "openai",
                            "model_id": self.model_id,
                            "trigger": trigger,
                            "request_sequence": request_sequence,
                        },
                    )
                )
                return _finish()

        before_item_count = len(self._input_items)
        compaction = await self._compact_inputs(
            system_prompt=system_prompt,
            input_items=compactable_items,
        )
        compacted_items = compaction.items
        if not compacted_items:
            raise RuntimeError(
                "OpenAI compaction returned no replacement items; "
                "refusing to discard compactable history"
            )
        self._input_items = [*immutable_prefix_items, *compacted_items, *raw_tail_items]
        self._pending_incremental_input_items = []
        self._last_response_start_index = len(immutable_prefix_items) + len(compacted_items)
        self._previous_response_id = None
        self._last_compaction_turn_number = request_sequence
        if trigger in {"compile_plateau", "cost_pressure"}:
            self._last_soft_compaction_turn_number = request_sequence
            self._last_soft_compaction_prompt_tokens = prompt_tokens
        if trigger == "compile_plateau":
            self._last_soft_compaction_failure_sig = last_compile_failure_sig

        usage = compaction.usage
        event = CompactionEvent(
            turn_before_request=completed_turns,
            trigger=trigger,
            provider="openai",
            model_id=compaction.model_id,
            usage=usage,
            trigger_prompt_tokens=prompt_tokens,
            before_next_input_tokens=None,
            after_next_input_tokens=None,
            estimated_saved_next_input_tokens=None,
            before_item_count=before_item_count,
            after_item_count=len(self._input_items),
            previous_response_id_cleared=True,
            immutable_prefix_item_count=len(immutable_prefix_items),
            compactable_item_count=len(compactable_items),
            raw_tail_item_count=len(raw_tail_items),
            retained_user_message_fingerprints=[
                *immutable_fingerprints,
                *raw_tail_fingerprints,
            ],
            request_sequence=request_sequence,
            guardrails={
                "min_turns_satisfied": True,
                "hard_trigger_tokens": decision.hard_trigger_tokens,
                "cost_trigger_tokens": decision.cost_trigger_tokens,
                "pressure_ratio": decision.pressure_ratio,
                "cache_ratio": decision.cache_ratio,
                "soft_failure_threshold": decision.soft_failure_threshold,
                "raw_tail_preserved": True,
            },
        )
        self.persistent_canonical_active = True
        result.compaction_event = event
        result.trace_events.append(
            ProviderTraceEvent(
                event_type="compaction",
                payload=event.to_dict(),
            )
        )
        return _finish(compact_ran=True)

    async def generate_with_tools(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
    ) -> dict:
        converted_tools = self._convert_tools(tools)
        self._append_new_inputs(messages)
        incremental_input_items = list(self._pending_incremental_input_items)

        request_payload = self._build_request_payload(
            system_prompt=system_prompt,
            tools=converted_tools,
            input_items=self._input_items,
        )
        incremental_request_payload = self._build_request_payload(
            system_prompt=system_prompt,
            tools=converted_tools,
            input_items=incremental_input_items,
            previous_response_id=self._previous_response_id,
        )
        fallback_request_payload = self._build_request_payload(
            system_prompt=system_prompt,
            tools=converted_tools,
            input_items=self._input_items,
        )

        async def _request_once() -> Any:
            try:
                return await self._request_with_transport(
                    request_payload=request_payload,
                    incremental_request_payload=incremental_request_payload,
                    fallback_request_payload=fallback_request_payload,
                )
            except Exception as exc:
                reasoning = request_payload.get("reasoning")
                if not (isinstance(reasoning, dict) and "summary" in reasoning):
                    raise
                if not _is_reasoning_summary_unsupported_error(exc):
                    raise
                logger.warning(
                    "OpenAI request rejected reasoning.summary for model=%s; retrying without it: %s",
                    self.model_id,
                    _format_retry_exception(exc),
                )
                return await self._request_with_transport(
                    request_payload=_payload_without_reasoning_summary(request_payload),
                    incremental_request_payload=_payload_without_reasoning_summary(
                        incremental_request_payload
                    ),
                    fallback_request_payload=_payload_without_reasoning_summary(
                        fallback_request_payload
                    ),
                )

        response = await _async_retry(
            _request_once,
            max_attempts=self.max_attempts,
            should_retry=_should_retry_openai_exception,
            base_delay=self.retry_base_seconds,
            max_delay=self.retry_max_seconds,
            logger=logger,
            context=f"openai[{self.transport}]",
        )

        response_start_index = len(self._input_items)
        provider_output_items = self._serialize_response_output(response)
        self._input_items.extend(provider_output_items)
        self._last_response_start_index = response_start_index
        self._previous_response_id = self._extract_response_id(response)
        self._last_usage = self._extract_usage(response)
        self._usage_validity = USAGE_OBSERVED
        self._pending_incremental_input_items = []

        converted = self._convert_response(response)
        meta = converted.get("_meta")
        if not isinstance(meta, dict):
            meta = {}
        converted["_meta"] = {
            **meta,
            "provider_output_items": json.loads(json.dumps(provider_output_items)),
            "openai_wire_fingerprint": self.openai_wire_fingerprint,
        }
        return converted

    async def close(self) -> None:
        await self._close_websocket()
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

    def _build_request_payload(
        self,
        *,
        system_prompt: str,
        tools: list[dict[str, Any]],
        input_items: list[dict[str, Any]],
        previous_response_id: Optional[str] = None,
    ) -> dict[str, Any]:
        request_payload: dict[str, Any] = {
            "model": self.model_id,
            "instructions": system_prompt,
            "tools": tools,
            "parallel_tool_calls": True,
            "input": input_items,
            "store": self.store,
            "include": ["reasoning.encrypted_content"],
        }

        reasoning: dict[str, Any] = {"effort": self.reasoning_effort}
        if self.reasoning_summary:
            reasoning["summary"] = self.reasoning_summary
        request_payload["reasoning"] = reasoning
        if self.prompt_cache_key:
            request_payload["prompt_cache_key"] = self.prompt_cache_key
        if self.prompt_cache_retention and self.model_id != "gpt-6-astra":
            request_payload["prompt_cache_retention"] = self.prompt_cache_retention
        if previous_response_id:
            request_payload["previous_response_id"] = previous_response_id
        return request_payload

    async def _compact_inputs(
        self,
        *,
        system_prompt: str,
        input_items: list[dict[str, Any]],
    ) -> _CompactionResult:
        validate_model(self.compaction_model_id, "compaction")

        async def _compact_once() -> Any:
            if self._client is None:
                raise RuntimeError("OpenAI compaction requires a real API client")
            request_kwargs: dict[str, Any] = {
                "model": self.compaction_model_id,
                "input": input_items,
                "instructions": system_prompt,
            }
            if self.prompt_cache_key:
                request_kwargs["prompt_cache_key"] = self.prompt_cache_key
            if self._client_is_async:
                return await self._client.responses.compact(**request_kwargs)
            return await asyncio.to_thread(self._client.responses.compact, **request_kwargs)

        async def _request_once() -> Any:
            return await _compact_once()

        response = await _async_retry(
            _request_once,
            max_attempts=self.max_attempts,
            should_retry=_should_retry_openai_exception,
            base_delay=self.retry_base_seconds,
            max_delay=self.retry_max_seconds,
            logger=logger,
            context=f"openai.compact[{self.transport}]",
        )
        return _CompactionResult(
            items=self._serialize_response_output(response),
            usage=self._extract_usage(response),
            model_id=self.compaction_model_id,
        )

    def _partition_compaction_items(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        immutable_prefix_count = self._immutable_prefix_count()
        raw_tail_start = self._last_response_start_index
        if raw_tail_start is None:
            raw_tail_start = len(self._input_items)
        raw_tail_start = max(immutable_prefix_count, min(raw_tail_start, len(self._input_items)))
        immutable_prefix_items = list(self._input_items[:immutable_prefix_count])
        compactable_items = list(self._input_items[immutable_prefix_count:raw_tail_start])
        raw_tail_items = list(self._input_items[raw_tail_start:])
        return immutable_prefix_items, compactable_items, raw_tail_items

    def _immutable_prefix_count(self) -> int:
        count = 0
        for item in self._input_items:
            if count >= 2:
                break
            if not openai_codec.is_user_message_item(item):
                break
            count += 1
        return count

    def _last_prompt_tokens(self) -> int | None:
        if not isinstance(self._last_usage, dict):
            return None
        prompt_tokens = self._last_usage.get("prompt_tokens")
        return prompt_tokens if isinstance(prompt_tokens, int) else None

    def _last_cached_tokens(self) -> int | None:
        if not isinstance(self._last_usage, dict):
            return None
        cached_tokens = self._last_usage.get("cached_tokens")
        return cached_tokens if isinstance(cached_tokens, int) else None

    async def _request_with_http(self, request_payload: dict[str, Any]) -> Any:
        if self._client is None:
            raise RuntimeError("OpenAI HTTP transport is unavailable in dry_run mode")
        if self._client_is_async:
            return await self._client.responses.create(**request_payload)
        return await asyncio.to_thread(self._client.responses.create, **request_payload)

    async def _request_with_stream(self, request_payload: dict[str, Any]) -> Any:
        if self._client is None:
            raise RuntimeError("OpenAI stream transport is unavailable in dry_run mode")
        if self._client_is_async:
            async with self._client.responses.stream(**request_payload) as stream:
                async for _event in stream:
                    pass
                return await stream.get_final_response()

        def _collect() -> Any:
            with self._client.responses.stream(**request_payload) as stream:
                for _event in stream:
                    pass
                return stream.get_final_response()

        return await asyncio.to_thread(_collect)

    async def _request_with_transport(
        self,
        *,
        request_payload: dict[str, Any],
        incremental_request_payload: dict[str, Any],
        fallback_request_payload: dict[str, Any],
    ) -> Any:
        if self.transport == "websocket":
            websocket_payload = (
                incremental_request_payload
                if incremental_request_payload.get("previous_response_id")
                else request_payload
            )
            request_coro = self._request_with_websocket(
                request_payload=websocket_payload,
                fallback_request_payload=fallback_request_payload,
            )
        elif self.transport == "stream":
            request_coro = self._request_with_stream(request_payload)
        else:
            request_coro = self._request_with_http(request_payload)

        if self.request_timeout_seconds and self.request_timeout_seconds > 0:
            return await asyncio.wait_for(
                request_coro,
                timeout=float(self.request_timeout_seconds),
            )
        return await request_coro

    async def _request_with_websocket(
        self,
        *,
        request_payload: dict[str, Any],
        fallback_request_payload: dict[str, Any],
    ) -> Any:
        try:
            return await self._send_websocket_request(request_payload=request_payload)
        except _OpenAIWebSocketError as exc:
            if exc.code == "websocket_connection_limit_reached":
                try:
                    return await self._send_websocket_request(
                        request_payload=request_payload, force_reconnect=True
                    )
                except _OpenAIWebSocketError as retry_exc:
                    if retry_exc.code == "previous_response_not_found":
                        self._log_websocket_full_context_fallback(
                            reason="previous_response_not_found_after_reconnect"
                        )
                        return await self._send_websocket_request(
                            request_payload=fallback_request_payload, force_reconnect=True
                        )
                    raise
            if exc.code == "previous_response_not_found":
                self._log_websocket_full_context_fallback(reason="previous_response_not_found")
                return await self._send_websocket_request(
                    request_payload=fallback_request_payload, force_reconnect=True
                )
            raise

    def _log_websocket_full_context_fallback(self, *, reason: str) -> None:
        logger.warning(
            "OpenAI websocket full-context fallback triggered: model=%s reason=%s previous_response_id=%s input_items=%s",
            self.model_id,
            reason,
            self._previous_response_id,
            len(self._input_items),
        )

    async def _send_websocket_request(
        self,
        *,
        request_payload: dict[str, Any],
        force_reconnect: bool = False,
    ) -> Any:
        websocket = await self._ensure_websocket_connection(force_reconnect=force_reconnect)
        event = {"type": "response.create", **request_payload}
        await websocket.send(json.dumps(event))
        return await self._receive_websocket_response(websocket)

    async def _ensure_websocket_connection(self, *, force_reconnect: bool = False) -> Any:
        if (
            not force_reconnect
            and self._websocket is not None
            and not getattr(self._websocket, "closed", False)
            and not self._websocket_connection_is_stale()
        ):
            return self._websocket

        await self._close_websocket()
        self._websocket = await self._open_websocket_connection()
        self._websocket_opened_at = time.monotonic()
        return self._websocket

    async def _open_websocket_connection(self) -> Any:
        if not self._api_key:
            raise RuntimeError("OpenAI websocket transport requires a real API key")
        try:
            import websockets  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "OpenAI websocket transport requires the `websockets` package."
            ) from exc

        return await websockets.connect(
            self._responses_websocket_url,
            additional_headers={"Authorization": f"Bearer {self._api_key}"},
            open_timeout=self.websocket_open_timeout_seconds,
            max_size=None,
        )

    def _websocket_connection_is_stale(self) -> bool:
        if self._websocket_opened_at is None:
            return False
        max_age = self.websocket_max_connection_age_seconds
        if max_age <= 0:
            return False
        return (time.monotonic() - self._websocket_opened_at) >= max_age

    async def _receive_websocket_response(self, websocket: Any) -> Any:
        while True:
            raw_message = await websocket.recv()
            if isinstance(raw_message, bytes):
                raw_message = raw_message.decode("utf-8")
            event = json.loads(raw_message)
            event_type = event.get("type")

            if event_type == "error":
                raise _OpenAIWebSocketError.from_event(event)

            if event_type == "response.completed":
                response_payload = event.get("response")
                if response_payload is None:
                    raise _OpenAIWebSocketError(
                        code="missing_response",
                        message="response.completed event did not include a response payload",
                    )
                return self._coerce_response_object(response_payload)

            if event_type == "response.incomplete":
                response_payload = event.get("response")
                if response_payload is None:
                    raise _OpenAIWebSocketError(
                        code="missing_response",
                        message="response.incomplete event did not include a response payload",
                    )
                return self._coerce_response_object(response_payload)

            if event_type == "response.failed":
                response_payload = event.get("response")
                raise _OpenAIWebSocketError(
                    code="response_failed",
                    message=_response_error_message(response_payload)
                    or "OpenAI response failed over websocket",
                    response=response_payload,
                )

    def _coerce_response_object(self, response_payload: Any) -> Any:
        if not isinstance(response_payload, dict):
            return response_payload
        try:
            from openai.types.responses.response import Response  # type: ignore

            return Response.model_validate(response_payload)
        except Exception:
            return response_payload

    async def _close_websocket(self) -> None:
        websocket = self._websocket
        self._websocket = None
        self._websocket_opened_at = None
        if websocket is None:
            return None
        close_method = getattr(websocket, "close", None)
        if close_method is None:
            return None
        result = close_method()
        if asyncio.iscoroutine(result):
            await result
        return None

    def _append_new_inputs(self, messages: list[dict]) -> list[dict[str, Any]]:

        new_items: list[dict[str, Any]] = []
        start = 0 if not self._input_items else self._last_message_count
        had_prior_input_items = bool(self._input_items)
        for message in messages[start:]:
            if not isinstance(message, dict):
                continue

            role = message.get("role")
            if role in {"user", "assistant"}:
                if had_prior_input_items and role == "assistant":
                    continue
                if role == "assistant":
                    replay_items = self._validated_provider_output_items(message)
                    if replay_items is not None:
                        self._input_items.extend(replay_items)
                        new_items.extend(replay_items)
                        continue
                content_parts = self._convert_message_content(message.get("content"))
                if role == "assistant":
                    content_parts = [
                        {**p, "type": "output_text"} if p.get("type") == "input_text" else p
                        for p in content_parts
                    ]
                if content_parts:
                    item = {"role": role, "content": content_parts}
                    self._input_items.append(item)
                    new_items.append(item)

                for call in message.get("tool_calls") or []:
                    if not isinstance(call, dict):
                        continue
                    call_id = call.get("id") or f"call_{uuid.uuid4().hex}"
                    if call.get("type") == "custom":
                        custom = call.get("custom") if isinstance(call.get("custom"), dict) else {}
                        item = {
                            "type": "custom_tool_call",
                            "call_id": call_id,
                            "name": custom.get("name", ""),
                            "input": custom.get("input", ""),
                        }
                        self._input_items.append(item)
                        new_items.append(item)
                    else:
                        func = (
                            call.get("function") if isinstance(call.get("function"), dict) else {}
                        )
                        item = {
                            "type": "function_call",
                            "call_id": call_id,
                            "name": func.get("name", ""),
                            "arguments": func.get("arguments", ""),
                        }
                        self._input_items.append(item)
                        new_items.append(item)

            elif role == "tool":
                call_id = message.get("tool_call_id") or message.get("call_id")
                if not call_id:
                    continue
                content = message.get("content")
                output: Any
                if isinstance(content, list):
                    converted = self._convert_message_content(content)
                    output = converted if converted else json.dumps(content)
                elif content is None:
                    output = ""
                elif isinstance(content, str):
                    output = content
                else:
                    try:
                        output = json.dumps(content)
                    except Exception:
                        output = str(content)
                if message.get("tool_type") == "custom":
                    out_str = output if isinstance(output, str) else json.dumps(content)
                    item = {
                        "type": "custom_tool_call_output",
                        "call_id": call_id,
                        "output": out_str,
                    }
                    self._input_items.append(item)
                    new_items.append(item)
                else:
                    item = {"type": "function_call_output", "call_id": call_id, "output": output}
                    self._input_items.append(item)
                    new_items.append(item)

        self._last_message_count = len(messages)
        self._pending_incremental_input_items.extend(new_items)
        return new_items

    def _validated_provider_output_items(
        self,
        message: dict[str, Any],
    ) -> list[dict[str, Any]] | None:
        meta = message.get("_meta")
        if not isinstance(meta, dict):
            return None
        if meta.get("openai_wire_fingerprint") != self.openai_wire_fingerprint:
            return None
        items = meta.get("provider_output_items")
        if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
            return None

        raw_call_ids = {
            str(item.get("call_id"))
            for item in items
            if item.get("type") in {"function_call", "custom_tool_call"} and item.get("call_id")
        }
        neutral_call_ids = {
            str(call.get("id"))
            for call in message.get("tool_calls") or []
            if isinstance(call, dict) and call.get("id")
        }
        if raw_call_ids != neutral_call_ids:
            return None
        return json.loads(json.dumps(items))

    def _convert_message_content(self, content: Any) -> list[dict[str, Any]]:
        return openai_codec.convert_message_content(content)

    def _convert_image_part(self, part: dict[str, Any]) -> Optional[dict[str, Any]]:
        return openai_codec.convert_image_part(part)

    def _convert_tools(self, tools: list[dict]) -> list[dict[str, Any]]:
        return openai_codec.convert_tools(tools)

    def _serialize_response_output(self, response: Any) -> list[dict[str, Any]]:
        return openai_codec.serialize_response_output(response)

    def _extract_usage(self, response: Any) -> Optional[dict[str, int]]:
        return openai_codec.extract_usage(response)

    def _convert_response(self, response: Any) -> dict[str, Any]:
        return openai_codec.convert_response(response)

    def _extract_response_id(self, response: Any) -> Optional[str]:
        return openai_codec.extract_response_id(response)


def _hard_pressure_threshold_for_model(model_id: str) -> int | None:
    override = _env_int_or_none("OPENAI_HARD_PRESSURE_THRESHOLD_TOKENS")
    if override is not None:
        return override
    if model_id in {"gpt-5.5", "gpt-5.5-2026-04-23", "gpt-5.6-sol", "gpt-6-astra"}:
        return DEFAULT_OPENAI_HARD_PRESSURE_TOKENS
    return None


def _optional_nonnegative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None


def _openai_wire_fingerprint(base_url: str | None) -> str:
    raw_url = (base_url or "https://api.openai.com/v1").strip()
    parsed = urlparse(raw_url)
    scheme = (parsed.scheme or "https").lower()
    host = (parsed.hostname or parsed.path.split("/", 1)[0]).lower()
    port = parsed.port
    default_port = (scheme == "https" and port == 443) or (scheme == "http" and port == 80)
    authority = host if port is None or default_port else f"{host}:{port}"
    origin = urlunparse((scheme, authority, "", "", "", ""))
    try:
        sdk_version = version("openai")
    except PackageNotFoundError:
        sdk_version = "unavailable"
    encoded = f"{origin}\n{sdk_version}".encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _context_window_tokens_for_model(model_id: str) -> int | None:
    override = _env_int_or_none("OPENAI_CONTEXT_WINDOW_TOKENS")
    if override is not None:
        return override
    if model_id in {"gpt-5.5", "gpt-5.5-2026-04-23", "gpt-5.6-sol", "gpt-6-astra"}:
        return DEFAULT_OPENAI_CONTEXT_TOKENS
    return None


def _normalize_reasoning_summary(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"0", "false", "none", "off"}:
        return None
    return text


def _normalize_prompt_cache_key(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _normalize_prompt_cache_retention(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"0", "false", "none", "off"}:
        return None
    return text


def _user_item_fingerprint(item: dict[str, Any]) -> str | None:
    if not openai_codec.is_user_message_item(item):
        return None
    payload = {
        "role": "user",
        "content": item.get("content"),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _user_item_fingerprints(items: list[dict[str, Any]]) -> list[str]:
    fingerprints: list[str] = []
    for item in items:
        fingerprint = _user_item_fingerprint(item)
        if fingerprint is not None and fingerprint not in fingerprints:
            fingerprints.append(fingerprint)
    return fingerprints


def _effort_from_thinking_level(thinking_level: str, model_id: str | None = None) -> str:
    if model_id == "gpt-6-astra":
        effort = str(thinking_level).strip().lower()
        effort = {"med": "medium", "mid": "medium", "none": "low", "minimal": "low"}.get(
            effort, effort
        )
        if effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ProviderConfigurationError(f"Unsupported GPT-6 Astra reasoning effort: {effort}")
        return effort
    return provider_reasoning_level(thinking_level, default=ThinkingLevel.MED)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except Exception:
        return default


def _env_int_or_none(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        value = int(raw.strip().replace("_", ""))
    except Exception:
        return None
    return value if value > 0 else None


def _normalize_transport(transport: str) -> str:
    value = (transport or "http").strip().lower()
    if value not in {"http", "websocket", "stream"}:
        raise ValueError(f"Unsupported OpenAI transport: {transport}")
    return value


def _responses_websocket_url(base_url: Optional[str]) -> str:
    root = (base_url or "https://api.openai.com/v1/").strip()
    parsed = urlparse(root)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = parsed.path.rstrip("/")
    if not path:
        path = "/v1"
    if not path.endswith("/responses"):
        if not path.endswith("/v1"):
            path = f"{path}/v1"
        path = f"{path}/responses"
    return urlunparse((scheme, parsed.netloc, path, "", "", ""))


def _payload_without_reasoning_summary(request_payload: dict[str, Any]) -> dict[str, Any]:
    updated = dict(request_payload)
    reasoning = updated.get("reasoning")
    if not isinstance(reasoning, dict) or "summary" not in reasoning:
        return updated
    new_reasoning = dict(reasoning)
    new_reasoning.pop("summary", None)
    updated["reasoning"] = new_reasoning
    return updated


def _extract_http_status(exc: BaseException) -> Optional[int]:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int) and 100 <= status <= 599:
        return status
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None) or getattr(response, "status", None)
        if isinstance(status, int) and 100 <= status <= 599:
            return status
    return None


def _is_reasoning_summary_unsupported_error(exc: BaseException) -> bool:
    status = _extract_http_status(exc)
    if status is not None and status not in {400, 422}:
        return False

    message = str(exc).strip().lower()
    if not message:
        return False

    if "reasoning.summary" in message:
        return any(
            needle in message
            for needle in (
                "unsupported",
                "not supported",
                "invalid",
                "unknown parameter",
                "not allowed",
                "unrecognized",
            )
        )

    if "reasoning" not in message or "summary" not in message:
        return False

    return any(
        needle in message
        for needle in (
            "unsupported",
            "not supported",
            "invalid",
            "unknown parameter",
            "not allowed",
            "unrecognized",
        )
    )


def _should_retry_openai_exception(exc: BaseException) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True

    message = str(exc).lower()
    if any(
        needle in message
        for needle in (
            "no tool output found",
            "no tool call found",
            "invalid_request_error",
            "invalid request error",
        )
    ):
        return False

    status = _extract_http_status(exc)
    if status is not None:
        if status in {408, 409, 425, 429, 500, 502, 503, 504}:
            return True
        if 400 <= status < 500:
            return False
        if status >= 500:
            return True

    if isinstance(exc, _OpenAIWebSocketError):
        if exc.code == "previous_response_not_found":
            return False
        if exc.code in {
            "websocket_connection_limit_reached",
            "rate_limit_exceeded",
            "server_error",
            "overloaded",
            "temporarily_unavailable",
        }:
            return True

    message = str(exc).lower()
    if any(
        needle in message
        for needle in (
            "overloaded",
            "temporarily unavailable",
            "service unavailable",
            "timeout",
            "timed out",
            "deadline exceeded",
            "rate limit",
            "too many requests",
            "connection reset",
            "connection aborted",
            "connection refused",
            "connection error",
            "internal error",
            "backend error",
        )
    ):
        return True

    if any(
        needle in message
        for needle in (
            "api key",
            "unauthorized",
            "permission denied",
            "forbidden",
            "invalid",
            "malformed",
            "not found",
        )
    ):
        return False

    return True


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
    fn: Any,
    *,
    max_attempts: int,
    should_retry: Any,
    base_delay: float,
    max_delay: float,
    sleep_fn: Any = asyncio.sleep,
    rng: Any = random.random,
    logger: Optional[logging.Logger] = None,
    context: str = "openai",
) -> Any:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be >= 1")

    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as exc:
            attempt += 1
            if attempt >= max_attempts or not bool(should_retry(exc)):
                raise

            cap = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay = max(0.0, cap * 0.5 + float(rng()) * cap * 0.5)
            if logger:
                logger.warning(
                    "%s failed (attempt %s/%s), retrying in %.2fs: %s",
                    context,
                    attempt,
                    max_attempts,
                    delay,
                    _format_retry_exception(exc),
                )
            await sleep_fn(delay)


def _response_error_message(response_payload: Any) -> str:
    if response_payload is None:
        return ""
    error = (
        response_payload.get("error")
        if isinstance(response_payload, dict)
        else getattr(response_payload, "error", None)
    )
    if error is None:
        return ""
    if isinstance(error, str):
        return error.strip()
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
        return json.dumps(error)
    message = getattr(error, "message", None)
    if isinstance(message, str) and message.strip():
        return message.strip()
    code = getattr(error, "code", None)
    if isinstance(code, str) and code.strip():
        return code.strip()
    return str(error)


class _OpenAIWebSocketError(RuntimeError):
    def __init__(
        self,
        *,
        code: Optional[str],
        message: str,
        status: Optional[int] = None,
        response: Any = None,
    ):
        self.code = code
        self.status = status
        self.response = response
        super().__init__(message)

    @classmethod
    def from_event(cls, event: dict[str, Any]) -> "_OpenAIWebSocketError":
        error = event.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if not isinstance(message, str) or not message.strip():
                message = json.dumps(error)
            code = error.get("code")
        else:
            message = str(error or "OpenAI websocket error")
            code = None
        status = event.get("status")
        return cls(
            code=str(code) if code is not None else None,
            message=message,
            status=int(status) if isinstance(status, int) else None,
        )
