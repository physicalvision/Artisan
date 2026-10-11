from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pipeline.context_policy import (
    consume_prepare_result,
    mask_stale_review_evidence,
    reset_tier_a_dedup,
    trace_full,
)
from pipeline.llm.errors import CostAccountingError, ProviderConfigurationError
from pipeline.timing import _null_measure

logger = logging.getLogger(__name__)


class MalformedToolCallError(Exception):

    def __init__(self, message: str, *, name: str, tool_id: Any) -> None:
        super().__init__(message)
        self.name = name
        self.tool_id = tool_id


class _ToolLoopHarness:

    ROLE: str = ""
    SYSTEM_PROMPT: str = ""
    _CASSETTE_KIND: str = "turn"
    supports_cost_compaction: bool = False

    recorder: Any = None

    def _start_tier_a_telemetry(self) -> None:
        self._tier_a_rollup: dict[str, dict[str, int]] = {}
        self._tier_a_rollup_active = True

    def _record_tier_a_savings(self, tool: str, saved_chars: int) -> None:
        saved = max(0, int(saved_chars))
        if not saved:
            return
        if not bool(getattr(self, "_tier_a_rollup_active", False)):
            self._start_tier_a_telemetry()
        rollup = getattr(self, "_tier_a_rollup", None)
        if not isinstance(rollup, dict):
            rollup = {}
            self._tier_a_rollup = rollup
        entry = rollup.setdefault(str(tool), {"action_count": 0, "saved_chars": 0})
        entry["action_count"] += 1
        entry["saved_chars"] += saved

    def _flush_tier_a_telemetry(self) -> None:
        trace = getattr(self, "trace", None)
        rollup = getattr(self, "_tier_a_rollup", None)
        try:
            if trace is None or not isinstance(rollup, dict):
                return
            for tool, entry in sorted(rollup.items()):
                try:
                    trace.write_event(
                        "tier_a_dedup",
                        {
                            "tool": tool,
                            "action_count": int(entry.get("action_count", 0)),
                            "saved_chars": int(entry.get("saved_chars", 0)),
                        },
                    )
                except Exception:  # noqa: BLE001 - telemetry must never break the run
                    logger.exception("Tier A telemetry flush failed (non-fatal)")
        finally:
            self._tier_a_rollup = {}
            self._tier_a_rollup_active = False

    async def _run_loop_with_telemetry(self, conversation: list[dict]):
        if not bool(getattr(self, "_tier_a_rollup_active", False)):
            self._start_tier_a_telemetry()
        try:
            return await self._run_loop(conversation)
        finally:
            self._flush_tier_a_telemetry()

    def _tools(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @property
    def _active_model_id(self) -> str | None:
        return getattr(getattr(self, "llm", None), "model_id", None)

    def _cassette_kind(self) -> str:
        return self._CASSETTE_KIND

    def _end_turn_after_nudge(self) -> bool:
        return False

    def _build_compaction_refresher(self) -> dict[str, Any] | None:
        return None

    def _build_task_rehydration(self) -> dict[str, Any] | None:
        return None

    def _handle_compaction_event(
        self,
        compaction_event: Any,
        conversation: list[dict[str, Any]],
    ) -> None:
        event_key = (
            getattr(compaction_event, "provider", None),
            getattr(compaction_event, "request_sequence", None) or id(compaction_event),
            getattr(compaction_event, "trigger", None),
        )
        handled = getattr(self, "_handled_compaction_events", None)
        if not isinstance(handled, set):
            handled = set()
            self._handled_compaction_events = handled
        if event_key in handled:
            return
        handled.add(event_key)

        reset_tier_a_dedup(self)
        self._expect_cold_next_call = True

        retained = getattr(compaction_event, "retained_user_message_fingerprints", None)
        task = getattr(self, "_round_task_message", None)
        fingerprint_fn = getattr(getattr(self, "llm", None), "user_message_fingerprint", None)
        if isinstance(retained, list) and isinstance(task, dict) and callable(fingerprint_fn):
            task_fingerprint = fingerprint_fn(task)
            if task_fingerprint and task_fingerprint not in retained:
                rehydrated = self._build_task_rehydration()
                if isinstance(rehydrated, dict):
                    conversation.append(rehydrated)
                    self._trace_msg(rehydrated)

        refresher = self._build_compaction_refresher()
        if isinstance(refresher, dict):
            conversation.append(refresher)
            self._trace_msg(refresher)

    def _assistant_nonempty(self, assistant: dict) -> bool:
        return any(
            k in assistant for k in ("content", "tool_calls", "thought_summary", "extra_content")
        )

    def _parse_tool_call(self, tc: dict) -> tuple[str, dict, Any]:
        name = self.codec.tool_call_name(tc)
        raw = (
            tc.get("function", {}).get("arguments")
            or tc.get("custom", {}).get("input")
            or tc.get("input")
            or "{}"
        )
        tool_id = tc.get("id") or tc.get("tool_call_id") or name
        if isinstance(raw, dict):
            return name, raw, tool_id
        try:
            args = json.loads(raw or "{}")
        except ValueError as exc:
            raise MalformedToolCallError(str(exc), name=name, tool_id=tool_id) from exc
        return name, args, tool_id

    def _tool_message(self, tool_id: Any, name: str, content: Any) -> dict:
        return {"role": "tool", "tool_call_id": tool_id, "name": name, "content": content}

    def _no_tool_nudge(self) -> str:
        raise NotImplementedError

    def _no_tool_images(self) -> list[Path]:
        return []

    def _no_tool_nudge_metadata(self) -> dict[str, Any]:
        return {}

    def _nudge_content(self, text: str, images: list[Path]) -> Any:
        if not images:
            return text
        return [{"type": "input_text", "text": text}] + [
            {"type": "input_image", "image_path": str(path), "detail": "high"} for path in images
        ]

    def _wrap_up_nudge(self) -> str | None:
        return None

    def _on_no_tool_calls(self, conversation: list[dict], turns: int):
        return None

    def _dispatch_tool(self, name: str, args: dict) -> tuple[str, list[Path], Any, Any]:
        raise NotImplementedError

    def _tool_ok(self, text_ack: str) -> bool:
        return not str(text_ack).startswith("ERROR")

    def _fallback_result(self, conversation: list[dict], turns: int):
        raise NotImplementedError

    def _cost_limit_result(self, conversation: list[dict], turns: int):
        raise NotImplementedError

    def _zero_cost_terminal_tools(self) -> frozenset[str]:
        return frozenset()

    def _current_total_cost_usd(self) -> float:
        if self.cost is None:
            return 0.0
        return self.cost.all_in_total_cost_usd

    def _trace_msg(self, msg: dict) -> None:
        if self.trace:
            self.trace.write_message(msg)

    def _shrink_for_payload_limit(self, exc: BaseException, conversation: list) -> bool:
        try:
            from pipeline.llm.errors import is_payload_too_large
        except Exception:  # noqa: BLE001 - provider not installed / not in use
            return False
        if not is_payload_too_large(exc) or getattr(self, "_payload_shrunk_this_turn", False):
            return False
        shrunk, expired = mask_stale_review_evidence(conversation)
        if not expired:
            return False
        conversation[:] = shrunk
        self._payload_shrunk_this_turn = True
        logger.warning(
            "request payload too large; expired %s stale review evidence part(s) and retrying",
            expired,
        )
        return True

    def _drain_provider_trace_events(self) -> None:
        drain = getattr(getattr(self, "llm", None), "drain_trace_events", None)
        if not callable(drain):
            return
        try:
            events = drain()
        except Exception:  # noqa: BLE001 - telemetry must never break the run
            logger.exception("provider trace drain failed (non-fatal)")
            return
        if not self.trace:
            return
        for name, payload in events:
            try:
                self.trace.write_event(name, payload)
            except Exception:  # noqa: BLE001
                logger.exception("provider trace write failed (non-fatal)")

    async def _generate(
        self, conversation: list[dict], tools: list[dict]
    ) -> tuple[dict, list, dict]:
        kind = self._cassette_kind()
        if self.cassette is not None and getattr(self.cassette, "replaying", False):
            rec = self.cassette.replay(self.ROLE, kind)
            return rec.get("assistant") or {}, rec.get("tool_calls") or [], rec.get("usage") or {}
        response = await self.llm.generate_with_tools(self.SYSTEM_PROMPT, conversation, tools)
        tool_calls = self.codec.extract_tool_calls(response)
        usage = self.codec.extract_usage(response) or {}
        assistant = self.codec.build_assistant_message(response)
        if self.cassette is not None and getattr(self.cassette, "recording", False):
            self.cassette.record(
                self.ROLE, kind, {"assistant": assistant, "tool_calls": tool_calls, "usage": usage}
            )
        return assistant, tool_calls, usage

    async def _run_loop(self, conversation: list[dict]):
        tools = self._tools()
        turns = 0
        self._conversation = conversation
        self.display.start()
        while turns < self.max_turns:
            turns += 1
            self._turn = turns
            self.display.start_turn(turns)
            wrap_up_start_turn = getattr(self, "wrap_up_start_turn", None)
            if (
                isinstance(wrap_up_start_turn, int)
                and wrap_up_start_turn < self.max_turns
                and turns == wrap_up_start_turn + 1
            ):
                wrap_up_text = self._wrap_up_nudge()
                if wrap_up_text:
                    wrap_up_message = {"role": "user", "content": wrap_up_text}
                    conversation.append(wrap_up_message)
                    self._trace_msg(wrap_up_message)
            _rec = self.recorder
            prep = getattr(self.llm, "prepare_next_request", None)
            if callable(prep):
                prepare_result = None
                try:
                    with _rec.measure() if _rec is not None else _null_measure() as _prep_s:
                        try:
                            prepare_result = await prep(
                                system_prompt=self.SYSTEM_PROMPT,
                                messages=conversation,
                                tools=tools,
                                completed_turns=turns - 1,
                                consecutive_compile_failure_count=(
                                    self._consecutive_compile_failures
                                ),
                                last_compile_failure_sig=self._last_compile_failure_sig,
                                entry_reason=getattr(self, "_entry_reason_once", None),
                            )
                        finally:
                            self._entry_reason_once = None
                except (ProviderConfigurationError, CostAccountingError):
                    raise
                except Exception:  # noqa: BLE001
                    logger.exception("prepare_next_request failed (non-fatal)")
                if prepare_result is not None:
                    compaction_event = getattr(prepare_result, "compaction_event", None)
                    if compaction_event is not None:
                        self._handle_compaction_event(compaction_event, conversation)
                    accounting = None
                    try:
                        accounting = consume_prepare_result(
                            prepare_result,
                            cost=self.cost,
                            display=self.display,
                            trace=self.trace,
                        )
                    except Exception as exc:  # noqa: BLE001 - never admit another paid call
                        raise CostAccountingError(
                            "Provider maintenance usage could not be accounted; stopping this run"
                        ) from exc
                    if _rec is not None and compaction_event is not None and accounting is not None:
                        status = accounting.compaction_accounting_status or "usage_unavailable"
                        _rec.record(
                            kind="maintenance",
                            name="compaction",
                            sample=_prep_s,
                            tokens=getattr(compaction_event, "usage", None),
                            cost_usd=(
                                accounting.compaction_cost_usd if status == "priced" else None
                            ),
                            trigger=getattr(compaction_event, "trigger", None),
                            model_id=getattr(compaction_event, "model_id", None),
                            provider=getattr(compaction_event, "provider", None),
                            accounting_status=status,
                            cumulative_cost_usd=round(self._current_total_cost_usd(), 6),
                        )
            if (
                self.cost is not None
                and self.max_cost_usd is not None
                and self._current_total_cost_usd() >= self.max_cost_usd
            ):
                self.display.end_turn(success=False, error="cost limit")
                return self._cost_limit_result(conversation, turns - 1)
            self.display.start_llm_wait()
            with _rec.measure() if _rec is not None else _null_measure() as _llm_s:
                try:
                    self._payload_shrunk_this_turn = False
                    try:
                        assistant, tool_calls, usage = await self._generate(conversation, tools)
                    except Exception as exc:  # noqa: BLE001 - re-raised unless recoverable
                        if not self._shrink_for_payload_limit(exc, conversation):
                            raise
                        assistant, tool_calls, usage = await self._generate(conversation, tools)
                finally:
                    self.display.stop_llm_wait()
                    self._drain_provider_trace_events()
            _llm_dt = _llm_s.wall_s

            if self._assistant_nonempty(assistant):
                conversation.append(assistant)
                self._trace_msg(assistant)

            turn_cost = 0.0
            if usage:
                turn_cost = self.cost.add_turn(usage).total_cost if self.cost else 0.0
                self.display.add_llm_call(usage, turn_cost, _llm_dt)
            expected_cold = bool(getattr(self, "_expect_cold_next_call", False))
            self._expect_cold_next_call = False
            if _rec is not None:
                _rec.record(
                    kind="llm",
                    name=self.ROLE,
                    sample=_llm_s,
                    tokens=usage or None,
                    cost_usd=turn_cost if usage else None,
                    model_id=self._active_model_id,
                    expected_cold=expected_cold,
                )
            response_over_cost = bool(
                usage
                and self.cost
                and self.max_cost_usd is not None
                and self._current_total_cost_usd() > self.max_cost_usd
            )

            if not tool_calls:
                if response_over_cost:
                    self.display.end_turn(success=False, error="cost limit")
                    return self._cost_limit_result(conversation, turns)
                terminal = self._on_no_tool_calls(conversation, turns)
                if terminal is not None:
                    self.display.end_turn(success=True)
                    return terminal
                nudge_metadata = self._no_tool_nudge_metadata()
                nudge_text = self._no_tool_nudge()
                nudge_images = self._no_tool_images()
                nudge = {
                    "role": "user",
                    "content": self._nudge_content(nudge_text, nudge_images),
                }
                if nudge_metadata:
                    nudge["_meta"] = nudge_metadata
                conversation.append(nudge)
                self._trace_msg(nudge)
                if self._end_turn_after_nudge():
                    self.display.end_turn(success=True)
                continue

            for tc in tool_calls:
                try:
                    name, args, tool_id = self._parse_tool_call(tc)
                except MalformedToolCallError as exc:
                    err = (
                        f"ERROR: could not parse the arguments of this tool call as JSON "
                        f"({exc}). The call was most likely truncated in transit. "
                        f"Re-issue it, and if the payload is large, split it into "
                        f"smaller calls."
                    )
                    logger.warning("malformed tool call %s: %s", exc.name, exc)
                    bad = self._tool_message(exc.tool_id, exc.name, err)
                    conversation.append(bad)
                    trace_full(self.trace, bad, None)
                    continue
                with _rec.measure() if _rec is not None else _null_measure() as _tool_s:
                    if response_over_cost and name not in self._zero_cost_terminal_tools():
                        text_ack = "ERROR: cost limit reached before tool dispatch"
                        imgs: list[Path] = []
                        terminal = None
                        full_ack = text_ack
                    else:
                        text_ack, imgs, terminal, full_ack = self._dispatch_tool(name, args)
                if (
                    isinstance(text_ack, str)
                    and isinstance(full_ack, str)
                    and len(full_ack) > len(text_ack)
                ):
                    self._record_tier_a_savings(name, len(full_ack) - len(text_ack))
                _tool_dt = _tool_s.wall_s
                ok = self._tool_ok(text_ack)
                _tool_s.ok = ok
                if _rec is not None:
                    _rec.record(kind="tool", name=name, sample=_tool_s, ok=ok)
                self.display.add_tool_call(name, args, ok, _tool_dt, result=text_ack)
                if imgs:
                    content: Any = [{"type": "input_text", "text": text_ack}] + [
                        {"type": "input_image", "image_path": str(p), "detail": "high"}
                        for p in imgs
                    ]
                else:
                    content = text_ack
                msg = self._tool_message(tool_id, name, content)
                conversation.append(msg)
                trace_full(self.trace, msg, None if imgs else full_ack)
                if terminal is not None:
                    self.display.end_turn(success=True)
                    return terminal
            if response_over_cost:
                self.display.end_turn(success=False, error="cost limit")
                return self._cost_limit_result(conversation, turns)
            self.display.end_turn(success=True)

        self.display.end_turn(success=False, error="max turns")
        return self._fallback_result(conversation, turns)
