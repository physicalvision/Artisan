from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from artisan.values import normalize_model_id
from pipeline.llm._image_tokens import (
    estimate_image_tokens,
    image_caps_for_model,
    sniff_image_dimensions,
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw.strip().replace("_", ""))
    except ValueError:
        return default


@dataclass(frozen=True, slots=True)
class CompressionPolicy:

    enabled: bool = True
    tool_output_hard_cap_chars: int = 10_000
    read_file_dedup: bool = True
    probe_dedup: bool = True
    boundary_reseed: bool = True
    keep_last_per_tool: dict[str, int] | int = field(
        default_factory=lambda: {"*": 2, "read_file": 1}
    )
    elide_tool_call_args: bool = True
    expire_review_evidence: bool = True
    keep_recent_renders: int = 0
    min_elide_chars: int = 500
    boundary_min_tokens: int = 30_000
    snapshot_max_chars: int = 6_000
    inject_snapshot: bool = False
    enable_native_summary: bool = True


def policy_from_env() -> CompressionPolicy:
    d = CompressionPolicy()
    keep_tool_raw = os.environ.get("ARTISAN_COMPRESSION_KEEP_TOOL")
    keep_last_per_tool: dict[str, int] | int = d.keep_last_per_tool
    if keep_tool_raw is not None:
        keep_last_per_tool = _env_int("ARTISAN_COMPRESSION_KEEP_TOOL", 2)
    return CompressionPolicy(
        enabled=_env_flag("ARTISAN_PIPELINE_COMPRESSION", d.enabled),
        tool_output_hard_cap_chars=_env_int(
            "ARTISAN_COMPRESSION_TOOL_CAP", d.tool_output_hard_cap_chars
        ),
        read_file_dedup=_env_flag("ARTISAN_COMPRESSION_READ_DEDUP", d.read_file_dedup),
        probe_dedup=_env_flag("ARTISAN_COMPRESSION_PROBE_DEDUP", d.probe_dedup),
        boundary_reseed=_env_flag("ARTISAN_COMPRESSION_BOUNDARY", d.boundary_reseed),
        keep_last_per_tool=keep_last_per_tool,
        elide_tool_call_args=_env_flag(
            "ARTISAN_COMPRESSION_TOOL_CALL_ARGS", d.elide_tool_call_args
        ),
        expire_review_evidence=_env_flag(
            "ARTISAN_COMPRESSION_REVIEW_EVIDENCE", d.expire_review_evidence
        ),
        keep_recent_renders=_env_int("ARTISAN_COMPRESSION_KEEP_RENDERS", d.keep_recent_renders),
        boundary_min_tokens=_env_int("ARTISAN_COMPRESSION_BOUNDARY_MIN", d.boundary_min_tokens),
        inject_snapshot=_env_flag("ARTISAN_COMPRESSION_SNAPSHOT", d.inject_snapshot),
        enable_native_summary=_env_flag(
            "ARTISAN_COMPRESSION_NATIVE_SUMMARY", d.enable_native_summary
        ),
    )


DEFAULT_COMPRESSION_POLICY = CompressionPolicy()
DISABLED_COMPRESSION_POLICY = CompressionPolicy(enabled=False)


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


def cap_tool_output(text: str, limit: int, *, label: str) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return (
        text[:limit] + f"\n\n[NOTE: {label} output truncated — showing the first {limit} of "
        f"{len(text)} characters to save context; the rest is omitted.]"
    )


def round_floats(obj: Any, ndigits: int = 6) -> Any:
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, ndigits) for v in obj]
    return obj


def dedup_read(
    full: str,
    *,
    path: str,
    key: Any,
    read_cache: dict[Any, str],
    read_seen_turn: dict[Any, int],
    this_turn: int,
    on_saved: Callable[[int], None] | None = None,
    source_digest: str | None = None,
    shown_range: str | None = None,
) -> str:
    digest = sha1_text(f"{source_digest}\0{full}" if source_digest is not None else full)
    if read_cache.get(key) == digest:
        compact = (
            f"[{path} is unchanged since you last read it this round — that "
            f"previously shown {len(full)}-char result"
            f"{f' ({shown_range})' if shown_range else ''} is still above; re-reading is a no-op until the "
            "file changes.]"
        )
        if on_saved is not None:
            on_saved(max(0, len(full) - len(compact)))
        return compact
    read_cache[key] = digest
    read_seen_turn[key] = this_turn
    return full


def trace_full(trace: Any, msg: dict[str, Any], full: Any = None) -> None:
    if trace is None:
        return
    if full is None or msg.get("content") == full:
        trace.write_message(msg)
    else:
        trace.write_message({**msg, "content": full})


_CHARS_PER_TOKEN = 4
_IMG_PART_TYPES = ("input_image", "image")


def _text_tokens(text: Any) -> int:
    if not text:
        return 0
    return len(str(text)) // _CHARS_PER_TOKEN


def _image_tokens_for_path(path: str | None, caps: tuple[int, int]) -> int:
    if not path:
        return caps[0]
    try:
        with open(path, "rb") as fh:
            head = fh.read(64)
    except OSError:
        return caps[0]
    dims = sniff_image_dimensions(head)
    if dims is None:
        return caps[0]
    return estimate_image_tokens(dims[0], dims[1], caps)


def _content_tokens(content: Any, caps: tuple[int, int]) -> int:
    if isinstance(content, str):
        return _text_tokens(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if not isinstance(part, dict):
                total += _text_tokens(part)
                continue
            if part.get("type") in _IMG_PART_TYPES:
                total += _image_tokens_for_path(part.get("image_path"), caps)
            else:
                total += _text_tokens(part.get("text", ""))
        return total
    return _text_tokens(content)


def estimate_conversation_tokens(
    conversation: list[dict[str, Any]], *, model_id: str = "claude-opus-4-8"
) -> int:
    caps = image_caps_for_model(model_id)
    total = 0
    for message in conversation:
        total += _content_tokens(message.get("content"), caps)
        tool_calls = message.get("tool_calls")
        if tool_calls:
            total += _text_tokens(json.dumps(tool_calls, ensure_ascii=False))
        total += _text_tokens(message.get("thought_summary"))
        total += 8
    return total


_RENDER_TOOLS = ("request_render", "crop_region")
_ROUND_RE = re.compile(r"_r(\d+)/")


def parse_render_round(image_path: str | None) -> int | None:
    if not image_path:
        return None
    match = _ROUND_RE.search(image_path)
    return int(match.group(1)) if match else None


def _meta_compressed(message: dict[str, Any]) -> str | None:
    meta = message.get("_meta")
    return meta.get("compressed") if isinstance(meta, dict) else None


def _is_render_tool_message(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return (
        message.get("role") == "tool"
        and isinstance(content, list)
        and any(isinstance(p, dict) and p.get("type") in _IMG_PART_TYPES for p in content)
    )


def mask_stale_renders(
    conversation: list[dict[str, Any]], *, keep_round: int, keep_recent: int = 0
) -> tuple[list[dict[str, Any]], int]:
    threshold = keep_round - keep_recent
    out: list[dict[str, Any]] = []
    masked = 0
    for message in conversation:
        if not _is_render_tool_message(message) or _meta_compressed(message) == "render_mask":
            out.append(message)
            continue
        content = message["content"]
        rounds = [
            r
            for r in (
                parse_render_round(p.get("image_path"))
                for p in content
                if isinstance(p, dict) and p.get("type") in _IMG_PART_TYPES
            )
            if r is not None
        ]
        newest = max(rounds) if rounds else None
        if newest is None or newest >= threshold:
            out.append(message)
            continue
        kept = [
            p for p in content if not (isinstance(p, dict) and p.get("type") in _IMG_PART_TYPES)
        ]
        img_count = len(content) - len(kept)
        names = ", ".join(
            os_basename(p.get("image_path"))
            for p in content
            if isinstance(p, dict) and p.get("type") in _IMG_PART_TYPES
        )
        kept.append(
            {
                "type": "input_text",
                "text": (
                    f"[{img_count} render(s) from round r{newest:02d} dropped to save context: "
                    f"{names}. Re-request_render to re-inspect.]"
                ),
            }
        )
        new_message = dict(message)
        new_message["content"] = kept
        new_message["_meta"] = {
            **(message.get("_meta") or {}),
            "compressed": "render_mask",
            "round": newest,
        }
        out.append(new_message)
        masked += 1
    return out, masked


def _review_evidence_round(part: object) -> int | None:
    if not isinstance(part, dict):
        return None
    meta = part.get("_meta")
    if not isinstance(meta, dict):
        return None
    round_n = meta.get("review_evidence_round")
    return round_n if isinstance(round_n, int) and not isinstance(round_n, bool) else None


def mask_stale_review_evidence(
    conversation: list[dict[str, Any]], *, keep_round: int | None = None
) -> tuple[list[dict[str, Any]], int]:
    out: list[dict[str, Any]] = []
    changed_messages = 0
    for message in conversation:
        content = message.get("content")
        if not isinstance(content, list):
            out.append(message)
            continue

        stale: list[tuple[dict[str, Any], int]] = []
        kept: list[object] = []
        for part in content:
            round_n = _review_evidence_round(part)
            if round_n is not None and (keep_round is None or round_n < keep_round):
                stale.append((part, round_n))
            else:
                kept.append(part)
        if not stale:
            out.append(message)
            continue

        rounds = sorted({round_n for _part, round_n in stale})
        image_names = [
            os_basename(part.get("image_path"))
            for part, _round_n in stale
            if part.get("type") in _IMG_PART_TYPES
        ]
        round_label = ", ".join(f"r{round_n:02d}" for round_n in rounds)
        image_label = ", ".join(image_names) if image_names else "no images"
        kept.append(
            {
                "type": "input_text",
                "text": (
                    f"[verify-first evidence from Review {round_label} expired at the next "
                    f"round boundary: {image_label}. Use current-round QA and renders when "
                    "available; gather missing evidence as needed.]"
                ),
                "_meta": {
                    "compressed": "review_evidence_mask",
                    "review_evidence_rounds": rounds,
                },
            }
        )
        new_message = dict(message)
        new_message["content"] = kept
        out.append(new_message)
        changed_messages += 1
    return out, changed_messages


def _tool_retention(keep_last_per_tool: dict[str, int] | int, tool_name: str) -> int:
    if isinstance(keep_last_per_tool, dict):
        value = keep_last_per_tool.get(tool_name, keep_last_per_tool.get("*", 2))
    else:
        value = keep_last_per_tool
    return max(0, int(value))


def elide_stale_tool_text(
    conversation: list[dict[str, Any]],
    *,
    keep_last_per_tool: dict[str, int] | int = 2,
    min_elide_chars: int = 500,
) -> tuple[list[dict[str, Any]], int]:
    idx_by_name: dict[str, list[int]] = {}
    for i, message in enumerate(conversation):
        if (
            message.get("role") == "tool"
            and isinstance(message.get("content"), str)
            and len(message["content"]) >= min_elide_chars
        ):
            idx_by_name.setdefault(message.get("name") or "?", []).append(i)
    stale: set[int] = set()
    for name, idxs in idx_by_name.items():
        keep = _tool_retention(keep_last_per_tool, name)
        if len(idxs) > keep:
            stale.update(idxs[: len(idxs) - keep])
    out: list[dict[str, Any]] = []
    replaced = 0
    for i, message in enumerate(conversation):
        if i in stale and _meta_compressed(message) != "tool_elide":
            name = message.get("name") or "?"
            new_message = dict(message)
            new_message["content"] = (
                f"[stale {name} output elided to save context — superseded by a later "
                f"call / the current state; re-run the tool if you need it again.]"
            )
            new_message["_meta"] = {**(message.get("_meta") or {}), "compressed": "tool_elide"}
            out.append(new_message)
            replaced += 1
        else:
            out.append(message)
    return out, replaced


def _motion_nudge_stub_content(content: Any, stub: str) -> Any:
    if not isinstance(content, list):
        return stub
    text_type = "input_text"
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in {"input_text", "text"}:
            text_type = str(part["type"])
            break
    return [{"type": text_type, "text": stub}]


def elide_stale_motion_selfcheck_nudges(
    conversation: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    indices: list[int] = []
    for index, message in enumerate(conversation):
        meta = message.get("_meta")
        if (
            message.get("role") == "user"
            and isinstance(meta, dict)
            and meta.get("source") == "harness_nudge"
            and meta.get("nudge_kind") == "motion_selfcheck"
        ):
            indices.append(index)
    if len(indices) <= 1:
        return conversation, 0

    newest_meta = conversation[indices[-1]].get("_meta") or {}
    newest_key = str(newest_meta.get("motion_evidence_key") or "")
    out = list(conversation)
    replaced = 0
    for index in indices[:-1]:
        message = conversation[index]
        meta = dict(message.get("_meta") or {})
        if meta.get("compressed") == "motion_selfcheck_elide":
            continue
        fingerprint = str(meta.get("diagnostic_fingerprint") or "unknown")
        evidence_key = str(meta.get("motion_evidence_key") or "")
        identity = f"fingerprint={fingerprint} motion_evidence_key={evidence_key or 'unknown'}"
        if evidence_key and evidence_key == newest_key:
            recovery = 'recover the full evidence with read_file("reports/motion_findings.json")'
        else:
            recovery = "the evidence key changed; run compile_model for current diagnostics"
        stub = f"[stale motion self-check elided; {identity}; {recovery}.]"
        new_message = dict(message)
        new_message["content"] = _motion_nudge_stub_content(message.get("content"), stub)
        new_message["_meta"] = {**meta, "compressed": "motion_selfcheck_elide"}
        out[index] = new_message
        replaced += 1
    if not replaced:
        return conversation, 0
    return out, replaced


_TOOL_CALL_ARGS_STUB = json.dumps(
    {"_stub": "args elided (edit already applied; see current file state)"},
    separators=(",", ":"),
)


def elide_stale_tool_call_args(
    conversation: list[dict[str, Any]],
    *,
    tools: tuple[str, ...] = ("replace", "write_file"),
    keep_last_per_tool: int = 2,
    min_elide_chars: int = 500,
) -> tuple[list[dict[str, Any]], int]:
    indexed: dict[str, list[tuple[int, int]]] = {}
    for message_index, message in enumerate(conversation):
        if message.get("role") != "assistant":
            continue
        for call_index, call in enumerate(message.get("tool_calls") or []):
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            arguments = function.get("arguments")
            arguments_text = (
                arguments
                if isinstance(arguments, str)
                else json.dumps(arguments, ensure_ascii=False, default=str)
            )
            if name in tools and len(arguments_text) >= min_elide_chars:
                indexed.setdefault(str(name), []).append((message_index, call_index))

    stale: set[tuple[int, int]] = set()
    keep = max(0, int(keep_last_per_tool))
    for positions in indexed.values():
        if len(positions) > keep:
            stale.update(positions[: len(positions) - keep])
    if not stale:
        return conversation, 0

    out = list(conversation)
    replaced = 0
    for message_index in sorted({message_index for message_index, _ in stale}):
        message = conversation[message_index]
        calls = list(message.get("tool_calls") or [])
        changed_call_ids: set[str] = set()
        for call_index, call in enumerate(calls):
            if (message_index, call_index) not in stale or not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            new_call = dict(call)
            new_call["function"] = {**function, "arguments": _TOOL_CALL_ARGS_STUB}
            calls[call_index] = new_call
            if call.get("id"):
                changed_call_ids.add(str(call["id"]))
            replaced += 1

        new_message = dict(message)
        new_message["tool_calls"] = calls
        meta = dict(message.get("_meta") or {})
        raw_items = meta.get("provider_output_items")
        if isinstance(raw_items, list):
            rewritten_items: list[Any] = []
            for raw_item in raw_items:
                if (
                    isinstance(raw_item, dict)
                    and raw_item.get("type") == "function_call"
                    and str(raw_item.get("call_id") or "") in changed_call_ids
                ):
                    rewritten_items.append({**raw_item, "arguments": _TOOL_CALL_ARGS_STUB})
                else:
                    rewritten_items.append(raw_item)
            meta["provider_output_items"] = rewritten_items
        meta["compressed"] = "tool_call_args"
        new_message["_meta"] = meta
        out[message_index] = new_message
    return out, replaced


def os_basename(path: str | None) -> str:
    if not path:
        return "?"
    return path.rsplit("/", 1)[-1]


def max_render_round(conversation: list[dict[str, Any]]) -> int:
    best = 0
    for message in conversation:
        if not _is_render_tool_message(message):
            continue
        for part in message["content"]:
            if isinstance(part, dict) and part.get("type") in _IMG_PART_TYPES:
                r = parse_render_round(part.get("image_path"))
                if r is not None:
                    best = max(best, r)
    return best


def assert_pairing_ok(conversation: list[dict[str, Any]]) -> None:
    answered = {
        m.get("tool_call_id")
        for m in conversation
        if m.get("role") == "tool" and m.get("tool_call_id")
    }
    for message in conversation:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            call_id = call.get("id") or call.get("tool_call_id")
            if call_id and call_id not in answered:
                raise AssertionError(f"dangling tool_call without output: {call_id!r}")




def build_state_snapshot(
    *,
    blueprint: dict[str, Any] | None = None,
    developer_note: dict[str, Any] | None = None,
    compile_summary: str | None = None,
    blocking_signals: list[str] | None = None,
    rubric: dict[str, int] | None = None,
    pending_issues: list[str] | None = None,
    model_note: str | None = None,
    max_chars: int = 6_000,
) -> str:
    lines: list[str] = ["<state_snapshot>"]
    if blueprint:
        brief = (blueprint.get("asset_brief") or {}).get("summary") or blueprint.get("prompt")
        if brief:
            lines.append(f"<goal>{brief}</goal>")
        parts = [p.get("name") for p in blueprint.get("parts", []) if p.get("name")]
        joints = [
            f"{j.get('name')}({j.get('type')})"
            for j in blueprint.get("joints", [])
            if j.get("name")
        ]
        if parts:
            lines.append(f"<parts>{', '.join(parts)}</parts>")
        if joints:
            lines.append(f"<joints>{', '.join(joints)}</joints>")
        acceptance = blueprint.get("acceptance") or []
        if acceptance:
            lines.append("<acceptance>" + "; ".join(str(a) for a in acceptance) + "</acceptance>")
    if model_note:
        lines.append(f"<model_state>{model_note}</model_state>")
    if compile_summary:
        cs = compile_summary
        if blocking_signals:
            cs += " | blocking: " + "; ".join(blocking_signals)
        lines.append(f"<compile_state>{cs}</compile_state>")
    if developer_note:
        changed = developer_note.get("changed")
        if changed:
            lines.append(f"<changed>{changed}</changed>")
        for key in ("intentional", "pushback"):
            vals = developer_note.get(key) or []
            if vals:
                lines.append(f"<{key}>" + "; ".join(str(v) for v in vals) + f"</{key}>")
    if rubric:
        lines.append("<rubric>" + ", ".join(f"{k}={v}" for k, v in rubric.items()) + "</rubric>")
    if pending_issues:
        lines.append(
            "<pending_issues>" + "; ".join(str(i) for i in pending_issues) + "</pending_issues>"
        )
    lines.append("</state_snapshot>")
    return "\n".join(lines)[:max_chars]


def reseed_at_boundary(
    conversation: list[dict[str, Any]],
    *,
    policy: CompressionPolicy,
    keep_round: int | None = None,
    model_id: str = "claude-opus-4-8",
    on_event: Callable[..., None] | None = None,
) -> list[dict[str, Any]]:
    if not policy.enabled or not policy.boundary_reseed:
        return conversation
    before = estimate_conversation_tokens(conversation, model_id=model_id)
    if before < policy.boundary_min_tokens:
        return conversation
    if keep_round is None:
        keep_round = max_render_round(conversation) + 1
    conv, masked = mask_stale_renders(
        conversation, keep_round=keep_round, keep_recent=policy.keep_recent_renders
    )
    conv, elided = elide_stale_tool_text(
        conv, keep_last_per_tool=policy.keep_last_per_tool, min_elide_chars=policy.min_elide_chars
    )
    elided_args = 0
    if policy.elide_tool_call_args:
        conv, elided_args = elide_stale_tool_call_args(
            conv,
            keep_last_per_tool=2,
            min_elide_chars=policy.min_elide_chars,
        )
    assert_pairing_ok(conv)
    if (masked or elided or elided_args) and on_event is not None:
        after = estimate_conversation_tokens(conv, model_id=model_id)
        on_event(
            before_tokens=before,
            after_tokens=after,
            masked_renders=masked,
            elided_tool_msgs=elided,
            elided_tool_call_args=elided_args,
            keep_round=keep_round,
        )
    return conv


def enable_native_compaction(
    llm: Any,
    provider: str,
    policy: CompressionPolicy,
    *,
    supports_cost_compaction: bool = False,
    trace: Any = None,
) -> None:
    enabled = False
    if (
        provider == "anthropic"
        and policy.enable_native_summary
        and llm is not None
        and hasattr(llm, "enable_compaction")
    ):
        llm.enable_compaction = True
        enabled = True
    if (
        provider == "openai"
        and supports_cost_compaction
        and policy.enabled
        and policy.enable_native_summary
        and llm is not None
        and bool(getattr(llm, "supports_native_cost_compaction", False))
    ):
        enable = getattr(llm, "enable_cost_compaction_for_session", None)
        if callable(enable):
            enable()
            enabled = True
    _trace_native_compaction_availability(trace, provider=provider, enabled=enabled)


def _trace_native_compaction_availability(trace: Any, *, provider: str, enabled: bool) -> None:
    if trace is None:
        return
    if enabled:
        availability, reason = "enabled", "native server-side compaction enabled for this run"
    else:
        availability = "disabled"
        reason = "provider supports native compaction but this run did not enable it"
    try:
        trace.write_event(
            "native_compaction_availability",
            {"provider": provider, "availability": availability, "reason": reason},
        )
    except Exception:  # noqa: BLE001 - telemetry must never break harness construction
        pass


@dataclass(frozen=True, slots=True)
class PrepareAccounting:

    maintenance_cost_usd: float = 0.0
    compaction_cost_usd: float = 0.0
    compaction_accounting_status: str | None = None


def consume_prepare_result(
    prepare_result: Any, *, cost: Any, display: Any, trace: Any = None
) -> PrepareAccounting:
    from pipeline.llm.errors import CostAccountingError

    if prepare_result is None:
        return PrepareAccounting()
    maintenance_cost_usd = 0.0
    for event in getattr(prepare_result, "trace_events", None) or []:
        event_type = getattr(event, "event_type", None)
        payload = getattr(event, "payload", None)
        if trace is not None and isinstance(event_type, str) and isinstance(payload, dict):
            try:
                trace.write_event(event_type, payload)
            except Exception:  # noqa: BLE001 — telemetry must never break the loop
                pass
    for event in getattr(prepare_result, "maintenance_events", None) or []:
        if cost is None and isinstance(event, dict) and event.get("usage"):
            raise CostAccountingError("Provider maintenance usage has no cost tracker")
        if cost is not None and isinstance(event, dict):
            try:
                maintenance_cost_usd += cost.add_maintenance_event(event).total_cost
            except Exception as exc:  # noqa: BLE001 - billing failures stop the run
                raise CostAccountingError(
                    "Provider maintenance usage could not be accounted"
                ) from exc
    compaction_event = getattr(prepare_result, "compaction_event", None)
    if compaction_event is None:
        return PrepareAccounting(maintenance_cost_usd=maintenance_cost_usd)
    billed = 0.0
    usage = getattr(compaction_event, "usage", None)
    if cost is None:
        raise CostAccountingError("Compaction usage has no cost tracker")
    if not isinstance(usage, dict) or any(
        not isinstance(usage.get(key), int) or usage[key] < 0
        for key in ("prompt_tokens", "candidates_tokens")
    ):
        raise CostAccountingError("Compaction response is missing valid input/output token usage")
    try:
        billed = cost.add_maintenance_event(compaction_event.to_dict()).total_cost
    except Exception as exc:  # noqa: BLE001 - billing failures stop the run
        raise CostAccountingError("Compaction usage could not be accounted") from exc
    accounting_status = "priced"
    maintenance_cost_usd += billed
    add = getattr(display, "add_compaction_event", None)
    if callable(add):
        display_usage = usage or {}
        display_billed = billed if accounting_status == "priced" else None
        try:
            add(
                trigger=compaction_event.trigger,
                estimated_saved_next_input_tokens=(
                    compaction_event.estimated_saved_next_input_tokens
                ),
                billed_cost=display_billed,
                previous_response_id_cleared=compaction_event.previous_response_id_cleared,
                usage_total_tokens=(
                    display_usage.get("total_tokens") if isinstance(display_usage, dict) else None
                ),
                estimate_error=compaction_event.estimate_error,
            )
        except Exception:  # noqa: BLE001 — telemetry must never break the loop
            pass
    return PrepareAccounting(
        maintenance_cost_usd=maintenance_cost_usd,
        compaction_cost_usd=billed,
        compaction_accounting_status=accounting_status,
    )


def emit_boundary_compaction_event(
    *,
    display: Any,
    before_tokens: int,
    after_tokens: int,
    masked_renders: int,
    elided_tool_msgs: int,
) -> None:
    add = getattr(display, "add_compaction_event", None)
    if not callable(add):
        return
    add(
        trigger="boundary_reseed",
        estimated_saved_next_input_tokens=max(0, before_tokens - after_tokens),
        billed_cost=0.0,
        previous_response_id_cleared=False,
        usage_total_tokens=None,
        estimate_error=None,
    )


def boundary_compact(harness: Any, conv: list[dict[str, Any]]) -> list[dict[str, Any]]:
    new, elided_motion_selfchecks = elide_stale_motion_selfcheck_nudges(conv)
    if not new:
        return new

    policy = getattr(harness, "_policy", None)
    if (
        policy is not None
        and policy.enabled
        and bool(getattr(policy, "expire_review_evidence", True))
    ):
        new, expired_evidence = mask_stale_review_evidence(new)
    else:
        expired_evidence = 0
    model_id = normalize_model_id(
        getattr(getattr(harness, "llm", None), "model_id", None) or "gpt-5.6-sol"
    )
    display = getattr(harness, "display", None)
    tier_b_changed = False
    boundary_stats = {
        "masked_renders": 0,
        "elided_tool_messages": 0,
        "elided_tool_call_args": 0,
        "elided_motion_selfchecks": elided_motion_selfchecks,
    }

    def _on_event(
        *,
        before_tokens,
        after_tokens,
        masked_renders,
        elided_tool_msgs,
        elided_tool_call_args,
        keep_round,
    ):
        nonlocal tier_b_changed
        tier_b_changed = True
        boundary_stats["masked_renders"] = int(masked_renders)
        boundary_stats["elided_tool_messages"] = int(elided_tool_msgs)
        boundary_stats["elided_tool_call_args"] = int(elided_tool_call_args)
        emit_boundary_compaction_event(
            display=display,
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            masked_renders=masked_renders,
            elided_tool_msgs=elided_tool_msgs,
        )

    if policy is not None and policy.enabled and policy.boundary_reseed:
        new = reseed_at_boundary(new, policy=policy, model_id=model_id, on_event=_on_event)
    if tier_b_changed or elided_motion_selfchecks:
        reset_tier_a_dedup(harness)
    if expired_evidence or tier_b_changed or elided_motion_selfchecks:
        harness._expect_cold_next_call = True
        trace = getattr(harness, "trace", None)
        if trace is not None:
            try:
                before_tokens = estimate_conversation_tokens(conv, model_id=model_id)
                after_tokens = estimate_conversation_tokens(new, model_id=model_id)
                trace.write_event(
                    "boundary_reseed",
                    {
                        "before_tokens": before_tokens,
                        "after_tokens": after_tokens,
                        "estimated_saved_next_input_tokens": max(0, before_tokens - after_tokens),
                        "masked_renders": boundary_stats["masked_renders"],
                        "elided_tool_messages": boundary_stats["elided_tool_messages"],
                        "elided_tool_call_args": boundary_stats["elided_tool_call_args"],
                        "elided_motion_selfchecks": elided_motion_selfchecks,
                        "expired_evidence": int(expired_evidence),
                    },
                )
            except Exception:  # noqa: BLE001 - telemetry must never break loading
                pass
        if elided_motion_selfchecks and trace is not None:
            try:
                trace.write_event(
                    "motion_selfcheck_elision",
                    {"elided_motion_selfchecks": elided_motion_selfchecks},
                )
            except Exception:  # noqa: BLE001 - telemetry must never break loading
                pass
        if getattr(harness, "provider", None) == "openai":
            llm = getattr(harness, "llm", None)
            reset = getattr(llm, "reset_canonical_history", None)
            if callable(reset):
                reset(preserve_policy_state=True)
            else:
                legacy_reset = getattr(llm, "reset_canonical_state", None)
                if callable(legacy_reset):
                    legacy_reset()
    return new


def reset_tier_a_dedup(harness: Any) -> None:
    for attr in (
        "_read_cache",
        "_read_seen_turn",
        "_read_coverage",
        "_probe_cache",
        "_probe_code_full_chars",
        "_qa_run_cache",
        "_lookup_result_chars",
        "_mass_audit_cache",
    ):
        cache = getattr(harness, attr, None)
        if isinstance(cache, dict):
            cache.clear()
    for attr in (
        "_examples_shown",
        "_examples_truncated",
        "_docs_read",
        "_lookup_shown",
        "_probe_code_seen",
        "_reported_motion_block_fingerprints",
    ):
        shown = getattr(harness, attr, None)
        if isinstance(shown, set):
            shown.clear()
    if hasattr(harness, "_probe_done"):
        harness._probe_done = False
    if hasattr(harness, "_probe_full_chars"):
        harness._probe_full_chars = 0
