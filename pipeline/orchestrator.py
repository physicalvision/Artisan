from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from rich.console import Console

from pipeline.contracts import TerminationReason
from pipeline.graph import StageOutcome, decide_next
from pipeline.policies import DEFAULT_POLICIES, Policies
from pipeline.state import (
    TERMINAL,
    Phase,
    PipelineState,
    agent_for_phase,
    load_checkpoint,
    save_checkpoint,
)

logger = logging.getLogger(__name__)
_CONSOLE = Console()

NodeFn = Callable[[PipelineState, "PipelineContext"], Awaitable[StageOutcome]]


def _append_event(state: PipelineState, phase: Phase, outcome: StageOutcome, transition) -> None:
    try:
        from pipeline import workspace

        path = workspace.workspace_root(Path(state.repo_root), state.run_id) / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        status = getattr(
            getattr(outcome, "status", None), "value", str(getattr(outcome, "status", ""))
        )
        event = {
            "ts": time.time(),
            "phase": phase.value,
            "agent": _agent_of(phase.value),
            "iter": state.iterations(phase),
            "status": status,
            "rating": getattr(outcome, "realism_rating", None),
            "next": transition.next_phase.value,
            "note": transition.note,
            "cost_usd": float(state.cost_usd),
            "termination_reason": (
                outcome.termination_reason.value if outcome.termination_reason else None
            ),
        }
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    except Exception:  # noqa: BLE001 — timeline logging must never break the run
        pass


def _append_cost_fuse_event(state: PipelineState, phase: Phase, outcome: StageOutcome) -> None:
    try:
        from pipeline import workspace

        path = workspace.workspace_root(Path(state.repo_root), state.run_id) / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "ts": time.time(),
            "phase": phase.value,
            "agent": _agent_of(phase.value),
            "iter": state.iterations(phase),
            "status": "cost_fuse",
            "rating": getattr(outcome, "realism_rating", None),
            "next": Phase.FAILED.value,
            "note": "cost fuse after completed node",
            "cost_usd": float(state.cost_usd),
            "fuse_timing": "post_node",
            "node_executed": True,
            "termination_reason": TerminationReason.COST_LIMIT.value,
        }
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    except Exception:  # noqa: BLE001 — timeline logging must never break the run
        pass


def _agent_of(phase: str) -> str:
    return agent_for_phase(phase).value


def _terminal_details(phase: Phase, status: str, summary: str) -> dict[str, str]:
    return {
        "phase": phase.value,
        "agent": _agent_of(phase.value),
        "status": status,
        "summary": summary,
    }


def _read_jsonl_generations(path: Path) -> list[dict[str, Any]]:
    paths = sorted(path.parent.glob(f"{path.stem}.r*{path.suffix}"))
    if path.exists():
        paths.append(path)
    records: list[dict[str, Any]] = []
    for candidate in paths:
        try:
            lines = candidate.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (TypeError, ValueError):
                continue
            if isinstance(record, dict):
                records.append(record)
    records.sort(
        key=lambda record: (
            float(record.get("ts")) if isinstance(record.get("ts"), (int, float)) else float("inf")
        )
    )
    return records


def _usage_value(tokens: Any, *names: str) -> int:
    if not isinstance(tokens, dict):
        return 0
    for name in names:
        value = tokens.get(name)
        if isinstance(value, (int, float)):
            return max(0, int(value))
    details = tokens.get("input_tokens_details")
    if isinstance(details, dict) and "cached_tokens" in names:
        value = details.get("cached_tokens")
        if isinstance(value, (int, float)):
            return max(0, int(value))
    return 0


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def _summarize_llm_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_total = 0
    prompt_usage_call_count = 0
    prompt_peak = 0
    cached_total = 0
    infra_prompt_total = 0
    infra_cached_total = 0
    expected_cold_calls = 0
    cache_breaks = 0
    previous_prompt: int | None = None
    for row in rows:
        tokens = row.get("tokens")
        prompt = _usage_value(tokens, "prompt_tokens", "input_tokens")
        cached = min(prompt, _usage_value(tokens, "cached_tokens"))
        expected_cold = bool(row.get("expected_cold", False))
        prompt_total += prompt
        if prompt > 0:
            prompt_usage_call_count += 1
        cached_total += cached
        prompt_peak = max(prompt_peak, prompt)
        if expected_cold:
            expected_cold_calls += 1
        else:
            infra_prompt_total += prompt
            infra_cached_total += cached
        if (
            previous_prompt is not None
            and previous_prompt > 20_000
            and not expected_cold
            and cached < 0.5 * min(previous_prompt, prompt)
        ):
            cache_breaks += 1
        previous_prompt = prompt
    return {
        "call_count": len(rows),
        "prompt_usage_call_count": prompt_usage_call_count,
        "prompt_tokens_total": prompt_total,
        "prompt_tokens_avg": (
            prompt_total / prompt_usage_call_count if prompt_usage_call_count else 0.0
        ),
        "prompt_tokens_peak": prompt_peak,
        "cached_tokens_total": cached_total,
        "effective_cache_ratio": (cached_total / prompt_total if prompt_total else 0.0),
        "infra_cache_ratio": (
            infra_cached_total / infra_prompt_total if infra_prompt_total else 0.0
        ),
        "expected_cold_calls": expected_cold_calls,
        "cache_breaks": cache_breaks,
    }


def _aggregate_compression_telemetry(trace_root: Path) -> dict[str, Any]:
    boundary = {
        "count": 0,
        "before_tokens": 0,
        "after_tokens": 0,
        "estimated_saved_next_input_tokens": 0,
        "masked_renders": 0,
        "elided_tool_messages": 0,
        "elided_tool_call_args": 0,
        "expired_evidence": 0,
    }
    tier_a = {"rollup_count": 0, "action_count": 0, "saved_chars": 0, "by_tool": {}}
    compaction = {
        "count": 0,
        "estimated_saved_next_input_tokens": 0,
        "post_compaction_observed_count": 0,
        "observations": [],
        "by_model": {},
    }

    trajectory: list[dict[str, Any]] = []
    for relative in ("trajectory.jsonl", "review/trajectory.jsonl"):
        agent = "review" if relative.startswith("review/") else "coding"
        for record in _read_jsonl_generations(trace_root / relative):
            trajectory.append({**record, "_compression_agent": agent})
    trajectory.sort(
        key=lambda record: (
            float(record.get("ts")) if isinstance(record.get("ts"), (int, float)) else float("inf")
        )
    )
    for event in trajectory:
        event_type = event.get("type")
        if event_type == "boundary_reseed":
            boundary["count"] += 1
            for key in boundary:
                if key == "count":
                    continue
                value = event.get(key, 0)
                if isinstance(value, (int, float)):
                    boundary[key] += int(value)
        elif event_type == "tier_a_dedup":
            actions = _nonnegative_int(event.get("action_count"))
            saved = _nonnegative_int(event.get("saved_chars"))
            tool = str(event.get("tool") or "unknown")
            tier_a["rollup_count"] += 1
            tier_a["action_count"] += actions
            tier_a["saved_chars"] += saved
            by_tool = tier_a["by_tool"].setdefault(tool, {"action_count": 0, "saved_chars": 0})
            by_tool["action_count"] += actions
            by_tool["saved_chars"] += saved
        elif event_type == "compaction":
            compaction["count"] += 1
            saved = event.get("estimated_saved_next_input_tokens", 0)
            if isinstance(saved, (int, float)):
                compaction["estimated_saved_next_input_tokens"] += max(0, int(saved))
            model_id = str(event.get("model_id") or "unknown")
            model_summary = compaction["by_model"].setdefault(
                model_id,
                {"count": 0, "cost_usd": 0.0},
            )
            model_summary["count"] += 1

    llm_rows = [
        row
        for row in _read_jsonl_generations(trace_root / "timing.jsonl")
        if row.get("kind") == "llm"
    ]
    maintenance_rows = [
        row
        for row in _read_jsonl_generations(trace_root / "timing.jsonl")
        if row.get("kind") == "maintenance" and row.get("name") == "compaction"
    ]
    for row in maintenance_rows:
        model_id = str(row.get("model_id") or "unknown")
        model_summary = compaction["by_model"].setdefault(
            model_id,
            {"count": 0, "cost_usd": 0.0},
        )
        cost_usd = row.get("cost_usd")
        if isinstance(cost_usd, (int, float)):
            model_summary["cost_usd"] += float(cost_usd)

    used_llm_rows: set[int] = set()
    for event in (record for record in trajectory if record.get("type") == "compaction"):
        event_ts = event.get("ts")
        agent = event.get("_compression_agent")
        if not isinstance(event_ts, (int, float)):
            continue
        matched_index: int | None = None
        for index, row in enumerate(llm_rows):
            if index in used_llm_rows or row.get("agent") != agent:
                continue
            row_ts = row.get("ts")
            if (
                isinstance(row_ts, (int, float))
                and row_ts >= event_ts
                and bool(row.get("expected_cold", False))
            ):
                matched_index = index
                break
        if matched_index is None:
            continue
        used_llm_rows.add(matched_index)
        row = llm_rows[matched_index]
        post_prompt_tokens = _usage_value(row.get("tokens"), "prompt_tokens", "input_tokens")
        trigger_prompt_tokens = event.get("trigger_prompt_tokens")
        observed_prompt_delta = (
            int(trigger_prompt_tokens) - post_prompt_tokens
            if isinstance(trigger_prompt_tokens, (int, float)) and post_prompt_tokens > 0
            else None
        )
        persisted_saved = event.get("estimated_saved_next_input_tokens")
        if (
            isinstance(observed_prompt_delta, int)
            and observed_prompt_delta > 0
            and (not isinstance(persisted_saved, (int, float)) or float(persisted_saved) <= 0.0)
        ):
            compaction["estimated_saved_next_input_tokens"] += observed_prompt_delta
        observation = {
            "agent": agent,
            "provider": event.get("provider"),
            "model_id": event.get("model_id"),
            "trigger": event.get("trigger"),
            "trigger_prompt_tokens": (
                int(trigger_prompt_tokens)
                if isinstance(trigger_prompt_tokens, (int, float))
                else None
            ),
            "post_compaction_first_prompt_tokens": post_prompt_tokens or None,
            "observed_prompt_delta": observed_prompt_delta,
        }
        compaction["observations"].append(observation)
        compaction["post_compaction_observed_count"] += 1
    rows_by_agent: dict[str, list[dict[str, Any]]] = {}
    for row in llm_rows:
        agent = str(row.get("agent") or row.get("name") or "unknown")
        rows_by_agent.setdefault(agent, []).append(row)
    by_agent = {
        agent: _summarize_llm_usage(agent_rows)
        for agent, agent_rows in sorted(rows_by_agent.items())
    }
    llm_usage = _summarize_llm_usage(llm_rows)
    llm_usage["cache_breaks"] = sum(int(summary["cache_breaks"]) for summary in by_agent.values())
    llm_usage["by_agent"] = by_agent
    return {
        "boundary_reseed": boundary,
        "tier_a_dedup": tier_a,
        "compaction": compaction,
        "llm_usage": llm_usage,
    }


def _write_run_summary(state: PipelineState) -> None:
    try:
        from pipeline import workspace
        from pipeline.state import PAUSE_GAP_CAP_S

        root = workspace.workspace_root(Path(state.repo_root), state.run_id)
        events: list[dict] = []
        epath = root / "events.jsonl"
        if epath.exists():
            for line in epath.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        events.sort(key=lambda e: e.get("ts", 0) or 0)
        start = 0
        last_cost = 0.0
        for i, e in enumerate(events):
            if e.get("cost_usd") is None:
                continue
            cost = float(e["cost_usd"])
            if cost < last_cost - 1e-6:
                start = i
            last_cost = cost
        events = events[start:]
        cost_by = {"design": 0.0, "coding": 0.0, "review": 0.0, "export": 0.0}
        time_by = {"design": 0.0, "coding": 0.0, "review": 0.0, "export": 0.0}
        redo = {"P4_build": 0, "P5_review": 0}
        prev_cost = 0.0
        prev_ts: float | None = None
        for e in events:
            ph = str(e.get("phase", ""))
            agent = _agent_of(ph)
            raw_cost = e.get("cost_usd")
            cur_cost = prev_cost if raw_cost is None else float(raw_cost)
            if cur_cost - prev_cost > 0:
                cost_by[agent] += cur_cost - prev_cost
            prev_cost = cur_cost
            ts = e.get("ts")
            if prev_ts is not None and ts is not None:
                dt = min(float(ts) - prev_ts, PAUSE_GAP_CAP_S)
                if dt > 0:
                    time_by[agent] += dt
            if ts is not None:
                prev_ts = float(ts)
            if e.get("status") == "redo" and ph in redo:
                redo[ph] += 1

        tool_seconds: dict[str, float] = {}
        llm_seconds = 0.0
        for sub in ("trajectory.jsonl", "review/trajectory.jsonl"):
            prev_ts: float | None = None
            for rec in _read_jsonl_generations(root / "trace" / sub):
                ts = rec.get("ts")
                msg = rec.get("message") if isinstance(rec.get("message"), dict) else {}
                role = msg.get("role")
                dur = rec.get("duration_s")
                if dur is None and isinstance(ts, (int, float)) and prev_ts is not None:
                    gap = float(ts) - prev_ts
                    dur = gap if 0 <= gap <= 1800 else None
                if isinstance(ts, (int, float)):
                    prev_ts = float(ts)
                if dur is None:
                    continue
                if role == "assistant":
                    llm_seconds += float(dur)
                elif role == "tool":
                    nm = str(msg.get("name") or "?")
                    tool_seconds[nm] = tool_seconds.get(nm, 0.0) + float(dur)

        summary = {
            "run_id": state.run_id,
            "status": state.status,
            "total_cost_usd": round(state.cost_usd, 6),
            "cost_by_agent": {k: round(v, 6) for k, v in cost_by.items()},
            "active_seconds_by_agent": {k: round(v, 1) for k, v in time_by.items()},
            "active_seconds_total": round(sum(time_by.values()), 1),
            "llm_seconds": round(llm_seconds, 1),
            "tool_seconds_by_name": {
                k: round(v, 1) for k, v in sorted(tool_seconds.items(), key=lambda kv: -kv[1])
            },
            "build_rounds": sum(1 for e in events if e.get("phase") == "P4_build"),
            "review_rounds": sum(1 for e in events if e.get("phase") == "P5_review"),
            "redo_counts": redo,
            "final_rating": state.last_realism_rating,
            "termination_reason": state.termination_reason,
            "terminal_details": dict(state.terminal_details),
            "compression": _aggregate_compression_telemetry(root / "trace"),
        }
        from pipeline.timing_summary import summarize_recorded_timings

        recorded = summarize_recorded_timings(
            _read_jsonl_generations(root / "trace" / "timing.jsonl")
        )
        if recorded is not None:
            summary["recorded_timing"] = recorded
        out = root / "trace" / "run_summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:  # noqa: BLE001 — observability must never break the run
        logger.exception("run summary write failed (non-fatal)")


def _finalize_cost_fused_run(state: PipelineState, ctx: "PipelineContext") -> None:
    try:
        from pipeline import workspace
        from sdk.runtime.release_candidate import (
            compute_candidate_identity,
            load_manifest,
            sha256_file,
        )

        if state.fallback_candidate.get("status") == "restored_after_failed_repair":
            logger.info("cost fuse finalization skipped: restored fallback is run-local only")
            return
        if not state.has_passed(Phase.P5_REVIEW):
            logger.info("cost fuse finalization skipped: P5 is not marked passed")
            return

        root = workspace.workspace_root(Path(state.repo_root), state.run_id)
        model = root / "model.py"
        blueprint = root / "blueprint.json"
        manifest_path = root / "candidate" / "manifest.json"
        if not model.is_file() or not blueprint.is_file() or not manifest_path.is_file():
            return
        try:
            manifest = load_manifest(manifest_path)
            if (
                manifest.status != "accepted"
                or not state.candidate_id
                or manifest.candidate_id != state.candidate_id
                or not state.reviewed_physics_report_sha256
                or manifest.physics_report_sha256 != state.reviewed_physics_report_sha256
            ):
                return
            current_identity = compute_candidate_identity(
                model,
                blueprint,
                formats=list(manifest.formats),
                accepted_fcl_pose_sha256=manifest.accepted_fcl_pose_sha256,
                simulator_collision_digest=manifest.simulator_collision_digest,
                collision_resolution_request_key=(manifest.collision_resolution_request_key),
            )
            if manifest.build_id != current_identity.build_id:
                return
            if not isinstance(manifest.physics_report_path, str):
                return
            report_relative = Path(manifest.physics_report_path)
            if report_relative.is_absolute() or ".." in report_relative.parts:
                return
            report_path = manifest_path.parent / report_relative
            if (
                not report_path.is_file()
                or sha256_file(report_path) != manifest.physics_report_sha256
            ):
                return
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                not isinstance(report, dict)
                or report.get("status") != "pass"
                or report.get("candidate_id") != manifest.candidate_id
            ):
                return
        except Exception:  # noqa: BLE001 — unreadable manifest -> nothing to trust
            return
        exports_dir = root / "exports"
        if (exports_dir / "export_manifest.json").exists():
            return
        rep = ctx.backend.publish_release_candidate(
            manifest_path,
            exports_dir,
            expected_candidate_id=manifest.candidate_id,
        )
        exports = rep.get("exports", {}) if isinstance(rep, dict) else {}
        (root / "reports").mkdir(parents=True, exist_ok=True)
        (root / "reports" / "P8_export_smoke.json").write_text(
            json.dumps(
                {
                    "phase": "P8_export_smoke",
                    "ok": bool(exports.get("glb")) and bool(exports.get("urdf")),
                    "exports": exports,
                    "skipped": rep.get("skipped", []) if isinstance(rep, dict) else [],
                    "candidate_id": (
                        (rep.get("candidate_id") or manifest.candidate_id)
                        if isinstance(rep, dict)
                        else manifest.candidate_id
                    ),
                    "publish_only": True,
                    "finalized_after_cost_fuse": True,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        logger.info("cost fuse: accepted candidate published to %s", exports_dir)
    except Exception:  # noqa: BLE001 — the wrap-up must never mask the fuse itself
        logger.exception("cost-fuse finalization failed (non-fatal)")


_MAX_UNBOUNDED_TRANSITIONS = 200


@dataclass(slots=True)
class PipelineContext:
    repo_root: Path
    backend: object
    policies: Policies = DEFAULT_POLICIES
    max_cost_usd: float | None = None


def _attempt_salvage(state: PipelineState, ctx: "PipelineContext") -> None:
    from pipeline.salvage import evaluate_salvage, publish_salvaged_candidate

    try:
        run_dir = Path(state.repo_root) / "data" / "cache" / "runs" / state.run_id
        decision = evaluate_salvage(run_dir)
        if not decision.salvageable:
            state.salvage = {"delivered": False, "refusal": decision.refusal}
            logger.info("salvage refused for %s: %s", state.run_id, decision.refusal)
            return
        publish_salvaged_candidate(run_dir, decision)
        from pipeline.ingest import ingest_run

        result = ingest_run(Path(state.repo_root), state.run_id, salvage=decision)
        state.salvage = {
            "delivered": True,
            "grade": decision.grade,
            "candidate_id": decision.candidate_id,
            "record_id": str(result.get("record_id") or ""),
            "open_blocking_issue_ids": list(decision.open_blocking_issue_ids),
        }
        logger.warning(
            "run %s delivered a DEGRADED asset (%s) as %s after %s",
            state.run_id,
            decision.grade,
            result.get("record_id"),
            state.termination_reason,
        )
    except Exception as exc:  # noqa: BLE001 - salvage must never mask the real outcome
        state.salvage = {"delivered": False, "refusal": f"{type(exc).__name__}: {exc}"}
        logger.exception("salvage attempt failed for %s", state.run_id)


async def run_pipeline(
    state: PipelineState,
    ctx: PipelineContext,
    nodes: Mapping[Phase, NodeFn],
) -> PipelineState:
    transitions = 0
    while state.phase not in TERMINAL:
        if ctx.max_cost_usd is None and transitions >= _MAX_UNBOUNDED_TRANSITIONS:
            exhausted_phase = state.phase
            state.phase = Phase.FAILED
            state.status = "failed"
            state.termination_reason = TerminationReason.HARD_FAILURE.value
            state.terminal_details = _terminal_details(
                exhausted_phase, "failed", "maximum unbounded transitions reached"
            )
            save_checkpoint(state)
            logger.error(
                "pipeline aborted without a cost ceiling: max transitions (%s)",
                _MAX_UNBOUNDED_TRANSITIONS,
            )
            return state

        phase = state.phase
        node = nodes.get(phase)
        if node is None:
            state.phase = Phase.FAILED
            state.status = "failed"
            state.termination_reason = TerminationReason.HARD_FAILURE.value
            state.terminal_details = _terminal_details(phase, "failed", "node is not registered")
            save_checkpoint(state)
            raise KeyError(f"no node registered for phase {phase.value}")

        if ctx.max_cost_usd is not None and state.cost_usd >= ctx.max_cost_usd:
            state.phase = Phase.FAILED
            state.status = "failed"
            state.termination_reason = TerminationReason.COST_LIMIT.value
            state.terminal_details = _terminal_details(
                phase, "failed", "cost limit reached before node execution"
            )
            save_checkpoint(state)
            logger.error(
                "cost fuse (pre-node): %.4f >= cap %.4f",
                state.cost_usd,
                ctx.max_cost_usd,
            )
            _finalize_cost_fused_run(state, ctx)
            _attempt_salvage(state, ctx)
            save_checkpoint(state)
            _write_run_summary(state)
            return state

        save_checkpoint(state)

        if phase is Phase.P9_FINAL:
            _write_run_summary(state)

        logger.info("phase %s (iter=%s)", phase.value, state.iterations(phase))
        _CONSOLE.rule(
            f"[bold cyan]{phase.value}[/]  (iter {state.iterations(phase)}, ${state.cost_usd:.3f})"
        )
        outcome = await node(state, ctx)

        if ctx.max_cost_usd is not None and state.cost_usd > ctx.max_cost_usd:
            _append_cost_fuse_event(state, phase, outcome)
            state.phase = Phase.FAILED
            state.status = "failed"
            state.termination_reason = TerminationReason.COST_LIMIT.value
            state.terminal_details = _terminal_details(
                phase, "failed", outcome.summary or "cost limit reached after node execution"
            )
            save_checkpoint(state)
            logger.error("cost fuse: %.4f > cap %.4f", state.cost_usd, ctx.max_cost_usd)
            _finalize_cost_fused_run(state, ctx)
            _attempt_salvage(state, ctx)
            save_checkpoint(state)
            _write_run_summary(state)
            return state

        transition = decide_next(state, phase, outcome, ctx.policies)
        logger.info("  → %s :: %s", transition.next_phase.value, transition.note)
        _CONSOLE.print(
            f"  [green]{phase.value}[/] → [bold]{transition.next_phase.value}[/]"
            f"  :: {transition.note}  [dim](${state.cost_usd:.3f})[/]"
        )
        _append_event(state, phase, outcome, transition)
        state.phase = transition.next_phase
        if transition.halt:
            state.status = "incomplete"
            state.termination_reason = (
                outcome.termination_reason or TerminationReason.HARD_FAILURE
            ).value
            state.terminal_details = _terminal_details(phase, "incomplete", outcome.summary)
            save_checkpoint(state)
            _attempt_salvage(state, ctx)
            save_checkpoint(state)
            _write_run_summary(state)
            return state
        if transition.next_phase is Phase.DONE:
            state.status = "done"
        elif transition.next_phase is Phase.FAILED:
            state.status = "failed"
            state.termination_reason = (
                outcome.termination_reason or TerminationReason.HARD_FAILURE
            ).value
            state.terminal_details = _terminal_details(phase, "failed", outcome.summary)
        save_checkpoint(state)
        transitions += 1

    _write_run_summary(state)
    return state


async def resume_pipeline(
    repo_root: Path,
    run_id: str,
    ctx: PipelineContext,
    nodes: Mapping[Phase, NodeFn],
    *,
    state: PipelineState | None = None,
) -> PipelineState:
    if state is None:
        state = load_checkpoint(repo_root, run_id)
    if state.phase in TERMINAL:
        logger.info("run %s already terminal (%s)", run_id, state.phase.value)
        return state
    state.status = "running"
    state.termination_reason = None
    state.terminal_details = {}
    logger.info("resuming run %s at phase %s", run_id, state.phase.value)
    return await run_pipeline(state, ctx, nodes)
