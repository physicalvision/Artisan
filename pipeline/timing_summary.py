from __future__ import annotations

import math
from typing import Any, Iterable


def interval_union_seconds(intervals: Iterable[tuple[float, float]]) -> float:
    total = 0.0
    end = -math.inf
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def summarize_recorded_timings(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    intervals = []
    llm_by_agent: dict[str, float] = {}
    llm_by_name: dict[str, float] = {}
    for row in rows:
        try:
            wall = float(row["wall_s"])
            end = float(row["ts"])
            start = float(row.get("started_at", end - wall))
        except (KeyError, ValueError, TypeError):
            continue
        if wall < 0 or not all(math.isfinite(v) for v in (wall, start, end)):
            continue
        intervals.append((start, start + wall))
        if row.get("kind") == "llm":
            agent = str(row.get("agent") or "unknown")
            name = str(row.get("name") or "unknown")
            llm_by_agent[agent] = llm_by_agent.get(agent, 0.0) + wall
            llm_by_name[name] = llm_by_name.get(name, 0.0) + wall
    if not intervals:
        return None
    coverage = interval_union_seconds(intervals)
    span = max(end for _, end in intervals) - min(start for start, _ in intervals)
    return {
        "schema_version": 1,
        "source": "trace/timing.jsonl and rotated generations",
        "llm_call_seconds_all_agents": round(sum(llm_by_agent.values()), 4),
        "llm_call_seconds_by_agent": llm_by_agent,
        "llm_call_seconds_by_name": llm_by_name,
        "instrumented_activity_union_s": round(coverage, 4),
        "first_to_last_recorded_activity_s": round(span, 4),
        "unrecorded_gaps_within_activity_span_s": round(max(0.0, span - coverage), 4),
        "scope": "LLM includes image requests; call sums are not concurrent elapsed time. "
        "Activity union excludes nested spans. Gaps include pauses and uninstrumented work. "
        "This is not total process wall time.",
    }
