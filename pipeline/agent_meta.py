from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

_AGENT_ORDER = {"design": 0, "coding": 1, "review": 2}


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def _new_bucket() -> dict[str, int]:
    return {"turns": 0, "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}


def _accumulate_usage(bucket: dict[str, int], usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    bucket["turns"] += 1
    bucket["input_tokens"] += int(usage.get("prompt_tokens") or 0)
    bucket["output_tokens"] += int(
        usage.get("candidates_tokens") or usage.get("completion_tokens") or 0
    )
    bucket["cached_tokens"] += int(usage.get("cached_tokens") or 0)


def read_run_summary(workspace: Path) -> dict[str, Any]:
    path = workspace / "trace" / "run_summary.json"
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def resolve_pipeline_provider_model(workspace: Path) -> tuple[str | None, str | None]:
    for obj in _iter_jsonl(workspace / "trace" / "design.jsonl"):
        if obj.get("type") in {"llm_call", "design_blueprint_turn"}:
            provider = obj.get("provider")
            model = obj.get("model_id")
            if provider or model:
                return provider, model
            break
    from pipeline.providers import pipeline_model, pipeline_provider

    return pipeline_provider(strict=False), pipeline_model()


def resolve_pipeline_models_by_agent(workspace: Path) -> dict[str, str]:
    models: dict[str, str] = {}
    for obj in _iter_jsonl(workspace / "trace" / "design.jsonl"):
        if obj.get("type") in {"llm_call", "design_blueprint_turn"} and obj.get("model_id"):
            models["design"] = str(obj["model_id"])
            break
    for obj in _iter_jsonl(workspace / "trace" / "timing.jsonl"):
        if obj.get("kind") != "llm":
            continue
        agent = obj.get("agent")
        model = obj.get("model_id")
        if agent and model and str(agent) not in models:
            models[str(agent)] = str(model)
    return models


def _tally_token_buckets(workspace: Path) -> dict[str, dict[str, int]]:
    buckets = {agent: _new_bucket() for agent in ("design", "coding", "review")}
    for obj in _iter_jsonl(workspace / "trace" / "design.jsonl"):
        if obj.get("type") in ("llm_call", "design_blueprint_turn"):
            _accumulate_usage(buckets["design"], obj.get("usage"))
    for agent, sub in (
        ("coding", "trace/trajectory.jsonl"),
        ("review", "trace/review/trajectory.jsonl"),
    ):
        for obj in _iter_jsonl(workspace / sub):
            if obj.get("type") != "message":
                continue
            msg = obj.get("message")
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                _accumulate_usage(buckets[agent], msg.get("usage"))
    return buckets


def build_pipeline_agent_meta(
    workspace: Path,
    state: dict[str, Any],
    *,
    provider: str | None,
    model: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = read_run_summary(workspace)
    cost_by_agent = summary.get("cost_by_agent") or {}
    seconds_by_agent = summary.get("active_seconds_by_agent") or {}
    buckets = _tally_token_buckets(workspace)
    models_by_agent = resolve_pipeline_models_by_agent(workspace)

    agents: list[dict[str, Any]] = []
    total_in = total_out = total_cached = 0
    for name in sorted(buckets, key=lambda a: _AGENT_ORDER.get(a, 9)):
        bucket = buckets[name]
        cost = round(float(cost_by_agent.get(name) or 0.0), 6)
        if bucket["turns"] == 0 and cost == 0.0:
            continue
        total_in += bucket["input_tokens"]
        total_out += bucket["output_tokens"]
        total_cached += bucket["cached_tokens"]
        agents.append(
            {
                "agent": name,
                "provider": provider,
                "model": models_by_agent.get(name) or model,
                "turns": bucket["turns"],
                "input_tokens": bucket["input_tokens"],
                "output_tokens": bucket["output_tokens"],
                "cached_tokens": bucket["cached_tokens"],
                "cost_usd": cost,
                "active_seconds": round(float(seconds_by_agent.get(name) or 0.0), 1),
            }
        )

    total_cost = state.get("cost_usd")
    if total_cost is None:
        total_cost = summary.get("total_cost_usd")
    active = float(state.get("active_seconds") or 0.0)
    if active <= 0.0:
        active = float(summary.get("active_seconds_total") or 0.0)
    elapsed = float(state.get("elapsed_seconds") or 0.0)
    if elapsed <= 0.0:
        elapsed = active
    cost_total = {
        "costs_usd": {"total": round(float(total_cost or 0.0), 6)},
        "tokens": {
            "prompt_tokens": total_in,
            "candidates_tokens": total_out,
            "total_tokens": total_in + total_out,
            "cached_tokens": total_cached,
        },
        "elapsed_seconds": round(elapsed, 1),
        "active_seconds": round(active, 1),
    }
    return agents, cost_total


def coding_turn_count(agents: list[dict[str, Any]]) -> int:
    for row in agents:
        if row.get("agent") == "coding":
            return int(row.get("turns") or 0)
    return 0
