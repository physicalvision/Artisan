from __future__ import annotations

from typing import Any

from pipeline.llm.errors import CostAccountingError


def require_token_usage(usage: Any, provider: str) -> dict[str, int]:
    if not isinstance(usage, dict) or any(
        type(usage.get(name)) is not int or usage[name] < 0
        for name in ("prompt_tokens", "candidates_tokens")
    ):
        raise CostAccountingError(f"{provider} response has missing or invalid token usage")
    if any(type(value) is not int or value < 0 for value in usage.values()):
        raise CostAccountingError(f"{provider} response has invalid usage counters")
    return usage


def require_reported_cost(usage: Any) -> dict[str, int]:
    if (
        not isinstance(usage, dict)
        or type(usage.get("cost_micro_usd")) is not int
        or usage["cost_micro_usd"] < 0
    ):
        raise CostAccountingError("OpenRouter response has missing or invalid reported cost")
    return usage
