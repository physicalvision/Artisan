from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from artisan.values import ProviderName, normalize_provider_name
from pipeline.llm.errors import CostAccountingError

MAX_COST_ENV_VAR = "ARTISAN_MAX_COST_USD"


OPENAI_GPT_5_3_CODEX_PRICING: dict[str, float] = {
    "input_uncached": 1.75,
    "input_cached": 0.175,
    "output": 14.00,
}
OPENAI_GPT_5_2_PRICING = OPENAI_GPT_5_3_CODEX_PRICING

OPENAI_GPT_5_4_PRICING: dict[str, float] = {
    "input_uncached": 2.50,
    "input_cached": 0.25,
    "output": 15.00,
    "prompt_tier_threshold_tokens": 272_000,
    "input_uncached_above_threshold": 5.00,
    "input_cached_above_threshold": 0.50,
    "output_above_threshold": 22.50,
}

OPENAI_GPT_5_5_PRICING: dict[str, float] = {
    "input_uncached": 5.00,
    "input_cached": 0.50,
    "output": 30.00,
    "prompt_tier_threshold_tokens": 272_000,
    "input_uncached_above_threshold": 10.00,
    "input_cached_above_threshold": 1.00,
    "output_above_threshold": 45.00,
}

OPENAI_GPT_5_6_SOL_PRICING: dict[str, float] = {
    "input_uncached": 4.00,
    "input_cached": 0.40,
    "input_cache_write": 5.00,
    "output": 20.00,
    "prompt_tier_threshold_tokens": 272_000,
    "input_uncached_above_threshold": 8.00,
    "input_cached_above_threshold": 0.80,
    "input_cache_write_above_threshold": 10.00,
    "output_above_threshold": 30.00,
}

OPENAI_GPT_5_6_TERRA_PRICING: dict[str, float] = {
    "input_uncached": 2.00,
    "input_cached": 0.20,
    "input_cache_write": 2.50,
    "output": 12.00,
    "prompt_tier_threshold_tokens": 272_000,
    "input_uncached_above_threshold": 4.00,
    "input_cached_above_threshold": 0.40,
    "input_cache_write_above_threshold": 5.00,
    "output_above_threshold": 18.00,
}

OPENAI_GPT_5_6_LUNA_PRICING: dict[str, float] = {
    "input_uncached": 0.20,
    "input_cached": 0.02,
    "input_cache_write": 0.25,
    "output": 1.20,
    "prompt_tier_threshold_tokens": 272_000,
    "input_uncached_above_threshold": 0.40,
    "input_cached_above_threshold": 0.04,
    "input_cache_write_above_threshold": 0.50,
    "output_above_threshold": 1.80,
}

OPENAI_GPT_6_ASTRA_PRICING: dict[str, float] = {
    "input_uncached": 10.00,
    "input_cached": 1.00,
    "input_cache_write": 12.50,
    "output": 50.00,
    "prompt_tier_threshold_tokens": 272_000,
    "input_uncached_above_threshold": 20.00,
    "input_cached_above_threshold": 2.00,
    "input_cache_write_above_threshold": 25.00,
    "output_above_threshold": 75.00,
}

ANTHROPIC_OPUS_PRICING: dict[str, float] = {
    "input_uncached": 5.0,
    "input_cached": 0.5,
    "input_cache_write": 6.25,
    "input_cache_write_1h": 10.0,
    "output": 25.0,
}
ANTHROPIC_SONNET_PRICING: dict[str, float] = {
    "input_uncached": 3.0,
    "input_cached": 0.3,
    "input_cache_write": 3.75,
    "input_cache_write_1h": 6.0,
    "output": 15.0,
}
ANTHROPIC_HAIKU_PRICING: dict[str, float] = {
    "input_uncached": 1.0,
    "input_cached": 0.1,
    "input_cache_write": 1.25,
    "input_cache_write_1h": 2.0,
    "output": 5.0,
}
GEMINI_3_5_FLASH_PRICING: dict[str, float] = {
    "input_uncached": 1.5,
    "input_cached": 0.15,
    "output": 9.0,
    "bill_reasoning_as_output": True,
}


@dataclass(slots=True)
class CostBreakdown:

    prompt_tokens: int = 0
    cached_tokens: int = 0
    uncached_prompt_tokens: int = 0
    candidates_tokens: int = 0
    total_tokens: int = 0
    input_uncached_cost: float = 0.0
    input_cached_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.uncached_prompt_tokens == 0 and self.prompt_tokens > 0:
            self.uncached_prompt_tokens = self.prompt_tokens - self.cached_tokens


def _breakdown_tokens_dict(breakdown: CostBreakdown) -> dict[str, int]:
    return {
        "prompt_tokens": breakdown.prompt_tokens,
        "cached_tokens": breakdown.cached_tokens,
        "uncached_prompt_tokens": breakdown.uncached_prompt_tokens,
        "candidates_tokens": breakdown.candidates_tokens,
        "total_tokens": breakdown.total_tokens,
    }


def _breakdown_costs_dict(breakdown: CostBreakdown) -> dict[str, float]:
    return {
        "input_uncached": round(breakdown.input_uncached_cost, 8),
        "input_cached": round(breakdown.input_cached_cost, 8),
        "output": round(breakdown.output_cost, 8),
        "total": round(breakdown.total_cost, 8),
    }


def _breakdown_dict(breakdown: CostBreakdown) -> dict[str, object]:
    return {
        "tokens": _breakdown_tokens_dict(breakdown),
        "costs_usd": _breakdown_costs_dict(breakdown),
    }


def _accumulate_breakdown(target: CostBreakdown, source: CostBreakdown) -> None:
    target.prompt_tokens += source.prompt_tokens
    target.cached_tokens += source.cached_tokens
    target.uncached_prompt_tokens += source.uncached_prompt_tokens
    target.candidates_tokens += source.candidates_tokens
    target.total_tokens += source.total_tokens
    target.input_uncached_cost += source.input_uncached_cost
    target.input_cached_cost += source.input_cached_cost
    target.output_cost += source.output_cost
    target.total_cost += source.total_cost


def calculate_cost(usage: dict[str, int], pricing: dict[str, float]) -> CostBreakdown:
    prompt_tokens = usage.get("prompt_tokens", 0)
    cached_tokens = usage.get("cached_tokens", 0)
    cache_creation_tokens = usage.get("cache_creation_input_tokens", 0)
    cache_creation_5m_tokens = usage.get("cache_creation_5m_input_tokens", 0)
    cache_creation_1h_tokens = usage.get("cache_creation_1h_input_tokens", 0)
    candidates_tokens = usage.get("candidates_tokens", 0)
    uncached_prompt_tokens = max(0, prompt_tokens - cached_tokens)
    regular_uncached_prompt_tokens = max(
        0,
        prompt_tokens - cached_tokens - cache_creation_tokens,
    )

    threshold = pricing.get("prompt_tier_threshold_tokens")
    if threshold is not None and prompt_tokens > int(threshold):
        input_uncached_rate = pricing.get(
            "input_uncached_above_threshold",
            pricing["input_uncached"],
        )
        input_cached_rate = pricing.get(
            "input_cached_above_threshold",
            pricing["input_cached"],
        )
        output_rate = pricing.get("output_above_threshold", pricing["output"])
    else:
        input_uncached_rate = pricing["input_uncached"]
        input_cached_rate = pricing["input_cached"]
        output_rate = pricing["output"]
    input_cache_write_rate = pricing.get("input_cache_write", input_uncached_rate)
    if threshold is not None and prompt_tokens > int(threshold):
        input_cache_write_rate = pricing.get(
            "input_cache_write_above_threshold",
            input_cache_write_rate,
        )
    input_cache_write_1h_rate = pricing.get("input_cache_write_1h", input_cache_write_rate)
    detailed_cache_creation_tokens = cache_creation_5m_tokens + cache_creation_1h_tokens
    unknown_cache_creation_tokens = max(0, cache_creation_tokens - detailed_cache_creation_tokens)
    cache_write_cost = (
        cache_creation_5m_tokens + unknown_cache_creation_tokens
    ) * input_cache_write_rate + cache_creation_1h_tokens * input_cache_write_1h_rate

    input_uncached_cost = (
        (regular_uncached_prompt_tokens * input_uncached_rate) + cache_write_cost
    ) / 1_000_000
    input_cached_cost = (cached_tokens * input_cached_rate) / 1_000_000
    billable_output_tokens = candidates_tokens
    if pricing.get("bill_reasoning_as_output"):
        billable_output_tokens += usage.get("reasoning_tokens", 0) or 0
    output_cost = (billable_output_tokens * output_rate) / 1_000_000
    total_cost = input_uncached_cost + input_cached_cost + output_cost

    return CostBreakdown(
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        uncached_prompt_tokens=uncached_prompt_tokens,
        candidates_tokens=candidates_tokens,
        total_tokens=usage.get("total_tokens", prompt_tokens + candidates_tokens),
        input_uncached_cost=input_uncached_cost,
        input_cached_cost=input_cached_cost,
        output_cost=output_cost,
        total_cost=total_cost,
    )


REPORTED_COST_USAGE_KEY = "cost_micro_usd"


def reported_breakdown(usage: dict[str, int]) -> CostBreakdown:
    prompt_tokens = usage.get("prompt_tokens", 0)
    cached_tokens = usage.get("cached_tokens", 0)
    candidates_tokens = usage.get("candidates_tokens", 0)
    return CostBreakdown(
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        uncached_prompt_tokens=max(0, prompt_tokens - cached_tokens),
        candidates_tokens=candidates_tokens,
        total_tokens=usage.get("total_tokens", prompt_tokens + candidates_tokens),
        total_cost=usage.get(REPORTED_COST_USAGE_KEY, 0) / 1_000_000,
    )


@dataclass(slots=True)
class CostTracker:

    model_id: str
    pricing: dict[str, float] | None
    cost_source: str = "priced"
    total_breakdown: CostBreakdown = field(default_factory=CostBreakdown)
    maintenance_breakdown: CostBreakdown = field(default_factory=CostBreakdown)
    maintenance_breakdown_by_model: dict[str, CostBreakdown] = field(default_factory=dict)
    turn_breakdowns: list[CostBreakdown] = field(default_factory=list)
    maintenance_events: list[dict[str, object]] = field(default_factory=list)

    @property
    def all_in_total_cost_usd(self) -> float:
        return self.total_breakdown.total_cost + self.maintenance_breakdown.total_cost

    def _require_pricing(self, context: str) -> dict[str, float]:
        if self.pricing is None:
            raise CostAccountingError(
                f"cost pricing unavailable for {self.model_id} ({context}): the provider "
                f"reported no {REPORTED_COST_USAGE_KEY} and no rate table is configured"
            )
        return self.pricing

    def add_turn(self, usage: dict[str, int]) -> CostBreakdown:
        if REPORTED_COST_USAGE_KEY in usage:
            turn_cost = reported_breakdown(usage)
            self.cost_source = "reported"
        else:
            turn_cost = calculate_cost(usage, self._require_pricing("turn"))
        self.turn_breakdowns.append(turn_cost)
        self.total_breakdown.prompt_tokens += turn_cost.prompt_tokens
        self.total_breakdown.cached_tokens += turn_cost.cached_tokens
        self.total_breakdown.uncached_prompt_tokens += turn_cost.uncached_prompt_tokens
        self.total_breakdown.candidates_tokens += turn_cost.candidates_tokens
        self.total_breakdown.total_tokens += turn_cost.total_tokens
        self.total_breakdown.input_uncached_cost += turn_cost.input_uncached_cost
        self.total_breakdown.input_cached_cost += turn_cost.input_cached_cost
        self.total_breakdown.output_cost += turn_cost.output_cost
        self.total_breakdown.total_cost += turn_cost.total_cost
        return turn_cost

    def format_cost_summary(self, turn_cost: CostBreakdown | None = None) -> str:
        if turn_cost is None:
            return f"Total cost: ${self.total_breakdown.total_cost:.6f}"
        return (
            f"Turn: ${turn_cost.total_cost:.6f} "
            f"(input: ${turn_cost.input_uncached_cost + turn_cost.input_cached_cost:.6f}, "
            f"output: ${turn_cost.output_cost:.6f}) | "
            f"Cumulative: ${self.total_breakdown.total_cost:.6f}"
        )

    def add_maintenance_event(self, event: dict[str, object]) -> CostBreakdown:
        usage = event.get("usage")
        usage_dict: dict[str, int] = {}
        if isinstance(usage, dict):
            for key in (
                "prompt_tokens",
                "cached_tokens",
                "cache_creation_input_tokens",
                "cache_creation_5m_input_tokens",
                "cache_creation_1h_input_tokens",
                "candidates_tokens",
                "reasoning_tokens",
                "total_tokens",
                REPORTED_COST_USAGE_KEY,
            ):
                value = usage.get(key)
                if isinstance(value, int):
                    usage_dict[key] = value

        event_model_id = event.get("model_id")
        event_provider = event.get("provider")
        has_event_model = isinstance(event_model_id, str) and bool(event_model_id.strip())
        if has_event_model:
            if not isinstance(event_provider, str) or not event_provider.strip():
                raise ValueError("maintenance event with model_id must include provider")
            maintenance_model_id = event_model_id
        else:
            maintenance_model_id = self.model_id

        if REPORTED_COST_USAGE_KEY in usage_dict:
            maintenance_cost = reported_breakdown(usage_dict)
            self.cost_source = "reported"
        elif not usage_dict:
            maintenance_cost = CostBreakdown()
        else:
            if has_event_model:
                event_pricing = pricing_for_provider_model(event_provider, event_model_id)
                if event_pricing is None:
                    raise CostAccountingError(
                        f"maintenance pricing unavailable for {event_provider}/{event_model_id}"
                    )
            else:
                event_pricing = self._require_pricing("maintenance event")
            maintenance_cost = calculate_cost(usage_dict, event_pricing)
        _accumulate_breakdown(self.maintenance_breakdown, maintenance_cost)
        model_breakdown = self.maintenance_breakdown_by_model.setdefault(
            maintenance_model_id,
            CostBreakdown(),
        )
        _accumulate_breakdown(model_breakdown, maintenance_cost)

        event_record = dict(event)
        if usage_dict:
            event_record["tokens"] = _breakdown_tokens_dict(maintenance_cost)
            event_record["costs_usd"] = _breakdown_costs_dict(maintenance_cost)
        else:
            event_record["tokens"] = None
            event_record["costs_usd"] = None
        self.maintenance_events.append(event_record)
        return maintenance_cost

    def all_in_total_breakdown(self) -> CostBreakdown:
        return CostBreakdown(
            prompt_tokens=self.total_breakdown.prompt_tokens
            + self.maintenance_breakdown.prompt_tokens,
            cached_tokens=self.total_breakdown.cached_tokens
            + self.maintenance_breakdown.cached_tokens,
            uncached_prompt_tokens=(
                self.total_breakdown.uncached_prompt_tokens
                + self.maintenance_breakdown.uncached_prompt_tokens
            ),
            candidates_tokens=(
                self.total_breakdown.candidates_tokens
                + self.maintenance_breakdown.candidates_tokens
            ),
            total_tokens=self.total_breakdown.total_tokens
            + self.maintenance_breakdown.total_tokens,
            input_uncached_cost=(
                self.total_breakdown.input_uncached_cost
                + self.maintenance_breakdown.input_uncached_cost
            ),
            input_cached_cost=(
                self.total_breakdown.input_cached_cost
                + self.maintenance_breakdown.input_cached_cost
            ),
            output_cost=self.total_breakdown.output_cost + self.maintenance_breakdown.output_cost,
            total_cost=self.all_in_total_cost_usd,
        )

    def to_dict(self) -> dict[str, object]:
        all_in_total = self.all_in_total_breakdown()
        return {
            "model_id": self.model_id,
            "cost_source": self.cost_source,
            "total": _breakdown_dict(self.total_breakdown),
            "maintenance_total": _breakdown_dict(self.maintenance_breakdown),
            "maintenance_by_model": {
                model_id: _breakdown_dict(breakdown)
                for model_id, breakdown in sorted(self.maintenance_breakdown_by_model.items())
            },
            "all_in_total": _breakdown_dict(all_in_total),
            "pricing": self.pricing,
            "turns": [
                {
                    **_breakdown_dict(breakdown),
                }
                for breakdown in self.turn_breakdowns
            ],
            "maintenance_events": self.maintenance_events,
        }

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def pricing_for_provider_model(provider: str, model_id: str) -> dict[str, float] | None:
    if not (provider or "").strip():
        return None
    try:
        provider_norm = normalize_provider_name(provider)
    except ValueError:
        return None
    model = str(model_id).strip()
    if provider_norm.value == "anthropic":
        return {
            "claude-opus-4-8": ANTHROPIC_OPUS_PRICING,
            "claude-opus-4-7": ANTHROPIC_OPUS_PRICING,
            "claude-opus-4-6": ANTHROPIC_OPUS_PRICING,
            "claude-sonnet-4-6": ANTHROPIC_SONNET_PRICING,
            "claude-haiku-4-5": ANTHROPIC_HAIKU_PRICING,
        }.get(model)
    if provider_norm.value == "gemini":
        return {"gemini-3.5-flash": GEMINI_3_5_FLASH_PRICING}.get(model)
    if provider_norm is not ProviderName.OPENAI:
        return None
    return {
        "gpt-5.2": OPENAI_GPT_5_2_PRICING,
        "gpt-5.3-codex": OPENAI_GPT_5_3_CODEX_PRICING,
        "gpt-5.4": OPENAI_GPT_5_4_PRICING,
        "gpt-5.5": OPENAI_GPT_5_5_PRICING,
        "gpt-5.5-2026-04-23": OPENAI_GPT_5_5_PRICING,
        "gpt-5.6-sol": OPENAI_GPT_5_6_SOL_PRICING,
        "gpt-5.6-terra": OPENAI_GPT_5_6_TERRA_PRICING,
        "gpt-5.6-luna": OPENAI_GPT_5_6_LUNA_PRICING,
        "gpt-6-astra": OPENAI_GPT_6_ASTRA_PRICING,
    }.get(model)


SELF_REPORTING_COST_PROVIDERS: frozenset[str] = frozenset({"openrouter"})


def provider_reports_cost(provider: str) -> bool:
    try:
        return normalize_provider_name(provider).value in SELF_REPORTING_COST_PROVIDERS
    except ValueError:
        return False


def build_cost_tracker(provider: str, model_id: str) -> CostTracker | None:
    pricing = pricing_for_provider_model(provider, model_id)
    if pricing is not None:
        return CostTracker(model_id=model_id, pricing=pricing)
    if provider_reports_cost(provider):
        return CostTracker(model_id=model_id, pricing=None, cost_source="reported")
    return None


def parse_max_cost_usd(value: object, *, label: str = "max_cost_usd") -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        amount = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {label}: {value!r}") from exc
    if amount <= 0:
        raise ValueError(f"{label} must be > 0")
    return amount


def max_cost_usd_from_env(env: dict[str, str] | None = None) -> float | None:
    values = os.environ if env is None else env
    return parse_max_cost_usd(values.get(MAX_COST_ENV_VAR), label=MAX_COST_ENV_VAR)
