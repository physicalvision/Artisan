from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SoftCompactionBand:

    min_pressure_ratio: float
    min_failure_streak: int
    min_compactable_items: int
    name: str


SOFT_COMPACTION_BANDS: tuple[SoftCompactionBand, ...] = (
    SoftCompactionBand(
        min_pressure_ratio=0.85,
        min_failure_streak=3,
        min_compactable_items=2,
        name="high_pressure",
    ),
    SoftCompactionBand(
        min_pressure_ratio=0.70,
        min_failure_streak=4,
        min_compactable_items=2,
        name="medium_pressure",
    ),
    SoftCompactionBand(
        min_pressure_ratio=0.55,
        min_failure_streak=5,
        min_compactable_items=3,
        name="early_pressure",
    ),
)

HARD_PRESSURE_TRIGGER_RATIO = 0.90
HIGH_CACHE_RATIO = 0.60
SOFT_COMPACTION_COOLDOWN_TURNS = 2
SOFT_COMPACTION_GROWTH_FACTOR = 1.20
COST_COMPACTION_MIN_COMPACTABLE_ITEMS = 2


@dataclass(frozen=True, slots=True)
class CompactionDecision:
    trigger: str | None
    reason: str
    pressure_ratio: float | None = None
    cache_ratio: float | None = None
    soft_failure_threshold: int | None = None
    min_compactable_items: int | None = None
    hard_trigger_tokens: int | None = None
    cost_trigger_tokens: int | None = None


def hard_pressure_trigger_tokens(hard_threshold: int | None) -> int | None:
    if not isinstance(hard_threshold, int) or hard_threshold <= 0:
        return None
    return max(1, math.ceil(hard_threshold * HARD_PRESSURE_TRIGGER_RATIO))


def prompt_cache_ratio(*, prompt_tokens: int | None, cached_tokens: int | None) -> float | None:
    if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
        return None
    if not isinstance(cached_tokens, int) or cached_tokens <= 0:
        return 0.0
    return min(1.0, max(0.0, cached_tokens / prompt_tokens))


def pressure_ratio(*, prompt_tokens: int | None, hard_threshold: int | None) -> float | None:
    if (
        not isinstance(prompt_tokens, int)
        or prompt_tokens < 0
        or not isinstance(hard_threshold, int)
        or hard_threshold <= 0
    ):
        return None
    return prompt_tokens / hard_threshold


def soft_compaction_band_for_pressure(value: float | None) -> SoftCompactionBand | None:
    if value is None:
        return None
    for band in SOFT_COMPACTION_BANDS:
        if value >= band.min_pressure_ratio:
            return band
    return None


def _soft_compaction_in_cooldown(
    *,
    prompt_tokens: int | None,
    turn_number: int,
    last_soft_compaction_turn_number: int | None,
    last_soft_compaction_prompt_tokens: int | None,
) -> bool:
    if last_soft_compaction_turn_number is None:
        return False
    turns_since_soft_compaction = turn_number - last_soft_compaction_turn_number
    if turns_since_soft_compaction >= SOFT_COMPACTION_COOLDOWN_TURNS:
        return False
    growth_floor: int | None = None
    if isinstance(last_soft_compaction_prompt_tokens, int):
        growth_floor = math.ceil(last_soft_compaction_prompt_tokens * SOFT_COMPACTION_GROWTH_FACTOR)
    return (
        growth_floor is None or not isinstance(prompt_tokens, int) or prompt_tokens < growth_floor
    )


def decide_compaction(
    *,
    prompt_tokens: int | None,
    cached_tokens: int | None,
    hard_threshold: int | None,
    consecutive_compile_failure_count: int,
    last_compile_failure_sig: str | None,
    last_soft_compaction_failure_sig: str | None,
    compactable_item_count: int,
    turn_number: int,
    last_soft_compaction_turn_number: int | None,
    last_soft_compaction_prompt_tokens: int | None,
    cost_trigger_tokens: int | None = None,
) -> CompactionDecision:
    current_pressure_ratio = pressure_ratio(
        prompt_tokens=prompt_tokens,
        hard_threshold=hard_threshold,
    )
    current_cache_ratio = prompt_cache_ratio(
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
    )
    hard_trigger_tokens = hard_pressure_trigger_tokens(hard_threshold)

    if (
        hard_trigger_tokens is not None
        and isinstance(prompt_tokens, int)
        and prompt_tokens >= hard_trigger_tokens
    ):
        return CompactionDecision(
            trigger="hard_pressure",
            reason="hard_pressure_band",
            pressure_ratio=current_pressure_ratio,
            cache_ratio=current_cache_ratio,
            hard_trigger_tokens=hard_trigger_tokens,
            cost_trigger_tokens=cost_trigger_tokens,
        )

    if (
        isinstance(cost_trigger_tokens, int)
        and cost_trigger_tokens > 0
        and isinstance(prompt_tokens, int)
        and prompt_tokens >= cost_trigger_tokens
    ):
        if compactable_item_count < COST_COMPACTION_MIN_COMPACTABLE_ITEMS:
            return CompactionDecision(
                trigger=None,
                reason="insufficient_compactable_history",
                pressure_ratio=current_pressure_ratio,
                cache_ratio=current_cache_ratio,
                min_compactable_items=COST_COMPACTION_MIN_COMPACTABLE_ITEMS,
                hard_trigger_tokens=hard_trigger_tokens,
                cost_trigger_tokens=cost_trigger_tokens,
            )
        if _soft_compaction_in_cooldown(
            prompt_tokens=prompt_tokens,
            turn_number=turn_number,
            last_soft_compaction_turn_number=last_soft_compaction_turn_number,
            last_soft_compaction_prompt_tokens=last_soft_compaction_prompt_tokens,
        ):
            return CompactionDecision(
                trigger=None,
                reason="soft_compaction_cooldown",
                pressure_ratio=current_pressure_ratio,
                cache_ratio=current_cache_ratio,
                min_compactable_items=COST_COMPACTION_MIN_COMPACTABLE_ITEMS,
                hard_trigger_tokens=hard_trigger_tokens,
                cost_trigger_tokens=cost_trigger_tokens,
            )
        return CompactionDecision(
            trigger="cost_pressure",
            reason="cost_trigger",
            pressure_ratio=current_pressure_ratio,
            cache_ratio=current_cache_ratio,
            min_compactable_items=COST_COMPACTION_MIN_COMPACTABLE_ITEMS,
            hard_trigger_tokens=hard_trigger_tokens,
            cost_trigger_tokens=cost_trigger_tokens,
        )

    if consecutive_compile_failure_count <= 0 or not last_compile_failure_sig:
        return CompactionDecision(
            trigger=None,
            reason="no_compile_plateau",
            pressure_ratio=current_pressure_ratio,
            cache_ratio=current_cache_ratio,
            hard_trigger_tokens=hard_trigger_tokens,
            cost_trigger_tokens=cost_trigger_tokens,
        )

    if last_compile_failure_sig == last_soft_compaction_failure_sig:
        return CompactionDecision(
            trigger=None,
            reason="signature_already_soft_compacted",
            pressure_ratio=current_pressure_ratio,
            cache_ratio=current_cache_ratio,
            hard_trigger_tokens=hard_trigger_tokens,
            cost_trigger_tokens=cost_trigger_tokens,
        )

    band = soft_compaction_band_for_pressure(current_pressure_ratio)
    if band is None:
        return CompactionDecision(
            trigger=None,
            reason="soft_pressure_too_low",
            pressure_ratio=current_pressure_ratio,
            cache_ratio=current_cache_ratio,
            hard_trigger_tokens=hard_trigger_tokens,
            cost_trigger_tokens=cost_trigger_tokens,
        )

    required_failure_streak = band.min_failure_streak
    if isinstance(current_cache_ratio, float) and current_cache_ratio >= HIGH_CACHE_RATIO:
        required_failure_streak += 1

    if consecutive_compile_failure_count < required_failure_streak:
        return CompactionDecision(
            trigger=None,
            reason="compile_plateau_below_threshold",
            pressure_ratio=current_pressure_ratio,
            cache_ratio=current_cache_ratio,
            soft_failure_threshold=required_failure_streak,
            min_compactable_items=band.min_compactable_items,
            hard_trigger_tokens=hard_trigger_tokens,
            cost_trigger_tokens=cost_trigger_tokens,
        )

    if compactable_item_count < band.min_compactable_items:
        return CompactionDecision(
            trigger=None,
            reason="insufficient_compactable_history",
            pressure_ratio=current_pressure_ratio,
            cache_ratio=current_cache_ratio,
            soft_failure_threshold=required_failure_streak,
            min_compactable_items=band.min_compactable_items,
            hard_trigger_tokens=hard_trigger_tokens,
            cost_trigger_tokens=cost_trigger_tokens,
        )

    if _soft_compaction_in_cooldown(
        prompt_tokens=prompt_tokens,
        turn_number=turn_number,
        last_soft_compaction_turn_number=last_soft_compaction_turn_number,
        last_soft_compaction_prompt_tokens=last_soft_compaction_prompt_tokens,
    ):
        return CompactionDecision(
            trigger=None,
            reason="soft_compaction_cooldown",
            pressure_ratio=current_pressure_ratio,
            cache_ratio=current_cache_ratio,
            soft_failure_threshold=required_failure_streak,
            min_compactable_items=band.min_compactable_items,
            hard_trigger_tokens=hard_trigger_tokens,
            cost_trigger_tokens=cost_trigger_tokens,
        )

    return CompactionDecision(
        trigger="compile_plateau",
        reason=band.name,
        pressure_ratio=current_pressure_ratio,
        cache_ratio=current_cache_ratio,
        soft_failure_threshold=required_failure_streak,
        min_compactable_items=band.min_compactable_items,
        hard_trigger_tokens=hard_trigger_tokens,
        cost_trigger_tokens=cost_trigger_tokens,
    )
