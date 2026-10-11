from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pipeline.contracts import Issue, TerminationReason
from pipeline.policies import DEFAULT_POLICIES, Policies
from pipeline.state import FORWARD_ORDER, TERMINAL, Phase, PipelineState


class StageStatus(str, Enum):
    PASS = "pass"
    REDO = "redo"
    FAIL = "fail"
    INCOMPLETE = "incomplete"


@dataclass(slots=True)
class StageOutcome:
    status: StageStatus
    summary: str = ""
    realism_rating: int | None = None
    rubric: dict[str, int] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)
    retry_in_place: bool = False
    termination_reason: TerminationReason | None = None

    def __post_init__(self) -> None:
        if self.status is StageStatus.INCOMPLETE and self.termination_reason is None:
            self.termination_reason = TerminationReason.MAX_TURNS
        elif self.status is StageStatus.FAIL and self.termination_reason is None:
            self.termination_reason = TerminationReason.HARD_FAILURE


REPAIR_TARGET: dict[Phase, Phase] = {
    Phase.P3_BLUEPRINT_REVIEW: Phase.P1_BLUEPRINT,
    Phase.P5_REVIEW: Phase.P4_BUILD,
}

SELF_REPAIR_PHASES: frozenset[Phase] = frozenset({Phase.P4_BUILD})


@dataclass(slots=True)
class Transition:
    next_phase: Phase
    note: str
    documented_limitation: bool = False
    halt: bool = False


def _forward(phase: Phase) -> Phase:
    idx = FORWARD_ORDER.index(phase)
    if idx + 1 < len(FORWARD_ORDER):
        return FORWARD_ORDER[idx + 1]
    return Phase.DONE


def decide_next(
    state: PipelineState,
    phase: Phase,
    outcome: StageOutcome,
    policies: Policies = DEFAULT_POLICIES,
) -> Transition:
    if phase in TERMINAL:
        return Transition(phase, "terminal")

    if outcome.status is StageStatus.INCOMPLETE:
        return Transition(phase, outcome.summary, halt=True)

    if outcome.status is StageStatus.FAIL:
        return Transition(Phase.FAILED, f"hard failure in {phase.value}: {outcome.summary}")

    if phase in SELF_REPAIR_PHASES and outcome.status is StageStatus.REDO:
        n = state.bump_iteration(phase)
        return Transition(phase, f"{phase.value} compile-repair iteration {n}")

    if phase in REPAIR_TARGET:
        rating = outcome.realism_rating
        has_blocking = any(i.is_blocking() for i in (outcome.issues or []))
        verdict = (
            policies.review_verdict(rating, outcome.rubric, has_blocking_issue=has_blocking)
            if rating is not None
            else ("pass" if outcome.status is StageStatus.PASS else "redo")
        )
        state.last_realism_rating = rating
        if verdict == "pass":
            state.mark_passed(phase)
            return _advance_with_cascade(state, phase, policies)
        if verdict == "redo":
            n = state.bump_iteration(phase)
            if rating is not None and not has_blocking:
                gating_vals = [
                    v
                    for k, v in (outcome.rubric or {}).items()
                    if k in policies.gating_rubric_dims and isinstance(v, int)
                ]
                cur_signal = {
                    "rating": rating,
                    "worst_gating": (min(gating_vals) if gating_vals else None),
                    "blocking": sum(1 for i in (outcome.issues or []) if i.is_blocking()),
                }
                if policies.review_made_progress(state.last_review_signal, cur_signal):
                    state.review_no_progress_streak = 0
                else:
                    state.review_no_progress_streak += 1
                state.last_review_signal = cur_signal
            elif has_blocking:
                state.last_review_signal = {}
                state.review_no_progress_streak = 0
            stalled = state.review_no_progress_streak >= policies.review_no_progress_budget
            if not has_blocking and (policies.budget_exhausted(n) or stalled):
                why = (
                    f"no progress for {state.review_no_progress_streak} rounds"
                    if stalled and not policies.budget_exhausted(n)
                    else f"budget exhausted ({n})"
                )
                state.mark_passed(phase)
                return _advance_with_cascade(
                    state,
                    phase,
                    policies,
                    note=f"{phase.value} {why}; documented limitation",
                    documented=True,
                )
            if outcome.retry_in_place:
                return Transition(
                    phase,
                    f"{phase.value} re-render & re-review (no actionable issues) iter {n}",
                )
            return Transition(
                REPAIR_TARGET[phase],
                f"{phase.value} redo (rating={rating}) → repair {REPAIR_TARGET[phase].value} "
                f"iter {n}",
            )
        state.mark_passed(phase)
        return _advance_with_cascade(
            state,
            phase,
            policies,
            note=f"{phase.value} acceptable (rating={rating}); proceeding",
            documented=bool(outcome.issues),
        )

    if outcome.status is StageStatus.PASS:
        state.mark_passed(phase)
        return _advance_with_cascade(state, phase, policies)

    n = state.bump_iteration(phase)
    if policies.budget_exhausted(n):
        return Transition(Phase.FAILED, f"{phase.value} stuck after {n} attempts")
    return Transition(phase, f"{phase.value} retry {n}")


def _advance_with_cascade(
    state: PipelineState,
    phase: Phase,
    policies: Policies,
    *,
    note: str = "",
    documented: bool = False,
) -> Transition:
    nxt = _forward(phase)
    return Transition(nxt, note or f"{phase.value} → {nxt.value}", documented_limitation=documented)
