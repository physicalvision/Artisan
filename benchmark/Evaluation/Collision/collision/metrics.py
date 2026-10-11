from __future__ import annotations

from enum import Enum
from typing import Iterable

from collision.backend import NarrowPhaseKind, NarrowPhaseResult


class EvidenceState(str, Enum):
    CLEAN = "clean"
    VIOLATING = "violating"
    INCONCLUSIVE = "inconclusive"
    EXCUSED = "excused"


def classify_collision(result: NarrowPhaseResult, *, epsilon: float) -> EvidenceState:
    if result.kind is NarrowPhaseKind.CONTAINED:
        return EvidenceState.VIOLATING
    if result.kind is NarrowPhaseKind.CONTAINMENT_UNCERTAIN:
        return EvidenceState.INCONCLUSIVE
    if result.kind is NarrowPhaseKind.CROSSING:
        if result.d_local is None:
            return EvidenceState.INCONCLUSIVE
        return EvidenceState.VIOLATING if result.d_local > epsilon else EvidenceState.CLEAN
    return EvidenceState.CLEAN


def aggregate_tuple(states: Iterable[EvidenceState]) -> EvidenceState:
    values = list(states)
    unmasked = [value for value in values if value is not EvidenceState.EXCUSED]
    if not unmasked:
        return EvidenceState.EXCUSED
    if EvidenceState.VIOLATING in unmasked:
        return EvidenceState.VIOLATING
    if EvidenceState.INCONCLUSIVE in unmasked:
        return EvidenceState.INCONCLUSIVE
    return EvidenceState.CLEAN


def aggregate_cell(states: Iterable[EvidenceState]) -> EvidenceState:
    unmasked = [value for value in states if value is not EvidenceState.EXCUSED]
    if not unmasked:
        return EvidenceState.INCONCLUSIVE
    if EvidenceState.VIOLATING in unmasked:
        return EvidenceState.VIOLATING
    if all((value is EvidenceState.CLEAN for value in unmasked)):
        return EvidenceState.CLEAN
    return EvidenceState.INCONCLUSIVE
