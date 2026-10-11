from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from collision.backend import NarrowPhaseResult
from collision.metrics import EvidenceState, classify_collision
from collision.model import EvalJoint, EvalModel


class MaskDecision(str, Enum):
    EXCUSED = "excused"
    NOT_EXCUSED = "not_excused"
    UNCERTAIN = "mask_uncertain"


@dataclass(frozen=True)
class RolePartition:
    body_roles: dict[str, str]

    def role_pair(self, body_pair: tuple[str, str]) -> tuple[str, str]:
        return canonical_pair(self.body_roles[body_pair[0]], self.body_roles[body_pair[1]])


def canonical_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def body_pair_key(left: str, right: str) -> str:
    return "|".join(canonical_pair(left, right))


def parse_body_pair(key: str) -> tuple[str, str]:
    left, right = key.split("|", 1)
    return canonical_pair(left, right)


def _children(model: EvalModel) -> dict[str, list[EvalJoint]]:
    result: dict[str, list[EvalJoint]] = {}
    for joint in model.joints:
        result.setdefault(joint.parent, []).append(joint)
    return result


def _subtree_links(model: EvalModel, joint: EvalJoint) -> set[str]:
    children = _children(model)
    result: set[str] = set()
    pending = [joint.child]
    while pending:
        link = pending.pop()
        if link in result:
            continue
        result.add(link)
        pending.extend((child.child for child in children.get(link, [])))
    return result


def _link_depths(model: EvalModel) -> dict[str, int]:
    children = _children(model)
    depths: dict[str, int] = {}
    pending = [(root, 0) for root in model.roots]
    while pending:
        link, depth = pending.pop()
        depths[link] = depth
        pending.extend(((joint.child, depth + 1) for joint in children.get(link, [])))
    return depths


def build_role_partition(model: EvalModel, matched_joint_roles: dict[str, str]) -> RolePartition:
    joint_by_key = model.joint_by_key
    depths = _link_depths(model)
    matched_subtrees = {
        key: _subtree_links(model, joint_by_key[key]) for key in matched_joint_roles
    }
    bodies = sorted({piece.body for piece in model.pieces})
    links_by_body = {
        body: {link for link, candidate in model.body_for_link.items() if candidate == body}
        for body in bodies
    }
    body_roles = {}
    for body in bodies:
        candidates = [key for key, links in matched_subtrees.items() if links_by_body[body] & links]
        if candidates:
            innermost = max(candidates, key=lambda key: depths[joint_by_key[key].child])
            body_roles[body] = matched_joint_roles[innermost]
        else:
            body_roles[body] = "base"
    return RolePartition(body_roles=body_roles)


def adjacent_body_pairs(model: EvalModel) -> set[tuple[str, str]]:
    result = set()
    for joint in model.joints:
        if joint.joint_type not in {"revolute", "continuous", "prismatic"}:
            continue
        left = model.body_for_link[joint.parent]
        right = model.body_for_link[joint.child]
        if left != right:
            result.add(canonical_pair(left, right))
    return result


def apply_collision_mask(generated: EvidenceState, decision: MaskDecision) -> EvidenceState:
    if decision is MaskDecision.EXCUSED:
        return EvidenceState.EXCUSED
    if decision is MaskDecision.UNCERTAIN:
        return (
            EvidenceState.CLEAN if generated is EvidenceState.CLEAN else EvidenceState.INCONCLUSIVE
        )
    return generated


def _role_pair_states(
    *,
    role_pair: tuple[str, str],
    results: dict[str, NarrowPhaseResult],
    partition: RolePartition,
    adjacent: set[tuple[str, str]],
    epsilon: float,
) -> list[EvidenceState]:
    target = canonical_pair(*role_pair)
    states = []
    for key, result in results.items():
        body_pair = parse_body_pair(key)
        if body_pair in adjacent:
            continue
        if body_pair[0] not in partition.body_roles or body_pair[1] not in partition.body_roles:
            continue
        if partition.role_pair(body_pair) == target:
            states.append(classify_collision(result, epsilon=epsilon))
    return states


def project_gt_collision_mask(
    *,
    role_pair: tuple[str, str],
    gt_results: dict[str, NarrowPhaseResult],
    gt_partition: RolePartition,
    gt_model: EvalModel,
    epsilon: float,
    gt_reference: dict[str, NarrowPhaseResult] | None = None,
) -> MaskDecision:
    adjacent = adjacent_body_pairs(gt_model)
    states = _role_pair_states(
        role_pair=role_pair,
        results=gt_results,
        partition=gt_partition,
        adjacent=adjacent,
        epsilon=epsilon,
    )
    if not states:
        return MaskDecision.NOT_EXCUSED
    if EvidenceState.VIOLATING not in states:
        return (
            MaskDecision.UNCERTAIN
            if EvidenceState.INCONCLUSIVE in states
            else MaskDecision.NOT_EXCUSED
        )
    if gt_reference is None:
        return MaskDecision.UNCERTAIN
    reference_states = _role_pair_states(
        role_pair=role_pair,
        results=gt_reference,
        partition=gt_partition,
        adjacent=adjacent,
        epsilon=epsilon,
    )
    if EvidenceState.VIOLATING in reference_states:
        return MaskDecision.NOT_EXCUSED
    if EvidenceState.INCONCLUSIVE in reference_states:
        return MaskDecision.UNCERTAIN
    return MaskDecision.EXCUSED
