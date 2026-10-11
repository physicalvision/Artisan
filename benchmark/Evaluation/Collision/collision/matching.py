from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from collision.kinematics import _origin_to_matrix
from collision.model import EvalJoint, EvalModel


@dataclass(frozen=True)
class SignedTransform:
    sign: int | None
    offset: float | None
    uncertain: bool
    reason: str | None = None


@dataclass(frozen=True)
class JointMatch:
    gt_key: str
    gen_key: str
    cost: float
    axis_error_rad: float
    pivot_error_ratio: float
    sign: int | None
    offset: float | None
    absolute_mapping_uncertain: bool


@dataclass(frozen=True)
class MatchTable:
    records: tuple[JointMatch, ...]
    unmatched_gt: tuple[str, ...]
    unmatched_gen: tuple[str, ...]
    frame_rotation: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class _JointFeatures:
    pivot: np.ndarray
    axis: np.ndarray
    centroid: np.ndarray
    extents: np.ndarray
    displacement_field: np.ndarray


def extract_signed_transform(
    *,
    gt_axis: np.ndarray,
    gen_axis: np.ndarray,
    frame_rotation: np.ndarray,
    gt_displacements: np.ndarray,
    gen_displacements: np.ndarray,
    axis_threshold: float = 0.2,
    displacement_threshold: float = 1e-09,
) -> SignedTransform:
    rotation = np.asarray(frame_rotation, dtype=float)
    gt_axis_aligned = rotation @ np.asarray(gt_axis, dtype=float)
    gt_axis_aligned /= max(float(np.linalg.norm(gt_axis_aligned)), 1e-15)
    gen_axis = np.asarray(gen_axis, dtype=float)
    gen_axis /= max(float(np.linalg.norm(gen_axis)), 1e-15)
    axis_dot = float(np.dot(gen_axis, gt_axis_aligned))
    axis_sign = None if abs(axis_dot) < axis_threshold else 1 if axis_dot >= 0.0 else -1
    gt_field = np.asarray(gt_displacements, dtype=float) @ rotation.T
    gen_field = np.asarray(gen_displacements, dtype=float)
    displacement_sign = None
    if gt_field.shape == gen_field.shape and gt_field.size:
        denominator = float(np.linalg.norm(gt_field) * np.linalg.norm(gen_field))
        if denominator > displacement_threshold:
            field_dot = float(np.sum(gt_field * gen_field) / denominator)
            if abs(field_dot) >= axis_threshold:
                displacement_sign = 1 if field_dot >= 0.0 else -1
    if axis_sign is None and displacement_sign is None:
        return SignedTransform(None, None, True, "axis and displacement fields are degenerate")
    if axis_sign is None or displacement_sign is None:
        return SignedTransform(None, None, True, "one sign source is degenerate")
    if axis_sign != displacement_sign:
        return SignedTransform(None, None, True, "axis and displacement signs disagree")
    return SignedTransform(axis_sign, 0.0, False)


def _rotation_candidates() -> tuple[np.ndarray, ...]:
    rotations = []
    for permutation in itertools.permutations(range(3)):
        base = np.eye(3)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            candidate = base * np.asarray(signs)[None, :]
            if np.linalg.det(candidate) > 0.5:
                rotations.append(candidate)
    return tuple(rotations)


def _subtree_links(model: EvalModel, joint: EvalJoint) -> set[str]:
    children: dict[str, list[str]] = {}
    for candidate in model.joints:
        children.setdefault(candidate.parent, []).append(candidate.child)
    result: set[str] = set()
    pending = [joint.child]
    while pending:
        link = pending.pop()
        if link in result:
            continue
        result.add(link)
        pending.extend(children.get(link, []))
    return result


def _joint_features(model: EvalModel, joint: EvalJoint) -> _JointFeatures:
    closed = model.closed_pose()
    rest_transforms = model.forward_kinematics(closed)
    frame = rest_transforms[joint.parent] @ _origin_to_matrix(joint.origin)
    axis = frame[:3, :3] @ np.asarray(joint.axis, dtype=float)
    axis /= max(float(np.linalg.norm(axis)), 1e-15)
    subtree = _subtree_links(model, joint)
    points = []
    for piece in model.pieces:
        if piece.link not in subtree:
            continue
        vertices = np.asarray(piece.mesh.vertices, dtype=float)
        if len(vertices) > 64:
            vertices = vertices[np.linspace(0, len(vertices) - 1, 64).astype(int)]
        points.append(
            vertices @ rest_transforms[piece.link][:3, :3].T + rest_transforms[piece.link][:3, 3]
        )
    world_points = (
        np.concatenate(points)
        if points
        else np.asarray(rest_transforms[joint.child][:3, 3], dtype=float)[None, :]
    )
    radial = world_points - frame[:3, 3]
    radial -= np.outer(radial @ axis, axis)
    ranked = np.argsort(-np.linalg.norm(radial, axis=1))
    chosen = world_points[ranked[: min(8, len(ranked))]]
    if len(chosen) < 8:
        chosen = np.concatenate([chosen, np.repeat(chosen[-1][None, :], 8 - len(chosen), axis=0)])
    lower, upper = joint.evaluation_range
    delta_magnitude = 0.05 * max(upper - lower, 1e-09)
    positive_room = upper - joint.closed
    negative_room = joint.closed - lower
    delta = (
        min(delta_magnitude, positive_room)
        if positive_room > 1e-12
        else -min(delta_magnitude, negative_room)
    )
    probe = dict(closed)
    probe[joint.key] = joint.closed + delta
    moved_child = model.forward_kinematics(probe)[joint.child]
    child_delta = moved_child @ np.linalg.inv(rest_transforms[joint.child])
    moved_points = chosen @ child_delta[:3, :3].T + child_delta[:3, 3]
    displacement_field = (moved_points - chosen) / delta
    return _JointFeatures(
        pivot=frame[:3, 3],
        axis=axis,
        centroid=world_points.mean(axis=0),
        extents=np.maximum(world_points.max(axis=0) - world_points.min(axis=0), 1e-12),
        displacement_field=displacement_field,
    )


def _types_compatible(left: EvalJoint, right: EvalJoint) -> bool:
    rotations = {"revolute", "continuous"}
    return left.joint_type == right.joint_type or (
        left.joint_type in rotations and right.joint_type in rotations
    )


def _cost(
    gt_joint: EvalJoint,
    gt: _JointFeatures,
    gen: _JointFeatures,
    rotation: np.ndarray,
    scale: float,
) -> tuple[float, float, float]:
    axis_dot = abs(float(np.dot(gen.axis, rotation @ gt.axis)))
    axis_error = math.acos(min(1.0, max(-1.0, axis_dot)))
    pivot_delta = gen.pivot - rotation @ gt.pivot
    if gt_joint.joint_type in {"revolute", "continuous"}:
        pivot_delta -= np.dot(pivot_delta, gen.axis) * gen.axis
    pivot_error = float(np.linalg.norm(pivot_delta) / scale)
    centroid_error = float(np.linalg.norm(gen.centroid - rotation @ gt.centroid) / scale)
    aligned_extents = np.abs(rotation) @ gt.extents
    extent_error = float(np.mean(np.abs(np.log(np.maximum(gen.extents, 1e-12) / aligned_extents))))
    gt_motion = (gt.displacement_field @ rotation.T).reshape(-1)
    gen_motion = gen.displacement_field.reshape(-1)
    denominator = float(np.linalg.norm(gt_motion) * np.linalg.norm(gen_motion))
    motion_error = (
        1.0
        if denominator <= 1e-12
        else 1.0 - abs(float(np.dot(gt_motion, gen_motion) / denominator))
    )
    total = (
        0.3 * axis_error / math.pi
        + 0.25 * pivot_error
        + 0.2 * centroid_error
        + 0.15 * min(extent_error, 4.0)
        + 0.1 * motion_error
    )
    return (total, axis_error, pivot_error)


def match_models(gt: EvalModel, gen: EvalModel, *, max_cost: float = 1.25) -> MatchTable:
    gt_joints, gen_joints = (gt.movable_joints, gen.movable_joints)
    identity = tuple((tuple((float(value) for value in row)) for row in np.eye(3)))
    if not gt_joints or not gen_joints:
        return MatchTable(
            records=(),
            unmatched_gt=tuple((joint.key for joint in gt_joints)),
            unmatched_gen=tuple((joint.key for joint in gen_joints)),
            frame_rotation=identity,
        )
    gt_features = [_joint_features(gt, joint) for joint in gt_joints]
    gen_features = [_joint_features(gen, joint) for joint in gen_joints]
    feature_extents = [feature.extents for feature in gt_features + gen_features]
    feature_centers = [feature.centroid for feature in gt_features + gen_features]
    scale = max(
        float(np.linalg.norm(np.max(feature_centers, axis=0) - np.min(feature_centers, axis=0))),
        max((float(np.linalg.norm(extents)) for extents in feature_extents)),
        1e-09,
    )
    best: tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    for rotation in _rotation_candidates():
        costs = np.full((len(gt_joints), len(gen_joints)), 1000000.0, dtype=float)
        for row, (gt_joint, gt_feature) in enumerate(zip(gt_joints, gt_features, strict=True)):
            for column, (gen_joint, gen_feature) in enumerate(
                zip(gen_joints, gen_features, strict=True)
            ):
                if _types_compatible(gt_joint, gen_joint):
                    costs[row, column] = _cost(gt_joint, gt_feature, gen_feature, rotation, scale)[
                        0
                    ]
        rows, columns = linear_sum_assignment(costs)
        total = float(costs[rows, columns].sum())
        if best is None or total < best[0]:
            best = (total, rotation, rows, columns, costs)
    assert best is not None
    _, rotation, rows, columns, costs = best
    records = []
    matched_gt: set[int] = set()
    matched_gen: set[int] = set()
    for row, column in zip(rows, columns, strict=True):
        gt_joint, gen_joint = (gt_joints[row], gen_joints[column])
        cost = float(costs[row, column])
        if cost > max_cost or not _types_compatible(gt_joint, gen_joint):
            continue
        _, axis_error, pivot_error = _cost(
            gt_joint, gt_features[row], gen_features[column], rotation, scale
        )
        signed = extract_signed_transform(
            gt_axis=gt_features[row].axis,
            gen_axis=gen_features[column].axis,
            frame_rotation=rotation,
            gt_displacements=gt_features[row].displacement_field,
            gen_displacements=gen_features[column].displacement_field,
        )
        offset = (
            gen_joint.closed - signed.sign * gt_joint.closed if signed.sign is not None else None
        )
        records.append(
            JointMatch(
                gt_key=gt_joint.key,
                gen_key=gen_joint.key,
                cost=cost,
                axis_error_rad=axis_error,
                pivot_error_ratio=pivot_error,
                sign=signed.sign,
                offset=offset,
                absolute_mapping_uncertain=signed.uncertain,
            )
        )
        matched_gt.add(row)
        matched_gen.add(column)
    records.sort(key=lambda record: (record.gt_key, record.gen_key))
    return MatchTable(
        records=tuple(records),
        unmatched_gt=tuple(
            (joint.key for index, joint in enumerate(gt_joints) if index not in matched_gt)
        ),
        unmatched_gen=tuple(
            (joint.key for index, joint in enumerate(gen_joints) if index not in matched_gen)
        ),
        frame_rotation=tuple((tuple((float(value) for value in row)) for row in rotation)),
    )
