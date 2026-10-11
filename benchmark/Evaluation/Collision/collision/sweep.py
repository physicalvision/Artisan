from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from collision.backend import (
    FCLBackend,
    NarrowPhaseKind,
    NarrowPhaseResult,
    transformed_bounds,
)
from collision.metrics import EvidenceState, classify_collision
from collision.grid import MultiGridCell, l2_cells, lmulti_cells, normalized_cells
from collision.model import EvalJoint, EvalModel, GeometryPiece
from collision.policy import BenchPolicy


@dataclass(frozen=True)
class CellResult:
    index: int
    t: float | None
    t_by_joint: dict[str, float]
    q_by_joint: dict[str, float]
    body_pairs: dict[str, NarrowPhaseResult]


@dataclass(frozen=True)
class JointSweep:
    joint_key: str
    swept_joint_keys: tuple[str, ...]
    dynamic_body_pairs: tuple[str, ...]
    cells: tuple[CellResult, ...]


def _aabb_gap(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.maximum(np.maximum(left[0] - right[1], right[0] - left[1]), 0.0)
    return float(np.linalg.norm(delta))


def _rotation_angle(rotation: np.ndarray) -> float:
    cosine = min(1.0, max(-1.0, (float(np.trace(rotation)) - 1.0) / 2.0))
    return math.acos(cosine)


class SweepEngine:
    def __init__(self, model: EvalModel, backend: FCLBackend, policy: BenchPolicy) -> None:
        self.model = model
        self.backend = backend
        self.policy = policy
        self._body_pieces: dict[str, list[GeometryPiece]] = {}
        for piece in model.pieces:
            self._body_pieces.setdefault(piece.body, []).append(piece)
        self._body_representative = {
            body: min(
                (link for link, candidate in model.body_for_link.items() if candidate == body)
            )
            for body in self._body_pieces
        }
        self._body_pairs = tuple(combinations(sorted(self._body_pieces), 2))
        self._scene_diagonal = self._compute_scene_diagonal()
        self._tau_float = policy.effective_tau_float(self._scene_diagonal)
        self._body_radius = self._compute_body_radii()
        self._rest_cache: dict[str, NarrowPhaseResult] | None = None
        self._pair_result_cache: dict[tuple[str, str, bytes, bytes], NarrowPhaseResult] = {}

    def _compute_scene_diagonal(self) -> float:
        transforms = self.model.forward_kinematics(self.model.closed_pose())
        bounds = []
        for piece in self.model.pieces:
            bounds.append(transformed_bounds(piece.mesh, transforms[piece.link]))
        if not bounds:
            return 1.0
        lower = np.min([bound[0] for bound in bounds], axis=0)
        upper = np.max([bound[1] for bound in bounds], axis=0)
        return max(float(np.linalg.norm(upper - lower)), 1e-09)

    @property
    def scene_diagonal(self) -> float:
        return self._scene_diagonal

    def _compute_body_radii(self) -> dict[str, float]:
        transforms = self.model.forward_kinematics(self.model.closed_pose())
        radii = {}
        for body, pieces in self._body_pieces.items():
            representative = transforms[self._body_representative[body]]
            inverse = np.linalg.inv(representative)
            points = []
            for piece in pieces:
                link_to_body = inverse @ transforms[piece.link]
                local_bounds = np.asarray(piece.mesh.bounds, dtype=float)
                corners = np.asarray(
                    [
                        [x, y, z]
                        for x in (local_bounds[0, 0], local_bounds[1, 0])
                        for y in (local_bounds[0, 1], local_bounds[1, 1])
                        for z in (local_bounds[0, 2], local_bounds[1, 2])
                    ]
                )
                points.append(corners @ link_to_body[:3, :3].T + link_to_body[:3, 3])
            radii[body] = max(float(np.linalg.norm(np.concatenate(points), axis=1).max()), 1e-09)
        return radii

    def _body_transform(self, transforms: dict[str, np.ndarray], body: str) -> np.ndarray:
        return transforms[self._body_representative[body]]

    def _evaluate_body_pair(
        self, left_body: str, right_body: str, transforms: dict[str, np.ndarray]
    ) -> NarrowPhaseResult:
        left_transform = np.ascontiguousarray(
            self._body_transform(transforms, left_body), dtype=float
        )
        right_transform = np.ascontiguousarray(
            self._body_transform(transforms, right_body), dtype=float
        )
        key = (
            left_body,
            right_body,
            left_transform.tobytes(),
            right_transform.tobytes(),
        )
        cached = self._pair_result_cache.get(key)
        if cached is not None:
            return cached
        result = self._evaluate_body_pair_uncached(left_body, right_body, transforms)
        self._pair_result_cache[key] = result
        return result

    def _evaluate_body_pair_uncached(
        self, left_body: str, right_body: str, transforms: dict[str, np.ndarray]
    ) -> NarrowPhaseResult:
        outcomes: list[NarrowPhaseResult] = []
        for left_piece in self._body_pieces[left_body]:
            left_transform = transforms[left_piece.link]
            left_bounds = transformed_bounds(left_piece.mesh, left_transform)
            for right_piece in self._body_pieces[right_body]:
                right_transform = transforms[right_piece.link]
                right_bounds = transformed_bounds(right_piece.mesh, right_transform)
                gap = _aabb_gap(left_bounds, right_bounds)
                if gap > self._tau_float:
                    outcomes.append(
                        NarrowPhaseResult(
                            kind=NarrowPhaseKind.SEPARATED,
                            min_distance=gap,
                            support_gap=gap,
                            reason="AABB lower bound exceeds float threshold",
                        )
                    )
                    continue
                outcomes.append(
                    self.backend.classify_meshes(
                        left_piece.mesh,
                        left_transform,
                        right_piece.mesh,
                        right_transform,
                        scene_diagonal=self._scene_diagonal,
                    )
                )
        epsilon = self.policy.effective_epsilon(self._scene_diagonal)
        violating = [value for value in outcomes if classify_collision(value, epsilon=epsilon) is EvidenceState.VIOLATING]
        if violating and not any(value.kind is NarrowPhaseKind.CROSSING for value in violating):
            return NarrowPhaseResult(kind=NarrowPhaseKind.CONTAINED, support_gap=0.0)
        crossing = [value for value in outcomes if value.kind is NarrowPhaseKind.CROSSING]
        if crossing:
            return NarrowPhaseResult(
                kind=NarrowPhaseKind.CROSSING,
                d_local=max((value.d_local or 0.0 for value in crossing)),
                d_sep=max((value.d_sep or 0.0 for value in crossing)),
                capped=any((value.capped for value in crossing)),
                support_gap=0.0,
                contact_count=sum((value.contact_count for value in crossing)),
                aabb_overlap_volume=max((value.aabb_overlap_volume or 0.0 for value in crossing)),
            )
        if any((value.kind is NarrowPhaseKind.CONTAINED for value in outcomes)):
            return NarrowPhaseResult(kind=NarrowPhaseKind.CONTAINED, support_gap=0.0)
        if any((value.kind is NarrowPhaseKind.CONTAINMENT_UNCERTAIN for value in outcomes)):
            return NarrowPhaseResult(
                kind=NarrowPhaseKind.CONTAINMENT_UNCERTAIN,
                reason="at least one collider pair has uncertain containment",
            )
        gap = min(
            (value.support_gap for value in outcomes if value.support_gap is not None),
            default=math.inf,
        )
        return NarrowPhaseResult(kind=NarrowPhaseKind.SEPARATED, min_distance=gap, support_gap=gap)

    def _evaluate_pairs(
        self, transforms: dict[str, np.ndarray], body_pairs: tuple[tuple[str, str], ...]
    ) -> dict[str, NarrowPhaseResult]:
        self.backend.begin_pose()
        return {
            f"{left}|{right}": self._evaluate_body_pair(left, right, transforms)
            for left, right in body_pairs
        }

    def rest_table(self) -> dict[str, NarrowPhaseResult]:
        if self._rest_cache is None:
            transforms = self.model.forward_kinematics(self.model.closed_pose())
            self._rest_cache = self._evaluate_pairs(transforms, self._body_pairs)
        return dict(self._rest_cache)

    def _dynamic_pairs(
        self, transforms_by_cell: tuple[dict[str, np.ndarray], ...]
    ) -> tuple[tuple[str, str], ...]:
        if len(transforms_by_cell) < 2:
            raise ValueError("dynamic-pair detection needs a baseline plus at least one cell")
        dynamic = []
        for left, right in self._body_pairs:
            relative = [
                np.linalg.inv(self._body_transform(transforms, left))
                @ self._body_transform(transforms, right)
                for transforms in transforms_by_cell
            ]
            first = relative[0]
            radius = min(self._body_radius[left], self._body_radius[right])
            varies = False
            for value in relative[1:]:
                delta = np.linalg.inv(first) @ value
                displacement_bound = float(np.linalg.norm(delta[:3, 3])) + radius * _rotation_angle(
                    delta[:3, :3]
                )
                if (
                    displacement_bound
                    >= self.policy.motion_translation_tolerance_m
                    + radius * self.policy.motion_rotation_tolerance_rad
                ):
                    varies = True
                    break
            if varies:
                dynamic.append((left, right))
        return tuple(dynamic)

    def scan_grid(
        self,
        *,
        unit_id: str,
        joint_keys: tuple[str, ...],
        cells: tuple[MultiGridCell, ...],
    ) -> JointSweep:
        if not joint_keys:
            raise ValueError("a sweep unit must contain at least one joint")
        closed = self.model.closed_pose()
        poses = tuple(({**closed, **cell.q_by_joint} for cell in cells))
        transforms_by_cell = tuple((self.model.forward_kinematics(pose) for pose in poses))
        baseline = (self.model.forward_kinematics(closed),) if len(transforms_by_cell) == 1 else ()
        dynamic_pairs = (
            self._dynamic_pairs(baseline + transforms_by_cell) if transforms_by_cell else ()
        )
        rest = self.rest_table()
        results = []
        for cell, transforms in zip(cells, transforms_by_cell, strict=True):
            at_closed = all(
                (
                    abs(cell.q_by_joint[key] - closed[key]) <= self.policy.locked_range_tolerance
                    for key in joint_keys
                )
            )
            if at_closed:
                body_pairs = {
                    f"{left}|{right}": rest[f"{left}|{right}"] for left, right in dynamic_pairs
                }
            else:
                body_pairs = self._evaluate_pairs(transforms, dynamic_pairs)
            results.append(
                CellResult(
                    index=cell.index,
                    t=next(iter(cell.t_by_joint.values())) if len(cell.t_by_joint) == 1 else None,
                    t_by_joint=dict(cell.t_by_joint),
                    q_by_joint=dict(cell.q_by_joint),
                    body_pairs=body_pairs,
                )
            )
        return JointSweep(
            joint_key=unit_id,
            swept_joint_keys=joint_keys,
            dynamic_body_pairs=tuple((f"{left}|{right}" for left, right in dynamic_pairs)),
            cells=tuple(results),
        )

    def scan_joint(self, joint: EvalJoint) -> JointSweep:
        cells = tuple(
            (
                MultiGridCell(
                    index=cell.index,
                    t_by_joint={joint.key: cell.t},
                    q_by_joint={joint.key: cell.q},
                    source="normalized",
                )
                for cell in normalized_cells(joint, count=self.policy.l1_cells)
            )
        )
        return self.scan_grid(unit_id=joint.key, joint_keys=(joint.key,), cells=cells)

    def scan_l2(self, left: EvalJoint, right: EvalJoint) -> JointSweep:
        joint_keys = (left.key, right.key)
        return self.scan_grid(
            unit_id="|".join(joint_keys),
            joint_keys=joint_keys,
            cells=l2_cells(left, right, count_per_axis=self.policy.l2_cells_per_axis),
        )

    def scan_lmulti(self, joints: tuple[EvalJoint, ...]) -> JointSweep:
        joint_keys = tuple((joint.key for joint in joints))
        return self.scan_grid(
            unit_id="|".join(joint_keys),
            joint_keys=joint_keys,
            cells=lmulti_cells(
                joints,
                sobol_count=self.policy.lmulti_sobol_cells,
                diagonal_count=self.policy.lmulti_diagonal_cells,
                seed=self.policy.random_seed,
            ),
        )
