from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.stats import qmc

from collision.model import EvalJoint


@dataclass(frozen=True)
class GridCell:
    index: int
    t: float
    q: float


@dataclass(frozen=True)
class MultiGridCell:
    index: int
    t_by_joint: dict[str, float]
    q_by_joint: dict[str, float]
    source: str


def normalized_cells(joint: EvalJoint, *, count: int) -> tuple[GridCell, ...]:
    if count < 2:
        raise ValueError("grid count must be at least two")
    if joint.locked:
        return ()
    if joint.continuous:
        values = []
        for index in range(count):
            q = 2.0 * math.pi * index / count
            if q > math.pi:
                q -= 2.0 * math.pi
            values.append(GridCell(index=index, t=index / count, q=q))
        return tuple(values)
    return tuple(
        (
            GridCell(
                index=index,
                t=index / (count - 1),
                q=joint.closed + index / (count - 1) * (joint.far - joint.closed),
            )
            for index in range(count)
        )
    )


def _q_at_t(joint: EvalJoint, t: float) -> float:
    if joint.continuous:
        return _wrap_pi(2.0 * math.pi * float(t))
    return joint.closed + float(t) * (joint.far - joint.closed)


def l2_cells(
    left: EvalJoint, right: EvalJoint, *, count_per_axis: int
) -> tuple[MultiGridCell, ...]:
    if count_per_axis < 2:
        raise ValueError("L2 count_per_axis must be at least two")
    values = np.linspace(0.0, 1.0, count_per_axis)
    return tuple(
        (
            MultiGridCell(
                index=index,
                t_by_joint={left.key: float(left_t), right.key: float(right_t)},
                q_by_joint={
                    left.key: _q_at_t(left, float(left_t)),
                    right.key: _q_at_t(right, float(right_t)),
                },
                source="cartesian",
            )
            for index, (left_t, right_t) in enumerate(product(values, repeat=2))
        )
    )


def lmulti_cells(
    joints: tuple[EvalJoint, ...], *, sobol_count: int, diagonal_count: int, seed: int
) -> tuple[MultiGridCell, ...]:
    if len(joints) < 3:
        return ()
    if sobol_count < 1 or diagonal_count < 2:
        raise ValueError("Lmulti requires positive Sobol and at least two diagonal cells")
    exponent = int(math.ceil(math.log2(sobol_count)))
    samples = qmc.Sobol(d=len(joints), scramble=True, seed=seed).random_base2(exponent)
    cells = []
    for index, sample in enumerate(samples[:sobol_count]):
        t_values = {joint.key: float(sample[offset]) for offset, joint in enumerate(joints)}
        cells.append(
            MultiGridCell(
                index=index,
                t_by_joint=t_values,
                q_by_joint={joint.key: _q_at_t(joint, t_values[joint.key]) for joint in joints},
                source="sobol",
            )
        )
    for diagonal_index, value in enumerate(np.linspace(0.0, 1.0, diagonal_count)):
        t_values = {joint.key: float(value) for joint in joints}
        cells.append(
            MultiGridCell(
                index=sobol_count + diagonal_index,
                t_by_joint=t_values,
                q_by_joint={joint.key: _q_at_t(joint, float(value)) for joint in joints},
                source="diagonal",
            )
        )
    return tuple(cells)


def _wrap_pi(value: float) -> float:
    wrapped = (value + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if wrapped <= -math.pi else wrapped
