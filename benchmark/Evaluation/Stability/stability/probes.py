from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from .loader import (
    JointEquality,
    LoadResult,
    compile_equality_episode,
    ground_contact_count,
    max_ground_penetration,
    place_on_ground,
    set_joint_configuration,
)
from .policy import StabilityPolicy

_WARNING_INDICES = tuple(
    int(mujoco.mjtWarning.__members__[f"mjWARN_{name}"])
    for name in ("BADQPOS", "BADQVEL", "BADQACC", "CNSTRFULL")
)


@dataclass(frozen=True)
class JointTarget:
    key: str
    joint_name: str
    target_state: str
    closed: float
    far: float | None


def compute_joint_target(
    *,
    joint_type: str,
    limited: bool,
    lower: float,
    upper: float,
    inset_fraction: float,
    key: str = "joint",
    joint_name: str | None = None,
) -> JointTarget:
    if joint_type not in {"revolute", "prismatic", "continuous"}:
        raise ValueError(f"unsupported scalar joint type: {joint_type}")
    if not math.isfinite(inset_fraction) or not 0.0 <= inset_fraction < 0.5:
        raise ValueError("limit inset fraction must be finite and in [0, 0.5)")
    compiled_name = joint_name or key
    if joint_type == "continuous":
        return JointTarget(
            key=key, joint_name=compiled_name, target_state="no_far", closed=0.0, far=None
        )
    if not (limited and math.isfinite(lower) and math.isfinite(upper) and (lower < upper)):
        return JointTarget(
            key=key, joint_name=compiled_name, target_state="bad_limits", closed=0.0, far=None
        )
    raw_closed = min(max(0.0, lower), upper)
    lower_distance = abs(raw_closed - lower)
    upper_distance = abs(upper - raw_closed)
    raw_far = lower if lower_distance > upper_distance else upper
    inset = inset_fraction * (upper - lower)

    def inset_endpoint(target: float) -> float:
        if math.isclose(target, lower, rel_tol=0.0, abs_tol=1e-12):
            return lower + inset
        if math.isclose(target, upper, rel_tol=0.0, abs_tol=1e-12):
            return upper - inset
        return target

    return JointTarget(
        key=key,
        joint_name=compiled_name,
        target_state="valid",
        closed=inset_endpoint(raw_closed),
        far=inset_endpoint(raw_far),
    )


class _NumericsAccumulator:
    def __init__(self, data: mujoco.MjData, policy: StabilityPolicy) -> None:
        self._policy = policy
        self._previous_warning = np.array(
            [int(data.warning[index].number) for index in _WARNING_INDICES], dtype=np.int64
        )
        self.warning_seen = False
        self.max_abs_qvel = 0.0
        self.max_abs_qacc = 0.0
        self.nonfinite_seen = False
        self._observe_state(data)

    @staticmethod
    def _finite_max_abs(values: NDArray[np.float64]) -> tuple[float, bool]:
        if values.size == 0:
            return (0.0, False)
        if bool(np.any(np.isnan(values))):
            return (math.nan, True)
        if bool(np.any(np.isinf(values))):
            return (math.inf, True)
        finite = np.isfinite(values)
        maximum = float(np.max(np.abs(values[finite]))) if np.any(finite) else 0.0
        return (maximum, not bool(np.all(finite)))

    @staticmethod
    def _merge_maximum(current: float, observed: float) -> float:
        if math.isnan(current) or math.isnan(observed):
            return math.nan
        return max(current, observed)

    def _observe_state(self, data: mujoco.MjData) -> None:
        qvel_max, qvel_nonfinite = self._finite_max_abs(np.asarray(data.qvel))
        qacc_max, qacc_nonfinite = self._finite_max_abs(np.asarray(data.qacc))
        self.max_abs_qvel = self._merge_maximum(self.max_abs_qvel, qvel_max)
        self.max_abs_qacc = self._merge_maximum(self.max_abs_qacc, qacc_max)
        self.nonfinite_seen |= qvel_nonfinite or qacc_nonfinite
        self.nonfinite_seen |= not bool(np.all(np.isfinite(data.qpos)))

    def observe_step(self, data: mujoco.MjData) -> None:
        current = np.array(
            [int(data.warning[index].number) for index in _WARNING_INDICES], dtype=np.int64
        )
        deltas = np.where(
            current >= self._previous_warning, current - self._previous_warning, current
        )
        self.warning_seen |= bool(np.any(deltas > 0))
        self._previous_warning = current
        self._observe_state(data)

    def numerical_failure(self) -> bool:
        return (
            self.nonfinite_seen
            or self.warning_seen
            or self.max_abs_qvel > self._policy.max_abs_qvel
            or (self.max_abs_qacc > self._policy.max_abs_qacc)
        )


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(element: ET.Element, name: str) -> ET.Element | None:
    return next((child for child in element if _local_name(child.tag) == name), None)


def inspect_joint_targets(
    loaded: LoadResult, *, policy: StabilityPolicy | None = None
) -> tuple[JointTarget, ...]:
    active_policy = policy or loaded.policy
    if active_policy != loaded.policy:
        raise ValueError("joint target policy must match the loaded asset policy")
    if loaded.model is None:
        raise ValueError("joint targets require a loaded asset")
    try:
        root = ET.parse(loaded.urdf_path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ValueError(f"cannot inspect URDF joints: {exc}") from exc
    declarations = {
        element.get("name", ""): element
        for element in root.iter()
        if _local_name(element.tag) == "joint"
    }
    targets: list[JointTarget] = []
    for joint_id in _scalar_joint_ids(loaded.model):
        if joint_id == loaded.free_joint_id:
            continue
        compiled_name = _joint_name(loaded.model, joint_id)
        key = compiled_name.removeprefix("asset_")
        declaration = declarations.get(key)
        declared_type = declaration.get("type") if declaration is not None else None
        if declared_type == "continuous":
            joint_type = "continuous"
        elif int(loaded.model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_SLIDE):
            joint_type = "prismatic"
        else:
            joint_type = "revolute"
        lower, upper = (float(value) for value in loaded.model.jnt_range[joint_id])
        limit = _child(declaration, "limit") if declaration is not None else None
        if limit is not None:
            try:
                lower = float(limit.get("lower", "nan"))
                upper = float(limit.get("upper", "nan"))
            except ValueError:
                lower, upper = (math.nan, math.nan)
        targets.append(
            compute_joint_target(
                joint_type=joint_type,
                limited=bool(loaded.model.jnt_limited[joint_id]),
                lower=lower,
                upper=upper,
                inset_fraction=active_policy.limit_inset_fraction,
                key=key,
                joint_name=compiled_name,
            )
        )
    return tuple(sorted(targets, key=lambda joint: joint.key))


def _scalar_joint_ids(model: mujoco.MjModel) -> tuple[int, ...]:
    scalar_types = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
    return tuple(
        (
            joint_id
            for joint_id in range(model.njnt)
            if int(model.jnt_type[joint_id]) in scalar_types
        )
    )


def _rotation_distance(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    relative = left.T @ right
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.acos(cosine)


def _swing_degrees(initial: NDArray[np.float64], current: NDArray[np.float64]) -> float:
    cosine = float(np.clip(np.dot(initial[:, 2], current[:, 2]), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _joint_name(model: mujoco.MjModel, joint_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"


class _EqualityTracker:
    def __init__(self, loaded: LoadResult, targets: Mapping[str, float]) -> None:
        if loaded.model is None:
            raise ValueError("equality tracking requires a compiled model")
        self._loaded = loaded
        self._targets = dict(targets)
        self._max_error = {name: 0.0 for name in targets}

    def observe(self, data: mujoco.MjData) -> None:
        model = self._loaded.model
        if model is None:
            return
        for joint_name, target in self._targets.items():
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_address = int(model.jnt_qposadr[joint_id])
            error = abs(float(data.qpos[qpos_address]) - target)
            self._max_error[joint_name] = max(self._max_error[joint_name], error)

    def invalid(self, policy: StabilityPolicy) -> bool:
        model = self._loaded.model
        if model is None:
            return True
        for joint_name, maximum in self._max_error.items():
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            tolerance = (
                policy.eq_tracking_tol_rad
                if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_HINGE)
                else policy.eq_tracking_tol_m
            )
            if maximum > tolerance:
                return True
        return False


def _run_configuration(
    loaded: LoadResult, targets: Mapping[str, float], timestep_s: float
) -> dict[str, Any]:
    policy = loaded.policy
    equalities = [JointEquality(joint_name=name, target=target) for name, target in targets.items()]
    episode = compile_equality_episode(loaded, equalities, timestep_s=timestep_s)
    model = episode.model
    data = mujoco.MjData(model)
    set_joint_configuration(episode, data, targets)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    place_on_ground(episode, data)
    timestep = float(model.opt.timestep)
    settle_steps = int(math.ceil(policy.settle_seconds / timestep))
    test_steps = int(math.ceil(policy.test_seconds / timestep))
    marker_id = episode.marker_site_id
    initial_rotation = np.asarray(data.site_xmat[marker_id], dtype=np.float64).reshape(3, 3).copy()
    reference_position: NDArray[np.float64] | None = None
    reference_rotation: NDArray[np.float64] | None = None
    max_position_deviation = 0.0
    max_rotation_deviation = 0.0
    max_swing = _swing_degrees(initial_rotation, initial_rotation)
    max_penetration = max_ground_penetration(episode, data)
    ground_contact_min: int | None = None
    tracker = _EqualityTracker(episode, targets) if targets else None
    if tracker is not None:
        tracker.observe(data)
    accumulator = _NumericsAccumulator(data, policy)
    for index in range(settle_steps + test_steps):
        mujoco.mj_step(model, data)
        accumulator.observe_step(data)
        position = np.asarray(data.site_xpos[marker_id], dtype=np.float64)
        rotation = np.asarray(data.site_xmat[marker_id], dtype=np.float64).reshape(3, 3)
        max_swing = max(max_swing, _swing_degrees(initial_rotation, rotation))
        max_penetration = max(max_penetration, max_ground_penetration(episode, data))
        if tracker is not None:
            tracker.observe(data)
        if index == settle_steps - 1:
            reference_position = position.copy()
            reference_rotation = rotation.copy()
        elif index >= settle_steps:
            if reference_position is None or reference_rotation is None:
                reference_position = position.copy()
                reference_rotation = rotation.copy()
            max_position_deviation = max(
                max_position_deviation, float(np.linalg.norm(position - reference_position))
            )
            max_rotation_deviation = max(
                max_rotation_deviation, _rotation_distance(reference_rotation, rotation)
            )
            count = ground_contact_count(episode, data)
            ground_contact_min = (
                count if ground_contact_min is None else min(ground_contact_min, count)
            )
        if accumulator.numerical_failure():
            break
    numerical = accumulator.numerical_failure()
    ground_contact_min = 0 if ground_contact_min is None else ground_contact_min
    if tracker is not None and tracker.invalid(policy):
        status, reason = ("invalid_probe", None)
    elif numerical:
        status, reason = ("fail", "numerical")
    elif max_swing > policy.tipped_swing_deg:
        status, reason = ("fail", "tipped")
    elif ground_contact_min < 1:
        status, reason = ("fail", "airborne")
    elif max_penetration > policy.max_ground_penetration_m:
        status, reason = ("fail", "penetration")
    elif (
        max_position_deviation > policy.max_pos_dev_m
        or max_rotation_deviation > policy.max_rot_dev_rad
    ):
        status, reason = ("fail", "unsettled")
    else:
        status, reason = ("pass", None)
    return {"status": status, "fail_reason": reason}


def run_configurations(loaded: LoadResult) -> dict[str, dict[str, Any]]:
    joints = inspect_joint_targets(loaded)
    selected = tuple((joint for joint in joints if joint.target_state != "bad_limits"))
    valid = tuple((joint for joint in selected if joint.target_state == "valid"))
    cells: dict[str, dict[str, Any]] = {}
    for name in ("Rest", "Open"):
        if name == "Open" and (not valid):
            cells[name] = {"status": "not_applicable", "fail_reason": None}
            continue
        targets = {
            joint.joint_name: float(joint.far)
            if name == "Open" and joint.target_state == "valid"
            else joint.closed
            for joint in selected
        }
        cell = _run_configuration(loaded, targets, loaded.policy.timestep_s)
        cells[name] = cell
    return cells
