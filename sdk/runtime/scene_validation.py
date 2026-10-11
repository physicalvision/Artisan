from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pipeline.contracts import SupportMode
from sdk.runtime.release_candidate import sha256_file


@dataclass(frozen=True, slots=True)
class SceneValidationPolicy:
    version: str = "scene-stability-v4"
    non_grounded_probe_seconds: float = 0.1
    max_non_grounded_abs_qvel: float = 100.0
    max_non_grounded_abs_qacc: float = 100000.0
    max_grounded_abs_qvel: float = 100.0
    max_grounded_abs_qacc: float = 100000.0
    quiescence_window_seconds: float = 0.4
    max_settle_seconds: float = 5.0
    contact_establish_seconds: float = 0.1
    max_tilt_degrees: float = 3.0
    max_xy_drift_abs_m: float = 0.01
    max_xy_drift_footprint_fraction: float = 0.01
    max_z_displacement_abs_m: float = 0.01
    max_final_linear_speed_m_s: float = 0.02
    max_final_angular_speed_rad_s: float = 0.05
    max_sustained_ground_penetration_m: float = 0.01
    min_sustained_ground_penetration_fraction: float = 0.8


DEFAULT_SCENE_POLICY = SceneValidationPolicy()

_ALWAYS_BLOCKING_GROUNDED_CODES = frozenset(
    {
        "non_finite",
        "physics_explosion",
        "pose_init_failed",
        "no_ground_contact",
        "ground_penetration",
    }
)
_ADVISORY_GROUNDED_CODES = frozenset(
    {
        "persistent_motion",
        "initial_com_outside_support",
        "final_com_outside_support",
        "tilt_exceeded",
        "xy_drift_exceeded",
        "root_z_displacement",
    }
)
assert _ALWAYS_BLOCKING_GROUNDED_CODES.isdisjoint(_ADVISORY_GROUNDED_CODES)


def _finding(code: str, detail: str, **measurements: Any) -> dict[str, Any]:
    return {"code": code, "detail": detail, **measurements}


def _partition_grounded_findings(
    findings: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blocking: list[dict[str, Any]] = []
    advisory: list[dict[str, Any]] = []
    for finding in findings:
        code = finding.get("code") if isinstance(finding, dict) else None
        if code in _ADVISORY_GROUNDED_CODES:
            advisory.append(finding)
        else:
            blocking.append(finding)
    return blocking, advisory


def _finite(data: Any) -> bool:
    import numpy as np

    return bool(
        np.isfinite(data.qpos).all()
        and np.isfinite(data.qvel).all()
        and np.isfinite(data.qacc).all()
    )


_FREE_DOF_COMPONENTS = ("tx", "ty", "tz", "rx", "ry", "rz")
_BALL_DOF_COMPONENTS = ("rx", "ry", "rz")


def _named_object(mujoco: Any, model: Any, object_type: Any, object_id: int, kind: str) -> str:
    name = mujoco.mj_id2name(model, object_type, object_id)
    return str(name) if name else f"{kind}#{object_id}"


def _dof_peak(mujoco: Any, model: Any, values: Any) -> dict[str, Any]:
    import numpy as np

    if len(values) == 0:
        return {
            "value": 0.0,
            "dof_index": None,
            "dof": "none",
            "joint": "none",
            "body": "none",
        }
    dof_index = int(np.argmax(np.abs(values)))
    joint_id = int(model.dof_jntid[dof_index])
    body_id = int(model.dof_bodyid[dof_index])
    joint = _named_object(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, joint_id, "joint")
    body = _named_object(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, body_id, "body")
    joint_type = int(model.jnt_type[joint_id])
    offset = dof_index - int(model.jnt_dofadr[joint_id])
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        component = _FREE_DOF_COMPONENTS[offset]
        dof = f"{joint}:{component}"
    elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        component = _BALL_DOF_COMPONENTS[offset]
        dof = f"{joint}:{component}"
    else:
        dof = joint
    return {
        "value": abs(float(values[dof_index])),
        "dof_index": dof_index,
        "dof": dof,
        "joint": joint,
        "body": body,
    }


def _larger_peak(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    return current if float(current["value"]) > float(previous["value"]) else previous


def _peak_measurements(
    qvel_peak: dict[str, Any],
    qacc_peak: dict[str, Any],
) -> dict[str, Any]:
    measurements: dict[str, Any] = {}
    for quantity, peak in (("qvel", qvel_peak), ("qacc", qacc_peak)):
        prefix = f"max_abs_{quantity}"
        measurements[prefix] = peak["value"]
        measurements[f"{prefix}_dof_index"] = peak["dof_index"]
        measurements[f"{prefix}_dof"] = peak["dof"]
        measurements[f"{prefix}_joint"] = peak["joint"]
        measurements[f"{prefix}_body"] = peak["body"]
    return measurements


def _explosion_evidence(
    qvel_peak: dict[str, Any],
    qacc_peak: dict[str, Any],
    *,
    qvel_limit: float,
    qacc_limit: float,
    time_s: float,
) -> dict[str, Any] | None:
    exceeded = [
        ("qvel", qvel_peak, qvel_limit),
        ("qacc", qacc_peak, qacc_limit),
    ]
    exceeded = [item for item in exceeded if float(item[1]["value"]) > item[2]]
    if not exceeded:
        return None
    quantity, peak, limit = max(
        exceeded,
        key=lambda item: float(item[1]["value"]) / max(float(item[2]), 1e-12),
    )
    return {
        "first_explosion_time_s": float(time_s),
        "offending_quantity": quantity,
        "offending_value": peak["value"],
        "offending_limit": limit,
        "offending_dof_index": peak["dof_index"],
        "offending_dof": peak["dof"],
        "offending_joint": peak["joint"],
        "offending_body": peak["body"],
    }


def _explosion_detail(evidence: dict[str, Any]) -> str:
    return (
        "physics explosion: "
        f"|{evidence['offending_quantity']}| at DOF {evidence['offending_dof']} "
        f"(body {evidence['offending_body']}) reached {evidence['offending_value']:.6g} "
        f"at t={evidence['first_explosion_time_s']:.6g}s "
        f"(limit {evidence['offending_limit']:.6g})"
    )


def _free_joint(mujoco: Any, model: Any) -> tuple[int, int] | None:
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return joint_id, int(model.jnt_bodyid[joint_id])
    return None


def _joint_pose(mujoco: Any, model: Any, data: Any, pose: dict[str, float]) -> None:
    for name, value in pose.items():
        joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name)))
        if joint_id < 0:
            raise ValueError(f"unknown representative-pose joint: {name}")
        joint_type = int(model.jnt_type[joint_id])
        if joint_type in {
            int(mujoco.mjtJoint.mjJNT_FREE),
            int(mujoco.mjtJoint.mjJNT_BALL),
        }:
            raise ValueError(f"representative pose requires a scalar joint: {name}")
        data.qpos[int(model.jnt_qposadr[joint_id])] = float(value)


def _root_speeds(model: Any, data: Any, free_joint_id: int) -> tuple[float, float]:
    import numpy as np

    dof_adr = int(model.jnt_dofadr[free_joint_id])
    linear = float(np.linalg.norm(data.qvel[dof_adr : dof_adr + 3]))
    angular = float(np.linalg.norm(data.qvel[dof_adr + 3 : dof_adr + 6]))
    return linear, angular


def _up_vector(quaternion: Any) -> tuple[float, float, float]:
    w, x, y, z = (float(value) for value in quaternion)
    return (
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    )


def _tilt_degrees(initial_quaternion: Any, current_quaternion: Any) -> float:
    initial = _up_vector(initial_quaternion)
    current = _up_vector(current_quaternion)
    dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(initial, current))))
    return math.degrees(math.acos(dot))


def _ground_contact_snapshot(
    model: Any,
    data: Any,
    *,
    ground_geom_id: int,
) -> tuple[list[tuple[float, float]], float]:
    points: set[tuple[float, float]] = set()
    max_penetration_m = 0.0
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        if ground_geom_id not in {int(contact.geom1), int(contact.geom2)}:
            continue
        points.add((round(float(contact.pos[0]), 7), round(float(contact.pos[1]), 7)))
        max_penetration_m = max(max_penetration_m, max(0.0, -float(contact.dist)))
    return sorted(points), max_penetration_m


def _cross(
    origin: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return unique
    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _signed_margin(point: tuple[float, float], hull: list[tuple[float, float]]) -> float | None:
    if len(hull) < 3:
        return None
    distances: list[float] = []
    for index, start in enumerate(hull):
        end = hull[(index + 1) % len(hull)]
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 1e-12:
            continue
        distances.append((dx * (point[1] - start[1]) - dy * (point[0] - start[0])) / length)
    return min(distances) if distances else None


def _footprint_span(hull: list[tuple[float, float]]) -> float | None:
    if len(hull) < 2:
        return None
    xs = [point[0] for point in hull]
    ys = [point[1] for point in hull]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def _initial_data(mujoco: Any, model: Any, pose: dict[str, float]) -> Any:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    _joint_pose(mujoco, model, data, pose)
    mujoco.mj_forward(model, data)
    return data


def _validate_non_grounded(
    mujoco: Any,
    model: Any,
    *,
    pose: dict[str, float],
    support_mode: SupportMode,
    policy: SceneValidationPolicy,
) -> dict[str, Any]:
    import numpy as np

    try:
        data = _initial_data(mujoco, model, pose)
    except Exception as exc:  # noqa: BLE001 - structured model error
        detail = f"pose initialization failed: {type(exc).__name__}: {exc}"
        return {
            "status": "fail",
            "stability_status": "not-applicable",
            "finite": False,
            "error": detail,
            "blocking_findings": [_finding("pose_init_failed", detail)],
            "advisory_findings": [],
            "requested_pose": dict(pose),
        }
    finite = _finite(data)
    max_qvel_norm = float(np.linalg.norm(data.qvel)) if finite else math.inf
    qvel_peak = _dof_peak(mujoco, model, data.qvel)
    qacc_peak = _dof_peak(mujoco, model, data.qacc)
    first_explosion = (
        _explosion_evidence(
            qvel_peak,
            qacc_peak,
            qvel_limit=policy.max_non_grounded_abs_qvel,
            qacc_limit=policy.max_non_grounded_abs_qacc,
            time_s=float(data.time),
        )
        if finite
        else None
    )
    exploded = first_explosion is not None
    timestep = max(float(model.opt.timestep), 1e-6)
    steps = max(1, math.ceil(policy.non_grounded_probe_seconds / timestep))
    for _ in range(steps):
        if not finite or exploded:
            break
        mujoco.mj_step(model, data)
        finite = _finite(data)
        if finite:
            max_qvel_norm = max(max_qvel_norm, float(np.linalg.norm(data.qvel)))
            current_qvel = _dof_peak(mujoco, model, data.qvel)
            current_qacc = _dof_peak(mujoco, model, data.qacc)
            qvel_peak = _larger_peak(qvel_peak, current_qvel)
            qacc_peak = _larger_peak(qacc_peak, current_qacc)
            first_explosion = _explosion_evidence(
                current_qvel,
                current_qacc,
                qvel_limit=policy.max_non_grounded_abs_qvel,
                qacc_limit=policy.max_non_grounded_abs_qacc,
                time_s=float(data.time),
            )
            exploded = first_explosion is not None
    passed = finite and not exploded
    measurements = _peak_measurements(qvel_peak, qacc_peak)
    error = None
    if not finite:
        error = "non-finite MuJoCo state"
    elif first_explosion is not None:
        error = _explosion_detail(first_explosion)
    blocking_findings: list[dict[str, Any]] = []
    if not finite:
        blocking_findings.append(_finding("non_finite", str(error)))
    elif first_explosion is not None:
        blocking_findings.append(
            _finding(
                "physics_explosion",
                str(error),
                **measurements,
                max_abs_qvel_limit=policy.max_non_grounded_abs_qvel,
                max_abs_qacc_limit=policy.max_non_grounded_abs_qacc,
                **first_explosion,
            )
        )
    return {
        "status": "pass" if passed else "fail",
        "stability_status": "not-applicable",
        "applicability": (
            "load-finiteness-internal"
            if support_mode is SupportMode.EXTERNAL_MOUNT
            else "load-finiteness-no-explosion"
        ),
        "finite": finite,
        "exploded": exploded,
        "simulated_seconds": float(data.time),
        "max_qvel_norm": max_qvel_norm,
        **measurements,
        "error": error,
        "blocking_findings": blocking_findings,
        "advisory_findings": [],
        "requested_pose": dict(pose),
    }


def _validate_grounded_pose(
    mujoco: Any,
    model: Any,
    *,
    free_joint_id: int,
    root_body_id: int,
    ground_geom_id: int,
    pose: dict[str, float],
    policy: SceneValidationPolicy,
) -> dict[str, Any]:
    import numpy as np

    try:
        data = _initial_data(mujoco, model, pose)
    except Exception as exc:  # noqa: BLE001 - structured model error
        detail = f"pose initialization failed: {type(exc).__name__}: {exc}"
        return {
            "status": "fail",
            "stability_status": "fail",
            "finite": False,
            "requested_pose": dict(pose),
            "blocking_findings": [_finding("pose_init_failed", detail)],
            "advisory_findings": [],
        }

    initial_position = np.asarray(data.xpos[root_body_id], dtype=float).copy()
    initial_quaternion = np.asarray(data.xquat[root_body_id], dtype=float).copy()
    initial_com = np.asarray(data.subtree_com[root_body_id], dtype=float).copy()
    initial_contacts, initial_ground_penetration = _ground_contact_snapshot(
        model,
        data,
        ground_geom_id=ground_geom_id,
    )
    initial_hull = _convex_hull(initial_contacts)
    initial_margin = _signed_margin((float(initial_com[0]), float(initial_com[1])), initial_hull)

    timestep = max(float(model.opt.timestep), 1e-6)
    max_steps = max(1, math.ceil(policy.max_settle_seconds / timestep))
    establish_steps = max(0, math.ceil(policy.contact_establish_seconds / timestep))
    window_steps = max(1, math.ceil(policy.quiescence_window_seconds / timestep))
    quiescent_steps = 0
    quiescent = False
    finite = _finite(data)
    qvel_peak = _dof_peak(mujoco, model, data.qvel)
    qacc_peak = _dof_peak(mujoco, model, data.qacc)
    first_explosion = (
        _explosion_evidence(
            qvel_peak,
            qacc_peak,
            qvel_limit=policy.max_grounded_abs_qvel,
            qacc_limit=policy.max_grounded_abs_qacc,
            time_s=float(data.time),
        )
        if finite
        else None
    )
    exploded = first_explosion is not None
    recent_ground_penetrations: deque[float] = deque(maxlen=window_steps)
    max_linear_speed = 0.0
    max_angular_speed = 0.0
    max_tilt = 0.0
    elapsed = 0.0

    for step in range(max_steps):
        if not finite or exploded:
            break
        mujoco.mj_step(model, data)
        elapsed = float(data.time)
        finite = _finite(data)
        if not finite:
            break
        current_qvel = _dof_peak(mujoco, model, data.qvel)
        current_qacc = _dof_peak(mujoco, model, data.qacc)
        qvel_peak = _larger_peak(qvel_peak, current_qvel)
        qacc_peak = _larger_peak(qacc_peak, current_qacc)
        _points, ground_penetration = _ground_contact_snapshot(
            model,
            data,
            ground_geom_id=ground_geom_id,
        )
        recent_ground_penetrations.append(ground_penetration)
        first_explosion = _explosion_evidence(
            current_qvel,
            current_qacc,
            qvel_limit=policy.max_grounded_abs_qvel,
            qacc_limit=policy.max_grounded_abs_qacc,
            time_s=elapsed,
        )
        exploded = first_explosion is not None
        linear_speed, angular_speed = _root_speeds(model, data, free_joint_id)
        tilt = _tilt_degrees(initial_quaternion, data.xquat[root_body_id])
        max_linear_speed = max(max_linear_speed, linear_speed)
        max_angular_speed = max(max_angular_speed, angular_speed)
        max_tilt = max(max_tilt, tilt)
        if exploded:
            break
        if (
            step >= establish_steps
            and linear_speed <= policy.max_final_linear_speed_m_s
            and angular_speed <= policy.max_final_angular_speed_rad_s
        ):
            quiescent_steps += 1
            if quiescent_steps >= window_steps:
                quiescent = True
                break
        else:
            quiescent_steps = 0

    final_position = np.asarray(data.xpos[root_body_id], dtype=float).copy()
    final_com = np.asarray(data.subtree_com[root_body_id], dtype=float).copy()
    linear_speed, angular_speed = _root_speeds(model, data, free_joint_id)
    tilt = _tilt_degrees(initial_quaternion, data.xquat[root_body_id])
    drift = float(np.linalg.norm(final_position[:2] - initial_position[:2]))
    z_displacement = float(final_position[2] - initial_position[2])
    contacts, final_ground_penetration = _ground_contact_snapshot(
        model,
        data,
        ground_geom_id=ground_geom_id,
    )
    if not recent_ground_penetrations:
        recent_ground_penetrations.append(final_ground_penetration)
    max_recent_ground_penetration = max(recent_ground_penetrations, default=0.0)
    penetration_exceedances = sum(
        depth > policy.max_sustained_ground_penetration_m for depth in recent_ground_penetrations
    )
    penetration_exceedance_fraction = (
        penetration_exceedances / len(recent_ground_penetrations)
        if recent_ground_penetrations
        else 0.0
    )
    hull = _convex_hull(contacts)
    margin = _signed_margin((float(final_com[0]), float(final_com[1])), hull)
    span = _footprint_span(initial_hull or hull)
    drift_limit = policy.max_xy_drift_abs_m
    if span is not None and span > 0.0:
        drift_limit = min(
            policy.max_xy_drift_abs_m,
            policy.max_xy_drift_footprint_fraction * span,
        )

    findings: list[dict[str, Any]] = []
    measurements = _peak_measurements(qvel_peak, qacc_peak)
    if not finite:
        findings.append(_finding("non_finite", "non-finite MuJoCo state"))
    if first_explosion is not None:
        findings.append(
            _finding(
                "physics_explosion",
                _explosion_detail(first_explosion),
                **measurements,
                max_abs_qvel_limit=policy.max_grounded_abs_qvel,
                max_abs_qacc_limit=policy.max_grounded_abs_qacc,
                **first_explosion,
            )
        )
    if finite and not exploded and not quiescent:
        findings.append(
            _finding(
                "persistent_motion",
                f"no {policy.quiescence_window_seconds:g}s quiescence "
                f"within {policy.max_settle_seconds:g}s",
                final_linear_speed_m_s=linear_speed,
                final_angular_speed_rad_s=angular_speed,
            )
        )
    if not contacts:
        findings.append(_finding("no_ground_contact", "no final ground contact"))
    if penetration_exceedance_fraction >= policy.min_sustained_ground_penetration_fraction:
        findings.append(
            _finding(
                "ground_penetration",
                "recent ground-contact penetration remained above the gross-failure threshold",
                max_recent_ground_penetration_m=max_recent_ground_penetration,
                ground_penetration_threshold_m=policy.max_sustained_ground_penetration_m,
                recent_ground_penetration_exceedance_fraction=(penetration_exceedance_fraction),
                ground_penetration_exceedance_fraction_threshold=(
                    policy.min_sustained_ground_penetration_fraction
                ),
            )
        )
    if finite and initial_margin is not None and initial_margin < 0.0:
        findings.append(
            _finding(
                "initial_com_outside_support",
                f"initial COM lies {-initial_margin:.6g} m outside the ground-contact support hull",
                initial_support_margin_m=initial_margin,
            )
        )
    if finite and margin is not None and margin < 0.0:
        findings.append(
            _finding(
                "final_com_outside_support",
                f"final COM lies {-margin:.6g} m outside the ground-contact support hull",
                support_margin_m=margin,
            )
        )
    if finite and tilt > policy.max_tilt_degrees:
        findings.append(
            _finding(
                "tilt_exceeded",
                f"root tilt {tilt:.6g} deg exceeds {policy.max_tilt_degrees:.6g} deg",
                tilt_degrees=tilt,
                tilt_limit_degrees=policy.max_tilt_degrees,
            )
        )
    if finite and drift > drift_limit:
        findings.append(
            _finding(
                "xy_drift_exceeded",
                f"root XY drift {drift:.6g} m exceeds {drift_limit:.6g} m",
                xy_drift_m=drift,
                xy_drift_limit_m=drift_limit,
            )
        )
    if finite and abs(z_displacement) > policy.max_z_displacement_abs_m:
        findings.append(
            _finding(
                "root_z_displacement",
                f"root Z displacement {z_displacement:.6g} m exceeds "
                f"{policy.max_z_displacement_abs_m:.6g} m in magnitude",
                z_displacement_m=z_displacement,
                z_displacement_limit_m=policy.max_z_displacement_abs_m,
            )
        )

    blocking_findings, advisory_findings = _partition_grounded_findings(findings)

    return {
        "status": "fail" if blocking_findings else "pass",
        "stability_status": (
            "fail" if blocking_findings else "advisory" if advisory_findings else "pass"
        ),
        "requested_pose": dict(pose),
        "finite": finite,
        "quiescent": quiescent,
        "time_to_quiescence_seconds": elapsed if quiescent else None,
        "simulated_seconds": elapsed,
        "initial_root_position": initial_position.tolist(),
        "final_root_position": final_position.tolist(),
        "xy_drift_m": drift,
        "xy_drift_limit_m": drift_limit,
        "z_displacement_m": z_displacement,
        "tilt_degrees": tilt,
        "max_transient_tilt_degrees": max_tilt,
        "final_linear_speed_m_s": linear_speed,
        "final_angular_speed_rad_s": angular_speed,
        "max_linear_speed_m_s": max_linear_speed,
        "max_angular_speed_rad_s": max_angular_speed,
        **measurements,
        "exploded": exploded,
        "initial_subtree_com": initial_com.tolist(),
        "final_subtree_com": final_com.tolist(),
        "initial_ground_contact_points_xy": [list(point) for point in initial_contacts],
        "initial_ground_contact_hull_xy": [list(point) for point in initial_hull],
        "initial_support_margin_m": initial_margin,
        "initial_ground_penetration_m": initial_ground_penetration,
        "ground_contact_points_xy": [list(point) for point in contacts],
        "ground_contact_hull_xy": [list(point) for point in hull],
        "support_margin_m": margin,
        "final_ground_penetration_m": final_ground_penetration,
        "max_recent_ground_penetration_m": max_recent_ground_penetration,
        "ground_penetration_threshold_m": policy.max_sustained_ground_penetration_m,
        "recent_ground_penetration_exceedance_fraction": penetration_exceedance_fraction,
        "ground_penetration_exceedance_fraction_threshold": (
            policy.min_sustained_ground_penetration_fraction
        ),
        "blocking_findings": blocking_findings,
        "advisory_findings": advisory_findings,
    }


def _pose_findings(
    pose_name: str,
    result: dict[str, Any],
    key: str,
    support_mode: str,
) -> list[dict[str, Any]]:
    aggregated: list[dict[str, Any]] = []
    for raw in result.get(key, []):
        finding = dict(raw) if isinstance(raw, dict) else _finding("malformed_finding", str(raw))
        finding["pose"] = pose_name
        finding["support_mode"] = support_mode
        finding["joint_pose"] = dict(result.get("requested_pose") or {})
        aggregated.append(finding)
    return aggregated


def _aggregate_pose_findings(report: dict[str, Any]) -> None:
    support_mode = str(report["support_mode"])
    blocking = _pose_findings("closed", dict(report["closed"]), "blocking_findings", support_mode)
    advisory = _pose_findings("closed", dict(report["closed"]), "advisory_findings", support_mode)
    for name, result in report["representative_open"].items():
        pose_name = f"representative_open:{name}"
        blocking.extend(_pose_findings(pose_name, dict(result), "blocking_findings", support_mode))
        advisory.extend(_pose_findings(pose_name, dict(result), "advisory_findings", support_mode))
    report["blocking_findings"] = blocking
    report["advisory_findings"] = advisory
    report["status"] = "fail" if blocking else "pass"


def validate_free_root_scene(
    scene_path: Path,
    *,
    support_mode: SupportMode | str,
    representative_poses: dict[str, dict[str, float]] | None = None,
    policy: SceneValidationPolicy = DEFAULT_SCENE_POLICY,
) -> dict[str, Any]:
    scene_path = Path(scene_path)
    try:
        mode = support_mode if isinstance(support_mode, SupportMode) else SupportMode(support_mode)
    except ValueError as exc:
        return {
            "status": "environment-failed",
            "support_mode": str(support_mode),
            "error": f"scene load failed: invalid support_mode: {exc}",
            "blocking_findings": [],
            "advisory_findings": [],
        }
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(scene_path))
    except Exception as exc:  # noqa: BLE001 - dependency/XML failures are structured
        return {
            "status": "environment-failed",
            "support_mode": mode.value,
            "error": f"scene load failed: {type(exc).__name__}: {exc}",
            "blocking_findings": [],
            "advisory_findings": [],
        }

    report: dict[str, Any] = {
        "status": "pass",
        "support_mode": mode.value,
        "scene_path": str(scene_path),
        "scene_sha256": sha256_file(scene_path),
        "mujoco_version": str(mujoco.__version__),
        "policy": asdict(policy),
        "blocking_findings": [],
        "advisory_findings": [],
        "representative_open": {},
    }
    free = _free_joint(mujoco, model)
    if mode is SupportMode.GROUNDED:
        ground_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "scene_ground"))
        if free is None or ground_id < 0:
            missing = "free root" if free is None else "scene_ground"
            report["status"] = "fail"
            report["error"] = f"grounded scene missing required {missing}"
            report["blocking_findings"] = [
                _finding("scene_structure_invalid", str(report["error"]))
            ]
            return report
        free_joint_id, root_body_id = free
        closed = _validate_grounded_pose(
            mujoco,
            model,
            free_joint_id=free_joint_id,
            root_body_id=root_body_id,
            ground_geom_id=ground_id,
            pose={},
            policy=policy,
        )
        report["closed"] = closed
        for name, pose in (representative_poses or {}).items():
            result = _validate_grounded_pose(
                mujoco,
                model,
                free_joint_id=free_joint_id,
                root_body_id=root_body_id,
                ground_geom_id=ground_id,
                pose=pose,
                policy=policy,
            )
            report["representative_open"][str(name)] = result
        _aggregate_pose_findings(report)
        return report

    report["closed"] = _validate_non_grounded(
        mujoco,
        model,
        pose={},
        support_mode=mode,
        policy=policy,
    )
    for name, pose in (representative_poses or {}).items():
        report["representative_open"][str(name)] = _validate_non_grounded(
            mujoco,
            model,
            pose=pose,
            support_mode=mode,
            policy=policy,
        )
    _aggregate_pose_findings(report)
    return report
