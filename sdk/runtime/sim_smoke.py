from __future__ import annotations

import hashlib
import json
import math
import tempfile
import xml.etree.ElementTree as ET
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Optional


def _enumerate_couplings(object_model: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for a in getattr(object_model, "articulations", None) or []:
        mim = getattr(a, "mimic", None)
        beh = getattr(a, "behavior", None)
        if mim is not None:
            poly = tuple(getattr(mim, "poly", ()) or ())
            out.append(
                {
                    "follower": a.name,
                    "driver": str(getattr(mim, "joint", "")),
                    "kind": "nonlinear" if poly else "proportional",
                    "mimic": {
                        "multiplier": float(getattr(mim, "multiplier", 1.0)),
                        "offset": float(getattr(mim, "offset", 0.0)),
                        "poly": poly,
                    },
                }
            )
        elif beh is not None:
            out.append(
                {
                    "follower": a.name,
                    "driver": str(getattr(beh, "driver", "")),
                    "kind": str(getattr(beh, "kind", "")),
                    "behavior": beh,
                }
            )
    return out


def _enumerate_articulation_joints(object_model: Any) -> list[str]:
    from sdk._core.v0.self_collision import _moving_joints

    behavior_followers = {
        a.name
        for a in getattr(object_model, "articulations", None) or []
        if getattr(a, "behavior", None) is not None
    }
    return [j.name for j in _moving_joints(object_model) if j.name not in behavior_followers]


def _gate_open_preset(m: Any, gate_joints: list[str]) -> dict[str, float]:
    preset: dict[str, float] = {}
    for g in gate_joints:
        jid = _jid(m, g)
        if jid < 0:
            continue
        lo, hi = float(m.jnt_range[jid][0]), float(m.jnt_range[jid][1])
        if lo == hi:
            preset[g] = math.pi / 2
        else:
            preset[g] = lo if abs(lo) > abs(hi) else hi
    return preset


_JAM_TRACK = 0.5
_MIN_BASELINE_TRAVEL = 1e-3
_JAM_SUSTAINED_MIN = 0.15
_JAM_FORCE_FLOOR = 50.0
_DRIVE_KP = 150.0
_DRIVE_KV = 15.0

_KINEMATIC_PENETRATION_POLICY = "candidate-mjcf-penetration-v3"
_MAX_REVOLUTE_STEP_RAD = 0.1
_MAX_SLIDE_STEP_M = 0.01


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _mjcf_mimic_relations(path: Path) -> dict[str, tuple[str, tuple[float, ...]]]:
    relations: dict[str, tuple[str, tuple[float, ...]]] = {}
    root = ET.parse(path).getroot()
    for equality in root.findall("./equality/joint"):
        follower = equality.get("joint1")
        driver = equality.get("joint2")
        if not follower or not driver:
            continue
        coefficients = tuple(
            float(value) for value in (equality.get("polycoef") or "0 1 0 0 0").split()
        )
        relations[follower] = (driver, coefficients)
    return relations


def _complete_coupled_pose(
    pose: dict[str, float],
    relations: dict[str, tuple[str, tuple[float, ...]]],
) -> dict[str, float]:
    completed = dict(pose)
    for follower, (driver, coefficients) in relations.items():
        if follower in completed or driver not in completed:
            continue
        q = completed[driver]
        completed[follower] = sum(
            coefficient * q**power for power, coefficient in enumerate(coefficients)
        )
    return completed


def _candidate_joint_poses(
    m: Any,
    relations: dict[str, tuple[str, tuple[float, ...]]],
    gate_map: dict[str, list[str]],
) -> list[dict[str, float]]:
    import mujoco

    poses: list[dict[str, float]] = [{}]
    for jid in range(int(m.njnt)):
        joint_type = int(m.jnt_type[jid])
        if joint_type not in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if not name:
            continue
        if name in relations:
            continue
        lo, hi = (float(v) for v in m.jnt_range[jid])
        if not bool(m.jnt_limited[jid]):
            if joint_type == int(mujoco.mjtJoint.mjJNT_HINGE):
                lo, hi = -math.pi, math.pi
            else:
                continue
        if not math.isfinite(lo + hi) or hi <= lo:
            continue
        step = (
            _MAX_REVOLUTE_STEP_RAD
            if joint_type == int(mujoco.mjtJoint.mjJNT_HINGE)
            else _MAX_SLIDE_STEP_M
        )
        count = max(1, int(math.ceil((hi - lo) / step)))
        for index in range(count + 1):
            q = lo + (hi - lo) * index / count
            pose = {name: q}
            pose.update(_gate_open_preset(m, gate_map.get(name, [])))
            poses.append(_complete_coupled_pose(pose, relations))
    return poses


def _set_scalar_joint_pose(m: Any, d: Any, pose: dict[str, float]) -> None:
    import mujoco

    for name, value in pose.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"unknown joint in sampled pose: {name}")
        joint_type = int(m.jnt_type[jid])
        if joint_type not in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            raise ValueError(f"sampled pose requires scalar hinge/slide joint: {name}")
        d.qpos[int(m.jnt_qposadr[jid])] = float(value)


def kinematic_penetration_sweep(
    mjcf_path: Path,
    *,
    poses: list[dict[str, float]] | None = None,
    gate_map: dict[str, list[str]] | None = None,
    penetration_depth_tol_m: float | None = None,
    cache_path: Path | None = None,
) -> dict[str, Any]:
    from artisan.physics_policy import (
        penetration_depth_tolerance_from_env,
        validate_penetration_depth_tolerance,
    )

    threshold = (
        penetration_depth_tolerance_from_env()
        if penetration_depth_tol_m is None
        else validate_penetration_depth_tolerance(penetration_depth_tol_m)
    )

    if poses is not None and not poses:
        return {
            "algorithm_version": _KINEMATIC_PENETRATION_POLICY,
            "status": "fail",
            "error": "explicit candidate pose list is empty",
            "accepted_input_poses": [],
            "completed_simulated_poses": [],
            "poses": [],
            "pose_count": 0,
            "blocking_findings": [
                {
                    "kind": "candidate_pose_contract",
                    "detail": "explicit candidate pose list must not be empty",
                }
            ],
            "cache_hit": False,
        }

    path = Path(mjcf_path)
    try:
        import mujoco

        xml_bytes = path.read_bytes()
        xml_sha256 = hashlib.sha256(xml_bytes).hexdigest()
        mimic_relations = _mjcf_mimic_relations(path)
        model = mujoco.MjModel.from_xml_path(str(path))
    except Exception as exc:  # noqa: BLE001 - reported as an explicit environment failure
        return {
            "status": "environment-failed",
            "error": f"MJCF load failed: {type(exc).__name__}: {exc}",
            "blocking_findings": [],
            "cache_hit": False,
        }

    sampled_poses = (
        poses
        if poses is not None
        else _candidate_joint_poses(model, mimic_relations, gate_map or {})
    )
    accepted_input_poses = [
        {str(name): float(value) for name, value in sorted(pose.items())} for pose in sampled_poses
    ]
    completed_simulated_poses = [
        {
            str(name): float(value)
            for name, value in sorted(_complete_coupled_pose(pose, mimic_relations).items())
        }
        for pose in accepted_input_poses
    ]
    accepted_input_pose_sha256 = hashlib.sha256(
        _canonical_json(accepted_input_poses).encode("utf-8")
    ).hexdigest()
    completed_simulated_pose_sha256 = hashlib.sha256(
        _canonical_json(completed_simulated_poses).encode("utf-8")
    ).hexdigest()
    identity = {
        "algorithm_version": _KINEMATIC_PENETRATION_POLICY,
        "mjcf_sha256": xml_sha256,
        "mujoco_version": str(mujoco.__version__),
        "penetration_depth_tol_m": threshold,
        "accepted_input_poses": accepted_input_poses,
        "accepted_input_pose_sha256": accepted_input_pose_sha256,
        "completed_simulated_poses": completed_simulated_poses,
        "completed_simulated_pose_sha256": completed_simulated_pose_sha256,
        "poses": completed_simulated_poses,
    }
    cache_key = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    if cache_path is not None:
        try:
            cached = json.loads(Path(cache_path).read_text(encoding="utf-8"))
            if isinstance(cached, dict) and cached.get("cache_key") == cache_key:
                return {**cached, "cache_hit": True}
        except Exception:  # noqa: BLE001 - a cache miss is not a validation failure
            pass

    worst_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    max_depth = 0.0
    try:
        for pose in completed_simulated_poses:
            data = mujoco.MjData(model)
            mujoco.mj_resetData(model, data)
            _set_scalar_joint_pose(model, data, pose)
            mujoco.mj_forward(model, data)
            for index in range(int(data.ncon)):
                contact = data.contact[index]
                depth = max(0.0, -float(contact.dist))
                max_depth = max(max_depth, depth)
                if depth <= threshold:
                    continue
                body_a, body_b = _contact_pair_parts(model, data, index)
                pair = (body_a, body_b)
                previous = worst_by_pair.get(pair)
                if previous is not None and float(previous["penetration_depth_m"]) >= depth:
                    continue
                geom_a = (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or "?"
                )
                geom_b = (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or "?"
                )
                worst_by_pair[pair] = {
                    "kind": "candidate_mjcf_penetration",
                    "pair": [body_a, body_b],
                    "geoms": [geom_a, geom_b],
                    "pose": pose,
                    "penetration_depth_m": depth,
                    "penetration_depth_tol_m": threshold,
                }
    except Exception as exc:  # noqa: BLE001 - malformed pose/model is explicit failure
        return {
            **identity,
            "cache_key": cache_key,
            "status": "environment-failed",
            "error": f"MJCF sweep failed: {type(exc).__name__}: {exc}",
            "blocking_findings": [],
            "cache_hit": False,
        }

    findings = sorted(
        worst_by_pair.values(),
        key=lambda finding: float(finding["penetration_depth_m"]),
        reverse=True,
    )
    report = {
        **identity,
        "cache_key": cache_key,
        "pose_count": len(completed_simulated_poses),
        "status": "fail" if findings else "pass",
        "max_penetration_depth_m": max_depth,
        "blocking_findings": findings,
        "cache_hit": False,
    }
    if cache_path is not None:
        try:
            cache = Path(cache_path)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(_canonical_json(report), encoding="utf-8")
        except Exception:  # noqa: BLE001 - caching is an optimization
            pass
    return report


def _boost_driver_actuator(m: Any, act_id: int) -> None:
    m.actuator_gainprm[act_id, 0] = _DRIVE_KP
    m.actuator_biasprm[act_id, 1] = -_DRIVE_KP
    m.actuator_biasprm[act_id, 2] = -_DRIVE_KV


def _prepare_rollout(m: Any) -> None:
    import mujoco

    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    _critically_damp_springs(m)


def _critically_damp_springs(m: Any) -> None:
    for jid in range(int(m.njnt)):
        k = float(m.jnt_stiffness[jid])
        if k <= 0.0:
            continue
        dofadr = int(m.jnt_dofadr[jid])
        if dofadr < 0:
            continue
        if float(m.dof_damping[dofadr]) > 1e-9:
            continue
        inertia = float(m.dof_M0[dofadr])
        if inertia <= 0.0:
            inertia = 1.0
        m.dof_damping[dofadr] = 2.0 * math.sqrt(k * inertia)


def _settle(m: Any, d: Any, *, seconds: float = 4.0, qvel_tol: float = 1e-4) -> str:
    import mujoco
    import numpy as np

    n = max(1, int(seconds / max(m.opt.timestep, 1e-6)))
    for _ in range(n):
        mujoco.mj_step(m, d)
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
            return "nan"
        if np.isfinite(qvel_tol) and float(np.max(np.abs(d.qvel))) < qvel_tol:
            return "ok"
    return "ok" if not np.isfinite(qvel_tol) else "unsettled"


def _jid(m: Any, name: str) -> int:
    import mujoco

    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)


def _aid(m: Any, name: str) -> int:
    import mujoco

    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


_DYNAMICS_ALGORITHM = "joint-dynamics-v3"
_DYNAMICS_POLICY: dict[str, object] = {
    "contacts_enabled": False,
    "below_breakaway_ratio": 0.7,
    "above_breakaway_ratio": 1.5,
    "breakaway_seconds": 1.0,
    "below_breakaway_drift_fraction": 0.01,
    "above_breakaway_motion_fraction": 0.02,
    "hold_seconds": 2.0,
    "hold_drift_fraction": 0.02,
    "hold_pose_count": 5,
    "decay_target_speed_fraction": 0.1,
    "decay_upper_time_ratio": 4.0,
}


def _load_dynamics_model(mjcf_bytes: bytes, assets: Optional[dict[str, bytes]]) -> Any:
    import mujoco

    asset_map = assets or {}
    basenames = [PurePosixPath(key.replace("\\", "/")).name for key in asset_map]
    if len(basenames) == len(set(basenames)):
        return mujoco.MjModel.from_xml_string(mjcf_bytes.decode("utf-8"), assets=asset_map)

    import xml.etree.ElementTree as ET

    root = ET.fromstring(mjcf_bytes)
    compiler = root.find("compiler")
    meshdir = str(compiler.get("meshdir", "") or "") if compiler is not None else ""
    meshdir_path = PurePosixPath(meshdir.replace("\\", "/"))
    if meshdir_path.is_absolute() or ".." in meshdir_path.parts:
        raise ValueError(f"unsafe MJCF meshdir for dynamics validation: {meshdir!r}")

    with tempfile.TemporaryDirectory(prefix="artisan-dynamics-") as scratch:
        scratch_path = Path(scratch)
        mjcf_path = scratch_path / "candidate.xml"
        mjcf_path.write_bytes(mjcf_bytes)
        asset_root = scratch_path.joinpath(*meshdir_path.parts)
        for logical_name, payload in asset_map.items():
            relative = PurePosixPath(str(logical_name).replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    f"unsafe MJCF asset path for dynamics validation: {logical_name!r}"
                )
            destination = asset_root.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        return mujoco.MjModel.from_xml_path(str(mjcf_path))


def _joint_coordinates(m: Any, joint: str) -> tuple[int, int, int, float, float, float, str] | None:
    import mujoco

    jid = _jid(m, joint)
    if jid < 0:
        return None
    joint_type = int(m.jnt_type[jid])
    if joint_type not in {
        int(mujoco.mjtJoint.mjJNT_HINGE),
        int(mujoco.mjtJoint.mjJNT_SLIDE),
    }:
        return None
    dof = int(m.jnt_dofadr[jid])
    qpos = int(m.jnt_qposadr[jid])
    limited = bool(m.jnt_limited[jid])
    if limited:
        lo, hi = float(m.jnt_range[jid][0]), float(m.jnt_range[jid][1])
    elif joint_type == int(mujoco.mjtJoint.mjJNT_HINGE):
        lo, hi = -math.pi, math.pi
    else:
        lo, hi = -0.5, 0.5
    rom = max(hi - lo, 1e-6)
    kind = "revolute" if joint_type == int(mujoco.mjtJoint.mjJNT_HINGE) else "prismatic"
    return jid, dof, qpos, lo, hi, rom, kind


def _disable_contacts(m: Any) -> None:
    import mujoco

    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)


def _frozen_dof_state(m: Any, d: Any, tested_jid: int) -> tuple[Any, Any, Any, Any]:
    import numpy as np

    qpos_mask = np.ones(int(m.nq), dtype=bool)
    qvel_mask = np.ones(int(m.nv), dtype=bool)
    qpos_start = int(m.jnt_qposadr[tested_jid])
    dof_start = int(m.jnt_dofadr[tested_jid])
    qpos_end = int(m.jnt_qposadr[tested_jid + 1]) if tested_jid + 1 < int(m.njnt) else int(m.nq)
    dof_end = int(m.jnt_dofadr[tested_jid + 1]) if tested_jid + 1 < int(m.njnt) else int(m.nv)
    qpos_mask[qpos_start:qpos_end] = False
    qvel_mask[dof_start:dof_end] = False
    return qpos_mask, d.qpos[qpos_mask].copy(), qvel_mask, d.qvel[qvel_mask].copy()


def _new_dynamics_diagnostics() -> dict[str, object]:
    return {
        "max_abs_qacc": 0.0,
        "max_abs_qvel": 0.0,
        "offending_dof": None,
        "offending_body": None,
    }


def _update_dynamics_diagnostics(m: Any, d: Any, diagnostics: dict[str, object]) -> None:
    import mujoco
    import numpy as np

    if int(m.nv) <= 0:
        return
    qacc = np.abs(np.asarray(d.qacc, dtype=float))
    qvel = np.abs(np.asarray(d.qvel, dtype=float))
    qacc_index = int(np.nanargmax(qacc)) if qacc.size else 0
    qvel_index = int(np.nanargmax(qvel)) if qvel.size else 0
    qacc_value = float(qacc[qacc_index]) if qacc.size else 0.0
    qvel_value = float(qvel[qvel_index]) if qvel.size else 0.0
    if qacc_value >= float(diagnostics["max_abs_qacc"]):
        diagnostics["max_abs_qacc"] = qacc_value
        jid = int(m.dof_jntid[qacc_index])
        diagnostics["offending_dof"] = (
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"dof:{qacc_index}"
        )
        body_id = int(m.jnt_bodyid[jid])
        diagnostics["offending_body"] = (
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body:{body_id}"
        )
    diagnostics["max_abs_qvel"] = max(float(diagnostics["max_abs_qvel"]), qvel_value)


def _step_isolated(
    m: Any,
    d: Any,
    frozen: tuple[Any, Any, Any, Any],
    diagnostics: dict[str, object],
) -> bool:
    import mujoco

    warning_index = int(mujoco.mjtWarning.mjWARN_BADQACC)
    warning_before = int(d.warning[warning_index].number)
    mujoco.mj_step(m, d)
    _update_dynamics_diagnostics(m, d, diagnostics)
    qpos_mask, qpos_values, qvel_mask, qvel_values = frozen
    d.qpos[qpos_mask] = qpos_values
    d.qvel[qvel_mask] = qvel_values
    mujoco.mj_forward(m, d)
    _update_dynamics_diagnostics(m, d, diagnostics)
    return int(d.warning[warning_index].number) > warning_before


def _badqacc_result(diagnostics: dict[str, object]) -> dict[str, object]:
    return {
        "status": "inconclusive",
        "reason": "badqacc-contamination",
        **diagnostics,
    }


def _retry_badqacc(runner: Any) -> dict[str, object]:
    first = runner(False)
    if first.get("reason") != "badqacc-contamination":
        return first
    retried = runner(True)
    retried["retry_with_prepare"] = True
    retried["first_attempt"] = first
    if retried.get("reason") == "badqacc-contamination":
        retried["status"] = "inconclusive"
        retried["reason"] = "persistent-badqacc"
    return retried


def _run_breakaway_once(
    mjcf_bytes: bytes,
    assets: Optional[dict[str, bytes]],
    joint: str,
    frictionloss: float,
    *,
    prepared: bool,
) -> dict[str, object]:
    import mujoco
    import numpy as np

    ratios = (
        ("below", float(_DYNAMICS_POLICY["below_breakaway_ratio"])),
        ("above", float(_DYNAMICS_POLICY["above_breakaway_ratio"])),
    )
    observations: dict[str, float] = {}
    diagnostics = _new_dynamics_diagnostics()
    rom = 1.0
    for label, ratio in ratios:
        m = _load_dynamics_model(mjcf_bytes, assets)
        _disable_contacts(m)
        if prepared:
            _prepare_rollout(m)
        coords = _joint_coordinates(m, joint)
        if coords is None:
            return {"status": "not-applicable", "reason": "joint-missing-or-unsupported"}
        _jid_value, dof, qpos, lo, hi, rom, joint_type = coords
        d = mujoco.MjData(m)
        d.qpos[qpos] = 0.5 * (lo + hi)
        d.qvel[dof] = 0.0
        mujoco.mj_forward(m, d)
        frozen = _frozen_dof_state(m, d, _jid_value)
        start = float(d.qpos[qpos])
        steps = max(1, int(float(_DYNAMICS_POLICY["breakaway_seconds"]) / m.opt.timestep))
        for _ in range(steps):
            d.qfrc_applied[:] = 0.0
            d.qfrc_applied[dof] = float(d.qfrc_bias[dof]) + ratio * frictionloss
            if _step_isolated(m, d, frozen, diagnostics):
                return _badqacc_result(diagnostics)
            if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
                return {"status": "fail", "reason": f"diverged-{label}"}
        observations[f"{label}_drift"] = abs(float(d.qpos[qpos]) - start)

    below_limit = float(_DYNAMICS_POLICY["below_breakaway_drift_fraction"]) * rom
    above_min = float(_DYNAMICS_POLICY["above_breakaway_motion_fraction"]) * rom
    passed = observations["below_drift"] <= below_limit and observations["above_drift"] >= above_min
    return {
        "status": "pass" if passed else "fail",
        "joint_type": joint_type,
        "frictionloss": frictionloss,
        **{key: round(value, 8) for key, value in observations.items()},
        "below_max": round(below_limit, 8),
        "above_min": round(above_min, 8),
        **diagnostics,
    }


def _run_breakaway(
    mjcf_bytes: bytes,
    assets: Optional[dict[str, bytes]],
    joint: str,
    frictionloss: float,
) -> dict[str, object]:
    return _retry_badqacc(
        lambda prepared: _run_breakaway_once(
            mjcf_bytes,
            assets,
            joint,
            frictionloss,
            prepared=prepared,
        )
    )


def _hold_poses(lo: float, hi: float, worst_pose: float) -> list[float]:
    span = hi - lo
    poses = [lo + span * fraction for fraction in (0.1, 0.3, 0.5, 0.7, 0.9)]
    if lo <= worst_pose <= hi:
        nearest = min(range(len(poses)), key=lambda index: abs(poses[index] - worst_pose))
        poses[nearest] = worst_pose
    return poses


def _run_hold_once(
    mjcf_bytes: bytes,
    assets: Optional[dict[str, bytes]],
    joint: str,
    hold: str,
    worst_pose: float,
    *,
    prepared: bool,
) -> dict[str, object]:
    import mujoco
    import numpy as np

    probe = _load_dynamics_model(mjcf_bytes, assets)
    coords = _joint_coordinates(probe, joint)
    if coords is None:
        return {"status": "not-applicable", "reason": "joint-missing-or-unsupported"}
    _jid_value, _dof, _qpos, lo, hi, rom, joint_type = coords
    observations: list[dict[str, float]] = []
    diagnostics = _new_dynamics_diagnostics()
    for pose in _hold_poses(lo, hi, worst_pose):
        m = _load_dynamics_model(mjcf_bytes, assets)
        _disable_contacts(m)
        if prepared:
            _prepare_rollout(m)
        current = _joint_coordinates(m, joint)
        assert current is not None
        _current_jid, dof, qpos, _lo, _hi, _rom, _joint_type = current
        d = mujoco.MjData(m)
        d.qpos[qpos] = pose
        d.qvel[dof] = 0.0
        mujoco.mj_forward(m, d)
        frozen = _frozen_dof_state(m, d, _current_jid)
        start = float(d.qpos[qpos])
        steps = max(1, int(float(_DYNAMICS_POLICY["hold_seconds"]) / m.opt.timestep))
        for _ in range(steps):
            if _step_isolated(m, d, frozen, diagnostics):
                return _badqacc_result(diagnostics)
            if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
                return {"status": "fail", "reason": "diverged"}
        observations.append({"pose": pose, "drift": abs(float(d.qpos[qpos]) - start)})
    max_drift = max(item["drift"] for item in observations)
    limit = float(_DYNAMICS_POLICY["hold_drift_fraction"]) * rom
    if hold != "holds_anywhere":
        status = "not-required"
    else:
        status = "pass" if max_drift <= limit else "fail"
    return {
        "status": status,
        "joint_type": joint_type,
        "max_drift": round(max_drift, 8),
        "max_drift_allowed": round(limit, 8),
        "poses": [
            {"pose": round(item["pose"], 8), "drift": round(item["drift"], 8)}
            for item in observations
        ],
        **diagnostics,
    }


def _run_hold(
    mjcf_bytes: bytes,
    assets: Optional[dict[str, bytes]],
    joint: str,
    hold: str,
    worst_pose: float,
) -> dict[str, object]:
    return _retry_badqacc(
        lambda prepared: _run_hold_once(
            mjcf_bytes,
            assets,
            joint,
            hold,
            worst_pose,
            prepared=prepared,
        )
    )


def _run_decay_once(
    mjcf_bytes: bytes,
    assets: Optional[dict[str, bytes]],
    joint: str,
    expected_decay_time: float,
    *,
    prepared: bool,
) -> dict[str, object]:
    import mujoco
    import numpy as np

    m = _load_dynamics_model(mjcf_bytes, assets)
    _disable_contacts(m)
    if prepared:
        _prepare_rollout(m)
    m.opt.gravity[:] = 0.0
    coords = _joint_coordinates(m, joint)
    if coords is None:
        return {"status": "not-applicable", "reason": "joint-missing-or-unsupported"}
    _jid_value, dof, qpos, lo, hi, rom, joint_type = coords
    d = mujoco.MjData(m)
    d.qpos[qpos] = 0.5 * (lo + hi)
    initial_speed = max(0.1, 0.25 * rom)
    d.qvel[dof] = initial_speed
    mujoco.mj_forward(m, d)
    frozen = _frozen_dof_state(m, d, _jid_value)
    diagnostics = _new_dynamics_diagnostics()
    target = initial_speed * float(_DYNAMICS_POLICY["decay_target_speed_fraction"])
    upper = max(
        m.opt.timestep, expected_decay_time * float(_DYNAMICS_POLICY["decay_upper_time_ratio"])
    )
    budget = max(upper * 1.25, 0.25)
    steps = max(1, int(budget / m.opt.timestep))
    elapsed: float | None = None
    for step in range(steps):
        if _step_isolated(m, d, frozen, diagnostics):
            return _badqacc_result(diagnostics)
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
            return {"status": "advisory", "reason": "diverged"}
        if abs(float(d.qvel[dof])) <= target:
            elapsed = (step + 1) * float(m.opt.timestep)
            break
    passed = elapsed is not None and elapsed <= upper
    return {
        "status": "pass" if passed else "advisory",
        "joint_type": joint_type,
        "initial_speed": round(initial_speed, 8),
        "target_speed": round(target, 8),
        "time_to_target": None if elapsed is None else round(elapsed, 8),
        "expected_decay_time": round(expected_decay_time, 8),
        "upper_time": round(upper, 8),
        **diagnostics,
    }


def _run_decay(
    mjcf_bytes: bytes,
    assets: Optional[dict[str, bytes]],
    joint: str,
    expected_decay_time: float,
) -> dict[str, object]:
    return _retry_badqacc(
        lambda prepared: _run_decay_once(
            mjcf_bytes,
            assets,
            joint,
            expected_decay_time,
            prepared=prepared,
        )
    )


def validate_joint_dynamics(
    mjcf_bytes: bytes,
    *,
    normalized_intents: dict[str, dict[str, object]],
    resolution_evidence: list[dict[str, object]],
    cache_dir: Path | None = None,
    assets: Optional[dict[str, bytes]] = None,
) -> dict[str, object]:
    import mujoco

    artifact_sha = sha256(mjcf_bytes).hexdigest()
    cache_material = _joint_dynamics_cache_material(
        artifact_sha,
        normalized_intents=normalized_intents,
        resolution_evidence=resolution_evidence,
        mujoco_version=mujoco.__version__,
    )
    cache_key = sha256(
        json.dumps(cache_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_path = Path(cache_dir) / f"{cache_key}.json" if cache_dir is not None else None
    if cache_path is not None and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and cached.get("cache_key") == cache_key:
                cached["cache_hit"] = True
                return cached
        except (OSError, json.JSONDecodeError):
            pass

    result: dict[str, object] = {
        **cache_material,
        "cache_key": cache_key,
        "cache_hit": False,
        "status": "no-applicable-checks",
        "joints": [],
    }
    try:
        _load_dynamics_model(mjcf_bytes, assets)
    except Exception as exc:  # noqa: BLE001 - load failure is returned as evidence
        result["status"] = "load-failed"
        result["load_error"] = f"{type(exc).__name__}: {exc}"
    else:
        resolved_by_joint = {
            str(item.get("joint")): item for item in resolution_evidence if item.get("joint")
        }
        joints: list[dict[str, object]] = []
        for joint, intent in sorted(normalized_intents.items()):
            resolved = resolved_by_joint.get(joint)
            if resolved is None:
                joints.append(
                    {
                        "joint": joint,
                        "intent": intent,
                        "resolved": None,
                        "status": "fail",
                        "reason": "missing-resolution-evidence",
                    }
                )
                continue
            breakaway = _run_breakaway(
                mjcf_bytes, assets, joint, float(resolved.get("frictionloss", 0.0))
            )
            hold = _run_hold(
                mjcf_bytes,
                assets,
                joint,
                str(intent.get("hold", "free")),
                float(resolved.get("worst_gravity_pose", 0.0)),
            )
            decay = _run_decay(mjcf_bytes, assets, joint, float(resolved.get("decay_time", 1.0)))
            checks = (breakaway, hold, decay)
            if any(check.get("status") in {"fail", "not-applicable"} for check in checks):
                status = "fail"
            elif any(check.get("status") == "inconclusive" for check in checks):
                status = "inconclusive"
            elif any(check.get("status") == "advisory" for check in checks):
                status = "advisory"
            else:
                status = "pass"
            joints.append(
                {
                    "joint": joint,
                    "intent": intent,
                    "resolved": resolved,
                    "breakaway": breakaway,
                    "hold": hold,
                    "decay": decay,
                    "status": status,
                }
            )
        result["joints"] = joints
        inconclusive_findings: list[dict[str, object]] = []
        for item in joints:
            if item.get("status") != "inconclusive":
                continue
            for check_name in ("breakaway", "hold", "decay"):
                check = item.get(check_name)
                if not isinstance(check, dict) or check.get("status") != "inconclusive":
                    continue
                dof = str(check.get("offending_dof") or item.get("joint") or "unknown")
                inconclusive_findings.append(
                    {
                        "joint": str(item.get("joint") or "unknown"),
                        "check": check_name,
                        "finding_code": f"dynamics_inconclusive:{dof}",
                        "offending_dof": dof,
                        "offending_body": check.get("offending_body"),
                        "max_abs_qacc": check.get("max_abs_qacc"),
                        "max_abs_qvel": check.get("max_abs_qvel"),
                    }
                )
        result["inconclusive_findings"] = inconclusive_findings
        statuses = {str(item.get("status")) for item in joints}
        if "fail" in statuses:
            result["status"] = "fail"
        elif statuses == {"inconclusive"}:
            result["status"] = "environment-failed"
            result["environment_error"] = "all applicable dynamics checks were inconclusive"
        elif "inconclusive" in statuses:
            result["status"] = "inconclusive"
        elif "advisory" in statuses:
            result["status"] = "advisory"
        elif joints:
            result["status"] = "pass"

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _joint_dynamics_cache_material(
    mjcf_sha256: str,
    *,
    normalized_intents: dict[str, dict[str, object]],
    resolution_evidence: list[dict[str, object]],
    mujoco_version: str,
) -> dict[str, object]:
    return {
        "algorithm": _DYNAMICS_ALGORITHM,
        "mujoco_version": mujoco_version,
        "mjcf_sha256": mjcf_sha256,
        "normalized_intents": normalized_intents,
        "resolution_evidence": resolution_evidence,
        "policy": _DYNAMICS_POLICY,
    }


def joint_dynamics_cache_key(
    mjcf_bytes: bytes,
    *,
    normalized_intents: dict[str, dict[str, object]],
    resolution_evidence: list[dict[str, object]],
) -> str:
    import mujoco

    material = _joint_dynamics_cache_material(
        sha256(mjcf_bytes).hexdigest(),
        normalized_intents=normalized_intents,
        resolution_evidence=resolution_evidence,
        mujoco_version=mujoco.__version__,
    )
    return sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _driver_actuator_id(m: Any, driver: str) -> int:
    aid = _aid(m, f"act_{driver}")
    return aid if aid >= 0 else _aid(m, f"rollout_drv_{driver}")


def _contact_pair_parts(m: Any, d: Any, i: int) -> tuple[str, str]:
    import mujoco

    c = d.contact[i]
    b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[c.geom1])) or "?"
    b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[c.geom2])) or "?"
    return (b1, b2) if b1 <= b2 else (b2, b1)


def _apply_preset(m: Any, d: Any, preset: Optional[dict[str, float]]) -> None:
    if not preset:
        return
    for name, q in preset.items():
        jid = _jid(m, name)
        if jid < 0:
            continue
        d.qpos[m.jnt_qposadr[jid]] = float(q)
        aid = _driver_actuator_id(m, name)
        if aid >= 0:
            d.ctrl[aid] = float(q)


def _joint_subtree_bodies(m: Any, joint: str) -> set[str]:
    import mujoco

    jid = _jid(m, joint)
    if jid < 0:
        return set()
    root_bid = int(m.jnt_bodyid[jid])
    out: set[str] = set()
    for bid in range(int(m.nbody)):
        b = bid
        while b > 0:
            if b == root_bid:
                out.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or "")
                break
            b = int(m.body_parentid[b])
    return out


def _drive_and_measure(
    m: Any,
    cp: dict[str, Any],
    *,
    contacts: bool,
    drive_steps: int,
    ramp_frac: float,
    preset: Optional[dict[str, float]] = None,
    subtree: Optional[set[str]] = None,
    disable_passive_dynamics: bool = False,
) -> Optional[dict[str, Any]]:
    import mujoco
    import numpy as np

    driver, follower = cp["driver"], cp["follower"]
    aid = _driver_actuator_id(m, driver)
    djid, fjid = _jid(m, driver), _jid(m, follower)
    if aid < 0 or djid < 0 or fjid < 0:
        return {"missing": True}
    dadr, fadr = m.jnt_qposadr[djid], m.jnt_qposadr[fjid]
    lo, hi = float(m.actuator_ctrlrange[aid][0]), float(m.actuator_ctrlrange[aid][1])
    _boost_driver_actuator(m, aid)

    saved_flags = int(m.opt.disableflags)
    if not contacts:
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    if disable_passive_dynamics:
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SPRING)
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS)
    try:
        d = mujoco.MjData(m)
        _apply_preset(m, d, preset)
        mujoco.mj_forward(m, d)
        q_rest, f_rest = float(d.qpos[dadr]), float(d.qpos[fadr])
        far = hi if abs(hi - q_rest) >= abs(lo - q_rest) else lo
        if abs(far - q_rest) < 1e-9:
            return {"no_travel": True}
        buf = np.zeros(6)
        peak_force, peak_pair, last_pair, in_contact, n_samples = 0.0, None, None, 0, 0
        for k in range(drive_steps):
            d.ctrl[aid] = q_rest + (far - q_rest) * min(1.0, k / max(1.0, drive_steps * ramp_frac))
            mujoco.mj_step(m, d)
            if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
                return None
            if contacts and k % 25 == 0:
                n_samples += 1
                ncon = int(d.ncon)
                step_force, step_pair, step_hit = 0.0, None, False
                for ci in range(ncon):
                    pair = _contact_pair_parts(m, d, ci)
                    if subtree is not None and pair[0] not in subtree and pair[1] not in subtree:
                        continue
                    step_hit = True
                    mujoco.mj_contactForce(m, d, ci, buf)
                    fn = abs(float(buf[0]))
                    if step_pair is None or fn > step_force:
                        step_force, step_pair = fn, pair
                if step_hit:
                    in_contact += 1
                if step_pair is not None:
                    last_pair = step_pair
                if step_force > peak_force:
                    peak_force, peak_pair = step_force, step_pair
    finally:
        m.opt.disableflags = saved_flags

    return {
        "follower_travel": abs(float(d.qpos[fadr]) - f_rest),
        "driver_reached": abs(float(d.qpos[dadr]) - q_rest) / max(abs(far - q_rest), 1e-9),
        "peak_force": peak_force,
        "peak_pair": peak_pair or last_pair,
        "sustained": (in_contact / n_samples) if n_samples else 0.0,
    }


def _check_free_motion(
    m: Any, d: Any, cp: dict[str, Any], *, drive_steps: int = 4000, ramp_frac: float = 0.75
) -> dict[str, Any]:
    off = _drive_and_measure(m, cp, contacts=False, drive_steps=drive_steps, ramp_frac=ramp_frac)
    if off is None:
        return {
            "check": "free_motion",
            "status": "inconclusive",
            "detail": "diverged (nan, contacts off)",
        }
    if off.get("missing"):
        return {
            "check": "free_motion",
            "status": "inconclusive",
            "detail": "driver actuator/joint missing",
        }
    if off.get("no_travel"):
        return {"check": "free_motion", "status": "inconclusive", "detail": "driver has no travel"}
    on = _drive_and_measure(m, cp, contacts=True, drive_steps=drive_steps, ramp_frac=ramp_frac)
    if on is None:
        return {
            "check": "free_motion",
            "status": "inconclusive",
            "detail": "diverged (nan, contacts on)",
        }

    foll_off = float(off["follower_travel"])
    foll_on = float(on["follower_travel"])
    peak_force = float(on["peak_force"])
    peak_pair = on["peak_pair"]
    sustained = float(on["sustained"])

    if foll_off < _MIN_BASELINE_TRAVEL:
        return {
            "check": "free_motion",
            "status": "inconclusive",
            "detail": (
                f"coupling did not track without contacts (follower moved {foll_off:.4g}, "
                f"driver reached {off['driver_reached']:.0%}) — wiring/export, not a jam"
            ),
        }

    track = foll_on / foll_off if foll_off > 1e-9 else 0.0
    detail = (
        f"follower_track={track:.0%} (on={foll_on:.4g} off={foll_off:.4g}) "
        f"peak_nonexempt_force={peak_force:.1f}N pair={peak_pair} sustained={sustained:.0%}"
    )
    impeded = track < _JAM_TRACK and sustained >= _JAM_SUSTAINED_MIN
    plowed = peak_force >= _JAM_FORCE_FLOOR and sustained >= _JAM_SUSTAINED_MIN
    if impeded or plowed:
        return {
            "check": "free_motion",
            "status": "jam",
            "detail": detail,
            "blocking_pair": list(peak_pair) if peak_pair else None,
            "peak_force_n": round(peak_force, 2),
            "follower_track": round(track, 3),
        }
    return {
        "check": "free_motion",
        "status": "pass",
        "detail": detail,
        "follower_track": round(track, 3),
    }


def _check_joint_articulation(
    m: Any,
    joint: str,
    *,
    preset: Optional[dict[str, float]] = None,
    drive_steps: int = 4000,
    ramp_frac: float = 0.75,
) -> dict[str, Any]:
    cp = {"driver": joint, "follower": joint}
    subtree = _joint_subtree_bodies(m, joint)
    off = _drive_and_measure(
        m,
        cp,
        contacts=False,
        drive_steps=drive_steps,
        ramp_frac=ramp_frac,
        preset=preset,
        disable_passive_dynamics=True,
    )
    if off is None:
        return {"check": "articulation", "status": "inconclusive", "detail": "diverged (nan, off)"}
    if off.get("missing"):
        return {
            "check": "articulation",
            "status": "inconclusive",
            "detail": "actuator/joint missing",
        }
    if off.get("no_travel"):
        return {"check": "articulation", "status": "inconclusive", "detail": "joint has no travel"}
    on = _drive_and_measure(
        m,
        cp,
        contacts=True,
        drive_steps=drive_steps,
        ramp_frac=ramp_frac,
        preset=preset,
        subtree=subtree,
        disable_passive_dynamics=True,
    )
    if on is None:
        return {"check": "articulation", "status": "inconclusive", "detail": "diverged (nan, on)"}

    travel_off = float(off["follower_travel"])
    travel_on = float(on["follower_travel"])
    peak_force = float(on["peak_force"])
    peak_pair = on["peak_pair"]
    sustained = float(on["sustained"])
    if travel_off < _MIN_BASELINE_TRAVEL:
        return {
            "check": "articulation",
            "status": "inconclusive",
            "detail": (
                f"joint did not move without contacts (moved {travel_off:.4g}, servo reached "
                f"{off['driver_reached']:.0%}) — under-powered rig, not a jam"
            ),
        }
    track = travel_on / travel_off if travel_off > 1e-9 else 0.0
    detail = (
        f"track={track:.0%} (on={travel_on:.4g} off={travel_off:.4g}) "
        f"resisting_pair={peak_pair} force={peak_force:.1f}N sustained={sustained:.0%}"
    )
    if track < _JAM_TRACK and sustained >= _JAM_SUSTAINED_MIN:
        return {
            "check": "articulation",
            "status": "jam",
            "detail": detail,
            "blocking_pair": list(peak_pair) if peak_pair else None,
            "peak_force_n": round(peak_force, 2),
            "track": round(track, 3),
            "reached_off": round(off["driver_reached"], 3),
        }
    return {"check": "articulation", "status": "pass", "detail": detail, "track": round(track, 3)}


def _check_spring_return(m: Any, d: Any, cp: dict[str, Any]) -> dict[str, Any]:
    import mujoco

    follower, beh = cp["follower"], cp["behavior"]
    fjid = _jid(m, follower)
    if fjid < 0:
        return {
            "check": "spring_return",
            "status": "inconclusive",
            "detail": "follower joint missing",
        }
    fadr = m.jnt_qposadr[fjid]
    dofadr = m.jnt_dofadr[fjid]
    target = float(getattr(beh, "target", 0.0))
    if m.jnt_limited[fjid]:
        flo, fhi = float(m.jnt_range[fjid][0]), float(m.jnt_range[fjid][1])
    else:
        flo, fhi = target - 1.0, target + 1.0
    start = flo if abs(fhi - target) < abs(flo - target) else fhi
    if abs(start - target) < 1e-9:
        return {
            "check": "spring_return",
            "status": "inconclusive",
            "detail": "no displacement room",
        }
    d.qpos[fadr] = start
    mujoco.mj_forward(m, d)
    st = _settle(m, d, seconds=4.0, qvel_tol=float("inf"))
    if st == "nan":
        return {"check": "spring_return", "status": "inconclusive", "detail": "diverged"}
    end = float(d.qpos[fadr])
    moved = (abs(start - target) - abs(end - target)) / max(abs(start - target), 1e-6)
    detail = f"start={start:.3f} end={end:.3f} target={target:.3f} moved={moved:.0%}"
    if cp["kind"] == "latch_release":
        detail += "; trigger not representable in MJCF — spring-release only"
    if moved > 0.5:
        return {"check": "spring_return", "status": "pass", "detail": detail}
    if abs(float(d.qvel[dofadr])) >= 5e-3:
        return {
            "check": "spring_return",
            "status": "inconclusive",
            "detail": detail + " (still moving)",
        }
    return {"check": "spring_return", "status": "fail", "detail": detail}


def _check_momentary(m: Any, d: Any, cp: dict[str, Any]) -> dict[str, Any]:
    import mujoco

    follower, beh = cp["follower"], cp["behavior"]
    aid, fjid = _aid(m, f"act_{follower}"), _jid(m, follower)
    if aid < 0 or fjid < 0:
        return {"check": "momentary", "status": "inconclusive", "detail": "actuator/joint missing"}
    fadr, dofadr = m.jnt_qposadr[fjid], m.jnt_dofadr[fjid]
    direction = float(getattr(beh, "direction", 1.0) or 1.0)
    speed = float(getattr(beh, "speed", 1.0) or 1.0)
    mujoco.mj_forward(m, d)
    q0 = float(d.qpos[fadr])
    d.ctrl[aid] = direction * speed
    if _settle(m, d, seconds=0.5, qvel_tol=float("inf")) == "nan":
        return {"check": "momentary", "status": "inconclusive", "detail": "diverged"}
    moved = abs(float(d.qpos[fadr]) - q0)
    v_release = abs(float(d.qvel[dofadr]))
    d.ctrl[aid] = 0.0
    _settle(m, d, seconds=1.0, qvel_tol=float("inf"))
    qv = abs(float(d.qvel[dofadr]))
    decay_tol = max(1e-2, 0.05 * abs(speed))
    decayed = qv < decay_tol or qv < 0.5 * v_release
    if moved <= 1e-3:
        status, why = "fail", "did not move under drive"
    elif not decayed:
        status, why = "fail", "did not stop after release (runaway velocity)"
    else:
        status, why = "pass", "moved and decayed"
    return {
        "check": "momentary",
        "status": status,
        "detail": (f"moved={moved:.4f} v_release={v_release:.4f} residual_qvel={qv:.4f} ({why})"),
    }


def _run_one(m: Any, cp: dict[str, Any]) -> dict[str, Any]:
    import mujoco

    d = mujoco.MjData(m)
    try:
        if cp["kind"] in ("proportional", "nonlinear"):
            res = _check_free_motion(m, d, cp)
        elif cp["kind"] in ("latch_release", "self_closer"):
            res = _check_spring_return(m, d, cp)
        elif cp["kind"] == "momentary":
            res = _check_momentary(m, d, cp)
        else:
            res = {"check": "unknown", "status": "inconclusive", "detail": f"kind {cp['kind']!r}"}
    except Exception as exc:  # noqa: BLE001 — one coupling's failure never blocks others
        res = {
            "check": cp.get("kind", "?"),
            "status": "inconclusive",
            "detail": f"error:{type(exc).__name__}",
        }
    return {
        "coupling": {"driver": cp["driver"], "follower": cp["follower"], "kind": cp["kind"]},
        **res,
    }


def _inject_driver_actuators(xml: str, drivers: list[str]) -> str:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml)
    have = {
        a.get("joint")
        for a in root.iter()
        if a.tag in ("position", "velocity", "motor") and a.get("joint")
    }
    joints = {j.get("name"): j for j in root.iter("joint") if j.get("name")}

    def _ctrlrange(drv: str) -> str:
        joint = joints[drv]
        baked = joint.get("range")
        if baked:
            return baked
        return "0 0.1" if joint.get("type") == "slide" else "0 6.283185307"

    need = [drv for drv in drivers if drv not in have and drv in joints]
    if not need:
        return xml
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    for drv in need:
        ET.SubElement(
            actuator,
            "position",
            {"name": f"rollout_drv_{drv}", "joint": drv, "kp": "200", "ctrlrange": _ctrlrange(drv)},
        )
    return ET.tostring(root, encoding="unicode")


def _stage_mjcf(
    object_model: Any,
    enriched: Any,
    asset_root: Path,
    out_dir: Path,
    declared_overlap_pairs: list[tuple[str, str]],
    *,
    drivers: Optional[list[str]] = None,
    collision_mode: str = "exempt",
) -> Path:
    from sdk._core.v0._mjcf_export import compile_object_to_mjcf_xml
    from sdk.runtime.render import _package_model_meshes

    if collision_mode == "full" and not any(
        getattr(p, "inertial", None) is not None for p in getattr(enriched, "parts", None) or []
    ):
        raise ValueError("un-enriched model staged for full-collision rollout (no inertial parts)")

    out_dir.mkdir(parents=True, exist_ok=True)
    xml = compile_object_to_mjcf_xml(
        object_model,
        compiled_model=enriched,
        asset_root=asset_root,
        declared_overlap_pairs=declared_overlap_pairs,
        collision_mode=collision_mode,
    )
    if drivers:
        xml = _inject_driver_actuators(xml, drivers)
    xml_path = out_dir / "object.xml"
    xml_path.write_text(xml, encoding="utf-8")
    _package_model_meshes(enriched, None, out_dir, "sdk")
    return xml_path


def coupling_rollout_evidence(
    model_path: Path, src: str, *, sdk_package: str = "sdk", scratch_dir: Optional[Path] = None
) -> dict[str, Any]:
    try:
        import mujoco  # noqa: F401
    except Exception:  # noqa: BLE001
        return {"coupling_rollout": [], "skipped_reason": "mujoco-import-failed"}

    from sdk.runtime.render import _compile_and_enrich
    from sdk.runtime.runtime import extract_declared_allowances

    try:
        object_model, enriched, asset_root, _notes = _compile_and_enrich(model_path, sdk_package)
    except Exception as exc:  # noqa: BLE001
        return {"coupling_rollout": [], "skipped_reason": f"enrich-error:{type(exc).__name__}"}
    if object_model is None or enriched is None:
        return {"coupling_rollout": [], "skipped_reason": "compile-enrich-failed"}

    couplings = _enumerate_couplings(object_model)
    if not couplings:
        return {"coupling_rollout": [], "skipped_reason": "no-couplings"}

    declared = [
        (al["a"], al["b"])
        for al in extract_declared_allowances(src)
        if al.get("kind") == "overlap" and al.get("a") and al.get("b")
    ]
    mimic_drivers = sorted(
        {cp["driver"] for cp in couplings if cp["kind"] in ("proportional", "nonlinear")}
    )
    staged = Path(scratch_dir) if scratch_dir else Path(tempfile.mkdtemp(prefix="rollout_"))
    try:
        import mujoco

        xml_path = _stage_mjcf(
            object_model, enriched, asset_root, staged, declared, drivers=mimic_drivers
        )
        m = mujoco.MjModel.from_xml_path(str(xml_path))
        _prepare_rollout(m)
    except Exception as exc:  # noqa: BLE001
        return {"coupling_rollout": [], "skipped_reason": f"stage/load-error:{type(exc).__name__}"}

    return {"coupling_rollout": [_run_one(m, cp) for cp in couplings], "skipped_reason": None}


def full_collision_articulation_evidence(
    model_path: Path,
    src: str,
    *,
    sdk_package: str = "sdk",
    scratch_dir: Optional[Path] = None,
    blueprint: Optional[dict] = None,
    cache_path: Optional[Path] = None,
) -> dict[str, Any]:
    import hashlib

    from sdk.runtime.runtime import _access_gate_signature

    cache_key = hashlib.sha1(
        (src + "|" + str(_access_gate_signature(blueprint))).encode("utf-8")
    ).hexdigest()
    if cache_path is not None:
        try:
            cached = json.loads(Path(cache_path).read_text(encoding="utf-8"))
            if isinstance(cached, dict) and cached.get("key") == cache_key:
                ev = cached.get("evidence")
                if isinstance(ev, dict):
                    return {
                        "articulation_rollout": list(ev.get("articulation_rollout", [])),
                        "status": ev.get("status", "inconclusive"),
                        "skipped_reason": ev.get("skipped_reason"),
                    }
        except Exception:  # noqa: BLE001 — missing / stale / corrupt cache: recompute
            pass

    result = _compute_articulation_evidence(
        model_path, src, sdk_package=sdk_package, scratch_dir=scratch_dir, blueprint=blueprint
    )
    if cache_path is not None:
        try:
            cp = Path(cache_path)
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps({"key": cache_key, "evidence": result}), encoding="utf-8")
        except Exception:  # noqa: BLE001 — caching is an optimization, never fatal
            pass
    return result


def _compute_articulation_evidence(
    model_path: Path,
    src: str,
    *,
    sdk_package: str,
    scratch_dir: Optional[Path],
    blueprint: Optional[dict],
) -> dict[str, Any]:
    try:
        import mujoco  # noqa: F401
    except Exception:  # noqa: BLE001
        return {
            "articulation_rollout": [],
            "status": "skipped",
            "skipped_reason": "mujoco-import-failed",
        }

    from sdk.runtime.render import _compile_and_enrich
    from sdk.runtime.runtime import _resolve_access_gate_map, extract_declared_allowances

    try:
        object_model, enriched, asset_root, _notes = _compile_and_enrich(model_path, sdk_package)
    except Exception as exc:  # noqa: BLE001
        return {
            "articulation_rollout": [],
            "status": "skipped",
            "skipped_reason": f"enrich-error:{type(exc).__name__}",
        }
    if object_model is None or enriched is None:
        return {
            "articulation_rollout": [],
            "status": "skipped",
            "skipped_reason": "compile-enrich-failed",
        }

    joints = _enumerate_articulation_joints(object_model)
    if not joints:
        return {"articulation_rollout": [], "status": "no-joints", "skipped_reason": None}

    gate_map, _unresolved = _resolve_access_gate_map(blueprint, object_model)
    declared = [
        (al["a"], al["b"])
        for al in extract_declared_allowances(src)
        if al.get("kind") == "overlap" and al.get("a") and al.get("b")
    ]
    staged = Path(scratch_dir) if scratch_dir else Path(tempfile.mkdtemp(prefix="articulation_"))
    try:
        import mujoco

        xml_path = _stage_mjcf(
            object_model,
            enriched,
            asset_root,
            staged,
            declared,
            drivers=joints,
            collision_mode="full",
        )
        m = mujoco.MjModel.from_xml_path(str(xml_path))
        _prepare_rollout(m)
    except ValueError as exc:
        return {"articulation_rollout": [], "status": "skipped", "skipped_reason": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {
            "articulation_rollout": [],
            "status": "skipped",
            "skipped_reason": f"stage/load-error:{type(exc).__name__}",
        }

    rollout: list[dict[str, Any]] = []
    for j in joints:
        preset = _gate_open_preset(m, gate_map.get(j, []))
        res = _check_joint_articulation(m, j, preset=preset)
        rollout.append({"joint": j, "gate_pose": preset or None, **res})
    return {
        "articulation_rollout": rollout,
        "status": _aggregate_rollout_status(rollout),
        "skipped_reason": None,
    }


def _aggregate_rollout_status(results: list[dict[str, Any]]) -> str:
    statuses = {r.get("status") for r in results}
    if "fail" in statuses or "jam" in statuses:
        return "fail"
    if "pass" in statuses:
        return "ok"
    return "inconclusive"
