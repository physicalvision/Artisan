from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pipeline.backends.base import CompileOutcome
from pipeline.operation_timing import operation_span, timed_operation

if TYPE_CHECKING:
    from sdk._core.v0.motion_engine.resolve import ResolvedCollisionManifest

DEFAULT_TARGET = "full"
COMPILE_FINDINGS_SCHEMA_VERSION = "compile-findings-v1"


def _representative_open_poses(
    sampled_poses: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    families: dict[tuple[str, ...], list[dict[str, float]]] = {}
    for pose in sampled_poses:
        if not pose:
            continue
        normalized = {str(name): float(value) for name, value in pose.items()}
        families.setdefault(tuple(sorted(normalized)), []).append(normalized)
    representative = {
        f"{'_'.join(names)}_representative_open": max(
            poses,
            key=lambda pose: sum(abs(value) for value in pose.values()),
        )
        for names, poses in sorted(families.items())
        if poses
    }
    return representative


def _blueprint_access_gate_map(blueprint: dict[str, Any]) -> dict[str, list[str]]:
    joints = blueprint.get("joints") or []
    joint_by_child = {
        str(joint.get("child")): str(joint.get("name"))
        for joint in joints
        if isinstance(joint, dict) and joint.get("child") and joint.get("name")
    }
    gate_map: dict[str, list[str]] = {}
    for part in blueprint.get("parts") or []:
        if not isinstance(part, dict) or not isinstance(part.get("motion_gate"), dict):
            continue
        gated_joint = joint_by_child.get(str(part.get("name")))
        if not gated_joint:
            continue
        gate_joints = [
            joint_by_child[str(name)]
            for name in part["motion_gate"].get("gates") or []
            if str(name) in joint_by_child
        ]
        if gate_joints:
            if gated_joint in gate_joints:
                raise ValueError(
                    f"target joint {gated_joint!r} cannot appear in its own gate combo"
                )
            gate_map[gated_joint] = gate_joints
    return gate_map


def _wrap_free_root_findings(findings: list[object]) -> list[dict[str, Any]]:
    wrapped: list[dict[str, Any]] = []
    for raw in findings:
        finding = (
            dict(raw)
            if isinstance(raw, dict)
            else {"code": "malformed_finding", "detail": str(raw)}
        )
        finding["kind"] = "free_root_stability"
        finding["rig"] = "free_root"
        wrapped.append(finding)
    return wrapped


def _dynamics_intent_advisory(
    normalized_intents: dict[str, Any], blueprint: dict[str, Any]
) -> dict[str, Any] | None:
    from sdk._core.v0.types import MotionFeel

    failures: list[str] = []
    mismatched_joints: set[str] = set()
    for raw_joint in blueprint.get("joints") or []:
        if not isinstance(raw_joint, dict) or not isinstance(
            raw_joint.get("dynamics_intent"), dict
        ):
            continue
        name = str(raw_joint.get("name") or "").strip()
        if not name:
            continue
        raw_intent = raw_joint["dynamics_intent"]
        actual = normalized_intents.get(name)
        if not isinstance(actual, dict):
            mismatched_joints.add(name)
            failures.append(f"joint {name!r}: normalized MotionFeel is missing")
            continue
        try:
            expected = MotionFeel.from_intent(
                effort=raw_intent.get("effort"),
                motion=raw_intent.get("motion"),
                hold=raw_intent.get("hold"),
                note=str(raw_intent.get("feel", "")),
            )
        except (TypeError, ValueError) as exc:
            mismatched_joints.add(name)
            failures.append(f"joint {name!r}: invalid dynamics_intent: {exc}")
            continue
        for field_name in ("effort", "motion", "hold"):
            actual_value = actual.get(field_name)
            expected_value = getattr(expected, field_name)
            if actual_value != expected_value:
                mismatched_joints.add(name)
                failures.append(
                    f"joint {name!r}: motion_feel.{field_name}={actual_value!r} != "
                    f"normalized blueprint value {expected_value!r}"
                )
    if not failures:
        return None
    return {
        "kind": "joint_dynamics",
        "code": "dynamics_intent_mismatch",
        "rig": "fixed_root",
        "joints": sorted(mismatched_joints),
        "evidence": failures,
        "detail": (
            "the optional Blueprint dynamics_intent and built MotionFeel differ; Review should "
            "judge whether the built behavior is reasonable for the requested object"
        ),
    }


def _candidate_collision_contract_status(staged: Any) -> dict[str, Any]:
    from sdk._core.v0.motion_engine.contract import (
        ContractUnsupported,
        contract_digest,
        contract_from_mjcf,
        first_contract_difference,
    )

    expected_contract = dict(staged.manifest.simulator_collision_contract)
    expected_digest = str(staged.manifest.simulator_collision_digest)
    parsed: dict[str, dict[str, object]] = {}
    digests: dict[str, str | None] = {"manifest": expected_digest}
    unsupported: ContractUnsupported | None = None
    for label, path, strip_scene in (
        ("object.xml", Path(staged.root) / "object.xml", False),
        ("object_scene.xml", Path(staged.root) / "object_scene.xml", True),
    ):
        try:
            contract = contract_from_mjcf(path, strip_scene_rig=strip_scene)
        except ContractUnsupported as exc:
            digests[label] = None
            unsupported = unsupported or exc
        else:
            parsed[label] = contract
            digests[label] = contract_digest(contract)
    if unsupported is not None:
        return {
            "state": "contract_unsupported",
            "envelope": {
                "expected_digest": expected_digest,
                "digests": digests,
                **unsupported.to_dict(),
            },
        }
    fixed_contract = parsed["object.xml"]
    scene_contract = parsed["object_scene.xml"]
    if any(value != expected_digest for key, value in digests.items() if key != "manifest"):
        first_difference = first_contract_difference(expected_contract, fixed_contract)
        if first_difference is None:
            first_difference = first_contract_difference(expected_contract, scene_contract)
        return {
            "state": "contract_mismatch",
            "envelope": {
                "digests": digests,
                "first_structural_difference": first_difference or "digest mismatch",
            },
        }
    return {"state": "ok", "envelope": {"digests": digests}}


def validate_staged_release_candidate(
    staged: Any,
    blueprint: dict[str, Any],
    *,
    accepted_fcl_poses: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    from pipeline.contracts import validate_support_mode
    from sdk.runtime.release_candidate import (
        VALIDATION_ALGORITHM_VERSION,
        hash_accepted_fcl_poses,
        normalize_accepted_fcl_poses,
        sha256_file,
    )
    from sdk.runtime.scene_validation import validate_free_root_scene
    from sdk.runtime.sim_smoke import kinematic_penetration_sweep, validate_joint_dynamics

    fixed_path = Path(staged.root) / "object.xml"
    scene_path = Path(staged.root) / "object_scene.xml"
    normalized_fcl_poses = normalize_accepted_fcl_poses(accepted_fcl_poses)
    pose_sha256 = hash_accepted_fcl_poses(normalized_fcl_poses)
    if pose_sha256 != staged.manifest.accepted_fcl_pose_sha256:
        raise ValueError("accepted FCL pose grid does not match staged candidate identity")
    collision_contract_status = _candidate_collision_contract_status(staged)
    if collision_contract_status["state"] != "ok":
        return {
            "candidate_id": staged.manifest.candidate_id,
            "build_id": staged.manifest.build_id,
            "model_sha256": staged.manifest.model_sha256,
            "object_mjcf_sha256": sha256_file(fixed_path),
            "scene_mjcf_sha256": sha256_file(scene_path),
            "algorithm_version": VALIDATION_ALGORITHM_VERSION,
            "validation_policy_sha256": staged.manifest.validation_policy_sha256,
            "accepted_fcl_pose_sha256": staged.manifest.accepted_fcl_pose_sha256,
            "accepted_fcl_poses": normalized_fcl_poses,
            "fixed_root_pose_source": "accepted-fcl",
            "simulator_collision_digest": staged.manifest.simulator_collision_digest,
            "collision_resolution_request_key": staged.manifest.collision_resolution_request_key,
            "mujoco_version": staged.manifest.mujoco_version,
            "status": "hard-failure",
            "candidate_validation_status": collision_contract_status,
            "fixed_root": {"status": "not-run", "blocking_findings": []},
            "free_root": {"status": "not-run", "blocking_findings": []},
            "joint_dynamics": {"status": "not-run", "joints": []},
            "physics_candidate": staged.export_report.get("physics_candidate"),
            "blocking_findings": [],
            "advisory_findings": [],
            "warnings": [],
        }
    fixed = kinematic_penetration_sweep(
        fixed_path,
        poses=normalized_fcl_poses if accepted_fcl_poses is not None else None,
        gate_map=_blueprint_access_gate_map(blueprint),
    )
    representative = _representative_open_poses(list(fixed.get("completed_simulated_poses") or []))
    try:
        support_mode = validate_support_mode(blueprint.get("asset_brief"))
        free = validate_free_root_scene(
            scene_path,
            support_mode=support_mode,
            representative_poses=(representative if fixed.get("status") == "pass" else None),
        )
        free["scene_path"] = "object_scene.xml"
    except Exception as exc:  # noqa: BLE001 - explicit infrastructure/contract failure
        free = {
            "status": "environment-failed",
            "blocking_findings": [],
            "advisory_findings": [],
            "error": f"free-root validation failed: {type(exc).__name__}: {exc}",
        }

    declared_dynamics = {
        str(joint.get("name"))
        for joint in (blueprint.get("joints") or [])
        if isinstance(joint, dict)
        and joint.get("name")
        and isinstance(joint.get("dynamics_intent"), dict)
    }
    normalized_intents: dict[str, Any] = {}
    candidate_summary = staged.export_report.get("physics_candidate")
    if not isinstance(candidate_summary, dict):
        dynamics: dict[str, Any] = {
            "status": "load-failed",
            "joints": [],
            "load_error": "staged export did not preserve PhysicsCandidate evidence",
        }
    else:
        normalized_intents = dict(candidate_summary.get("normalized_intents") or {})
        resolution_evidence = list(candidate_summary.get("resolution_evidence") or [])
        mesh_root = Path(staged.root) / "assets" / "meshes"
        assets = (
            {
                path.relative_to(mesh_root).as_posix(): path.read_bytes()
                for path in mesh_root.rglob("*")
                if path.is_file()
            }
            if mesh_root.is_dir()
            else {}
        )
        try:
            dynamics = validate_joint_dynamics(
                fixed_path.read_bytes(),
                normalized_intents=normalized_intents,
                resolution_evidence=resolution_evidence,
                assets=assets,
            )
        except Exception as exc:  # noqa: BLE001 - required exact-artifact validation
            dynamics = {
                "status": "load-failed",
                "joints": [],
                "load_error": f"{type(exc).__name__}: {exc}",
            }

    findings: list[dict[str, Any]] = []
    findings.extend(dict(value) for value in fixed.get("blocking_findings", []))
    findings.extend(_wrap_free_root_findings(list(free.get("blocking_findings", []))))
    advisory_findings = _wrap_free_root_findings(list(free.get("advisory_findings", [])))
    if isinstance(candidate_summary, dict):
        intent_advisory = _dynamics_intent_advisory(normalized_intents, blueprint)
        if intent_advisory is not None:
            advisory_findings.append(intent_advisory)
    explicit_undeclared_joints = sorted(
        name
        for name, intent in normalized_intents.items()
        if name not in declared_dynamics
        and isinstance(intent, dict)
        and intent.get("source") == "motion_feel"
    )
    synthesized_fallback_joints = {
        name
        for name, intent in normalized_intents.items()
        if name not in declared_dynamics
        and isinstance(intent, dict)
        and intent.get("source") == "synthesized_default"
    }
    undeclared_motion_feel_finding: dict[str, Any] | None = None
    if explicit_undeclared_joints:
        undeclared_motion_feel_finding = {
            "kind": "joint_dynamics",
            "code": "undeclared_motion_feel",
            "rig": "fixed_root",
            "joints": explicit_undeclared_joints,
            "failed_validation_joints": [],
            "detail": (
                "the built model explicitly declares MotionFeel where the optional blueprint "
                "dynamics_intent is absent; contacts-off validation is diagnostic only, so "
                "Review should judge whether that authored feel is reasonable for the requested "
                "object"
            ),
        }
        advisory_findings.append(undeclared_motion_feel_finding)
    dynamics_evidence_invalid = False
    if dynamics.get("status") == "fail":
        failed_items = [
            item
            for item in dynamics.get("joints", [])
            if isinstance(item, dict) and item.get("status") == "fail"
        ]
        failed_joints = [str(item.get("joint")) for item in failed_items if item.get("joint")]
        unexecuted_joints = {
            str(item.get("joint"))
            for item in failed_items
            if item.get("joint")
            and (
                item.get("reason") == "missing-resolution-evidence"
                or any(
                    isinstance(item.get(check_name), dict)
                    and item[check_name].get("status") == "not-applicable"
                    for check_name in ("breakaway", "hold", "decay")
                )
            )
        }
        failed_joint_set = set(failed_joints) - unexecuted_joints
        declared_advisory_joints = sorted(failed_joint_set & declared_dynamics)
        fallback_advisory_joints = sorted(failed_joint_set & synthesized_fallback_joints)
        explicit_advisory_joints = sorted(failed_joint_set & set(explicit_undeclared_joints))
        generic_advisory_joints = sorted(
            failed_joint_set
            - declared_dynamics
            - synthesized_fallback_joints
            - set(explicit_undeclared_joints)
        )
        if fallback_advisory_joints:
            advisory_findings.append(
                {
                    "kind": "joint_dynamics",
                    "code": "passive_fallback_threshold",
                    "rig": "fixed_root",
                    "joints": fallback_advisory_joints,
                    "detail": (
                        "contacts-off passive fallback exceeded one or more nominal thresholds; "
                        "this isolates intrinsic joint response and is not assembled-mechanism "
                        "acceptance, so Review should judge the behavior against intended use"
                    ),
                }
            )
        if declared_advisory_joints:
            advisory_findings.append(
                {
                    "kind": "joint_dynamics",
                    "code": "declared_dynamics_threshold",
                    "rig": "fixed_root",
                    "joints": declared_advisory_joints,
                    "detail": (
                        "contacts-off breakaway, hold, or decay exceeded a nominal threshold; "
                        "contact friction, guide forces, and mechanical retainers are absent, so "
                        "Review should treat this as diagnostic evidence rather than a P4 verdict"
                    ),
                }
            )
        if undeclared_motion_feel_finding is not None and explicit_advisory_joints:
            undeclared_motion_feel_finding["failed_validation_joints"] = explicit_advisory_joints
        if generic_advisory_joints:
            advisory_findings.append(
                {
                    "kind": "joint_dynamics",
                    "code": "contact_disabled_dynamics_threshold",
                    "rig": "fixed_root",
                    "joints": generic_advisory_joints,
                    "detail": (
                        "contacts-off breakaway, hold, or decay exceeded a nominal threshold; "
                        "the source classification is not recognized, but the complete numeric "
                        "result remains Review evidence rather than a P4 verdict"
                    ),
                }
            )
        malformed_failure = (
            not failed_items or len(failed_joints) != len(failed_items) or bool(unexecuted_joints)
        )
        if malformed_failure:
            dynamics_evidence_invalid = True
        elif (
            declared_advisory_joints
            or fallback_advisory_joints
            or explicit_advisory_joints
            or generic_advisory_joints
        ):
            dynamics = {
                **dynamics,
                "raw_status": "fail",
                "status": "advisory",
                "enforcement": "review_recommended",
            }
    environment_failed = (
        any(result.get("status") == "environment-failed" for result in (fixed, free))
        or dynamics.get("status")
        in {
            "load-failed",
            "environment-failed",
        }
        or dynamics_evidence_invalid
    )
    if dynamics.get("status") in {"load-failed", "environment-failed"}:
        findings.append(
            {
                "kind": "validation_environment",
                "rig": "joint_dynamics",
                "detail": str(
                    dynamics.get("load_error")
                    or dynamics.get("environment_error")
                    or "required validation did not run"
                ),
            }
        )
    if dynamics_evidence_invalid:
        findings.append(
            {
                "kind": "validation_environment",
                "rig": "joint_dynamics",
                "detail": "joint-dynamics validation was incomplete or returned malformed evidence",
            }
        )
    if environment_failed:
        for rig, result in (("fixed_root", fixed), ("free_root", free)):
            if result.get("status") == "environment-failed":
                findings.append(
                    {
                        "kind": "validation_environment",
                        "rig": rig,
                        "detail": str(result.get("error") or "required validation did not run"),
                    }
                )
    status = "environment-failed" if environment_failed else ("fail" if findings else "pass")
    return {
        "candidate_id": staged.manifest.candidate_id,
        "build_id": staged.manifest.build_id,
        "model_sha256": staged.manifest.model_sha256,
        "object_mjcf_sha256": sha256_file(fixed_path),
        "scene_mjcf_sha256": sha256_file(scene_path),
        "algorithm_version": VALIDATION_ALGORITHM_VERSION,
        "validation_policy_sha256": staged.manifest.validation_policy_sha256,
        "accepted_fcl_pose_sha256": staged.manifest.accepted_fcl_pose_sha256,
        "accepted_fcl_poses": normalized_fcl_poses,
        "fixed_root_pose_source": (
            "accepted-fcl" if accepted_fcl_poses is not None else "standalone-generated"
        ),
        "mujoco_version": staged.manifest.mujoco_version,
        "simulator_collision_digest": staged.manifest.simulator_collision_digest,
        "collision_resolution_request_key": staged.manifest.collision_resolution_request_key,
        "status": status,
        "candidate_validation_status": collision_contract_status,
        "fixed_root": fixed,
        "free_root": free,
        "joint_dynamics": dynamics,
        "physics_candidate": candidate_summary,
        "blocking_findings": findings,
        "advisory_findings": advisory_findings,
        "warnings": (
            list(candidate_summary.get("warnings") or [])
            if isinstance(candidate_summary, dict)
            else []
        ),
    }


def build_validated_release_candidate(
    model_path: Path,
    blueprint_path: Path,
    candidate_dir: Path,
    physics_report_path: Path,
    *,
    formats: list[str],
    sdk_package: str = "sdk",
    accepted_fcl_poses: list[dict[str, float]] | None = None,
    simulator_collision_digest: str | None = None,
    collision_resolution_request_key: str | None = None,
    resolved_manifest_digest: str | None = None,
) -> dict[str, Any]:
    from sdk.runtime.release_candidate import (
        accept_release_candidate,
        cached_release_candidate,
        compute_candidate_identity,
        load_manifest,
        stage_release_candidate,
    )

    model_path = Path(model_path)
    blueprint_path = Path(blueprint_path)
    candidate_dir = Path(candidate_dir)
    physics_report_path = Path(physics_report_path)
    collision_manifest = _candidate_motion_manifest(
        model_path,
        sdk_package=sdk_package,
        simulator_collision_digest=simulator_collision_digest,
        collision_resolution_request_key=collision_resolution_request_key,
        resolved_manifest_digest=resolved_manifest_digest,
    )
    identity = compute_candidate_identity(
        model_path,
        blueprint_path,
        formats=formats,
        accepted_fcl_poses=accepted_fcl_poses,
        simulator_collision_digest=collision_manifest.simulator_collision_digest,
        collision_resolution_request_key=collision_manifest.resolution_request_key,
        sdk_package=sdk_package,
    )
    cached = cached_release_candidate(candidate_dir, identity)
    if cached is not None:
        manifest = load_manifest(cached)
        embedded = cached.parent / str(manifest.physics_report_path)
        physics_report_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(embedded, physics_report_path)
        report = json.loads(embedded.read_text(encoding="utf-8"))
        return {
            "status": "pass",
            "candidate_id": manifest.candidate_id,
            "manifest_path": str(cached),
            "physics_report_path": str(physics_report_path),
            "physics_report": report,
            "cache_hit": True,
        }

    if physics_report_path.is_file():
        try:
            failed_report = json.loads(physics_report_path.read_text(encoding="utf-8"))
            expected_failure_identity = {
                "build_id": identity.build_id,
                "model_sha256": identity.model_sha256,
                "algorithm_version": identity.validation_algorithm_version,
                "validation_policy_sha256": identity.validation_policy_sha256,
                "accepted_fcl_pose_sha256": identity.accepted_fcl_pose_sha256,
                "simulator_collision_digest": identity.simulator_collision_digest,
                "collision_resolution_request_key": identity.collision_resolution_request_key,
                "mujoco_version": identity.mujoco_version,
            }
            if (
                isinstance(failed_report, dict)
                and all(
                    failed_report.get(key) == value
                    for key, value in expected_failure_identity.items()
                )
                and failed_report.get("status") == "fail"
                and bool(failed_report.get("blocking_findings"))
            ):
                return {
                    "status": "fail",
                    "candidate_id": str(failed_report.get("candidate_id") or ""),
                    "manifest_path": None,
                    "physics_report_path": str(physics_report_path),
                    "physics_report": failed_report,
                    "cache_hit": True,
                }
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    blueprint = json.loads(blueprint_path.read_text(encoding="utf-8"))
    if not isinstance(blueprint, dict):
        raise ValueError("blueprint must be a JSON object")
    staged = stage_release_candidate(
        model_path,
        blueprint_path,
        candidate_dir,
        formats=formats,
        sdk_package=sdk_package,
        accepted_fcl_poses=accepted_fcl_poses,
        simulator_collision_digest=identity.simulator_collision_digest,
        collision_resolution_request_key=identity.collision_resolution_request_key,
        collision_manifest=collision_manifest,
    )
    from sdk.runtime.candidate_validation_worker import validate_candidate_in_worker

    candidate_worker = validate_candidate_in_worker(
        staged,
        blueprint,
        accepted_fcl_poses=list(accepted_fcl_poses or []),
    )
    report = candidate_worker.report if candidate_worker.status == "ok" else None
    if report is None:
        assert candidate_worker is not None
        from sdk.runtime.release_candidate import hash_accepted_fcl_poses, sha256_file

        fixed_path = Path(staged.root) / "object.xml"
        scene_path = Path(staged.root) / "object_scene.xml"
        report = {
            "candidate_id": staged.manifest.candidate_id,
            "build_id": staged.manifest.build_id,
            "model_sha256": staged.manifest.model_sha256,
            "object_mjcf_sha256": sha256_file(fixed_path),
            "scene_mjcf_sha256": sha256_file(scene_path),
            "algorithm_version": staged.manifest.validation_algorithm_version,
            "validation_policy_sha256": staged.manifest.validation_policy_sha256,
            "accepted_fcl_pose_sha256": hash_accepted_fcl_poses(accepted_fcl_poses),
            "accepted_fcl_poses": list(accepted_fcl_poses or []),
            "fixed_root_pose_source": "accepted-fcl",
            "simulator_collision_digest": staged.manifest.simulator_collision_digest,
            "collision_resolution_request_key": staged.manifest.collision_resolution_request_key,
            "mujoco_version": staged.manifest.mujoco_version,
            "status": "environment-failed",
            "candidate_validation_status": {
                "state": candidate_worker.status,
                "envelope": candidate_worker.envelope,
            },
            "fixed_root": {"status": "not-run", "blocking_findings": []},
            "free_root": {"status": "not-run", "blocking_findings": []},
            "joint_dynamics": {"status": "not-run", "joints": []},
            "physics_candidate": staged.export_report.get("physics_candidate"),
            "blocking_findings": [
                {
                    "kind": "validation_environment",
                    "rig": "candidate",
                    "detail": candidate_worker.envelope,
                }
            ],
            "advisory_findings": [],
            "warnings": [],
        }
    report["candidate_id"] = staged.manifest.candidate_id
    report["build_id"] = staged.manifest.build_id
    if report["status"] == "fail":
        from sdk.runtime.collision_debug import build_collision_debug_bundle

        try:
            debug = build_collision_debug_bundle(
                staged.root,
                report,
                physics_report_path.parent / "collision_debug",
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics must not change the verdict
            report.setdefault("warnings", []).append(
                f"collision_debug_skip: {type(exc).__name__}: {exc}"
            )
        else:
            if debug is not None:
                report["collision_debug"] = debug
    staged_report = staged.root / ".physics_validation.tmp.json"
    staged_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    physics_report_path.parent.mkdir(parents=True, exist_ok=True)
    if report["status"] != "pass":
        shutil.copy2(staged_report, physics_report_path)
        shutil.rmtree(staged.root, ignore_errors=True)
        return {
            "status": report["status"],
            "candidate_id": staged.manifest.candidate_id,
            "manifest_path": None,
            "physics_report_path": str(physics_report_path),
            "physics_report": report,
            "cache_hit": False,
        }

    try:
        from sdk.runtime.release_candidate import verify_candidate_collision_identity

        verify_candidate_collision_identity(
            model_path,
            collision_manifest,
            simulator_collision_digest=identity.simulator_collision_digest,
            collision_resolution_request_key=identity.collision_resolution_request_key,
            sdk_package=sdk_package,
        )
        manifest_path = accept_release_candidate(staged, candidate_dir, staged_report)
    except Exception:
        shutil.rmtree(staged.root, ignore_errors=True)
        raise
    shutil.copy2(candidate_dir / "physics_validation.json", physics_report_path)
    return {
        "status": "pass",
        "candidate_id": staged.manifest.candidate_id,
        "manifest_path": str(manifest_path),
        "physics_report_path": str(physics_report_path),
        "physics_report": report,
        "cache_hit": False,
    }


def _candidate_motion_manifest(
    model_path: Path,
    *,
    sdk_package: str,
    simulator_collision_digest: str | None,
    collision_resolution_request_key: str | None,
    resolved_manifest_digest: str | None,
    required: bool = False,
) -> ResolvedCollisionManifest:
    from sdk._core.v0.motion_engine.resolve import (
        default_manifest_cache_dir,
        load_collision_manifest,
        load_or_create_collision_manifest,
    )
    from sdk.runtime.release_candidate import verify_candidate_collision_identity

    has_identity = (
        simulator_collision_digest is not None or collision_resolution_request_key is not None
    )
    manifest = (
        load_collision_manifest(default_manifest_cache_dir(), collision_resolution_request_key)
        if collision_resolution_request_key is not None
        else None
    )
    if not has_identity and not required:
        manifest = load_or_create_collision_manifest(model_path, sdk_package=sdk_package)
    verify_candidate_collision_identity(
        model_path,
        manifest,
        sdk_package=sdk_package,
        simulator_collision_digest=simulator_collision_digest,
        collision_resolution_request_key=collision_resolution_request_key,
        resolved_manifest_digest=resolved_manifest_digest,
        required=required or has_identity,
    )
    return manifest


def preview_validated_release_candidate(
    model_path: Path,
    blueprint_path: Path,
    preview_root: Path,
    *,
    formats: list[str],
    sdk_package: str = "sdk",
    accepted_fcl_poses: list[dict[str, float]] | None = None,
    simulator_collision_digest: str,
    collision_resolution_request_key: str,
    resolved_manifest_digest: str,
) -> dict[str, Any]:
    from sdk.runtime.release_candidate import stage_release_candidate

    model_path = Path(model_path)
    blueprint_path = Path(blueprint_path)
    preview_root = Path(preview_root)
    blueprint = json.loads(blueprint_path.read_text(encoding="utf-8"))
    if not isinstance(blueprint, dict):
        raise ValueError("blueprint must be a JSON object")
    collision_manifest = _candidate_motion_manifest(
        model_path,
        sdk_package=sdk_package,
        simulator_collision_digest=simulator_collision_digest,
        collision_resolution_request_key=collision_resolution_request_key,
        resolved_manifest_digest=resolved_manifest_digest,
        required=True,
    )
    preview_root.parent.mkdir(parents=True, exist_ok=True)
    transaction_root = Path(
        tempfile.mkdtemp(
            prefix=f".{preview_root.name}-",
            dir=preview_root.parent,
        )
    )
    try:
        staged = stage_release_candidate(
            model_path,
            blueprint_path,
            transaction_root / "candidate",
            formats=formats,
            sdk_package=sdk_package,
            accepted_fcl_poses=accepted_fcl_poses,
            simulator_collision_digest=simulator_collision_digest,
            collision_resolution_request_key=collision_resolution_request_key,
            collision_manifest=collision_manifest,
        )
        from sdk.runtime.candidate_validation_worker import validate_candidate_in_worker

        worker = validate_candidate_in_worker(
            staged,
            blueprint,
            accepted_fcl_poses=list(accepted_fcl_poses or []),
        )
        if worker.status != "ok" or worker.report is None:
            return {
                "status": "environment-failed",
                "physics_report": {
                    "status": "environment-failed",
                    "blocking_findings": [],
                },
            }
        report = worker.report
        return {
            "status": report.get("status"),
            "physics_report": report,
        }
    finally:
        shutil.rmtree(transaction_root, ignore_errors=True)


def _signal_line(sig: Any) -> str:
    code = getattr(sig, "code", "") or getattr(sig, "kind", "") or "SIGNAL"
    summary = getattr(sig, "summary", "") or ""
    details = getattr(sig, "details", "") or ""
    check = getattr(sig, "check_name", None)
    head = f"[{code}] {summary}".strip()
    if check:
        head = f"{head} ({check})"
    if details:
        head = f"{head} — {details}".strip()
    return head[:1000]


def _bundle_to_outcome(
    bundle: Any,
    *,
    report_path: Path | None,
    extra_warnings: list[str] | None = None,
) -> CompileOutcome:
    failures: list[str] = []
    warnings: list[str] = list(extra_warnings or [])
    for sig in getattr(bundle, "signals", ()):  # type: ignore[union-attr]
        line = _signal_line(sig)
        if getattr(sig, "severity", "warning") == "failure":
            failures.append(line)
        else:
            warnings.append(line)
    status = "success" if not failures else "failure"
    summary = (
        (getattr(bundle, "summary", "") or "compile success")
        if status == "success"
        else f"{len(failures)} blocking signal(s)"
    )
    return CompileOutcome(
        status=status,
        summary=summary,
        report_path=str(report_path) if report_path else None,
        blocking_signals=failures,
        warnings=warnings,
        raw=None,
    )


@timed_operation("compile.request")
def compile_report(
    model_path: Path,
    *,
    sdk_package: str = "sdk",
    target: str = DEFAULT_TARGET,
    include_diagnostics: bool = False,
) -> tuple[Any, Any, str | None, bool]:
    from pipeline.compiler import compile_urdf_report_maybe_timeout
    from pipeline.feedback import compile_signal_bundle_from_exception

    try:
        from pipeline.compiler import _env_float
        from sdk.runtime.performance_policy import PerformancePolicy

        compile_function = compile_urdf_report_maybe_timeout
        if (
            include_diagnostics
            and PerformancePolicy.from_env().shared_compile_diagnostics
            and _env_float("URDF_COMPILE_TIMEOUT_SECONDS", 300) > 0
        ):
            from sdk.runtime.compile_worker import compile_with_diagnostics

            compile_function = compile_with_diagnostics
        report = compile_function(Path(model_path), sdk_package=sdk_package, target=target)
        return report, report.signal_bundle, report.urdf_xml, True
    except Exception as exc:  # noqa: BLE001 — failure is an expected control path
        bundle = getattr(exc, "compile_signal_bundle", None)
        if bundle is None:
            bundle = compile_signal_bundle_from_exception(exc)
        urdf_xml = getattr(exc, "compiled_urdf_xml", None)
        if not isinstance(urdf_xml, str):
            urdf_xml = None
        return None, bundle, urdf_xml, False


def model_source_sha256(model_path: Path) -> str:
    return hashlib.sha256(model_path.read_bytes()).hexdigest()


def build_joint_dynamics_review_evidence(
    model_path: Path,
    *,
    sdk_package: str = "sdk",
    cache_path: Path | None = None,
    candidate: Any | None = None,
) -> dict[str, Any]:
    from sdk.runtime.physics_candidate import build_physics_candidate
    from sdk.runtime.sim_smoke import (
        _DYNAMICS_ALGORITHM,
        _DYNAMICS_POLICY,
        joint_dynamics_cache_key,
        validate_joint_dynamics,
    )

    model_path = Path(model_path)
    if candidate is None:
        source = model_path.read_text(encoding="utf-8")
        overlaps = [
            (item["a"], item["b"])
            for item in extract_declared_allowances(source)
            if item.get("kind") == "overlap" and item.get("a") and item.get("b")
        ]
        candidate = build_physics_candidate(
            model_path,
            sdk_package=sdk_package,
            declared_overlap_pairs=overlaps,
        )
    candidate_summary = candidate.summary()
    expected_key = joint_dynamics_cache_key(
        candidate.mjcf_bytes,
        normalized_intents=candidate.normalized_intents,
        resolution_evidence=candidate.resolution_evidence,
    )
    if cache_path is not None:
        try:
            cached = json.loads(Path(cache_path).read_text(encoding="utf-8"))
            if (
                isinstance(cached, dict)
                and cached.get("model_sha256") == candidate.model_sha256
                and cached.get("dependency_sha256") == candidate.dependency_sha256
                and cached.get("mjcf_sha256") == candidate.mjcf_sha256
                and cached.get("candidate_version") == candidate_summary["version"]
                and cached.get("mapping_version") == candidate_summary["mapping_version"]
                and cached.get("algorithm") == _DYNAMICS_ALGORITHM
                and cached.get("policy") == _DYNAMICS_POLICY
                and cached.get("cache_key") == expected_key
            ):
                cached["evidence_file_hit"] = True
                return cached
        except (OSError, json.JSONDecodeError):
            pass

    validation_cache = Path(cache_path).parent / "joint_dynamics_cache" if cache_path else None
    evidence = validate_joint_dynamics(
        candidate.mjcf_bytes,
        normalized_intents=candidate.normalized_intents,
        resolution_evidence=candidate.resolution_evidence,
        cache_dir=validation_cache,
        assets=candidate.mjcf_assets,
    )
    result = {
        **evidence,
        "model_sha256": candidate.model_sha256,
        "dependency_sha256": candidate.dependency_sha256,
        "mjcf_sha256": candidate.mjcf_sha256,
        "candidate_version": candidate_summary["version"],
        "mapping_version": candidate_summary["mapping_version"],
        "candidate_warnings": list(candidate.warnings),
        "evidence_file_hit": False,
    }
    if cache_path is not None:
        destination = Path(cache_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _persist_report(
    report_path: Path,
    model_sha256: str,
    ok: bool,
    urdf_xml: str | None,
    bundle: Any,
    *,
    policy_errors: list[str] | None = None,
) -> None:
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "ok": ok,
                    "model_sha256": model_sha256,
                    "has_urdf": bool(urdf_xml),
                    "bundle": bundle.to_dict() if hasattr(bundle, "to_dict") else None,
                    "policy_errors": list(policy_errors or []),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001 — report persistence is best-effort
        pass


def _compile_findings_payload(
    bundle: Any,
    *,
    model_sha256: str,
    compile_status: str,
    policy_errors: list[str],
) -> dict[str, Any]:
    raw_evidence = getattr(bundle, "evidence", None)
    evidence = dict(raw_evidence) if isinstance(raw_evidence, dict) else {}
    blocking_findings: list[dict[str, Any]] = []
    overlap_placement_records: list[dict[str, Any]] = []
    finding_keys: set[str] = set()
    overlap_record_keys: set[str] = set()

    def extend_unique(
        target: list[dict[str, Any]],
        seen: set[str],
        values: Any,
    ) -> None:
        if not isinstance(values, (list, tuple)):
            return
        for value in values:
            if not isinstance(value, dict):
                continue
            item = dict(value)
            identity = json.dumps(item, sort_keys=True, separators=(",", ":"), default=str)
            if identity in seen:
                continue
            seen.add(identity)
            target.append(item)

    for signal in getattr(bundle, "signals", ()):
        if getattr(signal, "severity", "warning") != "failure":
            continue
        signal_evidence = getattr(signal, "evidence", None)
        if isinstance(signal_evidence, dict):
            extend_unique(
                overlap_placement_records,
                overlap_record_keys,
                signal_evidence.get("overlap_placement_records"),
            )
            structured = signal_evidence.get("blocking_findings")
            if isinstance(structured, (list, tuple)) and structured:
                extend_unique(blocking_findings, finding_keys, structured)
                continue
        fallback = {
            "kind": str(getattr(signal, "kind", "unknown")),
            "code": str(getattr(signal, "code", "UNKNOWN")),
            "summary": str(getattr(signal, "summary", "")),
            "details": str(getattr(signal, "details", "")),
            "check_name": getattr(signal, "check_name", None),
        }
        extend_unique(blocking_findings, finding_keys, [fallback])
    extend_unique(
        blocking_findings,
        finding_keys,
        evidence.get("blocking_findings"),
    )
    for error in policy_errors:
        extend_unique(
            blocking_findings,
            finding_keys,
            [
                {
                    "kind": "declared_allowance_policy",
                    "code": "DECLARED_ALLOWANCE_POLICY",
                    "details": error,
                }
            ],
        )
    placement_records = [
        dict(item) for item in evidence.get("placement_records", ()) if isinstance(item, dict)
    ]
    return {
        "schema_version": COMPILE_FINDINGS_SCHEMA_VERSION,
        "model_sha256": model_sha256,
        "compile_status": compile_status,
        "findings": blocking_findings,
        "placement_records": placement_records,
        "overlap_placement_records": overlap_placement_records,
        "counts": {
            "blocking_findings": len(blocking_findings),
            "placement_records": len(placement_records),
            "overlap_placement_records": len(overlap_placement_records),
        },
    }


def _publish_compile_findings(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _merge_compile_failure_bundles(existing: Any, failure: Any) -> Any:
    from pipeline.models import CompileSignalBundle

    if not isinstance(existing, CompileSignalBundle):
        return failure
    signals = []
    seen_signals: set[str] = set()
    for signal in (*existing.signals, *failure.signals):
        identity = json.dumps(signal.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        if identity in seen_signals:
            continue
        seen_signals.add(identity)
        signals.append(signal)
    evidence: dict[str, Any] = {}
    for bundle in (existing, failure):
        for raw_key, value in bundle.evidence.items():
            key = str(raw_key)
            if isinstance(value, (list, tuple)):
                target = evidence.setdefault(key, [])
                if not isinstance(target, list):
                    continue
                seen_values = {
                    json.dumps(item, sort_keys=True, separators=(",", ":"), default=str)
                    for item in target
                }
                for item in value:
                    identity = json.dumps(item, sort_keys=True, separators=(",", ":"), default=str)
                    if identity in seen_values:
                        continue
                    seen_values.add(identity)
                    target.append(item)
                continue
            evidence.setdefault(key, value)
    return CompileSignalBundle(
        status="failure",
        summary=failure.summary,
        signals=tuple(signals),
        evidence=evidence,
    )


def compile_gate(
    model_path: Path,
    report_path: Path | None = None,
    policies: Any = None,
    *,
    sdk_package: str = "sdk",
    target: str = DEFAULT_TARGET,
) -> CompileOutcome:
    model_sha256 = model_source_sha256(model_path)
    report, bundle, urdf_xml, ok = compile_report(
        model_path, sdk_package=sdk_package, target=target
    )
    allowances, allowance_errors = _declared_allowance_diagnostics(
        model_path.read_text(encoding="utf-8")
    )
    del allowances
    if report_path is not None:
        _persist_report(
            report_path,
            model_sha256,
            ok and not allowance_errors,
            urdf_xml,
            bundle,
            policy_errors=allowance_errors,
        )
    extra = list(getattr(report, "warnings", []) or []) if report is not None else []
    outcome = _bundle_to_outcome(bundle, report_path=report_path, extra_warnings=extra)
    return _apply_allowance_compile_errors(outcome, allowance_errors)


def compile_for_agent(
    model_path: Path,
    report_path: Path | None = None,
    *,
    sdk_package: str = "sdk",
    target: str = DEFAULT_TARGET,
) -> tuple[CompileOutcome, str]:
    model_sha256 = model_source_sha256(model_path)
    report = None
    allowances: list[Any] = []
    allowance_errors: list[str] = []
    ok = False
    bundle = None
    try:
        report, bundle, urdf_xml, ok = compile_report(
            model_path, sdk_package=sdk_package, target=target, include_diagnostics=True
        )
        allowances, allowance_errors = _declared_allowance_diagnostics(
            model_path.read_text(encoding="utf-8")
        )
        if report_path is not None:
            _persist_report(
                report_path,
                model_sha256,
                ok and not allowance_errors,
                urdf_xml,
                bundle,
                policy_errors=allowance_errors,
            )
        extra = list(getattr(report, "warnings", []) or []) if report is not None else []
        outcome = _bundle_to_outcome(bundle, report_path=report_path, extra_warnings=extra)
        outcome = _apply_allowance_compile_errors(outcome, allowance_errors)
    except Exception as exc:  # noqa: BLE001 - publish a fresh failure sidecar for wrapper faults
        from pipeline.feedback import compile_signal_bundle_from_exception

        failure_bundle = compile_signal_bundle_from_exception(exc)
        bundle = _merge_compile_failure_bundles(bundle, failure_bundle)
        outcome = _bundle_to_outcome(bundle, report_path=report_path)
        allowances = []
        allowance_errors = []
        ok = False
    _publish_compile_findings(
        model_path.parent / "reports" / "compile_findings.json",
        _compile_findings_payload(
            bundle,
            model_sha256=model_sha256,
            compile_status=outcome.status,
            policy_errors=allowance_errors,
        ),
    )
    try:
        if ok and not getattr(bundle, "signals", None):
            text = ""
        else:
            text = render_signals(bundle)
    except Exception:  # noqa: BLE001 — fall back to the summary if rendering fails
        text = outcome.summary
    text = _inject_geometry_island_repair_guidance(text, bundle)
    if ok:
        diagnostics = getattr(report, "diagnostics", None)
        if diagnostics is not None:
            audit, topology = diagnostics.get("mass"), diagnostics.get("topology")
        else:
            audit, topology = _render_compile_diagnostic_blocks(model_path, sdk_package=sdk_package)
        if audit:
            text = f"{text}\n{audit}" if text else audit
        if topology:
            text = f"{text}\n{topology}" if text else topology
    allowance_block = _render_declared_allowances(allowances, errors=allowance_errors)
    text = f"{text}\n{allowance_block}" if text else allowance_block
    return outcome, text


_GEOMETRY_ISLAND_REPAIR_GUIDANCE = (
    "REPAIR GUIDANCE (advisory): This Part contains disconnected visual geometry. "
    "Inspect the named disconnected elements and determine which host element each one "
    "should physically touch. Author that contact as a relation first: anchor the "
    "element and its host, attach(...) them, and bind the result — or repair the "
    "existing Anchor selector, axis_map, or clearance when the relation is already "
    "there. Resize or extend the intended existing element when its geometry genuinely "
    "cannot span the gap. Correcting a hand-authored Origin(...) is the fallback, for "
    "an element with no host feature to query. Add a separate connector only when the "
    "design genuinely requires one and no existing element should span the gap. Then "
    "recompile and check inter-part overlap and motion clearance. Intra-part overlap "
    "can restore connectivity, but it is not a global clearance guarantee."
)

_GEOMETRY_ISLAND_LINE_RE = re.compile(
    r"^part=(?P<quote>['\"])(?P<part>.+?)(?P=quote) "
    r"connected=\d+/\d+ disconnected=\[(?P<elements>.*)\]$",
    re.MULTILINE,
)
_GEOMETRY_ISLAND_SUMMARY_LIMIT = 12


def _is_geometry_island_signal(signal: Any) -> bool:
    if getattr(signal, "kind", "") == "disconnected_geometry":
        return True
    check_name = str(getattr(signal, "check_name", "") or "")
    return check_name.startswith(
        (
            "warn_if_part_contains_disconnected_geometry_islands(",
            "fail_if_part_contains_disconnected_geometry_islands(",
        )
    )


def _inject_geometry_island_repair_guidance(text: str, bundle: Any) -> str:
    if "REPAIR GUIDANCE (advisory)" in text:
        return text
    island_signals = [
        signal for signal in getattr(bundle, "signals", ()) if _is_geometry_island_signal(signal)
    ]
    island_count = len(island_signals)
    if island_count == 0:
        return text
    affected_parts: set[str] = set()
    disconnected_elements: set[tuple[str, str]] = set()
    for signal in island_signals:
        for match in _GEOMETRY_ISLAND_LINE_RE.finditer(str(getattr(signal, "details", ""))):
            part = match.group("part")
            affected_parts.add(part)
            for raw_element in match.group("elements").split(","):
                element = raw_element.strip()
                if element:
                    disconnected_elements.add((part, element))
    affected_preview = [
        f"{part}:{element}"
        for part, element in sorted(disconnected_elements)[:_GEOMETRY_ISLAND_SUMMARY_LIMIT]
    ]
    omitted_elements = max(0, len(disconnected_elements) - len(affected_preview))
    affected_line = (
        "Affected disconnected elements: "
        + (", ".join(affected_preview) if affected_preview else "see failure details")
        + (f"; {omitted_elements} more in failure details" if omitted_elements else "")
        + "."
    )
    block = "\n".join(
        [
            "<geometry_island_repair_guidance>",
            f"Geometry-island signal groups: {island_count}; affected Parts: "
            f"{len(affected_parts)}; disconnected elements: {len(disconnected_elements)}.",
            affected_line,
            _GEOMETRY_ISLAND_REPAIR_GUIDANCE,
            "</geometry_island_repair_guidance>",
        ]
    )
    marker = "</summary>"
    if marker in text:
        return text.replace(marker, f"{block}\n{marker}", 1)
    return f"{block}\n{text}" if text else block


@timed_operation("compile.diagnostic_geometry")
def _compile_diagnostic_geometry(model_path: Path, *, sdk_package: str) -> tuple[Any, Path]:
    from pipeline.compiler import load_model_globals
    from sdk._core.v0.assets import activate_asset_session, asset_session_for_script
    from sdk._core.v0.exact_collisions import compile_object_model_with_exact_collisions

    asset_root = Path(model_path).resolve().parent
    with activate_asset_session(asset_session_for_script(Path(model_path))):
        model = load_model_globals(Path(model_path), sdk_package=sdk_package).get("object_model")
        if model is None:
            raise ValueError("model.py must define a top-level `object_model`")
        compiled = compile_object_model_with_exact_collisions(
            model, asset_root=asset_root, validate=False
        )
    return compiled, asset_root


def _render_compile_diagnostic_blocks(
    model_path: Path, *, sdk_package: str
) -> tuple[str | None, str | None]:
    from sdk.runtime.performance_policy import PerformancePolicy

    if not PerformancePolicy.from_env().reuse_diagnostic_geometry:
        return (
            _render_mass_audit_block(model_path, sdk_package=sdk_package),
            _render_collision_topology_block(model_path, sdk_package=sdk_package),
        )
    try:
        geometry = _compile_diagnostic_geometry(model_path, sdk_package=sdk_package)
    except Exception:  # noqa: BLE001 - an advisory cannot invalidate compilation
        return None, None
    return (
        _render_mass_audit_block(model_path, sdk_package=sdk_package, _geometry=geometry),
        _render_collision_topology_block(model_path, sdk_package=sdk_package, _geometry=geometry),
    )


@timed_operation("compile.mass_audit")
def _render_mass_audit_block(
    model_path: Path, *, sdk_package: str, _geometry: tuple[Any, Path] | None = None
) -> str | None:
    try:
        from sdk._core.v0.physics_export import mass_audit_rows

        compiled, asset_root = (
            _geometry
            if _geometry is not None
            else _compile_diagnostic_geometry(model_path, sdk_package=sdk_package)
        )
        rows = mass_audit_rows(compiled, asset_root=asset_root)
    except Exception:  # noqa: BLE001 — audit is advisory; never break compile feedback
        return None
    if not rows:
        return None
    lines = [
        "<mass_audit>",
        "Per-part mass self-check (ADVISORY, not a gate). "
        "implied_solidity = est_mass / (material_density x hull_volume):",
        "  ~1.0 you massed it ~solid; <1 hollow (normal for shells/doors/drawers);",
        "  >1 heavier than a solid block of its shell material — OK ONLY with hidden "
        "ballast/fill (motor, counterweight, battery), else you over-estimated.",
        f"  {'part':<22}{'hull_m3':>10}{'rho':>7}{'solid_kg':>10}{'est_kg':>9}{'solidity':>9}",
    ]
    for r in rows:
        est = r["est_mass_kg"]
        sol = r["implied_solidity"]
        lines.append(
            f"  {str(r['part'])[:22]:<22}{r['hull_volume_m3']:>10.4f}"
            f"{r['material_density']:>7.0f}{r['solid_mass_ceiling_kg']:>10.2f}"
            f"{(f'{est:.2f}' if est is not None else 'MISSING'):>9}"
            f"{(f'{sol:.2f}' if sol is not None else '-'):>9}"
        )
    lines.append("</mass_audit>")
    return "\n".join(lines)


@timed_operation("compile.collision_topology")
def _render_collision_topology_block(
    model_path: Path, *, sdk_package: str, _geometry: tuple[Any, Path] | None = None
) -> str | None:
    try:
        from sdk._core.v0.collision_mesh_repair import _topology_metrics
        from sdk._core.v0.inertia import geometry_to_trimesh
        from sdk._core.v0.types import Mesh

        compiled, asset_root = (
            _geometry
            if _geometry is not None
            else _compile_diagnostic_geometry(model_path, sdk_package=sdk_package)
        )
        rows: list[str] = []
        closed = 0
        for part in getattr(compiled, "parts", []) or []:
            for collision in getattr(part, "collisions", []) or []:
                geometry = getattr(collision, "geometry", None)
                if not isinstance(geometry, Mesh):
                    continue
                mesh = geometry_to_trimesh(geometry, asset_root=asset_root)
                if mesh is None:
                    continue
                metrics = _topology_metrics(mesh)
                if metrics.watertight:
                    closed += 1
                    continue
                required = geometry.collision_decomposition == "required"
                cavity = geometry.collision_cavity is not None
                expected = (
                    "bounded-repair->manifold->coacd;blocking-without-cavity-fallback"
                    if required or cavity
                    else "bounded-repair->manifold->coacd;explicit-convex-hull-fallback"
                )
                rows.append(
                    f"OPEN {part.name}/{collision.name or part.name} "
                    f"open_edges={metrics.boundary_edges} "
                    f"boundary_loops={metrics.boundary_loops} expected={expected}"
                )
    except Exception:  # noqa: BLE001 - diagnostics must not break compile
        return None
    lines = ["<collision_topology>", f"closed_meshes={closed}", *rows, "</collision_topology>"]
    return "\n".join(lines)


def render_signals(bundle: Any) -> str:
    from pipeline.feedback import render_compile_signals

    return render_compile_signals(bundle, compact_notes=True)


def load_object_model(model_path: Path, *, sdk_package: str = "sdk") -> Any:
    from pipeline.compiler import load_model_globals

    globals_dict = load_model_globals(Path(model_path), sdk_package=sdk_package)
    object_model = globals_dict.get("object_model")
    if object_model is None:
        raise ValueError("model.py must define a top-level `object_model`")
    return object_model


def _declared_allowance_diagnostics(
    src: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    out: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out, errors

    def _lit(node: ast.AST) -> tuple[bool, Any]:
        try:
            return True, ast.literal_eval(node)
        except (ValueError, SyntaxError):
            return False, None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        fn = node.func.attr
        if fn not in ("allow_overlap", "allow_isolated_part"):
            continue
        positional = [_lit(arg) for arg in node.args]
        keyword_values = {kw.arg: _lit(kw.value) for kw in node.keywords if kw.arg is not None}
        invalid = [
            f"arg {index + 1}" for index, (literal, _) in enumerate(positional) if not literal
        ]
        invalid.extend(
            f"keyword {name}" for name, (literal, _) in keyword_values.items() if not literal
        )
        if any(kw.arg is None for kw in node.keywords):
            invalid.append("**kwargs")
        if invalid:
            errors.append(
                f"line {getattr(node, 'lineno', '?')}: {fn} requires literal arguments; "
                f"non-literal {', '.join(invalid)}"
            )
            continue
        reason = keyword_values.get("reason", (True, ""))[1]
        elem_a = keyword_values.get("elem_a", (True, None))[1]
        elem_b = keyword_values.get("elem_b", (True, None))[1]
        args = [value for _, value in positional]
        if (
            fn == "allow_overlap"
            and len(args) >= 2
            and isinstance(args[0], str)
            and isinstance(args[1], str)
        ):
            out.append(
                {
                    "kind": "overlap",
                    "a": args[0],
                    "b": args[1],
                    "elem_a": elem_a if isinstance(elem_a, str) else None,
                    "elem_b": elem_b if isinstance(elem_b, str) else None,
                    "reason": reason or "",
                }
            )
        elif fn == "allow_isolated_part" and args and isinstance(args[0], str):
            out.append({"kind": "isolated", "part": args[0], "reason": reason or ""})
    return out, errors


def extract_declared_allowances(src: str) -> list[dict[str, Any]]:
    allowances, _errors = _declared_allowance_diagnostics(src)
    return allowances


def _apply_allowance_compile_errors(
    outcome: CompileOutcome,
    errors: list[str],
) -> CompileOutcome:
    if not errors:
        return outcome
    blocking = [*outcome.blocking_signals, *errors]
    return CompileOutcome(
        status="failure",
        summary=f"{len(blocking)} blocking signal(s)",
        report_path=outcome.report_path,
        blocking_signals=blocking,
        warnings=list(outcome.warnings),
        raw=outcome.raw,
    )


def _render_declared_allowances(
    allowances: list[dict[str, Any]],
    *,
    errors: list[str] | None = None,
) -> str:
    lines = ["<declared_allowances>"]
    if not allowances:
        lines.append("none")
    for item in allowances:
        if item["kind"] == "overlap":
            scope = ""
            if item.get("elem_a") or item.get("elem_b"):
                scope = f" elements=({item.get('elem_a') or '-'}, {item.get('elem_b') or '-'})"
            lines.append(
                f"overlap ({item['a']}, {item['b']}){scope} reason={(item.get('reason') or '')!r}"
            )
        else:
            lines.append(f"isolated {item['part']} reason={(item.get('reason') or '')!r}")
    for error in errors or []:
        lines.append(f"ERROR: {error}")
    lines.append("</declared_allowances>")
    return "\n".join(lines)


def _access_gate_signature(blueprint: dict | None) -> str | None:
    from pipeline.context_policy import sha1_text

    parts = (blueprint or {}).get("parts") if isinstance(blueprint, dict) else None
    decls: list[tuple[str, tuple[str, ...]]] = []
    for p in parts or []:
        if not isinstance(p, dict):
            continue
        mg = p.get("motion_gate")
        name = p.get("name")
        if isinstance(mg, dict) and isinstance(name, str):
            gates = tuple(sorted(g for g in (mg.get("gates") or []) if isinstance(g, str)))
            if gates:
                decls.append((name, gates))
    if not decls:
        return None
    decls.sort()
    return sha1_text(json.dumps(decls, sort_keys=True))


def read_gate_evidence(
    model_path: Path,
    *,
    scope: str = "all",
    blueprint_path: Path | None = None,
    motion_cache_path: Path | None = None,
    candidate_manifest_path: Path | None = None,
) -> dict[str, Any]:
    model_path = Path(model_path)
    root = model_path.parent
    blueprint_path = Path(blueprint_path or root / "blueprint.json")
    motion_cache_path = Path(motion_cache_path or root / "reports" / "motion_scan.json")
    candidate_manifest_path = Path(candidate_manifest_path or root / "candidate" / "manifest.json")
    if scope not in {"motion", "candidate", "all"}:
        return {"status": "STALE", "reason": f"unsupported scope: {scope}"}

    result: dict[str, Any] = {"scope": scope}
    statuses: list[str] = []
    if scope in {"motion", "all"}:
        reasons: list[str] = []
        envelope: dict[str, Any] = {}
        try:
            envelope = json.loads(motion_cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            reasons.append(f"motion envelope unavailable: {type(exc).__name__}")
        request_key: str | None = None
        expected_evidence_key: str | None = None
        if model_path.is_file():
            try:
                from sdk._core.v0.motion_engine.report import (
                    MOTION_EVIDENCE_SCHEMA_VERSION as V2_SCHEMA_VERSION,
                )
                from sdk._core.v0.motion_engine.report import (
                    blueprint_gate_identity,
                )
                from sdk._core.v0.motion_engine.report import (
                    motion_evidence_key as v2_evidence_key,
                )
                from sdk._core.v0.motion_engine.report import (
                    motion_request_key as v2_request_key,
                )

                blueprint = (
                    json.loads(blueprint_path.read_text(encoding="utf-8"))
                    if blueprint_path.is_file()
                    else {}
                )
                if envelope.get("schema_version") == V2_SCHEMA_VERSION:
                    from sdk._core.v0.motion_engine.pose_plan import PosePlanPolicy
                    from sdk._core.v0.motion_engine.resolve import (
                        default_manifest_cache_dir,
                        load_collision_manifest,
                        resolution_request_key,
                    )

                    resolution_key = resolution_request_key(model_path)
                    collision_manifest = load_collision_manifest(
                        default_manifest_cache_dir(), resolution_key
                    )
                    if collision_manifest is None:
                        reasons.append("resolved collision manifest unavailable")
                    else:
                        policy = PosePlanPolicy()
                        request_key = v2_request_key(
                            collision_manifest,
                            gate_identity=blueprint_gate_identity(blueprint),
                            policy=policy,
                            seed=policy.seed,
                        )
                        expected_evidence_key = v2_evidence_key(
                            request_key, collision_manifest.resolved_manifest_digest
                        )
                        if (
                            envelope.get("simulator_collision_digest")
                            != collision_manifest.simulator_collision_digest
                        ):
                            reasons.append("simulator collision digest mismatch")
            except Exception as exc:  # noqa: BLE001 - report stale with a concrete reason
                reasons.append(f"motion identity could not be computed: {type(exc).__name__}")
        else:
            reasons.append("model.py is missing")
        from sdk._core.v0.motion_engine.report import (
            MOTION_EVIDENCE_SCHEMA_VERSION as V2_SCHEMA_VERSION,
        )

        if envelope.get("schema_version") != V2_SCHEMA_VERSION:
            reasons.append("motion schema version mismatch")
        if request_key is not None and envelope.get("motion_request_key") != request_key:
            reasons.append("motion request identity mismatch")
        if (
            expected_evidence_key is not None
            and envelope.get("motion_evidence_key") != expected_evidence_key
        ):
            reasons.append("motion evidence identity mismatch")
        motion_status = "FRESH" if not reasons else "STALE"
        statuses.append(motion_status)
        result["motion"] = {
            "status": motion_status,
            "reasons": reasons,
            "motion_request_key": envelope.get("motion_request_key"),
            "motion_evidence_key": envelope.get("motion_evidence_key"),
            "requested_collision_mode": envelope.get("requested_collision_mode"),
            "actual_collision_mode": envelope.get("actual_collision_mode"),
        }

    if scope in {"candidate", "all"}:
        reasons = []
        manifest = None
        try:
            from sdk.runtime.release_candidate import (
                compute_candidate_identity,
                verify_release_candidate,
            )

            manifest = verify_release_candidate(candidate_manifest_path)
            current = compute_candidate_identity(
                model_path,
                blueprint_path,
                formats=list(manifest.formats),
                accepted_fcl_pose_sha256=manifest.accepted_fcl_pose_sha256,
            )
            if current.build_id != manifest.build_id:
                reasons.append("candidate input or policy identity mismatch")
        except Exception as exc:  # noqa: BLE001 - integrity failures are stale evidence
            reasons.append(f"candidate integrity mismatch: {type(exc).__name__}: {exc}")
        candidate_status = "FRESH" if not reasons else "STALE"
        statuses.append(candidate_status)
        result["candidate"] = {
            "status": candidate_status,
            "reasons": reasons,
            "candidate_id": getattr(manifest, "candidate_id", None),
            "physics_report_sha256": getattr(manifest, "physics_report_sha256", None),
        }

    result["status"] = (
        "FRESH" if statuses and all(item == "FRESH" for item in statuses) else "STALE"
    )
    return result


def _resolve_access_gate_map(
    blueprint: dict | None, model: object
) -> tuple[dict[str, list[str]], list[dict[str, str]]]:
    from sdk._core.v0.self_collision import _moving_joints

    parts = (blueprint or {}).get("parts") if isinstance(blueprint, dict) else None
    gate_map: dict[str, list[str]] = {}
    unresolved: list[dict[str, str]] = []
    if not parts:
        return gate_map, unresolved

    joint_by_child: dict[str, str] = {}
    for j in _moving_joints(model):
        child = getattr(j, "child", None)
        if isinstance(child, str):
            joint_by_child[child] = j.name

    for p in parts:
        if not isinstance(p, dict):
            continue
        mg = p.get("motion_gate")
        if not isinstance(mg, dict):
            continue
        gated_part = p.get("name")
        gates = [g for g in (mg.get("gates") or []) if isinstance(g, str)]
        if not isinstance(gated_part, str) or not gates:
            continue
        gated_joint = joint_by_child.get(gated_part)
        if gated_joint is None:
            unresolved.append(
                {"part": gated_part, "role": "gated", "detail": "no movable joint in built model"}
            )
            continue
        gate_joints: list[str] = []
        for g in gates:
            gj = joint_by_child.get(g)
            if gj is None:
                unresolved.append(
                    {
                        "part": g,
                        "role": "gate",
                        "detail": f"gate of '{gated_part}' has no movable joint in built model",
                    }
                )
            else:
                gate_joints.append(gj)
        if gate_joints:
            gate_map[gated_joint] = gate_joints
    return gate_map, unresolved


def build_motion_review_evidence(
    model_path: Path,
    src: str,
    *,
    seed: int = 0,
    sdk_package: str = "sdk",
    cache_path: Path | None = None,
    blueprint: dict | None = None,
) -> dict[str, Any]:
    from sdk._core.v0.motion_engine.executor import build_motion_evidence

    resolved_cache_path = (
        Path(cache_path)
        if cache_path is not None
        else (Path(model_path).parent / "reports" / "motion_scan.json")
    )
    return build_motion_evidence(
        Path(model_path),
        source=src,
        cache_path=resolved_cache_path,
        blueprint=blueprint,
        sdk_package=sdk_package,
        seed=seed,
        measure=operation_span,
    )


def build_rollout_review_evidence(
    model_path: Path, src: str, *, sdk_package: str = "sdk"
) -> dict[str, Any]:
    try:
        from sdk._dependencies import sim_extra_available
    except Exception:  # noqa: BLE001
        return {"coupling_rollout": [], "skipped_reason": "dependency-probe-unavailable"}
    if not sim_extra_available("mujoco"):
        return {"coupling_rollout": [], "skipped_reason": "mujoco-not-installed"}
    try:
        from sdk.runtime.sim_smoke import coupling_rollout_evidence
    except Exception as exc:  # noqa: BLE001 — rollout module unavailable
        return {
            "coupling_rollout": [],
            "skipped_reason": f"sim-smoke-unavailable:{type(exc).__name__}",
        }
    try:
        return coupling_rollout_evidence(model_path, src, sdk_package=sdk_package)
    except Exception as exc:  # noqa: BLE001 — rollout failure never blocks review
        return {"coupling_rollout": [], "skipped_reason": f"rollout-error:{type(exc).__name__}"}


def build_articulation_review_evidence(
    model_path: Path,
    src: str,
    *,
    sdk_package: str = "sdk",
    blueprint: dict | None = None,
    cache_path: Path | None = None,
) -> dict[str, Any]:
    try:
        from sdk._dependencies import sim_extra_available
    except Exception:  # noqa: BLE001
        return {
            "articulation_rollout": [],
            "status": "skipped",
            "skipped_reason": "dependency-probe-unavailable",
        }
    if not sim_extra_available("mujoco"):
        return {
            "articulation_rollout": [],
            "status": "skipped",
            "skipped_reason": "mujoco-not-installed",
        }
    try:
        from sdk.runtime.sim_smoke import full_collision_articulation_evidence
    except Exception as exc:  # noqa: BLE001
        return {
            "articulation_rollout": [],
            "status": "skipped",
            "skipped_reason": f"sim-smoke-unavailable:{type(exc).__name__}",
        }
    try:
        return full_collision_articulation_evidence(
            model_path, src, sdk_package=sdk_package, blueprint=blueprint, cache_path=cache_path
        )
    except Exception as exc:  # noqa: BLE001 — rollout failure never blocks the build/review
        return {
            "articulation_rollout": [],
            "status": "skipped",
            "skipped_reason": f"rollout-error:{type(exc).__name__}",
        }


def _apply_declared_allowances(ctx: Any, model_path: Path) -> None:
    try:
        src = model_path.read_text(encoding="utf-8")
    except OSError:
        return
    for al in extract_declared_allowances(src):
        try:
            if al["kind"] == "overlap":
                ctx.allow_overlap(
                    al["a"], al["b"], reason=al["reason"] or "declared in model run_tests()"
                )
            else:
                ctx.allow_isolated_part(
                    al["part"], reason=al["reason"] or "declared in model run_tests()"
                )
        except Exception:  # noqa: BLE001 — a malformed allowance must not break QA
            pass


def _warn_part_dimension_deviations(ctx: Any, model_path: Path) -> None:
    try:
        bp_path = Path(model_path).resolve().parent / "blueprint.json"
        if not bp_path.exists():
            return
        parts = (json.loads(bp_path.read_text(encoding="utf-8")) or {}).get("parts") or []
    except Exception:  # noqa: BLE001 — a missing/malformed blueprint must not break QA
        return

    HI, LO = 1.5, 0.66
    for p in parts:
        if not isinstance(p, dict):
            continue
        name = p.get("name")
        dims = p.get("dimensions")
        if not name or not isinstance(dims, dict):
            continue
        ref = [dims.get("x"), dims.get("y"), dims.get("z")]
        if any(not isinstance(r, (int, float)) or r <= 0 for r in ref):
            continue
        try:
            aabb = ctx.part_world_aabb(name)
        except Exception:  # noqa: BLE001
            aabb = None
        if not aabb:
            continue
        (mn, mx) = aabb
        built_max = max(mx[i] - mn[i] for i in range(3))
        ref_max = max(float(r) for r in ref)
        if ref_max <= 0:
            continue
        ratio = built_max / ref_max
        if ratio > HI or ratio < LO:
            ctx.warn(
                f"[size_sanity] part '{name}' largest built extent {built_max:.3f} m "
                f"vs blueprint largest reference {ref_max:.3f} m ({ratio:.1f}x) — VERIFY "
                f"this part: a builder bug OR an imprecise Design number. Non-blocking "
                f"heads-up, not a gate."
            )


def run_qa_tests(
    model_path: Path,
    test_code: str,
    *,
    sdk_package: str = "sdk",
    preserve_existing_meshes: bool = False,
) -> dict[str, Any]:
    import importlib

    from pipeline.compiler import load_model_globals
    from sdk._core.v0.assets import activate_asset_session, asset_session_for_script

    try:
        sdk_mod = importlib.import_module(sdk_package)
        test_context_cls = getattr(sdk_mod, "TestContext")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"cannot import {sdk_package}.TestContext: {exc}"}

    session = asset_session_for_script(
        Path(model_path), preserve_existing_base=preserve_existing_meshes
    )
    try:
        with activate_asset_session(session):
            globals_dict = load_model_globals(Path(model_path), sdk_package=sdk_package)
            model = globals_dict.get("object_model")
            if model is None:
                return {"ok": False, "error": "model.py defines no object_model"}
            ctx = test_context_cls(model, asset_root=Path(model_path).resolve().parent)
            _apply_declared_allowances(ctx, Path(model_path))
            _warn_part_dimension_deviations(ctx, Path(model_path))
            ns: dict[str, Any] = {}
            try:
                exec(compile(test_code, "<qa_tests>", "exec"), ns)  # noqa: S102 — sandboxed QA test
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": f"test code failed to load: {exc}"}
            qa = ns.get("qa_tests")
            if not callable(qa):
                return {"ok": False, "error": "test code must define qa_tests(ctx)"}
            try:
                qa(ctx)
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": f"qa_tests(ctx) raised: {exc}"}
            report = ctx.report()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    failures = []
    for f in getattr(report, "failures", ()) or ():
        failures.append({"name": getattr(f, "name", "?"), "details": getattr(f, "details", "")})
    return {
        "ok": bool(getattr(report, "passed", False)),
        "checks": [str(c) for c in getattr(report, "checks", ()) or ()],
        "failures": failures,
        "warnings": [str(w) for w in getattr(report, "warnings", ()) or ()],
    }


def probe_summary(
    model_path: Path,
    *,
    sdk_package: str = "sdk",
    collision_manifest: ResolvedCollisionManifest | None = None,
    preserve_existing_meshes: bool = False,
) -> dict[str, Any]:
    from sdk._core.v0.assets import activate_asset_session, asset_session_for_script

    try:
        if collision_manifest is not None:
            from sdk._core.v0.motion_engine.resolve import resolve_collision_geometry

            model = resolve_collision_geometry(
                model_path, sdk_package=sdk_package, manifest=collision_manifest
            ).raw_model
        else:
            with activate_asset_session(
                asset_session_for_script(
                    Path(model_path), preserve_existing_base=preserve_existing_meshes
                )
            ):
                model = load_object_model(model_path, sdk_package=sdk_package)
    except Exception as exc:  # noqa: BLE001
        return {"build_ok": False, "build_error": f"{type(exc).__name__}: {exc}"}

    model_parts = list(getattr(model, "parts", []) or [])
    parts = [getattr(p, "name", "?") for p in model_parts]
    visuals_by_part = {
        getattr(part, "name", "?"): [
            getattr(visual, "name", None) for visual in getattr(part, "visuals", []) or []
        ]
        for part in model_parts
    }
    joints = []
    for j in getattr(model, "articulations", []):
        jt = getattr(j, "articulation_type", None)
        limits = getattr(j, "motion_limits", None)
        mimic = getattr(j, "mimic", None)
        behavior = getattr(j, "behavior", None)
        motion_feel = getattr(j, "motion_feel", None)
        jmeta = getattr(j, "meta", None)
        cmeta = jmeta.get("coupling") if isinstance(jmeta, dict) else None
        joints.append(
            {
                "name": getattr(j, "name", "?"),
                "type": getattr(jt, "value", str(jt)),
                "parent": getattr(j, "parent", None),
                "child": getattr(j, "child", None),
                "axis": list(getattr(j, "axis", ()) or ()),
                "limit_lower": getattr(limits, "lower", None) if limits else None,
                "limit_upper": getattr(limits, "upper", None) if limits else None,
                "mimic": (
                    {
                        "joint": getattr(mimic, "joint", None),
                        "multiplier": getattr(mimic, "multiplier", None),
                        "offset": getattr(mimic, "offset", None),
                        "poly": list(getattr(mimic, "poly", ()) or ()),
                        "kind": cmeta.get("kind") if isinstance(cmeta, dict) else None,
                        "params": cmeta.get("params") if isinstance(cmeta, dict) else None,
                        "fit_error": cmeta.get("fit_error") if isinstance(cmeta, dict) else None,
                    }
                    if mimic is not None
                    else None
                ),
                "behavior": (
                    {
                        "kind": getattr(behavior, "kind", None),
                        "driver": getattr(behavior, "driver", None),
                        "target": getattr(behavior, "target", None),
                        "threshold": getattr(behavior, "threshold", None),
                        "direction": getattr(behavior, "direction", None),
                        "speed": getattr(behavior, "speed", None),
                        "strength": getattr(behavior, "strength", None),
                        "params": cmeta.get("params") if isinstance(cmeta, dict) else None,
                    }
                    if behavior is not None
                    else None
                ),
                "motion_feel": (
                    {
                        "effort": float(getattr(motion_feel, "effort", 0.0)),
                        "motion": float(getattr(motion_feel, "motion", 0.0)),
                        "hold": str(getattr(motion_feel, "hold", "free")),
                        "note": str(getattr(motion_feel, "note", "")),
                        "warnings": list(getattr(motion_feel, "warnings", ()) or ()),
                    }
                    if motion_feel is not None
                    else None
                ),
            }
        )
    return {
        "build_ok": True,
        "name": getattr(model, "name", None),
        "parts": parts,
        "visuals_by_part": visuals_by_part,
        "joints": joints,
    }
