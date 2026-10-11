from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from artisan.physics_policy import validate_penetration_depth_tolerance

if TYPE_CHECKING:
    from sdk._core.v0.motion_engine.resolve import ResolvedCollisionManifest

SCHEMA_VERSION = "release-candidate-v7"
EXPORTER_POLICY_VERSION = "sdk-export-v10"
VALIDATION_ALGORITHM_VERSION = "p4-physics-v17"


class CandidateError(RuntimeError):
    pass


class CandidateIdentityError(CandidateError):

    def __init__(self, details: dict[str, Any]) -> None:
        self.details = details
        super().__init__(json.dumps(details, sort_keys=True))


def verify_candidate_collision_identity(
    model_path: Path,
    manifest: ResolvedCollisionManifest | None,
    *,
    simulator_collision_digest: str | None,
    collision_resolution_request_key: str | None,
    sdk_package: str = "sdk",
    required: bool = False,
    resolved_manifest_digest: str | None = None,
) -> None:
    from sdk._core.v0.motion_engine.resolve import (
        canonical_bytes,
        default_manifest_cache_dir,
        frozen_input_identity,
        load_collision_manifest,
        verify_frozen_inputs,
    )

    observed_digest = manifest.simulator_collision_digest if manifest else None
    observed_key = manifest.resolution_request_key if manifest else None
    components: dict[str, Any] = {name: None for name in ("source", "geometry", "policy")}
    reasons: list[str] = []
    if (simulator_collision_digest is None) != (collision_resolution_request_key is None):
        reasons.append(
            "half identity: simulator digest and resolution key must be supplied together"
        )
    if required and (
        simulator_collision_digest is None or collision_resolution_request_key is None
    ):
        reasons.append("motion evidence requires both simulator digest and resolution key")
    if manifest is None:
        reasons.append("motion frozen build is unavailable; full motion revalidation required")
    else:
        current = frozen_input_identity(model_path, manifest, sdk_package=sdk_package)
        observed_key = hashlib.sha256(canonical_bytes(current)).hexdigest()
        reference = manifest
        if (
            collision_resolution_request_key
            and collision_resolution_request_key != manifest.resolution_request_key
        ):
            reference = load_collision_manifest(
                manifest.root.parent if manifest.root else default_manifest_cache_dir(),
                collision_resolution_request_key,
            )
        if reference is not None:
            components = {
                "source": reference.source_sha256 != current["source_sha256"],
                "geometry": reference.motion_geometry_sha256 != current["motion_geometry_sha256"],
                "policy": reference.policy_identity != current["policy_identity"],
            }
        try:
            verify_frozen_inputs(model_path, manifest, sdk_package=sdk_package)
        except (OSError, ValueError, TypeError) as exc:
            reasons.append(str(exc))
        if (
            resolved_manifest_digest is not None
            and resolved_manifest_digest != manifest.resolved_manifest_digest
        ):
            reasons.append("resolved manifest digest does not match motion evidence")
    if simulator_collision_digest is not None and simulator_collision_digest != observed_digest:
        reasons.append("simulator collision digest mismatch")
    if (
        collision_resolution_request_key is not None
        and collision_resolution_request_key != observed_key
    ):
        reasons.append("collision resolution request key mismatch")
    if reasons:
        raise CandidateIdentityError(
            {
                "kind": "candidate_identity",
                "reasons": reasons,
                "expected_simulator_collision_digest": simulator_collision_digest,
                "observed_simulator_collision_digest": observed_digest,
                "expected_resolution_request_key": collision_resolution_request_key,
                "observed_resolution_request_key": observed_key,
                "expected_resolved_manifest_digest": resolved_manifest_digest,
                "observed_resolved_manifest_digest": manifest.resolved_manifest_digest
                if manifest
                else None,
                "components": components,
            }
        )


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class CandidateIdentity:
    build_id: str
    model_sha256: str
    dependency_sha256: str
    blueprint_sha256: str
    exporter_policy_version: str
    validation_algorithm_version: str
    validation_policy_sha256: str
    accepted_fcl_pose_sha256: str
    simulator_collision_digest: str
    collision_resolution_request_key: str
    mujoco_version: str
    formats: tuple[str, ...]

    @property
    def candidate_id(self) -> str:
        return self.build_id


@dataclass(frozen=True, slots=True)
class ReleaseCandidateManifest:
    schema_version: str
    status: str
    candidate_id: str
    build_id: str
    artifact_set_sha256: str
    model_sha256: str
    dependency_sha256: str
    blueprint_sha256: str
    exporter_policy_version: str
    validation_algorithm_version: str
    validation_policy_sha256: str
    accepted_fcl_pose_sha256: str
    simulator_collision_digest: str
    collision_resolution_request_key: str
    simulator_collision_contract: dict[str, Any]
    mujoco_version: str
    formats: tuple[str, ...]
    artifacts: dict[str, ArtifactIdentity]
    physics_report_path: str | None = None
    physics_report_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["formats"] = list(self.formats)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReleaseCandidateManifest:
        artifacts = {
            str(key): ArtifactIdentity(**value)
            for key, value in dict(payload.get("artifacts") or {}).items()
        }
        return cls(
            schema_version=str(payload["schema_version"]),
            status=str(payload["status"]),
            candidate_id=str(payload["candidate_id"]),
            build_id=str(payload["build_id"]),
            artifact_set_sha256=str(payload["artifact_set_sha256"]),
            model_sha256=str(payload["model_sha256"]),
            dependency_sha256=str(payload["dependency_sha256"]),
            blueprint_sha256=str(payload["blueprint_sha256"]),
            exporter_policy_version=str(payload["exporter_policy_version"]),
            validation_algorithm_version=str(payload["validation_algorithm_version"]),
            validation_policy_sha256=str(payload["validation_policy_sha256"]),
            accepted_fcl_pose_sha256=str(payload["accepted_fcl_pose_sha256"]),
            simulator_collision_digest=str(payload["simulator_collision_digest"]),
            collision_resolution_request_key=str(payload["collision_resolution_request_key"]),
            simulator_collision_contract=dict(payload["simulator_collision_contract"]),
            mujoco_version=str(payload["mujoco_version"]),
            formats=tuple(str(value) for value in payload.get("formats") or []),
            artifacts=artifacts,
            physics_report_path=payload.get("physics_report_path"),
            physics_report_sha256=payload.get("physics_report_sha256"),
        )


@dataclass(frozen=True, slots=True)
class StagedReleaseCandidate:
    root: Path
    manifest_path: Path
    manifest: ReleaseCandidateManifest
    export_report: dict[str, Any]


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_payload(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _mujoco_version() -> str:
    try:
        import mujoco

        return str(mujoco.__version__)
    except Exception:  # noqa: BLE001 - version is part of identity, preflight gates availability
        return "unavailable"


def current_exporter_policy_version() -> str:
    from sdk._core.v0._coacd import coacd_policy_version
    from sdk._core.v0.collision_mesh_repair import repair_policy_identity
    from sdk.runtime.articulated_bake import bake_dependency_identity

    enabled = os.environ.get("ARTISAN_COLLISION_COACD", "1") != "0"
    repair_policy = _hash_payload(repair_policy_identity())[:16]
    bake_identity = bake_dependency_identity()
    return (
        f"{EXPORTER_POLICY_VERSION}+coacd-{int(enabled)}"
        f"+coacd-policy-{coacd_policy_version()}"
        f"+mesh-repair-{repair_policy}"
        f"+pbr-bake-{bake_identity['policy_version']}"
        f"+pbr-policy-{bake_identity['policy_sha256'][:16]}"
        f"+pbr-textures-{bake_identity['texture_library_sha256'][:16]}"
        f"+blender-{bake_identity['blender_binary_sha256'][:16]}"
    )


def normalize_accepted_fcl_poses(
    poses: list[dict[str, float]] | tuple[dict[str, float], ...] | None,
) -> list[dict[str, float]]:
    return [
        {str(name): float(value) for name, value in sorted(dict(pose).items())}
        for pose in (poses or [])
    ]


def hash_accepted_fcl_poses(
    poses: list[dict[str, float]] | tuple[dict[str, float], ...] | None,
) -> str:
    return _hash_payload(normalize_accepted_fcl_poses(poses))


def _validation_policy(accepted_fcl_pose_sha256: str) -> dict[str, Any]:
    from sdk._core.v0._motion_feel import MAPPING_VERSION
    from sdk._core.v0.geometry_qc import default_penetration_depth_tol_from_env
    from sdk.runtime.physics_candidate import PHYSICS_CANDIDATE_VERSION
    from sdk.runtime.scene_validation import DEFAULT_SCENE_POLICY
    from sdk.runtime.sim_smoke import _DYNAMICS_ALGORITHM, _DYNAMICS_POLICY

    return {
        "algorithm_version": VALIDATION_ALGORITHM_VERSION,
        "penetration_depth_tol_m": default_penetration_depth_tol_from_env(),
        "accepted_fcl_pose_sha256": accepted_fcl_pose_sha256,
        "physics_candidate_version": PHYSICS_CANDIDATE_VERSION,
        "motion_feel_mapping_version": MAPPING_VERSION,
        "collision_mode": "full",
        "joint_dynamics": {
            "algorithm": _DYNAMICS_ALGORITHM,
            "policy": _DYNAMICS_POLICY,
        },
        "scene": asdict(DEFAULT_SCENE_POLICY),
    }


def _candidate_identity_payload(
    *,
    model_sha256: str,
    dependency_sha256: str,
    blueprint_sha256: str,
    exporter_policy_version: str,
    validation_algorithm_version: str,
    validation_policy_sha256: str,
    accepted_fcl_pose_sha256: str,
    simulator_collision_digest: str,
    collision_resolution_request_key: str,
    mujoco_version: str,
    formats: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_sha256": model_sha256,
        "dependency_sha256": dependency_sha256,
        "blueprint_sha256": blueprint_sha256,
        "exporter_policy_version": exporter_policy_version,
        "validation_algorithm_version": validation_algorithm_version,
        "validation_policy_sha256": validation_policy_sha256,
        "accepted_fcl_pose_sha256": accepted_fcl_pose_sha256,
        "simulator_collision_digest": simulator_collision_digest,
        "collision_resolution_request_key": collision_resolution_request_key,
        "mujoco_version": mujoco_version,
        "formats": formats,
    }


def _dependency_manifest(
    model_path: Path, *, include_tasks: bool = True
) -> list[dict[str, object]]:
    root = Path(model_path).resolve().parent
    files: list[Path] = []
    for relative_root in (Path("inputs"), Path("references") / "inputs", Path("assets") / "meshes"):
        dependency_root = root / relative_root
        if dependency_root.is_dir():
            files.extend(
                path
                for path in dependency_root.rglob("*")
                if path.is_file()
                and not path.name.endswith(".cadquery.json")
                and ".cadquery_cache" not in path.parts
            )
    if include_tasks:
        tasks_path = root / "tasks.json"
        if tasks_path.is_file():
            files.append(tasks_path)
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(set(files))
    ]


def dependency_sha256_for_model(model_path: Path) -> str:
    return _hash_payload(_dependency_manifest(Path(model_path)))


def motion_geometry_sha256_for_model(model_path: Path) -> str:
    return _hash_payload(_dependency_manifest(Path(model_path), include_tasks=False))


def compute_candidate_identity(
    model_path: Path,
    blueprint_path: Path,
    *,
    formats: list[str] | tuple[str, ...],
    accepted_fcl_poses: list[dict[str, float]] | tuple[dict[str, float], ...] | None = None,
    accepted_fcl_pose_sha256: str | None = None,
    simulator_collision_digest: str | None = None,
    collision_resolution_request_key: str | None = None,
    sdk_package: str = "sdk",
) -> CandidateIdentity:
    model_path = Path(model_path)
    blueprint_path = Path(blueprint_path)
    normalized_formats = tuple(sorted(set(str(value) for value in formats)))
    blueprint_sha256 = (
        sha256_file(blueprint_path) if blueprint_path.is_file() else _hash_payload({})
    )
    mujoco_version = _mujoco_version()
    pose_sha256 = accepted_fcl_pose_sha256 or hash_accepted_fcl_poses(accepted_fcl_poses)
    validation_policy_sha256 = _hash_payload(_validation_policy(pose_sha256))
    exporter_policy_version = current_exporter_policy_version()
    if simulator_collision_digest is None or collision_resolution_request_key is None:
        from sdk._core.v0.motion_engine.resolve import load_or_create_collision_manifest

        collision_manifest = load_or_create_collision_manifest(model_path, sdk_package=sdk_package)
        verify_candidate_collision_identity(
            model_path,
            collision_manifest,
            simulator_collision_digest=simulator_collision_digest,
            collision_resolution_request_key=collision_resolution_request_key,
            sdk_package=sdk_package,
        )
        simulator_collision_digest = collision_manifest.simulator_collision_digest
        collision_resolution_request_key = collision_manifest.resolution_request_key
    assert collision_resolution_request_key is not None
    model_sha256 = sha256_file(model_path)
    dependency_sha256 = dependency_sha256_for_model(model_path)
    identity_payload = _candidate_identity_payload(
        model_sha256=model_sha256,
        dependency_sha256=dependency_sha256,
        blueprint_sha256=blueprint_sha256,
        exporter_policy_version=exporter_policy_version,
        validation_algorithm_version=VALIDATION_ALGORITHM_VERSION,
        validation_policy_sha256=validation_policy_sha256,
        accepted_fcl_pose_sha256=pose_sha256,
        simulator_collision_digest=simulator_collision_digest,
        collision_resolution_request_key=collision_resolution_request_key,
        mujoco_version=mujoco_version,
        formats=normalized_formats,
    )
    return CandidateIdentity(
        build_id=_hash_payload(identity_payload),
        model_sha256=model_sha256,
        dependency_sha256=dependency_sha256,
        blueprint_sha256=blueprint_sha256,
        exporter_policy_version=exporter_policy_version,
        validation_algorithm_version=VALIDATION_ALGORITHM_VERSION,
        validation_policy_sha256=validation_policy_sha256,
        accepted_fcl_pose_sha256=pose_sha256,
        simulator_collision_digest=simulator_collision_digest,
        collision_resolution_request_key=collision_resolution_request_key,
        mujoco_version=mujoco_version,
        formats=normalized_formats,
    )


def _artifact_identities(root: Path) -> dict[str, ArtifactIdentity]:
    artifacts: dict[str, ArtifactIdentity] = {}
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "manifest.json":
            continue
        artifacts[relative] = ArtifactIdentity(
            path=relative,
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
        )
    return artifacts


def _artifact_set_sha256(artifacts: dict[str, ArtifactIdentity]) -> str:
    return _hash_payload(
        {
            relative: {
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
            }
            for relative, artifact in sorted(artifacts.items())
            if relative != "physics_validation.json"
        }
    )


def _exact_candidate_id(*, build_id: str, artifact_set_sha256: str) -> str:
    return _hash_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "artifact_set_sha256": artifact_set_sha256,
        }
    )


def _write_manifest(path: Path, manifest: ReleaseCandidateManifest) -> None:
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")


def load_manifest(path: Path) -> ReleaseCandidateManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CandidateError(f"candidate manifest must be a JSON object: {path}")
    manifest = ReleaseCandidateManifest.from_dict(payload)
    if manifest.schema_version != SCHEMA_VERSION:
        raise CandidateError(f"unsupported candidate schema: {manifest.schema_version}")
    return manifest


def _required_export_keys(formats: tuple[str, ...]) -> set[str]:
    required = {fmt for fmt in formats if fmt in {"glb", "obj", "urdf", "mjcf"}}
    if "glb" in formats:
        required.update({"articulated_glb", "articulated_appearance"})
    if "mjcf" in formats:
        required.add("mjcf_scene")
    return required


def _required_artifact_paths(formats: tuple[str, ...]) -> set[str]:
    paths_by_format = {
        "glb": {
            "_pose_closed.glb",
            "object.articulated.glb",
            "object.articulated.appearance.json",
        },
        "obj": {"object.obj"},
        "urdf": {"object.urdf"},
        "mjcf": {"object.xml", "object_scene.xml"},
    }
    required: set[str] = set()
    for fmt in formats:
        required.update(paths_by_format.get(fmt, set()))
    return required


def _verify_physics_report_binding(
    manifest: ReleaseCandidateManifest,
    report: dict[str, Any],
) -> None:
    expected = {
        "candidate_id": manifest.candidate_id,
        "build_id": manifest.build_id,
        "model_sha256": manifest.model_sha256,
        "algorithm_version": manifest.validation_algorithm_version,
        "validation_policy_sha256": manifest.validation_policy_sha256,
        "accepted_fcl_pose_sha256": manifest.accepted_fcl_pose_sha256,
        "simulator_collision_digest": manifest.simulator_collision_digest,
        "collision_resolution_request_key": manifest.collision_resolution_request_key,
        "mujoco_version": manifest.mujoco_version,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise CandidateError(f"physics report {key} does not match candidate manifest")
    artifact_hashes = {
        "object_mjcf_sha256": manifest.artifacts.get("object.xml"),
        "scene_mjcf_sha256": manifest.artifacts.get("object_scene.xml"),
    }
    for key, artifact in artifact_hashes.items():
        if artifact is None or report.get(key) != artifact.sha256:
            raise CandidateError(f"physics report {key} does not match candidate artifact")
    accepted_poses = report.get("accepted_fcl_poses")
    if not isinstance(accepted_poses, list) or not accepted_poses:
        raise CandidateError("physics report has no accepted FCL pose grid")
    try:
        pose_sha256 = hash_accepted_fcl_poses(accepted_poses)
    except (TypeError, ValueError) as exc:
        raise CandidateError(f"physics report accepted FCL pose grid is invalid: {exc}") from exc
    if pose_sha256 != manifest.accepted_fcl_pose_sha256:
        raise CandidateError(
            "physics report accepted FCL pose grid does not match candidate manifest"
        )
    if report.get("fixed_root_pose_source") != "accepted-fcl":
        raise CandidateError("physics report fixed-root poses are not accepted-FCL evidence")
    for section_name in ("fixed_root", "free_root"):
        section = report.get(section_name)
        if not isinstance(section, dict) or section.get("status") != "pass":
            raise CandidateError(f"physics report {section_name} did not pass")
        if section.get("blocking_findings"):
            raise CandidateError(f"physics report {section_name} contains blocking findings")
    fixed_root = report["fixed_root"]
    if "penetration_depth_tol_m" in fixed_root:
        try:
            validate_penetration_depth_tolerance(fixed_root["penetration_depth_tol_m"])
        except ValueError as exc:
            raise CandidateError(f"physics report fixed_root tolerance is invalid: {exc}") from exc
    dynamics = report.get("joint_dynamics")
    if not isinstance(dynamics, dict):
        raise CandidateError("physics report joint_dynamics evidence is missing")
    if dynamics.get("status") not in {
        "pass",
        "advisory",
        "inconclusive",
        "no-applicable-checks",
    }:
        raise CandidateError("physics report joint_dynamics did not pass")
    object_artifact = manifest.artifacts.get("object.xml")
    if object_artifact is None or dynamics.get("mjcf_sha256") != object_artifact.sha256:
        raise CandidateError("physics report joint_dynamics MJCF does not match candidate artifact")
    fixed_root_input_poses = report["fixed_root"].get("accepted_input_poses")
    if not isinstance(fixed_root_input_poses, list):
        raise CandidateError("physics report fixed-root input pose grid is missing")
    try:
        normalized_fixed_root_poses = normalize_accepted_fcl_poses(fixed_root_input_poses)
    except (TypeError, ValueError) as exc:
        raise CandidateError(
            f"physics report fixed-root input pose grid is invalid: {exc}"
        ) from exc
    if normalized_fixed_root_poses != normalize_accepted_fcl_poses(accepted_poses):
        raise CandidateError(
            "physics report fixed-root input pose grid does not match accepted FCL poses"
        )
    if report["fixed_root"].get("accepted_input_pose_sha256") != hash_accepted_fcl_poses(
        normalized_fixed_root_poses
    ):
        raise CandidateError("physics report fixed-root input pose hash is invalid")
    completed_poses = report["fixed_root"].get("completed_simulated_poses")
    if not isinstance(completed_poses, list) or not completed_poses:
        raise CandidateError("physics report completed simulated pose grid is missing")
    try:
        normalized_completed_poses = normalize_accepted_fcl_poses(completed_poses)
    except (TypeError, ValueError) as exc:
        raise CandidateError(f"physics report completed pose grid is invalid: {exc}") from exc
    if report["fixed_root"].get("completed_simulated_pose_sha256") != hash_accepted_fcl_poses(
        normalized_completed_poses
    ):
        raise CandidateError("physics report completed simulated pose hash is invalid")
    from sdk._core.v0.motion_engine.contract import complete_joint_equalities

    expected_completed_poses = normalize_accepted_fcl_poses(
        [
            complete_joint_equalities(manifest.simulator_collision_contract, pose)
            for pose in normalized_fixed_root_poses
        ]
    )
    if normalized_completed_poses != expected_completed_poses:
        raise CandidateError(
            "physics report completed simulated poses do not match the candidate equality contract"
        )
    if report.get("blocking_findings"):
        raise CandidateError("physics report contains blocking findings")


def stage_release_candidate(
    model_path: Path,
    blueprint_path: Path,
    candidate_dir: Path,
    *,
    formats: list[str] | tuple[str, ...],
    sdk_package: str = "sdk",
    accepted_fcl_poses: list[dict[str, float]] | tuple[dict[str, float], ...] | None = None,
    simulator_collision_digest: str | None = None,
    collision_resolution_request_key: str | None = None,
    collision_manifest_cache_dir: Path | None = None,
    collision_coacd_cache_dir: Path | None = None,
    collision_manifest: ResolvedCollisionManifest | None = None,
) -> StagedReleaseCandidate:
    from sdk._core.v0.motion_engine.resolve import (
        default_manifest_cache_dir,
        load_collision_manifest,
        load_or_create_collision_manifest,
        resolution_request_key,
    )
    from sdk.runtime import render as sdk_render

    if collision_manifest is None:
        if simulator_collision_digest is not None or collision_resolution_request_key is not None:
            cache = collision_manifest_cache_dir or default_manifest_cache_dir()
            key = collision_resolution_request_key or resolution_request_key(
                Path(model_path),
                sdk_package=sdk_package,
                cache_dir=cache,
            )
            collision_manifest = load_collision_manifest(cache, key)
            if collision_manifest is None:
                current_key = resolution_request_key(
                    Path(model_path), sdk_package=sdk_package, cache_dir=cache
                )
                collision_manifest = load_collision_manifest(cache, current_key)
        else:
            collision_manifest = load_or_create_collision_manifest(
                Path(model_path),
                sdk_package=sdk_package,
                cache_dir=collision_manifest_cache_dir,
                coacd_cache_dir=collision_coacd_cache_dir,
            )
    verify_candidate_collision_identity(
        Path(model_path),
        collision_manifest,
        simulator_collision_digest=simulator_collision_digest,
        collision_resolution_request_key=collision_resolution_request_key,
        sdk_package=sdk_package,
    )
    assert collision_manifest is not None
    identity = compute_candidate_identity(
        model_path,
        blueprint_path,
        formats=formats,
        accepted_fcl_poses=accepted_fcl_poses,
        simulator_collision_digest=collision_manifest.simulator_collision_digest,
        collision_resolution_request_key=collision_manifest.resolution_request_key,
        sdk_package=sdk_package,
    )
    if collision_manifest.simulator_collision_digest != identity.simulator_collision_digest:
        raise CandidateError("resolved collision manifest changed during candidate identity build")
    candidate_dir = Path(candidate_dir)
    candidate_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{candidate_dir.name}-{identity.build_id[:12]}-", dir=candidate_dir.parent
        )
    )
    try:
        blueprint = json.loads(Path(blueprint_path).read_text(encoding="utf-8"))
        if not isinstance(blueprint, dict):
            raise CandidateError("blueprint must be a JSON object")
        report = sdk_render.export(
            Path(model_path),
            list(identity.formats),
            stage,
            sdk_package=sdk_package,
            collision_manifest=collision_manifest,
        )
        exports = dict(report.get("exports") or {}) if isinstance(report, dict) else {}
        missing = sorted(_required_export_keys(identity.formats) - set(exports))
        if missing:
            raise CandidateError(f"candidate export missing required artifacts: {missing}")
        from sdk.runtime.manip_manifest import derive_manifest, validate_manifest_against_urdf

        manipulation = derive_manifest(
            Path(model_path),
            blueprint,
            Path(model_path).parent,
            stage,
            sdk_package=sdk_package,
            collision_manifest=collision_manifest,
        )
        manipulation_path = stage / "manipulation.json"
        if manipulation_path.is_file():
            exports["manipulation"] = str(manipulation_path)
            report["exports"] = exports
        report["manipulation"] = manipulation
        if manipulation.get("operations", 0):
            urdf_path = stage / "object.urdf"
            if not urdf_path.is_file():
                raise CandidateError(
                    "candidate with manipulation operations requires an object.urdf artifact"
                )
            target_failures = validate_manifest_against_urdf(manipulation_path, urdf_path)
            if target_failures:
                raise CandidateError(
                    "candidate manipulation target does not match exact URDF: "
                    + "; ".join(target_failures)
                )
        verify_candidate_collision_identity(
            Path(model_path),
            collision_manifest,
            simulator_collision_digest=identity.simulator_collision_digest,
            collision_resolution_request_key=identity.collision_resolution_request_key,
            sdk_package=sdk_package,
        )
        if dependency_sha256_for_model(Path(model_path)) != identity.dependency_sha256:
            raise CandidateError("candidate dependencies changed during staging")
        artifacts = _artifact_identities(stage)
        artifact_set_sha256 = _artifact_set_sha256(artifacts)
        candidate_id = _exact_candidate_id(
            build_id=identity.build_id,
            artifact_set_sha256=artifact_set_sha256,
        )
        manifest = ReleaseCandidateManifest(
            schema_version=SCHEMA_VERSION,
            status="staged",
            candidate_id=candidate_id,
            build_id=identity.build_id,
            artifact_set_sha256=artifact_set_sha256,
            model_sha256=identity.model_sha256,
            dependency_sha256=identity.dependency_sha256,
            blueprint_sha256=identity.blueprint_sha256,
            exporter_policy_version=identity.exporter_policy_version,
            validation_algorithm_version=identity.validation_algorithm_version,
            validation_policy_sha256=identity.validation_policy_sha256,
            accepted_fcl_pose_sha256=identity.accepted_fcl_pose_sha256,
            simulator_collision_digest=identity.simulator_collision_digest,
            collision_resolution_request_key=identity.collision_resolution_request_key,
            simulator_collision_contract=collision_manifest.simulator_contract,
            mujoco_version=identity.mujoco_version,
            formats=identity.formats,
            artifacts=artifacts,
        )
        manifest_path = stage / "manifest.json"
        _write_manifest(manifest_path, manifest)
        return StagedReleaseCandidate(stage, manifest_path, manifest, report)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def verify_release_candidate(
    manifest_path: Path,
    *,
    require_accepted: bool = True,
) -> ReleaseCandidateManifest:
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    from sdk._core.v0.motion_engine.contract import contract_digest

    if (
        contract_digest(manifest.simulator_collision_contract)
        != manifest.simulator_collision_digest
    ):
        raise CandidateError("simulator collision contract does not match manifest digest")
    if require_accepted and manifest.status != "accepted":
        raise CandidateError(f"candidate is not accepted: {manifest.status}")
    expected_build_id = _hash_payload(
        _candidate_identity_payload(
            model_sha256=manifest.model_sha256,
            dependency_sha256=manifest.dependency_sha256,
            blueprint_sha256=manifest.blueprint_sha256,
            exporter_policy_version=manifest.exporter_policy_version,
            validation_algorithm_version=manifest.validation_algorithm_version,
            validation_policy_sha256=manifest.validation_policy_sha256,
            accepted_fcl_pose_sha256=manifest.accepted_fcl_pose_sha256,
            simulator_collision_digest=manifest.simulator_collision_digest,
            collision_resolution_request_key=manifest.collision_resolution_request_key,
            mujoco_version=manifest.mujoco_version,
            formats=manifest.formats,
        )
    )
    if manifest.build_id != expected_build_id:
        raise CandidateError("build_id does not match manifest identity fields")
    expected_artifact_set_sha256 = _artifact_set_sha256(manifest.artifacts)
    if manifest.artifact_set_sha256 != expected_artifact_set_sha256:
        raise CandidateError("artifact_set_sha256 does not match candidate artifacts")
    expected_candidate_id = _exact_candidate_id(
        build_id=manifest.build_id,
        artifact_set_sha256=manifest.artifact_set_sha256,
    )
    if manifest.candidate_id != expected_candidate_id:
        raise CandidateError("candidate_id does not match manifest identity fields")
    missing_artifacts = sorted(_required_artifact_paths(manifest.formats) - set(manifest.artifacts))
    if missing_artifacts:
        raise CandidateError(f"candidate manifest missing required artifacts: {missing_artifacts}")
    root = manifest_path.parent
    for relative, artifact in manifest.artifacts.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise CandidateError(f"candidate artifact path escapes root: {relative}")
        if artifact.path != relative:
            raise CandidateError(f"candidate artifact key/path mismatch: {relative}")
        path = root / relative
        if not path.is_file():
            raise CandidateError(f"candidate artifact missing: {relative}")
        if path.stat().st_size != artifact.size_bytes or sha256_file(path) != artifact.sha256:
            raise CandidateError(f"candidate artifact integrity mismatch: {relative}")
    if manifest.physics_report_path is not None:
        report_path = Path(manifest.physics_report_path)
        if report_path.is_absolute() or ".." in report_path.parts:
            raise CandidateError("candidate physics report path escapes root")
        report_path = root / report_path
        if not report_path.is_file():
            raise CandidateError("candidate physics report missing")
        if sha256_file(report_path) != manifest.physics_report_sha256:
            raise CandidateError("candidate physics report integrity mismatch")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict) or report.get("status") != "pass":
            raise CandidateError("accepted candidate physics report must have status=pass")
        _verify_physics_report_binding(manifest, report)
    elif require_accepted:
        raise CandidateError("accepted candidate has no physics report")
    return manifest


def _replace_directory(source: Path, destination: Path) -> None:
    backup: Path | None = None
    if destination.exists():
        backup = destination.parent / f".{destination.name}-backup-{uuid.uuid4().hex}"
        os.replace(destination, backup)
    try:
        os.replace(source, destination)
    except Exception:
        if backup is not None and backup.exists():
            os.replace(backup, destination)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def accept_release_candidate(
    staged: StagedReleaseCandidate,
    candidate_dir: Path,
    physics_report_path: Path,
) -> Path:
    report_path = Path(physics_report_path).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("status") != "pass":
        raise CandidateError("physics report must have status=pass")
    _verify_physics_report_binding(staged.manifest, report)
    verify_release_candidate(staged.manifest_path, require_accepted=False)
    embedded_report = staged.root / "physics_validation.json"
    if report_path != embedded_report.resolve():
        shutil.copy2(report_path, embedded_report)
    if report_path.parent == staged.root.resolve() and report_path != embedded_report.resolve():
        report_path.unlink(missing_ok=True)
    accepted = replace(
        staged.manifest,
        status="accepted",
        artifacts=_artifact_identities(staged.root),
        physics_report_path="physics_validation.json",
        physics_report_sha256=sha256_file(embedded_report),
    )
    _write_manifest(staged.manifest_path, accepted)
    verify_release_candidate(staged.manifest_path)
    candidate_dir = Path(candidate_dir)
    _replace_directory(staged.root, candidate_dir)
    manifest_path = candidate_dir / "manifest.json"
    verify_release_candidate(manifest_path)
    return manifest_path


def cached_release_candidate(
    candidate_dir: Path,
    identity: CandidateIdentity,
) -> Path | None:
    manifest_path = Path(candidate_dir) / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = verify_release_candidate(manifest_path)
    except (CandidateError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return manifest_path if manifest.build_id == identity.build_id else None


def _published_exports(out_dir: Path, manifest: ReleaseCandidateManifest) -> dict[str, str]:
    preferred = {
        "glb": "_pose_closed.glb",
        "articulated_glb": "object.articulated.glb",
        "articulated_appearance": "object.articulated.appearance.json",
        "obj": "object.obj",
        "urdf": "object.urdf",
        "mjcf": "object.xml",
        "mjcf_scene": "object_scene.xml",
        "usd": "object.usd",
        "manipulation": "manipulation.json",
    }
    available = set(manifest.artifacts)
    return {
        key: str(out_dir / relative) for key, relative in preferred.items() if relative in available
    }


def publish_release_candidate(
    manifest_path: Path,
    out_dir: Path,
    *,
    expected_candidate_id: str | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = verify_release_candidate(manifest_path)
    if expected_candidate_id is not None and manifest.candidate_id != expected_candidate_id:
        raise CandidateError("candidate does not match the ID accepted by Review")
    source_root = manifest_path.parent
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}-", dir=out_dir.parent))
    try:
        for relative, artifact in manifest.artifacts.items():
            source = source_root / relative
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if (
                destination.stat().st_size != artifact.size_bytes
                or sha256_file(destination) != artifact.sha256
            ):
                raise CandidateError(f"published candidate copy failed integrity check: {relative}")
        export_manifest = manifest.to_dict()
        export_manifest["source_manifest_sha256"] = sha256_file(manifest_path)
        (stage / "export_manifest.json").write_text(
            json.dumps(export_manifest, indent=2), encoding="utf-8"
        )
        _replace_directory(stage, out_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "candidate_id": manifest.candidate_id,
        "manifest": str(out_dir / "export_manifest.json"),
        "exports": _published_exports(out_dir, manifest),
        "skipped": ["usd"]
        if "usd" in manifest.formats and "object.usd" not in manifest.artifacts
        else [],
    }
