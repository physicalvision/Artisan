from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from pipeline.state import PipelineState
from sdk.runtime.release_candidate import (
    dependency_sha256_for_model,
    sha256_file,
    verify_release_candidate,
)

_FALLBACK_DIR = "fallback_candidate"
_FAILED_ATTEMPT_DIR = "failed_attempt"
_SNAPSHOT_METADATA = "snapshot.json"
_EVIDENCE_PATHS = (
    Path("_compile_report.json"),
    Path("reports") / "compile_findings.json",
    Path("reports") / "motion_findings.json",
    Path("reports") / "motion_scan.json",
)
_TRACKED_PATHS = (
    Path("model.py"),
    Path("blueprint.json"),
    Path("tasks.json"),
    Path("inputs"),
    Path("references") / "inputs",
    Path("assets") / "meshes",
    Path("candidate"),
    Path("reports") / "physics_validation.json",
    *_EVIDENCE_PATHS,
)


class CandidateRecoveryError(RuntimeError):
    pass


def _copy_tracked_paths(source_root: Path, destination_root: Path) -> None:
    for relative in _TRACKED_PATHS:
        source = source_root / relative
        if not source.exists():
            continue
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def _replace_snapshot(source_root: Path, destination_root: Path) -> None:
    destination_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination_root.name}-", dir=destination_root.parent)
    )
    try:
        _copy_tracked_paths(source_root, temporary)
        if destination_root.exists():
            backup = destination_root.with_name(f".{destination_root.name}-previous")
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(destination_root, backup)
            try:
                os.replace(temporary, destination_root)
            except Exception:
                os.replace(backup, destination_root)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(temporary, destination_root)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _restore_tracked_paths(snapshot_root: Path, workspace_root: Path) -> None:
    for relative in _TRACKED_PATHS:
        source = snapshot_root / relative
        destination = workspace_root / relative
        if destination.exists() or destination.is_symlink():
            _remove_path(destination)
        if not source.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def _evidence_sha256(root: Path) -> dict[str, str]:
    return {
        relative.as_posix(): sha256_file(path)
        for relative in _EVIDENCE_PATHS
        if (path := root / relative).is_file()
    }


def _validate_evidence_sha256(root: Path, expected: dict[str, str]) -> None:
    actual = _evidence_sha256(root)
    if actual != expected:
        raise CandidateRecoveryError("candidate evidence sidecars do not match the snapshot")


def _validated_reviewed_candidate(state: PipelineState, workspace_root: Path) -> Any:
    model_path = workspace_root / "model.py"
    blueprint_path = workspace_root / "blueprint.json"
    manifest_path = Path(
        state.candidate_manifest_path or workspace_root / "candidate/manifest.json"
    )
    manifest = verify_release_candidate(manifest_path)
    if not state.candidate_id or manifest.candidate_id != state.candidate_id:
        raise CandidateRecoveryError("reviewed candidate_id does not match its manifest")
    if (
        not state.reviewed_physics_report_sha256
        or manifest.physics_report_sha256 != state.reviewed_physics_report_sha256
    ):
        raise CandidateRecoveryError("reviewed physics digest does not match its manifest")
    report_path = Path(
        state.physics_report_path or workspace_root / "reports" / "physics_validation.json"
    )
    if not report_path.is_file() or sha256_file(report_path) != manifest.physics_report_sha256:
        raise CandidateRecoveryError("workspace physics report does not match its manifest")
    if not model_path.is_file() or sha256_file(model_path) != manifest.model_sha256:
        raise CandidateRecoveryError("reviewed model.py does not match its candidate")
    if dependency_sha256_for_model(model_path) != manifest.dependency_sha256:
        raise CandidateRecoveryError("reviewed source dependencies do not match its candidate")
    if not blueprint_path.is_file() or sha256_file(blueprint_path) != manifest.blueprint_sha256:
        raise CandidateRecoveryError("reviewed blueprint does not match its candidate")
    return manifest


def snapshot_reviewed_candidate(state: PipelineState, workspace_root: Path) -> dict[str, Any]:
    workspace_root = Path(workspace_root)
    manifest = _validated_reviewed_candidate(state, workspace_root)
    snapshot_root = workspace_root / _FALLBACK_DIR
    current = state.fallback_candidate
    if (
        current.get("status") == "available"
        and current.get("candidate_id") == manifest.candidate_id
        and isinstance(current.get("evidence_sha256"), dict)
        and (snapshot_root / "candidate" / "manifest.json").is_file()
        and (snapshot_root / "model.py").is_file()
    ):
        return dict(current)

    _replace_snapshot(workspace_root, snapshot_root)
    metadata: dict[str, Any] = {
        "status": "available",
        "candidate_id": manifest.candidate_id,
        "reviewed_physics_report_sha256": manifest.physics_report_sha256,
        "snapshot_root": str(snapshot_root),
        "model_path": str(snapshot_root / "model.py"),
        "candidate_manifest_path": str(snapshot_root / "candidate" / "manifest.json"),
        "physics_report_path": str(snapshot_root / "reports" / "physics_validation.json"),
        "evidence_sha256": _evidence_sha256(snapshot_root),
    }
    (snapshot_root / _SNAPSHOT_METADATA).write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    state.fallback_candidate = metadata
    return dict(metadata)


def restore_reviewed_candidate_after_failure(
    state: PipelineState,
    workspace_root: Path,
    *,
    failure_reason: str,
) -> dict[str, Any]:
    workspace_root = Path(workspace_root)
    snapshot = dict(state.fallback_candidate)
    snapshot_root_value = snapshot.get("snapshot_root")
    if snapshot.get("status") != "available" or not isinstance(snapshot_root_value, str):
        raise CandidateRecoveryError("no reviewed candidate snapshot is available")
    snapshot_root = Path(snapshot_root_value)
    snapshot_manifest_path = snapshot_root / "candidate" / "manifest.json"
    if not snapshot_manifest_path.is_file() or not (snapshot_root / "model.py").is_file():
        raise CandidateRecoveryError("reviewed candidate snapshot is incomplete")
    expected_evidence = snapshot.get("evidence_sha256")
    if isinstance(expected_evidence, dict):
        _validate_evidence_sha256(
            snapshot_root,
            {str(path): str(digest) for path, digest in expected_evidence.items()},
        )

    failed_attempt_root = workspace_root / _FAILED_ATTEMPT_DIR
    _replace_snapshot(workspace_root, failed_attempt_root)
    failure_metadata = {
        "failure_reason": failure_reason,
        "candidate_id": state.candidate_id,
        "candidate_manifest_path": state.candidate_manifest_path,
        "physics_report_path": state.physics_report_path,
    }
    (failed_attempt_root / "failure.json").write_text(
        json.dumps(failure_metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    _restore_tracked_paths(snapshot_root, workspace_root)
    restored_manifest = verify_release_candidate(workspace_root / "candidate" / "manifest.json")
    if restored_manifest.candidate_id != snapshot.get("candidate_id"):
        raise CandidateRecoveryError("restored candidate does not match the reviewed snapshot")
    state.candidate_id = restored_manifest.candidate_id
    state.candidate_manifest_path = str(workspace_root / "candidate" / "manifest.json")
    state.physics_report_path = str(workspace_root / "reports" / "physics_validation.json")
    state.reviewed_physics_report_sha256 = str(snapshot["reviewed_physics_report_sha256"])
    state.asset_build_path = str(workspace_root / "model.py")
    _validated_reviewed_candidate(state, workspace_root)
    if isinstance(expected_evidence, dict):
        _validate_evidence_sha256(
            workspace_root,
            {str(path): str(digest) for path, digest in expected_evidence.items()},
        )
    restored = {
        **snapshot,
        "status": "restored_after_failed_repair",
        "failure_reason": failure_reason,
        "failed_attempt_root": str(failed_attempt_root),
        "restored_model_path": state.asset_build_path,
        "restored_candidate_manifest_path": state.candidate_manifest_path,
        "restored_physics_report_path": state.physics_report_path,
    }
    state.fallback_candidate = restored
    return dict(restored)
