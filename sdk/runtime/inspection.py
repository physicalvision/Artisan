
from __future__ import annotations

import hashlib
import json
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from pipeline.operation_timing import operation_span, timed_operation


def inspection_input_key(model_path: Path) -> str:
    from sdk._core.v0.motion_engine.resolve import (
        default_manifest_cache_dir,
        frozen_input_identity,
        load_collision_manifest,
        motion_geometry_sha256_for_model,
    )

    source = model_path.read_bytes() if model_path.is_file() else b""
    snapshot_identity = b""
    evidence_path = model_path.parent / "reports/motion_scan.json"
    if evidence_path.is_file():
        try:
            evidence = json.loads(evidence_path.read_text())["evidence"]
            key = evidence["motion_identity"]["resolution_request_key"]
            manifest = load_collision_manifest(default_manifest_cache_dir(), key)
            if manifest is None:
                raise ValueError("Snapshot unavailable")
            snapshot_identity = json.dumps(
                frozen_input_identity(model_path, manifest),
                sort_keys=True,
            ).encode()
        except (KeyError, OSError, TypeError, ValueError):
            snapshot_identity = uuid.uuid4().hex.encode()
    return hashlib.sha256(
        source + motion_geometry_sha256_for_model(model_path).encode("ascii") + snapshot_identity
    ).hexdigest()


def automatic_snapshot_reference(
    model_path: Path,
    *,
    sdk_package: str = "sdk",
    require_current: bool = False,
) -> dict[str, str] | None:
    from sdk.runtime.performance_policy import PerformancePolicy

    if not PerformancePolicy.from_env().frozen_inspection:
        return None
    return snapshot_reference(model_path, sdk_package=sdk_package, require_current=require_current)


@timed_operation("inspection.verify")
def _load_reference(reference: dict[str, str], model_path: Path, sdk_package: str) -> Any:
    from sdk._core.v0.motion_engine.resolve import (
        load_collision_manifest,
        verify_frozen_inputs,
    )

    root = Path(reference["manifest_root"])
    manifest = load_collision_manifest(root.parent, reference["resolution_request_key"])
    if manifest is None or manifest.root != root:
        raise ValueError("Inspection snapshot is missing or corrupt")
    if manifest.resolved_manifest_digest != reference["resolved_manifest_digest"]:
        raise ValueError("Inspection snapshot digest mismatch")
    if manifest.simulator_collision_digest != reference["simulator_collision_digest"]:
        raise ValueError("Inspection simulator collision digest mismatch")
    verify_frozen_inputs(model_path, manifest, sdk_package=sdk_package)
    return manifest


@timed_operation("inspection.lookup")
def snapshot_reference(
    model_path: Path,
    *,
    sdk_package: str = "sdk",
    require_current: bool = False,
) -> dict[str, str] | None:
    from sdk._core.v0.motion_engine.resolve import (
        default_manifest_cache_dir,
        load_collision_manifest,
    )

    evidence_path = model_path.parent / "reports" / "motion_scan.json"
    if not evidence_path.exists():
        if require_current:
            raise ValueError("Required inspection motion evidence is missing")
        return None
    try:
        envelope = json.loads(evidence_path.read_text())
        evidence = envelope["evidence"]
        if evidence["motion_execution_status"]["status"] != "ok":
            raise ValueError("Inspection motion evidence is incomplete")
        identity = evidence["motion_identity"]
        key = identity["resolution_request_key"]
        manifest = load_collision_manifest(default_manifest_cache_dir(), key)
        if manifest is None or manifest.root is None:
            raise ValueError("Inspection frozen build is unavailable")
        reference = {
            "manifest_root": str(manifest.root),
            "resolution_request_key": key,
            "resolved_manifest_digest": identity["resolved_manifest_digest"],
            "simulator_collision_digest": identity["simulator_collision_digest"],
        }
        _load_reference(reference, model_path, sdk_package)
        return reference
    except (KeyError, OSError, TypeError, ValueError):
        if require_current:
            raise
        return None


@contextmanager
def private_snapshot(
    model_path: Path,
    reference: dict[str, str],
    *,
    sdk_package: str = "sdk",
) -> Iterator[tuple[Any, Path, Path]]:
    from sdk._core.v0.motion_engine.frozen_build import read_frozen_build

    manifest = _load_reference(reference, model_path, sdk_package)
    with tempfile.TemporaryDirectory(prefix="artisan-inspection-") as directory:
        private = Path(directory)
        with operation_span("inspection.copy"):
            for relative, expected in manifest.frozen_files.items():
                source = manifest.root / relative
                destination = private / relative
                if not destination.resolve().is_relative_to(private):
                    raise ValueError("Inspection snapshot contains an unsafe relative path")
                payload = source.read_bytes()
                if hashlib.sha256(payload).hexdigest() != expected:
                    raise ValueError("Inspection snapshot changed during copy")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
        with operation_span("inspection.decode"):
            raw, _resolved = read_frozen_build(private / "build.json")
        try:
            yield raw, private, manifest.root
        finally:
            _load_reference(reference, model_path, sdk_package)


def stable_snapshot_paths(value: Any, private_root: Path, original_root: Path) -> Any:
    if isinstance(value, str):
        prefix = str(private_root) + "/"
        return (
            str(original_root) + "/" + value[len(prefix) :] if value.startswith(prefix) else value
        )
    if isinstance(value, dict):
        return {k: stable_snapshot_paths(v, private_root, original_root) for k, v in value.items()}
    if isinstance(value, list):
        return [stable_snapshot_paths(v, private_root, original_root) for v in value]
    if isinstance(value, tuple):
        return tuple(stable_snapshot_paths(v, private_root, original_root) for v in value)
    return value
