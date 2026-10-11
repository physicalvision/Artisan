from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sdk._core.v0._mjcf_export import compile_object_to_mjcf_xml
from sdk._core.v0._motion_feel import MAPPING_VERSION, resolve_motion_feel
from sdk._core.v0.motion_engine.resolve import (
    ResolvedCollisionManifest,
    resolve_collision_geometry,
)
from sdk._core.v0.types import Mesh

PHYSICS_CANDIDATE_VERSION = "physics-candidate-v4"


@dataclass(slots=True)
class PhysicsCandidate:
    raw_model: object
    resolved_model: object
    asset_root: Path
    model_sha256: str
    dependency_sha256: str
    collision_mode: str
    mjcf_bytes: bytes
    mjcf_sha256: str
    mjcf_assets: dict[str, bytes]
    normalized_intents: dict[str, dict[str, object]]
    resolution_evidence: list[dict[str, object]]
    warnings: list[str]
    collision_manifest: ResolvedCollisionManifest

    def summary(self) -> dict[str, object]:
        return {
            "version": PHYSICS_CANDIDATE_VERSION,
            "mapping_version": MAPPING_VERSION,
            "model_sha256": self.model_sha256,
            "dependency_sha256": self.dependency_sha256,
            "collision_mode": self.collision_mode,
            "mjcf_sha256": self.mjcf_sha256,
            "normalized_intents": self.normalized_intents,
            "resolution_evidence": self.resolution_evidence,
            "warnings": self.warnings,
            "resolved_manifest_digest": self.collision_manifest.resolved_manifest_digest,
            "simulator_collision_digest": self.collision_manifest.simulator_collision_digest,
        }


def _normalized_intents(model: object) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for articulation in getattr(model, "articulations", []) or []:
        feel = getattr(articulation, "motion_feel", None)
        if feel is None:
            resolution = dict(
                (getattr(articulation, "meta", None) or {}).get("_motion_feel_resolution") or {}
            )
            if resolution.get("source") == "synthesized_default":
                result[str(articulation.name)] = {
                    "source": "synthesized_default",
                    "effort": None,
                    "motion": None,
                    "hold": "free",
                    "note": "light passive fallback; explicit MotionFeel preferred",
                    "warnings": [],
                }
            continue
        result[str(articulation.name)] = {
            "source": "motion_feel",
            "effort": float(feel.effort),
            "motion": float(feel.motion),
            "hold": str(feel.hold),
            "note": str(feel.note),
            "warnings": list(feel.warnings),
        }
    return result


def _dependency_manifest(model: object, asset_root: Path) -> list[dict[str, object]]:
    files: dict[str, Path] = {}
    for part in getattr(model, "parts", []) or []:
        elements = list(getattr(part, "visuals", []) or []) + list(
            getattr(part, "collisions", []) or []
        )
        for element in elements:
            geometry = getattr(element, "geometry", None)
            if not isinstance(geometry, Mesh):
                continue
            materialized = getattr(geometry, "materialized_path", None)
            filename = getattr(geometry, "filename", None)
            candidate = Path(materialized) if materialized else asset_root / str(filename or "")
            if candidate.is_file():
                files[str(filename or candidate.name)] = candidate
    return [
        {
            "path": name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        }
        for name, path in sorted(files.items())
    ]


def _mjcf_assets(model: object, asset_root: Path) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    meshdir = "assets/meshes/"
    for part in getattr(model, "parts", []) or []:
        elements = list(getattr(part, "visuals", []) or []) + list(
            getattr(part, "collisions", []) or []
        )
        for element in elements:
            geometry = getattr(element, "geometry", None)
            if not isinstance(geometry, Mesh):
                continue
            materialized = getattr(geometry, "materialized_path", None)
            filename = str(getattr(geometry, "filename", "") or "").replace("\\", "/")
            source = Path(materialized) if materialized else asset_root / filename
            if not filename or not source.is_file():
                continue
            payload = source.read_bytes()
            relative = filename[len(meshdir) :] if filename.startswith(meshdir) else filename
            assets.setdefault(relative, payload)
    return assets


def build_physics_candidate(
    model_path: Path,
    *,
    sdk_package: str = "sdk",
    declared_overlap_pairs: Iterable[tuple[str, str]] = (),
    collision_mode: str = "full",
    collision_manifest: ResolvedCollisionManifest | None = None,
) -> PhysicsCandidate:
    model_path = Path(model_path)
    source_bytes = model_path.read_bytes()
    model_sha256 = hashlib.sha256(source_bytes).hexdigest()
    overlaps = sorted(tuple(sorted((str(a), str(b)))) for a, b in declared_overlap_pairs)
    collision_resolution = resolve_collision_geometry(
        model_path,
        sdk_package=sdk_package,
        manifest=collision_manifest,
    )
    raw_model = collision_resolution.raw_model
    asset_root = collision_resolution.manifest.root
    if asset_root is None:
        raise ValueError("collision manifest has no frozen build root")
    resolved, resolution_evidence = resolve_motion_feel(collision_resolution.resolved_model)
    mjcf_xml = compile_object_to_mjcf_xml(
        raw_model,
        compiled_model=resolved,
        asset_root=asset_root,
        declared_overlap_pairs=overlaps,
        position_servos=False,
        collision_mode=collision_mode,
    )

    manifest = {
        "version": PHYSICS_CANDIDATE_VERSION,
        "mapping_version": MAPPING_VERSION,
        "collision_mode": collision_mode,
        "declared_overlap_pairs": overlaps,
        "assets": _dependency_manifest(resolved, asset_root),
    }
    dependency_sha256 = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    mjcf_bytes = mjcf_xml.encode("utf-8")
    feel_warnings = [
        f"motion_feel:{name}:{warning}"
        for name, intent in _normalized_intents(resolved).items()
        for warning in intent.get("warnings", [])
    ]
    return PhysicsCandidate(
        raw_model=raw_model,
        resolved_model=resolved,
        asset_root=asset_root,
        model_sha256=model_sha256,
        dependency_sha256=dependency_sha256,
        collision_mode=collision_mode,
        mjcf_bytes=mjcf_bytes,
        mjcf_sha256=hashlib.sha256(mjcf_bytes).hexdigest(),
        mjcf_assets=_mjcf_assets(resolved, asset_root),
        normalized_intents=_normalized_intents(resolved),
        resolution_evidence=resolution_evidence,
        warnings=[
            *collision_resolution.manifest.warnings,
            *collision_resolution.inertia_warnings,
            *feel_warnings,
        ],
        collision_manifest=collision_resolution.manifest,
    )
