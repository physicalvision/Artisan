from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
import shutil
import sysconfig
import tempfile
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterator, Mapping

from .._coacd import coacd_policy_version
from ..assets import AssetContext, activate_asset_session, asset_session_for_script
from ..collision_mesh_repair import repair_policy_identity
from ..exact_collisions import compile_object_model_with_exact_collisions
from ..geometry_qc import _collision_pair_metrics, _compiled_part_collision_entries
from ..physics_export import enrich_for_physics
from ..placement import PLACEMENT_MATH_IDENTITY
from ..types import Box, Collision, Cylinder, Mesh, Origin, Sphere
from .contract import (
    SIMULATOR_CONTRACT_SCHEMA_VERSION,
    canonical_bytes,
    canonical_payload,
    collision_geom_name,
    contract_digest,
    contract_from_model,
    source_element_id,
)
from .frozen_build import read_frozen_build, write_frozen_build

RESOLVED_MANIFEST_SCHEMA_VERSION = "resolved-collision-manifest-v3"
EXACT_LEGALITY_SCHEMA_VERSION = "exact-legality-v1"
SDK_COLLISION_API_VERSION = "sdk-collision-api-v1"
RESOLUTION_POLICY_VERSION = "collision-resolution-v5"
DEFAULT_CONNECTIVITY_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class ResolvedColliderRecord:
    part_name: str
    source_element_id: str
    resolved_piece_id: str
    resolution_kind: str
    geom_type: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    primitive_params: tuple[float, ...] = ()
    mesh_sha256: str | None = None
    mesh_relpath: str | None = None
    scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    provenance: tuple[str, ...] = ()
    source_visual_uid: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "part_name": self.part_name,
            "source_element_id": self.source_element_id,
            "resolved_piece_id": self.resolved_piece_id,
            "resolution_kind": self.resolution_kind,
            "geom_type": self.geom_type,
            "origin_xyz": list(self.origin_xyz),
            "origin_rpy": list(self.origin_rpy),
            "primitive_params": list(self.primitive_params),
            "mesh_sha256": self.mesh_sha256,
            "mesh_relpath": self.mesh_relpath,
            "scale_xyz": list(self.scale_xyz),
            "provenance": list(self.provenance),
            "source_visual_uid": self.source_visual_uid,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ResolvedColliderRecord:
        return cls(
            part_name=str(payload["part_name"]),
            source_element_id=str(payload["source_element_id"]),
            resolved_piece_id=str(payload["resolved_piece_id"]),
            resolution_kind=str(payload["resolution_kind"]),
            geom_type=str(payload["geom_type"]),
            origin_xyz=tuple(float(value) for value in payload["origin_xyz"]),  # type: ignore[arg-type]
            origin_rpy=tuple(float(value) for value in payload["origin_rpy"]),  # type: ignore[arg-type]
            primitive_params=tuple(
                float(value)
                for value in payload.get("primitive_params", [])  # type: ignore[arg-type]
            ),
            mesh_sha256=(str(payload["mesh_sha256"]) if payload.get("mesh_sha256") else None),
            mesh_relpath=(str(payload["mesh_relpath"]) if payload.get("mesh_relpath") else None),
            scale_xyz=tuple(
                float(value)
                for value in payload.get("scale_xyz", (1.0, 1.0, 1.0))  # type: ignore[arg-type]
            ),
            provenance=tuple(str(value) for value in payload.get("provenance", [])),  # type: ignore[arg-type]
            source_visual_uid=(
                str(payload["source_visual_uid"])
                if payload.get("source_visual_uid") is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class ExactLegalityTable:
    schema_version: str
    contact_tolerance: float
    entries: tuple[dict[str, object], ...]
    adjacency_edges: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contact_tolerance": self.contact_tolerance,
            "entries": [dict(value) for value in self.entries],
            "adjacency_edges": [list(value) for value in self.adjacency_edges],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ExactLegalityTable:
        return cls(
            schema_version=str(payload["schema_version"]),
            contact_tolerance=float(payload["contact_tolerance"]),
            entries=tuple(dict(value) for value in payload.get("entries", [])),  # type: ignore[arg-type]
            adjacency_edges=tuple(
                (str(value[0]), str(value[1]))
                for value in payload.get("adjacency_edges", [])  # type: ignore[arg-type]
            ),
        )


@dataclass(frozen=True, slots=True)
class ResolvedCollisionManifest:
    schema_version: str
    simulator_contract_schema_version: str
    resolution_request_key: str
    source_sha256: str
    motion_geometry_sha256: str
    policy_identity: dict[str, object]
    colliders: tuple[ResolvedColliderRecord, ...]
    exact_legality: ExactLegalityTable
    simulator_contract: dict[str, object]
    warnings: tuple[str, ...]
    simulator_collision_digest: str
    resolved_manifest_digest: str
    workspace_geometry_sha256: str
    frozen_build_sha256: str
    frozen_files: dict[str, str]
    mesh_inputs: dict[str, str]
    materialization_id: str
    root: Path | None = field(default=None, compare=False, repr=False)

    def digest_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "simulator_contract_schema_version": self.simulator_contract_schema_version,
            "resolution_request_key": self.resolution_request_key,
            "source_sha256": self.source_sha256,
            "motion_geometry_sha256": self.motion_geometry_sha256,
            "policy_identity": self.policy_identity,
            "colliders": [value.to_dict() for value in self.colliders],
            "exact_legality": self.exact_legality.to_dict(),
            "simulator_contract": self.simulator_contract,
            "warnings": list(self.warnings),
            "simulator_collision_digest": self.simulator_collision_digest,
            "workspace_geometry_sha256": self.workspace_geometry_sha256,
            "frozen_build_sha256": self.frozen_build_sha256,
            "frozen_files": self.frozen_files,
            "mesh_inputs": self.mesh_inputs,
            "materialization_id": self.materialization_id,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.digest_payload(), "resolved_manifest_digest": self.resolved_manifest_digest}

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object], *, root: Path | None = None
    ) -> ResolvedCollisionManifest:
        manifest = cls(
            schema_version=str(payload["schema_version"]),
            simulator_contract_schema_version=str(payload["simulator_contract_schema_version"]),
            resolution_request_key=str(payload["resolution_request_key"]),
            source_sha256=str(payload["source_sha256"]),
            motion_geometry_sha256=str(payload["motion_geometry_sha256"]),
            policy_identity=dict(payload.get("policy_identity") or {}),
            colliders=tuple(
                ResolvedColliderRecord.from_dict(value)
                for value in payload.get("colliders", [])  # type: ignore[arg-type]
            ),
            exact_legality=ExactLegalityTable.from_dict(dict(payload.get("exact_legality") or {})),
            simulator_contract=dict(payload.get("simulator_contract") or {}),
            warnings=tuple(str(value) for value in payload.get("warnings", [])),  # type: ignore[arg-type]
            simulator_collision_digest=str(payload["simulator_collision_digest"]),
            resolved_manifest_digest=str(payload["resolved_manifest_digest"]),
            workspace_geometry_sha256=str(payload["workspace_geometry_sha256"]),
            frozen_build_sha256=str(payload["frozen_build_sha256"]),
            frozen_files=dict(payload["frozen_files"]),
            mesh_inputs=dict(payload["mesh_inputs"]),
            materialization_id=str(payload["materialization_id"]),
            root=root,
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.schema_version != RESOLVED_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported resolved manifest schema: {self.schema_version}")
        if self.simulator_contract_schema_version != SIMULATOR_CONTRACT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported simulator contract schema: {self.simulator_contract_schema_version}"
            )
        if self.exact_legality.schema_version != EXACT_LEGALITY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported exact legality schema: {self.exact_legality.schema_version}"
            )
        expected_simulator = contract_digest(self.simulator_contract)
        if expected_simulator != self.simulator_collision_digest:
            raise ValueError("simulator collision digest does not match manifest contract")
        expected_manifest = hashlib.sha256(canonical_bytes(self.digest_payload())).hexdigest()
        if expected_manifest != self.resolved_manifest_digest:
            raise ValueError("resolved manifest digest does not match canonical payload")
        if self.root is not None:
            for relative, expected in self.frozen_files.items():
                path = self.root / relative
                if Path(relative).is_absolute() or ".." in Path(relative).parts:
                    raise ValueError(f"frozen file path escapes bundle: {relative}")
                if not path.is_file() or _hash_file(path) != expected:
                    raise ValueError(f"frozen build file digest mismatch: {relative}")
            if self.frozen_files.get("build.json") != self.frozen_build_sha256:
                raise ValueError("frozen model is not bound to manifest")
            for collider in self.colliders:
                if collider.mesh_relpath is None:
                    continue
                path = self.root / collider.mesh_relpath
                if not path.is_file():
                    raise ValueError(f"resolved collision mesh is missing: {path}")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != collider.mesh_sha256:
                    raise ValueError(f"resolved collision mesh digest mismatch: {path}")


@dataclass(slots=True)
class ResolvedCollisionGeometry:
    raw_model: object
    resolved_model: object
    manifest: ResolvedCollisionManifest
    inertia_warnings: list[str]


def fcl_backend_identity() -> dict[str, str]:
    try:
        version = importlib.metadata.version("python-fcl")
    except importlib.metadata.PackageNotFoundError:
        version = "absent"
    return {"python_fcl_version": version, "platform_tag": sysconfig.get_platform()}


def sdk_collision_identity(sdk_package: str) -> dict[str, str]:
    try:
        distribution_version = importlib.metadata.version("artisan-assets")
    except importlib.metadata.PackageNotFoundError:
        distribution_version = "editable"
    return {
        "package": str(sdk_package),
        "distribution": "artisan-assets",
        "distribution_version": distribution_version,
        "sdk_collision_api_version": SDK_COLLISION_API_VERSION,
    }


def resolution_policy_identity(
    sdk_package: str,
    *,
    contact_tolerance: float = DEFAULT_CONNECTIVITY_TOLERANCE,
) -> dict[str, object]:
    repair_digest = hashlib.sha256(canonical_bytes(repair_policy_identity())).hexdigest()
    return {
        "resolution_policy_version": RESOLUTION_POLICY_VERSION,
        "coacd_policy_version": coacd_policy_version(),
        "coacd_enabled": os.environ.get("ARTISAN_COLLISION_COACD", "1") != "0",
        "repair_policy_sha256": repair_digest,
        "contact_tolerance": float(contact_tolerance),
        "exact_legality_schema_version": EXACT_LEGALITY_SCHEMA_VERSION,
        "resolved_manifest_schema_version": RESOLVED_MANIFEST_SCHEMA_VERSION,
        "simulator_contract_schema_version": SIMULATOR_CONTRACT_SCHEMA_VERSION,
        "placement_math_identity": PLACEMENT_MATH_IDENTITY,
        "fcl_backend_identity": fcl_backend_identity(),
        "sdk_collision_identity": sdk_collision_identity(sdk_package),
    }


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def motion_geometry_sha256_for_model(model_path: Path) -> str:
    root = Path(model_path).resolve().parent
    rows: list[dict[str, object]] = []
    for relative_root in (Path("inputs"), Path("references") / "inputs", Path("assets") / "meshes"):
        dependency_root = root / relative_root
        if not dependency_root.is_dir():
            continue
        for path in sorted(value for value in dependency_root.rglob("*") if value.is_file()):
            if path.name.endswith(".cadquery.json") or ".cadquery_cache" in path.parts:
                continue
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": _hash_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return hashlib.sha256(canonical_bytes(rows)).hexdigest()


def resolution_request_key(
    model_path: Path,
    *,
    sdk_package: str = "sdk",
    contact_tolerance: float = DEFAULT_CONNECTIVITY_TOLERANCE,
    motion_geometry_sha256: str | None = None,
    cache_dir: Path | None = None,
) -> str:
    model_path = Path(model_path)
    payload = {
        "source_sha256": _hash_file(model_path),
        "motion_geometry_sha256": (
            motion_geometry_sha256 or motion_geometry_sha256_for_model(model_path)
        ),
        "policy_identity": resolution_policy_identity(
            sdk_package, contact_tolerance=contact_tolerance
        ),
    }
    lookup_key = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    cache = Path(cache_dir or default_manifest_cache_dir())
    binding = cache / "bindings" / f"{lookup_key}.json"
    try:
        key = str(json.loads(binding.read_text(encoding="utf-8"))["resolution_request_key"])
        manifest = load_collision_manifest(cache, key)
        if manifest is not None:
            verify_frozen_inputs(
                model_path, manifest, sdk_package=sdk_package, contact_tolerance=contact_tolerance
            )
            return key
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return lookup_key


class StaleInputError(ValueError):

    def __init__(self, details: dict[str, object]) -> None:
        self.details = details
        super().__init__(json.dumps(details, sort_keys=True))


def _current_file_hash(path: Path) -> str:
    return _hash_file(path) if path.is_file() else "missing"


def _geometry_identity(
    workspace_sha256: str,
    mesh_inputs: dict[str, str],
    frozen_files: dict[str, str],
    materialization_id: str,
) -> str:
    return hashlib.sha256(
        canonical_bytes(
            {
                "workspace_geometry_sha256": workspace_sha256,
                "mesh_inputs": mesh_inputs,
                "frozen_files": frozen_files,
                "materialization_id": materialization_id,
            }
        )
    ).hexdigest()


def frozen_input_identity(
    model_path: Path,
    manifest: ResolvedCollisionManifest,
    *,
    sdk_package: str = "sdk",
    contact_tolerance: float = DEFAULT_CONNECTIVITY_TOLERANCE,
) -> dict[str, object]:
    root = manifest.root
    frozen_files = {
        name: _current_file_hash(root / name) if root is not None else "missing"
        for name in manifest.frozen_files
    }
    mesh_inputs = {
        name: _current_file_hash(model_path.parent / name) for name in manifest.mesh_inputs
    }
    return {
        "source_sha256": _current_file_hash(model_path),
        "motion_geometry_sha256": _geometry_identity(
            motion_geometry_sha256_for_model(model_path),
            mesh_inputs,
            frozen_files,
            manifest.materialization_id,
        ),
        "policy_identity": resolution_policy_identity(
            sdk_package, contact_tolerance=contact_tolerance
        ),
    }


def verify_frozen_inputs(
    model_path: Path,
    manifest: ResolvedCollisionManifest,
    *,
    sdk_package: str = "sdk",
    contact_tolerance: float = DEFAULT_CONNECTIVITY_TOLERANCE,
) -> None:
    observed = frozen_input_identity(
        Path(model_path), manifest, sdk_package=sdk_package, contact_tolerance=contact_tolerance
    )
    components = {
        "source": observed["source_sha256"] != manifest.source_sha256,
        "geometry": observed["motion_geometry_sha256"] != manifest.motion_geometry_sha256,
        "policy": observed["policy_identity"] != manifest.policy_identity,
    }
    observed_key = hashlib.sha256(canonical_bytes(observed)).hexdigest()
    if any(components.values()) or observed_key != manifest.resolution_request_key:
        raise StaleInputError(
            {
                "reason": "frozen build inputs changed; full motion revalidation required",
                "components": components,
                "expected_resolution_request_key": manifest.resolution_request_key,
                "observed_resolution_request_key": observed_key,
            }
        )
    try:
        manifest.validate()
    except (OSError, ValueError, TypeError) as exc:
        raise StaleInputError({"reason": str(exc), "components": components}) from exc


def default_manifest_cache_dir() -> Path:
    from artisan.config import cache_dir

    configured = os.environ.get("ARTISAN_COLLISION_MANIFEST_CACHE")
    return Path(configured).expanduser() if configured else cache_dir() / "collision_manifest_v1"


@contextlib.contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _manifest_path(cache_dir: Path, request_key: str) -> Path:
    return Path(cache_dir) / request_key / "manifest.json"


def load_collision_manifest(
    cache_dir: Path,
    request_key: str,
) -> ResolvedCollisionManifest | None:
    manifest_path = _manifest_path(cache_dir, request_key)
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        manifest = ResolvedCollisionManifest.from_dict(payload, root=manifest_path.parent)
        if manifest.resolution_request_key != request_key:
            return None
        return manifest
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _seed_source_lineage(compiled_model: object) -> None:
    for part in getattr(compiled_model, "parts", []) or []:
        for index, collision in enumerate(getattr(part, "collisions", []) or []):
            collision.source_element_id = source_element_id(str(part.name), collision, index)
            collision.resolution_provenance = tuple(
                getattr(collision, "resolution_provenance", ()) or ("authored_collision",)
            )


def _entry_id(part_name: str, entry: object, index: int) -> str:
    item = getattr(entry, "item", None)
    source_id = getattr(item, "source_element_id", None) or f"{part_name}/element_{index:03d}"
    entry_name = getattr(entry, "name", None)
    if isinstance(entry_name, str) and "__component_" in entry_name:
        return f"{source_id}/{entry_name.rsplit('__component_', 1)[1]}"
    return str(source_id)


def _build_exact_legality_table(
    compiled_model: object,
    *,
    asset_root: Path,
    contact_tolerance: float,
) -> ExactLegalityTable:
    mesh_cache: dict[object, object] = {}
    entries_payload: list[dict[str, object]] = []
    adjacency: set[tuple[str, str]] = set()
    for part in getattr(compiled_model, "parts", []) or []:
        entries = _compiled_part_collision_entries(
            part,
            asset_root=asset_root,
            mesh_cache=mesh_cache,  # type: ignore[arg-type]
        )
        ids = [_entry_id(str(part.name), entry, index) for index, entry in enumerate(entries)]
        for index, (entry, entry_id) in enumerate(zip(entries, ids, strict=True)):
            mesh = getattr(entry, "world_mesh", None)
            vertices = getattr(mesh, "vertices", None)
            faces = getattr(mesh, "faces", None)
            entries_payload.append(
                {
                    "legacy_entry_id": entry_id,
                    "part_name": str(part.name),
                    "source_element_id": str(
                        getattr(getattr(entry, "item", None), "source_element_id", entry_id)
                    ),
                    "name": getattr(entry, "name", None),
                    "geometry_name": str(getattr(entry, "geometry_name", "")),
                    "aabb_min": [float(value) for value in entry.aabb[0]],
                    "aabb_max": [float(value) for value in entry.aabb[1]],
                    "topology": {
                        "vertices": int(len(vertices)) if vertices is not None else 0,
                        "faces": int(len(faces)) if faces is not None else 0,
                        "watertight": bool(getattr(entry, "watertight", False)),
                    },
                    "order": index,
                }
            )
        for left in range(len(entries)):
            for right in range(left + 1, len(entries)):
                collided, distance = _collision_pair_metrics(
                    entries[left].collision_obj, entries[right].collision_obj
                )
                if collided or distance <= float(contact_tolerance):
                    adjacency.add(tuple(sorted((ids[left], ids[right]))))
    return ExactLegalityTable(
        schema_version=EXACT_LEGALITY_SCHEMA_VERSION,
        contact_tolerance=float(contact_tolerance),
        entries=tuple(
            canonical_payload(value)
            for value in sorted(  # type: ignore[misc]
                entries_payload,
                key=lambda value: (str(value["part_name"]), int(value["order"])),
            )
        ),
        adjacency_edges=tuple(sorted(adjacency)),
    )


def _resolve_kind(collision: Collision) -> str:
    existing = getattr(collision, "resolution_kind", None)
    if existing in {"primitive", "convex_piece", "convex_hull"}:
        return str(existing)
    return "convex_hull" if isinstance(collision.geometry, Mesh) else "primitive"


def _finalize_resolved_metadata(resolved_model: object) -> None:
    for part in getattr(resolved_model, "parts", []) or []:
        collisions = list(getattr(part, "collisions", []) or [])
        if not collisions:
            raise RuntimeError(f"part has no publishable resolved collider: {part.name}")
        used_piece_ids: set[str] = set()
        for index, collision in enumerate(collisions):
            if not getattr(collision, "source_element_id", None):
                collision.source_element_id = source_element_id(str(part.name), collision, index)
            collision.resolution_kind = _resolve_kind(collision)
            base_piece_id = collision_geom_name(str(part.name), collision, index)
            piece_id = base_piece_id
            duplicate_index = 2
            while piece_id in used_piece_ids:
                piece_id = f"{base_piece_id}_{duplicate_index}"
                duplicate_index += 1
            used_piece_ids.add(piece_id)
            collision.resolved_piece_id = piece_id


def _copy_mesh_to_bundle(source: Path, bundle_root: Path) -> tuple[str, str]:
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    suffix = source.suffix.lower() or ".obj"
    relative = Path("assets/meshes") / f"{digest}{suffix}"
    destination = bundle_root / relative
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    return relative.as_posix(), digest


def _collider_records(
    resolved_model: object, *, asset_root: Path, bundle_root: Path
) -> tuple[ResolvedColliderRecord, ...]:
    records: list[ResolvedColliderRecord] = []
    for part in getattr(resolved_model, "parts", []) or []:
        for collision in getattr(part, "collisions", []) or []:
            geometry = collision.geometry
            geom_type: str
            primitive_params: tuple[float, ...] = ()
            mesh_sha256: str | None = None
            mesh_relpath: str | None = None
            scale = (1.0, 1.0, 1.0)
            if isinstance(geometry, Box):
                geom_type = "box"
                primitive_params = tuple(float(value) for value in geometry.size)
            elif isinstance(geometry, Cylinder):
                geom_type = "cylinder"
                primitive_params = (float(geometry.radius), float(geometry.length))
            elif isinstance(geometry, Sphere):
                geom_type = "sphere"
                primitive_params = (float(geometry.radius),)
            elif isinstance(geometry, Mesh):
                geom_type = "mesh"
                materialized = getattr(geometry, "materialized_path", None)
                source = (
                    Path(materialized)
                    if materialized
                    else asset_root / str(geometry.filename or "")
                )
                if not source.is_file():
                    raise FileNotFoundError(f"resolved collision mesh is unavailable: {source}")
                mesh_relpath, mesh_sha256 = _copy_mesh_to_bundle(source, bundle_root)
                scale = tuple(float(value) for value in (geometry.scale or (1.0, 1.0, 1.0)))
            else:
                raise TypeError(f"unsupported resolved collision type: {type(geometry).__name__}")
            records.append(
                ResolvedColliderRecord(
                    part_name=str(part.name),
                    source_element_id=str(collision.source_element_id),
                    resolved_piece_id=str(collision.resolved_piece_id),
                    resolution_kind=str(collision.resolution_kind),
                    geom_type=geom_type,
                    origin_xyz=tuple(float(value) for value in collision.origin.xyz),
                    origin_rpy=tuple(float(value) for value in collision.origin.rpy),
                    primitive_params=primitive_params,
                    mesh_sha256=mesh_sha256,
                    mesh_relpath=mesh_relpath,
                    scale_xyz=scale,
                    provenance=tuple(str(value) for value in collision.resolution_provenance),
                    source_visual_uid=getattr(collision, "source_visual_uid", None),
                )
            )
    return tuple(sorted(records, key=lambda value: value.resolved_piece_id))


def _manifest_from_resolution(
    resolved_model: object,
    *,
    request_key: str,
    source_sha256: str,
    motion_geometry_sha256: str,
    policy_identity: dict[str, object],
    exact_legality: ExactLegalityTable,
    warnings: list[str],
    asset_root: Path,
    bundle_root: Path,
    workspace_geometry_sha256: str,
    frozen_files: dict[str, str],
    mesh_inputs: dict[str, str],
    materialization_id: str,
) -> ResolvedCollisionManifest:
    colliders = _collider_records(resolved_model, asset_root=asset_root, bundle_root=bundle_root)
    from .._contact_exemptions import coupling_contact_pairs

    excludes = coupling_contact_pairs(resolved_model, asset_root=asset_root)
    simulator_contract = contract_from_model(
        resolved_model,
        asset_root=asset_root,
        contact_excludes=excludes,
        filterparent="disable",
    )
    simulator_digest = contract_digest(simulator_contract)
    partial = ResolvedCollisionManifest(
        schema_version=RESOLVED_MANIFEST_SCHEMA_VERSION,
        simulator_contract_schema_version=SIMULATOR_CONTRACT_SCHEMA_VERSION,
        resolution_request_key=request_key,
        source_sha256=source_sha256,
        motion_geometry_sha256=motion_geometry_sha256,
        policy_identity=policy_identity,
        colliders=colliders,
        exact_legality=exact_legality,
        simulator_contract=simulator_contract,
        warnings=tuple(warnings),
        simulator_collision_digest=simulator_digest,
        resolved_manifest_digest="",
        workspace_geometry_sha256=workspace_geometry_sha256,
        frozen_build_sha256=frozen_files["build.json"],
        frozen_files=frozen_files,
        mesh_inputs=mesh_inputs,
        materialization_id=materialization_id,
        root=bundle_root,
    )
    digest = hashlib.sha256(canonical_bytes(partial.digest_payload())).hexdigest()
    return replace(partial, resolved_manifest_digest=digest)


def _write_manifest(path: Path, manifest: ResolvedCollisionManifest) -> None:
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def _load_raw_model(model_path: Path, sdk_package: str) -> object:
    from pipeline.compiler import load_model_globals

    namespace = load_model_globals(model_path, sdk_package=sdk_package)
    raw_model = namespace.get("object_model")
    if raw_model is None:
        raise ValueError(f"{model_path} did not define object_model")
    return raw_model


def _freeze_model_meshes(
    model: object,
    *,
    asset_root: Path,
    bundle_root: Path,
    mesh_inputs: dict[str, str] | None = None,
) -> None:
    model.assets = AssetContext(bundle_root)
    for part in model.parts:
        part.assets = model.assets
        for element in [*part.visuals, *part.collisions]:
            geometry = element.geometry
            if not isinstance(geometry, Mesh):
                continue
            source = (
                Path(geometry.materialized_path)
                if geometry.materialized_path
                else (AssetContext(asset_root).mesh_path(str(geometry.filename), ensure_dir=False))
            )
            relative, digest = _copy_mesh_to_bundle(source, bundle_root)
            if mesh_inputs is not None:
                mesh_inputs[os.path.relpath(source, asset_root)] = digest
            element.geometry = replace(
                geometry, filename=relative, materialized_path=str(bundle_root / relative)
            )


def _create_manifest(
    model_path: Path,
    *,
    sdk_package: str,
    cache_dir: Path,
    coacd_cache_dir: Path | None,
    request_key: str,
    contact_tolerance: float,
) -> ResolvedCollisionManifest:
    source_sha256 = _hash_file(model_path)
    policy_identity = resolution_policy_identity(sdk_package, contact_tolerance=contact_tolerance)
    external_before = _external_geometry_sha256(model_path)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f".{request_key[:12]}-", dir=str(cache_dir)))
    try:
        with activate_asset_session(asset_session_for_script(model_path)):
            raw_model = _load_raw_model(model_path, sdk_package)
            compiled = compile_object_model_with_exact_collisions(
                raw_model, asset_root=model_path.parent, validate=False
            )
            mesh_inputs: dict[str, str] = {}
            _freeze_model_meshes(
                raw_model,
                asset_root=model_path.parent,
                bundle_root=temp_root,
                mesh_inputs=mesh_inputs,
            )
            _freeze_model_meshes(compiled, asset_root=model_path.parent, bundle_root=temp_root)
            _seed_source_lineage(compiled)
            exact_legality = _build_exact_legality_table(
                compiled,
                asset_root=temp_root,
                contact_tolerance=contact_tolerance,
            )
            resolved_model, warnings = enrich_for_physics(
                compiled,
                asset_root=temp_root,
                coacd_cache_dir=coacd_cache_dir,
            )
            _finalize_resolved_metadata(resolved_model)
            _freeze_model_meshes(resolved_model, asset_root=temp_root, bundle_root=temp_root)
            write_frozen_build(temp_root / "build.json", raw_model, resolved_model)
            frozen_files = {
                path.relative_to(temp_root).as_posix(): _hash_file(path)
                for path in sorted(temp_root.rglob("*"))
                if path.is_file()
            }
            workspace_sha = motion_geometry_sha256_for_model(model_path)
            materialization_id = uuid.uuid4().hex
            geometry_sha = _geometry_identity(
                workspace_sha, mesh_inputs, frozen_files, materialization_id
            )
            if _hash_file(model_path) != source_sha256 or (
                _external_geometry_sha256(model_path) != external_before
            ):
                raise StaleInputError(
                    {"reason": "source or external inputs changed during materialization"}
                )
            final_key = hashlib.sha256(
                canonical_bytes(
                    {
                        "source_sha256": source_sha256,
                        "motion_geometry_sha256": geometry_sha,
                        "policy_identity": policy_identity,
                    }
                )
            ).hexdigest()
            manifest = _manifest_from_resolution(
                resolved_model,
                request_key=final_key,
                source_sha256=source_sha256,
                motion_geometry_sha256=geometry_sha,
                policy_identity=policy_identity,
                exact_legality=exact_legality,
                warnings=warnings,
                asset_root=temp_root,
                bundle_root=temp_root,
                workspace_geometry_sha256=workspace_sha,
                frozen_files=frozen_files,
                mesh_inputs=mesh_inputs,
                materialization_id=materialization_id,
            )
        verify_frozen_inputs(
            model_path, manifest, sdk_package=sdk_package, contact_tolerance=contact_tolerance
        )
        _write_manifest(temp_root / "manifest.json", manifest)
        target = Path(cache_dir) / final_key
        os.replace(temp_root, target)
        loaded = load_collision_manifest(cache_dir, final_key)
        if loaded is None:
            raise RuntimeError("published collision manifest failed validation")
        lookup_key = hashlib.sha256(
            canonical_bytes(
                {
                    "source_sha256": source_sha256,
                    "motion_geometry_sha256": workspace_sha,
                    "policy_identity": policy_identity,
                }
            )
        ).hexdigest()
        binding = Path(cache_dir) / "bindings" / f"{lookup_key}.json"
        binding.parent.mkdir(parents=True, exist_ok=True)
        pending = binding.with_suffix(f".{materialization_id}.tmp")
        pending.write_text(json.dumps({"resolution_request_key": final_key}), encoding="utf-8")
        os.replace(pending, binding)
        return loaded
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)


def _external_geometry_sha256(model_path: Path) -> str:
    rows = {
        path.relative_to(model_path.parent).as_posix(): _hash_file(path)
        for relative in ("inputs", "references/inputs")
        for path in sorted((model_path.parent / relative).rglob("*"))
        if path.is_file()
    }
    return hashlib.sha256(canonical_bytes(rows)).hexdigest()


def load_or_create_collision_manifest(
    model_path: Path,
    *,
    sdk_package: str = "sdk",
    cache_dir: Path | None = None,
    coacd_cache_dir: Path | None = None,
    contact_tolerance: float = DEFAULT_CONNECTIVITY_TOLERANCE,
) -> ResolvedCollisionManifest:
    model_path = Path(os.path.abspath(model_path))
    resolved_cache_dir = Path(cache_dir or default_manifest_cache_dir()).expanduser()
    geometry_sha = motion_geometry_sha256_for_model(model_path)
    request_key = resolution_request_key(
        model_path,
        sdk_package=sdk_package,
        contact_tolerance=contact_tolerance,
        motion_geometry_sha256=geometry_sha,
        cache_dir=resolved_cache_dir,
    )
    cached = load_collision_manifest(resolved_cache_dir, request_key)
    if cached is not None:
        return cached
    lock_key = hashlib.sha256(str(model_path).encode()).hexdigest()
    lock_path = resolved_cache_dir / ".locks" / f"{lock_key}.lock"
    with _file_lock(lock_path):
        request_key = resolution_request_key(
            model_path,
            sdk_package=sdk_package,
            contact_tolerance=contact_tolerance,
            cache_dir=resolved_cache_dir,
        )
        cached = load_collision_manifest(resolved_cache_dir, request_key)
        if cached is not None:
            return cached
        return _create_manifest(
            model_path,
            sdk_package=sdk_package,
            cache_dir=resolved_cache_dir,
            coacd_cache_dir=coacd_cache_dir,
            request_key=request_key,
            contact_tolerance=contact_tolerance,
        )


def _geometry_from_record(record: ResolvedColliderRecord, manifest_root: Path) -> object:
    if record.geom_type == "box":
        return Box(size=tuple(record.primitive_params))
    if record.geom_type == "cylinder":
        return Cylinder(radius=record.primitive_params[0], length=record.primitive_params[1])
    if record.geom_type == "sphere":
        return Sphere(radius=record.primitive_params[0])
    if record.geom_type == "mesh" and record.mesh_relpath is not None:
        path = manifest_root / record.mesh_relpath
        return Mesh(
            filename=f"assets/meshes/collision_manifest/{path.name}",
            materialized_path=str(path),
            scale=record.scale_xyz,
            name=record.resolved_piece_id,
        )
    raise ValueError(f"invalid resolved collider record: {record.resolved_piece_id}")


def apply_resolved_collisions(
    compiled_model: object,
    manifest: ResolvedCollisionManifest,
) -> object:
    if manifest.root is None:
        raise ValueError("resolved manifest has no bundle root")
    by_part: dict[str, list[ResolvedColliderRecord]] = {}
    for record in manifest.colliders:
        by_part.setdefault(record.part_name, []).append(record)
    known_parts = {str(part.name) for part in getattr(compiled_model, "parts", []) or []}
    if set(by_part) != known_parts:
        raise ValueError(
            f"manifest/model part mismatch: manifest={sorted(by_part)} model={sorted(known_parts)}"
        )
    for part in getattr(compiled_model, "parts", []) or []:
        part.collisions = [
            Collision(
                geometry=_geometry_from_record(record, manifest.root),
                origin=Origin(xyz=record.origin_xyz, rpy=record.origin_rpy),
                name=record.resolved_piece_id,
                source_element_id=record.source_element_id,
                resolved_piece_id=record.resolved_piece_id,
                resolution_kind=record.resolution_kind,
                resolution_provenance=record.provenance,
                source_visual_uid=record.source_visual_uid,
            )
            for record in sorted(by_part[str(part.name)], key=lambda value: value.resolved_piece_id)
        ]
    return compiled_model


def resolve_collision_geometry(
    model_path: Path,
    *,
    sdk_package: str = "sdk",
    cache_dir: Path | None = None,
    manifest: ResolvedCollisionManifest | None = None,
) -> ResolvedCollisionGeometry:
    model_path = Path(model_path).resolve()
    resolved_manifest = manifest or load_or_create_collision_manifest(
        model_path, sdk_package=sdk_package, cache_dir=cache_dir
    )
    verify_frozen_inputs(model_path, resolved_manifest, sdk_package=sdk_package)
    if resolved_manifest.root is None:
        raise ValueError("resolved manifest has no frozen build root")
    raw_model, resolved = read_frozen_build(resolved_manifest.root / "build.json")
    resolved = apply_resolved_collisions(resolved, resolved_manifest)
    verify_frozen_inputs(model_path, resolved_manifest, sdk_package=sdk_package)
    return ResolvedCollisionGeometry(
        raw_model=raw_model,
        resolved_model=resolved,
        manifest=resolved_manifest,
        inertia_warnings=[],
    )
