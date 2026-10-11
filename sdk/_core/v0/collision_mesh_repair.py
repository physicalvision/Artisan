from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterator

import numpy as np

REPAIR_SCHEMA_VERSION = "collision-mesh-repair-v1"
WELD_TOLERANCE_RATIO = 1.0e-6
MAX_FILL_PERIMETER_RATIO = 0.05
MAX_FILL_LOOP_VERTICES = 64


@dataclass(frozen=True, slots=True)
class TopologyMetrics:
    vertices: int
    faces: int
    components: int
    boundary_edges: int
    boundary_loops: int
    boundary_perimeter: float
    watertight: bool
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class CollisionMeshRepairResult:
    path: Path | None
    status: str
    source_sha256: str
    repair_key: str
    before: TopologyMetrics
    after: TopologyMetrics | None
    weld_tolerance: float
    filled_loop_perimeters: tuple[float, ...]
    skipped_loop_perimeters: tuple[float, ...]
    manifold_status: str
    failure_reason: str | None = None


def _metrics_from_dict(payload: dict[str, Any] | None) -> TopologyMetrics | None:
    if payload is None:
        return None
    return TopologyMetrics(
        vertices=int(payload["vertices"]),
        faces=int(payload["faces"]),
        components=int(payload["components"]),
        boundary_edges=int(payload["boundary_edges"]),
        boundary_loops=int(payload["boundary_loops"]),
        boundary_perimeter=float(payload["boundary_perimeter"]),
        watertight=bool(payload["watertight"]),
        bbox_min=tuple(float(value) for value in payload["bbox_min"]),
        bbox_max=tuple(float(value) for value in payload["bbox_max"]),
    )


def _cached_repair(metadata: Path, output: Path) -> CollisionMeshRepairResult | None:
    if not metadata.is_file() or not output.is_file():
        return None
    try:
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        before = _metrics_from_dict(payload.get("before"))
        after = _metrics_from_dict(payload.get("after"))
        if before is None or after is None:
            return None
        return CollisionMeshRepairResult(
            path=output,
            status=str(payload["status"]),
            source_sha256=str(payload["source_sha256"]),
            repair_key=str(payload["repair_key"]),
            before=before,
            after=after,
            weld_tolerance=float(payload["weld_tolerance"]),
            filled_loop_perimeters=tuple(
                float(value) for value in payload.get("filled_loop_perimeters", [])
            ),
            skipped_loop_perimeters=tuple(
                float(value) for value in payload.get("skipped_loop_perimeters", [])
            ),
            manifold_status=str(payload["manifold_status"]),
            failure_reason=payload.get("failure_reason"),
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


@contextmanager
def _repair_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _record_attempt_event(action: str, key: str) -> None:
    path = os.environ.get("ARTISAN_REPAIR_ATTEMPT_LOG")
    if not path:
        return
    try:
        with Path(path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"action": action, "key": key}) + "\n")
    except OSError:
        pass


def manifold_version() -> str:
    try:
        return version("manifold3d")
    except PackageNotFoundError:
        return "unavailable"


def repair_policy_identity() -> dict[str, Any]:
    return {
        "schema": REPAIR_SCHEMA_VERSION,
        "weld_tolerance_ratio": WELD_TOLERANCE_RATIO,
        "max_fill_perimeter_ratio": MAX_FILL_PERIMETER_RATIO,
        "max_fill_loop_vertices": MAX_FILL_LOOP_VERTICES,
        "manifold_version": manifold_version(),
        "manifold_policy": "component-union",
        "fallback_policy": "required-blocking;ordinary-explicit-convex-hull-warning",
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _boundary_edges(mesh: Any) -> np.ndarray:
    if len(mesh.faces) == 0:
        return np.empty((0, 2), dtype=np.int64)
    counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
    return np.asarray(mesh.edges_unique[counts == 1], dtype=np.int64)


def _ordered_boundary_loops(mesh: Any) -> tuple[list[list[int]], int]:
    edges = _boundary_edges(mesh)
    if not len(edges):
        return [], 0
    adjacency: dict[int, list[int]] = {}
    for left, right in edges:
        adjacency.setdefault(int(left), []).append(int(right))
        adjacency.setdefault(int(right), []).append(int(left))

    loops: list[list[int]] = []
    visited: set[tuple[int, int]] = set()
    non_loop_components = 0
    for start in sorted(adjacency):
        available = [
            node for node in adjacency[start] if tuple(sorted((start, node))) not in visited
        ]
        if not available:
            continue
        if any(len(adjacency[node]) != 2 for node in _component_vertices(start, adjacency)):
            non_loop_components += 1
            for node in _component_vertices(start, adjacency):
                for other in adjacency[node]:
                    visited.add(tuple(sorted((node, other))))
            continue
        loop = [start]
        previous = -1
        current = start
        while True:
            candidates = [node for node in adjacency[current] if node != previous]
            if not candidates:
                break
            following = candidates[0]
            edge = tuple(sorted((current, following)))
            if edge in visited:
                break
            visited.add(edge)
            if following == start:
                loops.append(loop)
                break
            loop.append(following)
            previous, current = current, following
    return loops, non_loop_components


def _component_vertices(start: int, adjacency: dict[int, list[int]]) -> set[int]:
    pending = [start]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(node for node in adjacency[current] if node not in seen)
    return seen


def _loop_perimeter(vertices: np.ndarray, loop: list[int]) -> float:
    points = vertices[np.asarray(loop, dtype=np.int64)]
    return float(np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1).sum())


def _topology_metrics(mesh: Any) -> TopologyMetrics:
    loops, non_loop_components = _ordered_boundary_loops(mesh)
    bounds = np.asarray(mesh.bounds, dtype=float) if len(mesh.vertices) else np.zeros((2, 3))
    components = len(mesh.split(only_watertight=False)) if len(mesh.faces) else 0
    boundary_edges = _boundary_edges(mesh)
    if len(boundary_edges):
        vertices = np.asarray(mesh.vertices, dtype=float)
        perimeter = float(
            np.linalg.norm(
                vertices[boundary_edges[:, 0]] - vertices[boundary_edges[:, 1]],
                axis=1,
            ).sum()
        )
    else:
        perimeter = 0.0
    return TopologyMetrics(
        vertices=int(len(mesh.vertices)),
        faces=int(len(mesh.faces)),
        components=int(components),
        boundary_edges=int(len(boundary_edges)),
        boundary_loops=int(len(loops) + non_loop_components),
        boundary_perimeter=float(perimeter),
        watertight=bool(mesh.is_watertight),
        bbox_min=tuple(float(value) for value in bounds[0]),
        bbox_max=tuple(float(value) for value in bounds[1]),
    )


def _fill_small_planar_loops(
    mesh: Any,
    *,
    max_perimeter: float,
    planar_tolerance: float,
) -> tuple[list[float], list[float]]:
    loops, _non_loop_components = _ordered_boundary_loops(mesh)
    if not loops:
        return [], []
    vertices = np.asarray(mesh.vertices, dtype=float).tolist()
    faces = np.asarray(mesh.faces, dtype=np.int64).tolist()
    directed_edges = {
        (int(face[index]), int(face[(index + 1) % 3]))
        for face in np.asarray(mesh.faces, dtype=np.int64)
        for index in range(3)
    }
    filled: list[float] = []
    skipped: list[float] = []
    for loop in loops:
        perimeter = _loop_perimeter(np.asarray(mesh.vertices), loop)
        points = np.asarray(mesh.vertices)[np.asarray(loop, dtype=np.int64)]
        centered = points - points.mean(axis=0)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        nonplanarity = float(singular_values[-1]) if len(singular_values) >= 3 else 0.0
        if (
            perimeter > max_perimeter
            or len(loop) > MAX_FILL_LOOP_VERTICES
            or nonplanarity > planar_tolerance
        ):
            skipped.append(perimeter)
            continue
        if (loop[0], loop[1]) not in directed_edges:
            loop = list(reversed(loop))
        center_index = len(vertices)
        vertices.append(tuple(float(value) for value in points.mean(axis=0)))
        for index, current in enumerate(loop):
            following = loop[(index + 1) % len(loop)]
            faces.append((following, current, center_index))
        filled.append(perimeter)
    if filled:
        mesh.vertices = np.asarray(vertices, dtype=float)
        mesh.faces = np.asarray(faces, dtype=np.int64)
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals(multibody=True)
    return filled, skipped


def _normalize_with_manifold(mesh: Any) -> tuple[Any, str]:
    import manifold3d as m3d
    import trimesh

    manifolds: list[Any] = []
    for index, component in enumerate(mesh.split(only_watertight=False)):
        manifold = m3d.Manifold(
            m3d.Mesh(
                np.asarray(component.vertices, dtype=np.float32),
                np.asarray(component.faces, dtype=np.uint32),
            )
        )
        status = manifold.status()
        if status != m3d.Error.NoError:
            raise ValueError(f"component {index} manifold status={status}")
        manifolds.append(manifold)
    if not manifolds:
        raise ValueError("mesh has no closed components")
    normalized = m3d.Manifold.batch_boolean(manifolds, m3d.OpType.Add)
    status = normalized.status()
    if status != m3d.Error.NoError or normalized.is_empty():
        raise ValueError(f"union manifold status={status}")
    output = normalized.to_mesh()
    repaired = trimesh.Trimesh(
        vertices=np.asarray(output.vert_properties, dtype=float)[:, :3],
        faces=np.asarray(output.tri_verts, dtype=np.int64),
        process=False,
    )
    return repaired, str(status)


def _repair_collision_mesh_uncached(source: Path, *, cache_dir: Path) -> CollisionMeshRepairResult:
    import trimesh

    source = Path(source)
    source_sha256 = _sha256_file(source)
    policy = repair_policy_identity()
    repair_key = hashlib.sha256(
        json.dumps(
            {"source_sha256": source_sha256, "policy": policy},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    loaded = trimesh.load(source, force="mesh", process=False)
    before = _topology_metrics(loaded)
    diagonal = float(np.linalg.norm(np.asarray(before.bbox_max) - np.asarray(before.bbox_min)))
    weld_tolerance = max(diagonal * WELD_TOLERANCE_RATIO, 1.0e-12)
    if before.watertight and before.components <= 1:
        return CollisionMeshRepairResult(
            path=source,
            status="already_watertight",
            source_sha256=source_sha256,
            repair_key=repair_key,
            before=before,
            after=before,
            weld_tolerance=weld_tolerance,
            filled_loop_perimeters=(),
            skipped_loop_perimeters=(),
            manifold_status="not_required",
        )

    working = loaded.copy()
    working.update_faces(working.unique_faces())
    working.update_faces(working.nondegenerate_faces())
    working.remove_unreferenced_vertices()
    digits = max(0, int(math.ceil(-math.log10(weld_tolerance))))
    working.merge_vertices(digits_vertex=digits, merge_tex=True, merge_norm=True)
    working.remove_unreferenced_vertices()
    filled, skipped = _fill_small_planar_loops(
        working,
        max_perimeter=diagonal * MAX_FILL_PERIMETER_RATIO,
        planar_tolerance=max(weld_tolerance * 10.0, diagonal * 1.0e-4),
    )
    cleaned = _topology_metrics(working)
    if not cleaned.watertight:
        return CollisionMeshRepairResult(
            path=None,
            status="rollback",
            source_sha256=source_sha256,
            repair_key=repair_key,
            before=before,
            after=cleaned,
            weld_tolerance=weld_tolerance,
            filled_loop_perimeters=tuple(filled),
            skipped_loop_perimeters=tuple(skipped),
            manifold_status="not_run",
            failure_reason="bounded repair left open boundaries",
        )
    try:
        normalized, manifold_status = _normalize_with_manifold(working)
        after = _topology_metrics(normalized)
    except Exception as exc:  # noqa: BLE001 - certificate carries the exact rollback
        return CollisionMeshRepairResult(
            path=None,
            status="rollback",
            source_sha256=source_sha256,
            repair_key=repair_key,
            before=before,
            after=cleaned,
            weld_tolerance=weld_tolerance,
            filled_loop_perimeters=tuple(filled),
            skipped_loop_perimeters=tuple(skipped),
            manifold_status="failed",
            failure_reason=f"{type(exc).__name__}: {exc}",
        )

    bbox_delta = max(
        float(np.max(np.abs(np.asarray(after.bbox_min) - np.asarray(before.bbox_min)))),
        float(np.max(np.abs(np.asarray(after.bbox_max) - np.asarray(before.bbox_max)))),
    )
    guard_error: str | None = None
    if not after.watertight or after.boundary_edges:
        guard_error = "normalized mesh is not watertight"
    elif after.components > max(before.components, 1):
        guard_error = "component count increased"
    elif after.faces > max(before.faces * 2, before.faces + max(before.boundary_edges, 1) * 2):
        guard_error = "face-count topology delta exceeded"
    elif bbox_delta > max(weld_tolerance * 4.0, diagonal * 1.0e-6):
        guard_error = f"bounding-box delta exceeded ({bbox_delta:.6g})"
    if guard_error:
        return CollisionMeshRepairResult(
            path=None,
            status="rollback",
            source_sha256=source_sha256,
            repair_key=repair_key,
            before=before,
            after=after,
            weld_tolerance=weld_tolerance,
            filled_loop_perimeters=tuple(filled),
            skipped_loop_perimeters=tuple(skipped),
            manifold_status=manifold_status,
            failure_reason=guard_error,
        )

    output = Path(cache_dir) / "mesh_repair" / f"{repair_key}.obj"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = normalized.export(file_type="obj")
    if isinstance(payload, bytes):
        output.write_bytes(payload)
    else:
        output.write_text(str(payload), encoding="utf-8")
    return CollisionMeshRepairResult(
        path=output,
        status="repaired",
        source_sha256=source_sha256,
        repair_key=repair_key,
        before=before,
        after=after,
        weld_tolerance=weld_tolerance,
        filled_loop_perimeters=tuple(filled),
        skipped_loop_perimeters=tuple(skipped),
        manifold_status=manifold_status,
    )


def repair_collision_mesh(source: Path, *, cache_dir: Path) -> CollisionMeshRepairResult:
    source = Path(source)
    source_sha256 = _sha256_file(source)
    repair_key = hashlib.sha256(
        json.dumps(
            {"source_sha256": source_sha256, "policy": repair_policy_identity()},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    cache_root = Path(cache_dir) / "mesh_repair"
    output = cache_root / f"{repair_key}.obj"
    metadata = cache_root / f"{repair_key}.json"
    _record_attempt_event("attempted", repair_key)
    cached = _cached_repair(metadata, output)
    if cached is not None:
        return cached
    with _repair_lock(cache_root / ".locks" / f"{repair_key}.lock"):
        cached = _cached_repair(metadata, output)
        if cached is not None:
            return cached
        result = _repair_collision_mesh_uncached(source, cache_dir=cache_dir)
        if result.path is None or result.status != "repaired":
            return result
        payload = asdict(result)
        payload["path"] = output.name
        temp = metadata.with_suffix(".json.tmp")
        temp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(temp, metadata)
        _record_attempt_event("published", repair_key)
        return result
