from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from collision.policy import BenchPolicy


class NarrowPhaseKind(str, Enum):
    SEPARATED = "separated"
    CROSSING = "crossing"
    CONTAINED = "contained"
    CONTAINMENT_UNCERTAIN = "containment_uncertain"


@dataclass(frozen=True)
class NarrowPhaseResult:
    kind: NarrowPhaseKind
    min_distance: float | None = None
    d_local: float | None = None
    d_sep: float | None = None
    capped: bool = False
    support_gap: float | None = None
    contact_count: int = 0
    aabb_overlap_volume: float | None = None
    reason: str | None = None


@dataclass(frozen=True)
class _WorldMeshData:
    mesh: Any
    triangle_lower: np.ndarray
    triangle_upper: np.ndarray
    bounds: np.ndarray


def _world_mesh(mesh: Any, transform: np.ndarray) -> Any:
    result = mesh.copy()
    result.apply_transform(np.asarray(transform, dtype=float))
    return result


def _bounds_corners(bounds: np.ndarray) -> np.ndarray:
    lower, upper = np.asarray(bounds, dtype=float)
    return np.asarray(
        [
            [x, y, z]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ],
        dtype=float,
    )


def transformed_bounds(mesh: Any, transform: np.ndarray) -> np.ndarray:
    corners = _bounds_corners(np.asarray(mesh.bounds, dtype=float))
    world = corners @ transform[:3, :3].T + transform[:3, 3]
    return np.asarray([world.min(axis=0), world.max(axis=0)])


def _fcl_object(mesh: Any) -> Any:
    import fcl

    vertices = np.ascontiguousarray(np.asarray(mesh.vertices, dtype=float))
    faces = np.ascontiguousarray(np.asarray(mesh.faces, dtype=np.int32))
    model = fcl.BVHModel()
    model.beginModel(len(vertices), len(faces))
    model.addSubModel(vertices, faces)
    model.endModel()
    return fcl.CollisionObject(model, fcl.Transform())


def _collides(left: Any, right: Any, *, contacts: int = 1) -> tuple[bool, list[Any]]:
    import fcl

    request = fcl.CollisionRequest(num_max_contacts=contacts, enable_contact=contacts > 1)
    result = fcl.CollisionResult()
    hit = int(fcl.collide(left, right, request, result)) > 0
    return (hit, list(getattr(result, "contacts", []) or []))


def _distance(left: Any, right: Any) -> float:
    import fcl

    request = fcl.DistanceRequest(enable_nearest_points=False)
    result = fcl.DistanceResult()
    return max(0.0, float(fcl.distance(left, right, request, result)))


def _contains_bounds(outer: np.ndarray, inner: np.ndarray, tolerance: float) -> bool:
    return bool(
        np.all(outer[0] <= inner[0] + tolerance) and np.all(outer[1] >= inner[1] - tolerance)
    )


def _solid_winding(mesh: Any, points: np.ndarray, *, chunk_size: int = 128) -> np.ndarray:
    triangles = np.asarray(mesh.triangles, dtype=float)
    output = np.empty(len(points), dtype=float)
    for start in range(0, len(points), chunk_size):
        batch = np.asarray(points[start : start + chunk_size], dtype=float)
        a = triangles[None, :, 0, :] - batch[:, None, :]
        b = triangles[None, :, 1, :] - batch[:, None, :]
        c = triangles[None, :, 2, :] - batch[:, None, :]
        la = np.linalg.norm(a, axis=2)
        lb = np.linalg.norm(b, axis=2)
        lc = np.linalg.norm(c, axis=2)
        numerator = np.einsum("bti,bti->bt", a, np.cross(b, c))
        denominator = (
            la * lb * lc
            + np.einsum("bti,bti->bt", a, b) * lc
            + np.einsum("bti,bti->bt", b, c) * la
            + np.einsum("bti,bti->bt", c, a) * lb
        )
        angles = 2.0 * np.arctan2(numerator, denominator)
        output[start : start + len(batch)] = angles.sum(axis=1) / (4.0 * math.pi)
    return output


def _segment_hits_other_face(
    triangles: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    seed_face: int,
    tolerance: float,
) -> bool:
    direction = end - start
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    determinant = np.einsum("ij,ij->i", edge1, h)
    valid = np.abs(determinant) > tolerance
    inverse = np.zeros_like(determinant)
    inverse[valid] = 1.0 / determinant[valid]
    s = start - triangles[:, 0]
    u = inverse * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, edge1)
    v = inverse * np.einsum("ij,j->i", q, direction)
    distance = inverse * np.einsum("ij,ij->i", edge2, q)
    hits = valid & (u >= -tolerance) & (v >= -tolerance) & (u + v <= 1.0 + tolerance)
    hits &= (distance > tolerance) & (distance < 1.0 - tolerance)
    hits[seed_face] = False
    return bool(np.any(hits))


def _certified_interior_points(mesh: Any, policy: BenchPolicy) -> np.ndarray | None:
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        return None
    components = list(mesh.split(only_watertight=False))
    certified: list[np.ndarray] = []
    for component in components:
        if not component.is_watertight or not component.is_winding_consistent:
            return None
        triangles = np.asarray(component.triangles, dtype=float)
        centers = np.asarray(component.triangles_center, dtype=float)
        normals = np.asarray(component.face_normals, dtype=float)
        areas = np.asarray(component.area_faces, dtype=float)
        inward_sign = -1.0 if float(component.volume) >= 0.0 else 1.0
        scale = max(float(np.linalg.norm(component.extents)), 1e-09)
        offset = max(
            policy.containment_offset_ratio * scale,
            10.0 * policy.containment_segment_tolerance_m,
        )
        selected: list[np.ndarray] = []
        for face_index in np.argsort(-areas):
            point = centers[face_index] + inward_sign * offset * normals[face_index]
            if _segment_hits_other_face(
                triangles,
                centers[face_index],
                point,
                int(face_index),
                policy.containment_segment_tolerance_m,
            ):
                continue
            own_winding = abs(float(_solid_winding(component, point[None, :])[0]))
            if abs(own_winding - 1.0) > policy.containment_self_tolerance:
                continue
            selected.append(point)
            if len(selected) >= policy.containment_points:
                break
        if len(selected) < policy.containment_points:
            return None
        certified.extend(selected)
    return np.asarray(certified, dtype=float)


def _containment_verdict(outer: Any, inner: Any, policy: BenchPolicy) -> NarrowPhaseKind:
    if not outer.is_watertight or not outer.is_winding_consistent:
        return NarrowPhaseKind.CONTAINMENT_UNCERTAIN
    component_verdicts: list[str] = []
    for component in inner.split(only_watertight=False):
        points = _certified_interior_points(component, policy)
        if points is None:
            component_verdicts.append("uncertain")
            continue
        winding = np.abs(_solid_winding(outer, points))
        inside = winding > 0.5 + policy.containment_winding_tolerance
        outside = winding < 0.5 - policy.containment_winding_tolerance
        if bool(np.all(inside)):
            component_verdicts.append("inside")
        elif bool(np.all(outside)):
            component_verdicts.append("outside")
        else:
            component_verdicts.append("uncertain")
    if "inside" in component_verdicts:
        return NarrowPhaseKind.CONTAINED
    if component_verdicts and all((value == "outside" for value in component_verdicts)):
        return NarrowPhaseKind.SEPARATED
    return NarrowPhaseKind.CONTAINMENT_UNCERTAIN


def _set_transform(obj: Any, transform: np.ndarray) -> None:
    obj.setRotation(np.ascontiguousarray(transform[:3, :3], dtype=float))
    obj.setTranslation(np.ascontiguousarray(transform[:3, 3], dtype=float))


def _shifted_collision(
    obj_a: Any, obj_b: Any, right_transform: np.ndarray, shift: np.ndarray
) -> bool:
    obj_b.setTranslation(np.asarray(right_transform[:3, 3] + shift, dtype=float))
    hit, _ = _collides(obj_a, obj_b)
    return hit


def _separation_along(
    obj_a: Any,
    obj_b: Any,
    right_transform: np.ndarray,
    direction: np.ndarray,
    upper: float,
    policy: BenchPolicy,
) -> float | None:
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return None
    direction = direction / norm
    steps = np.linspace(upper / policy.separation_scan_steps, upper, policy.separation_scan_steps)
    collisions = [
        _shifted_collision(obj_a, obj_b, right_transform, direction * step) for step in steps
    ]
    if collisions[-1]:
        return None
    last_hit = max((index for index, value in enumerate(collisions) if value), default=-1)
    lower = 0.0 if last_hit < 0 else float(steps[last_hit])
    upper_bound = float(steps[last_hit + 1])
    while upper_bound - lower > policy.separation_resolution_m:
        middle = 0.5 * (lower + upper_bound)
        if _shifted_collision(obj_a, obj_b, right_transform, direction * middle):
            lower = middle
        else:
            upper_bound = middle
    if _shifted_collision(obj_a, obj_b, right_transform, direction * upper_bound):
        return None
    return upper_bound


def _separation_depth_objects(
    left_obj: Any,
    right_obj: Any,
    left_transform: np.ndarray,
    right_transform: np.ndarray,
    left_bounds: np.ndarray,
    right_bounds: np.ndarray,
    contacts: list[Any],
    policy: BenchPolicy,
) -> tuple[float, bool]:
    _set_transform(left_obj, left_transform)
    _set_transform(right_obj, right_transform)
    overlap = np.minimum(left_bounds[1], right_bounds[1]) - np.maximum(
        left_bounds[0], right_bounds[0]
    )
    cap = max(float(np.min(np.maximum(overlap, 0.0))), policy.separation_resolution_m)
    directions = [
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
    ]
    if contacts:
        mean_normal = np.mean(
            [np.asarray(contact.normal, dtype=float) for contact in contacts], axis=0
        )
        directions.extend((mean_normal, -mean_normal))
    center_delta = right_bounds.mean(axis=0) - left_bounds.mean(axis=0)
    directions.extend((center_delta, -center_delta))
    best: float | None = None
    for direction in directions:
        value = _separation_along(left_obj, right_obj, right_transform, direction, cap, policy)
        if value is not None and (best is None or value < best):
            best = value
    _set_transform(right_obj, right_transform)
    return (cap, True) if best is None else (best, False)


def _cropped_faces(
    triangle_lower: np.ndarray,
    triangle_upper: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, bytes, np.ndarray] | None:
    lower, upper = (triangle_lower, triangle_upper)
    keep = np.all((lower <= center + radius) & (upper >= center - radius), axis=1)
    if not np.any(keep):
        return None
    key = np.packbits(keep).tobytes()
    bounds = np.asarray([triangle_lower[keep].min(axis=0), triangle_upper[keep].max(axis=0)])
    return (np.flatnonzero(keep), key, bounds)


class FCLBackend:
    _LOCAL_BVH_CACHE_ENTRIES = 128

    def __init__(self, policy: BenchPolicy) -> None:
        self.policy = policy
        self._object_cache: dict[int, tuple[Any, Any]] = {}
        self._world_cache: dict[tuple[int, bytes], tuple[Any, _WorldMeshData]] = {}
        self._local_object_cache: OrderedDict[tuple[int, bytes], tuple[Any, Any]] = OrderedDict()
        self.local_bvh_builds = 0
        self.local_bvh_cache_hits = 0

    def begin_pose(self) -> None:
        self._world_cache.clear()

    def _world_data(self, mesh: Any, transform: np.ndarray) -> _WorldMeshData:
        key = (id(mesh), np.ascontiguousarray(transform, dtype=float).tobytes())
        cached = self._world_cache.get(key)
        if cached is not None and cached[0] is mesh:
            return cached[1]
        world = _world_mesh(mesh, transform)
        triangles = np.asarray(world.triangles)
        lower, upper = (triangles.min(axis=1), triangles.max(axis=1))
        data = _WorldMeshData(
            mesh=world,
            triangle_lower=lower,
            triangle_upper=upper,
            bounds=np.asarray([lower.min(axis=0), upper.max(axis=0)]),
        )
        self._world_cache[key] = (mesh, data)
        return data

    def _local_object(self, mesh: Any, face_key: bytes, face_indices: np.ndarray) -> Any:
        key = (id(mesh), face_key)
        cached = self._local_object_cache.pop(key, None)
        if cached is not None and cached[0] is mesh:
            self.local_bvh_cache_hits += 1
            self._local_object_cache[key] = cached
            return cached[1]
        local_mesh = mesh.submesh([face_indices], append=True, repair=False)
        obj = _fcl_object(local_mesh)
        self.local_bvh_builds += 1
        self._local_object_cache[key] = (mesh, obj)
        while len(self._local_object_cache) > self._LOCAL_BVH_CACHE_ENTRIES:
            self._local_object_cache.popitem(last=False)
        return obj

    def _object_for(self, mesh: Any) -> Any:
        key = id(mesh)
        cached = self._object_cache.get(key)
        if cached is None or cached[0] is not mesh:
            cached = (mesh, _fcl_object(mesh))
            self._object_cache[key] = cached
        return cached[1]

    def classify_meshes(
        self,
        left_mesh: Any,
        left_transform: np.ndarray,
        right_mesh: Any,
        right_transform: np.ndarray,
        *,
        scene_diagonal: float,
    ) -> NarrowPhaseResult:
        left_transform = np.asarray(left_transform, dtype=float)
        right_transform = np.asarray(right_transform, dtype=float)
        left_bounds = transformed_bounds(left_mesh, left_transform)
        right_bounds = transformed_bounds(right_mesh, right_transform)
        left_obj, right_obj = (
            self._object_for(left_mesh),
            self._object_for(right_mesh),
        )
        _set_transform(left_obj, left_transform)
        _set_transform(right_obj, right_transform)
        crossing, contacts = _collides(left_obj, right_obj, contacts=self.policy.max_contacts)
        if crossing:
            d_sep, capped = _separation_depth_objects(
                left_obj,
                right_obj,
                left_transform,
                right_transform,
                left_bounds,
                right_bounds,
                contacts,
                self.policy,
            )
            contact_points = np.asarray(
                [np.asarray(contact.pos, dtype=float) for contact in contacts],
                dtype=float,
            )
            d_local = d_sep
            if len(contact_points):
                left_world = self._world_data(left_mesh, left_transform)
                right_world = self._world_data(right_mesh, right_transform)
                radius = max(
                    self.policy.local_window_ratio * scene_diagonal,
                    self.policy.separation_resolution_m,
                )
                local_values: list[float] = []
                seen_windows: set[tuple[bytes, bytes]] = set()
                for center in contact_points[
                    np.linspace(0, len(contact_points) - 1, min(12, len(contact_points))).astype(
                        int
                    )
                ]:
                    cropped_left = _cropped_faces(
                        left_world.triangle_lower,
                        left_world.triangle_upper,
                        center,
                        radius,
                    )
                    cropped_right = _cropped_faces(
                        right_world.triangle_lower,
                        right_world.triangle_upper,
                        center,
                        radius,
                    )
                    if cropped_left is None or cropped_right is None:
                        continue
                    left_faces, left_key, local_left_bounds = cropped_left
                    right_faces, right_key, local_right_bounds = cropped_right
                    window_key = (left_key, right_key)
                    if window_key in seen_windows:
                        continue
                    seen_windows.add(window_key)
                    local_left_obj = self._local_object(left_mesh, left_key, left_faces)
                    local_right_obj = self._local_object(right_mesh, right_key, right_faces)
                    _set_transform(local_left_obj, left_transform)
                    _set_transform(local_right_obj, right_transform)
                    local_hit, local_contacts = _collides(
                        local_left_obj, local_right_obj, contacts=64
                    )
                    if local_hit:
                        value, _ = _separation_depth_objects(
                            local_left_obj,
                            local_right_obj,
                            left_transform,
                            right_transform,
                            local_left_bounds,
                            local_right_bounds,
                            local_contacts,
                            self.policy,
                        )
                        local_values.append(value)
                if local_values:
                    d_local = max(local_values)
            return NarrowPhaseResult(
                kind=NarrowPhaseKind.CROSSING,
                d_local=d_local,
                d_sep=d_sep,
                capped=capped,
                support_gap=0.0,
                contact_count=len(contacts),
                aabb_overlap_volume=float(
                    np.prod(
                        np.maximum(
                            np.minimum(left_bounds[1], right_bounds[1])
                            - np.maximum(left_bounds[0], right_bounds[0]),
                            0.0,
                        )
                    )
                ),
            )
        tolerance = self.policy.containment_segment_tolerance_m
        left_world: Any | None = None
        right_world: Any | None = None
        candidates: list[tuple[Any, Any]] = []
        if _contains_bounds(left_bounds, right_bounds, tolerance):
            left_world = self._world_data(left_mesh, left_transform).mesh
            right_world = self._world_data(right_mesh, right_transform).mesh
            candidates.append((left_world, right_world))
        if _contains_bounds(right_bounds, left_bounds, tolerance):
            if left_world is None:
                left_world = self._world_data(left_mesh, left_transform).mesh
            if right_world is None:
                right_world = self._world_data(right_mesh, right_transform).mesh
            candidates.append((right_world, left_world))
        if candidates:
            verdicts = [
                _containment_verdict(outer, inner, self.policy) for outer, inner in candidates
            ]
            if NarrowPhaseKind.CONTAINED in verdicts:
                return NarrowPhaseResult(
                    kind=NarrowPhaseKind.CONTAINED,
                    support_gap=0.0,
                    reason="certified inner component lies inside watertight outer",
                )
            if all((verdict is NarrowPhaseKind.SEPARATED for verdict in verdicts)):
                distance = _distance(left_obj, right_obj)
                return NarrowPhaseResult(
                    kind=NarrowPhaseKind.SEPARATED,
                    min_distance=distance,
                    support_gap=distance,
                )
            return NarrowPhaseResult(
                kind=NarrowPhaseKind.CONTAINMENT_UNCERTAIN,
                support_gap=None,
                reason="nested AABBs without a certifiable containment verdict",
            )
        distance = _distance(left_obj, right_obj)
        return NarrowPhaseResult(
            kind=NarrowPhaseKind.SEPARATED, min_distance=distance, support_gap=distance
        )
