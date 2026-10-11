from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from collision.kinematics import _motion_transform, _origin_to_matrix
from collision.policy import BenchPolicy
from collision.urdf import (
    GeometrySpec,
    JointSpec,
    Origin,
    RobotSpec,
    _resolve_mesh_source,
    parse_urdf,
)

_MOVABLE_TYPES = {"revolute", "continuous", "prismatic"}


class ModelRefusal(ValueError):
    pass


@dataclass(frozen=True)
class EvalJoint:
    index: int
    key: str
    source_name: str
    joint_type: str
    parent: str
    child: str
    origin: Origin
    axis: tuple[float, float, float]
    evaluation_range: tuple[float, float]
    continuous: bool
    locked: bool
    declared_mimic: object | None = None

    @property
    def closed(self) -> float:
        lower, upper = self.evaluation_range
        return min(max(0.0, lower), upper)

    @property
    def far(self) -> float:
        lower, upper = self.evaluation_range
        return upper if abs(upper - self.closed) >= abs(lower - self.closed) else lower


@dataclass
class GeometryPiece:
    key: str
    link: str
    body: str
    source_kind: str
    mesh: Any
    used_visual_fallback: bool


@dataclass
class EvalModel:
    source_path: Path
    robot: RobotSpec
    joints: tuple[EvalJoint, ...]
    pieces: tuple[GeometryPiece, ...]
    body_for_link: dict[str, str]
    roots: tuple[str, ...]
    declared_mimic_count: int

    @property
    def joint_by_key(self) -> dict[str, EvalJoint]:
        return {joint.key: joint for joint in self.joints}

    @property
    def movable_joints(self) -> tuple[EvalJoint, ...]:
        return tuple(
            (
                joint
                for joint in self.joints
                if joint.joint_type in _MOVABLE_TYPES and (not joint.locked)
            )
        )

    def closed_pose(self) -> dict[str, float]:
        return {
            joint.key: joint.closed for joint in self.joints if joint.joint_type in _MOVABLE_TYPES
        }

    def forward_kinematics(self, pose: dict[str, float]) -> dict[str, np.ndarray]:
        children: dict[str, list[EvalJoint]] = {}
        for joint in self.joints:
            children.setdefault(joint.parent, []).append(joint)
        transforms: dict[str, np.ndarray] = {}

        def visit(link: str, world: np.ndarray) -> None:
            transforms[link] = world
            for joint in children.get(link, []):
                source_joint = JointSpec(
                    name=joint.source_name,
                    joint_type=joint.joint_type,
                    parent=joint.parent,
                    child=joint.child,
                    origin=joint.origin,
                    axis=joint.axis,
                    limit={},
                    dynamics={},
                    mimic=None,
                )
                q = float(pose.get(joint.key, joint.closed))
                visit(
                    joint.child,
                    world @ _origin_to_matrix(joint.origin) @ _motion_transform(source_joint, q),
                )

        for root in self.roots:
            visit(root, np.eye(4))
        return transforms


class _UnionFind:
    def __init__(self, names: list[str]) -> None:
        self.parent = {name: name for name in names}

    def find(self, name: str) -> str:
        parent = self.parent[name]
        if parent != name:
            self.parent[name] = self.find(parent)
        return self.parent[name]

    def union(self, left: str, right: str) -> None:
        a, b = (self.find(left), self.find(right))
        if a != b:
            self.parent[b] = a


def _validate_structure(robot: RobotSpec) -> tuple[str, ...]:
    links = set(robot.links)
    parent_for_child: dict[str, str] = {}
    children: dict[str, list[str]] = {}
    for index, joint in enumerate(robot.joints):
        if joint.parent not in links or joint.child not in links:
            raise ModelRefusal(f"joint {joint.name!r} at index {index} references a missing link")
        if joint.child in parent_for_child:
            raise ModelRefusal(f"multi-parent link {joint.child!r}")
        parent_for_child[joint.child] = joint.parent
        children.setdefault(joint.parent, []).append(joint.child)
    roots = tuple(sorted(links - set(parent_for_child)))
    if not roots and links:
        raise ModelRefusal("joint graph has no root (cycle)")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(link: str) -> None:
        if link in visiting:
            raise ModelRefusal("joint graph contains a cycle")
        if link in visited:
            return
        visiting.add(link)
        for child in children.get(link, []):
            visit(child)
        visiting.remove(link)
        visited.add(link)

    for root in roots:
        visit(root)
    if visited != links:
        raise ModelRefusal("joint graph contains an unreachable cycle")
    return roots


def _eval_joint(joint: JointSpec, index: int, policy: BenchPolicy) -> EvalJoint:
    continuous = joint.joint_type == "continuous"
    lower = float(joint.limit.get("lower", -math.inf if continuous else 0.0))
    upper = float(joint.limit.get("upper", math.inf if continuous else 0.0))
    if not (math.isfinite(lower) and math.isfinite(upper)):
        continuous = True
        lower, upper = (-math.pi, math.pi)
    if lower > upper:
        raise ModelRefusal(
            f"joint {joint.name!r} at index {index} has lower > upper ({lower} > {upper})"
        )
    locked = upper - lower <= policy.locked_range_tolerance
    return EvalJoint(
        index=index,
        key=f"{joint.name}#{index}",
        source_name=joint.name,
        joint_type=joint.joint_type,
        parent=joint.parent,
        child=joint.child,
        origin=joint.origin,
        axis=joint.axis,
        evaluation_range=(lower, upper),
        continuous=continuous,
        locked=locked,
        declared_mimic=joint.mimic,
    )


def _as_meshes(value: Any) -> list[Any]:
    import trimesh

    if isinstance(value, trimesh.Trimesh):
        return [value]
    if isinstance(value, trimesh.Scene):
        return [mesh for mesh in value.dump() if isinstance(mesh, trimesh.Trimesh)]
    return []


def _cylinder_sections(radius: float, tolerance: float) -> int:
    if radius <= tolerance:
        return 12
    cosine = max(-1.0, min(1.0, 1.0 - tolerance / radius))
    return max(12, min(512, int(math.ceil(math.pi / math.acos(cosine)))))


def _geometry_meshes(geometry: GeometrySpec, urdf_dir: Path, policy: BenchPolicy) -> list[Any]:
    import trimesh

    attrs = geometry.attrs
    if geometry.kind == "mesh":
        filename = attrs.get("filename")
        if not filename:
            raise ModelRefusal("mesh geometry is missing filename")
        source = _resolve_mesh_source(filename, urdf_dir=urdf_dir)
        meshes = _as_meshes(trimesh.load(source, process=False))
        raw_scale = attrs.get("scale")
        if raw_scale:
            values = tuple((float(value) for value in raw_scale.split()))
            scale = values * 3 if len(values) == 1 else values
            if len(scale) != 3:
                raise ModelRefusal(f"invalid mesh scale {raw_scale!r}")
            for mesh in meshes:
                mesh.apply_scale(scale)
        return meshes
    if geometry.kind == "box":
        extents = [float(value) for value in attrs.get("size", "1 1 1").split()]
        return [trimesh.creation.box(extents=extents)]
    if geometry.kind == "sphere":
        radius = float(attrs.get("radius", "1"))
        ratio = max(radius / policy.primitive_chord_tolerance_m, 1.0)
        subdivisions = max(1, min(6, int(math.ceil(math.log2(ratio) / 2.0))))
        return [trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)]
    if geometry.kind == "cylinder":
        radius = float(attrs.get("radius", "1"))
        return [
            trimesh.creation.cylinder(
                radius=radius,
                height=float(attrs.get("length", "1")),
                sections=_cylinder_sections(radius, policy.primitive_chord_tolerance_m),
            )
        ]
    raise ModelRefusal(f"unsupported URDF geometry kind {geometry.kind!r}")


def _body_map(robot: RobotSpec) -> dict[str, str]:
    union = _UnionFind(list(robot.links))
    for joint in robot.joints:
        if joint.joint_type == "fixed":
            union.union(joint.parent, joint.child)
    groups: dict[str, list[str]] = {}
    for link in robot.links:
        groups.setdefault(union.find(link), []).append(link)
    canonical = {root: sorted(members)[0] for root, members in groups.items()}
    return {link: canonical[union.find(link)] for link in robot.links}


def _load_pieces(
    robot: RobotSpec, urdf_dir: Path, body_for_link: dict[str, str], policy: BenchPolicy
) -> tuple[GeometryPiece, ...]:
    pieces: list[GeometryPiece] = []
    for link_name, link in robot.links.items():
        sources = list(link.collisions)
        visual_fallback = not sources
        if visual_fallback:
            sources = list(link.visuals)
        for geometry_index, source in enumerate(sources):
            for piece_index, mesh in enumerate(_geometry_meshes(source.geometry, urdf_dir, policy)):
                mesh = mesh.copy()
                mesh.apply_transform(_origin_to_matrix(source.origin))
                mesh.remove_unreferenced_vertices()
                if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                    raise ModelRefusal(f"empty geometry on link {link_name!r}")
                pieces.append(
                    GeometryPiece(
                        key=f"{link_name}#{geometry_index}:{piece_index}",
                        link=link_name,
                        body=body_for_link[link_name],
                        source_kind=source.geometry.kind,
                        mesh=mesh,
                        used_visual_fallback=visual_fallback,
                    )
                )
    return tuple(pieces)


def load_eval_model(urdf_path: Path, policy: BenchPolicy) -> EvalModel:
    urdf_path = Path(urdf_path).expanduser().resolve()
    robot = parse_urdf(urdf_path)
    roots = _validate_structure(robot)
    joints = tuple((_eval_joint(joint, index, policy) for index, joint in enumerate(robot.joints)))
    body_for_link = _body_map(robot)
    model = EvalModel(
        source_path=urdf_path,
        robot=robot,
        joints=joints,
        pieces=_load_pieces(robot, urdf_path.parent, body_for_link, policy),
        body_for_link=body_for_link,
        roots=roots,
        declared_mimic_count=sum((joint.mimic is not None for joint in robot.joints)),
    )
    return model


def resolve_prediction(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file() and path.name == "object.urdf":
        return path
    if path.is_dir():
        candidates = [
            path / name
            for name in (
                "object.urdf",
                "exports/object.urdf",
                "workspace/exports/object.urdf",
            )
        ]
        found = [candidate for candidate in candidates if candidate.is_file()]
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise ValueError("Multiple exports found; pass the desired URDF file explicitly")
    raise FileNotFoundError(
        "Expected an Artisan object.urdf, export directory, workspace, or run directory"
    )
