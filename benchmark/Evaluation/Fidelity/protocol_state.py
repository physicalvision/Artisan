from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass(frozen=True)
class VisualSpec:
    link: str
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    visual_tf: np.ndarray


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_tf: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


def _parse_floats(text: str | None, default: Iterable[float]) -> np.ndarray:
    if not text:
        return np.asarray(tuple(default), dtype=np.float64)
    return np.asarray([float(x) for x in text.split()], dtype=np.float64)


def _origin_tf(elem: ET.Element | None) -> np.ndarray:
    xyz = _parse_floats(elem.get("xyz") if elem is not None else None, (0.0, 0.0, 0.0))
    rpy = _parse_floats(elem.get("rpy") if elem is not None else None, (0.0, 0.0, 0.0))
    tf = trimesh.transformations.euler_matrix(rpy[0], rpy[1], rpy[2], axes="sxyz")
    tf[:3, 3] = xyz
    return tf


def _box_mesh(size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.creation.box(extents=size)
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def _cylinder_mesh(radius: float, length: float) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=32)
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def _sphere_mesh(radius: float) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def _load_mesh_geometry(
    filename: str, urdf_dir: Path, scale: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    mesh_path = Path(filename)
    if not mesh_path.is_absolute():
        mesh_path = urdf_dir / mesh_path
    loaded = trimesh.load(mesh_path, process=False, force="scene")
    if isinstance(loaded, trimesh.Scene):
        meshes = []
        for node in loaded.graph.nodes_geometry:
            tf, geom_name = loaded.graph.get(frame_to=node)
            geom = loaded.geometry[geom_name]
            mesh = trimesh.Trimesh(geom.vertices.copy(), geom.faces.copy(), process=False)
            mesh.apply_transform(tf)
            meshes.append(mesh)
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = loaded
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if scale is not None:
        vertices = vertices * scale[None, :]
    return vertices, np.asarray(mesh.faces, dtype=np.int64)


def parse_urdf(urdf_path: Path) -> tuple[list[str], list[VisualSpec], list[JointSpec]]:
    root = ET.parse(urdf_path).getroot()
    links = [link.get("name") for link in root.findall("link") if link.get("name")]
    visuals: list[VisualSpec] = []
    urdf_dir = urdf_path.parent

    for link in root.findall("link"):
        link_name = link.get("name")
        if not link_name:
            continue
        for idx, visual in enumerate(link.findall("visual")):
            geom = visual.find("geometry")
            if geom is None:
                continue
            if (box := geom.find("box")) is not None:
                vertices, faces = _box_mesh(_parse_floats(box.get("size"), (1.0, 1.0, 1.0)))
            elif (cyl := geom.find("cylinder")) is not None:
                vertices, faces = _cylinder_mesh(float(cyl.get("radius")), float(cyl.get("length")))
            elif (sph := geom.find("sphere")) is not None:
                vertices, faces = _sphere_mesh(float(sph.get("radius")))
            elif (mesh := geom.find("mesh")) is not None:
                scale = (
                    _parse_floats(mesh.get("scale"), (1.0, 1.0, 1.0)) if mesh.get("scale") else None
                )
                vertices, faces = _load_mesh_geometry(mesh.get("filename"), urdf_dir, scale)
            else:
                continue
            visuals.append(
                VisualSpec(
                    link=link_name,
                    name=visual.get("name") or f"{link_name}_visual_{idx}",
                    vertices=vertices,
                    faces=faces,
                    visual_tf=_origin_tf(visual.find("origin")),
                )
            )

    joints: list[JointSpec] = []
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        lower = float(limit.get("lower", "0")) if limit is not None else 0.0
        upper = float(limit.get("upper", "0")) if limit is not None else 0.0
        joints.append(
            JointSpec(
                name=joint.get("name") or "",
                joint_type=joint.get("type") or "fixed",
                parent=joint.find("parent").get("link"),
                child=joint.find("child").get("link"),
                origin_tf=_origin_tf(joint.find("origin")),
                axis=_parse_floats(
                    joint.find("axis").get("xyz") if joint.find("axis") is not None else None,
                    (0.0, 0.0, 1.0),
                ),
                lower=lower,
                upper=upper,
            )
        )

    return links, visuals, joints
