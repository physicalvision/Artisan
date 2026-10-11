from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from pipeline.articulated_glb import (
    pose_articulated_glb,
    read_glb_json,
    rewrite_glb_json,
)


def articulation_metadata(node: dict[str, Any]) -> dict[str, Any]:
    extras = node.get("extras", {})
    return extras.get("articulation", {})


def prediction_exports(path: Path) -> tuple[Path, Path]:
    if path.is_file():
        if path.suffix.lower() != ".glb":
            raise ValueError("Prediction must be an articulated GLB or an Ours export directory")
        glb = path
    else:
        candidates = [
            path / "object.articulated.glb",
            path / "exports/object.articulated.glb",
            path / "workspace/exports/object.articulated.glb",
        ]
        found = [candidate for candidate in candidates if candidate.is_file()]
        if len(found) != 1:
            raise ValueError(
                "Ours input requires exactly one object.articulated.glb; URDF-only input is unsupported"
            )
        glb = found[0]
    urdf = glb.with_name("object.urdf")
    if not urdf.is_file():
        raise FileNotFoundError(f"Articulated GLB requires its object.urdf sidecar: {urdf}")
    document = read_glb_json(glb)
    if not any(
        articulation_metadata(node).get("kind") == "link" for node in document.get("nodes", [])
    ):
        raise ValueError("Prediction GLB requires articulation link metadata")
    return urdf, glb


def vector(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    value = np.array(default if text is None else [float(v) for v in text.split()])
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("Expected three finite coordinates")
    return value


def kinematics(
    path: Path,
    state: str,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    robot = ET.parse(path).getroot()
    links = [link.attrib["name"] for link in robot.findall("link")]
    joints = {joint.attrib["name"]: joint for joint in robot.findall("joint")}
    if len(set(links)) != len(links) or len(joints) != len(robot.findall("joint")):
        raise ValueError("Duplicate link or joint names")
    incoming: dict[str, ET.Element] = {}
    for joint in joints.values():
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        if parent not in links or child not in links or child in incoming:
            raise ValueError("Invalid URDF tree")
        if joint.attrib["type"] not in {"fixed", "revolute", "continuous", "prismatic"}:
            raise ValueError("Unsupported URDF joint type")
        incoming[child] = joint
    roots = set(links) - set(incoming)
    if len(roots) != 1:
        raise ValueError("Expected one URDF root")

    def coordinate(name: str) -> float:
        joint = joints[name]
        kind = joint.attrib["type"]
        if kind == "fixed":
            value = 0.0
        elif kind == "continuous":
            value = 0.0 if state == "close" else math.tau
        else:
            limit = joint.find("limit")
            if limit is None:
                raise ValueError(f"Missing limits: {name}")
            lower, upper = float(limit.attrib["lower"]), float(limit.attrib["upper"])
            if not np.isfinite([lower, upper]).all() or lower > upper:
                raise ValueError(f"Invalid limits: {name}")
            value = lower if state == "close" else upper
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite coordinate: {name}")
        return value

    transforms: dict[str, np.ndarray] = {}
    groups: dict[str, str] = {}

    def visit(link: str, active: frozenset[str] = frozenset()) -> None:
        if link in transforms:
            return
        if link in active:
            raise ValueError("Cyclic URDF tree")
        joint = incoming.get(link)
        if joint is None:
            transforms[link] = np.eye(4)
            groups[link] = link
            return
        parent = joint.find("parent").attrib["link"]
        visit(parent, active | {link})
        origin = joint.find("origin")
        xyz = vector(None if origin is None else origin.get("xyz"), (0.0, 0.0, 0.0))
        rpy = vector(None if origin is None else origin.get("rpy"), (0.0, 0.0, 0.0))
        transform = trimesh.transformations.euler_matrix(*rpy, axes="sxyz")
        transform[:3, 3] = xyz
        motion = np.eye(4)
        kind = joint.attrib["type"]
        if kind != "fixed":
            axis_element = joint.find("axis")
            axis = vector(
                None if axis_element is None else axis_element.get("xyz"), (1.0, 0.0, 0.0)
            )
            if np.linalg.norm(axis) == 0:
                raise ValueError("Zero joint axis")
            axis = axis / np.linalg.norm(axis)
            q = coordinate(joint.attrib["name"])
            if kind == "prismatic":
                motion[:3, 3] = axis * q
            else:
                motion = trimesh.transformations.rotation_matrix(q, axis)
        transforms[link] = transforms[parent] @ transform @ motion
        groups[link] = groups[parent] if kind == "fixed" else link

    for link in links:
        visit(link)
    return transforms, groups


def prepare_prediction(path: Path, output: Path) -> tuple[dict[str, Path], dict[str, str]]:
    urdf, glb = prediction_exports(path)
    compatible = output / "articulation_compatible.glb"

    def aliases(document: dict[str, Any]) -> None:
        for node in document.get("nodes", []):
            if node.get("name") == "__articraft_urdf_z_up__":
                node["name"] = "__artisan_urdf_z_up__"
            extras = node.get("extras", {})
            metadata = articulation_metadata(node)
            if metadata:
                extras["articulation"] = metadata

    rewrite_glb_json(glb, compatible, aliases)
    glb = compatible
    document = read_glb_json(glb)
    link_nodes = {
        node["name"]: articulation_metadata(node)["link"]
        for node in document["nodes"]
        if articulation_metadata(node).get("kind") == "link"
    }
    if not link_nodes:
        raise ValueError("Expected Artisan articulated GLB link metadata")
    poses = {}
    for state in ("close", "open"):
        transforms, groups = kinematics(urdf, state)
        if set(link_nodes.values()) != set(transforms):
            raise ValueError("URDF and articulated GLB link sets differ")
        poses[state] = output / f"prediction_{state}.glb"
        pose_articulated_glb(glb, poses[state], transforms)
    return poses, {node: groups[link] for node, link in link_nodes.items()}


def transform_glb(source: Path, target: Path, matrix: np.ndarray) -> None:
    def update(document: dict[str, Any]) -> None:
        frames = [
            node
            for node in document["nodes"]
            if node.get("name") in {"__artisan_urdf_z_up__", "__articraft_urdf_z_up__"}
        ]
        if frames:
            if len(frames) != 1:
                raise ValueError("Expected one coordinate frame")
            frames[0]["matrix"] = np.asarray(matrix, dtype=np.float64).T.reshape(-1).tolist()
            for key in ("translation", "rotation", "scale"):
                frames[0].pop(key, None)
            return
        scene = document["scenes"][document.get("scene", 0)]
        for index in scene["nodes"]:
            node = document["nodes"][index]
            if "matrix" in node:
                local = np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T
            else:
                x, y, z, w = node.get("rotation", (0.0, 0.0, 0.0, 1.0))
                local = trimesh.transformations.quaternion_matrix((w, x, y, z))
                local[:3, :3] *= np.asarray(node.get("scale", (1.0, 1.0, 1.0)))
                local[:3, 3] = node.get("translation", (0.0, 0.0, 0.0))
            node["matrix"] = (matrix @ local).T.reshape(-1).tolist()
            for key in ("translation", "rotation", "scale"):
                node.pop(key, None)

    rewrite_glb_json(source, target, update)


def mesh_groups(path: Path, link_groups: dict[str, str] | None) -> dict[str, str]:
    document = read_glb_json(path)
    nodes = document["nodes"]
    parents = {}
    for i, node in enumerate(nodes):
        for child in node.get("children", []):
            if child in parents:
                raise ValueError("GLB node has multiple parents")
            parents[child] = i
    result = {}
    for i, node in enumerate(nodes):
        if "mesh" not in node:
            continue
        name = node.get("name")
        if not name or name in result:
            raise ValueError("Mesh nodes require unique names")
        if link_groups is None:
            result[name] = name.split("/", 1)[0]
            continue
        current = i
        visited = set()
        while nodes[current].get("name") not in link_groups:
            if current in visited or current not in parents:
                raise ValueError(f"Mesh has no owning link: {name}")
            visited.add(current)
            current = parents[current]
        result[name] = link_groups[nodes[current]["name"]]
    return result
