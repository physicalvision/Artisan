from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from assets import mesh_groups
from pipeline.articulated_glb import read_glb_json


@dataclass
class Samples:
    close: np.ndarray
    open: np.ndarray
    ids: np.ndarray
    names: list[str]


def mesh_transforms(path: Path) -> dict[str, np.ndarray]:
    document = read_glb_json(path)
    nodes = document["nodes"]
    transforms: dict[str, np.ndarray] = {}
    visited: set[int] = set()

    def visit(index: int, parent: np.ndarray) -> None:
        if index in visited:
            raise ValueError("GLB nodes must form a tree")
        visited.add(index)
        node = nodes[index]
        if "matrix" in node:
            local = np.asarray(node["matrix"], dtype=float).reshape(4, 4).T
        else:
            x, y, z, w = node.get("rotation", (0.0, 0.0, 0.0, 1.0))
            local = trimesh.transformations.quaternion_matrix((w, x, y, z))
            local[:3, :3] *= np.asarray(node.get("scale", (1.0, 1.0, 1.0)))
            local[:3, 3] = node.get("translation", (0.0, 0.0, 0.0))
        world = parent @ local
        if not np.isfinite(world).all():
            raise ValueError("Nonfinite GLB node transform")
        if "mesh" in node:
            name = node.get("name")
            if not name or name in transforms:
                raise ValueError("GLB mesh nodes require unique names")
            transforms[name] = world
        for child in node.get("children", []):
            visit(child, world)

    for root in document["scenes"][document.get("scene", 0)]["nodes"]:
        visit(root, np.eye(4))
    return transforms


def sample_pair(
    close: Path, opened: Path, count: int, seed: int, groups: dict[str, str] | None = None
) -> Samples:
    scenes = [trimesh.load(p, force="scene", process=False) for p in (close, opened)]
    world_transforms = [mesh_transforms(path) for path in (close, opened)]
    names = sorted(scenes[0].graph.nodes_geometry)
    if not names or set(names) != set(scenes[1].graph.nodes_geometry):
        raise ValueError("Close/open mesh-node sets differ or are empty")
    ownership = mesh_groups(close, groups)
    if ownership != mesh_groups(opened, groups) or set(ownership) != set(names):
        raise ValueError("Close/open part ownership differs")
    parts = sorted(set(ownership.values()))
    if groups is None:
        root = next(
            (name for name in parts if "body" in name.lower() or name == "part_root"), parts[0]
        )
        parts = [root, *[name for name in parts if name != root]]
    vertices, faces, areas = [], [], []
    for name in names:
        pair = []
        pair_faces = []
        for scene, transforms in zip(scenes, world_transforms, strict=True):
            _, key = scene.graph.get(name)
            mesh = scene.geometry[key]
            pair.append(trimesh.transform_points(mesh.vertices, transforms[name]))
            pair_faces.append(np.asarray(mesh.faces))
        if pair[0].shape != pair[1].shape or not np.array_equal(*pair_faces):
            raise ValueError(f"Close/open mesh topology differs: {name}")
        if not np.isfinite(pair).all():
            raise ValueError("Nonfinite mesh vertices")
        vertices.append(pair)
        faces.append(pair_faces[0])
        areas.append(trimesh.triangles.area(pair[0][pair_faces[0]]))
    weights = np.array([a.sum() for a in areas])
    if weights.sum() <= 0:
        raise ValueError("Zero surface area")
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(count, weights / weights.sum())
    states: list[list[np.ndarray]] = [[], []]
    ids = []
    for name, pair, triangles, area, n in zip(names, vertices, faces, areas, counts):
        if not n:
            continue
        selected = rng.choice(len(triangles), size=n, p=area / area.sum())
        uv = rng.random((n, 2))
        flipped = uv.sum(axis=1) > 1
        uv[flipped] = 1 - uv[flipped]
        bary = np.column_stack((uv, 1 - uv.sum(axis=1)))
        for index in range(2):
            states[index].append((pair[index][triangles[selected]] * bary[:, :, None]).sum(axis=1))
        ids.append(np.full(n, parts.index(ownership[name]), dtype=np.int32))
    all_ids = np.concatenate(ids)
    if len(np.unique(all_ids)) != len(parts):
        raise ValueError("Sampling missed a part; increase --points")
    return Samples(np.concatenate(states[0]), np.concatenate(states[1]), all_ids, parts)


def chamfer_squared(a: np.ndarray, b: np.ndarray) -> float:
    ab = cKDTree(b).query(a, workers=1)[0]
    ba = cKDTree(a).query(b, workers=1)[0]
    return float((np.mean(ab**2) + np.mean(ba**2)) / 2)


def aabb_iou(a: np.ndarray, b: np.ndarray) -> float:
    lo_a, hi_a = a.min(axis=0), a.max(axis=0)
    lo_b, hi_b = b.min(axis=0), b.max(axis=0)
    overlap = np.maximum(0, np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b)).prod()
    union = (hi_a - lo_a).prod() + (hi_b - lo_b).prod() - overlap
    return float(overlap / union) if union > 0 else 0.0


def evaluate_3d(pred: Samples, gt: Samples, mapping: dict[int, int] | None = None) -> dict:
    if mapping is None:
        pred_centers = np.array(
            [pred.close[pred.ids == i].mean(axis=0) for i in range(len(pred.names))]
        )
        gt_centers = np.array([gt.close[gt.ids == i].mean(axis=0) for i in range(len(gt.names))])
        rows, cols = linear_sum_assignment(
            np.linalg.norm(gt_centers[:, None] - pred_centers[None, :], axis=2)
        )
        mapping = dict(zip(rows.tolist(), cols.tolist()))
    if any(
        i < 0 or i >= len(gt.names) or j < 0 or j >= len(pred.names) for i, j in mapping.items()
    ):
        raise ValueError("Part matching refers to an unknown part")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Part matching must be one-to-one")
    penalty = float((np.linalg.norm(np.ptp(pred.close, axis=0)) / 2) ** 2)
    result = {
        "matching": {gt.names[i]: pred.names[j] for i, j in mapping.items()},
    }
    for state in ("close", "open"):
        p, g = getattr(pred, state), getattr(gt, state)
        cd, iou = [], []
        for i, name in enumerate(gt.names):
            if i not in mapping:
                distance, overlap = penalty, 0.0
            else:
                a, b = g[gt.ids == i], p[pred.ids == mapping[i]]
                distance, overlap = chamfer_squared(a, b), aabb_iou(a, b)
            cd.append(distance)
            iou.append(overlap)
        result[state] = {
            "object_cd": chamfer_squared(p, g),
            "part_cd": float(np.mean(cd)),
            "part_miou": float(np.mean(iou)),
        }
    return result
