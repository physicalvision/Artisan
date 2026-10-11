from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from alignment import _subsample_points, apply_similarity
from protocol_state import parse_urdf
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

KD_TREE_WORKERS = 1
URDF_TO_GLTF = np.array(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)))


def _triangle_sample(
    vertices: np.ndarray, faces: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    tri = vertices[faces]
    face_area = trimesh.triangles.area(tri)
    face_probs = face_area / face_area.sum()
    face_idx = rng.choice(len(faces), size=count, replace=True, p=face_probs)
    weights = rng.random((count, 2))
    over = weights.sum(axis=1) > 1.0
    weights[over] = 1.0 - weights[over]
    bary = np.column_stack((weights[:, 0], weights[:, 1], 1.0 - weights.sum(axis=1)))
    points = (vertices[faces[face_idx]] * bary[:, :, None]).sum(axis=1)
    return points


def sample_urdf_parts(
    urdf_path: Path,
    num_points: int,
    seed: int,
) -> dict[str, Any]:
    links, visuals, joints = parse_urdf(urdf_path)
    from assets import kinematics

    close_tf, welded = kinematics(urdf_path, "close")
    open_tf, _ = kinematics(urdf_path, "open")
    root = next(link for link in links if link not in {j.child for j in joints})
    names = {link: ("root" if group == root else group) for link, group in welded.items()}
    part_names = ["root"] + sorted(set(names.values()) - {"root"})
    link_to_part = {link: part_names.index(name) for link, name in names.items()}
    rng = np.random.default_rng(seed)
    areas = []
    for visual in visuals:
        tri = visual.vertices[visual.faces]
        areas.append(trimesh.triangles.area(tri).sum())
    counts = rng.multinomial(num_points, np.asarray(areas) / np.sum(areas))

    close_points, open_points, part_ids = [], [], []
    for visual, count in zip(visuals, counts):
        if count == 0:
            continue
        local = _triangle_sample(visual.vertices, visual.faces, count, rng)
        local_h = np.column_stack((local, np.ones(count)))
        close_full_tf = close_tf[visual.link] @ visual.visual_tf
        open_full_tf = open_tf[visual.link] @ visual.visual_tf
        close_points.append((close_full_tf @ local_h.T).T[:, :3])
        open_points.append((open_full_tf @ local_h.T).T[:, :3])
        part_ids.append(np.full(count, link_to_part[visual.link], dtype=np.int32))

    return {
        "close": np.concatenate(close_points, axis=0),
        "open": np.concatenate(open_points, axis=0),
        "part_ids": np.concatenate(part_ids, axis=0),
        "part_names": part_names,
    }


def sample_gt_scene_parts(
    close_path: Path, open_path: Path, num_points: int, seed: int
) -> dict[str, Any]:
    close_scene = trimesh.load(close_path, process=False, force="scene")
    open_scene = trimesh.load(open_path, process=False, force="scene")
    nodes = sorted(close_scene.graph.nodes_geometry)
    prefixes = sorted({node.split("/", 1)[0] for node in nodes})
    root_prefix = next((prefix for prefix in prefixes if "body" in prefix.lower()), prefixes[0])
    part_names = [root_prefix] + [prefix for prefix in prefixes if prefix != root_prefix]
    prefix_to_part = {name: idx for idx, name in enumerate(part_names)}

    rng = np.random.default_rng(seed)
    node_data, areas = [], []
    for node in nodes:
        close_tf, close_geom = close_scene.graph.get(frame_to=node)
        open_tf, open_geom = open_scene.graph.get(frame_to=node)
        close_mesh = close_scene.geometry[close_geom]
        open_mesh = open_scene.geometry[open_geom]
        if (
            close_mesh.vertices.shape != open_mesh.vertices.shape
            or close_mesh.faces.shape != open_mesh.faces.shape
        ):
            raise ValueError(f"Topology mismatch between close/open GT node {node}")
        tri = close_mesh.vertices[close_mesh.faces]
        node_data.append((node, close_mesh, open_mesh, close_tf, open_tf))
        areas.append(trimesh.triangles.area(tri).sum())
    counts = rng.multinomial(num_points, np.asarray(areas) / np.sum(areas))

    close_points, open_points, part_ids = [], [], []
    for (node, close_mesh, open_mesh, close_tf, open_tf), count in zip(node_data, counts):
        if count == 0:
            continue

        tri = close_mesh.vertices[close_mesh.faces]
        face_area = trimesh.triangles.area(tri)
        face_probs = face_area / face_area.sum()
        face_idx = rng.choice(len(close_mesh.faces), size=count, replace=True, p=face_probs)
        weights = rng.random((count, 2))
        over = weights.sum(axis=1) > 1.0
        weights[over] = 1.0 - weights[over]
        bary = np.column_stack((weights[:, 0], weights[:, 1], 1.0 - weights.sum(axis=1)))
        close_local = (close_mesh.vertices[close_mesh.faces[face_idx]] * bary[:, :, None]).sum(
            axis=1
        )
        open_local = (open_mesh.vertices[open_mesh.faces[face_idx]] * bary[:, :, None]).sum(axis=1)
        close_points.append(trimesh.transform_points(close_local, close_tf))
        open_points.append(trimesh.transform_points(open_local, open_tf))
        prefix = node.split("/", 1)[0]
        part_ids.append(np.full(count, prefix_to_part[prefix], dtype=np.int32))

    return {
        "close": np.concatenate(close_points, axis=0),
        "open": np.concatenate(open_points, axis=0),
        "part_ids": np.concatenate(part_ids, axis=0),
        "part_names": part_names,
    }


def _upright_yaw_rotations() -> list[tuple[int, np.ndarray]]:
    rotations = []
    for yaw_degrees in (0, 90, 180, 270):
        angle = np.deg2rad(yaw_degrees)
        cosine, sine = np.cos(angle), np.sin(angle)
        yaw = np.asarray(
            (
                (cosine, 0.0, sine),
                (0.0, 1.0, 0.0),
                (-sine, 0.0, cosine),
            ),
            dtype=np.float64,
        )
        rotations.append((yaw_degrees, yaw @ URDF_TO_GLTF))
    return rotations


def _fixed_rotation_similarity_icp(
    src_states: list[np.ndarray],
    dst_states: list[np.ndarray],
    rotation: np.ndarray,
    iterations: int = 25,
    trim_fraction: float = 0.9,
    max_points_per_state: int = 10000,
) -> dict[str, Any]:
    src_states = [_subsample_points(points, max_points_per_state) for points in src_states]
    dst_states = [_subsample_points(points, max_points_per_state) for points in dst_states]
    rotated_states = [points @ rotation.T for points in src_states]
    rotated_all = np.concatenate(rotated_states, axis=0)
    dst_all = np.concatenate(dst_states, axis=0)
    src_center = rotated_all.mean(axis=0)
    dst_center = dst_all.mean(axis=0)
    src_radius = np.sqrt(np.mean(np.sum((rotated_all - src_center) ** 2, axis=1)))
    dst_radius = np.sqrt(np.mean(np.sum((dst_all - dst_center) ** 2, axis=1)))
    scale = float(dst_radius / max(src_radius, 1e-12))
    translation = dst_center - scale * src_center
    previous_score = np.inf
    dst_trees = [cKDTree(points) for points in dst_states]

    for _ in range(iterations):
        corr_src, corr_dst, corr_dist = [], [], []
        for rotated, dst, dst_tree in zip(rotated_states, dst_states, dst_trees):
            transformed = rotated * scale + translation
            src_to_dst_dist, src_to_dst_idx = dst_tree.query(transformed, workers=KD_TREE_WORKERS)
            transformed_tree = cKDTree(transformed)
            dst_to_src_dist, dst_to_src_idx = transformed_tree.query(dst, workers=KD_TREE_WORKERS)
            corr_src.extend((rotated, rotated[dst_to_src_idx]))
            corr_dst.extend((dst[src_to_dst_idx], dst))
            corr_dist.extend((src_to_dst_dist, dst_to_src_dist))

        src_pairs = np.concatenate(corr_src, axis=0)
        dst_pairs = np.concatenate(corr_dst, axis=0)
        distances = np.concatenate(corr_dist, axis=0)
        if trim_fraction < 1.0:
            keep = distances <= np.quantile(distances, trim_fraction)
            src_pairs = src_pairs[keep]
            dst_pairs = dst_pairs[keep]

        src_pair_center = src_pairs.mean(axis=0)
        dst_pair_center = dst_pairs.mean(axis=0)
        src_centered = src_pairs - src_pair_center
        dst_centered = dst_pairs - dst_pair_center
        updated_scale = float(
            np.sum(src_centered * dst_centered) / max(np.sum(src_centered * src_centered), 1e-12)
        )
        updated_scale = max(updated_scale, 1e-8)
        updated_translation = dst_pair_center - updated_scale * src_pair_center
        score = float(np.mean(distances))
        scale, translation = updated_scale, updated_translation
        if np.isfinite(previous_score) and abs(previous_score - score) <= 1e-7 * max(
            1.0, previous_score
        ):
            break
        previous_score = score

    return {
        "rotation": rotation,
        "scale": scale,
        "translation": translation,
    }


def _part_l2_chamfer(points_a: np.ndarray, points_b: np.ndarray, max_points: int = 500) -> float:
    points_a = _subsample_points(points_a, max_points)
    points_b = _subsample_points(points_b, max_points)
    tree_b = cKDTree(points_b)
    tree_a = cKDTree(points_a)
    dist_ab = tree_b.query(points_a, workers=KD_TREE_WORKERS)[0]
    dist_ba = tree_a.query(points_b, workers=KD_TREE_WORKERS)[0]
    return float((dist_ab.mean() + dist_ba.mean()) / 2.0)


def _part_balanced_alignment_score(
    pred: dict[str, Any], gt: dict[str, Any], fit: dict[str, Any], reversed_states: bool
) -> dict[str, Any]:
    src_close = pred["open"] if reversed_states else pred["close"]
    src_open = pred["close"] if reversed_states else pred["open"]
    pred_close = apply_similarity(src_close, fit)
    pred_open = apply_similarity(src_open, fit)
    pred_parts = np.unique(pred["part_ids"])
    gt_parts = np.unique(gt["part_ids"])
    close_cost = np.zeros((len(pred_parts), len(gt_parts)), dtype=np.float64)
    for row, pred_part in enumerate(pred_parts):
        pred_mask = pred["part_ids"] == pred_part
        for column, gt_part in enumerate(gt_parts):
            gt_mask = gt["part_ids"] == gt_part
            close_cost[row, column] = _part_l2_chamfer(pred_close[pred_mask], gt["close"][gt_mask])

    rows, columns = linear_sum_assignment(close_cost)
    close_values, open_values = [], []
    for row, column in zip(rows, columns):
        pred_part = pred_parts[row]
        gt_part = gt_parts[column]
        pred_mask = pred["part_ids"] == pred_part
        gt_mask = gt["part_ids"] == gt_part
        close_value = float(close_cost[row, column])
        open_value = _part_l2_chamfer(pred_open[pred_mask], gt["open"][gt_mask])
        close_values.append(close_value)
        open_values.append(open_value)

    missing_count = max(len(pred_parts), len(gt_parts)) - len(rows)
    unmatched_penalty = 0.5 * float(
        np.linalg.norm(gt["close"].max(axis=0) - gt["close"].min(axis=0))
    )
    close_values.extend([unmatched_penalty] * missing_count)
    open_values.extend([unmatched_penalty] * missing_count)
    close_score = float(np.mean(close_values))
    open_score = float(np.mean(open_values))
    return {
        "part_balanced_score": (close_score + open_score) / 2.0,
    }


def fit_upright_part_balanced_similarity(
    pred: dict[str, Any], gt: dict[str, Any], reversed_states: bool = False
) -> dict[str, Any]:
    src_close = pred["open"] if reversed_states else pred["close"]
    src_open = pred["close"] if reversed_states else pred["open"]
    candidates = []
    for _, rotation in _upright_yaw_rotations():
        fit = _fixed_rotation_similarity_icp(
            [src_close, src_open],
            [gt["close"], gt["open"]],
            rotation,
        )
        fit.update(_part_balanced_alignment_score(pred, gt, fit, reversed_states))
        candidates.append(fit)

    return min(candidates, key=lambda item: float(item["part_balanced_score"]))
