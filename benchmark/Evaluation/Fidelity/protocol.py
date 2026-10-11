from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from alignment import apply_similarity, fit_multistart_trajectory_similarity
from geometry import Samples
from protocol_config import PROTOCOL_VERSION
from protocol_sampling import (
    fit_upright_part_balanced_similarity,
    sample_gt_scene_parts,
    sample_urdf_parts,
)

from assets import kinematics, prepare_prediction, transform_glb


def sample_lightwheel_gt(close: Path, opened: Path, count: int, seed: int) -> dict:
    from protocol_lw_geometry import Piece, sample_pieces

    scenes = [trimesh.load(path, process=False, force="scene") for path in (close, opened)]
    nodes = sorted(scenes[0].graph.nodes_geometry)
    if nodes != sorted(scenes[1].graph.nodes_geometry):
        raise ValueError("GT close/open mesh nodes differ")
    pieces = []
    for node in nodes:
        transforms, meshes = [], []
        for scene in scenes:
            transform, name = scene.graph.get(frame_to=node)
            transforms.append(transform)
            meshes.append(scene.geometry[name])
        if meshes[0].vertices.shape != meshes[1].vertices.shape or not np.array_equal(
            meshes[0].faces, meshes[1].faces
        ):
            raise ValueError(f"GT topology differs: {node}")
        pieces.append(
            Piece(
                node.split("/", 1)[0],
                np.asarray(meshes[0].faces),
                trimesh.transform_points(meshes[0].vertices, transforms[0]),
                trimesh.transform_points(meshes[1].vertices, transforms[1]),
            )
        )
    return sample_pieces(pieces, count, seed)


def sample_lightwheel_prediction(urdf: Path, count: int, seed: int) -> dict:
    from protocol_lw_predict import load_prediction, sample_local

    robot, partition = load_prediction(urdf)
    local = sample_local(robot, partition, urdf.parent, count, seed)
    states = {}
    homogeneous = np.column_stack((local.points, np.ones(len(local.points))))
    for state in ("close", "open"):
        transforms, _ = kinematics(urdf, state)
        points = np.empty_like(local.points)
        for index, link in enumerate(local.link_names):
            mask = local.link_index == index
            points[mask] = (transforms[link] @ homogeneous[mask].T).T[:, :3]
        states[state] = points
    return {**states, "part_ids": local.part_ids, "part_names": list(local.part_names)}


def as_samples(data: dict) -> Samples:

    ids, inverse = np.unique(data["part_ids"], return_inverse=True)
    return Samples(
        data["close"], data["open"], inverse.astype(np.int32), [data["part_names"][i] for i in ids]
    )


def choose_alignment(pred: dict, gt: dict, dataset: str) -> tuple[dict[str, Any], bool]:
    def candidate(reverse: bool) -> dict[str, Any]:
        a, b = (pred["open"], pred["close"]) if reverse else (pred["close"], pred["open"])
        if dataset == "LightWheel":
            return fit_upright_part_balanced_similarity(pred, gt, reverse)
        return fit_multistart_trajectory_similarity(
            a, b, gt["close"], gt["open"], initialization="axis24_pca", workers=1
        )

    key = "part_balanced_score" if dataset == "LightWheel" else "trajectory_score"
    direct = candidate(False)
    reversed_fit = candidate(True)
    required = max(1e-5, 0.01 * float(direct[key]))
    improvement = float(direct[key] - reversed_fit[key])
    reverse = improvement > required
    return (reversed_fit if reverse else direct), reverse


def create_protocol_poses(
    row: dict[str, Any], args: argparse.Namespace, work: Path, paths: dict[str, Path]
) -> dict[str, Path]:
    from results import atomic_json

    raw = work / "raw"
    raw.mkdir()
    dataset = args.dataset or ("LightWheel" if row["gt"].parent.name == "LightWheel" else "ArtVIP")
    gt = (sample_lightwheel_gt if dataset == "LightWheel" else sample_gt_scene_parts)(
        paths["gt_close"], paths["gt_open"], args.points, args.seed
    )
    poses, _ = prepare_prediction(row["prediction"], raw)
    pred = (sample_lightwheel_prediction if dataset == "LightWheel" else sample_urdf_parts)(
        paths["urdf"], args.points, args.seed
    )
    fit, reverse = choose_alignment(pred, gt, dataset)
    if reverse:
        pred["close"], pred["open"] = pred["open"], pred["close"]
        poses["close"], poses["open"] = poses["open"], poses["close"]
    center = (gt["close"].min(axis=0) + gt["close"].max(axis=0)) / 2
    size = float(np.ptp(gt["close"], axis=0).max())
    if size <= 1e-12:
        raise ValueError("Degenerate GT bounds")
    norm, alignment = np.eye(4), np.eye(4)
    scale = 1.0 / size
    norm[:3, :3] = np.eye(3) * scale
    norm[:3, 3] = -center * scale
    alignment[:3, :3] = fit["scale"] * fit["rotation"]
    alignment[:3, 3] = fit["translation"]
    target = work / "eval_poses"
    target.mkdir()
    for state in ("close", "open"):
        pred[state] = (apply_similarity(pred[state], fit) - center) / size
        gt[state] = (gt[state] - center) / size
        for role, source, transform in (
            ("pred", poses[state], norm @ alignment),
            ("gt", paths[f"gt_{state}"], norm),
        ):
            transform_glb(
                source, target / (f"gt_{state}.glb" if role == "gt" else f"{state}.glb"), transform
            )
    atomic_json(
        target / "poses.json",
        {
            "protocol": PROTOCOL_VERSION,
            "normalization": {"center": center, "longest_side": size},
            "points": args.points,
            "seed": args.seed,
        },
    )
    np.savez_compressed(
        work / "points_3d.npz",
        **{
            f"{role}_{field}": np.asarray(getattr(as_samples(data), field))
            for role, data in (("pred", pred), ("gt", gt))
            for field in ("close", "open", "ids", "names")
        },
    )
    return {"eval_poses": target, "evaluation/points_3d.npz": work / "points_3d.npz"}
