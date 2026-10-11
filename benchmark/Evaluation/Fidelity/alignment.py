from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.spatial import cKDTree


def chamfer(points_a: np.ndarray, points_b: np.ndarray) -> dict[str, float]:
    tree_b = cKDTree(points_b)
    tree_a = cKDTree(points_a)
    dist_ab = tree_b.query(points_a, workers=1)[0]
    dist_ba = tree_a.query(points_b, workers=1)[0]
    return {
        "l2_mean": float((dist_ab.mean() + dist_ba.mean()) / 2.0),
        "l2_squared_mean": float(((dist_ab**2).mean() + (dist_ba**2).mean()) / 2.0),
        "pred_to_gt_l2_mean": float(dist_ab.mean()),
        "gt_to_pred_l2_mean": float(dist_ba.mean()),
    }


def signed_permutation_matrices() -> list[np.ndarray]:
    mats = []
    for perm in itertools.permutations(range(3)):
        base = np.zeros((3, 3), dtype=np.float64)
        for i, j in enumerate(perm):
            base[i, j] = 1.0
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            mats.append(np.diag(signs) @ base)
    return mats


def proper_axis_rotation_matrices() -> list[np.ndarray]:
    return [matrix for matrix in signed_permutation_matrices() if np.linalg.det(matrix) > 0.0]


def _subsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[indices]


def _estimate_similarity(src: np.ndarray, dst: np.ndarray) -> dict[str, np.ndarray | float]:
    src_center = src.mean(axis=0)
    dst_center = dst.mean(axis=0)
    src_centered = src - src_center
    dst_centered = dst - dst_center
    covariance = dst_centered.T @ src_centered / max(len(src), 1)
    u, singular_values, vt = np.linalg.svd(covariance)
    handedness = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        handedness[-1] = -1.0
    rotation = u @ np.diag(handedness) @ vt
    variance = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
    scale = float(np.dot(singular_values, handedness) / max(variance, 1e-12))
    translation = dst_center - scale * (src_center @ rotation.T)
    return {"rotation": rotation, "scale": scale, "translation": translation}


def _initial_similarity(
    src_states: list[np.ndarray], dst_states: list[np.ndarray], rotation: np.ndarray
) -> dict:
    src = np.concatenate(src_states, axis=0)
    dst = np.concatenate(dst_states, axis=0)
    src_center = src.mean(axis=0)
    dst_center = dst.mean(axis=0)
    src_radius = float(np.sqrt(np.mean(np.sum((src - src_center) ** 2, axis=1))))
    dst_radius = float(np.sqrt(np.mean(np.sum((dst - dst_center) ** 2, axis=1))))
    scale = dst_radius / max(src_radius, 1e-12)
    return {
        "rotation": rotation,
        "scale": scale,
        "translation": dst_center - scale * (src_center @ rotation.T),
    }


def _pca_initial_rotations(
    src_states: list[np.ndarray], dst_states: list[np.ndarray]
) -> list[np.ndarray]:
    src = np.concatenate(src_states, axis=0)
    dst = np.concatenate(dst_states, axis=0)
    _, src_basis = np.linalg.eigh(np.cov(src, rowvar=False))
    _, dst_basis = np.linalg.eigh(np.cov(dst, rowvar=False))
    src_basis = src_basis[:, ::-1]
    dst_basis = dst_basis[:, ::-1]
    rotations = []
    for signs in itertools.product((-1.0, 1.0), repeat=3):
        rotation = dst_basis @ np.diag(signs) @ src_basis.T
        if np.linalg.det(rotation) > 0.0:
            rotations.append(rotation)
    return rotations


def _state_aware_score(
    src_states: list[np.ndarray], dst_states: list[np.ndarray], fit: dict, workers: int
) -> float:
    scores = []
    for src, dst in zip(src_states, dst_states):
        transformed = apply_similarity(src, fit)
        dist_src = cKDTree(dst).query(transformed, workers=workers)[0]
        dist_dst = cKDTree(transformed).query(dst, workers=workers)[0]
        scores.append(float((dist_src.mean() + dist_dst.mean()) / 2.0))
    return float(np.mean(scores))


def _similarity_icp(
    src_states: list[np.ndarray],
    dst_states: list[np.ndarray],
    initial_fit: dict,
    iterations: int,
    trim_fraction: float,
    workers: int,
) -> dict:
    fit = {
        key: np.asarray(value).copy() if isinstance(value, np.ndarray) else value
        for key, value in initial_fit.items()
    }
    dst_trees = [cKDTree(points) for points in dst_states]
    previous_score = math.inf
    completed_iterations = 0
    for iteration in range(iterations):
        corr_src = []
        corr_dst = []
        corr_dist = []
        for src, dst, dst_tree in zip(src_states, dst_states, dst_trees):
            transformed = apply_similarity(src, fit)
            src_to_dst_dist, src_to_dst_idx = dst_tree.query(transformed, workers=workers)
            transformed_tree = cKDTree(transformed)
            dst_to_src_dist, dst_to_src_idx = transformed_tree.query(dst, workers=workers)
            corr_src.extend((src, src[dst_to_src_idx]))
            corr_dst.extend((dst[src_to_dst_idx], dst))
            corr_dist.extend((src_to_dst_dist, dst_to_src_dist))
        src_pairs = np.concatenate(corr_src, axis=0)
        dst_pairs = np.concatenate(corr_dst, axis=0)
        distances = np.concatenate(corr_dist, axis=0)
        if trim_fraction < 1.0:
            threshold = float(np.quantile(distances, trim_fraction))
            keep = distances <= threshold
            src_pairs = src_pairs[keep]
            dst_pairs = dst_pairs[keep]
        updated = _estimate_similarity(src_pairs, dst_pairs)
        score = _state_aware_score(src_states, dst_states, updated, workers)
        fit = updated
        completed_iterations = iteration + 1
        if math.isfinite(previous_score) and abs(previous_score - score) <= 1e-07 * max(
            1.0, previous_score
        ):
            break
        previous_score = score
    fit["icp_iterations"] = completed_iterations
    fit["score"] = _state_aware_score(src_states, dst_states, fit, workers)
    return fit


def fit_multistart_trajectory_similarity(
    src_close: np.ndarray,
    src_open: np.ndarray,
    dst_close: np.ndarray,
    dst_open: np.ndarray,
    initialization: str = "axis24_pca",
    coarse_points_per_state: int = 2500,
    refine_points_per_state: int = 10000,
    coarse_iterations: int = 8,
    refine_iterations: int = 25,
    top_k: int = 4,
    trim_fraction: float = 0.9,
    workers: int = -1,
) -> dict[str, np.ndarray | float]:
    if initialization not in {"identity", "axis24", "axis24_pca"}:
        raise ValueError(f"Unsupported alignment initialization: {initialization}")
    coarse_src = [
        _subsample_points(src_close, coarse_points_per_state),
        _subsample_points(src_open, coarse_points_per_state),
    ]
    coarse_dst = [
        _subsample_points(dst_close, coarse_points_per_state),
        _subsample_points(dst_open, coarse_points_per_state),
    ]
    rotations = [np.eye(3)] if initialization == "identity" else proper_axis_rotation_matrices()
    if initialization == "axis24_pca":
        for rotation in _pca_initial_rotations(coarse_src, coarse_dst):
            if not any((np.allclose(rotation, existing, atol=1e-08) for existing in rotations)):
                rotations.append(rotation)
    candidates = []
    for index, rotation in enumerate(rotations):
        initial_fit = _initial_similarity(coarse_src, coarse_dst, rotation)
        fit = _similarity_icp(
            coarse_src, coarse_dst, initial_fit, coarse_iterations, trim_fraction, workers
        )
        fit["initialization_index"] = index
        candidates.append(fit)
    candidates.sort(key=lambda item: float(item["score"]))
    refine_src = [
        _subsample_points(src_close, refine_points_per_state),
        _subsample_points(src_open, refine_points_per_state),
    ]
    refine_dst = [
        _subsample_points(dst_close, refine_points_per_state),
        _subsample_points(dst_open, refine_points_per_state),
    ]
    refined = []
    for candidate in candidates[: max(1, min(top_k, len(candidates)))]:
        fit = _similarity_icp(
            refine_src, refine_dst, candidate, refine_iterations, trim_fraction, workers
        )
        fit["initialization_index"] = candidate["initialization_index"]
        refined.append(fit)
    best = min(refined, key=lambda item: float(item["score"]))
    best["close_score"] = chamfer(apply_similarity(src_close, best), dst_close)["l2_mean"]
    best["open_score"] = chamfer(apply_similarity(src_open, best), dst_open)["l2_mean"]
    best["trajectory_score"] = float((best["close_score"] + best["open_score"]) / 2.0)
    best["initialization"] = initialization
    best["num_initializations"] = len(rotations)
    best["coarse_candidate_scores"] = [
        {"initialization_index": int(item["initialization_index"]), "score": float(item["score"])}
        for item in candidates
    ]
    return best


def apply_similarity(points: np.ndarray, fit: dict[str, np.ndarray | float]) -> np.ndarray:
    return points @ fit["rotation"].T * fit["scale"] + fit["translation"]
