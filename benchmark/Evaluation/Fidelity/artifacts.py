from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from geometry import Samples

IMAGE_METRICS = {
    "mask_iou": "mask_iou",
    "ssim_full": "ssim",
    "full_psnr": "psnr_full",
    "lpips": "lpips",
    "clip_cosine": "clip",
}


def read_metrics(path: Path, kind: str) -> dict[str, Any]:
    if kind in {"material_direct", "material_transfer"}:
        from material_metrics import read_material

        return read_material(path, mode=kind.removeprefix("material_"))
    data = json.loads(path.read_text())
    if {"close", "open"}.issubset(data):
        return data
    if kind == "2d" and "pair_metrics" in data:
        result = {}
        for state in ("close", "open"):
            pairs = [
                {target: values[source] for source, target in IMAGE_METRICS.items()}
                for name, values in sorted(data["pair_metrics"].items())
                if name.split("/", 1)[0] == state
            ]
            if not pairs:
                raise ValueError(f"Missing {state} image metrics: {path}")
            result[state] = {
                "pairs": pairs,
                "mean": {key: float(np.mean([row[key] for row in pairs])) for key in pairs[0]},
            }
        return result
    if kind == "3d" and "metrics" in data:
        values = data["metrics"]
        return {
            state: {
                target: values[f"{prefix}_{source}"]
                for source, target in (
                    ("oCD", "object_cd"),
                    ("pCD", "part_cd"),
                    ("pIoU", "part_miou"),
                )
            }
            for state, prefix in (("close", "rest"), ("open", "open"))
        }
    raise ValueError(f"Unsupported metric structure: {path}")


def load_points(path: Path) -> tuple[Samples, Samples]:
    samples = []
    with np.load(path, allow_pickle=False) as data:
        for role in ("pred", "gt"):
            if f"{role}_ids" in data:
                ids = data[f"{role}_ids"]
                names = data[f"{role}_names"].tolist()
            else:
                unique, ids = np.unique(data[f"{role}_part_ids"], return_inverse=True)
                names = [str(int(value)) for value in unique]
            samples.append(Samples(data[f"{role}_close"], data[f"{role}_open"], ids, names))
    return samples[0], samples[1]


def saved_matching(directory: Path, pred: Samples, gt: Samples) -> dict[int, int] | None:
    record = directory / "matching.json"
    if record.is_file():
        pairs = json.loads(record.read_text())
    else:
        record = directory / "metrics_3d.json"
        if not record.is_file():
            return None
        data = json.loads(record.read_text())
        if "matching" in data:
            pairs = data["matching"]
        elif "actual_ID_pairs" in data.get("evaluation_config", {}):
            pairs = {
                str(row["gt_id"]): str(row["pred_id"])
                for row in data["evaluation_config"]["actual_ID_pairs"]
                if row["pred_id"] is not None and row["pred_id"] >= 0
            }
        else:
            return None
    gt_indices = {str(name): index for index, name in enumerate(gt.names)}
    pred_indices = {str(name): index for index, name in enumerate(pred.names)}
    try:
        return {gt_indices[str(left)]: pred_indices[str(right)] for left, right in pairs.items()}
    except KeyError as exc:
        raise ValueError(f"Saved matching refers to an unknown point-cloud part: {exc}") from exc
