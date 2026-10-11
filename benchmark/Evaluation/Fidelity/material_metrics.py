from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion

CHANNELS = ("base_color", "metallic", "roughness")
STATES = ("close", "open")
VIEWS = ("front", "iso", "top")
SETTINGS = {
    "resolution": 512,
    "engine": "cycles",
    "samples": 1,
    "device": "CPU",
    "denoise": False,
    "alpha_threshold": 0.5,
    "erosion": 3,
    "metallic_threshold": 1e-4,
    "psnr_peak": 1.0,
    "mse_floor": 1e-12,
}


def _comparison(mode: str) -> str:
    if mode not in {"direct", "transfer"}:
        raise ValueError(f"Unsupported material mode: {mode}")
    return "original_geometry" if mode == "direct" else "part_transfer_gt_geometry"


def _score(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    count = int(mask.sum())
    if not count:
        return {"valid_pixels": 0, "mse": None, "psnr": None}
    delta = pred[mask].astype(np.float64) - gt[mask].astype(np.float64)
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite material values in scoring region")
    mse = float(np.mean(delta * delta))
    return {
        "valid_pixels": count,
        "mse": mse,
        "psnr": -10.0 * math.log10(max(mse, SETTINGS["mse_floor"])),
    }


def _values(views: dict[str, Any], channel: str) -> list[float]:
    values = []
    for row in views.values():
        item = row[channel]
        value = item.get("psnr")
        if item.get("valid_pixels", 0) > 0 and value is not None:
            if not math.isfinite(float(value)):
                raise ValueError(f"Nonfinite {channel} PSNR")
            values.append(float(value))
    return values


def _result(views: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    aggregate = {}
    for channel in CHANNELS:
        values = _values(views, channel)
        if channel == "metallic" and not values and settings["mode"] == "direct":
            ordinary = _values(views, "metallic_interior")
            if ordinary and math.isclose(mean(ordinary), 120.0, abs_tol=1e-12):
                values = [120.0]
        aggregate[channel] = mean(values) if values else None
    return {"aggregate": aggregate}


def score_buffers(
    directory: Path, views: Sequence[str] = VIEWS, mode: str = "direct"
) -> dict[str, Any]:
    comparison = _comparison(mode)
    scores = {}
    for state in STATES:
        for view in views:
            folder = directory / state / view
            pred_path = folder / "pred_buffers.npz"
            if not pred_path.is_file():
                name = "ours_buffers.npz" if mode == "direct" else "ours_on_gt_buffers.npz"
                pred_path = folder / name
            with (
                np.load(pred_path, allow_pickle=False) as pred,
                np.load(folder / "gt_buffers.npz", allow_pickle=False) as gt,
            ):
                pred_mask, gt_mask = pred["mask"], gt["mask"]
                if pred_mask.ndim != 2 or pred_mask.shape != gt_mask.shape:
                    raise ValueError(f"Material mask shape mismatch: {state}/{view}")
                if not np.isfinite(pred_mask).all() or not np.isfinite(gt_mask).all():
                    raise ValueError(f"Nonfinite material masks: {state}/{view}")
                interior = binary_erosion(
                    (pred_mask >= SETTINGS["alpha_threshold"])
                    & (gt_mask >= SETTINGS["alpha_threshold"]),
                    structure=np.ones((3, 3), dtype=bool),
                    iterations=SETTINGS["erosion"],
                    border_value=0,
                )
                row = {}
                for channel in CHANNELS:
                    p, g = pred[channel], gt[channel]
                    expected = pred_mask.shape + ((3,) if channel == "base_color" else ())
                    if p.shape != expected or g.shape != expected:
                        raise ValueError(f"Material channel shape mismatch: {channel}")
                    if channel == "metallic" and mode == "direct":
                        row["metallic_interior"] = _score(p, g, interior)
                        active = interior & (
                            (np.abs(p) > SETTINGS["metallic_threshold"])
                            | (np.abs(g) > SETTINGS["metallic_threshold"])
                        )
                        row[channel] = _score(p, g, active)
                    else:
                        row[channel] = _score(p, g, interior)
                scores[f"{state}/{view}"] = row
    settings = {
        **SETTINGS,
        "mode": mode,
        "comparison_mode": comparison,
        "states": list(STATES),
        "views": list(views),
    }
    return _result(scores, settings)


def read_material(path: Path, mode: str = "direct") -> dict[str, Any]:
    comparison = _comparison(mode)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "aggregate" in data:
        return {"aggregate": {key: data["aggregate"][key] for key in CHANNELS}}
    settings = data.get("settings", {})
    stored_mode = settings.get("comparison_mode", data.get("comparison_mode"))
    allowed = {comparison, "surface_transfer"} if mode == "transfer" else {comparison}
    if stored_mode is not None and stored_mode not in allowed:
        raise ValueError(f"Expected {mode} material evaluation: {path}")
    if settings.get("mode", mode) != mode:
        raise ValueError(f"Expected {mode} material evaluation: {path}")
    raw = data.get("views", data.get("metrics"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Missing material view metrics: {path}")
    views = {}
    for key, row in raw.items():
        if key.split("/", 1)[0] not in STATES:
            continue
        if "surface_transfer" in row:
            if mode != "transfer":
                raise ValueError(f"Expected direct material evaluation: {path}")
            row = row["surface_transfer"]
        elif mode == "transfer" and (stored_mode is None or "views" not in data):
            raise ValueError(f"Missing material transfer results: {path}")
        channels = row.get("channels")
        if channels is not None:
            scores = {
                channel: channels[channel][
                    "active_interior"
                    if channel == "metallic" and mode == "direct"
                    else "interior_intersection"
                ]
                for channel in CHANNELS
            }
            if mode == "direct":
                scores["metallic_interior"] = channels["metallic"]["interior_intersection"]
        else:
            keys = (*CHANNELS, "metallic_interior") if mode == "direct" else CHANNELS
            scores = {channel: row[channel] for channel in keys}
        views[key] = {
            channel: {field: item.get(field) for field in ("valid_pixels", "mse", "psnr")}
            for channel, item in scores.items()
        }
    if not views:
        raise ValueError(f"Missing material view metrics: {path}")
    return _result(views, {**settings, "mode": mode, "comparison_mode": comparison})
