from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def read_camera(gt: Path, normalization: dict[str, Any] | None) -> dict[str, Any] | None:
    path = gt / "camera.json"
    if not path.is_file():
        return None
    camera = json.loads(path.read_text())
    if normalization is None:
        raise ValueError("Saved camera requires pose normalization; use --refresh to prepare poses")
    scale = (
        1.0 / float(normalization["longest_side"])
        if "longest_side" in normalization
        else float(normalization["scale"])
    )
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Camera normalization scale must be finite and positive")
    return {
        "center": camera["center"],
        "position": camera["position"],
        "normalization": {"center": normalization["center"], "scale": scale},
        "mask_mode": "geometry",
    }


def pose_normalization(root: Path) -> dict[str, Any] | None:
    poses = root / "eval_poses/poses.json"
    if poses.is_file():
        return json.loads(poses.read_text()).get("normalization")
    metrics = root / "evaluation/metrics_3d.json"
    if metrics.is_file():
        data = json.loads(metrics.read_text())
        return data.get("evaluation_config", {}).get("normalization")
    return None
