from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

from artifacts import load_points
from material_metrics import score_buffers
from protocol_config import blender_bin

from assets import kinematics, prediction_exports


def buffers_exist(root: Path, views: Sequence[str], mode: str) -> bool:
    roles = ("pred", "ours") if mode == "direct" else ("pred", "ours_on_gt")
    return all(
        (root / state / view / "gt_buffers.npz").is_file()
        and any((root / state / view / f"{role}_buffers.npz").is_file() for role in roles)
        for state in ("close", "open")
        for view in views
    )


def saved_transfer_parts(root: Path) -> tuple[list[str], list[str]] | None:
    sources = (
        (root / "evaluation/material_transfer/render_results.json", "correspondence"),
        (root / "evaluation/metrics_3d.json", "evaluation_config"),
    )
    for path, key in sources:
        if path.is_file():
            config = json.loads(path.read_text()).get(key) or {}
            if config.get("pred_part_names") and config.get("gt_part_names"):
                return config["pred_part_names"], config["gt_part_names"]
    return None


def transfer_parts(row: dict[str, Any], reuse: bool) -> tuple[list[str], list[str]]:
    root = row["root"]
    saved = saved_transfer_parts(root) if reuse else None
    if saved is not None:
        return saved
    pred, gt = load_points(root / "evaluation/points_3d.npz")
    urdf, _ = prediction_exports(row["prediction"])
    import xml.etree.ElementTree as ET

    robot = ET.parse(urdf).getroot()
    children = {joint.find("child").attrib["link"] for joint in robot.findall("joint")}
    roots = [
        link.attrib["name"] for link in robot.findall("link") if link.attrib["name"] not in children
    ]
    fixed = next((name for name in ("root", *roots) if name in pred.names), None)
    if len(roots) != 1 or fixed is None:
        raise ValueError("Missing root part in material-transfer geometry")
    names = [fixed, *[name for name in pred.names if name != fixed]]
    body = next((name for name in gt.names if "body" in name.lower()), gt.names[0])
    return names, [body, *[name for name in gt.names if name != body]]


def transfer_link_parts(urdf: Path, part_names: list[str]) -> dict[str, int]:
    _, groups = kinematics(urdf, "close")
    return {
        link: part_names.index(group) if group in part_names else 0
        for link, group in groups.items()
    }


def evaluate_material(row: dict[str, Any], args: argparse.Namespace, mode: str) -> dict[str, Any]:
    from results import atomic_json, publish_files, render_inputs

    root = row["root"]
    directory = root / "evaluation" / f"material_{mode}"
    if not args.refresh and buffers_exist(directory, args.views, mode):
        return score_buffers(directory, args.views, mode=mode)
    with tempfile.TemporaryDirectory(
        prefix="material-", dir=root / ".cache", ignore_cleanup_errors=True
    ) as temporary:
        work = Path(temporary)
        config = {
            **render_inputs(row, args),
            "output": str(work / "buffers"),
            "transfer": mode == "transfer",
        }
        if mode == "transfer":
            config["pred_part_names"], config["gt_part_names"] = transfer_parts(
                row, reuse=not args.refresh
            )
            urdf, _ = prediction_exports(row["prediction"])
            config["link_to_part"] = transfer_link_parts(urdf, config["pred_part_names"])
            saved = directory / "render_results.json"
            if saved.is_file() and not args.refresh:
                config["correspondence"] = json.loads(saved.read_text()).get("correspondence")
        atomic_json(work / "config.json", config)
        with (work / "render.log").open("x") as log:
            process = subprocess.run(
                [
                    blender_bin(),
                    "--background",
                    "--factory-startup",
                    "--python-exit-code",
                    "1",
                    "--python",
                    str(Path(__file__).with_name("render_material.py")),
                    "--",
                    "--manifest",
                    str(work / "config.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if process.returncode:
            publish_files(root, {f"evaluation/material_{mode}_error.log": work / "render.log"})
            raise RuntimeError(
                f"Blender failed ({process.returncode}); see evaluation/material_{mode}_error.log"
            )
        result = score_buffers(work / "buffers", args.views, mode=mode)
        publish_files(
            root,
            {
                f"evaluation/material_{mode}": work / "buffers",
                f"evaluation/material_{mode}/render.log": work / "render.log",
            },
        )
    return result
