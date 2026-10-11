from __future__ import annotations

import json
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def _geom_inventory(mjcf_path: Path) -> list[dict[str, Any]]:
    root = ET.parse(mjcf_path).getroot()
    rows: list[dict[str, Any]] = []
    worldbody = root.find("worldbody")
    if worldbody is None:
        return rows

    def visit(body: ET.Element, parents: list[str]) -> None:
        body_name = body.get("name") or "<unnamed>"
        lineage = [*parents, body_name]
        for geom in body.findall("geom"):
            rows.append(
                {
                    "geom": geom.get("name"),
                    "body": body_name,
                    "body_path": lineage,
                    "class": geom.get("class"),
                    "group": geom.get("group"),
                    "type": geom.get("type", "sphere"),
                    "mesh": geom.get("mesh"),
                }
            )
        for child in body.findall("body"):
            visit(child, lineage)

    for body in worldbody.findall("body"):
        visit(body, [])
    return rows


def _apply_pose(model: Any, data: Any, pose: dict[str, Any]) -> None:
    import mujoco

    for joint_name, value in pose.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_name))
        if joint_id < 0:
            continue
        qpos_adr = int(model.jnt_qposadr[joint_id])
        data.qpos[qpos_adr] = float(value)
    mujoco.mj_forward(model, data)


def _render_images(bundle_root: Path, finding: dict[str, Any]) -> dict[str, str]:
    import mujoco
    import numpy as np
    from PIL import Image

    model = mujoco.MjModel.from_xml_path(str(bundle_root / "object.xml"))
    data = mujoco.MjData(model)
    _apply_pose(model, data, dict(finding.get("pose") or {}))

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.asarray(model.stat.center, dtype=float)
    camera.distance = max(0.25, float(model.stat.extent) * 2.4)
    camera.azimuth = 135.0
    camera.elevation = -25.0

    geoms = [str(value) for value in finding.get("geoms") or []]
    positions = []
    for geom_name in geoms:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id >= 0:
            positions.append(np.asarray(data.geom_xpos[geom_id], dtype=float))

    outputs: dict[str, str] = {}
    original_rgba = model.geom_rgba.copy()
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        collision = mujoco.MjvOption()
        collision.geomgroup[:] = 0
        collision.geomgroup[3] = 1
        model.geom_rgba[model.geom_group == 3] = (0.88, 0.12, 0.08, 1.0)
        renderer.update_scene(data, camera=camera, scene_option=collision)
        collision_path = bundle_root / "collision_only.png"
        Image.fromarray(renderer.render()).save(collision_path)
        outputs["collision_only"] = collision_path.name

        overlay = mujoco.MjvOption()
        overlay.geomgroup[:] = 0
        overlay.geomgroup[2] = 1
        overlay.geomgroup[3] = 1
        model.geom_rgba[:] = original_rgba
        model.geom_rgba[model.geom_group == 3] = (0.95, 0.10, 0.05, 0.45)
        renderer.update_scene(data, camera=camera, scene_option=overlay)
        overlay_path = bundle_root / "collision_overlay.png"
        Image.fromarray(renderer.render()).save(overlay_path)
        outputs["collision_overlay"] = overlay_path.name

        if positions:
            camera.lookat[:] = np.mean(positions, axis=0)
            camera.distance = max(0.08, float(model.stat.extent) * 0.9)
        renderer.update_scene(data, camera=camera, scene_option=overlay)
        closeup_path = bundle_root / "failing_pose_closeup.png"
        Image.fromarray(renderer.render()).save(closeup_path)
        outputs["failing_pose_closeup"] = closeup_path.name
    model.geom_rgba[:] = original_rgba
    return outputs


def build_collision_debug_bundle(
    staged_root: Path,
    report: dict[str, Any],
    output_root: Path,
    *,
    render_images: bool = True,
) -> dict[str, Any] | None:
    findings = [
        dict(finding)
        for finding in report.get("blocking_findings", [])
        if isinstance(finding, dict) and finding.get("kind") == "candidate_mjcf_penetration"
    ]
    if not findings or not (Path(staged_root) / "object.xml").is_file():
        return None

    candidate_id = str(report.get("candidate_id") or "unknown")
    destination = Path(output_root) / candidate_id
    if not destination.exists():
        Path(output_root).mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{candidate_id[:12]}-", dir=Path(output_root)))
        try:
            shutil.copytree(staged_root, temporary / "candidate", dirs_exist_ok=True)
            os.replace(temporary / "candidate", destination)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "status": "failed_run_cache_only",
        "canonical_publish_allowed": False,
        "findings": findings,
        "warnings": list(report.get("warnings") or []),
        "geom_inventory": _geom_inventory(destination / "object.xml"),
        "images": {},
    }
    if render_images:
        try:
            manifest["images"] = _render_images(destination, findings[0])
        except Exception as exc:  # noqa: BLE001 - the preserved evidence remains useful
            manifest["render_error"] = f"{type(exc).__name__}: {exc}"
    manifest_path = destination / "collision_debug.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "status": manifest["status"],
        "candidate_id": candidate_id,
        "root": str(destination),
        "manifest_path": str(manifest_path),
        "images": dict(manifest["images"]),
        "render_error": manifest.get("render_error"),
    }


__all__ = ["build_collision_debug_bundle"]
