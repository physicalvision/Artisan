from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from xml.etree import ElementTree as ET

from pipeline.gates import (
    WHOLE_BODY_JOINT_TYPE,
    WHOLE_BODY_TARGET,
    _moving_ancestor_path,
    _walk_to_actuated_joint,
    manipulation_binding_check,
)

if TYPE_CHECKING:
    from sdk._core.v0.motion_engine.resolve import ResolvedCollisionManifest

_SCHEMA_VERSION = "2.0"


def _read_tasks_json(workspace: Path) -> tuple[bool, Any, str | None]:
    path = Path(workspace) / "tasks.json"
    if not path.is_file():
        return False, None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return True, None, f"tasks.json could not be read: {type(exc).__name__}: {exc}"
    return True, data, None


def _project_tasks(tasks: list[Any], joints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    for i, t in enumerate(tasks):
        if not isinstance(t, dict):
            continue
        raw_contact_target = t.get("contact_target")
        if isinstance(raw_contact_target, dict):
            link = raw_contact_target.get("link")
            raw_visuals = raw_contact_target.get("visuals")
            contact_target = {
                "link": link,
                "visuals": list(raw_visuals) if isinstance(raw_visuals, list) else raw_visuals,
            }
        else:
            link = None
            contact_target = raw_contact_target
        target = t.get("target_joint")
        if not isinstance(target, str):
            target = _walk_to_actuated_joint(link, joints)
        operations.append(
            {
                "task_id": t.get("task_id") or f"task_{i}",
                "instruction": t.get("instruction"),
                "contact_target": contact_target,
                "target_joint": target,
                "target_kind": (WHOLE_BODY_JOINT_TYPE if target == WHOLE_BODY_TARGET else "joint"),
                "success": t.get("success"),
                "task_intent": t.get("task_intent"),
            }
        )
        if "articulated_path" in t:
            operations[-1]["articulated_path"] = list(t["articulated_path"])
    return operations


def _write(out_dir: Path, manifest: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manipulation.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def validate_manifest_against_urdf(
    manifest_path: str | Path,
    urdf_path: str | Path,
) -> list[str]:
    manifest_path = Path(manifest_path)
    urdf_path = Path(urdf_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"manipulation manifest could not be read: {type(exc).__name__}: {exc}"]
    if not isinstance(manifest, dict) or manifest.get("schema_version") != _SCHEMA_VERSION:
        return [f"manipulation manifest schema_version must equal {_SCHEMA_VERSION!r}"]
    operations = manifest.get("operations")
    if not isinstance(operations, list):
        return ["manipulation manifest operations must be a list"]
    try:
        root = ET.parse(urdf_path).getroot()
    except (OSError, ET.ParseError) as exc:
        return [f"candidate URDF could not be read: {type(exc).__name__}: {exc}"]

    links_by_name: dict[str, list[ET.Element]] = {}
    for link_element in root.findall("link"):
        link_name = link_element.get("name")
        if isinstance(link_name, str):
            links_by_name.setdefault(link_name, []).append(link_element)

    failures: list[str] = []
    joint_graph = [
        {
            "name": element.get("name"),
            "type": element.get("type"),
            "parent": element.find("parent").get("link"),
            "child": element.find("child").get("link"),
        }
        for element in root.findall("joint")
        if element.find("parent") is not None and element.find("child") is not None
    ]
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict):
            failures.append(f"operations[{index}] is not an object")
            continue
        task_id = operation.get("task_id") or f"operations[{index}]"
        contact_target = operation.get("contact_target")
        if not isinstance(contact_target, dict):
            failures.append(f"{task_id}: contact_target is not an object")
            continue
        if set(contact_target) != {"link", "visuals"}:
            failures.append(f"{task_id}: contact_target must contain exactly link and visuals")
            continue
        link_name = contact_target.get("link")
        if "articulated_path" in operation:
            path = operation["articulated_path"]
            target = operation.get("target_joint")
            resolved = (
                _moving_ancestor_path(link_name, target, joint_graph)
                if isinstance(link_name, str) and isinstance(target, str)
                else None
            )
            if not isinstance(path, list) or len(path) < 2 or path != resolved:
                failures.append(f"{task_id}: articulated_path does not match candidate URDF")
        visual_names = contact_target.get("visuals")
        if (
            not isinstance(link_name, str)
            or not link_name.strip()
            or not isinstance(visual_names, list)
            or not visual_names
            or any(not isinstance(name, str) or not name.strip() for name in visual_names)
            or len(set(visual_names)) != len(visual_names)
        ):
            failures.append(f"{task_id}: contact_target link/visuals are malformed")
            continue
        matching_links = links_by_name.get(link_name, [])
        if len(matching_links) != 1:
            failures.append(
                f"{task_id}: target link {link_name!r} has {len(matching_links)} matches in URDF"
            )
            continue
        exported_visuals = [visual.get("name") for visual in matching_links[0].findall("visual")]
        for visual_name in visual_names:
            matches = sum(candidate == visual_name for candidate in exported_visuals)
            if matches != 1:
                failures.append(
                    f"{task_id}: target visual {visual_name!r} on link {link_name!r} "
                    f"has {matches} matches in URDF"
                )
    return failures


def derive_manifest(
    model_path: str | Path,
    blueprint: dict[str, Any] | None,
    workspace: str | Path,
    out_dir: str | Path,
    *,
    sdk_package: str = "sdk",
    collision_manifest: ResolvedCollisionManifest | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    blueprint = blueprint or {}

    from sdk.runtime.runtime import probe_summary

    probe = probe_summary(
        Path(model_path),
        sdk_package=sdk_package,
        **({"collision_manifest": collision_manifest} if collision_manifest is not None else {}),
    )
    parts: list[str] = probe.get("parts", []) if probe.get("build_ok") else []
    joints: list[dict[str, Any]] = probe.get("joints", []) if probe.get("build_ok") else []
    visuals_by_part: dict[str, list[str | None]] = (
        probe.get("visuals_by_part", {}) if probe.get("build_ok") else {}
    )

    tasks_present, tasks_json, tasks_error = _read_tasks_json(workspace)
    warnings: list[str] = []
    if not probe.get("build_ok"):
        warnings.append(f"probe failed: {probe.get('build_error', 'unknown')}")

    if tasks_present:
        derivation = "coding_tasks_json"
        binding = manipulation_binding_check(
            tasks_json,
            parts,
            joints,
            visuals_by_part=visuals_by_part,
        )
        if tasks_error is not None:
            operations = []
            warnings.append(tasks_error)
        elif binding.passed:
            operations = _project_tasks(tasks_json["tasks"], joints)
        else:
            operations = []
            warnings.extend(binding.failures)
    else:
        operations = []
        derivation = "no_authored_tasks"
        warnings.append("tasks.json absent; no manipulation operations emitted")

    if not operations:
        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "source": "artisan_pipeline",
            "derivation": derivation,
            "operations": [],
        }
        if warnings:
            manifest["warnings"] = warnings
        _write(out_dir, manifest)
        return {
            "manifest": str(out_dir / "manipulation.json"),
            "operations": 0,
            "skipped": "no-tasks",
        }

    object_id = probe.get("name") or (blueprint.get("asset_brief") or {}).get("name")
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "object_id": object_id,
        "source": "artisan_pipeline",
        "derivation": derivation,
        "operations": operations,
    }
    if warnings:
        manifest["warnings"] = warnings
    _write(out_dir, manifest)
    return {"manifest": str(out_dir / "manipulation.json"), "operations": len(operations)}
