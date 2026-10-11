from __future__ import annotations

import atexit
import hashlib
import json
import os
import select
import shutil
import signal
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from pipeline.operation_timing import timed_operation
from sdk.runtime._render_device import RenderDevicePolicy
from sdk.runtime._render_protocol import (
    RenderIntegrityError,
    input_identity,
    stage_input,
    validate_response,
    verify_input,
    write_report,
)

if TYPE_CHECKING:
    from sdk._core.v0.motion_engine.resolve import ResolvedCollisionManifest

_DRIVER = Path(__file__).with_name("_render_driver.py")
_SERVER = Path(__file__).with_name("_render_server.py")
_TEXTURES_DIR = Path(__file__).with_name("assets") / "textures"
_READY_MARKER = "__SERVER_READY__"
_DONE_MARKER = "__RENDER_DONE__"
OPEN_FRACTION = 1.0

_Z_UP_TO_Y_UP = [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]

_FOCUS_MARKER = "__artisan_focus__"
_FOCUS_PART_PAD = 1.15
_FOCUS_JOINT_FACTOR = 0.4
_FOCUS_MIN_RADIUS = 1e-3


def _open_pose(model: Any) -> dict[str, float]:
    pose: dict[str, float] = {}
    for j in getattr(model, "articulations", []):
        jt = getattr(j, "articulation_type", None)
        jt_name = getattr(jt, "value", str(jt)).lower()
        if jt_name in ("fixed", "floating"):
            continue
        if getattr(j, "mimic", None) is not None:
            continue
        lim = getattr(j, "motion_limits", None)
        lo = getattr(lim, "lower", None) if lim else None
        hi = getattr(lim, "upper", None) if lim else None
        name = getattr(j, "name", None)
        if not name:
            continue
        if lo is None or hi is None:
            pose[name] = OPEN_FRACTION
        else:
            pose[name] = float(lo) + OPEN_FRACTION * (float(hi) - float(lo))
    return pose


def _redirect_followers_to_drivers(model: Any, pose: dict[str, float]) -> dict[str, float]:
    if not pose:
        return pose
    from sdk._core.v0._coupling import driver_pose_for_follower

    followers = {
        getattr(j, "name", None): j
        for j in getattr(model, "articulations", []) or []
        if getattr(j, "mimic", None) is not None
    }
    if not any(k in followers for k in pose):
        return pose
    out: dict[str, float] = {}
    redirected: dict[str, float] = {}
    for k, v in pose.items():
        j = followers.get(k)
        if j is not None:
            driver = str(getattr(getattr(j, "mimic", None), "joint", "")).strip()
            qd = driver_pose_for_follower(model, k, float(v))
            if driver and qd is not None:
                redirected[driver] = qd
                continue
        out[k] = v
    out.update(redirected)
    return out


def _geometry_to_trimesh(geometry: Any, asset_root: Path):
    import trimesh

    from sdk._core.v0.placement import _load_trimesh_mesh
    from sdk._core.v0.types import Box, Cylinder, Mesh, Sphere

    if isinstance(geometry, Mesh):
        return _load_trimesh_mesh(geometry, asset_root=asset_root)
    if isinstance(geometry, Box):
        return trimesh.creation.box(extents=[float(v) for v in geometry.size])
    if isinstance(geometry, Cylinder):
        return trimesh.creation.cylinder(
            radius=float(geometry.radius), height=float(geometry.length)
        )
    if isinstance(geometry, Sphere):
        return trimesh.creation.icosphere(radius=float(geometry.radius))
    return None


_MATERIAL_LIB_ON = os.environ.get("ARTISAN_MATERIAL_LIB", "1") != "0"


def _resolve_material(material_ref: Any, materials_by_name: dict[str, Any]):
    if material_ref is None:
        return None
    if isinstance(material_ref, str):
        return materials_by_name.get(material_ref)
    return material_ref


def _material_rgba(mat_obj: Any):
    rgba = getattr(mat_obj, "rgba", None)
    if rgba is None:
        return None
    vals = [float(c) for c in rgba]
    if len(vals) == 3:
        vals.append(1.0)
    return vals[:4]


def _visual_material(
    mat_obj: Any,
    material_name: str,
) -> tuple[Any | None, dict[str, Any], dict[str, Any]]:
    import trimesh

    if mat_obj is None:
        return None, {}, {}
    rgba = _material_rgba(mat_obj)
    appearance_fields = (
        "material_class",
        "metallic",
        "roughness",
        "emissive",
        "transmission",
        "ior",
        "texture_set",
    )
    if rgba is None and not any(
        getattr(mat_obj, field, None) is not None for field in appearance_fields
    ):
        return None, {}, {}

    base = rgba if rgba is not None else [0.8, 0.8, 0.8, 1.0]
    renderer_spec: dict[str, Any] = {}
    texture_set: str | None = None
    if _MATERIAL_LIB_ON:
        from sdk._core.v0.types import (
            resolve_material_glass,
            resolve_material_pbr,
            resolve_material_texture,
        )

        metallic, roughness, emissive = resolve_material_pbr(mat_obj)
        glass = resolve_material_glass(mat_obj)
        if glass:
            renderer_spec.update(glass)
        texture_set = resolve_material_texture(mat_obj)
        if texture_set:
            texture_dir = _TEXTURES_DIR / texture_set
            if (texture_dir / "color.png").is_file():
                renderer_spec["texture_dir"] = str(texture_dir)
    else:
        metallic, roughness, emissive = (0.0, 0.8, 0.0)

    pbr_kwargs: dict[str, Any] = {
        "name": material_name,
        "baseColorFactor": [float(value) for value in base],
        "metallicFactor": float(metallic),
        "roughnessFactor": float(roughness),
    }
    if emissive and emissive > 0.0:
        multiplier = min(1.0, float(emissive))
        pbr_kwargs["emissiveFactor"] = [min(1.0, float(value) * multiplier) for value in base[:3]]
        renderer_spec["emission"] = {
            "strength": float(emissive),
            "color": [float(value) for value in base[:3]],
        }

    source_spec: dict[str, Any] = {
        "source_name": str(getattr(mat_obj, "name", material_name)),
        "material_class": getattr(mat_obj, "material_class", None),
        "base_color_factor": [float(value) for value in base],
        "metallic_factor": float(metallic),
        "roughness_factor": float(roughness),
        "emissive_strength": float(emissive),
        "texture_set": texture_set,
    }
    for field in (
        "transmission",
        "ior",
        "attenuation_color",
        "attenuation_distance",
    ):
        if field in renderer_spec:
            source_spec[field] = renderer_spec[field]
    return trimesh.visual.material.PBRMaterial(**pbr_kwargs), renderer_spec, source_spec


def _resolve_focus(model: Any, focus: list[str]) -> dict[str, list[str]]:
    part_names = {getattr(p, "name", None) for p in getattr(model, "parts", [])}
    part_names.discard(None)
    joint_names = {getattr(j, "name", None) for j in getattr(model, "articulations", [])}
    joint_names.discard(None)
    parts: list[str] = []
    joints: list[str] = []
    for token in focus:
        name = str(token)
        if name in part_names:
            parts.append(name)
        elif name in joint_names:
            joints.append(name)
        else:
            raise ValueError(
                f"unknown focus target {name!r}; "
                f"valid parts={sorted(part_names)}, joints={sorted(joint_names)}"
            )
    return {"parts": parts, "joints": joints}


def _focus_center_radius(model: Any, world_tf, part_bounds: dict, focus_spec: dict):
    import numpy as np

    from sdk._core.v0.geometry_qc import _mat4_vec3

    def _scene_radius() -> float:
        if not part_bounds:
            return 1.0
        allmin = np.min([b[0] for b in part_bounds.values()], axis=0)
        allmax = np.max([b[1] for b in part_bounds.values()], axis=0)
        return float(np.max(allmax - allmin)) or 1.0

    boxes: list[tuple] = []
    for pname in focus_spec.get("parts", []):
        b = part_bounds.get(pname)
        if b is None:
            continue
        boxes.append(((b[0] + b[1]) / 2.0, (b[1] - b[0]) / 2.0 * _FOCUS_PART_PAD))

    joints = {getattr(j, "name", None): j for j in getattr(model, "articulations", [])}
    for jname in focus_spec.get("joints", []):
        j = joints.get(jname)
        if j is None:
            continue
        origin = getattr(j, "origin", None)
        xyz = tuple(float(v) for v in getattr(origin, "xyz", (0.0, 0.0, 0.0)))
        ptf = world_tf.get(getattr(j, "parent", None))
        anchor = np.asarray(_mat4_vec3(ptf, xyz) if ptf is not None else xyz, dtype=float)
        cb = part_bounds.get(getattr(j, "child", None))
        if cb is not None:
            r = max(float(np.max(cb[1] - cb[0])) * _FOCUS_JOINT_FACTOR, _FOCUS_MIN_RADIUS)
        else:
            r = max(0.1 * _scene_radius(), _FOCUS_MIN_RADIUS)
        boxes.append((anchor, np.array([r, r, r], dtype=float)))

    if not boxes:
        return None
    mins = np.min([c - h for c, h in boxes], axis=0)
    maxs = np.max([c + h for c, h in boxes], axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins) / 2.0), _FOCUS_MIN_RADIUS)
    return center, radius


_SEGMENT_PALETTE = [
    (0.90, 0.12, 0.12),
    (0.12, 0.45, 0.95),
    (0.16, 0.78, 0.28),
    (0.98, 0.72, 0.06),
    (0.72, 0.22, 0.88),
    (0.06, 0.82, 0.80),
    (0.98, 0.45, 0.10),
    (0.95, 0.28, 0.62),
    (0.55, 0.78, 0.10),
    (0.10, 0.30, 0.75),
    (0.80, 0.55, 0.30),
    (0.50, 0.50, 0.55),
]


def _scene_for_pose(
    model: Any,
    pose: dict[str, float],
    asset_root: Path,
    focus_spec: dict | None = None,
    segment: bool = False,
):
    import numpy as np
    import trimesh

    from sdk._core.v0.geometry_qc import _origin_to_mat4, compute_part_world_transforms

    pose = _redirect_followers_to_drivers(model, pose)
    world_tf = compute_part_world_transforms(model, pose)
    part_bounds: dict[str, Any] = {}
    materials_by_name = {getattr(m, "name", ""): m for m in getattr(model, "materials", []) or []}
    scene = trimesh.Scene()
    skipped = 0
    mat_specs: dict[str, dict] = {}
    for part_index, part in enumerate(getattr(model, "parts", [])):
        pname = getattr(part, "name", None)
        if pname is None:
            continue
        w = np.asarray(world_tf.get(pname, np.eye(4)), dtype=float)
        for idx, visual in enumerate(getattr(part, "visuals", []) or []):
            tm = _geometry_to_trimesh(getattr(visual, "geometry", None), asset_root)
            if tm is None:
                skipped += 1
                continue
            local = np.asarray(_origin_to_mat4(getattr(visual, "origin")), dtype=float)
            tm = tm.copy()
            tm.apply_transform(w @ local)
            b = getattr(tm, "bounds", None)
            if b is not None:
                prev = part_bounds.get(pname)
                if prev is None:
                    part_bounds[pname] = np.array(b, dtype=float)
                else:
                    prev[0] = np.minimum(prev[0], b[0])
                    prev[1] = np.maximum(prev[1], b[1])
            if segment:
                col = _SEGMENT_PALETTE[part_index % len(_SEGMENT_PALETTE)]
                seg_mat = trimesh.visual.material.PBRMaterial(
                    name=f"{pname}_seg_mat",
                    baseColorFactor=[col[0], col[1], col[2], 1.0],
                    metallicFactor=0.0,
                    roughnessFactor=0.55,
                    emissiveFactor=[col[0] * 0.18, col[1] * 0.18, col[2] * 0.18],
                )
                tm.visual = trimesh.visual.TextureVisuals(material=seg_mat)
                mat_specs[f"{pname}_seg_mat"] = {"emission": {"strength": 0.18, "color": list(col)}}
                scene.add_geometry(tm, node_name=f"{pname}.{idx}", geom_name=f"{pname}.{idx}")
                continue
            mat_obj = _resolve_material(getattr(visual, "material", None), materials_by_name)
            if mat_obj is not None:
                material_name = f"{pname}_{idx}_mat"
                material, renderer_spec, _source_spec = _visual_material(
                    mat_obj,
                    material_name,
                )
                if material is not None:
                    tm.visual = trimesh.visual.TextureVisuals(material=material)
                if renderer_spec:
                    mat_specs[material_name] = renderer_spec
            scene.add_geometry(tm, node_name=f"{pname}.{idx}", geom_name=f"{pname}.{idx}")
    if focus_spec:
        cr = _focus_center_radius(model, world_tf, part_bounds, focus_spec)
        if cr is not None:
            center, radius = cr
            marker = trimesh.creation.icosphere(radius=float(radius), subdivisions=1)
            marker.apply_translation([float(center[0]), float(center[1]), float(center[2])])
            scene.add_geometry(marker, node_name=_FOCUS_MARKER, geom_name=_FOCUS_MARKER)
    return scene, skipped, mat_specs


def _joint_metadata(joint: Any) -> dict[str, Any]:
    joint_type = getattr(joint, "articulation_type", None)
    payload: dict[str, Any] = {
        "name": str(getattr(joint, "name", "")),
        "type": str(getattr(joint_type, "value", joint_type)),
        "axis": [float(value) for value in getattr(joint, "axis", (0.0, 0.0, 1.0))],
    }
    limits = getattr(joint, "motion_limits", None)
    if limits is not None:
        payload["limit"] = {
            "lower": getattr(limits, "lower", None),
            "upper": getattr(limits, "upper", None),
        }
    mimic = getattr(joint, "mimic", None)
    if mimic is not None:
        payload["mimic"] = {
            "joint": str(getattr(mimic, "joint", "")),
            "multiplier": float(getattr(mimic, "multiplier", 1.0)),
            "offset": float(getattr(mimic, "offset", 0.0)),
        }
    return payload


_BLENDER_NAME_LIMIT = 63


def _blender_safe_node_name(name: str) -> str:
    raw = name.encode("utf-8")
    if len(raw) <= _BLENDER_NAME_LIMIT:
        return name
    digest = hashlib.sha1(raw).hexdigest()[:8]
    keep = _BLENDER_NAME_LIMIT - len(digest) - 1
    return f"{raw[:keep].decode('utf-8', 'ignore')}~{digest}"


def _build_articulated_scene(
    model: Any,
    asset_root: Path,
) -> tuple[Any, dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    import numpy as np
    import trimesh

    from pipeline.articulated_glb import (
        ARTICULATED_ROOT_NODE,
        LINK_NODE_PREFIX,
        VISUAL_NODE_PREFIX,
    )
    from sdk._core.v0.geometry_qc import _origin_to_mat4, compute_part_world_transforms

    parts = {
        str(getattr(part, "name")): part
        for part in getattr(model, "parts", [])
        if getattr(part, "name", None) is not None
    }
    joints = list(getattr(model, "articulations", []) or [])
    joint_by_child = {
        str(getattr(joint, "child")): joint
        for joint in joints
        if getattr(joint, "child", None) is not None
    }
    children_by_parent: dict[str, list[str]] = {}
    for child, joint in joint_by_child.items():
        children_by_parent.setdefault(str(getattr(joint, "parent")), []).append(child)

    rest_world = {
        name: np.asarray(matrix, dtype=float)
        for name, matrix in compute_part_world_transforms(model, {}).items()
    }
    scene = trimesh.Scene(base_frame="world")
    scene.graph.update(
        frame_to=ARTICULATED_ROOT_NODE,
        frame_from="world",
        matrix=np.asarray(_Z_UP_TO_Y_UP, dtype=float),
    )

    link_metadata: dict[str, dict[str, Any]] = {}
    visited: set[str] = set()

    def _add_link(link_name: str, parent_link: str | None) -> None:
        if link_name in visited:
            return
        visited.add(link_name)
        parent_frame = (
            ARTICULATED_ROOT_NODE
            if parent_link is None
            else _blender_safe_node_name(f"{LINK_NODE_PREFIX}{parent_link}")
        )
        local = rest_world.get(link_name, np.eye(4))
        if parent_link is not None:
            local = np.linalg.inv(rest_world.get(parent_link, np.eye(4))) @ local
        scene.graph.update(
            frame_to=_blender_safe_node_name(f"{LINK_NODE_PREFIX}{link_name}"),
            frame_from=parent_frame,
            matrix=local,
        )
        joint = joint_by_child.get(link_name)
        metadata: dict[str, Any] = {
            "kind": "link",
            "link": link_name,
            "parent_link": parent_link,
        }
        if joint is not None:
            metadata["joint"] = _joint_metadata(joint)
        link_metadata[link_name] = metadata
        for child in sorted(children_by_parent.get(link_name, [])):
            _add_link(child, link_name)

    roots = sorted(set(parts) - set(joint_by_child))
    for root in roots:
        _add_link(root, None)
    for name in sorted(set(parts) - visited):
        _add_link(name, None)

    materials_by_name = {
        getattr(material, "name", ""): material
        for material in getattr(model, "materials", []) or []
    }
    renderer_specs: dict[str, dict[str, Any]] = {}
    source_specs: dict[str, dict[str, Any]] = {}
    shared_materials: dict[tuple[str, str], tuple[str, Any]] = {}
    for link_name, part in parts.items():
        link_node = _blender_safe_node_name(f"{LINK_NODE_PREFIX}{link_name}")
        for index, visual in enumerate(getattr(part, "visuals", []) or []):
            mesh = _geometry_to_trimesh(getattr(visual, "geometry", None), asset_root)
            if mesh is None:
                continue
            mesh = mesh.copy()
            material_name = f"{link_name}_{index}_mat"
            mat_obj = _resolve_material(getattr(visual, "material", None), materials_by_name)
            material, renderer_spec, source_spec = _visual_material(mat_obj, material_name)
            if material is None:
                material = trimesh.visual.material.PBRMaterial(
                    name=material_name,
                    baseColorFactor=[0.8, 0.8, 0.8, 1.0],
                    metallicFactor=0.0,
                    roughnessFactor=0.8,
                )
                source_spec = {
                    "source_name": "",
                    "material_class": None,
                    "base_color_factor": [0.8, 0.8, 0.8, 1.0],
                    "metallic_factor": 0.0,
                    "roughness_factor": 0.8,
                    "emissive_strength": 0.0,
                    "texture_set": None,
                }
            visual_name = str(getattr(visual, "name", None) or f"visual_{index}")
            appearance = json.dumps((source_spec, renderer_spec), sort_keys=True, default=str)
            shared = shared_materials.get((link_name, appearance))
            if shared is None:
                shared_materials[(link_name, appearance)] = (material_name, material)
                if renderer_spec:
                    renderer_specs[material_name] = renderer_spec
                if source_spec:
                    source_spec["link"] = link_name
                    source_spec["visuals"] = [visual_name]
                    source_specs[material_name] = source_spec
            else:
                material_name, material = shared
                backing = source_specs.get(material_name)
                if backing is not None:
                    backing["visuals"].append(visual_name)
            mesh.visual = trimesh.visual.TextureVisuals(material=material)
            node_name = _blender_safe_node_name(
                f"{VISUAL_NODE_PREFIX}{link_name}::{index}::{visual_name}"
            )
            scene.add_geometry(
                mesh,
                node_name=node_name,
                geom_name=node_name,
                parent_node_name=link_node,
                transform=np.asarray(_origin_to_mat4(getattr(visual, "origin")), dtype=float),
            )
    return scene, renderer_specs, source_specs, link_metadata


def _export_articulated_glb(model: Any, asset_root: Path, out_dir: Path) -> Path:
    from pipeline.articulated_glb import (
        ARTICULATED_APPEARANCE_FILENAME,
        ARTICULATED_GLB_FILENAME,
        add_link_node_metadata,
        apply_standard_material_extensions,
    )

    scene, renderer_specs, source_specs, link_metadata = _build_articulated_scene(
        model,
        asset_root,
    )
    glb_path = out_dir / ARTICULATED_GLB_FILENAME
    scene.export(glb_path)
    apply_standard_material_extensions(glb_path, renderer_specs)
    add_link_node_metadata(glb_path, link_metadata)
    from sdk.runtime.articulated_bake import (
        ArticulatedBakeError,
        BakePolicy,
        bake_articulated_glb,
    )

    bake_result = bake_articulated_glb(
        glb_path,
        renderer_specs,
        source_specs,
    )
    if bake_result is not None:
        apply_standard_material_extensions(glb_path, renderer_specs)
        add_link_node_metadata(glb_path, link_metadata)
        final_size = glb_path.stat().st_size
        hard_limit = BakePolicy.from_env().hard_size_bytes
        if final_size > hard_limit:
            raise ArticulatedBakeError(
                f"final articulated GLB exceeds hard size limit: {final_size} > {hard_limit}"
            )
        bake_result["glb_size_bytes"] = final_size
    baked_channels = dict((bake_result or {}).get("baked_channels") or {})
    material_pages = dict((bake_result or {}).get("material_pages") or {})
    for material_name, material_spec in source_specs.items():
        channels = list(baked_channels.get(material_name) or [])
        material_spec["representation"] = "baked" if channels else "numeric"
        material_spec["baked_channels"] = channels
        if material_name in material_pages:
            material_spec["atlas_page"] = int(material_pages[material_name])
    (out_dir / ARTICULATED_APPEARANCE_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": "artisan-articulated-appearance-v2",
                "runtime_required": False,
                "glb": ARTICULATED_GLB_FILENAME,
                "glb_sha256": hashlib.sha256(glb_path.read_bytes()).hexdigest(),
                "texture_mode": (
                    str(bake_result.get("texture_mode"))
                    if bake_result is not None
                    else "numeric-only"
                ),
                "coordinate_system": {
                    "authoring_up": "+Z",
                    "gltf_up": "+Y",
                    "unit": "meter",
                },
                "links": link_metadata,
                "materials": source_specs,
                "bake": bake_result,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return glb_path


@timed_operation("render.pose_glbs")
def _build_pose_glbs(
    model_path: Path,
    pose_by_key: dict[str, dict[str, float]],
    out_dir: Path,
    sdk_package: str,
    focus_spec: dict | None = None,
    segment: bool = False,
    object_model: Any | None = None,
    object_asset_root: Path | None = None,
) -> dict[str, Path]:
    from pipeline.compiler import load_model_globals
    from sdk._core.v0.assets import activate_asset_session, asset_session_for_script

    out_dir.mkdir(parents=True, exist_ok=True)
    session = asset_session_for_script(Path(model_path))
    glbs: dict[str, Path] = {}
    with activate_asset_session(session):
        model = object_model
        if model is None:
            globals_dict = load_model_globals(Path(model_path), sdk_package=sdk_package)
            model = globals_dict.get("object_model")
        if model is None:
            raise ValueError("model.py must define a top-level `object_model`")
        asset_root = object_asset_root or Path(model_path).resolve().parent
        for key, pose in pose_by_key.items():
            scene, _skipped, mat_specs = _scene_for_pose(
                model, pose, asset_root, focus_spec, segment=segment
            )
            scene.apply_transform(_Z_UP_TO_Y_UP)
            glb_path = out_dir / f"_pose_{key}.glb"
            scene.export(glb_path)
            from pipeline.articulated_glb import apply_standard_material_extensions

            apply_standard_material_extensions(glb_path, mat_specs)
            glbs[key] = glb_path
            sidecar = out_dir / f"{glb_path.stem}.materials.json"
            if mat_specs:
                sidecar.write_text(json.dumps(mat_specs, indent=2), encoding="utf-8")
            elif sidecar.exists():
                sidecar.unlink()
    return glbs


def _export_closed_pose_artifacts(
    model: Any,
    *,
    asset_root: Path,
    out_dir: Path,
    want_glb: bool,
    want_obj: bool,
) -> dict[str, str]:
    exports: dict[str, str] = {}
    scene, _skipped, mat_specs = _scene_for_pose(model, {}, asset_root)

    if want_obj:
        obj_path = out_dir / "object.obj"
        scene.export(obj_path)
        exports["obj"] = str(obj_path)

    if want_glb:
        glb_scene = scene.copy()
        glb_scene.apply_transform(_Z_UP_TO_Y_UP)
        glb_path = out_dir / "_pose_closed.glb"
        glb_scene.export(glb_path)
        from pipeline.articulated_glb import apply_standard_material_extensions

        apply_standard_material_extensions(glb_path, mat_specs)
        exports["glb"] = str(glb_path)
        sidecar = out_dir / f"{glb_path.stem}.materials.json"
        if mat_specs:
            sidecar.write_text(json.dumps(mat_specs, indent=2), encoding="utf-8")
        elif sidecar.exists():
            sidecar.unlink()
        articulated_path = _export_articulated_glb(model, asset_root, out_dir)
        exports["articulated_glb"] = str(articulated_path)
        from pipeline.articulated_glb import ARTICULATED_APPEARANCE_FILENAME

        exports["articulated_appearance"] = str(out_dir / ARTICULATED_APPEARANCE_FILENAME)
    return exports


def _load_object_model(model_path: Path, sdk_package: str) -> tuple[Any, Path]:
    from pipeline.compiler import load_model_globals
    from sdk._core.v0.assets import activate_asset_session, asset_session_for_script

    session = asset_session_for_script(Path(model_path))
    with activate_asset_session(session):
        globals_dict = load_model_globals(Path(model_path), sdk_package=sdk_package)
    object_model = globals_dict.get("object_model")
    if object_model is None:
        raise ValueError("model.py must define a top-level `object_model`")
    return object_model, Path(model_path).resolve().parent


def _pose_key_for_view(view: str) -> str:
    return "open" if view.endswith("_open") else "closed"


class _BlenderDaemon:

    def __init__(self, policy: RenderDevicePolicy | None = None) -> None:
        self.proc: subprocess.Popen | None = None
        self.device_policy = policy or RenderDevicePolicy.from_environment()
        self._stdout_buffer = b""

    def _alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _wait_for(self, marker: str, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        assert self.proc is not None and self.proc.stdout is not None
        while time.monotonic() < deadline:
            while b"\n" in self._stdout_buffer:
                line, self._stdout_buffer = self._stdout_buffer.split(b"\n", 1)
                if line.decode("utf-8", errors="replace").strip() == marker:
                    return True
            if self.proc.poll() is not None:
                return False
            r, _, _ = select.select(
                [self.proc.stdout], [], [], min(1.0, max(0.0, deadline - time.monotonic()))
            )
            if not r:
                continue
            chunk = os.read(self.proc.stdout.fileno(), 65536)
            if not chunk:
                return False
            self._stdout_buffer += chunk
        return False

    def _start(self, timeout_s: float = 180.0) -> None:
        from pipeline.blender.runtime import blender_bin

        self.proc = subprocess.Popen(
            [blender_bin(), "--background", "--factory-startup", "--python", str(_SERVER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=_device_environment(self.device_policy),
            start_new_session=True,
        )
        self._stdout_buffer = b""
        if not self._wait_for(_READY_MARKER, timeout_s=min(180.0, timeout_s)):
            self._kill()
            raise RuntimeError("Blender render server failed to start")

    def render(
        self,
        glb: Path,
        views: list[str],
        out_dir: Path,
        report: Path,
        camera: str | None,
        timeout_s: float,
        *,
        request_id: str,
        expected_input: dict[str, Any],
    ) -> dict:
        deadline = time.monotonic() + timeout_s
        if not self._alive():
            self._start(timeout_s=timeout_s)
        req = {
            "glb": str(glb),
            "views": list(views),
            "out_dir": str(out_dir),
            "report": str(report),
            "camera": camera or "",
            "request_id": request_id,
            "input_identity": expected_input,
        }
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        if not self._wait_for(
            f"{_DONE_MARKER} {request_id}", timeout_s=max(0.0, deadline - time.monotonic())
        ):
            self._kill()
            raise RuntimeError("Blender render server did not finish in time")
        return json.loads(report.read_text(encoding="utf-8"))

    def _kill(self) -> None:
        if self.proc is not None:
            _stop_render_process(self.proc)
        self.proc = None
        self._stdout_buffer = b""

    def close(self) -> None:
        if self._alive():
            try:
                assert self.proc is not None and self.proc.stdin is not None
                self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                self._kill()
        if self.proc is not None:
            _close_render_pipes(self.proc)
        self.proc = None


_DAEMON: _BlenderDaemon | None = None
_RENDER_LOCK = threading.Lock()


def _get_daemon(policy: RenderDevicePolicy | None = None) -> _BlenderDaemon:
    global _DAEMON
    policy = policy or RenderDevicePolicy.from_environment()
    if _DAEMON is not None and _DAEMON.device_policy != policy:
        _DAEMON.close()
        _DAEMON = None
    if _DAEMON is None:
        _DAEMON = _BlenderDaemon(policy)
        atexit.register(_DAEMON.close)
    return _DAEMON


def _device_environment(policy: RenderDevicePolicy) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        ARTISAN_RENDER_DEVICE=policy.mode,
        ARTISAN_RENDER_GPU_BACKEND=policy.backend,
        ARTISAN_RENDER_GPU_INDEX=str(policy.gpu_index),
    )
    return environment


def _close_render_pipes(process: subprocess.Popen) -> None:
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            stream.close()


def _stop_render_process(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)
    _close_render_pipes(process)


@timed_operation("render.driver")
def _run_render_driver(
    glb: Path,
    views: list[str],
    out_dir: Path,
    timeout_s: int = 900,
    *,
    camera: str | None = None,
    mode: str = "review",
) -> dict:
    started = time.monotonic()
    deadline = started + timeout_s
    if not _RENDER_LOCK.acquire(timeout=max(0.0, timeout_s)):
        raise TimeoutError("Render request expired in the worker queue")
    try:
        return _render_transaction(
            Path(glb), views, Path(out_dir), camera=camera, mode=mode, deadline=deadline
        )
    finally:
        _RENDER_LOCK.release()


def _render_transaction(
    glb: Path,
    views: list[str],
    out_dir: Path,
    *,
    camera: str | None,
    mode: str,
    deadline: float,
) -> dict:
    expected = input_identity(glb)
    requested_policy = RenderDevicePolicy.from_environment()
    policy = requested_policy
    fallback_reason: str | None = None
    out_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Render request deadline exhausted")
        request_id = uuid.uuid4().hex
        attempt_root = out_dir / ".render_attempts" / request_id
        attempt_root.mkdir(parents=True)
        staged_glb = attempt_root / glb.name
        staged_identity = stage_input(glb, staged_glb, expected)
        try:
            report = _execute_render_attempt(
                staged_glb,
                views,
                attempt_root,
                camera=camera,
                mode=mode,
                deadline=deadline,
                request_id=request_id,
                expected_input=staged_identity,
                policy=policy,
                force_one_shot=attempt > 0,
            )
            images = validate_response(
                report,
                request_id=request_id,
                expected_input=staged_identity,
                views=views,
                attempt_root=attempt_root,
            )
            actual = (report.get("device") or {}).get("actual")
            if policy.mode == "gpu" and actual != policy.backend:
                raise RenderIntegrityError("GPU request was not rendered on the requested backend")
            if policy.mode == "cpu" and actual != "CPU":
                raise RenderIntegrityError("CPU request was not rendered on the CPU")
            verify_input(glb, expected)
            published = {}
            for view, source in images.items():
                destination = out_dir / source.name
                temporary = out_dir / f".{source.name}.{request_id}.tmp"
                shutil.copy2(source, temporary)
                temporary.replace(destination)
                published[view] = str(destination)
            report.update(
                glb=str(glb),
                staged_glb=str(staged_glb),
                renders=published,
                requested_device=requested_policy.to_dict(),
                attempt=attempt,
                fallback_reason=fallback_reason,
                source_input_identity=expected,
            )
            write_report(out_dir / f"_render_{glb.stem}.json", report)
            return report
        except RenderIntegrityError:
            if _DAEMON is not None:
                _DAEMON._kill()
            raise
        except Exception as exc:
            if _DAEMON is not None:
                _DAEMON._kill()
            if attempt or deadline <= time.monotonic():
                raise
            fallback_reason = f"{type(exc).__name__}: {exc}"
            if requested_policy.mode == "auto":
                policy = RenderDevicePolicy(mode="cpu")
        except BaseException:
            if _DAEMON is not None:
                _DAEMON._kill()
            raise
    raise RuntimeError("Render attempts exhausted")


def _execute_render_attempt(
    glb: Path,
    views: list[str],
    out_dir: Path,
    *,
    camera: str | None,
    mode: str,
    deadline: float,
    request_id: str,
    expected_input: dict[str, Any],
    policy: RenderDevicePolicy,
    force_one_shot: bool,
) -> dict:
    from pipeline.blender.runtime import blender_bin

    report = out_dir / f"_render_{glb.stem}.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    use_daemon = (
        mode == "review"
        and os.environ.get("ARTISAN_BLENDER_NO_DAEMON") != "1"
        and not force_one_shot
    )
    if use_daemon:
        return _get_daemon(policy).render(
            glb,
            views,
            out_dir,
            report,
            camera,
            max(0.0, deadline - time.monotonic()),
            request_id=request_id,
            expected_input=expected_input,
        )

    request_file = out_dir / "request.json"
    write_report(request_file, {"request_id": request_id, "input_identity": expected_input})

    cmd = [
        blender_bin(),
        "--background",
        "--factory-startup",
        "--python",
        str(_DRIVER),
        "--",
        "--glb",
        str(glb),
        "--views",
        ",".join(views),
        "--out-dir",
        str(out_dir),
        "--report",
        str(report),
        "--request-file",
        str(request_file),
    ]
    if camera:
        cmd += ["--camera", camera]
    if mode and mode != "review":
        cmd += ["--mode", mode]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_device_environment(policy),
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=max(0.0, deadline - time.monotonic()))
    except BaseException:
        _stop_render_process(proc)
        raise
    if not report.exists():
        raise RuntimeError(
            f"render driver produced no report (rc={proc.returncode}).\n"
            f"STDOUT:\n{stdout[-2000:]}\nSTDERR:\n{stderr[-2000:]}"
        )
    return json.loads(report.read_text(encoding="utf-8"))


def _rename_segment_outputs(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        dst = p.with_name(f"{p.stem}_seg{p.suffix}")
        try:
            p.replace(dst)
            out.append(dst)
        except OSError:
            out.append(p)
    return out


@contextmanager
def _render_model(
    model_path: Path,
    sdk_package: str,
    inspection_snapshot: dict[str, str] | None,
) -> Iterator[tuple[Any, Path]]:
    if inspection_snapshot is not None:
        from sdk.runtime.inspection import private_snapshot

        with private_snapshot(model_path, inspection_snapshot, sdk_package=sdk_package) as data:
            yield data[0], data[1]
        return
    from pipeline.compiler import load_model_globals
    from sdk._core.v0.assets import activate_asset_session, asset_session_for_script

    with activate_asset_session(asset_session_for_script(model_path)):
        model = load_model_globals(model_path, sdk_package=sdk_package).get("object_model")
        if model is None:
            raise ValueError("model.py must define a top-level `object_model`")
        yield model, model_path.resolve().parent


def render_views(
    model_path: Path,
    views: list[str],
    out_dir: Path,
    *,
    sdk_package: str = "sdk",
    pose: dict[str, float] | None = None,
    camera: dict[str, float] | None = None,
    focus: list[str] | None = None,
    mode: str = "review",
    segment: bool = False,
    inspection_snapshot: dict[str, str] | None = None,
) -> list[Path]:
    model_path = Path(model_path)
    out_dir = Path(out_dir)

    from sdk.runtime.performance_policy import PerformancePolicy

    reuse_model = inspection_snapshot is not None or PerformancePolicy.from_env().reuse_render_model
    with _render_model(model_path, sdk_package, inspection_snapshot) as (model, asset_root):
        focus_spec = None
        if focus:
            focus_spec = _resolve_focus(model, focus)

        if camera is not None:
            pose_by_key = {"custom": dict(pose) if pose else {}}
            glbs = _build_pose_glbs(
                model_path,
                pose_by_key,
                out_dir,
                sdk_package,
                focus_spec,
                segment=segment,
                object_model=model if reuse_model else None,
                object_asset_root=asset_root,
            )
            name = views[0] if views else "free_cam"
            cam_arg = (
                f"{float(camera.get('azimuth_deg', 45.0))},"
                f"{float(camera.get('elevation_deg', 20.0))},"
                f"{float(camera.get('distance_mult', 3.0))}"
            )
            report = _run_render_driver(glbs["custom"], [name], out_dir, camera=cam_arg, mode=mode)
            out = [Path(p) for p in report.get("renders", {}).values() if Path(p).exists()]
            return _rename_segment_outputs(out) if segment else out

        if pose is not None:
            pose_by_key = {"custom": dict(pose)}
            view_keys = {v: "custom" for v in views}
        else:
            open_pose = _open_pose(model)
            pose_by_key = {"closed": {}, "open": open_pose}
            view_keys = {v: _pose_key_for_view(v) for v in views}
            used = set(view_keys.values())
            pose_by_key = {k: v for k, v in pose_by_key.items() if k in used}

        glbs = _build_pose_glbs(
            model_path,
            pose_by_key,
            out_dir,
            sdk_package,
            focus_spec,
            segment=segment,
            object_model=model if reuse_model else None,
            object_asset_root=asset_root,
        )

        rendered: list[Path] = []
        for key, glb in glbs.items():
            group = [v for v, k in view_keys.items() if k == key]
            if not group:
                continue
            report = _run_render_driver(glb, group, out_dir, mode=mode)
            for p in report.get("renders", {}).values():
                if Path(p).exists():
                    rendered.append(Path(p))
        return _rename_segment_outputs(rendered) if segment else rendered


def _package_urdf_meshes(model_path: Path, out_dir: Path, sdk_package: str) -> int:
    import shutil

    from pipeline.compiler import load_model_globals
    from sdk._core.v0.assets import activate_asset_session, asset_session_for_script
    from sdk._core.v0.types import Mesh

    copied = 0
    session = asset_session_for_script(Path(model_path))
    with activate_asset_session(session):
        globals_dict = load_model_globals(Path(model_path), sdk_package=sdk_package)
        model = globals_dict.get("object_model")
        if model is None:
            return 0
        for part in getattr(model, "parts", []):
            elems = list(getattr(part, "visuals", []) or []) + list(
                getattr(part, "collisions", []) or []
            )
            for el in elems:
                geo = getattr(el, "geometry", None)
                if not isinstance(geo, Mesh):
                    continue
                rel = getattr(geo, "filename", None)
                src = getattr(geo, "materialized_path", None)
                if not rel or not src:
                    continue
                dst = out_dir / str(rel)
                try:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
                    copied += 1
                except Exception:  # noqa: BLE001 — best-effort per mesh
                    pass
    return copied


def export_per_visual_meshes(
    model_path: Path,
    out_dir: Path,
    sdk_package: str,
    *,
    object_model: Any | None = None,
    asset_root: Path | None = None,
) -> dict[tuple[str, int], str]:
    import trimesh

    from pipeline.compiler import load_model_globals
    from sdk._core.v0.assets import activate_asset_session, asset_session_for_script

    meshes_rel: dict[tuple[str, int], str] = {}

    def _export(model: Any, resolved_asset_root: Path) -> None:
        materials_by_name = {
            getattr(m, "name", ""): m for m in getattr(model, "materials", []) or []
        }
        mesh_dir = out_dir / "assets" / "meshes"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        for part in getattr(model, "parts", []):
            pname = getattr(part, "name", None)
            if pname is None:
                continue
            for idx, visual in enumerate(getattr(part, "visuals", []) or []):
                tm = _geometry_to_trimesh(getattr(visual, "geometry", None), resolved_asset_root)
                if tm is None:
                    continue
                tm = tm.copy()
                mat_obj = _resolve_material(getattr(visual, "material", None), materials_by_name)
                rgba = _material_rgba(mat_obj) if mat_obj is not None else None
                base = rgba if rgba is not None else [0.8, 0.8, 0.8, 1.0]
                tm.visual = trimesh.visual.TextureVisuals(
                    material=trimesh.visual.material.SimpleMaterial(
                        name=f"{pname}_{idx}_mat",
                        diffuse=[float(c) for c in base],
                    )
                )
                rel = f"assets/meshes/{pname}__{idx}.obj"
                tm.export(out_dir / rel, include_texture=True)
                meshes_rel[(pname, idx)] = rel

    if object_model is not None and asset_root is not None:
        _export(object_model, Path(asset_root))
        return meshes_rel

    session = asset_session_for_script(Path(model_path))
    with activate_asset_session(session):
        globals_dict = load_model_globals(Path(model_path), sdk_package=sdk_package)
        model = globals_dict.get("object_model")
        if model is None:
            return meshes_rel
        _export(model, Path(model_path).resolve().parent)
    return meshes_rel


def _rewrite_urdf_visuals(urdf_xml: str, meshes_rel: dict[tuple[str, int], str]) -> str:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(urdf_xml)
    for link in root.findall("link"):
        lname = link.get("name")
        for vi, visual in enumerate(link.findall("visual")):
            geom = visual.find("geometry")
            rel = meshes_rel.get((lname, vi))
            if geom is None or rel is None:
                continue
            for child in list(geom):
                geom.remove(child)
            ET.SubElement(geom, "mesh", {"filename": rel})
    return ET.tostring(root, encoding="unicode")


def default_export_formats() -> list[str]:
    return ["glb", "obj", "urdf", "mjcf", "usd"]


def _compile_and_enrich(model_path: Path, sdk_package: str):
    from sdk.runtime.physics_candidate import build_physics_candidate

    candidate = build_physics_candidate(Path(model_path), sdk_package=sdk_package)
    return (
        candidate.raw_model,
        candidate.resolved_model,
        candidate.asset_root,
        candidate.warnings,
    )


def _declared_overlap_pairs(model_path: Path) -> list[tuple[str, str]]:
    from sdk.runtime import runtime as sdk_runtime

    try:
        source = Path(model_path).read_text(encoding="utf-8")
        return [
            (item["a"], item["b"])
            for item in sdk_runtime.extract_declared_allowances(source)
            if item.get("kind") == "overlap" and item.get("a") and item.get("b")
        ]
    except Exception:  # noqa: BLE001 — exclusion hints are best-effort
        return []


def _package_model_meshes(enriched: Any, model_path: Path, out_dir: Path, sdk_package: str) -> int:
    import shutil

    from sdk._core.v0.types import Mesh

    if enriched is None:
        return _package_urdf_meshes(model_path, out_dir, sdk_package)
    copied = 0
    for part in getattr(enriched, "parts", []):
        for el in list(getattr(part, "visuals", []) or []) + list(
            getattr(part, "collisions", []) or []
        ):
            geo = getattr(el, "geometry", None)
            if not isinstance(geo, Mesh):
                continue
            rel = getattr(geo, "filename", None)
            src = getattr(geo, "materialized_path", None)
            if not rel or not src:
                continue
            dst = out_dir / str(rel)
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                copied += 1
            except Exception:  # noqa: BLE001 — best-effort per mesh
                pass
    return copied


def export(
    model_path: Path,
    formats: list[str],
    out_dir: Path,
    *,
    sdk_package: str = "sdk",
    collision_manifest: ResolvedCollisionManifest | None = None,
) -> dict[str, Any]:
    from sdk.runtime import runtime as sdk_runtime

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exports: dict[str, str] = {}
    skipped: list[str] = []

    want_urdf = "urdf" in formats
    want_mjcf = "mjcf" in formats
    want_usd = "usd" in formats

    object_model = enriched = asset_root = candidate = None
    enrich_notes: list[str] = []
    declared_overlap_pairs = _declared_overlap_pairs(model_path)
    if want_urdf or want_mjcf or want_usd or collision_manifest is not None:
        try:
            from sdk.runtime.physics_candidate import build_physics_candidate

            candidate = build_physics_candidate(
                model_path,
                sdk_package=sdk_package,
                declared_overlap_pairs=declared_overlap_pairs,
                collision_mode="full",
                collision_manifest=collision_manifest,
            )
            object_model = candidate.raw_model
            enriched = candidate.resolved_model
            asset_root = candidate.asset_root
            enrich_notes = candidate.warnings
        except Exception as exc:  # noqa: BLE001 — enrichment must never break export
            if collision_manifest is not None:
                raise
            skipped.append(f"physics-enrich-failed:{type(exc).__name__}")
        if enriched is not None:
            _package_model_meshes(enriched, model_path, out_dir, sdk_package)

    if want_urdf:
        urdf_xml: Any = None
        if enriched is not None and object_model is not None:
            try:
                from sdk._core.v0._urdf_export import compile_object_to_urdf_xml

                urdf_xml = compile_object_to_urdf_xml(
                    object_model, compiled_model=enriched, asset_root=asset_root
                )
            except Exception:  # noqa: BLE001 — fall back to the compiled URDF below
                urdf_xml = None
        if not isinstance(urdf_xml, str):
            if collision_manifest is not None:
                raise ValueError("frozen build URDF export failed; refusing generator fallback")
            _r, _b, legacy_xml, _ok = sdk_runtime.compile_report(
                model_path, sdk_package=sdk_package
            )
            urdf_xml = legacy_xml if isinstance(legacy_xml, str) else None
            if enriched is None:
                _package_urdf_meshes(model_path, out_dir, sdk_package)
        if isinstance(urdf_xml, str):
            if os.environ.get("ARTISAN_URDF_VISUAL_MESH", "1") != "0":
                try:
                    meshes_rel = export_per_visual_meshes(
                        model_path,
                        out_dir,
                        sdk_package,
                        object_model=object_model,
                        asset_root=asset_root,
                    )
                    if meshes_rel:
                        urdf_xml = _rewrite_urdf_visuals(urdf_xml, meshes_rel)
                        exports["urdf_visual_meshes"] = str(len(meshes_rel))
                except Exception:  # noqa: BLE001 — visual-mesh rewrite must not break export
                    pass
            (out_dir / "object.urdf").write_text(urdf_xml, encoding="utf-8")
            exports["urdf"] = str(out_dir / "object.urdf")

    if want_mjcf:
        if enriched is not None and object_model is not None:
            try:
                from sdk._core.v0._mjcf_export import compile_object_to_mjcf_xml

                assert candidate is not None
                (out_dir / "object.xml").write_bytes(candidate.mjcf_bytes)
                exports["mjcf"] = str(out_dir / "object.xml")

                try:
                    scene_xml = compile_object_to_mjcf_xml(
                        object_model,
                        compiled_model=enriched,
                        asset_root=asset_root,
                        position_servos=False,
                        scene_mode=True,
                    )
                    (out_dir / "object_scene.xml").write_text(scene_xml, encoding="utf-8")
                    exports["mjcf_scene"] = str(out_dir / "object_scene.xml")
                except Exception as exc:  # noqa: BLE001 — scene is best-effort
                    skipped.append(f"mjcf_scene:{type(exc).__name__}")
            except Exception as exc:  # noqa: BLE001
                skipped.append(f"mjcf:{type(exc).__name__}")
        else:
            skipped.append("mjcf:no-model")

    if want_usd:
        from sdk._dependencies import sim_extra_available

        if not sim_extra_available("pxr"):
            skipped.append("usd (uv sync --extra usd)")
        elif enriched is not None and object_model is not None:
            try:
                from sdk._core.v0._usd_export import compile_object_to_usd

                usd_path = compile_object_to_usd(
                    object_model,
                    out_dir / "object.usd",
                    asset_root=asset_root,
                    compiled_model=enriched,
                    declared_overlap_pairs=declared_overlap_pairs,
                )
                exports["usd"] = str(usd_path)
            except Exception as exc:  # noqa: BLE001
                skipped.append(f"usd:{type(exc).__name__}")
        else:
            skipped.append("usd:no-model")

    if enrich_notes:
        exports["physics_warnings"] = str(len(enrich_notes))

    want_glb = "glb" in formats
    want_obj = "obj" in formats
    if want_glb or want_obj:
        try:
            if object_model is None or asset_root is None:
                object_model, asset_root = _load_object_model(model_path, sdk_package)
            exports.update(
                _export_closed_pose_artifacts(
                    object_model,
                    asset_root=asset_root,
                    out_dir=out_dir,
                    want_glb=want_glb,
                    want_obj=want_obj,
                )
            )
        except Exception as exc:  # noqa: BLE001 — preserve best-effort general export API
            for fmt in ("glb", "obj"):
                if fmt in formats:
                    skipped.append(f"{fmt}:{type(exc).__name__}")

    for fmt in formats:
        if fmt not in exports and not any(s.split(":")[0].split(" ")[0] == fmt for s in skipped):
            skipped.append(fmt)

    coupling_notes = _coupling_linearization_notes(object_model)
    return {
        "exports": exports,
        "skipped": skipped,
        "physics_notes": enrich_notes,
        "physics_candidate": candidate.summary() if candidate is not None else None,
        "coupling_notes": coupling_notes,
    }


def _coupling_linearization_notes(model: Any) -> list[str]:
    notes: list[str] = []
    for art in getattr(model, "articulations", None) or []:
        mimic = getattr(art, "mimic", None)
        if mimic is None or not getattr(mimic, "poly", ()):
            continue
        meta = getattr(art, "meta", None)
        cp = meta.get("coupling") if isinstance(meta, dict) else None
        kind = cp.get("kind") if isinstance(cp, dict) else "nonlinear"
        fit_err = cp.get("fit_error") if isinstance(cp, dict) else None
        msg = (
            f"coupling {getattr(art, 'name', '?')!r} ({kind}): URDF/USD/viewer use a "
            f"linear (secant) approximation; MJCF carries the full curve"
        )
        if isinstance(fit_err, (int, float)):
            msg += f" (degree-4 fit error {fit_err * 100:.1f}% of stroke)"
        notes.append(msg)
    return notes


def crop_image(
    image_path: Path,
    box: tuple[float, float, float, float],
    out_dir: Path | None = None,
) -> Path:
    from PIL import Image

    image_path = Path(image_path)
    out_dir = Path(out_dir) if out_dir is not None else image_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as im:
        w, h = im.size
        left, top, right, bottom = box
        left, right = sorted((max(0.0, min(1.0, left)), max(0.0, min(1.0, right))))
        top, bottom = sorted((max(0.0, min(1.0, top)), max(0.0, min(1.0, bottom))))
        px = (
            int(left * w),
            int(top * h),
            max(int(right * w), int(left * w) + 1),
            max(int(bottom * h), int(top * h) + 1),
        )
        cropped = im.crop(px)
        out = out_dir / f"{image_path.stem}_crop.png"
        cropped.save(out)
    return out
