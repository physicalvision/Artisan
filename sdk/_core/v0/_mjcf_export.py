
from __future__ import annotations

import hashlib
import math
import os
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

from ._contact_exemptions import contact_exempt_pairs as _contact_exempt_pairs_impl
from ._coupling import driver_effective_ranges
from ._mesh_hull import check_mesh_file_hull
from .articulated_object import ArticulatedObject
from .exact_collisions import compile_object_model_with_exact_collisions
from .geometry_qc import _rpy_matrix
from .motion_engine.contract import collision_geom_name, mesh_asset_identity
from .types import (
    ArticulationType,
    Box,
    Cylinder,
    Material,
    Mesh,
    Sphere,
)

_MESHDIR = "assets/meshes"


def compile_object_to_mjcf_xml(
    object_model: ArticulatedObject,
    *,
    asset_root: str | os.PathLike[str] | Path | None = None,
    meshdir: str = _MESHDIR,
    include_actuators: bool = True,
    position_servos: bool = True,
    validate: bool = True,
    compiled_model: ArticulatedObject | None = None,
    declared_overlap_pairs: Iterable[tuple[str, str]] | None = None,
    scene_mode: bool = False,
    collision_mode: str | None = None,
) -> str:
    if collision_mode is None:
        collision_mode = "full" if scene_mode else "exempt"
    if collision_mode not in ("full", "exempt"):
        raise ValueError(f"collision_mode must be 'full' or 'exempt', got {collision_mode!r}")
    full_collision = collision_mode == "full"
    if compiled_model is None:
        if validate:
            object_model.validate(strict=True)
        compiled_model = compile_object_model_with_exact_collisions(
            object_model, asset_root=asset_root, validate=False
        )

    materials_by_name = {
        getattr(m, "name", ""): m for m in getattr(compiled_model, "materials", []) or []
    }

    root = ET.Element("mujoco", {"model": object_model.name})
    ET.SubElement(
        root,
        "compiler",
        {
            "angle": "radian",
            "meshdir": meshdir,
            "autolimits": "true",
            "balanceinertia": "true",
        },
    )
    if full_collision:
        _option = ET.SubElement(root, "option")
        ET.SubElement(_option, "flag", {"filterparent": "disable"})
    _append_defaults(root)

    shell_assets: set[str] = set()
    mesh_registry = _collect_meshes(
        compiled_model,
        meshdir=meshdir,
        asset_root=Path(asset_root) if asset_root is not None else None,
        shell_assets=shell_assets,
    )
    _require_explicit_inertial_for_shell(compiled_model, mesh_registry, shell_assets)
    if mesh_registry:
        asset = ET.SubElement(root, "asset")
        declared_assets: set[str] = set()
        for mesh_name, file_rel, scale in mesh_registry.values():
            if mesh_name in declared_assets:
                continue
            declared_assets.add(mesh_name)
            attrs = {"name": mesh_name, "file": file_rel}
            if scale != (1.0, 1.0, 1.0):
                attrs["scale"] = _fmt_vec(scale)
            if mesh_name in shell_assets:
                attrs["inertia"] = "shell"
            ET.SubElement(asset, "mesh", attrs)

    worldbody = ET.SubElement(root, "worldbody")
    parts_by_name = {p.name: p for p in compiled_model.parts}
    children_of, joint_for_child = _articulation_maps(compiled_model)
    roots = _root_parts(compiled_model, joint_for_child)
    effective_ranges = driver_effective_ranges(compiled_model)
    geom_names: set[str] = set()
    for part in roots:
        _append_body(
            worldbody,
            part,
            parts_by_name=parts_by_name,
            children_of=children_of,
            joint_for_child=joint_for_child,
            mesh_registry=mesh_registry,
            materials_by_name=materials_by_name,
            geom_names=geom_names,
            effective_ranges=effective_ranges,
        )

    if include_actuators:
        _append_actuators(root, compiled_model, effective_ranges, position_servos=position_servos)
    _append_equality(root, compiled_model)
    if scene_mode:
        _append_scene_rig(worldbody)
    if full_collision:
        from ._contact_exemptions import coupling_contact_pairs

        _append_contact_excludes(
            root, sorted(coupling_contact_pairs(compiled_model, asset_root=asset_root))
        )
    else:
        _append_contact_excludes(
            root,
            _contact_exclude_pairs(
                compiled_model,
                asset_root=asset_root,
                declared_overlap_pairs=declared_overlap_pairs,
            ),
        )

    _indent(root)
    return ET.tostring(root, encoding="unicode")


def _append_scene_rig(worldbody: ET.Element) -> None:
    for body in list(worldbody.findall("body")):
        if body.find("freejoint") is None and body.find("joint") is None:
            body.insert(0, ET.Element("freejoint", {"name": f"{body.get('name', 'root')}_free"}))
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "scene_ground",
            "type": "plane",
            "size": "0 0 0.05",
            "pos": "0 0 0",
            "rgba": "0.55 0.55 0.58 1",
        },
    )
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "scene_light",
            "pos": "0 0 4",
            "dir": "0 0 -1",
            "directional": "true",
            "diffuse": "0.8 0.8 0.8",
        },
    )


def _append_defaults(root: ET.Element) -> None:
    default = ET.SubElement(root, "default")
    vis = ET.SubElement(default, "default", {"class": "visual"})
    ET.SubElement(vis, "geom", {"group": "2", "contype": "0", "conaffinity": "0"})
    col = ET.SubElement(default, "default", {"class": "collision"})
    ET.SubElement(col, "geom", {"group": "3"})


def _mesh_name_from_rel(file_rel: str) -> str:
    stem = file_rel.replace("\\", "/")
    for ch in "/.":
        stem = stem.replace(ch, "_")
    return stem.strip("_")


def _strip_meshdir(filename: str, meshdir: str) -> str:
    fn = str(filename).replace("\\", "/")
    prefix = meshdir.rstrip("/") + "/"
    if fn.startswith(prefix):
        return fn[len(prefix) :]
    return fn


def _collect_meshes(
    compiled_model: ArticulatedObject,
    *,
    meshdir: str,
    asset_root: Path | None,
    shell_assets: set[str] | None = None,
) -> dict[tuple[str, tuple[float, float, float]], tuple[str, str, tuple[float, float, float]]]:
    registry: dict[
        tuple[str, tuple[float, float, float]],
        tuple[str, str, tuple[float, float, float]],
    ] = {}
    content_assets: dict[
        tuple[str, tuple[float, float, float]],
        tuple[str, str, tuple[float, float, float]],
    ] = {}
    used_names: set[str] = set()
    for part in compiled_model.parts:
        elems = list(part.visuals) + list(part.collisions)
        for el in elems:
            geom = getattr(el, "geometry", None)
            if not isinstance(geom, Mesh) or not geom.filename:
                continue
            filename, scale = mesh_asset_identity(geom)
            key = (filename, scale)
            if key in registry:
                continue
            file_rel = _strip_meshdir(filename, meshdir)
            materialized = getattr(geom, "materialized_path", None)
            source_path = (
                Path(materialized)
                if materialized and Path(materialized).is_file()
                else ((asset_root / filename) if asset_root is not None else Path(filename))
            )
            content_digest = (
                hashlib.sha256(source_path.read_bytes()).hexdigest()
                if source_path.is_file()
                else hashlib.sha256(filename.encode("utf-8")).hexdigest()
            )
            content_key = (content_digest, scale)
            existing_asset = content_assets.get(content_key)
            if existing_asset is not None:
                registry[key] = existing_asset
                continue
            name = _mesh_name_from_rel(file_rel)
            if scale != (1.0, 1.0, 1.0):
                scale_id = hashlib.sha256(
                    ",".join(repr(float(value)) for value in scale).encode("utf-8")
                ).hexdigest()[:10]
                name = f"{name}_scale_{scale_id}"
            base = name or "mesh"
            i = 0
            while name in used_names or not name:
                i += 1
                name = f"{base}_{i}"
            used_names.add(name)
            registry[key] = (name, file_rel, scale)
            content_assets[content_key] = registry[key]
            if (
                shell_assets is not None
                and source_path.is_file()
                and _needs_shell_inertia(content_digest, source_path, scale)
            ):
                shell_assets.add(name)
    return registry


_SHELL_INERTIA_CACHE: dict[tuple[str, tuple[float, float, float]], bool] = {}


def _needs_shell_inertia(
    content_digest: str, source_path: Path, scale: tuple[float, float, float]
) -> bool:
    key = (content_digest, scale)
    cached = _SHELL_INERTIA_CACHE.get(key)
    if cached is None:
        check = check_mesh_file_hull(source_path, scale)
        cached = check is not None and check.kind != "ok"
        _SHELL_INERTIA_CACHE[key] = cached
    return cached


def _require_explicit_inertial_for_shell(
    compiled_model: ArticulatedObject,
    mesh_registry: dict[
        tuple[str, tuple[float, float, float]],
        tuple[str, str, tuple[float, float, float]],
    ],
    shell_assets: set[str],
) -> None:
    if not shell_assets:
        return
    for part in compiled_model.parts:
        if getattr(part, "inertial", None) is not None:
            continue
        for element in [*part.visuals, *part.collisions]:
            geom = getattr(element, "geometry", None)
            if not isinstance(geom, Mesh) or not geom.filename:
                continue
            entry = mesh_registry.get(mesh_asset_identity(geom))
            if entry is not None and entry[0] in shell_assets:
                raise ValueError(
                    f"mesh asset {entry[0]!r} needs inertia='shell' but body "
                    f"{part.name!r} has no explicit inertial"
                )


def _articulation_maps(
    compiled_model: ArticulatedObject,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    children_of: dict[str, list[Any]] = {}
    joint_for_child: dict[str, Any] = {}
    for art in compiled_model.articulations:
        children_of.setdefault(art.parent, []).append(art)
        joint_for_child[art.child] = art
    return children_of, joint_for_child


def _root_parts(compiled_model: ArticulatedObject, joint_for_child: dict[str, Any]) -> list[Any]:
    roots = [p for p in compiled_model.parts if p.name not in joint_for_child]
    return roots or list(compiled_model.parts[:1])


def _append_body(
    parent_elem: ET.Element,
    part: Any,
    *,
    parts_by_name: dict[str, Any],
    children_of: dict[str, list[Any]],
    joint_for_child: dict[str, Any],
    mesh_registry: dict[
        tuple[str, tuple[float, float, float]],
        tuple[str, str, tuple[float, float, float]],
    ],
    materials_by_name: dict[str, Any],
    geom_names: set[str],
    effective_ranges: dict[str, tuple[float, float]],
) -> None:
    incoming = joint_for_child.get(part.name)
    attrs = {"name": part.name}
    if incoming is not None:
        pos, quat = _origin_pos_quat(incoming.origin)
        attrs["pos"] = pos
        if quat is not None:
            attrs["quat"] = quat
    body = ET.SubElement(parent_elem, "body", attrs)

    if incoming is not None:
        _append_joint(body, incoming, effective_ranges)

    if getattr(part, "inertial", None) is not None:
        _append_inertial(body, part.inertial)

    for visual in part.visuals:
        _append_geom(
            body,
            visual,
            "visual",
            body_name=part.name,
            geom_names=geom_names,
            mesh_registry=mesh_registry,
            materials_by_name=materials_by_name,
        )
    for collision_index, collision in enumerate(part.collisions):
        _append_geom(
            body,
            collision,
            "collision",
            body_name=part.name,
            geom_names=geom_names,
            mesh_registry=mesh_registry,
            materials_by_name=materials_by_name,
            element_index=collision_index,
        )

    for art in children_of.get(part.name, []):
        child_part = parts_by_name.get(art.child)
        if child_part is not None:
            _append_body(
                body,
                child_part,
                parts_by_name=parts_by_name,
                children_of=children_of,
                joint_for_child=joint_for_child,
                mesh_registry=mesh_registry,
                materials_by_name=materials_by_name,
                geom_names=geom_names,
                effective_ranges=effective_ranges,
            )


def _append_joint(
    body: ET.Element, art: Any, effective_ranges: dict[str, tuple[float, float]]
) -> None:
    jt = (
        art.articulation_type
        if isinstance(art.articulation_type, ArticulationType)
        else ArticulationType(str(art.articulation_type))
    )
    if jt is ArticulationType.FIXED:
        return
    if jt is ArticulationType.FLOATING:
        ET.SubElement(body, "freejoint", {"name": art.name})
        return

    attrs = {
        "name": art.name,
        "type": "slide" if jt is ArticulationType.PRISMATIC else "hinge",
        "axis": _fmt_vec(_normalize_axis(art.axis)),
        "pos": "0 0 0",
    }
    lim = art.motion_limits
    baked = effective_ranges.get(art.name)
    if baked is not None:
        attrs["range"] = f"{_fmt(baked[0])} {_fmt(baked[1])}"
    elif (
        jt in (ArticulationType.REVOLUTE, ArticulationType.PRISMATIC)
        and lim is not None
        and lim.lower is not None
        and lim.upper is not None
    ):
        attrs["range"] = f"{_fmt(lim.lower)} {_fmt(lim.upper)}"
    else:
        attrs["limited"] = "false"
    props = art.motion_properties
    if props is not None:
        if props.damping is not None:
            attrs["damping"] = _fmt(props.damping)
        if props.friction is not None:
            attrs["frictionloss"] = _fmt(props.friction)
        resolution = art.meta.get("_motion_feel_resolution") if isinstance(art.meta, dict) else None
        if isinstance(resolution, dict):
            solref = resolution.get("solreffriction")
            solimp = resolution.get("solimpfriction")
            if isinstance(solref, list) and len(solref) == 2:
                attrs["solreffriction"] = _fmt_vec(solref)
            if isinstance(solimp, list) and len(solimp) == 5:
                attrs["solimpfriction"] = _fmt_vec(solimp)
        if getattr(props, "spring_stiffness", None) is not None:
            attrs["stiffness"] = _fmt(props.spring_stiffness)
        if getattr(props, "spring_ref", None) is not None:
            attrs["springref"] = _fmt(props.spring_ref)
    ET.SubElement(body, "joint", attrs)


def _append_inertial(body: ET.Element, inertial: Any) -> None:
    pos, _quat = _origin_pos_quat(inertial.origin)
    i = inertial.inertia
    fullinertia = " ".join(_fmt(v) for v in (i.ixx, i.iyy, i.izz, i.ixy, i.ixz, i.iyz))
    ET.SubElement(
        body,
        "inertial",
        {"pos": pos, "mass": _fmt(inertial.mass), "fullinertia": fullinertia},
    )


def _unique_name(base: str, used: set[str]) -> str:
    candidate = base
    i = 2
    while candidate in used:
        candidate = f"{base}_{i}"
        i += 1
    used.add(candidate)
    return candidate


def _append_geom(
    body: ET.Element,
    element: Any,
    cls: str,
    *,
    body_name: str,
    geom_names: set[str],
    mesh_registry: dict[
        tuple[str, tuple[float, float, float]],
        tuple[str, str, tuple[float, float, float]],
    ],
    materials_by_name: dict[str, Any],
    element_index: int = 0,
) -> None:
    geom = getattr(element, "geometry", None)
    attrs: dict[str, str] = {"class": cls}
    name = getattr(element, "name", None)
    if cls == "collision":
        attrs["name"] = _unique_name(
            collision_geom_name(body_name, element, element_index), geom_names
        )
    elif name:
        attrs["name"] = _unique_name(f"{cls}_{body_name}_{name}", geom_names)
    pos, quat = _origin_pos_quat(getattr(element, "origin", None))
    if pos != "0 0 0":
        attrs["pos"] = pos
    if quat is not None:
        attrs["quat"] = quat

    if isinstance(geom, Box):
        attrs["type"] = "box"
        sx, sy, sz = (float(v) for v in geom.size)
        attrs["size"] = f"{_fmt(sx / 2)} {_fmt(sy / 2)} {_fmt(sz / 2)}"
    elif isinstance(geom, Cylinder):
        attrs["type"] = "cylinder"
        attrs["size"] = f"{_fmt(geom.radius)} {_fmt(float(geom.length) / 2)}"
    elif isinstance(geom, Sphere):
        attrs["type"] = "sphere"
        attrs["size"] = _fmt(geom.radius)
    elif isinstance(geom, Mesh) and geom.filename and mesh_asset_identity(geom) in mesh_registry:
        attrs["type"] = "mesh"
        attrs["mesh"] = mesh_registry[mesh_asset_identity(geom)][0]
    else:
        return

    if cls == "visual":
        rgba = _material_rgba(getattr(element, "material", None), materials_by_name)
        if rgba is not None:
            attrs["rgba"] = _fmt_vec(rgba)
    ET.SubElement(body, "geom", attrs)


def _append_actuators(
    root: ET.Element,
    compiled_model: ArticulatedObject,
    effective_ranges: dict[str, tuple[float, float]],
    position_servos: bool = True,
) -> None:
    def _behavior(a: Any) -> Any:
        return getattr(a, "behavior", None)

    movable = [
        a
        for a in compiled_model.articulations
        if _coerce(a.articulation_type) not in (ArticulationType.FIXED, ArticulationType.FLOATING)
        and getattr(a, "mimic", None) is None
        and _behavior(a) is None
    ]
    if not position_servos:
        movable = []
    momentary = [
        a
        for a in compiled_model.articulations
        if _behavior(a) is not None and getattr(_behavior(a), "kind", "") == "momentary"
    ]
    if not movable and not momentary:
        return
    actuator = ET.SubElement(root, "actuator")
    for art in movable:
        attrs = {"name": f"act_{art.name}", "joint": art.name, "kp": "10"}
        baked = effective_ranges.get(art.name)
        lim = art.motion_limits
        if baked is not None:
            attrs["ctrlrange"] = f"{_fmt(baked[0])} {_fmt(baked[1])}"
        elif lim is not None and lim.lower is not None and lim.upper is not None:
            attrs["ctrlrange"] = f"{_fmt(lim.lower)} {_fmt(lim.upper)}"
        ET.SubElement(actuator, "position", attrs)
    for art in momentary:
        b = _behavior(art)
        speed = float(getattr(b, "speed", 1.0) or 1.0)
        ET.SubElement(
            actuator,
            "velocity",
            {
                "name": f"act_{art.name}",
                "joint": art.name,
                "kv": "1",
                "ctrlrange": f"{_fmt(-speed)} {_fmt(speed)}",
            },
        )


def _append_equality(root: ET.Element, compiled_model: ArticulatedObject) -> None:
    mimics = [a for a in compiled_model.articulations if getattr(a, "mimic", None) is not None]
    if not mimics:
        return
    equality = ET.SubElement(root, "equality")
    for art in mimics:
        m = art.mimic
        coeffs = list(m.poly) if m.poly else [m.offset, m.multiplier]
        coeffs = (coeffs + [0.0, 0.0, 0.0, 0.0, 0.0])[:5]
        ET.SubElement(
            equality,
            "joint",
            {
                "joint1": art.name,
                "joint2": m.joint,
                "polycoef": " ".join(_fmt(c) for c in coeffs),
            },
        )


def _contact_exclude_pairs(
    compiled_model: ArticulatedObject,
    *,
    asset_root: str | os.PathLike[str] | Path | None,
    declared_overlap_pairs: Iterable[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    return _contact_exempt_pairs_impl(
        compiled_model,
        asset_root=asset_root,
        declared_overlap_pairs=declared_overlap_pairs,
    )


def _append_contact_excludes(root: ET.Element, pairs: list[tuple[str, str]]) -> None:
    if not pairs:
        return
    contact = ET.SubElement(root, "contact")
    for a, b in pairs:
        ET.SubElement(contact, "exclude", {"body1": a, "body2": b})


def _coerce(value: Any) -> ArticulationType:
    return value if isinstance(value, ArticulationType) else ArticulationType(str(value))


def _resolve_material(ref: Any, materials_by_name: dict[str, Any]) -> Optional[Material]:
    if ref is None:
        return None
    if isinstance(ref, str):
        return materials_by_name.get(ref)
    return ref


def _material_rgba(ref: Any, materials_by_name: dict[str, Any]) -> Optional[tuple[float, ...]]:
    mat = _resolve_material(ref, materials_by_name)
    rgba = getattr(mat, "rgba", None) if mat is not None else None
    if rgba is None:
        return None
    vals = tuple(float(v) for v in rgba)
    if len(vals) == 3:
        vals = vals + (1.0,)
    return vals


def _normalize_axis(axis: Any) -> tuple[float, float, float]:
    x, y, z = (float(axis[0]), float(axis[1]), float(axis[2]))
    mag = math.sqrt(x * x + y * y + z * z)
    if mag <= 1e-12:
        return (x, y, z)
    return (x / mag, y / mag, z / mag)


def _origin_pos_quat(origin: Any) -> tuple[str, Optional[str]]:
    if origin is None:
        return "0 0 0", None
    pos = _fmt_vec(tuple(float(v) for v in origin.xyz))
    rpy = tuple(float(v) for v in origin.rpy)
    if rpy == (0.0, 0.0, 0.0):
        return pos, None
    return pos, _fmt_vec(_quat_from_rpy(rpy))


def _quat_from_rpy(rpy: tuple[float, float, float]) -> tuple[float, float, float, float]:
    r = _rpy_matrix(rpy[0], rpy[1], rpy[2])
    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2][1] - r[1][2]) / s
        y = (r[0][2] - r[2][0]) / s
        z = (r[1][0] - r[0][1]) / s
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0
        w = (r[2][1] - r[1][2]) / s
        x = 0.25 * s
        y = (r[0][1] + r[1][0]) / s
        z = (r[0][2] + r[2][0]) / s
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0
        w = (r[0][2] - r[2][0]) / s
        x = (r[0][1] + r[1][0]) / s
        y = 0.25 * s
        z = (r[1][2] + r[2][1]) / s
    else:
        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0
        w = (r[1][0] - r[0][1]) / s
        x = (r[0][2] + r[2][0]) / s
        y = (r[1][2] + r[2][1]) / s
        z = 0.25 * s
    return (w, x, y, z)


def _fmt(value: float) -> str:
    return f"{float(value):.6g}"


def _fmt_vec(values: Any) -> str:
    return " ".join(_fmt(v) for v in values)


def _indent(elem: ET.Element, level: int = 0) -> None:
    pad = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad
