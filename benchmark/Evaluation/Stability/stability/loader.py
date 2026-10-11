from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

from .policy import StabilityPolicy

_GROUND_NAME = "bench_ground"
_MARKER_NAME = "asset_canonical_marker"
_FREE_JOINT_NAME = "asset_freejoint"


@dataclass
class LoadResult:
    urdf_path: Path
    policy: StabilityPolicy
    model: mujoco.MjModel
    asset_geom_ids: tuple[int, ...]
    ground_geom_id: int
    free_joint_id: int
    marker_site_id: int
    parent_spec: mujoco.MjSpec


@dataclass(frozen=True)
class JointEquality:
    joint_name: str
    target: float
    active: bool = True


def _assign_options(spec: mujoco.MjSpec, policy: StabilityPolicy) -> None:
    option = spec.option
    option.timestep = policy.timestep_s
    option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    option.iterations = policy.iterations
    option.tolerance = policy.tolerance
    option.gravity = list(policy.gravity)
    option.disableflags = int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
    option.enableflags = 0


def _build_scene(
    child: mujoco.MjSpec, policy: StabilityPolicy
) -> tuple[mujoco.MjSpec, mujoco.MjsSite]:
    parent = mujoco.MjSpec()
    _assign_options(parent, policy)
    for index, geom in enumerate(child.geoms):
        if not geom.name:
            geom.name = f"geom_{index}"
        _assign_geom_contact(
            geom, friction=policy.asset_geom_friction, policy=policy, contype=0, conaffinity=1
        )
    ground = parent.worldbody.add_geom(
        name=_GROUND_NAME, type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.0, 0.0, 0.05]
    )
    _assign_geom_contact(
        ground, friction=policy.ground_friction, policy=policy, contype=1, conaffinity=0
    )
    frame = parent.worldbody.add_frame()
    root = frame.attach_body(child.worldbody, "asset_", "")
    root.add_freejoint(name=_FREE_JOINT_NAME)
    marker = root.add_site(name=_MARKER_NAME, pos=[0.0, 0.0, 0.0], size=[0.002])
    return (parent, marker)


def _read_urdf(path: Path) -> tuple[mujoco.MjSpec, bool]:
    source = ET.parse(path).getroot()
    if source.tag.rsplit("}", 1)[-1] != "robot":
        raise ValueError("expected a URDF robot")
    if any(
        (
            Path(element.get("filename", "")).suffix.lower() == ".dae"
            for element in source.iter()
            if element.tag.rsplit("}", 1)[-1] == "mesh"
        )
    ):
        raise ValueError("unsupported mesh format: .dae")
    has_collision = any(
        (element.tag.rsplit("}", 1)[-1] == "collision" for element in source.iter())
    )
    try:
        child = mujoco.MjSpec.from_file(str(path))
    except ValueError:
        normalized = _deduplicated_urdf(path)
        if normalized is None:
            raise
        child = mujoco.MjSpec.from_string(normalized[0])
    use_visual = not len(child.geoms) and (not has_collision)
    if use_visual:
        child = mujoco.MjSpec.from_string(_visual_urdf(path))
    return (child, use_visual)


def _scene_result(
    path: Path, policy: StabilityPolicy, spec: mujoco.MjSpec, model: mujoco.MjModel
) -> LoadResult:
    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, _GROUND_NAME)
    free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, _FREE_JOINT_NAME)
    marker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, _MARKER_NAME)
    if min(ground_id, free_id, marker_id) < 0:
        raise RuntimeError("compiled scene is missing required elements")
    return LoadResult(
        urdf_path=path,
        policy=policy,
        model=model,
        asset_geom_ids=tuple((index for index in range(model.ngeom) if index != ground_id)),
        ground_geom_id=ground_id,
        free_joint_id=free_id,
        marker_site_id=marker_id,
        parent_spec=spec,
    )


def compile_equality_episode(
    loaded: LoadResult, equalities: Sequence[JointEquality], *, timestep_s: float
) -> LoadResult:
    spec = loaded.parent_spec.copy()
    spec.option.timestep = timestep_s
    preliminary = spec.compile()
    names: dict[str, str] = {}
    for index, request in enumerate(equalities):
        joint_id = mujoco.mj_name2id(preliminary, mujoco.mjtObj.mjOBJ_JOINT, request.joint_name)
        if joint_id < 0:
            raise ValueError(f"unknown equality joint: {request.joint_name}")
        address = int(preliminary.jnt_qposadr[joint_id])
        name = f"bench_joint_eq_{index}"
        spec.add_equality(
            name=name,
            type=mujoco.mjtEq.mjEQ_JOINT,
            objtype=mujoco.mjtObj.mjOBJ_JOINT,
            name1=request.joint_name,
            data=[request.target - float(preliminary.qpos0[address]), *[0.0] * 10],
            active=request.active,
            solref=list(loaded.policy.eq_solref),
            solimp=list(loaded.policy.eq_solimp),
        )
        names[request.joint_name] = name
    model = spec.compile()
    episode = _scene_result(loaded.urdf_path, loaded.policy, spec, model)
    if any(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name) < 0 for name in names.values()
    ):
        raise RuntimeError("compiled scene is missing a requested equality")
    return episode


def _assign_compiler(spec: mujoco.MjSpec, flags: dict[str, bool | float | str]) -> None:
    compiler = spec.compiler
    compiler.balanceinertia = bool(flags["balanceinertia"])
    compiler.boundmass = float(flags["boundmass"])
    compiler.boundinertia = float(flags["boundinertia"])
    compiler.inertiafromgeom = (
        mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_TRUE
        if flags["inertiafromgeom"] == "true"
        else mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_AUTO
    )
    compiler.settotalmass = float(flags["settotalmass"])


def _assign_geom_contact(
    geom: mujoco.MjsGeom,
    *,
    friction: tuple[float, float, float],
    policy: StabilityPolicy,
    contype: int,
    conaffinity: int,
) -> None:
    geom.friction = list(friction)
    geom.solref = list(policy.contact_solref)
    geom.solimp = list(policy.contact_solimp)
    geom.condim = policy.condim
    geom.margin = policy.contact_margin
    geom.gap = policy.contact_gap
    geom.priority = policy.geom_priority
    geom.solmix = policy.geom_solmix
    geom.contype = contype
    geom.conaffinity = conaffinity


def _xml_element_identity(element: ET.Element) -> tuple[object, ...]:
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        (element.text or "").strip(),
        tuple((_xml_element_identity(child) for child in element)),
    )


def _drop_exact_duplicate_joints(root: ET.Element) -> int | None:
    seen: dict[str, tuple[object, ...]] = {}
    removed = 0
    for joint in list(root.findall("joint")):
        name = joint.get("name")
        if name is None:
            continue
        identity = _xml_element_identity(joint)
        previous = seen.get(name)
        if previous is None:
            seen[name] = identity
        elif previous == identity:
            root.remove(joint)
            removed += 1
        else:
            return None
    return removed


def _deduplicated_urdf(path: Path) -> tuple[str, int] | None:
    root = ET.parse(path).getroot()
    removed = _drop_exact_duplicate_joints(root)
    if removed is None:
        return None
    if not removed:
        return None
    block = ET.Element("mujoco")
    ET.SubElement(
        block,
        "compiler",
        {"discardvisual": "true", "meshdir": str(path.parent), "strippath": "false"},
    )
    root.insert(0, block)
    return (ET.tostring(root, encoding="unicode"), removed)


def _geom_world_span(
    model: mujoco.MjModel, data: mujoco.MjData, geom_id: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    origin = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        vertices = np.asarray(model.mesh_vert[start : start + count], dtype=np.float64)
        world = vertices.reshape(-1, 3) @ rotation.T + origin
        return (world.min(axis=0), world.max(axis=0))
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        extent = np.full(3, float(size[0]))
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        extent = np.abs(rotation[:, 2]) * float(size[1]) + float(size[0])
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        radial = np.linalg.norm(rotation[:, :2], axis=1)
        extent = np.abs(rotation[:, 2]) * float(size[1]) + radial * float(size[0])
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        extent = np.linalg.norm(rotation * size[np.newaxis, :], axis=1)
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        extent = np.abs(rotation) @ size
    else:
        local_center = np.asarray(model.geom_aabb[geom_id, :3], dtype=np.float64)
        local_halfsize = np.asarray(model.geom_aabb[geom_id, 3:], dtype=np.float64)
        center = origin + rotation @ local_center
        halfsize = np.abs(rotation) @ local_halfsize
        return (center - halfsize, center + halfsize)
    return (origin - extent, origin + extent)


def _geom_world_bounds(
    model: mujoco.MjModel, data: mujoco.MjData, geom_ids: tuple[int, ...]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if not geom_ids:
        raise ValueError("asset has no collision geometry")
    spans = [_geom_world_span(model, data, geom_id) for geom_id in geom_ids]
    return (
        np.min([span[0] for span in spans], axis=0),
        np.max([span[1] for span in spans], axis=0),
    )


def collision_aabb(
    loaded: LoadResult, data: mujoco.MjData
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if loaded.model is None:
        raise ValueError("cannot compute an AABB for an unloaded asset")
    return _geom_world_bounds(loaded.model, data, loaded.asset_geom_ids)


def _assign_closed_qpos(
    model: mujoco.MjModel, data: mujoco.MjData, policy: StabilityPolicy
) -> None:
    data.qpos[:] = model.qpos0
    scalar_types = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) not in scalar_types:
            continue
        target = 0.0
        if bool(model.jnt_limited[joint_id]):
            lower, upper = (float(value) for value in model.jnt_range[joint_id])
            if not (math.isfinite(lower) and math.isfinite(upper) and (lower < upper)):
                continue
            target = min(max(0.0, lower), upper)
            inset = policy.limit_inset_fraction * (upper - lower)
            if math.isclose(target, lower, rel_tol=0.0, abs_tol=1e-12):
                target = lower + inset
            elif math.isclose(target, upper, rel_tol=0.0, abs_tol=1e-12):
                target = upper - inset
        data.qpos[int(model.jnt_qposadr[joint_id])] = target


def _set_marker_from_preliminary_model(
    model: mujoco.MjModel, marker: mujoco.MjsSite, policy: StabilityPolicy
) -> None:
    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, _GROUND_NAME)
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "asset_world")
    asset_geom_ids = tuple((index for index in range(model.ngeom) if index != ground_id))
    data = mujoco.MjData(model)
    _assign_closed_qpos(model, data, policy)
    mujoco.mj_forward(model, data)
    lower, upper = _geom_world_bounds(model, data, asset_geom_ids)
    world_center = (lower + upper) * 0.5
    root_rotation = np.asarray(data.xmat[root_id], dtype=np.float64).reshape(3, 3)
    root_position = np.asarray(data.xpos[root_id], dtype=np.float64)
    marker.pos = (root_rotation.T @ (world_center - root_position)).tolist()


def set_joint_configuration(
    loaded: LoadResult, data: mujoco.MjData, targets: Mapping[str, float]
) -> None:
    if loaded.model is None:
        raise ValueError("cannot configure an unloaded asset")
    data.qpos[:] = loaded.model.qpos0
    for joint_name, target in targets.items():
        joint_id = mujoco.mj_name2id(loaded.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"unknown configuration joint: {joint_name}")
        joint_type = int(loaded.model.jnt_type[joint_id])
        if joint_type not in {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}:
            raise ValueError(f"configuration joint must be scalar: {joint_name}")
        data.qpos[int(loaded.model.jnt_qposadr[joint_id])] = target


def place_on_ground(loaded: LoadResult, data: mujoco.MjData) -> float:
    if loaded.model is None or loaded.free_joint_id is None:
        raise ValueError("cannot place an unloaded asset")
    mujoco.mj_forward(loaded.model, data)
    lower, _upper = collision_aabb(loaded, data)
    delta_z = loaded.policy.placement_clearance_m - float(lower[2])
    qpos_address = int(loaded.model.jnt_qposadr[loaded.free_joint_id])
    data.qpos[qpos_address + 2] += delta_z
    mujoco.mj_forward(loaded.model, data)
    return delta_z


def ground_contact_count(loaded: LoadResult, data: mujoco.MjData) -> int:
    if loaded.ground_geom_id is None:
        return 0
    ground_id = loaded.ground_geom_id
    asset_ids = set(loaded.asset_geom_ids)
    return sum(
        (
            1
            for contact in data.contact
            if int(contact.geom1) == ground_id
            and int(contact.geom2) in asset_ids
            or (int(contact.geom2) == ground_id and int(contact.geom1) in asset_ids)
        )
    )


def max_ground_penetration(loaded: LoadResult, data: mujoco.MjData) -> float:
    if loaded.ground_geom_id is None:
        return 0.0
    ground_id = loaded.ground_geom_id
    asset_ids = set(loaded.asset_geom_ids)
    penetrations = [
        max(0.0, -float(contact.dist))
        for contact in data.contact
        if int(contact.geom1) == ground_id
        and int(contact.geom2) in asset_ids
        or (int(contact.geom2) == ground_id and int(contact.geom1) in asset_ids)
    ]
    return max(penetrations, default=0.0)


def _visual_urdf(path: Path) -> str:
    root = ET.parse(path).getroot()
    if _drop_exact_duplicate_joints(root) is None:
        raise ValueError("conflicting duplicate joint definitions")
    block = ET.Element("mujoco")
    ET.SubElement(
        block,
        "compiler",
        {"discardvisual": "false", "meshdir": str(path.parent), "strippath": "false"},
    )
    root.insert(0, block)
    return ET.tostring(root, encoding="unicode")


def load_urdf(path: Path, policy: StabilityPolicy | None = None) -> LoadResult:
    policy = policy or StabilityPolicy()
    path = path.absolute()
    if path.suffix.lower() != ".urdf":
        raise ValueError("expected a .urdf asset")
    child, _ = _read_urdf(path)
    parent, marker = _build_scene(child, policy)
    flags = {
        "balanceinertia": False,
        "boundmass": 0.0,
        "boundinertia": 0.0,
        "inertiafromgeom": "auto",
        "settotalmass": -1.0,
    }
    _assign_compiler(child, flags)
    _assign_compiler(parent, flags)
    preliminary = parent.compile()
    _set_marker_from_preliminary_model(preliminary, marker, policy)
    return _scene_result(path, policy, parent, parent.compile())
