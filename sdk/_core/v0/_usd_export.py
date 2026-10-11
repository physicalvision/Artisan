
from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from ._contact_exemptions import contact_exempt_pairs
from ._coupling import driver_effective_ranges
from .articulated_object import ArticulatedObject
from .exact_collisions import compile_object_model_with_exact_collisions
from .geometry_qc import _rpy_matrix
from .inertia import geometry_to_trimesh
from .types import ArticulationType, Box, Cylinder, Mesh, Sphere

logger = logging.getLogger(__name__)

_PHYSX_MIMIC_INSTANCE = "rotZ"


def compile_object_to_usd(
    object_model: ArticulatedObject,
    out_path: str | os.PathLike[str] | Path,
    *,
    asset_root: str | os.PathLike[str] | Path | None = None,
    compiled_model: ArticulatedObject | None = None,
    declared_overlap_pairs: Iterable[tuple[str, str]] | None = None,
    validate: bool = True,
) -> Path:
    from sdk._dependencies import require_usd

    require_usd(feature="USD export")
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    if compiled_model is None:
        if validate:
            object_model.validate(strict=True)
        compiled_model = compile_object_model_with_exact_collisions(
            object_model, asset_root=asset_root, validate=False
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    root_asset = _resolve_asset_root(compiled_model, asset_root)
    materials_by_name = {
        getattr(m, "name", ""): m for m in getattr(compiled_model, "materials", []) or []
    }

    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    robot_name = _sanitize(object_model.name) or "Robot"
    robot_path = f"/{robot_name}"
    robot = UsdGeom.Xform.Define(stage, robot_path)
    stage.SetDefaultPrim(robot.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(robot.GetPrim())

    parts_by_name = {p.name: p for p in compiled_model.parts}
    children_of: dict[str, list[Any]] = {}
    joint_for_child: dict[str, Any] = {}
    for art in compiled_model.articulations:
        children_of.setdefault(art.parent, []).append(art)
        joint_for_child[art.child] = art
    roots = [p for p in compiled_model.parts if p.name not in joint_for_child] or list(
        compiled_model.parts[:1]
    )

    ctx = _Ctx(
        stage=stage,
        gf=Gf,
        usdgeom=UsdGeom,
        usdphysics=UsdPhysics,
        asset_root=root_asset,
        materials_by_name=materials_by_name,
        body_paths={},
        effective_ranges=driver_effective_ranges(compiled_model),
    )
    for part in roots:
        _author_body(ctx, part, robot_path, children_of, parts_by_name, joint_for_child)

    for art in compiled_model.articulations:
        _author_joint(ctx, art)

    _author_mimic_joints(ctx, compiled_model)

    _author_contact_filters(
        ctx,
        compiled_model,
        contact_exempt_pairs(
            compiled_model,
            asset_root=root_asset,
            declared_overlap_pairs=declared_overlap_pairs,
        ),
    )

    stage.GetRootLayer().Save()
    return out_path


def _author_contact_filters(
    ctx: _Ctx, compiled_model: ArticulatedObject, pairs: list[tuple[str, str]]
) -> int:
    UsdPhysics = ctx.usdphysics
    count = 0
    for a, b in pairs:
        pa = ctx.body_paths.get(a)
        pb = ctx.body_paths.get(b)
        if pa is None or pb is None:
            continue
        prim = ctx.stage.GetPrimAtPath(pa)
        if not prim or not prim.IsValid():
            continue
        api = UsdPhysics.FilteredPairsAPI.Apply(prim)
        api.CreateFilteredPairsRel().AddTarget(pb)
        count += 1
    return count


class _Ctx:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


def _author_body(
    ctx: _Ctx,
    part: Any,
    parent_path: str,
    children_of: dict[str, list[Any]],
    parts_by_name: dict[str, Any],
    joint_for_child: dict[str, Any],
) -> None:
    UsdGeom = ctx.usdgeom
    UsdPhysics = ctx.usdphysics
    Gf = ctx.gf

    body_path = f"{parent_path}/{_sanitize(part.name)}"
    ctx.body_paths[part.name] = body_path
    xform = UsdGeom.Xform.Define(ctx.stage, body_path)
    prim = xform.GetPrim()

    incoming = joint_for_child.get(part.name)
    if incoming is not None:
        tx, ty, tz = (float(v) for v in incoming.origin.xyz)
        xform.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))
        if tuple(float(v) for v in incoming.origin.rpy) != (0.0, 0.0, 0.0):
            w, x, y, z = _mat3_to_quat(_rpy_matrix(*incoming.origin.rpy))
            xform.AddOrientOp().Set(Gf.Quatf(w, x, y, z))

    UsdPhysics.RigidBodyAPI.Apply(prim)
    _author_mass(ctx, prim, getattr(part, "inertial", None))

    for idx, visual in enumerate(part.visuals):
        _author_geom(ctx, body_path, f"vis_{idx}", visual, collision=False)
    for idx, collision in enumerate(part.collisions):
        _author_geom(ctx, body_path, f"col_{idx}", collision, collision=True)

    for art in children_of.get(part.name, []):
        child = parts_by_name.get(art.child)
        if child is not None:
            _author_body(ctx, child, body_path, children_of, parts_by_name, joint_for_child)


def _author_mass(ctx: _Ctx, prim: Any, inertial: Any) -> None:
    if inertial is None:
        return
    UsdPhysics = ctx.usdphysics
    Gf = ctx.gf
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(float(inertial.mass))
    com = tuple(float(v) for v in inertial.origin.xyz)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*com))
    moments, axes_quat = _principal_inertia(inertial.inertia)
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*moments))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(*axes_quat))


def _author_geom(ctx: _Ctx, body_path: str, name: str, element: Any, *, collision: bool) -> None:
    UsdGeom = ctx.usdgeom
    UsdPhysics = ctx.usdphysics
    Gf = ctx.gf
    geom = getattr(element, "geometry", None)
    path = f"{body_path}/{name}"

    gprim = None
    extra_scale: Optional[tuple[float, float, float]] = None
    if isinstance(geom, Box):
        gprim = UsdGeom.Cube.Define(ctx.stage, path)
        gprim.CreateSizeAttr(1.0)
        extra_scale = tuple(float(v) for v in geom.size)
    elif isinstance(geom, Cylinder):
        gprim = UsdGeom.Cylinder.Define(ctx.stage, path)
        gprim.CreateRadiusAttr(float(geom.radius))
        gprim.CreateHeightAttr(float(geom.length))
        gprim.CreateAxisAttr(UsdGeom.Tokens.z)
    elif isinstance(geom, Sphere):
        gprim = UsdGeom.Sphere.Define(ctx.stage, path)
        gprim.CreateRadiusAttr(float(geom.radius))
    elif isinstance(geom, Mesh):
        gprim = _author_mesh_geom(ctx, path, geom)
    if gprim is None:
        return

    ox, oy, oz = (float(v) for v in element.origin.xyz)
    if (ox, oy, oz) != (0.0, 0.0, 0.0):
        gprim.AddTranslateOp().Set(Gf.Vec3d(ox, oy, oz))
    if tuple(float(v) for v in element.origin.rpy) != (0.0, 0.0, 0.0):
        w, x, y, z = _mat3_to_quat(_rpy_matrix(*element.origin.rpy))
        gprim.AddOrientOp().Set(Gf.Quatf(w, x, y, z))
    if extra_scale is not None:
        gprim.AddScaleOp().Set(Gf.Vec3f(*extra_scale))

    prim = gprim.GetPrim()
    if collision:
        UsdPhysics.CollisionAPI.Apply(prim)
        if isinstance(geom, Mesh):
            mca = UsdPhysics.MeshCollisionAPI.Apply(prim)
            approx = (
                UsdPhysics.Tokens.convexHull
                if "/collision/" in str(geom.filename or "")
                else UsdPhysics.Tokens.convexDecomposition
            )
            mca.CreateApproximationAttr(approx)
    else:
        UsdGeom.Gprim(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)
        rgba = _material_rgba(getattr(element, "material", None), ctx.materials_by_name)
        if rgba is not None:
            UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*rgba[:3])])


def _author_mesh_geom(ctx: _Ctx, path: str, geom: Any) -> Any:
    UsdGeom = ctx.usdgeom
    Gf = ctx.gf
    tm = geometry_to_trimesh(geom, asset_root=ctx.asset_root)
    if tm is None or len(getattr(tm, "faces", [])) == 0:
        return None
    mesh = UsdGeom.Mesh.Define(ctx.stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*[float(c) for c in v]) for v in tm.vertices])
    faces = tm.faces
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    indices: list[int] = []
    for f in faces:
        indices.extend(int(i) for i in f)
    mesh.CreateFaceVertexIndicesAttr(indices)
    return mesh


def _author_joint(ctx: _Ctx, art: Any) -> None:
    UsdPhysics = ctx.usdphysics
    Gf = ctx.gf
    jt = art.articulation_type
    jt = jt if isinstance(jt, ArticulationType) else ArticulationType(str(jt))
    parent_path = ctx.body_paths.get(art.parent)
    child_path = ctx.body_paths.get(art.child)
    if parent_path is None or child_path is None:
        return
    joint_path = f"{child_path}/joint_{_sanitize(art.name)}"

    if jt is ArticulationType.FIXED:
        joint = UsdPhysics.FixedJoint.Define(ctx.stage, joint_path)
        _set_joint_bodies(joint, parent_path, child_path, art, Gf, align=None)
        return
    if jt is ArticulationType.FLOATING:
        return

    is_prismatic = jt is ArticulationType.PRISMATIC
    joint = (
        UsdPhysics.PrismaticJoint.Define(ctx.stage, joint_path)
        if is_prismatic
        else UsdPhysics.RevoluteJoint.Define(ctx.stage, joint_path)
    )
    align = _quat_align_z_to(tuple(float(v) for v in art.axis))
    joint.CreateAxisAttr(ctx.usdgeom.Tokens.z)
    _set_joint_bodies(joint, parent_path, child_path, art, Gf, align=align)

    eff = getattr(ctx, "effective_ranges", {}).get(art.name)
    lim = art.motion_limits
    bounds: tuple[float, float] | None = None
    if eff is not None:
        bounds = (float(eff[0]), float(eff[1]))
    elif lim is not None and lim.lower is not None and lim.upper is not None:
        bounds = (float(lim.lower), float(lim.upper))
    if bounds is not None:
        lo, hi = bounds
        if is_prismatic:
            joint.CreateLowerLimitAttr(lo)
            joint.CreateUpperLimitAttr(hi)
        else:
            joint.CreateLowerLimitAttr(math.degrees(lo))
            joint.CreateUpperLimitAttr(math.degrees(hi))

    behavior = getattr(art, "behavior", None)
    if behavior is not None:
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear" if is_prismatic else "angular")
        drive.CreateTypeAttr("force")
        _author_behavior_drive(drive, art, is_prismatic)
        _author_behavior_attrs(joint.GetPrim(), behavior)
        return
    if getattr(art, "mimic", None) is None:
        _author_passive_motion_feel(joint.GetPrim(), art, is_prismatic, UsdPhysics)


_ANGULAR_TYPES = {ArticulationType.REVOLUTE, ArticulationType.CONTINUOUS}


def _coerce_atype(art: Any) -> ArticulationType:
    t = art.articulation_type
    return t if isinstance(t, ArticulationType) else ArticulationType(str(t))


def _isaac45_joint_friction(effort: float) -> float:
    value = float(effort)
    if not math.isfinite(value):
        raise ValueError(f"MotionFeel effort must be finite, got {value!r}")
    return min(1.0, max(0.0, value))


def _apply_physx_joint_friction(prim: Any, coefficient: float) -> None:
    physx_schema = None
    try:
        from pxr import PhysxSchema as physx_schema  # type: ignore  # noqa: PLC0415
    except ImportError:
        physx_schema = None
    if physx_schema is not None:
        api = physx_schema.PhysxJointAPI.Apply(prim)
        api.CreateJointFrictionAttr(float(coefficient))
        return

    from pxr import Sdf  # noqa: PLC0415

    prim.CreateAttribute("physxJoint:jointFriction", Sdf.ValueTypeNames.Float).Set(
        float(coefficient)
    )
    try:
        prim.AddAppliedSchema("PhysxJointAPI")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "USD dynamics: could not apply PhysxJointAPI on %s (%s); joint friction "
            "may be ignored by Isaac/PhysX",
            prim.GetPath(),
            exc,
        )


def _author_passive_motion_feel(prim: Any, art: Any, is_prismatic: bool, UsdPhysics: Any) -> None:
    from pxr import Sdf  # noqa: PLC0415

    feel = getattr(art, "motion_feel", None)
    if feel is None:
        return
    coefficient = _isaac45_joint_friction(float(feel.effort))
    _apply_physx_joint_friction(prim, coefficient)

    namespace = "artisan:motionFeel:"
    prim.CreateAttribute(namespace + "profile", Sdf.ValueTypeNames.Token).Set("isaac-4.5-passive")
    prim.CreateAttribute(namespace + "frictionMapping", Sdf.ValueTypeNames.Token).Set(
        "dimensionless-best-effort"
    )
    if str(feel.hold) == "holds_anywhere":
        prim.CreateAttribute(namespace + "hold", Sdf.ValueTypeNames.Token).Set("holds_anywhere")
        prim.CreateAttribute(namespace + "holdSupport", Sdf.ValueTypeNames.Token).Set(
            "unsupported-passive-isaac45"
        )

    properties = getattr(art, "motion_properties", None)
    damping = float(getattr(properties, "damping", None) or 0.0)
    if damping <= 0.0:
        return
    if not is_prismatic:
        damping *= math.pi / 180.0
    drive = UsdPhysics.DriveAPI.Apply(prim, "linear" if is_prismatic else "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(damping)
    drive.CreateTargetVelocityAttr(0.0)


def _author_mimic_joints(ctx: _Ctx, compiled_model: ArticulatedObject) -> int:
    by_name = {a.name: a for a in compiled_model.articulations}
    deg = math.degrees(1.0)
    declared = sum(1 for a in compiled_model.articulations if getattr(a, "mimic", None) is not None)
    count = 0
    for art in compiled_model.articulations:
        mimic = getattr(art, "mimic", None)
        if mimic is None:
            continue
        source = by_name.get(str(getattr(mimic, "joint", "")).strip())
        if source is None:
            logger.warning(
                "USD mimic: %r references missing source %r — skipped", art.name, mimic.joint
            )
            continue
        follower_body = ctx.body_paths.get(art.child)
        driver_body = ctx.body_paths.get(source.child)
        if follower_body is None or driver_body is None:
            logger.warning("USD mimic: %r missing body path(s) — coupling dropped", art.name)
            continue
        follower_joint = f"{follower_body}/joint_{_sanitize(art.name)}"
        driver_joint = f"{driver_body}/joint_{_sanitize(source.name)}"
        prim = ctx.stage.GetPrimAtPath(follower_joint)
        if not prim or not prim.IsValid():
            logger.warning(
                "USD mimic: follower joint prim %s not found — coupling dropped", follower_joint
            )
            continue
        k_follower = deg if _coerce_atype(art) in _ANGULAR_TYPES else 1.0
        k_driver = deg if _coerce_atype(source) in _ANGULAR_TYPES else 1.0
        gearing = -float(mimic.multiplier) * k_follower / k_driver
        offset_usd = -float(mimic.offset) * k_follower
        _apply_physx_mimic(prim, _PHYSX_MIMIC_INSTANCE, driver_joint, gearing, offset_usd)
        count += 1
    if count != declared:
        logger.warning("USD mimic: authored %d of %d declared couplings", count, declared)
    return count


def _author_behavior_drive(drive: Any, art: Any, is_prismatic: bool) -> None:
    behavior = art.behavior
    k = 1.0 if is_prismatic else math.degrees(1.0)
    if getattr(behavior, "kind", "") == "momentary":
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(100.0)
        vel = float(getattr(behavior, "direction", 1.0) or 1.0) * float(
            getattr(behavior, "speed", 1.0) or 1.0
        )
        drive.CreateTargetVelocityAttr(vel * k)
        return
    mp = getattr(art, "motion_properties", None)
    stiffness = float(getattr(mp, "spring_stiffness", None) or 0.0)
    target = getattr(behavior, "target", None)
    drive.CreateStiffnessAttr(stiffness)
    drive.CreateDampingAttr(max(1.0, 0.1 * stiffness))
    if isinstance(target, (int, float)):
        drive.CreateTargetPositionAttr(float(target) * k)


def _author_behavior_attrs(prim: Any, behavior: Any) -> None:
    from pxr import Sdf  # noqa: PLC0415

    ns = "physics:behavior:"
    prim.CreateAttribute(ns + "kind", Sdf.ValueTypeNames.Token).Set(str(behavior.kind))
    prim.CreateAttribute(ns + "driver", Sdf.ValueTypeNames.String).Set(str(behavior.driver))
    for fld in ("target", "threshold", "direction", "speed"):
        v = getattr(behavior, fld, None)
        if isinstance(v, (int, float)):
            prim.CreateAttribute(ns + fld, Sdf.ValueTypeNames.Float).Set(float(v))
    prim.CreateAttribute(ns + "strength", Sdf.ValueTypeNames.Token).Set(str(behavior.strength))


def _apply_physx_mimic(
    prim: Any, instance: str, driver_joint_path: str, gearing: float, offset_usd: float
) -> None:
    physx_schema = None
    try:
        from pxr import PhysxSchema as physx_schema  # type: ignore  # noqa: PLC0415
    except ImportError:
        physx_schema = None
    if physx_schema is not None:
        api = physx_schema.PhysxMimicJointAPI.Apply(prim, instance)
        api.CreateGearingAttr(float(gearing))
        api.CreateOffsetAttr(float(offset_usd))
        api.CreateReferenceJointRel().SetTargets([driver_joint_path])
        return

    from pxr import Sdf  # noqa: PLC0415

    base = f"physxMimicJoint:{instance}:"
    prim.CreateRelationship(base + "referenceJoint").SetTargets([driver_joint_path])
    prim.CreateAttribute(base + "gearing", Sdf.ValueTypeNames.Float).Set(float(gearing))
    prim.CreateAttribute(base + "offset", Sdf.ValueTypeNames.Float).Set(float(offset_usd))
    try:
        prim.AddAppliedSchema(f"PhysxMimicJointAPI:{instance}")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "USD mimic: could not apply PhysxMimicJointAPI:%s on %s (%s); "
            "the coupling may be ignored by Isaac/PhysX",
            instance,
            prim.GetPath(),
            exc,
        )


def _set_joint_bodies(
    joint: Any, parent_path: str, child_path: str, art: Any, Gf, *, align
) -> None:
    from pxr import Sdf

    joint.CreateBody0Rel().SetTargets([Sdf.Path(parent_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(child_path)])
    px, py, pz = (float(v) for v in art.origin.xyz)
    origin_R = _rpy_matrix(*[float(v) for v in art.origin.rpy])
    if align is None:
        rot0 = _mat3_to_quat(origin_R)
        rot1 = (1.0, 0.0, 0.0, 0.0)
    else:
        align_R = _quat_to_mat3(align)
        rot0 = _mat3_to_quat(_mat3_mul(origin_R, align_R))
        rot1 = align
    joint.CreateLocalPos0Attr(Gf.Vec3f(px, py, pz))
    joint.CreateLocalRot0Attr(Gf.Quatf(*rot0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(*rot1))


def _sanitize(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]", "_", str(name))
    if s and s[0].isdigit():
        s = "_" + s
    return s


def _resolve_asset_root(compiled_model: Any, asset_root: Any) -> Path:
    from .assets import coerce_asset_context

    ctx = coerce_asset_context(asset_root)
    if ctx is not None:
        return Path(ctx.asset_root)
    model_ctx = coerce_asset_context(getattr(compiled_model, "assets", None))
    if model_ctx is not None:
        return Path(model_ctx.asset_root)
    if isinstance(asset_root, (str, os.PathLike)):
        return Path(asset_root)
    return Path(".").resolve()


def _resolve_material(ref: Any, materials_by_name: dict[str, Any]) -> Any:
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
    return vals if len(vals) >= 3 else None


def _principal_inertia(inertia: Any) -> tuple[tuple[float, float, float], tuple[float, ...]]:
    import numpy as np

    t = np.array(
        [
            [inertia.ixx, inertia.ixy, inertia.ixz],
            [inertia.ixy, inertia.iyy, inertia.iyz],
            [inertia.ixz, inertia.iyz, inertia.izz],
        ],
        dtype=float,
    )
    vals, vecs = np.linalg.eigh(t)
    if np.linalg.det(vecs) < 0:
        vecs[:, 0] = -vecs[:, 0]
    moments = (float(vals[0]), float(vals[1]), float(vals[2]))
    return moments, _mat3_to_quat(
        tuple(tuple(float(vecs[i][j]) for j in range(3)) for i in range(3))
    )


def _mat3_mul(a: Any, b: Any) -> tuple:
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3)
    )


def _mat3_to_quat(r: Any) -> tuple[float, float, float, float]:
    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w, x, y, z = (
            0.25 * s,
            (r[2][1] - r[1][2]) / s,
            (r[0][2] - r[2][0]) / s,
            (r[1][0] - r[0][1]) / s,
        )
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0
        w, x, y, z = (
            (r[2][1] - r[1][2]) / s,
            0.25 * s,
            (r[0][1] + r[1][0]) / s,
            (r[0][2] + r[2][0]) / s,
        )
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0
        w, x, y, z = (
            (r[0][2] - r[2][0]) / s,
            (r[0][1] + r[1][0]) / s,
            0.25 * s,
            (r[1][2] + r[2][1]) / s,
        )
    else:
        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0
        w, x, y, z = (
            (r[1][0] - r[0][1]) / s,
            (r[0][2] + r[2][0]) / s,
            (r[1][2] + r[2][1]) / s,
            0.25 * s,
        )
    return (w, x, y, z)


def _quat_to_mat3(q: tuple[float, float, float, float]) -> tuple:
    w, x, y, z = q
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def _quat_align_z_to(a: tuple[float, float, float]) -> tuple[float, float, float, float]:
    ax, ay, az = a
    mag = math.sqrt(ax * ax + ay * ay + az * az)
    if mag <= 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    ax, ay, az = ax / mag, ay / mag, az / mag
    d = az
    if d > 0.999999:
        return (1.0, 0.0, 0.0, 0.0)
    if d < -0.999999:
        return (0.0, 1.0, 0.0, 0.0)
    cx, cy, cz = -ay, ax, 0.0
    s = math.sqrt((1.0 + d) * 2.0)
    w = s * 0.5
    inv = 1.0 / s
    q = (w, cx * inv, cy * inv, cz * inv)
    n = math.sqrt(sum(c * c for c in q))
    return tuple(c / n for c in q)  # type: ignore[return-value]
