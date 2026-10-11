
from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import Any, Optional

from artisan.physics_policy import (
    DEFAULT_PENETRATION_DEPTH_TOL_M,
    penetration_depth_tolerance_from_env,
)

from .assets import AssetContext, coerce_asset_context
from .geometry_qc import _mat4_mul, _origin_to_mat4
from .inertia import (
    ABSOLUTE_MAX_DENSITY,
    DEFAULT_DENSITY,
    density_for_material,
    geometry_to_trimesh,
    mesh_inertial_from_trimesh,
)
from .types import Box, Collision, Inertia, Inertial, Mesh, Origin

_CONVEX_SKIP_TOL = 1e-6


def enrich_for_physics(
    compiled_model: Any,
    *,
    asset_root: Optional[AssetContext | str | Path] = None,
    density_override: Optional[float] = None,
    force_inertia: bool = False,
    collision_coacd: Optional[bool] = None,
    coacd_cache_dir: Optional[str | Path] = None,
) -> tuple[Any, list[str]]:
    warnings: list[str] = []
    if density_override is None:
        env_density = os.environ.get("ARTISAN_INERTIA_DENSITY")
        if env_density:
            try:
                density_override = float(env_density)
            except ValueError:
                pass

    root = _resolve_asset_root(compiled_model, asset_root)
    coacd_on = _coacd_enabled(collision_coacd)
    cache_dir = _coacd_cache_dir(coacd_cache_dir)
    materials_by_name = {
        getattr(m, "name", ""): m for m in getattr(compiled_model, "materials", []) or []
    }
    coacd_available: Optional[bool] = None

    for part in getattr(compiled_model, "parts", []):
        warnings.extend(_recover_primitive_collisions(part))

        note = _inject_part_inertial(
            part,
            asset_root=root,
            materials_by_name=materials_by_name,
            density_override=density_override,
            force=force_inertia,
        )
        if note:
            warnings.append(note)

        required = _required_decomposition_collisions(part)
        if required and not coacd_on:
            raise RuntimeError(
                f"required collision decomposition disabled for {part.name}: {', '.join(required)}"
            )
        if coacd_on:
            if coacd_available is None:
                from sdk._dependencies import sim_extra_available

                coacd_available = sim_extra_available("coacd")
                if not coacd_available:
                    warnings.append("coacd_skip: required coacd dependency missing (uv sync)")
            if required and coacd_available is False:
                raise RuntimeError(
                    f"required collision decomposition unavailable for {part.name}: "
                    f"{', '.join(required)}"
                )
            if coacd_available:
                warnings.extend(_coacd_part_collisions(part, asset_root=root, cache_dir=cache_dir))

        warnings.extend(_admit_mesh_collisions(part, asset_root=root))

        warnings.extend(_collision_qa_notes(part, asset_root=root))
    return compiled_model, warnings


def enrich_inertia_for_physics(
    compiled_model: Any,
    *,
    asset_root: Optional[AssetContext | str | Path] = None,
    density_override: Optional[float] = None,
    force_inertia: bool = False,
) -> tuple[Any, list[str]]:

    warnings: list[str] = []
    if density_override is None:
        env_density = os.environ.get("ARTISAN_INERTIA_DENSITY")
        if env_density:
            try:
                density_override = float(env_density)
            except ValueError:
                pass
    root = _resolve_asset_root(compiled_model, asset_root)
    materials_by_name = {
        getattr(material, "name", ""): material
        for material in getattr(compiled_model, "materials", []) or []
    }
    for part in getattr(compiled_model, "parts", []) or []:
        warnings.extend(_recover_primitive_collisions(part))
        note = _inject_part_inertial(
            part,
            asset_root=root,
            materials_by_name=materials_by_name,
            density_override=density_override,
            force=force_inertia,
        )
        if note:
            warnings.append(note)
    return compiled_model, warnings


def _required_decomposition_collisions(part: Any) -> list[str]:
    return [
        str(collision.name or part.name)
        for collision in getattr(part, "collisions", []) or []
        if isinstance(getattr(collision, "geometry", None), Mesh)
        and collision.geometry.collision_decomposition == "required"
    ]


def _collision_qa_notes(part: Any, *, asset_root: Path) -> list[str]:
    import numpy as np

    notes: list[str] = []
    for col in getattr(part, "collisions", []) or []:
        geom = getattr(col, "geometry", None)
        if not isinstance(geom, Mesh):
            continue
        if "/collision/" in str(getattr(geom, "filename", "") or ""):
            continue
        tm = geometry_to_trimesh(geom, asset_root=asset_root)
        if tm is None:
            continue
        if not bool(getattr(tm, "is_watertight", False)):
            notes.append(
                f"collision_qa: {part.name}/{col.name} open mesh collider (hull-fill risk)"
            )
            continue
        try:
            with np.errstate(invalid="ignore", divide="ignore"):
                vol = float(getattr(tm, "volume", 0.0) or 0.0)
                hull_vol = float(getattr(tm.convex_hull, "volume", 0.0) or 0.0)
            if hull_vol > 1e-12 and vol > 1e-12 and (vol / hull_vol) < 0.5:
                notes.append(
                    f"collision_qa: {part.name}/{col.name} concave mesh collider "
                    f"vol/hull={vol / hull_vol:.2f}"
                )
        except Exception:  # noqa: BLE001 — unprobeable mesh: no QA note
            pass
    return notes


def _coacd_enabled(flag: Optional[bool]) -> bool:
    if flag is not None:
        return bool(flag)
    return os.environ.get("ARTISAN_COLLISION_COACD", "1") != "0"


def nonwatertight_coacd_enabled() -> bool:
    return True


def _coacd_cache_dir(explicit: Optional[str | Path]) -> Path:
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get("ARTISAN_COACD_CACHE")
    if env:
        return Path(env)
    from artisan.config import cache_dir

    return cache_dir() / "collision_coacd"


def _resolve_asset_root(
    compiled_model: Any, asset_root: Optional[AssetContext | str | Path]
) -> Path:
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


def _density_for_part(
    part: Any, materials_by_name: dict[str, Any], *, override: Optional[float]
) -> float:
    if override is not None:
        return float(override)
    meta = getattr(part, "meta", None) or {}
    if "density" in meta:
        try:
            return float(meta["density"])
        except (TypeError, ValueError):
            pass
    for visual in getattr(part, "visuals", []) or []:
        mat = _resolve_material(getattr(visual, "material", None), materials_by_name)
        if mat is not None and getattr(mat, "material_class", None):
            return density_for_material(mat)
    return DEFAULT_DENSITY


def mass_audit_rows(
    compiled_model: Any,
    *,
    asset_root: Optional[AssetContext | str | Path] = None,
    density_override: Optional[float] = None,
) -> list[dict]:
    import trimesh

    root = _resolve_asset_root(compiled_model, asset_root)
    materials_by_name = {
        getattr(m, "name", ""): m for m in getattr(compiled_model, "materials", []) or []
    }
    rows: list[dict] = []
    for part in getattr(compiled_model, "parts", []):
        geoms = [c.geometry for c in getattr(part, "collisions", []) or []]
        if not geoms:
            geoms = [v.geometry for v in getattr(part, "visuals", []) or []]
        meshes = []
        for g in geoms:
            tm = geometry_to_trimesh(g, asset_root=root)
            if tm is not None:
                meshes.append(tm)
        if not meshes:
            continue
        combined = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        try:
            v_hull = float(combined.convex_hull.volume)
        except Exception:  # noqa: BLE001 — unprobeable hull: skip this part
            continue
        rho = _density_for_part(part, materials_by_name, override=density_override)
        ceiling = rho * v_hull
        est = getattr(part, "est_mass_kg", None)
        solidity = (float(est) / ceiling) if (est is not None and ceiling > 1e-12) else None
        rows.append(
            {
                "part": getattr(part, "name", "?"),
                "hull_volume_m3": v_hull,
                "material_density": rho,
                "solid_mass_ceiling_kg": ceiling,
                "est_mass_kg": float(est) if est is not None else None,
                "implied_solidity": solidity,
            }
        )
    return rows


def _inject_part_inertial(
    part: Any,
    *,
    asset_root: Path,
    materials_by_name: dict[str, Any],
    density_override: Optional[float],
    force: bool,
) -> Optional[str]:
    import numpy as np
    import trimesh

    if getattr(part, "inertial", None) is not None and not force:
        return None

    meta = getattr(part, "meta", None) or {}

    geoms_origins = [(c.geometry, c.origin) for c in getattr(part, "collisions", []) or []]
    if not geoms_origins:
        geoms_origins = [(v.geometry, v.origin) for v in getattr(part, "visuals", []) or []]

    meshes = []
    for geom, origin in geoms_origins:
        tm = geometry_to_trimesh(geom, asset_root=asset_root)
        if tm is None:
            continue
        tm = tm.copy()
        tm.apply_transform(np.asarray(_origin_to_mat4(origin), dtype=float))
        meshes.append(tm)

    if not meshes:
        part.inertial = _fallback_inertial()
        return f"inertia_fallback: {part.name} empty(default)"

    combined = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    density = _density_for_part(part, materials_by_name, override=density_override)

    note: Optional[str] = None
    est = getattr(part, "est_mass_kg", None)
    explicit_mass = est if est is not None else (meta["mass"] if "mass" in meta else None)
    if explicit_mass is not None:
        try:
            part.inertial = _scaled_inertial(combined, float(explicit_mass))
        except (TypeError, ValueError):
            explicit_mass = None
    if explicit_mass is None:
        inertial, approx = mesh_inertial_from_trimesh(combined, density=density)
        part.inertial = inertial
        if approx:
            note = f"inertia_approx: {part.name} {approx} (density={density:g})"

    clamp_note = _clamp_inertial_to_ceiling(part, combined)
    return clamp_note or note


def _scaled_inertial(tm: Any, mass: float) -> Inertial:
    inertial, _ = mesh_inertial_from_trimesh(tm, density=1.0)
    vol_mass = max(inertial.mass, 1e-12)
    scale = mass / vol_mass
    i = inertial.inertia
    return Inertial(
        mass=mass,
        inertia=Inertia(
            ixx=i.ixx * scale,
            ixy=i.ixy * scale,
            ixz=i.ixz * scale,
            iyy=i.iyy * scale,
            iyz=i.iyz * scale,
            izz=i.izz * scale,
        ),
        origin=inertial.origin,
    )


def _clamp_inertial_to_ceiling(part: Any, combined: Any) -> Optional[str]:
    inertial = getattr(part, "inertial", None)
    if inertial is None:
        return None
    try:
        v_hull = float(combined.convex_hull.volume)
    except Exception:  # noqa: BLE001 — unprobeable hull: skip clamp
        return None
    if v_hull <= 1e-9:
        return None
    mass_cap = ABSOLUTE_MAX_DENSITY * v_hull
    if inertial.mass > mass_cap:
        rho_impl = inertial.mass / v_hull
        part.inertial = _scaled_inertial(combined, mass_cap)
        return (
            f"inertia_clamp: {part.name} rho_impl={rho_impl:.0f} -> cap@{ABSOLUTE_MAX_DENSITY:.0f}"
        )
    return None


def _fallback_inertial() -> Inertial:
    from .inertia import MIN_MASS

    i = 1e-6
    return Inertial(
        mass=MIN_MASS,
        inertia=Inertia(ixx=i, ixy=0.0, ixz=0.0, iyy=i, iyz=0.0, izz=i),
        origin=Origin(),
    )




def _recover_primitive_collisions(part: Any) -> list[str]:
    notes: list[str] = []
    new_collisions: list[Any] = []
    for col in getattr(part, "collisions", []) or []:
        geom = getattr(col, "geometry", None)
        prims = getattr(geom, "collision_primitives", None) if isinstance(geom, Mesh) else None
        if prims:
            base = _origin_to_mat4(col.origin)
            base_name = col.name or part.name
            for k, (leg_geom, leg_origin) in enumerate(prims):
                composed = _mat4_mul(base, _origin_to_mat4(leg_origin))
                new_collisions.append(
                    Collision(
                        geometry=copy.deepcopy(leg_geom),
                        origin=_mat4_to_origin(composed),
                        name=f"{base_name}_cp{k}",
                        source_element_id=getattr(col, "source_element_id", None),
                        source_visual_uid=getattr(col, "source_visual_uid", None),
                        resolution_kind="primitive",
                        resolution_provenance=("analytic_primitives",),
                    )
                )
            notes.append(
                f"analytic_primitives: {part.name}/{col.name} -> {len(prims)} primitive colliders"
            )
            continue
        prim = getattr(geom, "source_geometry", None) if isinstance(geom, Mesh) else None
        if prim is None:
            new_collisions.append(col)
            continue
        composed = _origin_to_mat4(col.origin)
        st = getattr(geom, "source_transform", None)
        if st is not None:
            composed = _mat4_mul(composed, _as_mat4_rows(st))
        if not _is_rigid_mat4(composed):
            new_collisions.append(col)
            continue
        new_collisions.append(
            Collision(
                geometry=copy.deepcopy(prim),
                origin=_mat4_to_origin(composed),
                name=col.name,
                source_element_id=getattr(col, "source_element_id", None),
                source_visual_uid=getattr(col, "source_visual_uid", None),
                resolution_kind="primitive",
                resolution_provenance=("primitive_provenance",),
            )
        )
        notes.append(f"provenance: {part.name}/{col.name} mesh->primitive")
    part.collisions = new_collisions
    return notes


def _as_mat4_rows(m: Any) -> tuple:
    return tuple(tuple(float(v) for v in row) for row in m)


def _is_rigid_mat4(m: Any, *, tol: float = 1e-4) -> bool:
    cols = [(m[0][j], m[1][j], m[2][j]) for j in range(3)]
    for c in cols:
        if abs(math.sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]) - 1.0) > tol:
            return False

    def _dot(a: tuple, b: tuple) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    return (
        abs(_dot(cols[0], cols[1])) < tol
        and abs(_dot(cols[0], cols[2])) < tol
        and abs(_dot(cols[1], cols[2])) < tol
    )


def _mat4_to_origin(m: Any) -> Origin:
    x, y, z = float(m[0][3]), float(m[1][3]), float(m[2][3])
    r00, r10, r20 = float(m[0][0]), float(m[1][0]), float(m[2][0])
    r11, r12 = float(m[1][1]), float(m[1][2])
    r21, r22 = float(m[2][1]), float(m[2][2])
    sp = max(-1.0, min(1.0, -r20))
    pitch = math.asin(sp)
    if abs(sp) < 0.999999:
        roll = math.atan2(r21, r22)
        yaw = math.atan2(r10, r00)
    else:
        roll = math.atan2(-r12, r11)
        yaw = 0.0
    return Origin(xyz=(x, y, z), rpy=(roll, pitch, yaw))


def _coacd_part_collisions(part: Any, *, asset_root: Path, cache_dir: Path) -> list[str]:
    from ._coacd import decompose_mesh
    from .collision_mesh_repair import repair_collision_mesh

    notes: list[str] = []
    new_collisions: list[Any] = []
    for col in getattr(part, "collisions", []) or []:
        geom = getattr(col, "geometry", None)
        if not isinstance(geom, Mesh) or not _should_decompose(geom, asset_root):
            new_collisions.append(col)
            continue
        src = _resolve_mesh_source(geom, asset_root)
        required = geom.collision_decomposition == "required"
        if src is None:
            if required:
                raise RuntimeError(
                    f"required collision decomposition source unavailable: {part.name}/{col.name}"
                )
            notes.append(f"coacd_skip: {part.name}/{col.name} mesh source unavailable")
            new_collisions.append(col)
            continue
        working_src = src
        try:
            repair = repair_collision_mesh(src, cache_dir=cache_dir)
        except Exception as exc:  # noqa: BLE001 - policy below decides fallback vs block
            repair = None
            repair_error = f"{type(exc).__name__}: {exc}"
        else:
            repair_error = repair.failure_reason
            notes.append(
                f"collision_repair: {part.name}/{col.name} status={repair.status} "
                f"open_edges={repair.before.boundary_edges}->"
                f"{repair.after.boundary_edges if repair.after else 'unknown'} "
                f"filled_loops={len(repair.filled_loop_perimeters)}"
            )
            if repair.path is not None:
                working_src = repair.path
        if repair is None or repair.path is None:
            if required:
                if _append_collision_fallback(
                    new_collisions, part=part, collision=col, reason="repair_failed"
                ):
                    notes.append(f"cavity_fallback: {part.name}/{col.name} repair_failed")
                    continue
                raise RuntimeError(
                    f"required collision mesh repair failed: {part.name}/{col.name}: "
                    f"{repair_error or 'unknown error'}"
                )
            notes.append(
                f"convex_hull_fallback: {part.name}/{col.name} repair rollback "
                f"({repair_error or 'unknown error'})"
            )
            new_collisions.append(col)
            continue
        try:
            pieces = decompose_mesh(working_src, cache_dir=cache_dir)
        except Exception as exc:  # noqa: BLE001 — never break export on a CoACD failure
            if required:
                if _append_collision_fallback(
                    new_collisions, part=part, collision=col, reason="coacd_failed"
                ):
                    notes.append(f"cavity_fallback: {part.name}/{col.name} coacd_failed")
                    continue
                raise RuntimeError(
                    f"required collision decomposition failed: {part.name}/{col.name}"
                ) from exc
            notes.append(
                f"convex_hull_fallback: {part.name}/{col.name} CoACD failed ({type(exc).__name__})"
            )
            new_collisions.append(col)
            continue
        if not pieces:
            if required:
                if _append_collision_fallback(
                    new_collisions, part=part, collision=col, reason="coacd_empty"
                ):
                    notes.append(f"cavity_fallback: {part.name}/{col.name} coacd_empty")
                    continue
                raise RuntimeError(
                    f"required collision decomposition returned no pieces: {part.name}/{col.name}"
                )
            notes.append(
                f"convex_hull_fallback: {part.name}/{col.name} CoACD returned no convex pieces"
            )
            new_collisions.append(col)
            continue
        cavity = geom.collision_cavity
        if cavity is not None:
            penetration = _collision_cavity_penetration(pieces, cavity)
            tolerance = _penetration_depth_tolerance()
            if penetration >= tolerance:
                if _append_collision_fallback(
                    new_collisions, part=part, collision=col, reason="cavity_filled"
                ):
                    notes.append(
                        f"cavity_fallback: {part.name}/{col.name} penetration={penetration:.6g}m"
                    )
                    continue
                raise RuntimeError(
                    f"required collision decomposition fills cavity: {part.name}/{col.name} "
                    f"penetration={penetration:.6g}m tolerance={tolerance:.6g}m"
                )
            notes.append(f"cavity_probe: {part.name}/{col.name} penetration={penetration:.6g}m")
        key = pieces[0].parent.name
        base_name = col.name or part.name
        for k, piece in enumerate(pieces):
            new_collisions.append(
                Collision(
                    geometry=Mesh(
                        filename=f"assets/meshes/collision/{key}/{piece.name}",
                        name=f"{base_name}_cx{k}",
                        materialized_path=str(piece),
                    ),
                    origin=copy.deepcopy(col.origin),
                    name=f"{base_name}_cx{k}",
                    source_element_id=getattr(col, "source_element_id", None),
                    source_visual_uid=getattr(col, "source_visual_uid", None),
                    resolution_kind="convex_piece",
                    resolution_provenance=("collision_repair", "coacd"),
                )
            )
        notes.append(f"coacd: {part.name}/{col.name} -> {len(pieces)} convex pieces")
    part.collisions = new_collisions
    return notes


def _append_collision_fallback(
    target: list[Any],
    *,
    part: Any,
    collision: Any,
    reason: str,
) -> bool:
    geometry = getattr(collision, "geometry", None)
    fallback = geometry.collision_fallback_primitives if isinstance(geometry, Mesh) else None
    if not fallback:
        return False
    base = _origin_to_mat4(collision.origin)
    base_name = collision.name or part.name
    for index, (primitive, origin) in enumerate(fallback):
        composed = _mat4_mul(base, _origin_to_mat4(origin))
        target.append(
            Collision(
                geometry=copy.deepcopy(primitive),
                origin=_mat4_to_origin(composed),
                name=f"{base_name}_fallback_{reason}_{index}",
                source_element_id=getattr(collision, "source_element_id", None),
                source_visual_uid=getattr(collision, "source_visual_uid", None),
                resolution_kind="primitive",
                resolution_provenance=("declared_fallback", reason),
            )
        )
    return True


def _admit_mesh_collisions(part: Any, *, asset_root: Path) -> list[str]:
    from ._mesh_hull import check_mesh_file_hull

    collisions = list(getattr(part, "collisions", []) or [])
    checks: list[Any] = []
    for col in collisions:
        geom = getattr(col, "geometry", None)
        src = _resolve_mesh_source(geom, asset_root) if isinstance(geom, Mesh) else None
        checks.append(check_mesh_file_hull(src, geom.scale) if src is not None else None)
    residue: dict[str, list[Any]] = {}
    survivors: dict[str, int] = {}
    for col, check in zip(collisions, checks, strict=True):
        key = _collision_element_key(col)
        if check is not None and check.is_residue:
            residue.setdefault(key, []).append(col)
        else:
            survivors[key] = survivors.get(key, 0) + 1

    notes: list[str] = []
    admitted: list[Any] = []
    for col, check in zip(collisions, checks, strict=True):
        if check is None or check.kind == "ok":
            admitted.append(col)
            continue
        label = f"{part.name}/{col.name or 'collision'}"
        if check.is_residue:
            key = _collision_element_key(col)
            if survivors.get(key, 0):
                notes.append(
                    f"collision_residue_dropped: {label} extents={_format_extent(check.extent)}"
                )
            elif residue[key][0] is col:
                box = _residue_element_box(residue[key], asset_root=asset_root)
                admitted.append(box)
                notes.append(
                    f"collision_residue_box: {label} {len(residue[key])} residue piece(s) "
                    f"-> box {_format_extent(box.geometry.size)}"
                )
            continue
        if check.kind == "thin":
            notes.append(
                f"collision_thin: {label} hull_volume={check.volume:.3g}m3 -> mjcf shell inertia"
            )
            admitted.append(col)
            continue
        box = _aabb_box_collision(
            col, extent=check.extent, centre=check.centre, provenance="unhullable_box"
        )
        admitted.append(box)
        notes.append(f"collision_unhullable: {label} -> box {_format_extent(box.geometry.size)}")
    part.collisions = admitted
    return notes


def _collision_element_key(col: Any) -> str:
    return str(
        getattr(col, "source_element_id", None)
        or getattr(col, "source_visual_uid", None)
        or col.name
    )


def _format_extent(values: Any) -> str:
    return "x".join(f"{float(value):.4g}" for value in values) + "m"


def _aabb_box_collision(
    col: Any,
    *,
    extent: tuple[float, float, float],
    centre: tuple[float, float, float],
    provenance: str,
) -> Collision:
    from ._mesh_hull import MIN_SHEET_THICKNESS

    frame = _origin_to_mat4(col.origin)
    xyz = tuple(
        float(col.origin.xyz[row]) + sum(frame[row][axis] * centre[axis] for axis in range(3))
        for row in range(3)
    )
    return Collision(
        geometry=Box(size=tuple(max(float(value), MIN_SHEET_THICKNESS) for value in extent)),
        origin=Origin(xyz=xyz, rpy=tuple(col.origin.rpy)),
        name=col.name,
        source_element_id=getattr(col, "source_element_id", None),
        source_visual_uid=getattr(col, "source_visual_uid", None),
        resolution_kind="primitive",
        resolution_provenance=(*(getattr(col, "resolution_provenance", ()) or ()), provenance),
    )


def _residue_element_box(pieces: list[Any], *, asset_root: Path) -> Collision:
    import numpy as np

    from ._mesh_hull import load_mesh_vertices

    reference = np.asarray(_origin_to_mat4(pieces[0].origin), dtype=float)
    to_reference = np.linalg.inv(reference)
    points = []
    for piece in pieces:
        vertices = load_mesh_vertices(
            _resolve_mesh_source(piece.geometry, asset_root), piece.geometry.scale
        )
        frame = to_reference @ np.asarray(_origin_to_mat4(piece.origin), dtype=float)
        points.append(vertices @ frame[:3, :3].T + frame[:3, 3])
    stacked = np.concatenate(points)
    lower, upper = stacked.min(axis=0), stacked.max(axis=0)
    return _aabb_box_collision(
        pieces[0],
        extent=tuple(float(value) for value in upper - lower),
        centre=tuple(float(value) for value in (upper + lower) / 2.0),
        provenance="residue_box",
    )


def _should_decompose(geom: Mesh, asset_root: Path) -> bool:
    if geom.collision_decomposition == "required":
        return True
    tm = geometry_to_trimesh(geom, asset_root=asset_root)
    if tm is None:
        return False
    if not bool(getattr(tm, "is_watertight", False)):
        return True
    try:
        import numpy as np

        with np.errstate(invalid="ignore", divide="ignore"):
            vol = float(getattr(tm, "volume", 0.0) or 0.0)
            hull_vol = float(getattr(tm.convex_hull, "volume", 0.0) or 0.0)
        if hull_vol <= 1e-12 or vol <= 1e-12:
            return False
        ratio = vol / hull_vol
        if ratio > 1.0 + _CONVEX_SKIP_TOL:
            return True
        return ratio < 1.0 - _CONVEX_SKIP_TOL
    except Exception:  # noqa: BLE001 — treat unprobeable meshes as not-decomposable
        return False


def _penetration_depth_tolerance() -> float:
    value = penetration_depth_tolerance_from_env()
    return value if value > 0.0 else DEFAULT_PENETRATION_DEPTH_TOL_M


def _collision_cavity_penetration(
    pieces: list[Path],
    cavity: tuple[tuple[float, float, float], tuple[float, float, float], float, float],
) -> float:
    import numpy as np
    import trimesh

    origin, axis, radius, length = cavity
    transform = trimesh.geometry.align_vectors((0.0, 0.0, 1.0), axis)
    if transform is None:
        transform = np.eye(4, dtype=float)
    transform = np.asarray(transform, dtype=float)
    transform[:3, 3] = np.asarray(origin, dtype=float)
    probe = trimesh.creation.cylinder(
        radius=float(radius), height=float(length), sections=32, transform=transform
    )
    manager = trimesh.collision.CollisionManager()
    for index, path in enumerate(pieces):
        manager.add_object(f"piece_{index}", trimesh.load_mesh(path, force="mesh"))
    hit, contacts = manager.in_collision_single(probe, return_data=True)
    if not hit:
        return 0.0
    return max((float(contact.depth) for contact in contacts), default=0.0)


def _resolve_mesh_source(geom: Mesh, asset_root: Path) -> Optional[Path]:
    src = getattr(geom, "materialized_path", None)
    if src and Path(src).exists():
        return Path(src)
    fn = getattr(geom, "filename", None)
    if fn:
        candidate = (Path(asset_root) / str(fn)).resolve()
        if candidate.exists():
            return candidate
    return None
