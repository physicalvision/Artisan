
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .types import Box, Cylinder, Geometry, Inertia, Inertial, Material, Mesh, Origin, Sphere

_CLASS_DENSITY: dict[str, float] = {
    "steel": 7850.0,
    "stainless_steel": 7900.0,
    "aluminum": 2700.0,
    "brushed_aluminum": 2700.0,
    "brass": 8500.0,
    "copper": 8960.0,
    "chrome": 7190.0,
    "gold": 19300.0,
    "iron": 7870.0,
    "anodized_metal": 2700.0,
    "painted_metal": 7800.0,
    "matte_plastic": 1050.0,
    "glossy_plastic": 1050.0,
    "abs_plastic": 1050.0,
    "translucent_plastic": 1100.0,
    "rubber": 1100.0,
    "silicone": 1200.0,
    "wood": 650.0,
    "painted_wood": 700.0,
    "ceramic": 2300.0,
    "porcelain": 2400.0,
    "concrete": 2400.0,
    "stone": 2600.0,
    "marble": 2700.0,
    "fabric": 300.0,
    "leather": 900.0,
    "paper": 800.0,
    "cardboard": 700.0,
    "glass": 2500.0,
    "frosted_glass": 2500.0,
    "tinted_glass": 2500.0,
    "emissive": 1200.0,
}

DENSITY_TABLE_VERSION = 1
DEFAULT_DENSITY = 700.0
MIN_MASS = 1e-4
MIN_DIAG_INERTIA = 1e-9
ABSOLUTE_MAX_DENSITY = 23000.0


def density_for_material_class(material_class: Optional[str]) -> float:
    if material_class is None:
        return DEFAULT_DENSITY
    return _CLASS_DENSITY.get(str(material_class).strip().lower(), DEFAULT_DENSITY)


def density_for_material(material: Optional[Material]) -> float:
    cls = getattr(material, "material_class", None) if material is not None else None
    return density_for_material_class(cls)


def geometry_to_trimesh(geometry: Geometry, *, asset_root):
    import trimesh

    from .placement import _load_trimesh_mesh

    if isinstance(geometry, Mesh):
        mp = getattr(geometry, "materialized_path", None)
        if mp and Path(mp).exists():
            tm = trimesh.load_mesh(mp, force="mesh")
            if geometry.scale is not None:
                tm = tm.copy()
                tm.apply_scale(geometry.scale)
            return tm
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


def mesh_inertial_from_trimesh(
    tm, *, density: float, min_mass: float = MIN_MASS
) -> tuple[Inertial, Optional[str]]:
    import numpy as np

    with np.errstate(invalid="ignore", divide="ignore"):
        approx: Optional[str] = None
        source = tm
        watertight = bool(getattr(tm, "is_watertight", False))
        volume = float(getattr(tm, "volume", 0.0) or 0.0)
        if not watertight or volume <= 1e-12:
            hull = None
            try:
                hull = tm.convex_hull
            except Exception:  # noqa: BLE001 — degenerate mesh; fall through to bbox
                hull = None
            if hull is not None and float(getattr(hull, "volume", 0.0) or 0.0) > 1e-12:
                source = hull
                approx = "hull"
            else:
                approx = "bbox"

        if approx == "bbox":
            ext = np.maximum(np.asarray(getattr(tm, "extents", (0.0, 0.0, 0.0)), dtype=float), 1e-3)
            box = Box(size=(float(ext[0]), float(ext[1]), float(ext[2])))
            mass = max(density * float(ext[0] * ext[1] * ext[2]), min_mass)
            inertial = Inertial.from_geometry(box, mass)
            try:
                com = np.asarray(tm.bounds, dtype=float).mean(axis=0)
                inertial.origin = Origin(xyz=(float(com[0]), float(com[1]), float(com[2])))
            except Exception:  # noqa: BLE001 — keep identity origin if bounds unavailable
                pass
            return inertial, approx

        src = source.copy()
        try:
            src.density = float(density)
        except Exception:  # noqa: BLE001 — density setter unavailable; scale manually
            pass
        mass = max(float(getattr(src, "mass", density * max(volume, 0.0))), min_mass)
        com = np.asarray(getattr(src, "center_mass", (0.0, 0.0, 0.0)), dtype=float)
        inertia_t = _positive_definite_inertia(np.asarray(src.moment_inertia, dtype=float))
        inertial = Inertial(
            mass=mass,
            inertia=Inertia(
                ixx=float(inertia_t[0, 0]),
                ixy=float(inertia_t[0, 1]),
                ixz=float(inertia_t[0, 2]),
                iyy=float(inertia_t[1, 1]),
                iyz=float(inertia_t[1, 2]),
                izz=float(inertia_t[2, 2]),
            ),
            origin=Origin(xyz=(float(com[0]), float(com[1]), float(com[2]))),
        )
        return inertial, approx


def _positive_definite_inertia(inertia_t):
    import numpy as np

    sym = 0.5 * (inertia_t + inertia_t.T)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.maximum(vals, MIN_DIAG_INERTIA)
    spd = (vecs * vals) @ vecs.T
    return 0.5 * (spd + spd.T)
