from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

from .assets import AssetContext, coerce_asset_context
from .errors import ValidationError

Vec3 = Tuple[float, float, float]
Mat4 = Tuple[Tuple[float, float, float, float], ...]


def _as_vec3(values: Sequence[float], *, name: str) -> Vec3:
    if len(values) != 3:
        raise ValidationError(f"{name} must have 3 elements")
    return (float(values[0]), float(values[1]), float(values[2]))


def _as_mat4(values: Sequence[Sequence[float]], *, name: str) -> Mat4:
    if len(values) != 4:
        raise ValidationError(f"{name} must have 4 rows")
    rows: list[tuple[float, float, float, float]] = []
    for idx, row in enumerate(values):
        if len(row) != 4:
            raise ValidationError(f"{name}[{idx}] must have 4 elements")
        rows.append((float(row[0]), float(row[1]), float(row[2]), float(row[3])))
    return (rows[0], rows[1], rows[2], rows[3])


@dataclass(frozen=True)
class Origin:
    xyz: Vec3 = (0.0, 0.0, 0.0)
    rpy: Vec3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "xyz", _as_vec3(self.xyz, name="origin.xyz"))
        object.__setattr__(self, "rpy", _as_vec3(self.rpy, name="origin.rpy"))


@dataclass(frozen=True)
class Box:
    size: Vec3


@dataclass(frozen=True, init=False)
class Cylinder:
    radius: float
    length: float

    def __init__(
        self,
        radius: float,
        length: float | None = None,
        *,
        height: float | None = None,
    ) -> None:
        if length is not None and height is not None:
            raise TypeError("Cylinder() accepts only one of 'length' or alias 'height'")
        resolved_length = height if length is None else length
        if resolved_length is None:
            raise TypeError("Cylinder() missing required argument: 'length'")
        object.__setattr__(self, "radius", float(radius))
        object.__setattr__(self, "length", float(resolved_length))

    @property
    def height(self) -> float:
        return self.length


@dataclass(frozen=True)
class Sphere:
    radius: float


PrimitiveGeometry = Union[Box, Cylinder, Sphere]


@dataclass(frozen=True)
class Mesh:
    filename: Union[str, os.PathLike[str], None] = None
    name: Optional[str] = None
    scale: Optional[Vec3] = None
    source_geometry: Optional[PrimitiveGeometry] = None
    source_transform: Optional[Mat4] = None
    materialized_path: Optional[str] = field(default=None, compare=False, repr=False)
    collision_primitives: Optional[Tuple[Tuple[PrimitiveGeometry, "Origin"], ...]] = field(
        default=None, compare=False, repr=False
    )
    collision_decomposition: Literal["auto", "required"] = "auto"
    collision_cavity: Optional[Tuple[Vec3, Vec3, float, float]] = field(
        default=None, compare=False, repr=False
    )
    collision_fallback_primitives: Optional[Tuple[Tuple[PrimitiveGeometry, "Origin"], ...]] = field(
        default=None, compare=False, repr=False
    )
    anchor_sidecar: Optional[Union[str, os.PathLike[str]]] = None

    def __post_init__(self) -> None:
        filename_value = self.filename
        filename = os.fspath(filename_value) if filename_value is not None else ""
        logical_name = None if self.name is None else str(self.name).strip()
        if not filename and not logical_name:
            raise ValidationError("mesh.filename or mesh.name is required")
        if filename:
            object.__setattr__(self, "filename", filename)
        else:
            object.__setattr__(self, "filename", None)
        if logical_name is not None:
            if not logical_name:
                raise ValidationError("mesh.name must be non-empty when provided")
            object.__setattr__(self, "name", logical_name)
        if self.scale is not None:
            object.__setattr__(self, "scale", _as_vec3(self.scale, name="mesh.scale"))
        if self.source_geometry is not None and not isinstance(
            self.source_geometry, (Box, Cylinder, Sphere)
        ):
            raise ValidationError("mesh.source_geometry must be a Box, Cylinder, Sphere, or None")
        if self.source_transform is not None:
            if self.source_geometry is None:
                raise ValidationError("mesh.source_transform requires mesh.source_geometry")
            object.__setattr__(
                self,
                "source_transform",
                _as_mat4(self.source_transform, name="mesh.source_transform"),
            )
        if self.materialized_path is not None:
            object.__setattr__(self, "materialized_path", os.fspath(self.materialized_path))
        if self.anchor_sidecar is not None:
            anchor_sidecar = os.fspath(self.anchor_sidecar).strip()
            if not anchor_sidecar:
                raise ValidationError("mesh.anchor_sidecar must be non-empty when provided")
            object.__setattr__(self, "anchor_sidecar", anchor_sidecar)
        if self.collision_primitives is not None:
            coerced = tuple((prim, origin) for (prim, origin) in self.collision_primitives)
            for prim, origin in coerced:
                if not isinstance(prim, (Box, Cylinder, Sphere)):
                    raise ValidationError(
                        "mesh.collision_primitives entries must be (Box|Cylinder|Sphere, Origin)"
                    )
                if not isinstance(origin, Origin):
                    raise ValidationError("mesh.collision_primitives origin must be an Origin")
            object.__setattr__(self, "collision_primitives", coerced or None)
        if self.collision_decomposition not in {"auto", "required"}:
            raise ValidationError("mesh.collision_decomposition must be 'auto' or 'required'")
        if self.collision_cavity is not None:
            origin, axis, radius, length = self.collision_cavity
            origin_v = _as_vec3(origin, name="mesh.collision_cavity.origin")
            axis_v = _as_vec3(axis, name="mesh.collision_cavity.axis")
            magnitude = math.sqrt(sum(float(value) * float(value) for value in axis_v))
            if magnitude <= 1e-12:
                raise ValidationError("mesh.collision_cavity.axis must be non-zero")
            radius_f, length_f = float(radius), float(length)
            if radius_f <= 0.0 or length_f <= 0.0:
                raise ValidationError("mesh.collision_cavity radius and length must be positive")
            object.__setattr__(
                self,
                "collision_cavity",
                (
                    origin_v,
                    tuple(float(value) / magnitude for value in axis_v),
                    radius_f,
                    length_f,
                ),
            )
            if self.collision_decomposition != "required":
                raise ValidationError(
                    "mesh.collision_cavity requires collision_decomposition='required'"
                )
        if self.collision_fallback_primitives is not None:
            fallback = tuple(self.collision_fallback_primitives)
            for primitive, origin in fallback:
                if not isinstance(primitive, (Box, Cylinder, Sphere)) or not isinstance(
                    origin, Origin
                ):
                    raise ValidationError(
                        "mesh.collision_fallback_primitives entries must be "
                        "(Box|Cylinder|Sphere, Origin)"
                    )
            if self.collision_decomposition != "required":
                raise ValidationError(
                    "mesh.collision_fallback_primitives require collision_decomposition='required'"
                )
            object.__setattr__(self, "collision_fallback_primitives", fallback or None)


Geometry = Union[Box, Cylinder, Sphere, Mesh]


def _normalize_material_rgba(
    rgba: Optional[Sequence[float]],
    *,
    field_name: str = "material.rgba",
) -> Optional[Tuple[float, float, float, float]]:
    if rgba is None:
        return None
    normalized = tuple(float(v) for v in rgba)
    if len(normalized) not in (3, 4):
        raise ValidationError(f"{field_name} must have 3 or 4 values")
    if len(normalized) == 3:
        normalized = normalized + (1.0,)
    return normalized


_CLASS_PBR: Dict[str, Tuple[float, float, float]] = {
    "steel": (1.0, 0.35, 0.0),
    "stainless_steel": (1.0, 0.22, 0.0),
    "aluminum": (1.0, 0.35, 0.0),
    "brushed_aluminum": (1.0, 0.45, 0.0),
    "brass": (1.0, 0.30, 0.0),
    "copper": (1.0, 0.30, 0.0),
    "chrome": (1.0, 0.08, 0.0),
    "gold": (1.0, 0.20, 0.0),
    "iron": (1.0, 0.55, 0.0),
    "anodized_metal": (1.0, 0.40, 0.0),
    "painted_metal": (0.0, 0.45, 0.0),
    "matte_plastic": (0.0, 0.70, 0.0),
    "glossy_plastic": (0.0, 0.25, 0.0),
    "abs_plastic": (0.0, 0.50, 0.0),
    "translucent_plastic": (0.0, 0.35, 0.0),
    "rubber": (0.0, 0.90, 0.0),
    "silicone": (0.0, 0.60, 0.0),
    "wood": (0.0, 0.55, 0.0),
    "painted_wood": (0.0, 0.45, 0.0),
    "ceramic": (0.0, 0.30, 0.0),
    "porcelain": (0.0, 0.18, 0.0),
    "concrete": (0.0, 0.85, 0.0),
    "stone": (0.0, 0.80, 0.0),
    "marble": (0.0, 0.25, 0.0),
    "fabric": (0.0, 0.95, 0.0),
    "leather": (0.0, 0.60, 0.0),
    "paper": (0.0, 0.85, 0.0),
    "cardboard": (0.0, 0.90, 0.0),
    "glass": (0.0, 0.03, 0.0),
    "frosted_glass": (0.0, 0.35, 0.0),
    "tinted_glass": (0.0, 0.05, 0.0),
    "emissive": (0.0, 0.50, 5.0),
}

_CLASS_GLASS_DEFAULTS: Dict[str, Tuple[float, float]] = {
    "glass": (1.0, 1.45),
    "frosted_glass": (1.0, 1.45),
    "tinted_glass": (1.0, 1.50),
    "translucent_plastic": (0.5, 1.45),
}

_KNOWN_MATERIAL_CLASSES = frozenset(_CLASS_PBR)

_CLASS_TEXTURE_DEFAULT: Dict[str, str] = {
    "wood": "wood",
    "painted_wood": "wood",
    "fabric": "fabric",
    "leather": "leather",
    "marble": "marble",
    "concrete": "concrete",
    "stone": "concrete",
    "brushed_aluminum": "brushed_metal",
    "anodized_metal": "brushed_metal",
}

_DEFAULT_PBR: Tuple[float, float, float] = (0.0, 0.8, 0.0)


def _clamp01(value: Optional[float], *, name: str) -> Optional[float]:
    if value is None:
        return None
    v = float(value)
    if v < 0.0 or v > 1.0:
        raise ValidationError(f"{name} must be in [0, 1] (got {v})")
    return v


@dataclass(init=False)
class Material:
    name: str
    rgba: Optional[Tuple[float, float, float, float]] = None
    texture: Optional[str] = None
    material_class: Optional[str] = None
    metallic: Optional[float] = None
    roughness: Optional[float] = None
    emissive: Optional[float] = None
    texture_set: Optional[str] = None
    transmission: Optional[float] = None
    ior: Optional[float] = None
    attenuation_color: Optional[Tuple[float, float, float]] = None
    attenuation_distance: Optional[float] = None

    def __init__(
        self,
        name: str,
        rgba: Optional[Sequence[float]] = None,
        texture: Optional[str] = None,
        *,
        color: Optional[Sequence[float]] = None,
        material_class: Optional[str] = None,
        metallic: Optional[float] = None,
        roughness: Optional[float] = None,
        emissive: Optional[float] = None,
        texture_set: Optional[str] = None,
        transmission: Optional[float] = None,
        ior: Optional[float] = None,
        attenuation_color: Optional[Sequence[float]] = None,
        attenuation_distance: Optional[float] = None,
    ) -> None:
        if rgba is not None and color is not None:
            raise ValidationError("Material cannot set both rgba and color")
        self.name = str(name)
        self.rgba = _normalize_material_rgba(rgba if rgba is not None else color)
        self.texture = texture
        self.material_class = material_class
        self.metallic = metallic
        self.roughness = roughness
        self.emissive = emissive
        self.texture_set = texture_set
        self.transmission = transmission
        self.ior = ior
        self.attenuation_color = attenuation_color
        self.attenuation_distance = attenuation_distance
        self.__post_init__()

    def __post_init__(self) -> None:
        self.name = str(self.name)
        if not self.name:
            raise ValidationError("material.name is required")
        self.rgba = _normalize_material_rgba(self.rgba)
        if self.material_class is not None:
            self.material_class = str(self.material_class).strip().lower()
            if self.material_class not in _KNOWN_MATERIAL_CLASSES:
                raise ValidationError(
                    f"material.material_class {self.material_class!r} is not a known class; "
                    f"choose one of {sorted(_KNOWN_MATERIAL_CLASSES)}"
                )
        self.metallic = _clamp01(self.metallic, name="material.metallic")
        self.roughness = _clamp01(self.roughness, name="material.roughness")
        self.transmission = _clamp01(self.transmission, name="material.transmission")
        if self.emissive is not None:
            self.emissive = float(self.emissive)
            if self.emissive < 0.0:
                raise ValidationError("material.emissive must be >= 0")
        if self.ior is not None:
            self.ior = float(self.ior)
            if self.ior <= 0.0:
                raise ValidationError("material.ior must be > 0")
        if self.attenuation_color is not None:
            self.attenuation_color = _as_vec3(
                self.attenuation_color, name="material.attenuation_color"
            )
        if self.attenuation_distance is not None:
            self.attenuation_distance = float(self.attenuation_distance)
            if self.attenuation_distance <= 0.0:
                raise ValidationError("material.attenuation_distance must be > 0")


def resolve_material_pbr(material: "Material") -> Tuple[float, float, float]:
    metallic, roughness, emissive = _DEFAULT_PBR
    cls = getattr(material, "material_class", None)
    if cls in _CLASS_PBR:
        metallic, roughness, emissive = _CLASS_PBR[cls]
    m = getattr(material, "metallic", None)
    r = getattr(material, "roughness", None)
    e = getattr(material, "emissive", None)
    if m is not None:
        metallic = float(m)
    if r is not None:
        roughness = float(r)
    if e is not None:
        emissive = float(e)
    return (metallic, roughness, emissive)


def resolve_material_glass(material: "Material") -> Optional[Dict[str, object]]:
    cls = getattr(material, "material_class", None)
    transmission = getattr(material, "transmission", None)
    ior = getattr(material, "ior", None)
    if transmission is None and cls in _CLASS_GLASS_DEFAULTS:
        default_t, default_ior = _CLASS_GLASS_DEFAULTS[cls]
        transmission = default_t
        if ior is None:
            ior = default_ior
    if not transmission:
        return None
    spec: Dict[str, object] = {
        "transmission": float(transmission),
        "ior": float(ior if ior is not None else 1.45),
    }
    ac = getattr(material, "attenuation_color", None)
    ad = getattr(material, "attenuation_distance", None)
    if ac is not None:
        spec["attenuation_color"] = [float(c) for c in ac]
    if ad is not None:
        spec["attenuation_distance"] = float(ad)
    return spec


def resolve_material_texture(material: "Material") -> Optional[str]:
    ts = getattr(material, "texture_set", None)
    if ts:
        return str(ts)
    cls = getattr(material, "material_class", None)
    if cls in _CLASS_TEXTURE_DEFAULT:
        return _CLASS_TEXTURE_DEFAULT[cls]
    return None


MaterialRef = Union[Material, str]


def _coerce_visual_material_ref(
    *,
    material: Optional[MaterialRef],
    color: object,
) -> Optional[MaterialRef]:
    if material is not None and color is not None:
        raise TypeError("Part.visual() accepts only one of 'material' or alias 'color'")
    if color is None:
        return material
    if isinstance(color, (Material, str)):
        return color
    if isinstance(color, Sequence) and not isinstance(color, (str, bytes)):
        return Material(name="inline_color", rgba=tuple(float(v) for v in color))
    raise TypeError(
        "Part.visual() alias 'color' must be a material name, Material, or 3/4-value color tuple"
    )


@dataclass
class Visual:
    geometry: Geometry
    origin: Origin = field(default_factory=Origin)
    material: Optional[MaterialRef] = None
    name: Optional[str] = None
    visual_uid: Optional[str] = field(default=None, compare=False, repr=False)
    _visual_uid_committed: Optional[str] = field(
        default=None, init=False, compare=False, repr=False
    )
    _visual_uid_locked: bool = field(default=False, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_visual_uid_committed", self.visual_uid)
        object.__setattr__(self, "_visual_uid_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        if name == "visual_uid" and getattr(self, "_visual_uid_locked", False):
            current = getattr(self, "visual_uid", None)
            if value != current:
                if current is None:
                    raise ValidationError(
                        "Visual UID requires model-owned assignment after initialization"
                    )
                raise ValidationError("Visual UID is write-once")
        super().__setattr__(name, value)

    def _assign_visual_uid(self, value: str) -> None:
        current = self.visual_uid
        if current is not None and current != value:
            raise ValidationError("Visual UID is write-once")
        object.__setattr__(self, "visual_uid", value)
        object.__setattr__(self, "_visual_uid_committed", value)


@dataclass
class Collision:
    geometry: Geometry
    origin: Origin = field(default_factory=Origin)
    name: Optional[str] = None
    source_element_id: Optional[str] = field(default=None, compare=False, repr=False)
    resolved_piece_id: Optional[str] = field(default=None, compare=False, repr=False)
    resolution_kind: Optional[str] = field(default=None, compare=False, repr=False)
    resolution_provenance: Tuple[str, ...] = field(default_factory=tuple, compare=False, repr=False)
    source_visual_uid: Optional[str] = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class Inertia:
    ixx: float
    ixy: float
    ixz: float
    iyy: float
    iyz: float
    izz: float


@dataclass
class Inertial:
    mass: float
    inertia: Inertia
    origin: Origin = field(default_factory=Origin)

    @staticmethod
    def from_geometry(
        geometry: Geometry, mass: float, *, origin: Optional[Origin] = None
    ) -> "Inertial":
        mass = float(mass)
        if mass <= 0:
            raise ValidationError("Mass must be positive")

        if isinstance(geometry, Box):
            x, y, z = _as_vec3(geometry.size, name="box.size")
            ixx = (1.0 / 12.0) * mass * (y * y + z * z)
            iyy = (1.0 / 12.0) * mass * (x * x + z * z)
            izz = (1.0 / 12.0) * mass * (x * x + y * y)
        elif isinstance(geometry, Cylinder):
            r = float(geometry.radius)
            length = float(geometry.length)
            ixx = (1.0 / 12.0) * mass * (3 * r * r + length * length)
            iyy = ixx
            izz = 0.5 * mass * r * r
        elif isinstance(geometry, Sphere):
            r = float(geometry.radius)
            ixx = (2.0 / 5.0) * mass * r * r
            iyy = ixx
            izz = ixx
        else:
            raise ValidationError("Cannot derive inertia for mesh geometry")

        inertia = Inertia(ixx=ixx, ixy=0.0, ixz=0.0, iyy=iyy, iyz=0.0, izz=izz)
        return Inertial(mass=mass, inertia=inertia, origin=origin or Origin())

    @staticmethod
    def from_trimesh(tm, *, density: float, min_mass: float = 1e-4) -> "Inertial":
        from .inertia import mesh_inertial_from_trimesh

        inertial, _approx = mesh_inertial_from_trimesh(tm, density=density, min_mass=min_mass)
        return inertial


class ArticulationType(str, Enum):
    REVOLUTE = "revolute"
    CONTINUOUS = "continuous"
    PRISMATIC = "prismatic"
    FIXED = "fixed"
    FLOATING = "floating"


def _coerce_articulation_type(
    value: Union["ArticulationType", str],
) -> ArticulationType:
    try:
        return value if isinstance(value, ArticulationType) else ArticulationType(str(value))
    except ValueError as exc:
        raise ValidationError(f"Unknown articulation type: {value}") from exc


@dataclass(frozen=True)
class MotionLimits:
    effort: float = 1.0
    velocity: float = 1.0
    lower: Optional[float] = None
    upper: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "effort", float(self.effort))
        object.__setattr__(self, "velocity", float(self.velocity))
        if self.lower is not None:
            object.__setattr__(self, "lower", float(self.lower))
        if self.upper is not None:
            object.__setattr__(self, "upper", float(self.upper))


@dataclass(frozen=True)
class MotionProperties:
    damping: Optional[float] = None
    friction: Optional[float] = None
    spring_stiffness: Optional[float] = None
    spring_ref: Optional[float] = None


EFFORT_ANCHORS = {"light": 0.15, "medium": 0.40, "firm": 0.65, "stiff": 0.85}
MOTION_ANCHORS = {"coasting": 0.10, "smooth": 0.35, "damped": 0.70}
HOLD_KINDS = ("free", "holds_anywhere")


def _feel_scalar(
    value: str | float | None,
    anchors: Dict[str, float],
    default: float,
) -> tuple[float, Optional[str]]:
    if value is None:
        return default, None
    if isinstance(value, str):
        key = value.strip().lower()
        if key in anchors:
            return anchors[key], None
        return default, f"unknown feel tier: {value!r}"
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return default, f"invalid feel scalar: {value!r}"
    if not math.isfinite(scalar):
        return default, f"non-finite feel scalar: {value!r}"
    clamped = min(1.0, max(0.0, scalar))
    warning = None if clamped == scalar else f"feel scalar clamped: {value!r}"
    return clamped, warning


@dataclass(frozen=True)
class MotionFeel:

    effort: float = 0.40
    motion: float = 0.35
    hold: str = "free"
    note: str = ""
    warnings: tuple[str, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        effort = float(self.effort)
        motion = float(self.motion)
        if not math.isfinite(effort) or not 0.0 <= effort <= 1.0:
            raise ValidationError("motion_feel.effort must be finite and within [0, 1]")
        if not math.isfinite(motion) or not 0.0 <= motion <= 1.0:
            raise ValidationError("motion_feel.motion must be finite and within [0, 1]")
        hold = str(self.hold).strip().lower()
        if hold not in HOLD_KINDS:
            raise ValidationError(f"motion_feel.hold must be one of {HOLD_KINDS}")
        object.__setattr__(self, "effort", effort)
        object.__setattr__(self, "motion", motion)
        object.__setattr__(self, "hold", hold)
        object.__setattr__(self, "note", str(self.note))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))

    @classmethod
    def from_intent(
        cls,
        effort: str | float | None = None,
        motion: str | float | None = None,
        hold: Optional[str] = None,
        note: str = "",
    ) -> "MotionFeel":
        effort_value, effort_warning = _feel_scalar(effort, EFFORT_ANCHORS, 0.40)
        motion_value, motion_warning = _feel_scalar(motion, MOTION_ANCHORS, 0.35)
        hold_value = str(hold or "free").strip().lower()
        hold_warning: Optional[str] = None
        if hold_value not in HOLD_KINDS:
            hold_warning = f"unknown hold kind: {hold!r}"
            hold_value = "free"
        return cls(
            effort=effort_value,
            motion=motion_value,
            hold=hold_value,
            note=note,
            warnings=tuple(
                warning
                for warning in (effort_warning, motion_warning, hold_warning)
                if warning is not None
            ),
        )


def _coerce_motion_feel(value: object) -> Optional[MotionFeel]:
    if value is None or isinstance(value, MotionFeel):
        return value
    if isinstance(value, dict):
        allowed = {"effort", "motion", "hold", "note"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValidationError(f"motion_feel has unknown field(s): {unknown}")
        return MotionFeel.from_intent(**value)
    raise ValidationError("motion_feel must be a MotionFeel, dict, or None")


BEHAVIOR_KINDS = ("latch_release", "self_closer", "momentary")


@dataclass(frozen=True)
class ArticulationBehavior:

    kind: str
    driver: str
    target: Optional[float] = None
    threshold: Optional[float] = None
    direction: Optional[float] = None
    speed: Optional[float] = None
    strength: str = "firm"

    def __post_init__(self) -> None:
        if self.kind not in BEHAVIOR_KINDS:
            raise ValidationError(
                f"behavior.kind must be one of {BEHAVIOR_KINDS}, got {self.kind!r}"
            )
        driver = str(self.driver).strip()
        if not driver:
            raise ValidationError("behavior.driver is required")
        object.__setattr__(self, "driver", driver)
        if self.strength not in ("soft", "firm", "strong"):
            raise ValidationError(
                f"behavior.strength must be 'soft', 'firm' or 'strong', got {self.strength!r}"
            )
        for fld in ("target", "threshold", "direction", "speed"):
            v = getattr(self, fld)
            if v is not None and not math.isfinite(float(v)):
                raise ValidationError(f"behavior.{fld} must be finite (got {v!r})")


@dataclass(frozen=True)
class Mimic:

    joint: str
    multiplier: float = 1.0
    offset: float = 0.0
    poly: tuple[float, ...] = ()
    linear_multiplier: Optional[float] = None

    def __post_init__(self) -> None:
        joint_name = str(self.joint).strip()
        if not joint_name:
            raise ValidationError("mimic.joint is required")
        object.__setattr__(self, "joint", joint_name)
        poly = tuple(float(c) for c in self.poly)
        if poly:
            if any(not math.isfinite(c) for c in poly):
                raise ValidationError(f"mimic.poly coefficients must be finite (got {self.poly!r})")
            if all(c == 0.0 for c in poly):
                raise ValidationError("mimic.poly must have at least one non-zero coefficient")
            lm = self.linear_multiplier
            if lm is not None and not math.isfinite(float(lm)):
                raise ValidationError(f"mimic.linear_multiplier must be finite (got {lm!r})")
            object.__setattr__(self, "poly", poly)
            object.__setattr__(self, "offset", poly[0])
            object.__setattr__(
                self,
                "multiplier",
                float(lm) if lm is not None else (poly[1] if len(poly) > 1 else 0.0),
            )
            return
        multiplier = float(self.multiplier)
        offset = float(self.offset)
        if not math.isfinite(multiplier) or multiplier == 0.0:
            raise ValidationError(
                f"mimic.multiplier must be a non-zero finite number (got {self.multiplier!r})"
            )
        if not math.isfinite(offset):
            raise ValidationError(f"mimic.offset must be finite (got {self.offset!r})")
        object.__setattr__(self, "multiplier", multiplier)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "poly", ())


@dataclass
class Part:
    name: str
    visuals: List[Visual] = field(default_factory=list)
    collisions: List[Collision] = field(default_factory=list)
    inertial: Optional[Inertial] = None
    meta: Dict[str, object] = field(default_factory=dict)
    assets: Optional[AssetContext] = None
    est_mass_kg: Optional[float] = None

    def __post_init__(self) -> None:
        self.assets = coerce_asset_context(self.assets)
        if self.est_mass_kg is not None:
            self.est_mass_kg = float(self.est_mass_kg)
            if self.est_mass_kg <= 0.0:
                raise ValidationError(f"part.est_mass_kg must be > 0 (got {self.est_mass_kg})")

    def visual(
        self,
        geometry: Geometry,
        *,
        origin: Optional[Origin] = None,
        material: Optional[MaterialRef] = None,
        color: object = None,
        name: Optional[str] = None,
    ) -> Visual:
        visual = Visual(
            geometry=geometry,
            origin=origin or Origin(),
            material=_coerce_visual_material_ref(material=material, color=color),
            name=name,
        )
        self.visuals.append(visual)
        return visual

    def get_visual(self, name: str) -> Visual:
        key = str(name).strip()
        if not key:
            raise ValidationError("visual name is required")
        for visual in self.visuals:
            if visual.name == key:
                return visual
        raise ValidationError(f"Unknown visual on part {self.name!r}: {name!r}")


@dataclass(eq=False)
class Articulation:
    name: str
    articulation_type: Union[ArticulationType, str]
    parent: str
    child: str
    origin: Origin = field(default_factory=Origin)
    axis: Vec3 = (0.0, 0.0, 1.0)
    motion_limits: Optional[MotionLimits] = None
    motion_properties: Optional[MotionProperties] = None
    motion_feel: Optional[MotionFeel] = None
    mimic: Optional[Mimic] = None
    behavior: Optional["ArticulationBehavior"] = None
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.articulation_type = _coerce_articulation_type(self.articulation_type)
        self.axis = _as_vec3(self.axis, name="articulation.axis")
        self.motion_feel = _coerce_motion_feel(self.motion_feel)
        if self.motion_feel is not None:
            if self.articulation_type not in {
                ArticulationType.REVOLUTE,
                ArticulationType.CONTINUOUS,
                ArticulationType.PRISMATIC,
            }:
                raise ValidationError("motion_feel is supported only on a plain movable joint")
            properties = self.motion_properties
            if properties is not None and (
                properties.damping is not None or properties.friction is not None
            ):
                raise ValidationError(
                    "motion_feel cannot be combined with explicit damping/friction"
                )
        if self.mimic is not None and not isinstance(self.mimic, Mimic):
            if isinstance(self.mimic, str):
                self.mimic = Mimic(joint=self.mimic)
            else:
                joint = getattr(self.mimic, "joint", None)
                if joint is None:
                    raise ValidationError("articulation.mimic must be a Mimic or mimic-like object")
                self.mimic = Mimic(
                    joint=joint,
                    multiplier=getattr(self.mimic, "multiplier", 1.0),
                    offset=getattr(self.mimic, "offset", 0.0),
                )
        if self.motion_feel is not None and self.mimic is not None:
            raise ValidationError("motion_feel is not supported on a mimic follower")
        if self.motion_feel is not None and self.behavior is not None:
            raise ValidationError("motion_feel is not supported on a behavior-owned joint")

    @property
    def joint_type(self) -> ArticulationType:
        return self.articulation_type

    @joint_type.setter
    def joint_type(self, value: Union[ArticulationType, str]) -> None:
        self.articulation_type = _coerce_articulation_type(value)

    @property
    def limit(self) -> Optional[MotionLimits]:
        return self.motion_limits

    @limit.setter
    def limit(self, value: Optional[MotionLimits]) -> None:
        self.motion_limits = value

    @property
    def dynamics(self) -> Optional[MotionProperties]:
        return self.motion_properties

    @dynamics.setter
    def dynamics(self, value: Optional[MotionProperties]) -> None:
        self.motion_properties = value
