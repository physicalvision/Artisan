from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import warnings
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Literal, Mapping, Optional, Tuple, Union

from ._mesh_provenance import decompose_mesh_primitive_provenance
from .assets import AssetContext, resolve_asset_root, resolve_mesh_path
from .errors import ValidationError
from .geometry_qc import (
    AABB,
    _axis_angle_matrix,
    _geometry_local_aabb,
    _mat4_mul,
    _mat4_vec3,
    _origin_to_mat4,
    _transform_aabb,
    link_local_aabbs,
)
from .mesh import (
    ExtrudeGeometry,
    ExtrudeWithHolesGeometry,
    MeshGeometry,
    _ensure_ccw,
    _profile_points_2d,
    _triangulate_polygon,
)
from .types import Articulation, Box, Cylinder, Geometry, Mat4, Mesh, Origin, Part, Sphere, Visual

Vec3 = Tuple[float, float, float]
Mat3 = Tuple[Vec3, Vec3, Vec3]
SurfaceSubject = Union[Part, Visual, Geometry]
MeshSubject = Union[MeshGeometry, Mesh]
SurfaceWrapMapping = Literal["auto", "intrinsic", "nearest"]
_EPS = 1e-9
_TRIMESH_CACHE: dict[Path, tuple[tuple[int, int, int], object]] = {}
logger = logging.getLogger(__name__)
PLACEMENT_MATH_IDENTITY = "relational-placement-v1"
PARALLEL_FACE_DOT_THRESHOLD = 0.999

FrameKind = Literal["part_local", "parent_link_local", "shared_spec"]
AnchorKind = Literal["point", "plane", "axis"]
SelectorKind = Literal["face", "edge", "corner"]
AnchorSourceKind = Literal[
    "analytic_face",
    "analytic_edge",
    "analytic_corner",
    "part_bounds_face",
    "part_bounds_edge",
    "part_bounds_corner",
    "cadquery_face",
    "cadquery_edge",
    "cadquery_corner",
    "surface",
    "mesh_planar_patch",
    "mesh_feature_edge",
    "mesh_feature_corner",
    "semantic",
    "bounds",
    "explicit",
    "internal_surface_mount",
]
AnchorOwnerKind = Literal["visual", "part", "joint", "standalone", "unregistered"]
_ANCHOR_OWNER_KINDS = {"visual", "part", "joint", "standalone", "unregistered"}
_SIGNED_AXIS_VECTORS: dict[str, Vec3] = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}


def _signed_axis_key(value: str, *, name: str) -> str:
    key = str(value).strip().lower()
    if key not in _SIGNED_AXIS_VECTORS:
        expected = ", ".join(sorted(_SIGNED_AXIS_VECTORS))
        raise ValidationError(f"{name} must be one of: {expected}")
    return key


def _axis_index(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[axis[-1]]


@dataclass(frozen=True)
class FrameRef:

    kind: FrameKind
    owner: object

    def __post_init__(self) -> None:
        if self.kind not in {"part_local", "parent_link_local", "shared_spec"}:
            raise ValidationError(f"Unsupported placement frame kind: {self.kind!r}")

    @classmethod
    def part_local(cls, owner: object) -> FrameRef:
        return cls("part_local", owner)

    @classmethod
    def parent_link_local(cls, owner: object) -> FrameRef:
        return cls("parent_link_local", owner)

    @classmethod
    def shared_spec(cls, owner: object) -> FrameRef:
        return cls("shared_spec", owner)

    def matches(self, other: FrameRef) -> bool:
        if self.kind != other.kind:
            return False
        if self.owner is other.owner:
            return True
        if isinstance(self.owner, (str, int)) and isinstance(other.owner, type(self.owner)):
            return bool(self.owner == other.owner)
        return False


@dataclass(frozen=True)
class RigidFrame:

    position: Vec3 = (0.0, 0.0, 0.0)
    basis: Mat3 = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    def __post_init__(self) -> None:
        position = _as_vec3(self.position, name="rigid_frame.position")
        try:
            columns = tuple(
                _as_vec3(column, name=f"rigid_frame.basis[{index}]")
                for index, column in enumerate(self.basis)
            )
        except TypeError as exc:
            raise ValidationError("rigid_frame.basis must contain 3 axes") from exc
        if len(columns) != 3:
            raise ValidationError("rigid_frame.basis must contain 3 axes")
        if not all(math.isfinite(value) for value in position for _ in (0,)):
            raise ValidationError("rigid_frame.position must be finite")
        if not all(math.isfinite(value) for column in columns for value in column):
            raise ValidationError("rigid_frame.basis must be finite")
        for index, column in enumerate(columns):
            if abs(_norm(column) - 1.0) > 1e-7:
                raise ValidationError(f"rigid_frame.basis[{index}] must be unit length")
        if any(abs(_dot(columns[a], columns[b])) > 1e-7 for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValidationError("rigid_frame.basis axes must be orthogonal")
        if _dot(_cross(columns[0], columns[1]), columns[2]) < 1.0 - 1e-7:
            raise ValidationError("rigid_frame.basis must be right-handed")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "basis", (columns[0], columns[1], columns[2]))

    @classmethod
    def identity(cls) -> RigidFrame:
        return cls()

    @classmethod
    def from_origin(cls, origin: Origin) -> RigidFrame:
        matrix = _origin_to_mat4(origin)
        rotation = _mat3_from_mat4(matrix)
        return cls(position=origin.xyz, basis=_mat3_columns(rotation))

    def to_origin(self) -> Origin:
        return Origin(xyz=self.position, rpy=_mat3_to_rpy(_mat3_from_columns(*self.basis)))

    def axis(self, axis: str) -> Vec3:
        key = _signed_axis_key(axis, name="axis")
        base = self.basis[_axis_index(key)]
        sign = -1.0 if key.startswith("-") else 1.0
        return (sign * base[0], sign * base[1], sign * base[2])


@dataclass(frozen=True)
class AnchorSelector:

    kind: SelectorKind
    axes: tuple[str, ...]
    uv: Optional[tuple[float, float]] = None
    t: Optional[float] = None

    def __post_init__(self) -> None:
        expected_count = {"face": 1, "edge": 2, "corner": 3}.get(self.kind)
        if expected_count is None:
            raise ValidationError(f"Unsupported Anchor selector kind: {self.kind!r}")
        normalized = tuple(
            _signed_axis_key(axis, name=f"selector.axes[{index}]")
            for index, axis in enumerate(self.axes)
        )
        if len(normalized) != expected_count:
            raise ValidationError(f"{self.kind} selector requires {expected_count} axes")
        if len({_axis_index(axis) for axis in normalized}) != len(normalized):
            raise ValidationError(f"{self.kind} selector axes must refer to different axes")
        if self.uv is not None:
            uv = _as_pair(self.uv, name="selector.uv")
            if any(value < 0.0 or value > 1.0 for value in uv):
                raise ValidationError("selector.uv is outside [0, 1]")
            object.__setattr__(self, "uv", uv)
        if self.t is not None:
            t = float(self.t)
            if not 0.0 <= t <= 1.0:
                raise ValidationError("selector.t is outside [0, 1]")
            object.__setattr__(self, "t", t)
        object.__setattr__(self, "axes", normalized)

    @classmethod
    def face(cls, axis: str, *, uv: tuple[float, float] = (0.5, 0.5)) -> AnchorSelector:
        return cls("face", (axis,), uv=uv)

    @classmethod
    def edge(cls, first: str, second: str, *, t: float = 0.5) -> AnchorSelector:
        return cls("edge", (first, second), t=t)

    @classmethod
    def corner(cls, first: str, second: str, third: str) -> AnchorSelector:
        return cls("corner", (first, second, third))


@dataclass(frozen=True)
class ControlledSubspace:

    frame: FrameRef
    translation_axes: tuple[Vec3, ...] = ()
    rotation_axes: tuple[Vec3, ...] = ()


def _stable_origin_value(origin: Origin) -> dict[str, list[float]]:
    return {
        "xyz": [float(component) for component in origin.xyz],
        "rpy": [float(component) for component in origin.rpy],
    }


def _stable_geometry_value(geometry: Geometry) -> dict[str, object]:
    if isinstance(geometry, Box):
        return {"type": "box", "size": [float(value) for value in geometry.size]}
    if isinstance(geometry, Cylinder):
        return {
            "type": "cylinder",
            "radius": float(geometry.radius),
            "length": float(geometry.length),
        }
    if isinstance(geometry, Sphere):
        return {"type": "sphere", "radius": float(geometry.radius)}

    def serialized_primitives(
        values: Optional[Tuple[Tuple[Union[Box, Cylinder, Sphere], Origin], ...]],
    ) -> object:
        if not values:
            return None
        return [
            {
                "geometry": _stable_geometry_value(primitive),
                "origin": _stable_origin_value(origin),
            }
            for primitive, origin in values
        ]

    return {
        "type": "mesh",
        "filename": None if geometry.filename is None else str(geometry.filename),
        "name": geometry.name,
        "scale": (None if geometry.scale is None else [float(value) for value in geometry.scale]),
        "source_geometry": (
            None
            if geometry.source_geometry is None
            else _stable_geometry_value(geometry.source_geometry)
        ),
        "source_transform": (
            None
            if geometry.source_transform is None
            else [[float(component) for component in row] for row in geometry.source_transform]
        ),
        "collision_primitives": serialized_primitives(geometry.collision_primitives),
        "collision_decomposition": geometry.collision_decomposition,
        "collision_cavity": (
            None
            if geometry.collision_cavity is None
            else [
                [float(value) for value in geometry.collision_cavity[0]],
                [float(value) for value in geometry.collision_cavity[1]],
                float(geometry.collision_cavity[2]),
                float(geometry.collision_cavity[3]),
            ]
        ),
        "collision_fallback_primitives": serialized_primitives(
            geometry.collision_fallback_primitives
        ),
        "anchor_sidecar": (
            None if geometry.anchor_sidecar is None else str(geometry.anchor_sidecar)
        ),
    }


def _stable_source_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Visual):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "geometry": _stable_geometry_value(value.geometry),
            "origin": _stable_origin_value(value.origin),
            "material": _stable_source_value(value.material),
            "name": value.name,
        }
    if isinstance(value, (list, tuple)):
        return [_stable_source_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _stable_source_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    values = getattr(value, "__dict__", None)
    if isinstance(values, dict):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                key: _stable_source_value(item)
                for key, item in sorted(values.items())
                if not key.startswith("_")
                and isinstance(item, (type(None), bool, int, float, str, Path, list, tuple, dict))
            },
        }
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _standalone_owner_ref(
    subject: object,
    *,
    local_pose: RigidFrame,
    selector: object = None,
    stable_subject_identity: str | None = None,
) -> str:
    qualified = f"{type(subject).__module__}.{type(subject).__qualname__}"
    payload = {
        "subject": (
            _stable_source_value(subject)
            if stable_subject_identity is None
            else {
                "type": qualified,
                "stable_identity": stable_subject_identity,
            }
        ),
        "selector": _stable_source_value(selector),
        "local_pose": _stable_source_value(local_pose),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    registered = isinstance(subject, (Box, Cylinder, Sphere, Mesh, Visual))
    prefix = "standalone" if registered else "unregistered"
    return f"{prefix}:{qualified}:{digest}"


@dataclass(frozen=True)
class AnchorSource:
    source_kind: AnchorSourceKind
    owner_ref: str
    subject: object = field(repr=False, compare=False)
    owner_kind: AnchorOwnerKind | None = None

    def __post_init__(self) -> None:
        prefix = self.owner_ref.partition(":")[0]
        if prefix not in _ANCHOR_OWNER_KINDS:
            raise ValidationError(f"Unsupported Anchor owner_ref prefix: {prefix!r}")
        inferred = prefix
        if self.owner_kind is None:
            object.__setattr__(self, "owner_kind", inferred)
            return
        if self.owner_kind not in _ANCHOR_OWNER_KINDS:
            raise ValidationError(f"Unsupported Anchor owner kind: {self.owner_kind!r}")
        if prefix in _ANCHOR_OWNER_KINDS and self.owner_kind != inferred:
            raise ValidationError(
                "Anchor owner_kind must match the owner_ref prefix "
                f"({self.owner_kind!r} != {prefix!r})"
            )

    def to_dict(
        self,
        *,
        owner_ref: Optional[str] = None,
        owner_kind: AnchorOwnerKind | None = None,
    ) -> dict[str, str]:
        return {
            "source_kind": self.source_kind,
            "owner_kind": str(owner_kind or self.owner_kind),
            "owner_ref": self.owner_ref if owner_ref is None else owner_ref,
        }


def _anchor_source(
    subject: object,
    source_kind: AnchorSourceKind,
    *,
    local_pose: RigidFrame,
    selector: object = None,
    owner_ref: Optional[str] = None,
) -> AnchorSource:
    if owner_ref is None:
        if isinstance(subject, Part):
            owner_ref = f"part:{subject.name}"
        elif isinstance(subject, Articulation):
            owner_ref = f"joint:{subject.name}"
        else:
            owner_ref = _standalone_owner_ref(
                subject,
                local_pose=local_pose,
                selector=selector,
            )
    return AnchorSource(source_kind=source_kind, owner_ref=owner_ref, subject=subject)


@dataclass(frozen=True, eq=False)
class _FrozenMapping(Mapping[str, object]):
    _items: tuple[tuple[str, object], ...]

    def __getitem__(self, key: str) -> object:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)


def _freeze_provenance_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _FrozenMapping(
            tuple((str(key), _freeze_provenance_value(item)) for key, item in value.items())
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_provenance_value(item) for item in value)
    return value


def _thaw_provenance_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_provenance_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_provenance_value(item) for item in value]
    return value


@dataclass(frozen=True)
class PlacementStep:
    operation: str
    subject_source: AnchorSource
    host_sources: tuple[AnchorSource, ...]
    commanded_clearance: Optional[float] = None
    commanded_axis_map: Optional[tuple[tuple[str, str], ...]] = None
    effective_axis_map: tuple[tuple[str, str], ...] = ()
    orientation_measurement: Optional[Mapping[str, object]] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "host_sources", tuple(self.host_sources))
        if self.commanded_axis_map is not None:
            object.__setattr__(
                self,
                "commanded_axis_map",
                tuple(tuple(pair) for pair in self.commanded_axis_map),
            )
        object.__setattr__(
            self,
            "effective_axis_map",
            tuple(tuple(pair) for pair in self.effective_axis_map),
        )
        if self.orientation_measurement is None:
            return
        frozen = _freeze_provenance_value(self.orientation_measurement)
        if not isinstance(frozen, Mapping):
            raise TypeError("orientation_measurement must be a mapping")
        object.__setattr__(self, "orientation_measurement", frozen)


@dataclass(frozen=True)
class Anchor:
    pose: RigidFrame
    frame: FrameRef
    kind: AnchorKind
    subject: object = field(repr=False, compare=False)
    subject_pose: RigidFrame = field(repr=False, compare=False)
    local_pose: RigidFrame = field(repr=False, compare=False)
    source: Optional[AnchorSource] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.source is None:
            object.__setattr__(
                self,
                "source",
                _anchor_source(
                    self.subject,
                    "explicit",
                    local_pose=self.local_pose,
                ),
            )

    def offset(
        self,
        *,
        translate: Vec3 = (0.0, 0.0, 0.0),
        clearance: float = 0.0,
        uv: tuple[float, float] = (0.0, 0.0),
    ) -> Anchor:
        local_translate = _as_vec3(translate, name="translate")
        uv_offset = _as_pair(uv, name="uv")
        delta = (
            local_translate[0] + uv_offset[0],
            local_translate[1] + uv_offset[1],
            local_translate[2] + float(clearance),
        )
        world_delta = _mat3_vec3(_mat3_from_columns(*self.pose.basis), delta)
        shifted = RigidFrame(
            position=tuple(self.pose.position[index] + world_delta[index] for index in range(3)),  # type: ignore[arg-type]
            basis=self.pose.basis,
        )
        return Anchor(
            pose=shifted,
            frame=self.frame,
            kind=self.kind,
            subject=self.subject,
            subject_pose=self.subject_pose,
            local_pose=_relative_rigid_frame(self.subject_pose, shifted),
            source=self.source,
        )


@dataclass(frozen=True)
class PlacementResult:
    subject: object
    frame: FrameRef
    pose: RigidFrame
    write_subspace: Optional[ControlledSubspace] = None
    provenance_steps: tuple[PlacementStep, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance_steps", tuple(self.provenance_steps))
        if self.write_subspace is None:
            object.__setattr__(self, "write_subspace", ControlledSubspace(self.frame))
        elif not self.write_subspace.frame.matches(self.frame):
            raise ValidationError("write_subspace frame must match the placement result frame")

    @property
    def origin(self) -> Origin:
        return self.pose.to_origin()


@dataclass(frozen=True)
class PlacementTarget:
    kind: Literal["visual", "joint"]
    target: Union[Visual, Articulation]

    @classmethod
    def visual(cls, visual: Visual) -> PlacementTarget:
        return cls("visual", visual)

    @classmethod
    def joint(cls, joint: Articulation) -> PlacementTarget:
        return cls("joint", joint)


@dataclass(frozen=True)
class SurfaceFrame:
    point: Vec3
    normal: Vec3
    tangent_u: Vec3
    tangent_v: Vec3


@dataclass(frozen=True)
class _SurfaceHit:
    point: Vec3
    normal: Vec3
    distance: float


def _identity4() -> Mat4:
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _as_pair(values: object, *, name: str) -> tuple[float, float]:
    try:
        if len(values) != 2:  # type: ignore[arg-type]
            raise ValidationError(f"{name} must have 2 elements")
        return (float(values[0]), float(values[1]))  # type: ignore[index]
    except TypeError as exc:
        raise ValidationError(f"{name} must have 2 elements") from exc


def _as_vec3(values: object, *, name: str) -> Vec3:
    try:
        if len(values) != 3:  # type: ignore[arg-type]
            raise ValidationError(f"{name} must have 3 elements")
        return (
            float(values[0]),  # type: ignore[index]
            float(values[1]),  # type: ignore[index]
            float(values[2]),  # type: ignore[index]
        )
    except TypeError as exc:
        raise ValidationError(f"{name} must have 3 elements") from exc


def _aabb_union(a: list[AABB]) -> AABB:
    if not a:
        raise ValidationError("Cannot union empty AABB list")
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    for mn, mx in a:
        for i in range(3):
            mins[i] = min(mins[i], mn[i])
            maxs[i] = max(maxs[i], mx[i])
    return (mins[0], mins[1], mins[2]), (maxs[0], maxs[1], maxs[2])


def _aabb_center(aabb: AABB) -> Vec3:
    (mn, mx) = aabb
    return ((mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5, (mn[2] + mx[2]) * 0.5)


def _aabb_size(aabb: AABB) -> Vec3:
    (mn, mx) = aabb
    return (mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])


def _normalize_axes_spec(
    axes: Union[str, Iterable[str]],
    *,
    name: str,
) -> tuple[int, ...]:
    if isinstance(axes, str):
        tokens = [char.lower() for char in axes if not char.isspace()]
    else:
        tokens = [str(axis).strip().lower() for axis in axes]
    if not tokens:
        raise ValidationError(f"{name} must specify at least one axis")
    axis_map = {"x": 0, "y": 1, "z": 2}
    normalized: list[int] = []
    for token in tokens:
        if token not in axis_map:
            raise ValidationError(f"{name} must use only axes x, y, z")
        idx = axis_map[token]
        if idx not in normalized:
            normalized.append(idx)
    return tuple(normalized)


def _dot(a: Vec3, b: Vec3) -> float:
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(v: Vec3) -> float:
    return math.sqrt(_dot(v, v))


def _normalize(v: Vec3, *, name: str) -> Vec3:
    n = _norm(v)
    if n <= _EPS:
        raise ValidationError(f"{name} must be non-zero")
    return (v[0] / n, v[1] / n, v[2] / n)


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(float(value), float(lo)), float(hi))


def _mat3_from_mat4(mat: Mat4) -> Mat3:
    return (
        (float(mat[0][0]), float(mat[0][1]), float(mat[0][2])),
        (float(mat[1][0]), float(mat[1][1]), float(mat[1][2])),
        (float(mat[2][0]), float(mat[2][1]), float(mat[2][2])),
    )


def _mat3_transpose(mat: Mat3) -> Mat3:
    return (
        (mat[0][0], mat[1][0], mat[2][0]),
        (mat[0][1], mat[1][1], mat[2][1]),
        (mat[0][2], mat[1][2], mat[2][2]),
    )


def _mat3_vec3(mat: Mat3, vec: Vec3) -> Vec3:
    return (
        mat[0][0] * vec[0] + mat[0][1] * vec[1] + mat[0][2] * vec[2],
        mat[1][0] * vec[0] + mat[1][1] * vec[1] + mat[1][2] * vec[2],
        mat[2][0] * vec[0] + mat[2][1] * vec[1] + mat[2][2] * vec[2],
    )


def _mat3_mul(a: Mat3, b: Mat3) -> Mat3:
    rows: list[Vec3] = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j])
        rows.append((row[0], row[1], row[2]))
    return (rows[0], rows[1], rows[2])


def _mat3_from_columns(a: Vec3, b: Vec3, c: Vec3) -> Mat3:
    return (
        (a[0], b[0], c[0]),
        (a[1], b[1], c[1]),
        (a[2], b[2], c[2]),
    )


def _mat3_columns(matrix: Mat3) -> Mat3:
    return (
        (matrix[0][0], matrix[1][0], matrix[2][0]),
        (matrix[0][1], matrix[1][1], matrix[2][1]),
        (matrix[0][2], matrix[1][2], matrix[2][2]),
    )


def _compose_rigid_frames(parent: RigidFrame, child: RigidFrame) -> RigidFrame:
    parent_rotation = _mat3_from_columns(*parent.basis)
    child_rotation = _mat3_from_columns(*child.basis)
    rotated_position = _mat3_vec3(parent_rotation, child.position)
    return RigidFrame(
        position=(
            parent.position[0] + rotated_position[0],
            parent.position[1] + rotated_position[1],
            parent.position[2] + rotated_position[2],
        ),
        basis=_mat3_columns(_mat3_mul(parent_rotation, child_rotation)),
    )


def _relative_rigid_frame(parent: RigidFrame, child: RigidFrame) -> RigidFrame:
    parent_rotation = _mat3_from_columns(*parent.basis)
    inverse_rotation = _mat3_transpose(parent_rotation)
    delta = (
        child.position[0] - parent.position[0],
        child.position[1] - parent.position[1],
        child.position[2] - parent.position[2],
    )
    child_rotation = _mat3_from_columns(*child.basis)
    return RigidFrame(
        position=_mat3_vec3(inverse_rotation, delta),
        basis=_mat3_columns(_mat3_mul(inverse_rotation, child_rotation)),
    )


def _mat3_to_rpy(mat: Mat3) -> Vec3:
    pitch = math.asin(_clamp(-mat[2][0], -1.0, 1.0))
    cp = math.cos(pitch)
    if abs(cp) > 1e-8:
        roll = math.atan2(mat[2][1], mat[2][2])
        yaw = math.atan2(mat[1][0], mat[0][0])
        return (roll, pitch, yaw)

    roll = 0.0
    if pitch >= 0.0:
        yaw = math.atan2(-mat[0][1], mat[1][1])
    else:
        yaw = math.atan2(mat[0][1], mat[1][1])
    return (roll, pitch, yaw)


def _mat4_inverse_rigid(tf: Mat4) -> Mat4:
    rot = _mat3_from_mat4(tf)
    rot_t = _mat3_transpose(rot)
    trans = (float(tf[0][3]), float(tf[1][3]), float(tf[2][3]))
    inv_trans = _mat3_vec3(rot_t, (-trans[0], -trans[1], -trans[2]))
    return (
        (rot_t[0][0], rot_t[0][1], rot_t[0][2], inv_trans[0]),
        (rot_t[1][0], rot_t[1][1], rot_t[1][2], inv_trans[1]),
        (rot_t[2][0], rot_t[2][1], rot_t[2][2], inv_trans[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def _mat4_transform_direction(tf: Mat4, vec: Vec3) -> Vec3:
    rot = _mat3_from_mat4(tf)
    return _mat3_vec3(rot, vec)


def _resolve_asset_root_for_subject(
    asset_root: Optional[Union[Path, str, AssetContext]],
    subject: SurfaceSubject,
) -> Optional[Path]:
    if isinstance(subject, Part):
        return resolve_asset_root(asset_root, subject)
    if asset_root is None:
        return None
    if isinstance(asset_root, AssetContext):
        return asset_root.asset_root
    return Path(asset_root)


def _iter_subject_items(
    subject: SurfaceSubject,
    *,
    prefer_collisions: bool,
) -> list[tuple[Geometry, Mat4]]:
    if isinstance(subject, Part):
        if prefer_collisions and subject.collisions:
            items = list(subject.collisions)
        elif subject.visuals:
            items = list(subject.visuals)
        else:
            items = list(subject.collisions)
        return [
            (item.geometry, _origin_to_mat4(getattr(item, "origin", Origin())))
            for item in items
            if getattr(item, "geometry", None) is not None
        ]

    if isinstance(subject, Visual):
        return [(subject.geometry, _origin_to_mat4(subject.origin))]

    return [(subject, _identity4())]


def _expand_mesh_provenance(geometry: Geometry, tf: Mat4) -> tuple[Geometry, Mat4]:
    if isinstance(geometry, Mesh):
        decomposed = decompose_mesh_primitive_provenance(geometry)
        if decomposed is not None:
            primitive, local_tf = decomposed
            return primitive, _mat4_mul(tf, local_tf)
    return geometry, tf


def _subject_world_aabb(
    subject: SurfaceSubject,
    *,
    asset_root: Optional[Path],
    prefer_collisions: bool,
) -> AABB:
    aabbs: list[AABB] = []
    obj_cache: dict[Path, AABB] = {}
    for geometry, tf in _iter_subject_items(subject, prefer_collisions=prefer_collisions):
        geometry, tf = _expand_mesh_provenance(geometry, tf)
        local_aabb = _geometry_local_aabb(geometry, asset_root=asset_root, _obj_cache=obj_cache)
        aabbs.append(_transform_aabb(local_aabb, tf))
    if not aabbs:
        raise ValidationError("subject has no geometry to compute AABB")
    return _aabb_union(aabbs)


def link_local_aabb(
    link: Part,
    *,
    asset_root: Optional[Union[Path, str, AssetContext]] = None,
    prefer_collisions: bool = True,
) -> Optional[AABB]:
    root = resolve_asset_root(asset_root, link)
    aabbs = link_local_aabbs(
        link,
        asset_root=root,
        prefer_collisions=prefer_collisions,
    )
    if not aabbs:
        return None
    return _aabb_union(aabbs)


def part_local_aabb(
    part: Part,
    *,
    asset_root: Optional[Union[Path, str, AssetContext]] = None,
    prefer_collisions: bool = True,
) -> Optional[AABB]:
    return link_local_aabb(
        part,
        asset_root=asset_root,
        prefer_collisions=prefer_collisions,
    )


def align_centers(
    child_aabb: AABB,
    parent_aabb: AABB,
    *,
    axes: Union[str, Iterable[str]] = ("x", "y", "z"),
) -> Origin:
    axis_indices = _normalize_axes_spec(axes, name="axes")
    child_center = _aabb_center(child_aabb)
    parent_center = _aabb_center(parent_aabb)
    xyz = [0.0, 0.0, 0.0]
    for axis_idx in axis_indices:
        xyz[axis_idx] = float(parent_center[axis_idx]) - float(child_center[axis_idx])
    return Origin(xyz=(xyz[0], xyz[1], xyz[2]))


def align_centers_xy(child_aabb: AABB, parent_aabb: AABB) -> Origin:
    warnings.warn(
        "align_centers_xy(...) is deprecated; use align_centers(..., axes=('x', 'y')) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return align_centers(child_aabb, parent_aabb, axes=("x", "y"))


def place_on_top(
    child_aabb: AABB,
    parent_aabb: AABB,
    *,
    clearance: float = 0.0,
    align_xy: bool = True,
) -> Origin:
    warnings.warn(
        "place_on_top(...) is deprecated; prefer place_on_surface(...) or place_on_face(...) "
        "for new models.",
        DeprecationWarning,
        stacklevel=2,
    )
    (cmin, _cmax) = child_aabb
    (_pmin, pmax) = parent_aabb
    dz = float(pmax[2]) + float(clearance) - float(cmin[2])
    dx = 0.0
    dy = 0.0
    if align_xy:
        cx, cy, _ = _aabb_center(child_aabb)
        px, py, _ = _aabb_center(parent_aabb)
        dx = px - cx
        dy = py - cy
    return Origin(xyz=(dx, dy, dz))


def place_in_front_of(
    child_aabb: AABB,
    parent_aabb: AABB,
    *,
    gap: float = 0.0,
    align_yz: bool = True,
) -> Origin:
    warnings.warn(
        "place_in_front_of(...) is deprecated; prefer place_on_surface(...) or place_on_face(...) "
        "for new models.",
        DeprecationWarning,
        stacklevel=2,
    )
    (cmin, _cmax) = child_aabb
    (_pmin, pmax) = parent_aabb
    dx = float(pmax[0]) + float(gap) - float(cmin[0])
    dy = 0.0
    dz = 0.0
    if align_yz:
        cx, cy, cz = _aabb_center(child_aabb)
        px, py, pz = _aabb_center(parent_aabb)
        dy = py - cy
        dz = pz - cz
    return Origin(xyz=(dx, dy, dz))


_FACE_TABLE: dict[str, tuple[int, float, Vec3]] = {
    "+x": (0, +1.0, (0.0, math.pi / 2.0, 0.0)),
    "-x": (0, -1.0, (0.0, -math.pi / 2.0, 0.0)),
    "+y": (1, +1.0, (-math.pi / 2.0, 0.0, 0.0)),
    "-y": (1, -1.0, (math.pi / 2.0, 0.0, 0.0)),
    "+z": (2, +1.0, (0.0, 0.0, 0.0)),
    "-z": (2, -1.0, (math.pi, 0.0, 0.0)),
}

_FACE_TANGENT_AXES: dict[str, tuple[int, int]] = {
    "+x": (1, 2),
    "-x": (1, 2),
    "+y": (0, 2),
    "-y": (0, 2),
    "+z": (0, 1),
    "-z": (0, 1),
}


def _selector_anchor_local_pose(
    selector: AnchorSelector, aabb: AABB
) -> tuple[RigidFrame, AnchorKind]:
    mn, mx = aabb
    center = _aabb_center(aabb)
    position = [center[0], center[1], center[2]]

    if selector.kind == "face":
        axis = selector.axes[0]
        axis_index = _axis_index(axis)
        sign = -1.0 if axis.startswith("-") else 1.0
        position[axis_index] = mx[axis_index] if sign > 0.0 else mn[axis_index]
        tangent_a, tangent_b = _FACE_TANGENT_AXES[axis]
        u, v = selector.uv or (0.5, 0.5)
        position[tangent_a] = mn[tangent_a] + (mx[tangent_a] - mn[tangent_a]) * u
        position[tangent_b] = mn[tangent_b] + (mx[tangent_b] - mn[tangent_b]) * v
        basis = RigidFrame.from_origin(Origin(rpy=_FACE_TABLE[axis][2])).basis
        return RigidFrame(position=tuple(position), basis=basis), "plane"  # type: ignore[arg-type]

    if selector.kind == "edge":
        first, second = selector.axes
        for axis in (first, second):
            axis_index = _axis_index(axis)
            position[axis_index] = mx[axis_index] if axis.startswith("+") else mn[axis_index]
        remaining = ({0, 1, 2} - {_axis_index(first), _axis_index(second)}).pop()
        t = selector.t if selector.t is not None else 0.5
        position[remaining] = mn[remaining] + (mx[remaining] - mn[remaining]) * t
        first_normal = _SIGNED_AXIS_VECTORS[first]
        second_normal = _SIGNED_AXIS_VECTORS[second]
        line_direction = _normalize(_cross(first_normal, second_normal), name="edge direction")
        return (
            RigidFrame(
                position=tuple(position),  # type: ignore[arg-type]
                basis=(first_normal, second_normal, line_direction),
            ),
            "axis",
        )

    for axis in selector.axes:
        axis_index = _axis_index(axis)
        position[axis_index] = mx[axis_index] if axis.startswith("+") else mn[axis_index]
    return RigidFrame(position=tuple(position)), "point"  # type: ignore[arg-type]


def anchor(
    subject: SurfaceSubject,
    selector: AnchorSelector,
    *,
    frame: Optional[FrameRef] = None,
    asset_root: Optional[Union[str, Path, AssetContext]] = None,
    prefer_collisions: bool = False,
) -> Anchor:

    resolved_frame = frame or FrameRef.part_local(subject)
    if isinstance(subject, Visual):
        geometry = subject.geometry
        if isinstance(geometry, Mesh):
            raise ValidationError(
                "Unannotated Mesh does not support analytic face/edge/corner selectors"
            )
        if not isinstance(geometry, (Box, Cylinder, Sphere)):
            raise ValidationError(
                f"{type(geometry).__name__} does not support analytic face/edge/corner selectors"
            )
        root = _resolve_asset_root_for_subject(asset_root, subject)
        local_aabb = _geometry_local_aabb(geometry, asset_root=root, _obj_cache={})
        subject_pose = RigidFrame.from_origin(subject.origin)
    elif isinstance(subject, Part):
        root = _resolve_asset_root_for_subject(asset_root, subject)
        local_aabb = _subject_world_aabb(
            subject,
            asset_root=root,
            prefer_collisions=prefer_collisions,
        )
        subject_pose = RigidFrame.identity()
    elif isinstance(subject, (Box, Cylinder, Sphere)):
        local_aabb = _geometry_local_aabb(subject, asset_root=None, _obj_cache={})
        subject_pose = RigidFrame.identity()
    else:
        raise ValidationError(
            f"{type(subject).__name__} does not support analytic face/edge/corner selectors"
        )

    if selector.kind in {"edge", "corner"} and not isinstance(
        subject.geometry if isinstance(subject, Visual) else subject,
        (Box, Part),
    ):
        raise ValidationError(
            f"{type(subject).__name__} does not support {selector.kind} selectors"
        )
    if selector.kind == "face" and not isinstance(
        subject.geometry if isinstance(subject, Visual) else subject,
        (Box, Part),
    ):
        axis = selector.axes[0]
        if isinstance(subject.geometry if isinstance(subject, Visual) else subject, Cylinder):
            if axis not in {"+z", "-z"}:
                raise ValidationError("Cylinder face selectors support only +z and -z")
        elif selector.uv != (0.5, 0.5):
            raise ValidationError("Curved analytic face selectors require uv=(0.5, 0.5)")

    local_pose, kind = _selector_anchor_local_pose(selector, local_aabb)
    absolute_pose = _compose_rigid_frames(subject_pose, local_pose)
    return Anchor(
        pose=absolute_pose,
        frame=resolved_frame,
        kind=kind,
        subject=subject,
        subject_pose=subject_pose,
        local_pose=local_pose,
        source=_anchor_source(
            subject,
            (
                f"part_bounds_{selector.kind}"
                if isinstance(subject, Part)
                else f"analytic_{selector.kind}"
            ),  # type: ignore[arg-type]
            local_pose=local_pose,
            selector=selector,
        ),
    )


def _normalize_axis_map(axis_map: Optional[Mapping[str, str]]) -> tuple[tuple[str, str], ...]:
    if axis_map is None:
        return (("+x", "+x"), ("+y", "+y"), ("+z", "+z"))
    if not axis_map:
        raise ValidationError("axis_map must contain at least one axis mapping")
    mappings = tuple(
        (
            _signed_axis_key(source, name="axis_map source"),
            _signed_axis_key(target, name="axis_map target"),
        )
        for source, target in axis_map.items()
    )
    source_indices = [_axis_index(source) for source, _ in mappings]
    target_indices = [_axis_index(target) for _, target in mappings]
    if len(set(source_indices)) != len(source_indices):
        raise ValidationError("axis_map contains collinear subject axes")
    if len(set(target_indices)) != len(target_indices):
        raise ValidationError("axis_map contains collinear target axes")
    return mappings


def _rotation_between(source: Vec3, target: Vec3) -> Mat3:
    source_unit = _normalize(source, name="source axis")
    target_unit = _normalize(target, name="target axis")
    cosine = _clamp(_dot(source_unit, target_unit), -1.0, 1.0)
    if cosine >= 1.0 - 1e-12:
        return _mat3_from_columns(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    cross = _cross(source_unit, target_unit)
    if cosine <= -1.0 + 1e-12:
        candidate = min(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            key=lambda axis: abs(_dot(source_unit, axis)),
        )
        rotation_axis = _normalize(_cross(source_unit, candidate), name="antiparallel axis")
        return _axis_angle_matrix(rotation_axis, math.pi)
    rotation_axis = _normalize(cross, name="rotation axis")
    return _axis_angle_matrix(rotation_axis, math.acos(cosine))


def _rotation_from_axis_correspondences(
    subject_anchor: Anchor,
    target_anchor: Anchor,
    mappings: tuple[tuple[str, str], ...],
) -> Mat3:
    first_source = subject_anchor.local_pose.axis(mappings[0][0])
    second_source = subject_anchor.local_pose.axis(mappings[1][0])
    first_target = target_anchor.pose.axis(mappings[0][1])
    second_target = target_anchor.pose.axis(mappings[1][1])
    source_basis = _mat3_from_columns(
        first_source,
        second_source,
        _normalize(_cross(first_source, second_source), name="subject mapped basis"),
    )
    target_basis = _mat3_from_columns(
        first_target,
        second_target,
        _normalize(_cross(first_target, second_target), name="target mapped basis"),
    )
    rotation = _mat3_mul(target_basis, _mat3_transpose(source_basis))
    for source_axis, target_axis in mappings[2:]:
        actual = _mat3_vec3(rotation, subject_anchor.local_pose.axis(source_axis))
        expected = target_anchor.pose.axis(target_axis)
        if _dot(actual, expected) < 1.0 - 1e-7:
            raise ValidationError("axis_map contains contradictory axis mappings")
    return rotation


def attach(
    subject_anchor: Anchor,
    target_anchor: Anchor,
    *,
    axis_map: Optional[Mapping[str, str]] = None,
    translate: Vec3 = (0.0, 0.0, 0.0),
    clearance: float = 0.0,
    uv_offset: tuple[float, float] = (0.0, 0.0),
    from_placement: Optional[PlacementResult] = None,
) -> PlacementResult:

    if not subject_anchor.frame.matches(target_anchor.frame):
        raise ValidationError("subject and target Anchors must use the same consumption frame")
    base_pose = subject_anchor.subject_pose
    if from_placement is not None:
        if from_placement.subject is not subject_anchor.subject:
            raise ValidationError("from_placement subject does not match the subject Anchor")
        if not from_placement.frame.matches(subject_anchor.frame):
            raise ValidationError("from_placement frame does not match the subject Anchor frame")
        base_pose = from_placement.pose

    point_translation_only = axis_map is None and (
        subject_anchor.kind == "point" or target_anchor.kind == "point"
    )
    if axis_map is not None and (subject_anchor.kind == "point" or target_anchor.kind == "point"):
        raise ValidationError("point Anchors do not define axes for axis_map")
    mappings = _normalize_axis_map(axis_map)
    if point_translation_only:
        rotation = _mat3_from_columns(*base_pose.basis)
        rotation_axes = ()
    elif len(mappings) == 1:
        current_anchor = _compose_rigid_frames(base_pose, subject_anchor.local_pose)
        adjustment = _rotation_between(
            current_anchor.axis(mappings[0][0]),
            target_anchor.pose.axis(mappings[0][1]),
        )
        rotation = _mat3_mul(adjustment, _mat3_from_columns(*base_pose.basis))
        target_axis = target_anchor.pose.axis(mappings[0][1])
        rotation_axes = tuple(
            axis for axis in target_anchor.pose.basis if abs(_dot(axis, target_axis)) < 0.5
        )
    else:
        rotation = _rotation_from_axis_correspondences(subject_anchor, target_anchor, mappings)
        rotation_axes = target_anchor.pose.basis

    frame_translate = _as_vec3(translate, name="translate")
    u, v = _as_pair(uv_offset, name="uv_offset")
    local_offset = (
        frame_translate[0] + u,
        frame_translate[1] + v,
        frame_translate[2] + float(clearance),
    )
    target_offset = _mat3_vec3(_mat3_from_columns(*target_anchor.pose.basis), local_offset)
    desired_anchor_position = tuple(
        target_anchor.pose.position[index] + target_offset[index] for index in range(3)
    )
    rotated_local_anchor = _mat3_vec3(rotation, subject_anchor.local_pose.position)
    subject_position = tuple(
        desired_anchor_position[index] - rotated_local_anchor[index] for index in range(3)
    )
    pose = RigidFrame(
        position=subject_position,  # type: ignore[arg-type]
        basis=_mat3_columns(rotation),
    )
    orientation_measurement: Optional[dict[str, object]] = None
    face_sources = {"analytic_face", "cadquery_face"}
    if (
        subject_anchor.source is not None
        and target_anchor.source is not None
        and subject_anchor.source.source_kind in face_sources
        and target_anchor.source.source_kind in face_sources
    ):
        resolved_subject_anchor = _compose_rigid_frames(pose, subject_anchor.local_pose)
        subject_normal = _normalize(
            resolved_subject_anchor.axis("+z"), name="subject mating normal"
        )
        target_normal = _normalize(target_anchor.pose.axis("+z"), name="target mating normal")
        normal_dot = _dot(subject_normal, target_normal)
        if normal_dot > PARALLEL_FACE_DOT_THRESHOLD:
            relative = _relative_rigid_frame(target_anchor.pose, resolved_subject_anchor)
            orientation_measurement = {
                "subject_normal": list(subject_normal),
                "target_normal": list(target_normal),
                "normal_dot": normal_dot,
                "commanded_axis_map": (dict(axis_map) if axis_map is not None else None),
                "resolved_relative_basis": [list(axis) for axis in relative.basis],
                "resolved_relative_rpy": list(relative.to_origin().rpy),
            }
    return PlacementResult(
        subject=subject_anchor.subject,
        frame=subject_anchor.frame,
        pose=pose,
        write_subspace=ControlledSubspace(
            frame=subject_anchor.frame,
            translation_axes=target_anchor.pose.basis,
            rotation_axes=rotation_axes,
        ),
        provenance_steps=(from_placement.provenance_steps if from_placement else ())
        + (
            PlacementStep(
                operation="attach",
                subject_source=subject_anchor.source,
                host_sources=(target_anchor.source,),
                commanded_clearance=float(clearance),
                commanded_axis_map=(
                    tuple((str(key), str(value)) for key, value in axis_map.items())
                    if axis_map is not None
                    else None
                ),
                effective_axis_map=mappings,
                orientation_measurement=orientation_measurement,
            ),
        ),  # type: ignore[arg-type]
    )


def _operation_frame(subject: object, frame: Optional[FrameRef]) -> FrameRef:
    return frame or FrameRef.part_local(subject)


def _operation_base_pose(
    subject: object,
    frame: FrameRef,
    from_placement: Optional[PlacementResult],
) -> RigidFrame:
    if from_placement is not None:
        if from_placement.subject is not subject:
            raise ValidationError("from_placement subject does not match the operation subject")
        if not from_placement.frame.matches(frame):
            raise ValidationError("from_placement frame does not match the operation frame")
        return from_placement.pose
    if isinstance(subject, Visual):
        return RigidFrame.from_origin(subject.origin)
    return RigidFrame.identity()


def _direction_in_consumption_frame(direction: Union[str, Vec3]) -> Vec3:
    if isinstance(direction, str):
        return _axis_vector(direction)
    return _normalize(_as_vec3(direction, name="direction"), name="direction")


def gap(
    start: Anchor,
    end: Anchor,
    *,
    axis: Optional[Union[str, Vec3]] = None,
) -> float:

    if not start.frame.matches(end.frame):
        raise ValidationError("gap Anchors must use the same consumption frame")
    delta = tuple(end.pose.position[index] - start.pose.position[index] for index in range(3))
    if axis is None:
        return _norm(delta)  # type: ignore[arg-type]
    direction = (
        start.pose.axis(axis) if isinstance(axis, str) else _normalize(axis, name="gap axis")
    )
    return abs(_dot(delta, direction))  # type: ignore[arg-type]


def center_between(
    subject: object,
    start: Anchor,
    end: Anchor,
    *,
    axis: Optional[Union[str, Vec3]] = None,
    frame: Optional[FrameRef] = None,
    from_placement: Optional[PlacementResult] = None,
) -> PlacementResult:

    if not start.frame.matches(end.frame):
        raise ValidationError("center_between Anchors must use the same consumption frame")
    resolved_frame = frame or start.frame
    if not resolved_frame.matches(start.frame):
        raise ValidationError("center_between frame must match both Anchor frames")
    base_pose = _operation_base_pose(subject, resolved_frame, from_placement)
    delta = tuple(end.pose.position[index] - start.pose.position[index] for index in range(3))
    if axis is None:
        direction = _normalize(delta, name="center_between Anchor separation")  # type: ignore[arg-type]
    elif isinstance(axis, str):
        direction = start.pose.axis(axis)
    else:
        direction = _normalize(_as_vec3(axis, name="axis"), name="axis")
    midpoint = tuple(
        (start.pose.position[index] + end.pose.position[index]) * 0.5 for index in range(3)
    )
    correction = _dot(
        tuple(midpoint[index] - base_pose.position[index] for index in range(3)),  # type: ignore[arg-type]
        direction,
    )
    position = tuple(
        base_pose.position[index] + direction[index] * correction for index in range(3)
    )
    return PlacementResult(
        subject=subject,
        frame=resolved_frame,
        pose=RigidFrame(position=position, basis=base_pose.basis),  # type: ignore[arg-type]
        write_subspace=ControlledSubspace(
            frame=resolved_frame,
            translation_axes=(direction,),
        ),
        provenance_steps=(from_placement.provenance_steps if from_placement else ())
        + (
            PlacementStep(
                operation="center_between",
                subject_source=_anchor_source(
                    subject,
                    "explicit",
                    local_pose=base_pose,
                ),
                host_sources=(start.source, end.source),
            ),
        ),  # type: ignore[arg-type]
    )


def _subject_items_at_pose(
    subject: object,
    pose: RigidFrame,
    *,
    prefer_collisions: bool,
) -> list[tuple[Geometry, Mat4]]:
    pose_transform = _origin_to_mat4(pose.to_origin())
    if isinstance(subject, Visual):
        return [(subject.geometry, pose_transform)]
    if isinstance(subject, Part):
        return [
            (geometry, _mat4_mul(pose_transform, local_transform))
            for geometry, local_transform in _iter_subject_items(
                subject,
                prefer_collisions=prefer_collisions,
            )
        ]
    if isinstance(subject, (Box, Cylinder, Sphere, Mesh)):
        return [(subject, pose_transform)]
    raise ValidationError(f"Unsupported snap subject type: {type(subject).__name__}")


def _projection_bounds_for_items(
    items: Iterable[tuple[Geometry, Mat4]],
    *,
    direction: Vec3,
    asset_root: Optional[Path],
) -> tuple[float, float]:
    minimum = float("inf")
    maximum = float("-inf")
    opposite = (-direction[0], -direction[1], -direction[2])
    for geometry, transform in items:
        minimum = min(
            minimum,
            _support_min_projection_on_element(
                geometry,
                transform,
                axis=direction,
                asset_root=asset_root,
            ),
        )
        maximum = max(
            maximum,
            -_support_min_projection_on_element(
                geometry,
                transform,
                axis=opposite,
                asset_root=asset_root,
            ),
        )
    if minimum == float("inf") or maximum == float("-inf"):
        raise ValidationError("snap subject or target has no geometry")
    return minimum, maximum


def _snap_target_name(target: object, index: int) -> str:
    name = getattr(target, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return f"{type(target).__name__}_{index}"


def snap(
    subject: object,
    *,
    against: Iterable[SurfaceSubject],
    direction: Union[str, Vec3],
    until: Literal["gap"] = "gap",
    gap: float = 0.0,
    max_travel: Optional[float] = None,
    frame: Optional[FrameRef] = None,
    from_placement: Optional[PlacementResult] = None,
    asset_root: Optional[Union[str, Path, AssetContext]] = None,
    prefer_collisions: bool = False,
) -> PlacementResult:

    if until != "gap":
        raise ValidationError("snap until must be 'gap'")
    requested_gap = float(gap)
    if requested_gap < 0.0:
        raise ValidationError("snap gap must be non-negative")
    if max_travel is not None and float(max_travel) < 0.0:
        raise ValidationError("snap max_travel must be non-negative")
    resolved_frame = _operation_frame(subject, frame)
    base_pose = _operation_base_pose(subject, resolved_frame, from_placement)
    travel_direction = _direction_in_consumption_frame(direction)
    subject_root = (
        _resolve_asset_root_for_subject(asset_root, subject)
        if isinstance(subject, (Part, Visual, Box, Cylinder, Sphere, Mesh))
        else None
    )
    subject_items = _subject_items_at_pose(
        subject,
        base_pose,
        prefer_collisions=prefer_collisions,
    )
    _, subject_maximum = _projection_bounds_for_items(
        subject_items,
        direction=travel_direction,
        asset_root=subject_root,
    )
    lateral_u, lateral_v = _build_surface_tangents(
        travel_direction,
        (0.0, 0.0, 1.0),
    )
    subject_lateral = (
        _projection_bounds_for_items(
            subject_items,
            direction=lateral_u,
            asset_root=subject_root,
        ),
        _projection_bounds_for_items(
            subject_items,
            direction=lateral_v,
            asset_root=subject_root,
        ),
    )

    candidates: list[tuple[float, str, SurfaceSubject]] = []
    for index, target in enumerate(against):
        target_root = _resolve_asset_root_for_subject(asset_root, target)
        target_items = _iter_subject_items(target, prefer_collisions=prefer_collisions)
        target_minimum, _ = _projection_bounds_for_items(
            target_items,
            direction=travel_direction,
            asset_root=target_root,
        )
        target_lateral = (
            _projection_bounds_for_items(
                target_items,
                direction=lateral_u,
                asset_root=target_root,
            ),
            _projection_bounds_for_items(
                target_items,
                direction=lateral_v,
                asset_root=target_root,
            ),
        )
        if any(
            subject_bounds[1] < target_bounds[0] - 1e-9
            or target_bounds[1] < subject_bounds[0] - 1e-9
            for subject_bounds, target_bounds in zip(subject_lateral, target_lateral)
        ):
            continue
        required = target_minimum - requested_gap - subject_maximum
        if required >= -1e-9:
            candidates.append((max(0.0, required), _snap_target_name(target, index), target))
    if not candidates:
        raise ValidationError("snap blocked: no target lies ahead in the requested direction")
    candidates.sort(key=lambda item: (item[0], item[1]))
    travel, target_name, _target = candidates[0]
    ambiguous = [name for candidate, name, _ in candidates if abs(candidate - travel) <= 1e-9]
    if len(ambiguous) > 1:
        raise ValidationError(f"snap ambiguity: equally near targets {', '.join(ambiguous)}")
    if max_travel is not None and travel > float(max_travel) + 1e-12:
        achieved = float(max_travel)
        deficit = travel - achieved
        raise ValidationError(
            f"snap blocked_by={target_name} achieved={achieved:.6g} deficit={deficit:.6g}"
        )
    position = tuple(
        base_pose.position[index] + travel_direction[index] * travel for index in range(3)
    )
    return PlacementResult(
        subject=subject,
        frame=resolved_frame,
        pose=RigidFrame(position=position, basis=base_pose.basis),  # type: ignore[arg-type]
        write_subspace=ControlledSubspace(
            frame=resolved_frame,
            translation_axes=(travel_direction,),
        ),
        provenance_steps=(from_placement.provenance_steps if from_placement else ())
        + (
            PlacementStep(
                operation="snap",
                subject_source=_anchor_source(
                    subject,
                    "bounds",
                    local_pose=base_pose,
                ),
                host_sources=(
                    _anchor_source(
                        _target,
                        "bounds",
                        local_pose=RigidFrame.identity(),
                    ),
                ),
                commanded_clearance=requested_gap,
            ),
        ),  # type: ignore[arg-type]
    )


def _bind_placement_target(
    model: object,
    target: PlacementTarget,
    result: PlacementResult,
) -> None:

    parts = list(getattr(model, "parts", ()))
    if target.kind == "visual":
        visual = target.target
        if not isinstance(visual, Visual):
            raise ValidationError("PlacementTarget.visual requires a Visual")
        owners = [part for part in parts if any(item is visual for item in part.visuals)]
        if len(owners) != 1:
            raise ValidationError("bound Visual must belong to exactly one model Part")
        part = owners[0]
        expected_frame = FrameRef.part_local(part)
        if result.subject is not visual:
            raise ValidationError("placement result subject does not match the bound Visual")
        if not result.frame.matches(expected_frame):
            raise ValidationError(
                "placement result frame must be the Visual's owning part-local frame"
            )
        visual.origin = result.origin
        return

    joint = target.target
    if not isinstance(joint, Articulation):
        raise ValidationError("PlacementTarget.joint requires an Articulation")
    if not any(candidate is joint for candidate in getattr(model, "articulations", ())):
        raise ValidationError("bound joint must belong to the model")
    parent_candidates = [part for part in parts if part.name == joint.parent]
    child_candidates = [part for part in parts if part.name == joint.child]
    if len(parent_candidates) != 1 or len(child_candidates) != 1:
        raise ValidationError("bound joint must reference unique parent and child Parts")
    expected_frame = FrameRef.parent_link_local(parent_candidates[0])
    if result.subject is not joint and result.subject is not child_candidates[0]:
        raise ValidationError("placement result subject must be the bound joint or its child Part")
    if not result.frame.matches(expected_frame):
        raise ValidationError("placement result frame must be the joint parent-link-local frame")
    joint.origin = result.origin


def _model_source_owner_identity(
    model: object, source: AnchorSource
) -> tuple[AnchorOwnerKind, str]:
    subject = source.subject
    parts = list(getattr(model, "parts", ()))
    if isinstance(subject, Visual):
        owners = [part for part in parts if any(item is subject for item in part.visuals)]
        if len(owners) == 1 and subject.visual_uid is not None:
            return "visual", f"visual:{owners[0].name}:{subject.visual_uid}"
    if isinstance(subject, Part) and any(part is subject for part in parts):
        return "part", f"part:{subject.name}"
    if isinstance(subject, Articulation) and any(
        joint is subject for joint in getattr(model, "articulations", ())
    ):
        return "joint", f"joint:{subject.name}"
    if source.owner_kind in {"visual", "part", "joint"}:
        raise ValidationError(
            "Anchor source claims a model-owned identity but its subject does not "
            "belong to the current model"
        )
    return source.owner_kind or "unregistered", source.owner_ref


def _model_source_payload(model: object, source: AnchorSource) -> dict[str, str]:
    owner_kind, owner_ref = _model_source_owner_identity(model, source)
    return source.to_dict(owner_kind=owner_kind, owner_ref=owner_ref)


def _placement_step_payload(model: object, step: PlacementStep) -> dict[str, object]:
    return {
        "operation": step.operation,
        "subject_source": _model_source_payload(model, step.subject_source),
        "host_sources": [_model_source_payload(model, source) for source in step.host_sources],
        "commanded_clearance": step.commanded_clearance,
        "commanded_axis_map": (
            dict(step.commanded_axis_map) if step.commanded_axis_map is not None else None
        ),
        "effective_axis_map": dict(step.effective_axis_map),
        "orientation_measurement": _thaw_provenance_value(step.orientation_measurement),
    }


def _placement_bind_record(
    model: object,
    target: PlacementTarget,
    result: PlacementResult,
    *,
    ordinal: int,
) -> dict[str, object]:
    if target.kind == "visual":
        visual = target.target
        owners = [
            part
            for part in getattr(model, "parts", ())
            if any(item is visual for item in part.visuals)
        ]
        if len(owners) != 1 or not isinstance(visual, Visual) or visual.visual_uid is None:
            raise ValidationError("bound Visual must have one stable model-owned UID")
        target_ref = f"visual:{owners[0].name}:{visual.visual_uid}"
        display_name = visual.name
    else:
        joint = target.target
        if not isinstance(joint, Articulation):
            raise ValidationError("bound joint must belong to the model")
        target_ref = f"joint:{joint.name}"
        display_name = joint.name
    return {
        "ordinal": ordinal,
        "target_ref": target_ref,
        "display_name": display_name,
        "final_pose": {
            "xyz": list(result.origin.xyz),
            "rpy": list(result.origin.rpy),
        },
        "provenance_steps": [
            _placement_step_payload(model, step) for step in result.provenance_steps
        ],
        "superseded": False,
    }


def place_on_face(
    parent_link: Part,
    face: str,
    *,
    face_pos: tuple[float, float] = (0.0, 0.0),
    proud: float = 0.0,
    asset_root: Optional[Union[str, Path, AssetContext]] = None,
    prefer_collisions: bool = True,
) -> Origin:
    face_key = face.strip().lower()
    if face_key not in _FACE_TABLE:
        raise ValidationError(
            f"Invalid face {face!r}; expected one of: {', '.join(sorted(_FACE_TABLE))}"
        )

    tang_a, tang_b = _FACE_TANGENT_AXES[face_key]
    face_pos_pair = _as_pair(face_pos, name="face_pos")

    face_anchor = anchor(
        parent_link,
        AnchorSelector.face(face_key),
        frame=FrameRef.part_local(parent_link),
        asset_root=asset_root,
        prefer_collisions=prefer_collisions,
    )
    xyz = list(face_anchor.pose.position)
    xyz[tang_a] = face_pos_pair[0]
    xyz[tang_b] = face_pos_pair[1]
    positioned = Anchor(
        pose=RigidFrame(position=(xyz[0], xyz[1], xyz[2]), basis=face_anchor.pose.basis),
        frame=face_anchor.frame,
        kind=face_anchor.kind,
        subject=face_anchor.subject,
        subject_pose=face_anchor.subject_pose,
        local_pose=RigidFrame(position=(xyz[0], xyz[1], xyz[2]), basis=face_anchor.pose.basis),
        source=face_anchor.source,
    ).offset(clearance=proud)
    return positioned.pose.to_origin()


def place_on_face_uv(
    parent_link: Part,
    face: str,
    *,
    uv: tuple[float, float] = (0.5, 0.5),
    uv_margin: float | tuple[float, float] = 0.0,
    proud: float = 0.0,
    asset_root: Optional[Union[str, Path, AssetContext]] = None,
    prefer_collisions: bool = True,
) -> Origin:
    face_key = face.strip().lower()
    if face_key not in _FACE_TABLE:
        raise ValidationError(
            f"Invalid face {face!r}; expected one of: {', '.join(sorted(_FACE_TABLE))}"
        )

    if isinstance(uv_margin, tuple):
        mu, mv = _as_pair(uv_margin, name="uv_margin")
    else:
        mu = mv = float(uv_margin)
    mu = max(0.0, min(0.49, mu))
    mv = max(0.0, min(0.49, mv))

    u_raw, v_raw = _as_pair(uv, name="uv")
    u = max(0.0, min(1.0, u_raw))
    v = max(0.0, min(1.0, v_raw))
    u = mu + (1.0 - 2.0 * mu) * u
    v = mv + (1.0 - 2.0 * mv) * v

    face_anchor = anchor(
        parent_link,
        AnchorSelector.face(face_key, uv=(u, v)),
        frame=FrameRef.part_local(parent_link),
        asset_root=asset_root,
        prefer_collisions=prefer_collisions,
    )
    return face_anchor.offset(clearance=proud).pose.to_origin()


def _axis_vector(axis: str) -> Vec3:
    axis_key = axis.strip().lower()
    table = {
        "+x": (1.0, 0.0, 0.0),
        "-x": (-1.0, 0.0, 0.0),
        "+y": (0.0, 1.0, 0.0),
        "-y": (0.0, -1.0, 0.0),
        "+z": (0.0, 0.0, 1.0),
        "-z": (0.0, 0.0, -1.0),
    }
    if axis_key not in table:
        raise ValidationError(f"Invalid axis {axis!r}; expected one of: {', '.join(sorted(table))}")
    return table[axis_key]


def _child_local_basis_for_axis(axis: str) -> Mat3:
    axis_key = axis.strip().lower()
    basis = {
        "+z": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        "-z": ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
        "+x": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        "-x": ((0.0, 1.0, 0.0), (0.0, 0.0, -1.0), (-1.0, 0.0, 0.0)),
        "+y": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        "-y": ((0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
    }
    if axis_key not in basis:
        raise ValidationError(f"Invalid axis {axis!r}; expected one of: {', '.join(sorted(basis))}")
    return basis[axis_key]


def _build_surface_tangents(normal: Vec3, up_hint: Vec3) -> tuple[Vec3, Vec3]:
    up = _normalize(up_hint, name="up_hint")
    projected = (
        up[0] - normal[0] * _dot(up, normal),
        up[1] - normal[1] * _dot(up, normal),
        up[2] - normal[2] * _dot(up, normal),
    )
    if _norm(projected) <= _EPS:
        fallback = (1.0, 0.0, 0.0)
        if abs(_dot(fallback, normal)) >= 0.95:
            fallback = (0.0, 1.0, 0.0)
        projected = _cross(fallback, normal)
    tangent_u = _normalize(projected, name="surface tangent")
    tangent_v = _normalize(_cross(normal, tangent_u), name="surface bitangent")
    return tangent_u, tangent_v


def _rotation_for_surface_frame(
    frame: SurfaceFrame,
    *,
    child_axis: str,
    spin: float,
) -> Mat3:
    local_basis = _child_local_basis_for_axis(child_axis)
    world_basis = _mat3_from_columns(frame.tangent_u, frame.tangent_v, frame.normal)
    local_basis_mat = _mat3_from_columns(
        local_basis[0],
        local_basis[1],
        local_basis[2],
    )
    base = _mat3_mul(world_basis, _mat3_transpose(local_basis_mat))
    if abs(float(spin)) <= _EPS:
        return base
    spin_world = _axis_angle_matrix(frame.normal, float(spin))
    return _mat3_mul(spin_world, base)


def _resolve_mesh_path(mesh: Mesh, asset_root: Optional[Path]) -> Path:
    return resolve_mesh_path(mesh.filename, assets=asset_root)


def _mesh_cache_fingerprint(mesh_path: Path) -> tuple[int, int, int]:
    try:
        stat = mesh_path.stat()
        content_crc = zlib.crc32(mesh_path.read_bytes())
    except FileNotFoundError as exc:
        raise ValidationError(f"Mesh file not found: {mesh_path}") from exc
    return int(stat.st_mtime_ns), int(stat.st_size), content_crc


def _load_trimesh_mesh(mesh: Mesh, *, asset_root: Optional[Path]):
    mesh_path = _resolve_mesh_path(mesh, asset_root)
    fingerprint = _mesh_cache_fingerprint(mesh_path)
    cached_entry = _TRIMESH_CACHE.get(mesh_path)
    cached = None
    if cached_entry is not None and cached_entry[0] == fingerprint:
        cached = cached_entry[1]

    if cached is None:
        try:
            import trimesh
        except Exception as exc:  # pragma: no cover - import failure is environment-specific
            raise RuntimeError(
                "Surface mesh placement requires the optional 'trimesh' dependency"
            ) from exc
        loaded = trimesh.load_mesh(mesh_path, force="mesh")
        if not isinstance(loaded, trimesh.Trimesh):
            raise ValidationError(
                f"Expected mesh geometry at {mesh_path}, got {type(loaded).__name__}"
            )
        _TRIMESH_CACHE[mesh_path] = (fingerprint, loaded)
        cached = loaded
    if mesh.scale is None:
        return cached
    scaled = cached.copy()
    scaled.apply_scale(mesh.scale)
    return scaled


def _closest_point_on_box(point: Vec3, size: Vec3) -> tuple[Vec3, Vec3]:
    hx, hy, hz = float(size[0]) * 0.5, float(size[1]) * 0.5, float(size[2]) * 0.5
    x, y, z = point
    clamped = (
        _clamp(x, -hx, hx),
        _clamp(y, -hy, hy),
        _clamp(z, -hz, hz),
    )
    inside = abs(x) <= hx and abs(y) <= hy and abs(z) <= hz
    if inside:
        face_dist = (hx - abs(x), hy - abs(y), hz - abs(z))
        axis = min(range(3), key=lambda idx: face_dist[idx])
        sign = 1.0 if point[axis] >= 0.0 else -1.0
        nearest = [x, y, z]
        nearest[axis] = (hx, hy, hz)[axis] * sign
        normal = [0.0, 0.0, 0.0]
        normal[axis] = sign
        return (nearest[0], nearest[1], nearest[2]), (normal[0], normal[1], normal[2])

    dx = x - clamped[0]
    dy = y - clamped[1]
    dz = z - clamped[2]
    axis = max(range(3), key=lambda idx: abs((dx, dy, dz)[idx]))
    delta = (dx, dy, dz)[axis]
    sign = 1.0 if delta >= 0.0 else -1.0
    normal = [0.0, 0.0, 0.0]
    normal[axis] = sign
    return clamped, (normal[0], normal[1], normal[2])


def _closest_point_on_cylinder(point: Vec3, radius: float, length: float) -> tuple[Vec3, Vec3]:
    x, y, z = point
    half = float(length) * 0.5
    radial = math.hypot(x, y)
    if radial <= _EPS:
        rx, ry = 1.0, 0.0
    else:
        rx, ry = x / radial, y / radial

    side = (rx * radius, ry * radius, _clamp(z, -half, half))
    side_d2 = (x - side[0]) ** 2 + (y - side[1]) ** 2 + (z - side[2]) ** 2

    def cap_candidate(sign: float) -> tuple[Vec3, float]:
        cap_z = sign * half
        if radial <= radius:
            cap_xy = (x, y)
        else:
            cap_xy = (rx * radius, ry * radius)
        cap = (cap_xy[0], cap_xy[1], cap_z)
        d2 = (x - cap[0]) ** 2 + (y - cap[1]) ** 2 + (z - cap[2]) ** 2
        return cap, d2

    top, top_d2 = cap_candidate(+1.0)
    bottom, bottom_d2 = cap_candidate(-1.0)
    if side_d2 <= top_d2 and side_d2 <= bottom_d2:
        return side, (rx, ry, 0.0)
    if top_d2 <= bottom_d2:
        return top, (0.0, 0.0, 1.0)
    return bottom, (0.0, 0.0, -1.0)


def _query_mesh_surface_local(
    mesh: Mesh, point_local: Vec3, *, asset_root: Optional[Path]
) -> _SurfaceHit:
    local_mesh = _load_trimesh_mesh(mesh, asset_root=asset_root)
    try:
        query = local_mesh.nearest.on_surface([point_local])
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on native environment
        if exc.name == "rtree":
            raise RuntimeError(
                "Mesh surface placement requires the optional 'rtree' dependency"
            ) from exc
        if exc.name == "scipy":
            raise RuntimeError(
                "Mesh surface placement requires the optional 'scipy' dependency"
            ) from exc
        raise
    closest = query[0][0]
    distance = float(query[1][0])
    face_index = int(query[2][0])
    normal = tuple(float(v) for v in local_mesh.face_normals[face_index])
    to_query = (
        float(point_local[0]) - float(closest[0]),
        float(point_local[1]) - float(closest[1]),
        float(point_local[2]) - float(closest[2]),
    )
    if _dot(normal, to_query) < 0.0:
        normal = (-normal[0], -normal[1], -normal[2])
    normal = _normalize(normal, name="mesh face normal")
    return _SurfaceHit(
        point=(float(closest[0]), float(closest[1]), float(closest[2])),
        normal=normal,
        distance=distance,
    )


def _query_surface_on_element(
    geometry: Geometry,
    tf: Mat4,
    *,
    point_hint: Vec3,
    asset_root: Optional[Path],
) -> _SurfaceHit:
    geometry, tf = _expand_mesh_provenance(geometry, tf)
    inv_tf = _mat4_inverse_rigid(tf)
    point_local = _mat4_vec3(inv_tf, point_hint)

    if isinstance(geometry, Sphere):
        radius = float(geometry.radius)
        direction = point_local
        if _norm(direction) <= _EPS:
            direction = (1.0, 0.0, 0.0)
        normal_local = _normalize(direction, name="sphere query direction")
        surface_local = (
            normal_local[0] * radius,
            normal_local[1] * radius,
            normal_local[2] * radius,
        )
        distance = _norm(
            (
                point_local[0] - surface_local[0],
                point_local[1] - surface_local[1],
                point_local[2] - surface_local[2],
            )
        )
    elif isinstance(geometry, Box):
        surface_local, normal_local = _closest_point_on_box(point_local, geometry.size)
        distance = _norm(
            (
                point_local[0] - surface_local[0],
                point_local[1] - surface_local[1],
                point_local[2] - surface_local[2],
            )
        )
    elif isinstance(geometry, Cylinder):
        surface_local, normal_local = _closest_point_on_cylinder(
            point_local,
            radius=float(geometry.radius),
            length=float(geometry.length),
        )
        distance = _norm(
            (
                point_local[0] - surface_local[0],
                point_local[1] - surface_local[1],
                point_local[2] - surface_local[2],
            )
        )
    elif isinstance(geometry, Mesh):
        hit = _query_mesh_surface_local(geometry, point_local, asset_root=asset_root)
        surface_local = hit.point
        normal_local = hit.normal
        distance = hit.distance
    else:
        raise ValidationError(f"Unsupported surface geometry type: {type(geometry).__name__}")

    point_world = _mat4_vec3(tf, surface_local)
    normal_world = _normalize(
        _mat4_transform_direction(tf, normal_local),
        name="surface normal",
    )
    return _SurfaceHit(point=point_world, normal=normal_world, distance=distance)


def _query_subject_surface_at_point(
    subject: SurfaceSubject,
    *,
    point_hint: Vec3,
    asset_root: Optional[Union[str, Path, AssetContext]],
    prefer_collisions: bool,
) -> _SurfaceHit:
    root = _resolve_asset_root_for_subject(asset_root, subject)
    hits: list[_SurfaceHit] = []
    for geometry, tf in _iter_subject_items(subject, prefer_collisions=prefer_collisions):
        hits.append(_query_surface_on_element(geometry, tf, point_hint=point_hint, asset_root=root))
    if not hits:
        raise ValidationError("target has no geometry to query")
    best_distance = min(hit.distance for hit in hits)
    selected = [hit for hit in hits if hit.distance - best_distance <= 1e-9]
    if (
        len(selected) > 1
        and any(
            _norm(
                tuple(hit.point[index] - selected[0].point[index] for index in range(3))  # type: ignore[arg-type]
            )
            > 1e-7
            or _dot(hit.normal, selected[0].normal) < 1.0 - 1e-7
            for hit in selected[1:]
        )
    ):
        raise ValidationError("ambiguity: equally near surface candidates")
    return selected[0]


def _query_subject_surface_in_direction(
    subject: SurfaceSubject,
    *,
    direction: Vec3,
    asset_root: Optional[Union[str, Path, AssetContext]],
    prefer_collisions: bool,
) -> _SurfaceHit:

    root = _resolve_asset_root_for_subject(asset_root, subject)
    direction_world = _normalize(direction, name="direction")
    hits: list[tuple[float, _SurfaceHit]] = []
    obj_cache: dict[Path, AABB] = {}
    for geometry, transform in _iter_subject_items(subject, prefer_collisions=prefer_collisions):
        geometry, transform = _expand_mesh_provenance(geometry, transform)
        local_aabb = _geometry_local_aabb(geometry, asset_root=root, _obj_cache=obj_cache)
        local_center = _aabb_center(local_aabb)
        local_radius = max(_norm(_aabb_size(local_aabb)), 1e-3)
        local_direction = _normalize(
            _mat4_transform_direction(_mat4_inverse_rigid(transform), direction_world),
            name="source-local direction",
        )
        local_query = (
            local_center[0] + local_direction[0] * local_radius * 4.0,
            local_center[1] + local_direction[1] * local_radius * 4.0,
            local_center[2] + local_direction[2] * local_radius * 4.0,
        )
        hit = _query_surface_on_element(
            geometry,
            transform,
            point_hint=_mat4_vec3(transform, local_query),
            asset_root=root,
        )
        projection = _dot(hit.point, direction_world)
        hits.append((projection, hit))
    if not hits:
        raise ValidationError("target has no geometry to query")
    best_projection = max(projection for projection, _hit in hits)
    selected = [hit for projection, hit in hits if best_projection - projection <= 1e-9]
    if (
        len(selected) > 1
        and any(
            _norm(
                tuple(hit.point[index] - selected[0].point[index] for index in range(3))  # type: ignore[arg-type]
            )
            > 1e-7
            or _dot(hit.normal, selected[0].normal) < 1.0 - 1e-7
            for hit in selected[1:]
        )
    ):
        raise ValidationError("ambiguity: equally exposed surface candidates")
    return selected[0]


def surface_frame(
    target: SurfaceSubject,
    *,
    point_hint: Optional[Vec3] = None,
    direction: Optional[Vec3] = None,
    asset_root: Optional[Union[str, Path, AssetContext]] = None,
    prefer_collisions: bool = False,
    up_hint: Vec3 = (0.0, 0.0, 1.0),
) -> SurfaceFrame:
    resolved = _surface_anchor(
        target,
        point_hint=point_hint,
        direction=direction,
        asset_root=asset_root,
        prefer_collisions=prefer_collisions,
        up_hint=up_hint,
    )
    return SurfaceFrame(
        point=resolved.pose.position,
        normal=resolved.pose.axis("+z"),
        tangent_u=resolved.pose.axis("+x"),
        tangent_v=resolved.pose.axis("+y"),
    )


def _surface_anchor(
    target: SurfaceSubject,
    *,
    point_hint: Optional[Vec3] = None,
    direction: Optional[Vec3] = None,
    asset_root: Optional[Union[str, Path, AssetContext]] = None,
    prefer_collisions: bool = False,
    up_hint: Vec3 = (0.0, 0.0, 1.0),
    frame: Optional[FrameRef] = None,
) -> Anchor:
    if (point_hint is None) == (direction is None):
        raise ValidationError("Exactly one of point_hint or direction must be provided")

    root = _resolve_asset_root_for_subject(asset_root, target)
    if direction is not None:
        best = _query_subject_surface_in_direction(
            target,
            direction=direction,
            asset_root=root,
            prefer_collisions=prefer_collisions,
        )
    else:
        assert point_hint is not None
        best = _query_subject_surface_at_point(
            target,
            point_hint=point_hint,
            asset_root=root,
            prefer_collisions=prefer_collisions,
        )

    tangent_u, tangent_v = _build_surface_tangents(best.normal, up_hint)
    pose = RigidFrame(
        position=best.point,
        basis=(tangent_u, tangent_v, best.normal),
    )
    subject_pose = (
        RigidFrame.from_origin(target.origin)
        if isinstance(target, Visual)
        else RigidFrame.identity()
    )
    return Anchor(
        pose=pose,
        frame=frame or FrameRef.part_local(target),
        kind="plane",
        subject=target,
        subject_pose=subject_pose,
        local_pose=_relative_rigid_frame(subject_pose, pose),
        source=_anchor_source(
            target,
            "surface",
            local_pose=_relative_rigid_frame(subject_pose, pose),
        ),
    )


def surface_anchor(
    target: SurfaceSubject,
    *,
    point_hint: Optional[Vec3] = None,
    direction: Optional[Vec3] = None,
    asset_root: Optional[Union[str, Path, AssetContext]] = None,
    prefer_collisions: bool = False,
    up_hint: Vec3 = (0.0, 0.0, 1.0),
    frame: Optional[FrameRef] = None,
) -> Anchor:

    return _surface_anchor(
        target,
        point_hint=point_hint,
        direction=direction,
        asset_root=asset_root,
        prefer_collisions=prefer_collisions,
        up_hint=up_hint,
        frame=frame,
    )


def _mesh_subject(
    subject: Union[Mesh, Visual],
) -> tuple[Mesh, RigidFrame]:
    if isinstance(subject, Mesh):
        return subject, RigidFrame.identity()
    if isinstance(subject, Visual) and isinstance(subject.geometry, Mesh):
        return subject.geometry, RigidFrame.from_origin(subject.origin)
    raise ValidationError("Mesh Anchor operations require a Mesh or a Visual containing a Mesh")


def _mesh_anchor_result(
    subject: Union[Mesh, Visual],
    *,
    local_pose: RigidFrame,
    kind: AnchorKind,
    frame: Optional[FrameRef],
    source_kind: AnchorSourceKind,
) -> Anchor:
    _mesh, subject_pose = _mesh_subject(subject)
    return Anchor(
        pose=_compose_rigid_frames(subject_pose, local_pose),
        frame=frame or FrameRef.part_local(subject),
        kind=kind,
        subject=subject,
        subject_pose=subject_pose,
        local_pose=local_pose,
        source=_anchor_source(subject, source_kind, local_pose=local_pose),
    )


def explicit_anchor(
    subject: object,
    local_pose: Union[RigidFrame, Origin],
    *,
    frame: FrameRef,
    kind: AnchorKind = "point",
    subject_pose: Optional[Union[RigidFrame, Origin]] = None,
    source_kind: AnchorSourceKind = "explicit",
    owner_ref: Optional[str] = None,
) -> Anchor:

    if kind not in {"point", "plane", "axis"}:
        raise ValidationError(f"Unsupported explicit Anchor kind: {kind!r}")
    resolved_local = (
        local_pose if isinstance(local_pose, RigidFrame) else RigidFrame.from_origin(local_pose)
    )
    if subject_pose is None:
        resolved_subject = (
            RigidFrame.from_origin(subject.origin)
            if isinstance(subject, Visual)
            else RigidFrame.identity()
        )
    else:
        resolved_subject = (
            subject_pose
            if isinstance(subject_pose, RigidFrame)
            else RigidFrame.from_origin(subject_pose)
        )
    return Anchor(
        pose=_compose_rigid_frames(resolved_subject, resolved_local),
        frame=frame,
        kind=kind,
        subject=subject,
        subject_pose=resolved_subject,
        local_pose=resolved_local,
        source=_anchor_source(
            subject,
            source_kind,
            local_pose=resolved_local,
            owner_ref=owner_ref,
        ),
    )


def bounds_anchor(
    subject: Union[Mesh, Visual],
    selector: AnchorSelector,
    *,
    frame: Optional[FrameRef] = None,
    asset_root: Optional[Union[str, Path, AssetContext]] = None,
) -> Anchor:

    mesh, _subject_pose = _mesh_subject(subject)
    root = _resolve_asset_root_for_subject(asset_root, subject)
    local_aabb = _geometry_local_aabb(mesh, asset_root=root, _obj_cache={})
    local_pose, kind = _selector_anchor_local_pose(selector, local_aabb)
    return _mesh_anchor_result(
        subject,
        local_pose=local_pose,
        kind=kind,
        frame=frame,
        source_kind="bounds",
    )


def _anchor_sidecar_path(
    mesh: Mesh,
    *,
    asset_root: Optional[Union[str, Path, AssetContext]],
) -> Path:
    sidecar = mesh.anchor_sidecar
    if sidecar is None:
        raise ValidationError("no-feature: Mesh has no semantic Anchor sidecar")
    path = Path(sidecar)
    if path.is_absolute():
        return path.resolve()
    root = resolve_asset_root(asset_root, mesh)
    if root is None:
        raise ValidationError("Mesh Anchor sidecar is relative but no asset_root is available")
    return (root / path).resolve()


def validate_mesh_anchor_sidecar(
    mesh_path: Union[str, Path],
    sidecar_path: Union[str, Path],
) -> dict[str, tuple[RigidFrame, AnchorKind]]:

    mesh_path = Path(mesh_path).resolve()
    sidecar_path = Path(sidecar_path).resolve()
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Invalid Mesh Anchor sidecar {sidecar_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "artisan-mesh-anchors-v1":
        raise ValidationError("Mesh Anchor sidecar has an unsupported schema_version")
    entries = payload.get("anchors")
    if not isinstance(entries, list):
        raise ValidationError("Mesh Anchor sidecar anchors must be a list")
    loaded_mesh = _load_trimesh_mesh(Mesh(filename=str(mesh_path)), asset_root=None)
    parsed: dict[str, tuple[RigidFrame, AnchorKind]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValidationError(f"Mesh Anchor sidecar anchors[{index}] must be an object")
        name = str(entry.get("name", "")).strip()
        if not re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", name):
            raise ValidationError(
                f"Mesh Anchor sidecar anchors[{index}] name must be stable snake_case"
            )
        if name in parsed:
            raise ValidationError(f"Mesh Anchor sidecar has duplicate Anchor name {name}")
        kind = str(entry.get("kind", "")).strip()
        if kind not in {"point", "plane", "axis"}:
            raise ValidationError(f"Mesh Anchor {name!r} has invalid kind {kind!r}")
        try:
            pose = RigidFrame(
                position=_as_vec3(entry.get("position"), name=f"Anchor {name!r} position"),
                basis=tuple(
                    _as_vec3(axis, name=f"Anchor {name!r} basis[{axis_index}]")
                    for axis_index, axis in enumerate(entry.get("basis", ()))
                ),  # type: ignore[arg-type]
            )
        except (TypeError, ValidationError) as exc:
            raise ValidationError(f"Invalid semantic Anchor {name!r}: {exc}") from exc
        surface_bound = entry.get("surface_bound")
        if not isinstance(surface_bound, bool):
            raise ValidationError(f"Mesh Anchor {name!r} surface_bound must be boolean")
        tolerance = float(entry.get("tolerance", 0.0))
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValidationError(f"Mesh Anchor {name!r} tolerance must be positive and finite")
        if kind == "axis" and surface_bound:
            raise ValidationError(f"axis Anchor {name!r} cannot be surface_bound")
        if surface_bound:
            _closest, distances, triangles = loaded_mesh.nearest.on_surface([pose.position])
            residual = float(distances[0])
            if residual > tolerance:
                raise ValidationError(
                    f"semantic Anchor {name!r} is off the final Mesh by {residual:.6g} m "
                    f"(tolerance={tolerance:.6g})"
                )
            if kind == "plane":
                normal = tuple(
                    float(value) for value in loaded_mesh.face_normals[int(triangles[0])]
                )
                if _dot(pose.axis("+z"), normal) < 0.9:  # type: ignore[arg-type]
                    raise ValidationError(
                        f"semantic plane Anchor {name!r} normal does not match the final Mesh"
                    )
        parsed[name] = (pose, kind)  # type: ignore[assignment]
    return parsed


def _axis_anchor_pose(position: Vec3, axis: Vec3, up: Vec3) -> RigidFrame:
    z_axis = _normalize(axis, name="semantic axis")
    projected_up = tuple(up[index] - z_axis[index] * _dot(up, z_axis) for index in range(3))
    if _norm(projected_up) <= _EPS:
        fallback = min(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            key=lambda candidate: abs(_dot(candidate, z_axis)),
        )
        projected_up = tuple(
            fallback[index] - z_axis[index] * _dot(fallback, z_axis) for index in range(3)
        )
    y_axis = _normalize(projected_up, name="semantic axis up")  # type: ignore[arg-type]
    x_axis = _normalize(_cross(y_axis, z_axis), name="semantic axis x")
    return RigidFrame(position=position, basis=(x_axis, y_axis, z_axis))


def semantic_anchor(
    subject: object,
    name: str,
    *,
    frame: Optional[FrameRef] = None,
    asset_root: Optional[Union[str, Path, AssetContext]] = None,
) -> Anchor:

    from ._mesh.hinges import HingePin
    from ._mesh.slides import SlideRail

    key = str(name).strip()
    if isinstance(subject, HingePin):
        if key != "pin_axis":
            raise ValidationError("no-feature: HingePin semantic Anchor; candidates: pin_axis")
        pose = _axis_anchor_pose(subject.anchor, subject.axis, (0.0, 1.0, 0.0))
        return explicit_anchor(
            subject,
            pose,
            frame=frame or FrameRef.shared_spec(subject),
            kind="axis",
            source_kind="semantic",
        )
    if isinstance(subject, SlideRail):
        if key != "runner_axis":
            raise ValidationError("no-feature: SlideRail semantic Anchor; candidates: runner_axis")
        pose = _axis_anchor_pose(subject.anchor, subject.slide_axis, subject.up)
        return explicit_anchor(
            subject,
            pose,
            frame=frame or FrameRef.shared_spec(subject),
            kind="axis",
            source_kind="semantic",
        )
    if not isinstance(subject, (Mesh, Visual)):
        raise ValidationError(
            f"{type(subject).__name__} does not expose semantic Anchors through this adapter"
        )

    mesh, _subject_pose = _mesh_subject(subject)
    if mesh.scale is not None:
        if max(mesh.scale) - min(mesh.scale) > 1e-9:
            raise ValidationError("semantic Mesh Anchors do not support non-uniform Mesh.scale")
        scale = float(mesh.scale[0])
    else:
        scale = 1.0
    root = _resolve_asset_root_for_subject(asset_root, subject)
    mesh_path = resolve_mesh_path(mesh.filename, assets=root)
    sidecar_path = _anchor_sidecar_path(mesh, asset_root=root)
    parsed = validate_mesh_anchor_sidecar(mesh_path, sidecar_path)
    if key not in parsed:
        candidates = ", ".join(sorted(parsed)) or "none"
        raise ValidationError(f"no-feature: semantic Anchor {key!r}; candidates: {candidates}")
    local_pose, kind = parsed[key]
    if scale != 1.0:
        local_pose = RigidFrame(
            position=tuple(component * scale for component in local_pose.position),  # type: ignore[arg-type]
            basis=local_pose.basis,
        )
    return _mesh_anchor_result(
        subject,
        local_pose=local_pose,
        kind=kind,
        frame=frame,
        source_kind="semantic",
    )


def _mesh_query_local_inputs(
    subject_pose: RigidFrame,
    *,
    direction: Optional[Vec3],
    point_hint: Optional[Vec3],
) -> tuple[Optional[Vec3], Optional[Vec3]]:
    inverse_rotation = _mat3_transpose(_mat3_from_columns(*subject_pose.basis))
    local_direction = (
        _normalize(_mat3_vec3(inverse_rotation, direction), name="source-local direction")
        if direction is not None
        else None
    )
    if point_hint is None:
        local_hint = None
    else:
        delta = tuple(point_hint[index] - subject_pose.position[index] for index in range(3))
        local_hint = _mat3_vec3(inverse_rotation, delta)  # type: ignore[arg-type]
    return local_direction, local_hint


def _select_unique_feature(
    candidates: list[tuple[RigidFrame, AnchorKind, str]],
    *,
    feature: str,
    direction: Optional[Vec3],
    point_hint: Optional[Vec3],
    tolerance: float,
) -> tuple[RigidFrame, AnchorKind]:
    if not candidates:
        raise ValidationError(f"no-feature: Mesh has no {feature}")
    if direction is not None:
        scored = [
            (_dot(candidate.position, direction), candidate, kind, label)
            for candidate, kind, label in candidates
        ]
        best_score = max(item[0] for item in scored)
        selected = [item for item in scored if best_score - item[0] <= tolerance]
    elif point_hint is not None:
        scored = [
            (
                _norm(
                    tuple(candidate.position[index] - point_hint[index] for index in range(3))  # type: ignore[arg-type]
                ),
                candidate,
                kind,
                label,
            )
            for candidate, kind, label in candidates
        ]
        best_score = min(item[0] for item in scored)
        selected = [item for item in scored if item[0] - best_score <= tolerance]
    else:
        selected = [(0.0, candidate, kind, label) for candidate, kind, label in candidates]
    if len(selected) != 1:
        labels = ", ".join(item[3] for item in selected[:12])
        raise ValidationError(f"ambiguity: Mesh {feature} candidates: {labels}")
    return selected[0][1], selected[0][2]


def mesh_feature_anchor(
    subject: Union[Mesh, Visual],
    feature: Literal["planar_patch", "feature_edge", "feature_corner"],
    *,
    direction: Optional[Vec3] = None,
    point_hint: Optional[Vec3] = None,
    frame: Optional[FrameRef] = None,
    asset_root: Optional[Union[str, Path, AssetContext]] = None,
    up_hint: Vec3 = (0.0, 0.0, 1.0),
    planar_normal_tolerance: float = 1e-4,
    feature_angle_degrees: float = 30.0,
    selection_tolerance: float = 1e-6,
) -> Anchor:

    if feature not in {"planar_patch", "feature_edge", "feature_corner"}:
        raise ValidationError(f"Unsupported Mesh feature selector: {feature!r}")
    if direction is not None and point_hint is not None:
        raise ValidationError("Mesh feature selector accepts only one of direction or point_hint")
    mesh, subject_pose = _mesh_subject(subject)
    root = _resolve_asset_root_for_subject(asset_root, subject)
    loaded = _load_trimesh_mesh(mesh, asset_root=root)
    local_direction, local_hint = _mesh_query_local_inputs(
        subject_pose,
        direction=direction,
        point_hint=point_hint,
    )
    inverse_rotation = _mat3_transpose(_mat3_from_columns(*subject_pose.basis))
    local_up = _normalize(_mat3_vec3(inverse_rotation, up_hint), name="source-local up_hint")
    candidates: list[tuple[RigidFrame, AnchorKind, str]] = []

    if feature == "planar_patch":
        rejected_hole = 0
        mesh_area = max(float(loaded.area), _EPS)
        for facet_index, face_ids in enumerate(loaded.facets):
            area = float(loaded.facets_area[facet_index])
            if area / mesh_area < 1e-4:
                continue
            normal = tuple(float(value) for value in loaded.facets_normal[facet_index])
            if (
                local_direction is not None
                and _dot(normal, local_direction) < 1.0 - planar_normal_tolerance
            ):
                continue
            face_indices = [int(value) for value in face_ids]
            weights = [float(loaded.area_faces[index]) for index in face_indices]
            weight_sum = sum(weights)
            centroid = tuple(
                sum(
                    float(loaded.triangles_center[face_index][axis]) * weight
                    for face_index, weight in zip(face_indices, weights)
                )
                / weight_sum
                for axis in range(3)
            )
            _closest, distances, _triangles = loaded.nearest.on_surface([centroid])
            if float(distances[0]) > selection_tolerance:
                rejected_hole += 1
                continue
            tangent_u, tangent_v = _build_surface_tangents(normal, local_up)  # type: ignore[arg-type]
            candidates.append(
                (
                    RigidFrame(
                        position=centroid,  # type: ignore[arg-type]
                        basis=(tangent_u, tangent_v, normal),  # type: ignore[arg-type]
                    ),
                    "plane",
                    f"facet_{facet_index}",
                )
            )
        if not candidates and rejected_hole:
            raise ValidationError("no-feature: planar patch centroid lies in a hole")

    elif feature == "feature_edge":
        threshold = math.radians(float(feature_angle_degrees))
        for edge_index, (edge, angle) in enumerate(
            zip(loaded.face_adjacency_edges, loaded.face_adjacency_angles)
        ):
            if float(angle) < threshold:
                continue
            first = tuple(float(value) for value in loaded.vertices[int(edge[0])])
            second = tuple(float(value) for value in loaded.vertices[int(edge[1])])
            line = _normalize(
                tuple(second[index] - first[index] for index in range(3)),  # type: ignore[arg-type]
                name="feature edge",
            )
            point = tuple((first[index] + second[index]) * 0.5 for index in range(3))
            tangent_u, tangent_v = _build_surface_tangents(line, local_up)
            candidates.append(
                (
                    RigidFrame(
                        position=point,  # type: ignore[arg-type]
                        basis=(tangent_u, tangent_v, line),
                    ),
                    "axis",
                    f"edge_{edge_index}",
                )
            )

    else:
        threshold = math.radians(float(feature_angle_degrees))
        degrees: dict[int, int] = {}
        for edge, angle in zip(loaded.face_adjacency_edges, loaded.face_adjacency_angles):
            if float(angle) < threshold:
                continue
            for vertex_index in (int(edge[0]), int(edge[1])):
                degrees[vertex_index] = degrees.get(vertex_index, 0) + 1
        for vertex_index, degree in sorted(degrees.items()):
            if degree < 3:
                continue
            point = tuple(float(value) for value in loaded.vertices[vertex_index])
            candidates.append((RigidFrame(position=point), "point", f"corner_{vertex_index}"))

    local_pose, kind = _select_unique_feature(
        candidates,
        feature=feature,
        direction=local_direction,
        point_hint=local_hint,
        tolerance=float(selection_tolerance),
    )
    return _mesh_anchor_result(
        subject,
        local_pose=local_pose,
        kind=kind,
        frame=frame,
        source_kind={
            "planar_patch": "mesh_planar_patch",
            "feature_edge": "mesh_feature_edge",
            "feature_corner": "mesh_feature_corner",
        }[feature],  # type: ignore[arg-type]
    )


def _support_min_projection_on_element(
    geometry: Geometry,
    tf: Mat4,
    *,
    axis: Vec3,
    asset_root: Optional[Path],
) -> float:
    geometry, tf = _expand_mesh_provenance(geometry, tf)
    center = (float(tf[0][3]), float(tf[1][3]), float(tf[2][3]))
    rot = _mat3_from_mat4(tf)
    axis_local = _mat3_vec3(_mat3_transpose(rot), axis)

    if isinstance(geometry, Sphere):
        return _dot(center, axis) - float(geometry.radius)
    if isinstance(geometry, Box):
        hx, hy, hz = (
            float(geometry.size[0]) * 0.5,
            float(geometry.size[1]) * 0.5,
            float(geometry.size[2]) * 0.5,
        )
        half = hx * abs(axis_local[0]) + hy * abs(axis_local[1]) + hz * abs(axis_local[2])
        return _dot(center, axis) - half
    if isinstance(geometry, Cylinder):
        radial = math.hypot(axis_local[0], axis_local[1])
        half = float(geometry.radius) * radial + (float(geometry.length) * 0.5) * abs(axis_local[2])
        return _dot(center, axis) - half
    if isinstance(geometry, Mesh):
        local_mesh = _load_trimesh_mesh(geometry, asset_root=asset_root)
        min_proj = float("inf")
        for vertex in local_mesh.vertices:
            local_vertex = (float(vertex[0]), float(vertex[1]), float(vertex[2]))
            world_vertex = _mat4_vec3(tf, local_vertex)
            min_proj = min(min_proj, _dot(world_vertex, axis))
        if min_proj == float("inf"):
            raise ValidationError("mesh geometry has no vertices")
        return min_proj
    raise ValidationError(f"Unsupported child geometry type: {type(geometry).__name__}")


def _subject_min_projection(
    subject: SurfaceSubject,
    *,
    axis: Vec3,
    asset_root: Optional[Path],
    prefer_collisions: bool,
) -> float:
    projections = [
        _support_min_projection_on_element(
            geometry,
            tf,
            axis=axis,
            asset_root=asset_root,
        )
        for geometry, tf in _iter_subject_items(subject, prefer_collisions=prefer_collisions)
    ]
    if not projections:
        raise ValidationError("child has no geometry to compute support distance")
    return min(projections)


def _mesh_geometry_from_subject(
    mesh: MeshSubject,
    *,
    asset_root: Optional[Union[str, Path, AssetContext]],
) -> MeshGeometry:
    if isinstance(mesh, MeshGeometry):
        return mesh.copy()
    if isinstance(mesh, Mesh):
        local_mesh = _load_trimesh_mesh(
            mesh,
            asset_root=_resolve_asset_root_for_subject(asset_root, mesh),
        )
        vertices = [
            (float(vertex[0]), float(vertex[1]), float(vertex[2])) for vertex in local_mesh.vertices
        ]
        faces = [(int(face[0]), int(face[1]), int(face[2])) for face in local_mesh.faces]
        return MeshGeometry(vertices=vertices, faces=faces)
    raise ValidationError(f"mesh must be a MeshGeometry or Mesh, got {type(mesh).__name__}")


def _normalize_surface_wrap_mapping(mapping: SurfaceWrapMapping | str) -> SurfaceWrapMapping:
    mapping_key = str(mapping).strip().lower()
    if mapping_key not in {"auto", "intrinsic", "nearest"}:
        raise ValidationError("mapping must be one of: 'auto', 'intrinsic', 'nearest'")
    return mapping_key  # type: ignore[return-value]


def _resolve_surface_max_edge(
    *,
    surface_max_edge: Optional[float],
    max_edge: Optional[float],
) -> Optional[float]:
    if surface_max_edge is None:
        return max_edge
    if max_edge is None:
        return surface_max_edge
    if abs(float(surface_max_edge) - float(max_edge)) > 1e-12:
        raise ValidationError("Provide only one of surface_max_edge or max_edge")
    return surface_max_edge


def _subdivide_mesh_geometry_to_size(
    geometry: MeshGeometry,
    *,
    max_edge: Optional[float],
) -> MeshGeometry:
    if max_edge is None:
        return geometry
    max_edge_value = float(max_edge)
    if max_edge_value <= 0.0:
        raise ValidationError("max_edge must be positive")
    if not geometry.vertices or not geometry.faces:
        return geometry
    try:
        import trimesh
    except Exception as exc:  # pragma: no cover - import failure is environment-specific
        raise RuntimeError(
            "Mesh surface wrapping with max_edge requires the optional 'trimesh' dependency"
        ) from exc
    vertices, faces = trimesh.remesh.subdivide_to_size(
        geometry.vertices,
        geometry.faces,
        max_edge=max_edge_value,
    )
    return MeshGeometry(
        vertices=[(float(vertex[0]), float(vertex[1]), float(vertex[2])) for vertex in vertices],
        faces=[(int(face[0]), int(face[1]), int(face[2])) for face in faces],
    )


def _normalize_profile_points(
    profile: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    try:
        return _ensure_ccw(_profile_points_2d(profile))
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc


def _normalize_hole_profiles(
    hole_profiles: Iterable[Iterable[tuple[float, float]]],
) -> list[list[tuple[float, float]]]:
    holes: list[list[tuple[float, float]]] = []
    for hole_profile in hole_profiles:
        holes.append(_normalize_profile_points(hole_profile))
    return holes


def _fallback_profile_solid(
    profile: list[tuple[float, float]],
    *,
    hole_profiles: list[list[tuple[float, float]]],
    thickness: float,
) -> MeshGeometry:
    if hole_profiles:
        solid = ExtrudeWithHolesGeometry(
            profile,
            hole_profiles,
            thickness,
            cap=True,
            center=False,
            closed=True,
        )
    else:
        solid = ExtrudeGeometry.from_z0(profile, thickness, cap=True, closed=True)
    solid.translate(0.0, 0.0, -float(thickness))
    return solid


def _point_distance_to_segment_xy(
    point: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> tuple[float, float]:
    seg_x = float(b[0]) - float(a[0])
    seg_y = float(b[1]) - float(a[1])
    seg_len2 = seg_x * seg_x + seg_y * seg_y
    if seg_len2 <= _EPS:
        dx = float(point[0]) - float(a[0])
        dy = float(point[1]) - float(a[1])
        return math.hypot(dx, dy), 0.0
    rel_x = float(point[0]) - float(a[0])
    rel_y = float(point[1]) - float(a[1])
    t = _clamp((rel_x * seg_x + rel_y * seg_y) / seg_len2, 0.0, 1.0)
    closest = (float(a[0]) + seg_x * t, float(a[1]) + seg_y * t)
    return math.hypot(float(point[0]) - closest[0], float(point[1]) - closest[1]), t


def _boundary_loop_for_planar_mesh(
    geometry: MeshGeometry,
    *,
    boundary_profile: list[tuple[float, float]],
) -> list[int]:
    edge_counts: dict[tuple[int, int], int] = {}
    for a, b, c in geometry.faces:
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_counts[key] = edge_counts.get(key, 0) + 1

    boundary_vertices = sorted(
        {index for edge, count in edge_counts.items() if count == 1 for index in edge}
    )
    if not boundary_vertices:
        raise ValidationError("profile mesh has no boundary loop")

    cumulative = [0.0]
    for i, a in enumerate(boundary_profile):
        b = boundary_profile[(i + 1) % len(boundary_profile)]
        cumulative.append(
            cumulative[-1] + math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
        )

    perimeter = cumulative[-1]
    if perimeter <= _EPS:
        raise ValidationError("profile perimeter must be non-zero")

    ordered = []
    for index in boundary_vertices:
        point = (float(geometry.vertices[index][0]), float(geometry.vertices[index][1]))
        best_distance = float("inf")
        best_arclength = 0.0
        for segment_index, a in enumerate(boundary_profile):
            b = boundary_profile[(segment_index + 1) % len(boundary_profile)]
            distance, t = _point_distance_to_segment_xy(point, a, b)
            if distance + 1e-9 < best_distance:
                best_distance = distance
                best_arclength = (
                    cumulative[segment_index]
                    + (cumulative[segment_index + 1] - cumulative[segment_index]) * t
                )
        if best_distance > 1e-5:
            raise ValidationError("profile mesh boundary deviated from source profile")
        if best_arclength >= perimeter - 1e-8:
            best_arclength = 0.0
        ordered.append((best_arclength, index))

    ordered.sort(key=lambda item: item[0])
    loop = [index for _, index in ordered]

    area = 0.0
    for i, index in enumerate(loop):
        x1, y1, _ = geometry.vertices[index]
        x2, y2, _ = geometry.vertices[loop[(i + 1) % len(loop)]]
        area += x1 * y2 - x2 * y1
    if area < 0.0:
        loop.reverse()
    return loop


def _planar_profile_cap(
    profile: Iterable[tuple[float, float]],
    *,
    max_edge: Optional[float],
) -> MeshGeometry:
    points = _normalize_profile_points(profile)
    try:
        faces = _triangulate_polygon(points)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc

    cap = MeshGeometry(
        vertices=[(float(x), float(y), 0.0) for x, y in points],
        faces=[(int(a), int(b), int(c)) for a, b, c in faces],
    )
    if not cap.faces:
        raise ValidationError("profile triangulation produced no faces")
    return _subdivide_mesh_geometry_to_size(cap, max_edge=max_edge)


def _solid_from_planar_profile_cap(
    cap: MeshGeometry,
    *,
    boundary_profile: list[tuple[float, float]],
    thickness: float,
) -> MeshGeometry:
    thickness_value = float(thickness)
    if thickness_value <= 0.0:
        raise ValidationError("thickness must be positive")

    loop = _boundary_loop_for_planar_mesh(cap, boundary_profile=boundary_profile)
    vertex_count = len(cap.vertices)
    solid = MeshGeometry(
        vertices=[
            *[(float(x), float(y), 0.0) for x, y, _ in cap.vertices],
            *[(float(x), float(y), -thickness_value) for x, y, _ in cap.vertices],
        ],
        faces=[],
    )

    for a, b, c in cap.faces:
        solid.add_face(a, b, c)

    offset = vertex_count
    for a, b, c in cap.faces:
        solid.add_face(offset + c, offset + b, offset + a)

    for i, a in enumerate(loop):
        b = loop[(i + 1) % len(loop)]
        solid.add_face(a, b, offset + b)
        solid.add_face(a, offset + b, offset + a)
    return solid


def _subject_single_world_sphere(
    subject: SurfaceSubject,
    *,
    prefer_collisions: bool,
) -> tuple[Vec3, float] | None:
    items = _iter_subject_items(subject, prefer_collisions=prefer_collisions)
    if len(items) != 1:
        return None
    geometry, tf = _expand_mesh_provenance(items[0][0], items[0][1])
    if not isinstance(geometry, Sphere):
        return None
    center = _mat4_vec3(tf, (0.0, 0.0, 0.0))
    return center, float(geometry.radius)


def _subject_single_world_cylinder(
    subject: SurfaceSubject,
    *,
    prefer_collisions: bool,
) -> tuple[Vec3, Vec3, float, float] | None:
    items = _iter_subject_items(subject, prefer_collisions=prefer_collisions)
    if len(items) != 1:
        return None
    geometry, tf = _expand_mesh_provenance(items[0][0], items[0][1])
    if not isinstance(geometry, Cylinder):
        return None
    center = _mat4_vec3(tf, (0.0, 0.0, 0.0))
    axis_world = _normalize(
        _mat4_transform_direction(tf, (0.0, 0.0, 1.0)),
        name="cylinder axis",
    )
    return center, axis_world, float(geometry.radius), float(geometry.length)


def _wrap_mesh_onto_sphere(
    geometry: MeshGeometry,
    *,
    frame: SurfaceFrame,
    sphere_center: Vec3,
    sphere_radius: float,
    child_axis: str,
    visible_relief: float,
    spin: float,
) -> MeshGeometry:
    rot = _rotation_for_surface_frame(frame, child_axis=child_axis, spin=float(spin))
    axis_local = _axis_vector(child_axis)
    outer_support = max(_dot(vertex, axis_local) for vertex in geometry.vertices)

    wrapped_vertices: list[Vec3] = []
    for vertex in geometry.vertices:
        local_proj = _dot(vertex, axis_local)
        depth_behind = outer_support - local_proj
        in_plane_local = (
            float(vertex[0]) - axis_local[0] * local_proj,
            float(vertex[1]) - axis_local[1] * local_proj,
            float(vertex[2]) - axis_local[2] * local_proj,
        )
        tangent_offset = _mat3_vec3(rot, in_plane_local)
        tangent_distance = _norm(tangent_offset)
        if tangent_distance <= _EPS:
            surface_normal = frame.normal
        else:
            tangent_dir = (
                tangent_offset[0] / tangent_distance,
                tangent_offset[1] / tangent_distance,
                tangent_offset[2] / tangent_distance,
            )
            angle = tangent_distance / float(sphere_radius)
            surface_normal = _normalize(
                (
                    math.cos(angle) * frame.normal[0] + math.sin(angle) * tangent_dir[0],
                    math.cos(angle) * frame.normal[1] + math.sin(angle) * tangent_dir[1],
                    math.cos(angle) * frame.normal[2] + math.sin(angle) * tangent_dir[2],
                ),
                name="sphere wrap normal",
            )
        surface_point = (
            sphere_center[0] + surface_normal[0] * float(sphere_radius),
            sphere_center[1] + surface_normal[1] * float(sphere_radius),
            sphere_center[2] + surface_normal[2] * float(sphere_radius),
        )
        wrapped_vertices.append(
            (
                surface_point[0] + surface_normal[0] * (float(visible_relief) - depth_behind),
                surface_point[1] + surface_normal[1] * (float(visible_relief) - depth_behind),
                surface_point[2] + surface_normal[2] * (float(visible_relief) - depth_behind),
            )
        )

    return MeshGeometry(vertices=wrapped_vertices, faces=list(geometry.faces))


def _wrap_mesh_onto_cylinder(
    geometry: MeshGeometry,
    *,
    frame: SurfaceFrame,
    cylinder_center: Vec3,
    cylinder_axis: Vec3,
    cylinder_radius: float,
    cylinder_length: float,
    child_axis: str,
    visible_relief: float,
    spin: float,
) -> MeshGeometry | None:
    if abs(_dot(frame.normal, cylinder_axis)) >= 0.2:
        return None

    rel_anchor = (
        frame.point[0] - cylinder_center[0],
        frame.point[1] - cylinder_center[1],
        frame.point[2] - cylinder_center[2],
    )
    anchor_axial = _dot(rel_anchor, cylinder_axis)
    anchor_radial = (
        rel_anchor[0] - cylinder_axis[0] * anchor_axial,
        rel_anchor[1] - cylinder_axis[1] * anchor_axial,
        rel_anchor[2] - cylinder_axis[2] * anchor_axial,
    )
    if _norm(anchor_radial) <= _EPS:
        return None

    anchor_normal = _normalize(anchor_radial, name="cylinder wrap normal")
    circum_axis = _normalize(
        _cross(cylinder_axis, anchor_normal),
        name="cylinder circumferential tangent",
    )
    half_length = float(cylinder_length) * 0.5
    rot = _rotation_for_surface_frame(frame, child_axis=child_axis, spin=float(spin))
    axis_local = _axis_vector(child_axis)
    outer_support = max(_dot(vertex, axis_local) for vertex in geometry.vertices)

    wrapped_vertices: list[Vec3] = []
    for vertex in geometry.vertices:
        local_proj = _dot(vertex, axis_local)
        depth_behind = outer_support - local_proj
        in_plane_local = (
            float(vertex[0]) - axis_local[0] * local_proj,
            float(vertex[1]) - axis_local[1] * local_proj,
            float(vertex[2]) - axis_local[2] * local_proj,
        )
        tangent_offset = _mat3_vec3(rot, in_plane_local)
        circum_distance = _dot(tangent_offset, circum_axis)
        axial_distance = _dot(tangent_offset, cylinder_axis)
        axial = anchor_axial + axial_distance
        if abs(axial) > half_length + 1e-9:
            return None

        angle = circum_distance / float(cylinder_radius)
        turn = _axis_angle_matrix(cylinder_axis, angle)
        surface_normal = _normalize(
            _mat3_vec3(turn, anchor_normal),
            name="cylinder wrap rotated normal",
        )
        surface_point = (
            cylinder_center[0]
            + cylinder_axis[0] * axial
            + surface_normal[0] * float(cylinder_radius),
            cylinder_center[1]
            + cylinder_axis[1] * axial
            + surface_normal[1] * float(cylinder_radius),
            cylinder_center[2]
            + cylinder_axis[2] * axial
            + surface_normal[2] * float(cylinder_radius),
        )
        wrapped_vertices.append(
            (
                surface_point[0] + surface_normal[0] * (float(visible_relief) - depth_behind),
                surface_point[1] + surface_normal[1] * (float(visible_relief) - depth_behind),
                surface_point[2] + surface_normal[2] * (float(visible_relief) - depth_behind),
            )
        )

    return MeshGeometry(vertices=wrapped_vertices, faces=list(geometry.faces))


def _wrap_mesh_onto_surface_nearest(
    geometry: MeshGeometry,
    *,
    frame: SurfaceFrame,
    target: SurfaceSubject,
    child_axis: str,
    visible_relief: float,
    spin: float,
    asset_root: Optional[Union[str, Path, AssetContext]],
    prefer_collisions: bool,
) -> MeshGeometry:
    rot = _rotation_for_surface_frame(frame, child_axis=child_axis, spin=float(spin))
    axis_local = _axis_vector(child_axis)
    outer_support = max(_dot(vertex, axis_local) for vertex in geometry.vertices)

    wrapped_vertices: list[Vec3] = []
    for vertex in geometry.vertices:
        local_proj = _dot(vertex, axis_local)
        depth_behind = outer_support - local_proj
        in_plane_local = (
            float(vertex[0]) - axis_local[0] * local_proj,
            float(vertex[1]) - axis_local[1] * local_proj,
            float(vertex[2]) - axis_local[2] * local_proj,
        )
        plane_offset = _mat3_vec3(rot, in_plane_local)
        query_point = (
            frame.point[0] + plane_offset[0],
            frame.point[1] + plane_offset[1],
            frame.point[2] + plane_offset[2],
        )
        hit = _query_subject_surface_at_point(
            target,
            point_hint=query_point,
            asset_root=asset_root,
            prefer_collisions=prefer_collisions,
        )
        wrapped_vertices.append(
            (
                hit.point[0] + hit.normal[0] * (float(visible_relief) - depth_behind),
                hit.point[1] + hit.normal[1] * (float(visible_relief) - depth_behind),
                hit.point[2] + hit.normal[2] * (float(visible_relief) - depth_behind),
            )
        )

    return MeshGeometry(vertices=wrapped_vertices, faces=list(geometry.faces))


def wrap_mesh_onto_surface(
    mesh: MeshSubject,
    target: SurfaceSubject,
    *,
    point_hint: Optional[Vec3] = None,
    direction: Optional[Vec3] = None,
    child_axis: str = "+z",
    visible_relief: float = 0.0,
    mapping: SurfaceWrapMapping | str = "auto",
    surface_max_edge: Optional[float] = None,
    max_edge: Optional[float] = None,
    spin: float = 0.0,
    asset_root: Optional[Union[str, Path, AssetContext]] = None,
    prefer_collisions: bool = False,
    up_hint: Vec3 = (0.0, 0.0, 1.0),
) -> MeshGeometry:
    mapping_mode = _normalize_surface_wrap_mapping(mapping)
    frame = surface_frame(
        target,
        point_hint=point_hint,
        direction=direction,
        asset_root=asset_root,
        prefer_collisions=prefer_collisions,
        up_hint=up_hint,
    )
    resolved_max_edge = _resolve_surface_max_edge(
        surface_max_edge=surface_max_edge,
        max_edge=max_edge,
    )
    geometry = _mesh_geometry_from_subject(mesh, asset_root=asset_root)
    geometry = _subdivide_mesh_geometry_to_size(geometry, max_edge=resolved_max_edge)
    if not geometry.vertices:
        raise ValidationError("mesh has no vertices")
    if mapping_mode != "nearest":
        sphere_info = _subject_single_world_sphere(
            target,
            prefer_collisions=prefer_collisions,
        )
        if sphere_info is not None:
            sphere_center, sphere_radius = sphere_info
            return _wrap_mesh_onto_sphere(
                geometry,
                frame=frame,
                sphere_center=sphere_center,
                sphere_radius=sphere_radius,
                child_axis=child_axis,
                visible_relief=float(visible_relief),
                spin=float(spin),
            )

        cylinder_info = _subject_single_world_cylinder(
            target,
            prefer_collisions=prefer_collisions,
        )
        if cylinder_info is not None:
            cylinder_center, cylinder_axis, cylinder_radius, cylinder_length = cylinder_info
            intrinsic = _wrap_mesh_onto_cylinder(
                geometry,
                frame=frame,
                cylinder_center=cylinder_center,
                cylinder_axis=cylinder_axis,
                cylinder_radius=cylinder_radius,
                cylinder_length=cylinder_length,
                child_axis=child_axis,
                visible_relief=float(visible_relief),
                spin=float(spin),
            )
            if intrinsic is not None:
                return intrinsic

        if mapping_mode == "intrinsic":
            raise ValidationError(
                "Intrinsic surface wrapping is only supported for spheres and cylinder sidewalls"
            )

    return _wrap_mesh_onto_surface_nearest(
        geometry,
        frame=frame,
        target=target,
        child_axis=child_axis,
        visible_relief=float(visible_relief),
        spin=float(spin),
        asset_root=asset_root,
        prefer_collisions=prefer_collisions,
    )


def wrap_profile_onto_surface(
    profile: Iterable[tuple[float, float]],
    target: SurfaceSubject,
    *,
    thickness: float,
    hole_profiles: Iterable[Iterable[tuple[float, float]]] = (),
    point_hint: Optional[Vec3] = None,
    direction: Optional[Vec3] = None,
    visible_relief: float = 0.0,
    mapping: SurfaceWrapMapping | str = "auto",
    surface_max_edge: Optional[float] = None,
    max_edge: Optional[float] = None,
    spin: float = 0.0,
    asset_root: Optional[Union[str, Path, AssetContext]] = None,
    prefer_collisions: bool = False,
    up_hint: Vec3 = (0.0, 0.0, 1.0),
) -> MeshGeometry:
    thickness_value = float(thickness)
    if thickness_value <= 0.0:
        raise ValidationError("thickness must be positive")
    resolved_max_edge = _resolve_surface_max_edge(
        surface_max_edge=surface_max_edge,
        max_edge=max_edge,
    )
    points = _normalize_profile_points(profile)
    holes = _normalize_hole_profiles(hole_profiles)
    if holes:
        logger.warning(
            "wrap_profile_onto_surface using extrude fallback because hole_profiles were provided: hole_count=%s thickness=%.6g",
            len(holes),
            thickness_value,
        )
        fallback = _fallback_profile_solid(
            points,
            hole_profiles=holes,
            thickness=thickness_value,
        )
        return wrap_mesh_onto_surface(
            fallback,
            target,
            point_hint=point_hint,
            direction=direction,
            child_axis="+z",
            visible_relief=visible_relief,
            mapping=mapping,
            surface_max_edge=resolved_max_edge,
            spin=spin,
            asset_root=asset_root,
            prefer_collisions=prefer_collisions,
            up_hint=up_hint,
        )

    cap = _planar_profile_cap(points, max_edge=resolved_max_edge)
    try:
        solid = _solid_from_planar_profile_cap(
            cap,
            boundary_profile=points,
            thickness=thickness_value,
        )
        return wrap_mesh_onto_surface(
            solid,
            target,
            point_hint=point_hint,
            direction=direction,
            child_axis="+z",
            visible_relief=visible_relief,
            mapping=mapping,
            spin=spin,
            asset_root=asset_root,
            prefer_collisions=prefer_collisions,
            up_hint=up_hint,
        )
    except ValidationError as exc:
        message = str(exc)
        if "profile mesh boundary" not in message:
            raise
        logger.warning(
            "wrap_profile_onto_surface using extrude fallback after planar cap reconstruction failed: %s",
            message,
        )

    fallback = _fallback_profile_solid(
        points,
        hole_profiles=[],
        thickness=thickness_value,
    )
    return wrap_mesh_onto_surface(
        fallback,
        target,
        point_hint=point_hint,
        direction=direction,
        child_axis="+z",
        visible_relief=visible_relief,
        mapping=mapping,
        surface_max_edge=resolved_max_edge,
        spin=spin,
        asset_root=asset_root,
        prefer_collisions=prefer_collisions,
        up_hint=up_hint,
    )


def place_on_surface(
    child: SurfaceSubject,
    target: SurfaceSubject,
    *,
    point_hint: Optional[Vec3] = None,
    direction: Optional[Vec3] = None,
    child_axis: str = "+z",
    clearance: float = 0.0,
    spin: float = 0.0,
    asset_root: Optional[Union[str, Path, AssetContext]] = None,
    prefer_collisions: bool = False,
    child_prefer_collisions: bool = False,
    up_hint: Vec3 = (0.0, 0.0, 1.0),
) -> Origin:
    common_frame = FrameRef.part_local(target)
    target_anchor = _surface_anchor(
        target,
        point_hint=point_hint,
        direction=direction,
        asset_root=asset_root,
        prefer_collisions=prefer_collisions,
        up_hint=up_hint,
        frame=common_frame,
    )
    child_root = _resolve_asset_root_for_subject(asset_root, child)
    child_axis_vec = _axis_vector(child_axis)
    mount_proj = _subject_min_projection(
        child,
        axis=child_axis_vec,
        asset_root=child_root,
        prefer_collisions=child_prefer_collisions,
    )
    child_basis = _child_local_basis_for_axis(child_axis)
    child_anchor = Anchor(
        pose=RigidFrame(
            position=tuple(component * mount_proj for component in child_axis_vec),  # type: ignore[arg-type]
            basis=child_basis,
        ),
        frame=common_frame,
        kind="plane",
        subject=child,
        subject_pose=RigidFrame.identity(),
        local_pose=RigidFrame(
            position=tuple(component * mount_proj for component in child_axis_vec),  # type: ignore[arg-type]
            basis=child_basis,
        ),
        source=_anchor_source(
            child,
            "internal_surface_mount",
            local_pose=RigidFrame(
                position=tuple(component * mount_proj for component in child_axis_vec),  # type: ignore[arg-type]
                basis=child_basis,
            ),
        ),
    )
    if abs(float(spin)) > _EPS:
        spin_rotation = _axis_angle_matrix(target_anchor.pose.axis("+z"), float(spin))
        spun_rotation = _mat3_mul(
            spin_rotation,
            _mat3_from_columns(*target_anchor.pose.basis),
        )
        target_anchor = Anchor(
            pose=RigidFrame(
                position=target_anchor.pose.position,
                basis=_mat3_columns(spun_rotation),
            ),
            frame=target_anchor.frame,
            kind=target_anchor.kind,
            subject=target_anchor.subject,
            subject_pose=target_anchor.subject_pose,
            local_pose=target_anchor.local_pose,
            source=target_anchor.source,
        )
    return attach(child_anchor, target_anchor, clearance=clearance).origin


def proud_for_flush_mount(
    child_link: Part,
    *,
    axis: str = "z",
    clearance: float = 0.0,
    asset_root: Optional[Union[str, Path, AssetContext]] = None,
    prefer_collisions: bool = True,
) -> float:
    warnings.warn(
        "proud_for_flush_mount(...) is deprecated; use attach(..., clearance=...) "
        "with an explicit child support Anchor instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    axis_key = axis.strip().lower()
    axis_idx = {"x": 0, "y": 1, "z": 2}.get(axis_key)
    if axis_idx is None:
        raise ValidationError(f"Invalid axis {axis!r}; expected 'x', 'y', or 'z'.")

    root = resolve_asset_root(asset_root, child_link)
    aabb = link_local_aabb(child_link, asset_root=root, prefer_collisions=prefer_collisions)
    if aabb is None:
        raise ValidationError(f"Child link {child_link.name!r} has no geometry to compute AABB")

    (mn, mx) = aabb
    thickness = float(mx[axis_idx]) - float(mn[axis_idx])
    return 0.5 * thickness + float(clearance)
