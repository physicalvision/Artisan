
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from sdk._dependencies import require_cadquery

from ..types import Box, Cylinder, Origin
from .cadquery_helpers import _mesh_geometry_from_cadquery_model
from .common import _set_collision_primitives
from .primitives import MeshGeometry, _adopt_mesh_geometry
from .specs import HingeHolePattern, HingePinStyle

Vec3 = Tuple[float, float, float]

__all__ = ["HingePin", "HingeLeaf", "HingeHalfGeometry"]

_DEFAULT_PIN_DIAMETER = 0.006
_DEFAULT_KNUCKLE_DIAMETER = 0.010
_DEFAULT_CLEARANCE = 3e-4


def _normalize(a: Sequence[float]) -> Vec3:
    m = math.sqrt(sum(float(x) * float(x) for x in a))
    if m < 1e-12:
        raise ValueError("axis/direction must be non-zero")
    return (float(a[0]) / m, float(a[1]) / m, float(a[2]) / m)


def _axis_rotations_deg(axis: Vec3) -> Tuple[float, float]:
    _ax, _ay, az = axis
    pitch = math.degrees(math.acos(max(-1.0, min(1.0, az))))
    yaw = (
        0.0
        if abs(math.sin(math.radians(pitch))) < 1e-9
        else math.degrees(math.atan2(axis[1], axis[0]))
    )
    return pitch, yaw


def _principal_index(v: Vec3) -> Optional[int]:
    idx = max(range(3), key=lambda k: abs(v[k]))
    if abs(v[idx]) > 0.999 and all(abs(v[k]) < 1e-3 for k in range(3) if k != idx):
        return idx
    return None


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _leaf_frame(axis: Vec3, toward: Vec3) -> Tuple[Vec3, Vec3]:
    dot = sum(toward[k] * axis[k] for k in range(3))
    perp = tuple(toward[k] - dot * axis[k] for k in range(3))
    if math.sqrt(sum(c * c for c in perp)) < 1e-6:
        raise ValueError("HingeLeaf.toward must not be parallel to the pin axis")
    t = _normalize(perp)
    return t, _normalize(_cross(axis, t))


def _frame_rpy(x: Vec3, y: Vec3, z: Vec3) -> Vec3:
    flat = math.hypot(y[2], z[2])
    pitch = math.atan2(-x[2], flat)
    if flat < 1e-9:
        sign = -1.0 if x[2] > 0.0 else 1.0
        return (math.atan2(sign * y[0], y[1]), pitch, 0.0)
    return (math.atan2(y[2], z[2]), pitch, math.atan2(x[1], x[0]))


@dataclass(frozen=True)
class HingeLeaf:

    width: float
    thickness: float
    toward: Vec3
    length: Optional[float] = None
    holes: Optional[HingeHolePattern] = None


@dataclass(frozen=True)
class HingePin:

    axis: Vec3
    anchor: Vec3
    length: float
    knuckle_count: int = 5
    pin_diameter: float = _DEFAULT_PIN_DIAMETER
    knuckle_diameter: float = _DEFAULT_KNUCKLE_DIAMETER
    clearance: float = _DEFAULT_CLEARANCE
    pin_style: Optional[HingePinStyle] = None
    part_span: Optional[float] = None

    @classmethod
    def from_anchor(
        cls,
        mount: object,
        *,
        length: float,
        axis: Optional[Vec3] = None,
        knuckle_count: int = 5,
        pin_diameter: float = _DEFAULT_PIN_DIAMETER,
        knuckle_diameter: float = _DEFAULT_KNUCKLE_DIAMETER,
        clearance: float = _DEFAULT_CLEARANCE,
        pin_style: Optional[HingePinStyle] = None,
        part_span: Optional[float] = None,
    ) -> HingePin:

        from ..placement import Anchor

        if not isinstance(mount, Anchor):
            raise TypeError("mount must be an Anchor")
        resolved_axis = axis if axis is not None else mount.pose.axis("+z")
        return cls(
            axis=_normalize(resolved_axis),
            anchor=mount.pose.position,
            length=float(length),
            knuckle_count=int(knuckle_count),
            pin_diameter=float(pin_diameter),
            knuckle_diameter=float(knuckle_diameter),
            clearance=float(clearance),
            pin_style=pin_style,
            part_span=part_span,
        )

    @classmethod
    def from_edge(
        cls,
        edge: object,
        *,
        length: float,
        axis: Optional[Vec3] = None,
        knuckle_count: int = 5,
        pin_diameter: float = _DEFAULT_PIN_DIAMETER,
        knuckle_diameter: float = _DEFAULT_KNUCKLE_DIAMETER,
        clearance: float = _DEFAULT_CLEARANCE,
        pin_style: Optional[HingePinStyle] = None,
        part_span: Optional[float] = None,
    ) -> HingePin:

        from ..placement import Anchor

        if not isinstance(edge, Anchor) or edge.kind != "axis":
            raise TypeError("edge must be an axis Anchor")

        return cls.from_anchor(
            edge,
            length=length,
            axis=axis,
            knuckle_count=knuckle_count,
            pin_diameter=pin_diameter,
            knuckle_diameter=knuckle_diameter,
            clearance=clearance,
            pin_style=pin_style,
            part_span=part_span,
        )

    def __post_init__(self) -> None:
        try:
            segment_aspect = self._segment_length() / float(self.knuckle_diameter)
        except (TypeError, ValueError, ZeroDivisionError):
            return
        if segment_aspect > 2.5:
            warnings.warn(
                "HingePin knuckle segment aspect exceeds 2.5; increase knuckle_count "
                "or knuckle_diameter for a supported barrel.",
                UserWarning,
                stacklevel=2,
            )
        if self.part_span is not None:
            if float(self.part_span) <= 0.0:
                raise ValueError("HingePin.part_span must be positive when provided")
            if float(self.length) > 0.6 * float(self.part_span):
                warnings.warn(
                    "HingePin length exceeds 0.6 of part_span; verify this is an "
                    "intentional continuous hinge or use multiple supported hinges.",
                    UserWarning,
                    stacklevel=2,
                )

    def _unit(self) -> Vec3:
        return _normalize(self.axis)

    def _segment_length(self) -> float:
        n = int(self.knuckle_count)
        if n < 3:
            raise ValueError("knuckle_count must be at least 3")
        seg = (float(self.length) - float(self.clearance) * (n - 1)) / n
        if seg <= 0:
            raise ValueError("clearance/knuckle_count leave no knuckle length")
        return seg

    def _segment_offsets(self) -> list[float]:
        n = int(self.knuckle_count)
        seg = self._segment_length()
        start = -float(self.length) / 2 + seg / 2
        return [start + i * (seg + float(self.clearance)) for i in range(n)]

    def bearing_pairs(self, *, parent_elem: str, child_elem: str) -> list[Tuple[str, str]]:
        return []

    def declare_bearings(
        self,
        ctx,
        *,
        parent,
        parent_elem: str,
        child,
        child_elem: str,
        reason: str = "hinge bearing contact at the shared pin",
    ) -> None:
        for a, b in self.bearing_pairs(parent_elem=parent_elem, child_elem=child_elem):
            ctx.allow_overlap(parent, child, elem_a=a, elem_b=b, reason=reason)


class HingeHalfGeometry(MeshGeometry):

    def __init__(
        self,
        pin: HingePin,
        *,
        role: str,
        leaf: Optional[HingeLeaf] = None,
        include_pin: bool = False,
        bore: Optional[bool] = None,
    ):
        super().__init__()
        if role not in ("outer", "inner"):
            raise ValueError("role must be 'outer' or 'inner'")
        cq = require_cadquery(feature="HingeHalfGeometry")
        axis = pin._unit()
        pitch, yaw = _axis_rotations_deg(axis)
        seg = pin._segment_length()

        def _along(shape, offset: float):
            s = shape.rotate((0, 0, 0), (0, 1, 0), pitch).rotate((0, 0, 0), (0, 0, 1), yaw)
            return s.translate((offset * axis[0], offset * axis[1], offset * axis[2]))

        def _cyl(radius: float, height: float):
            return cq.Workplane("XY").circle(radius).extrude(height / 2.0, both=True)

        def _tube(outer_r: float, inner_r: float, height: float):
            outer = cq.Workplane("XY").circle(outer_r).extrude(height / 2.0, both=True)
            hole = cq.Workplane("XY").circle(inner_r).extrude(height / 2.0 + 1e-4, both=True)
            return outer.cut(hole)

        do_bore = (not include_pin) if bore is None else bool(bore)
        knuckle_r = float(pin.knuckle_diameter) / 2.0
        bore_r = float(pin.pin_diameter) / 2.0 + float(pin.clearance)
        if do_bore and bore_r >= knuckle_r:
            raise ValueError(
                "hinge knuckle bore (pin_diameter/2 + clearance) is >= knuckle radius: "
                "increase knuckle_diameter or reduce pin_diameter/clearance so a wall remains"
            )

        knuckle_rpy = (0.0, math.radians(pitch), math.radians(yaw))
        collision_prims: list[Tuple[object, Origin]] = []

        shape = None
        for i, off in enumerate(pin._segment_offsets()):
            if (role == "outer") != (i % 2 == 0):
                continue
            solid = _tube(knuckle_r, bore_r, seg) if do_bore else _cyl(knuckle_r, seg)
            knuckle = _along(solid, off)
            shape = knuckle if shape is None else shape.union(knuckle)
            collision_prims.append(
                (
                    Cylinder(radius=knuckle_r, length=seg),
                    Origin(xyz=(off * axis[0], off * axis[1], off * axis[2]), rpy=knuckle_rpy),
                )
            )

        if include_pin:
            style = pin.pin_style or HingePinStyle()
            pin_len = float(pin.length) + 2.0 * float(style.exposed_end)
            pin_shape = _along(_cyl(float(pin.pin_diameter) / 2.0, pin_len), 0.0)
            shape = pin_shape if shape is None else shape.union(pin_shape)

        if leaf is not None:
            toward = _normalize(leaf.toward)
            ai, ti = _principal_index(axis), _principal_index(toward)
            principal = ai is not None and ti is not None and ai != ti
            leaf_normal: Optional[Vec3] = None
            if not principal:
                toward, leaf_normal = _leaf_frame(axis, toward)
            length = float(leaf.length) if leaf.length is not None else float(pin.length)
            kept = [
                o for i, o in enumerate(pin._segment_offsets()) if (role == "outer") == (i % 2 == 0)
            ]
            if kept:
                reach = max(abs(min(kept) - seg / 2.0), abs(max(kept) + seg / 2.0))
                if length / 2.0 < reach - 1e-9:
                    raise ValueError(
                        f"HingeLeaf.length ({length:g}) is too short to bridge this half's "
                        f"knuckles (needs >= {2.0 * reach:g}); the outer knuckles would island. "
                        "Increase leaf.length (the default, = pin.length, spans them all)."
                    )
            width, thickness = float(leaf.width), float(leaf.thickness)
            stack = thickness / 2.0 + float(pin.clearance)
            sign = 1.0 if role == "outer" else -1.0
            if principal:
                ni = 3 - ai - ti
                dims = [thickness, thickness, thickness]
                dims[ai] = length
                dims[ti] = width
                normal = [0.0, 0.0, 0.0]
                normal[ni] = sign * stack
                leaf_rpy = (0.0, 0.0, 0.0)
            else:
                assert leaf_normal is not None
                dims = [length, width, thickness]
                normal = [sign * stack * c for c in leaf_normal]
                leaf_rpy = _frame_rpy(axis, toward, leaf_normal)

            def _leaf_center(toward_off: float) -> Tuple[float, float, float]:
                return (
                    toward[0] * toward_off + normal[0],
                    toward[1] * toward_off + normal[1],
                    toward[2] * toward_off + normal[2],
                )

            embed = min(0.25 * knuckle_r, 0.5 * width)
            visual_center = _leaf_center(knuckle_r + width / 2.0 - embed)
            if principal:
                plate = cq.Workplane("XY").box(dims[0], dims[1], dims[2]).translate(visual_center)
            else:
                frame = cq.Plane(origin=(0, 0, 0), xDir=axis, normal=leaf_normal)
                plate = cq.Workplane(frame).box(*dims).translate(visual_center)
            other_offsets = [
                o for j, o in enumerate(pin._segment_offsets()) if (role == "outer") != (j % 2 == 0)
            ]
            notch_r = knuckle_r + float(pin.clearance)
            for o in other_offsets:
                plate = plate.cut(_along(_cyl(notch_r, seg + float(pin.clearance)), o))
            shape = plate if shape is None else shape.union(plate)
            collision_center = _leaf_center(knuckle_r + width / 2.0)
            collision_prims.append(
                (Box((dims[0], dims[1], dims[2])), Origin(xyz=collision_center, rpy=leaf_rpy))
            )

        if shape is None:
            raise ValueError("HingeHalfGeometry produced no geometry (no knuckles for this role)")

        _adopt_mesh_geometry(self, _mesh_geometry_from_cadquery_model(shape))
        if collision_prims:
            _set_collision_primitives(self, collision_prims)
