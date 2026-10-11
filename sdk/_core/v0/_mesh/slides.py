
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from sdk._dependencies import require_cadquery

from ..types import Box, Origin
from .cadquery_helpers import _mesh_geometry_from_cadquery_model
from .common import _set_collision_primitives
from .primitives import MeshGeometry, _adopt_mesh_geometry

Vec3 = Tuple[float, float, float]

__all__ = ["SlideRail", "SlideRailGeometry"]


def _normalize(a: Sequence[float]) -> Vec3:
    m = math.sqrt(sum(float(x) * float(x) for x in a))
    if m < 1e-12:
        raise ValueError("axis/direction must be non-zero")
    return (float(a[0]) / m, float(a[1]) / m, float(a[2]) / m)


def _principal_index(v: Vec3) -> Optional[int]:
    idx = max(range(3), key=lambda k: abs(v[k]))
    if abs(v[idx]) > 0.999 and all(abs(v[k]) < 1e-3 for k in range(3) if k != idx):
        return idx
    return None


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _rail_frame(slide: Vec3, up: Vec3) -> Tuple[Vec3, Vec3]:
    dot = sum(up[k] * slide[k] for k in range(3))
    perp = tuple(up[k] - dot * slide[k] for k in range(3))
    if math.sqrt(sum(c * c for c in perp)) < 1e-6:
        raise ValueError("SlideRail.up must not be parallel to slide_axis")
    u = _normalize(perp)
    return u, _normalize(_cross(u, slide))


def _frame_rpy(x: Vec3, y: Vec3, z: Vec3) -> Vec3:
    flat = math.hypot(y[2], z[2])
    pitch = math.atan2(-x[2], flat)
    if flat < 1e-9:
        sign = -1.0 if x[2] > 0.0 else 1.0
        return (math.atan2(sign * y[0], y[1]), pitch, 0.0)
    return (math.atan2(y[2], z[2]), pitch, math.atan2(x[1], x[0]))


@dataclass(frozen=True)
class SlideRail:

    slide_axis: Vec3
    up: Vec3
    anchor: Vec3
    length: float
    width: float
    height: float
    runner_length: Optional[float] = None
    wall: float = 0.006
    clearance: float = 5e-4
    lip: float = 0.004
    capture: bool = True
    mount_offset: Optional[float] = None
    stem_width: Optional[float] = None
    travel: Optional[float] = None
    retained_margin: Optional[float] = None

    def _axes(self) -> Tuple[int, int, int, Vec3, Vec3, Vec3]:
        s = _normalize(self.slide_axis)
        u = _normalize(self.up)
        si, ui = _principal_index(s), _principal_index(u)
        if si is None or ui is None or si == ui:
            u, lat = _rail_frame(s, u)
            return -1, -1, -1, s, lat, u
        li = 3 - si - ui
        lat = [0.0, 0.0, 0.0]
        lat[li] = 1.0
        return si, li, ui, s, (lat[0], lat[1], lat[2]), u

    def _runner_length(self) -> float:
        return float(self.runner_length) if self.runner_length is not None else float(self.length)

    def _slot_inner(self) -> float:
        return float(self.width) - 2.0 * float(self.lip) - 2.0 * float(self.clearance)

    def _mount_offset(self) -> float:
        if self.mount_offset is not None:
            return float(self.mount_offset)
        return float(self.height) / 2.0 + float(self.wall) + 4.0 * float(self.clearance)

    def _stem_width(self) -> float:
        if self.stem_width is not None:
            return float(self.stem_width)
        return (
            self._slot_inner()
            if self.capture
            else (float(self.width) - 2.0 * float(self.clearance))
        )

    def runner_mount_local(self) -> Vec3:
        _si, _li, _ui, _s, _l, u = self._axes()
        m = self._mount_offset()
        return (u[0] * m, u[1] * m, u[2] * m)

    def _validate(self) -> None:
        for name, v in (
            ("length", self.length),
            ("width", self.width),
            ("height", self.height),
            ("wall", self.wall),
            ("clearance", self.clearance),
        ):
            if float(v) <= 0.0:
                raise ValueError(f"SlideRail.{name} must be positive")
        if self._runner_length() <= 0.0:
            raise ValueError("SlideRail.runner_length must be positive")
        if self.travel is not None:
            travel = float(self.travel)
            if travel <= 0.0:
                raise ValueError("SlideRail.travel must be positive when provided")
            margin = (
                float(self.retained_margin)
                if self.retained_margin is not None
                else max(0.01, 2.0 * float(self.wall))
            )
            if margin < 0.0:
                raise ValueError("SlideRail.retained_margin must be non-negative")
            if self._runner_length() < travel + margin:
                warnings.warn(
                    "SlideRail runner_length does not cover travel plus retained margin; "
                    "increase runner_length or reduce the declared travel.",
                    UserWarning,
                    stacklevel=2,
                )
        if 2.0 * float(self.clearance) >= float(self.width):
            raise ValueError("SlideRail.clearance leaves no runner width")
        if 2.0 * float(self.clearance) >= float(self.height):
            raise ValueError("SlideRail.clearance leaves no runner height")
        if self.capture and float(self.lip) <= float(self.clearance):
            raise ValueError(
                "SlideRail.lip must exceed clearance to actually capture the runner "
                "(set capture=False for an open U-channel)"
            )
        if self.capture and self._slot_inner() <= 0.0:
            raise ValueError(
                "SlideRail capture lips leave no slot for the runner stem; reduce lip or "
                "increase width"
            )
        if self.mount_offset is not None:
            mo = float(self.mount_offset)
            min_mo = float(self.height) / 2.0 + float(self.wall)
            if mo < min_mo:
                raise ValueError(
                    f"SlideRail.mount_offset ({mo:g}) must clear the channel: it needs to be "
                    f">= height/2 + wall ({min_mo:g}) so the stem exits the channel and the "
                    "body-mount face sits outside the cavity (a smaller value buries the mount "
                    "face inside the channel, so the body collides the walls)"
                )
        sw = self._stem_width()
        if sw <= 0.0:
            raise ValueError("SlideRail.stem_width must be positive")
        max_sw = (
            self._slot_inner()
            if self.capture
            else (float(self.width) - 2.0 * float(self.clearance))
        )
        if sw > max_sw + 1e-9:
            raise ValueError(
                "SlideRail.stem_width is too wide: the stem must fit "
                + (
                    "through the capture slot (width - 2*lip - 2*clearance) to pass the lips"
                    if self.capture
                    else "between the side walls (width - 2*clearance)"
                )
            )


class SlideRailGeometry(MeshGeometry):

    def __init__(self, rail: SlideRail, *, role: str):
        super().__init__()
        if role not in ("channel", "runner"):
            raise ValueError("role must be 'channel' or 'runner'")
        rail._validate()
        cq = require_cadquery(feature="SlideRailGeometry")
        si, li, ui, s_unit, l_unit, u_unit = rail._axes()

        L = float(rail.length)
        W = float(rail.width)
        H = float(rail.height)
        wall = float(rail.wall)
        clr = float(rail.clearance)
        lip = float(rail.lip)

        def _center(so: float, lo: float, uo: float) -> Vec3:
            return (
                s_unit[0] * so + l_unit[0] * lo + u_unit[0] * uo,
                s_unit[1] * so + l_unit[1] * lo + u_unit[1] * uo,
                s_unit[2] * so + l_unit[2] * lo + u_unit[2] * uo,
            )

        principal = si >= 0

        def _dims(ds: float, dl: float, du: float) -> Tuple[float, float, float]:
            if not principal:
                return (ds, dl, du)
            d = [0.0, 0.0, 0.0]
            d[si], d[li], d[ui] = ds, dl, du
            return (d[0], d[1], d[2])

        boxes: List[Tuple[Tuple[float, float, float], float, float, float]] = []
        if role == "channel":
            boxes.append((_dims(L, W + 2 * wall, wall), 0.0, 0.0, -(H / 2 + wall / 2)))
            boxes.append((_dims(L, wall, H), 0.0, -(W / 2 + wall / 2), 0.0))
            boxes.append((_dims(L, wall, H), 0.0, +(W / 2 + wall / 2), 0.0))
            if rail.capture:
                lip_lo = W / 2 + wall / 2 - lip / 2
                boxes.append((_dims(L, wall + lip, wall), 0.0, -lip_lo, +(H / 2 + wall / 2)))
                boxes.append((_dims(L, wall + lip, wall), 0.0, +lip_lo, +(H / 2 + wall / 2)))
        else:
            Lr = rail._runner_length()
            boxes.append((_dims(Lr, W - 2 * clr, H - 2 * clr), 0.0, 0.0, 0.0))
            mount = rail._mount_offset()
            if mount > (H / 2 - clr):
                stem_dl = rail._stem_width()
                boxes.append((_dims(Lr, stem_dl, mount), 0.0, 0.0, mount / 2.0))

        shape = None
        collision_prims: List[Tuple[object, Origin]] = []
        frame = None if principal else cq.Plane(origin=(0, 0, 0), xDir=s_unit, normal=u_unit)
        rpy = (0.0, 0.0, 0.0) if principal else _frame_rpy(s_unit, l_unit, u_unit)
        for dims, so, lo, uo in boxes:
            center = _center(so, lo, uo)
            plane = cq.Workplane("XY") if principal else cq.Workplane(frame)
            solid = plane.box(dims[0], dims[1], dims[2]).translate(center)
            shape = solid if shape is None else shape.union(solid)
            collision_prims.append((Box(dims), Origin(xyz=center, rpy=rpy)))

        if shape is None:
            raise ValueError("SlideRailGeometry produced no geometry")

        _adopt_mesh_geometry(self, _mesh_geometry_from_cadquery_model(shape))
        _set_collision_primitives(self, collision_prims)
