from __future__ import annotations

import math
from dataclasses import dataclass

from sdk._dependencies import require_cadquery

from ..types import Box, Cylinder, Origin
from .cadquery_helpers import _mesh_geometry_from_cadquery_model
from .common import _set_collision_primitives
from .primitives import MeshGeometry, _adopt_mesh_geometry

__all__ = ["ThreadedSleeveFit", "ThreadedSleeveGeometry"]


@dataclass(frozen=True)
class ThreadedSleeveFit:

    male_radius: float
    engagement_length: float
    clearance: float = 8e-4
    thread_depth: float = 8e-4
    thread_pitch: float = 3e-3
    wall: float = 2e-3
    socket_top: float = 2e-3
    collider_segments: int = 16

    @property
    def male_outer_radius(self) -> float:
        return float(self.male_radius) + float(self.thread_depth)

    @property
    def socket_inner_radius(self) -> float:
        return self.male_outer_radius + float(self.clearance)

    @property
    def socket_outer_radius(self) -> float:
        return self.socket_inner_radius + float(self.wall)

    @property
    def axis(self) -> tuple[float, float, float]:
        return (0.0, 0.0, 1.0)

    def _validate(self) -> None:
        for name in (
            "male_radius",
            "engagement_length",
            "clearance",
            "thread_depth",
            "thread_pitch",
            "wall",
            "socket_top",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"ThreadedSleeveFit.{name} must be a positive finite value")
        if int(self.collider_segments) < 8:
            raise ValueError("ThreadedSleeveFit.collider_segments must be at least 8")
        if float(self.thread_depth) >= float(self.male_radius):
            raise ValueError("ThreadedSleeveFit.thread_depth must be smaller than male_radius")


class ThreadedSleeveGeometry(MeshGeometry):

    def __init__(self, fit: ThreadedSleeveFit, *, role: str):
        super().__init__()
        if role not in {"plug", "socket"}:
            raise ValueError("role must be 'plug' or 'socket'")
        fit._validate()
        cq = require_cadquery(feature="ThreadedSleeveGeometry")

        length = float(fit.engagement_length)
        if role == "plug":
            shape = cq.Workplane("XY").cylinder(
                length,
                float(fit.male_radius),
                centered=(True, True, True),
            )
            ridge_count = max(1, int(length / float(fit.thread_pitch)))
            ridge_height = min(float(fit.thread_pitch) * 0.45, length / ridge_count * 0.8)
            for index in range(ridge_count):
                z = -length / 2.0 + (index + 0.5) * length / ridge_count
                ridge = (
                    cq.Workplane("XY")
                    .cylinder(
                        ridge_height,
                        fit.male_outer_radius,
                        centered=(True, True, True),
                    )
                    .translate((0.0, 0.0, z))
                )
                shape = shape.union(ridge)
            collision_primitives = [
                (Cylinder(radius=fit.male_outer_radius, length=length), Origin())
            ]
        else:
            outer = cq.Workplane("XY").cylinder(
                length,
                fit.socket_outer_radius,
                centered=(True, True, True),
            )
            cavity = cq.Workplane("XY").cylinder(
                length + 2.0 * float(fit.socket_top),
                fit.socket_inner_radius,
                centered=(True, True, True),
            )
            shape = outer.cut(cavity)
            top_z = length / 2.0 + float(fit.socket_top) / 2.0
            top = (
                cq.Workplane("XY")
                .cylinder(
                    float(fit.socket_top),
                    fit.socket_outer_radius,
                    centered=(True, True, True),
                )
                .translate((0.0, 0.0, top_z))
            )
            shape = shape.union(top)

            collision_primitives = []
            segments = int(fit.collider_segments)
            radius = fit.socket_inner_radius + float(fit.wall) / 2.0
            tangent = 2.0 * fit.socket_inner_radius * math.tan(math.pi / segments)
            for index in range(segments):
                theta = 2.0 * math.pi * index / segments
                center = (radius * math.cos(theta), radius * math.sin(theta), 0.0)
                collision_primitives.append(
                    (
                        Box((float(fit.wall), tangent, length)),
                        Origin(xyz=center, rpy=(0.0, 0.0, theta)),
                    )
                )
            collision_primitives.append(
                (
                    Cylinder(radius=fit.socket_outer_radius, length=float(fit.socket_top)),
                    Origin(xyz=(0.0, 0.0, top_z)),
                )
            )

        _adopt_mesh_geometry(self, _mesh_geometry_from_cadquery_model(shape))
        _set_collision_primitives(self, collision_primitives)
