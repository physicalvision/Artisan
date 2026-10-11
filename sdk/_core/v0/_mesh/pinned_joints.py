
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence, Tuple

from sdk._dependencies import require_cadquery

from ..types import Box, Cylinder, Origin
from .cadquery_helpers import _mesh_geometry_from_cadquery_model
from .common import _rpy_from_matrix, _set_collision_primitives
from .primitives import MeshGeometry, _adopt_mesh_geometry

Vec3 = Tuple[float, float, float]
PinnedJointStyle = Literal["clevis", "pivot_fork", "trunnion"]
PinnedJointRole = Literal["support", "member"]
PinnedJointRetention = Literal["permanent", "removable"]

__all__ = ["PinnedJointSpec", "PinnedJointGeometry"]


def _vec3(value: Sequence[float], *, name: str) -> Vec3:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return (float(value[0]), float(value[1]), float(value[2]))


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize(value: Vec3, *, name: str) -> Vec3:
    magnitude = math.sqrt(_dot(value, value))
    if magnitude <= 1e-12:
        raise ValueError(f"{name} must be non-zero")
    return tuple(component / magnitude for component in value)  # type: ignore[return-value]


@dataclass(frozen=True)
class PinnedJointSpec:

    style: PinnedJointStyle = "clevis"
    axis: Vec3 = (1.0, 0.0, 0.0)
    anchor: Vec3 = (0.0, 0.0, 0.0)
    opening_direction: Optional[Vec3] = None
    pin_diameter: float = 0.010
    lug_diameter: float = 0.024
    member_width: float = 0.014
    cheek_thickness: float = 0.007
    radial_clearance: float = 5e-4
    axial_clearance: float = 5e-4
    support_arm_length: float = 0.040
    support_mount_thickness: float = 0.008
    member_neck_length: float = 0.030
    member_neck_thickness: float = 0.010
    retention: PinnedJointRetention = "permanent"

    def __post_init__(self) -> None:
        if self.style not in {"clevis", "pivot_fork", "trunnion"}:
            raise ValueError("style must be 'clevis', 'pivot_fork', or 'trunnion'")
        if self.retention not in {"permanent", "removable"}:
            raise ValueError("retention must be 'permanent' or 'removable'")
        if self.style == "trunnion" and self.retention != "permanent":
            raise ValueError("trunnion joints require retention='permanent'")
        _vec3(self.anchor, name="anchor")
        self._frame_matrix()

        positive = {
            "pin_diameter": self.pin_diameter,
            "lug_diameter": self.lug_diameter,
            "member_width": self.member_width,
            "cheek_thickness": self.cheek_thickness,
            "radial_clearance": self.radial_clearance,
            "axial_clearance": self.axial_clearance,
            "support_arm_length": self.support_arm_length,
            "support_mount_thickness": self.support_mount_thickness,
            "member_neck_length": self.member_neck_length,
            "member_neck_thickness": self.member_neck_thickness,
        }
        for name, value in positive.items():
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive")

        bore_diameter = float(self.pin_diameter) + 2.0 * float(self.radial_clearance)
        if bore_diameter >= float(self.lug_diameter):
            raise ValueError("pin diameter plus radial clearance leaves no lug wall")
        if float(self.member_neck_thickness) >= float(self.lug_diameter):
            raise ValueError("member_neck_thickness must be less than lug_diameter")
        if float(self.support_mount_thickness) >= float(self.support_arm_length):
            raise ValueError("support_mount_thickness must be less than support_arm_length")
        if (
            float(self.support_arm_length) - float(self.support_mount_thickness)
            <= float(self.lug_diameter) * 0.5
        ):
            raise ValueError("support_arm_length leaves no arm between mount and joint")

    def _frame_matrix(self) -> tuple[Vec3, Vec3, Vec3]:
        axis = _normalize(_vec3(self.axis, name="axis"), name="axis")
        if self.opening_direction is None:
            references = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
            reference = min(references, key=lambda candidate: abs(_dot(axis, candidate)))
        else:
            reference = _vec3(self.opening_direction, name="opening_direction")
        projected = tuple(
            reference[index] - _dot(reference, axis) * axis[index] for index in range(3)
        )
        opening = _normalize(projected, name="opening_direction perpendicular to axis")
        normal = _normalize(_cross(axis, opening), name="joint frame normal")
        return (
            (axis[0], opening[0], normal[0]),
            (axis[1], opening[1], normal[1]),
            (axis[2], opening[2], normal[2]),
        )

    @property
    def joint_axis_local(self) -> Vec3:
        return (1.0, 0.0, 0.0)

    @property
    def gap_width(self) -> float:
        return float(self.member_width) + 2.0 * float(self.axial_clearance)

    @property
    def overall_width(self) -> float:
        return self.gap_width + 2.0 * float(self.cheek_thickness)

    def joint_origin(self) -> Origin:
        return Origin(
            xyz=_vec3(self.anchor, name="anchor"), rpy=_rpy_from_matrix(self._frame_matrix())
        )


def _box(cq: Any, size: Vec3, center: Vec3) -> Any:
    return cq.Workplane("XY").box(*size).translate(center)


def _x_cylinder(cq: Any, radius: float, length: float, center_x: float = 0.0) -> Any:
    return (
        cq.Workplane("YZ")
        .circle(float(radius))
        .extrude(float(length) * 0.5, both=True)
        .translate((float(center_x), 0.0, 0.0))
    )


class PinnedJointGeometry(MeshGeometry):

    def __init__(self, spec: PinnedJointSpec, *, role: PinnedJointRole):
        super().__init__()
        if role not in {"support", "member"}:
            raise ValueError("role must be 'support' or 'member'")
        self.spec = spec
        self.role = role
        cq = require_cadquery(feature="PinnedJointGeometry")
        shape, collision_primitives = (
            self._support_shape(cq) if role == "support" else self._member_shape(cq)
        )
        _adopt_mesh_geometry(self, _mesh_geometry_from_cadquery_model(shape))
        _set_collision_primitives(self, collision_primitives)

    def mount_local(self) -> Origin:
        if self.role == "support":
            return Origin(xyz=(0.0, -float(self.spec.support_arm_length), 0.0))
        return Origin(
            xyz=(
                0.0,
                float(self.spec.lug_diameter) * 0.5 + float(self.spec.member_neck_length),
                0.0,
            )
        )

    def mount_size(self) -> tuple[float, float]:
        if self.role == "support":
            return (self.spec.overall_width, float(self.spec.lug_diameter))
        return (float(self.spec.member_width), float(self.spec.member_neck_thickness))

    def _support_shape(self, cq: Any) -> tuple[Any, list[tuple[object, Origin]]]:
        spec = self.spec
        lug_radius = float(spec.lug_diameter) * 0.5
        pin_radius = float(spec.pin_diameter) * 0.5
        bore_radius = pin_radius + float(spec.radial_clearance)
        cheek = float(spec.cheek_thickness)
        gap = spec.gap_width
        overall = spec.overall_width
        arm_length = float(spec.support_arm_length)
        mount = float(spec.support_mount_thickness)
        x_centers = (-(gap + cheek) * 0.5, (gap + cheek) * 0.5)
        bridge = _box(
            cq,
            (overall, mount, 2.0 * lug_radius),
            (0.0, -arm_length + mount * 0.5, 0.0),
        )
        shape = bridge
        collision_primitives: list[tuple[object, Origin]] = [
            (
                Box((overall, mount, 2.0 * lug_radius)),
                Origin(xyz=(0.0, -arm_length + mount * 0.5, 0.0)),
            )
        ]

        for x_center in x_centers:
            if spec.style == "clevis":
                stem_min = -arm_length + mount
                stem_max = -lug_radius * 0.75
                stem_length = stem_max - stem_min
                stem = _box(
                    cq,
                    (cheek, stem_length, 2.0 * lug_radius),
                    (x_center, (stem_min + stem_max) * 0.5, 0.0),
                )
                boss = _x_cylinder(cq, lug_radius, cheek, x_center)
                tine = stem.union(boss)
                collision_stem_max = -lug_radius
                collision_stem_length = collision_stem_max - stem_min
                collision_primitives.extend(
                    [
                        (
                            Box((cheek, collision_stem_length, 2.0 * lug_radius)),
                            Origin(
                                xyz=(
                                    x_center,
                                    (stem_min + collision_stem_max) * 0.5,
                                    0.0,
                                )
                            ),
                        ),
                        (
                            Cylinder(radius=lug_radius, length=cheek),
                            Origin(xyz=(x_center, 0.0, 0.0), rpy=(0.0, math.pi * 0.5, 0.0)),
                        ),
                    ]
                )
            else:
                tine_min = -arm_length + mount
                tine_max = lug_radius
                tine_length = tine_max - tine_min
                tine = _box(
                    cq,
                    (cheek, tine_length, 2.0 * lug_radius),
                    (x_center, (tine_min + tine_max) * 0.5, 0.0),
                )
                if spec.style == "trunnion":
                    collision_primitives.extend(
                        self._trunnion_cheek_primitives(
                            x_center=x_center,
                            tine_min=tine_min,
                            tine_max=tine_max,
                            bore_radius=bore_radius,
                        )
                    )
                else:
                    collision_primitives.append(
                        (
                            Box((cheek, tine_length, 2.0 * lug_radius)),
                            Origin(xyz=(x_center, (tine_min + tine_max) * 0.5, 0.0)),
                        )
                    )
            shape = shape.union(tine)

        bore = _x_cylinder(cq, bore_radius, overall + 0.004)
        shape = shape.cut(bore)
        if spec.style != "trunnion":
            shape = shape.union(self._retained_pin_shape(cq))
        return shape, collision_primitives

    def _trunnion_cheek_primitives(
        self,
        *,
        x_center: float,
        tine_min: float,
        tine_max: float,
        bore_radius: float,
    ) -> list[tuple[object, Origin]]:
        cheek = float(self.spec.cheek_thickness)
        lug_radius = float(self.spec.lug_diameter) * 0.5
        rear_length = -bore_radius - tine_min
        front_length = tine_max - bore_radius
        side_height = lug_radius - bore_radius
        return [
            (
                Box((cheek, rear_length, 2.0 * lug_radius)),
                Origin(xyz=(x_center, tine_min + rear_length * 0.5, 0.0)),
            ),
            (
                Box((cheek, front_length, 2.0 * lug_radius)),
                Origin(xyz=(x_center, bore_radius + front_length * 0.5, 0.0)),
            ),
            (
                Box((cheek, 2.0 * bore_radius, side_height)),
                Origin(xyz=(x_center, 0.0, bore_radius + side_height * 0.5)),
            ),
            (
                Box((cheek, 2.0 * bore_radius, side_height)),
                Origin(xyz=(x_center, 0.0, -bore_radius - side_height * 0.5)),
            ),
        ]

    def _retained_pin_shape(self, cq: Any) -> Any:
        spec = self.spec
        pin_radius = float(spec.pin_diameter) * 0.5
        head_radius = min(pin_radius * 1.6, float(spec.lug_diameter) * 0.45)
        head_thickness = max(float(spec.pin_diameter) * 0.18, 8e-4)
        overall = spec.overall_width
        pin = _x_cylinder(cq, pin_radius, overall + 2.0 * head_thickness)
        left_head = _x_cylinder(cq, head_radius, head_thickness, -(overall + head_thickness) * 0.5)
        if spec.retention == "permanent":
            right_radius = head_radius
        else:
            right_radius = min(head_radius, pin_radius * 1.25)
        right_head = _x_cylinder(cq, right_radius, head_thickness, (overall + head_thickness) * 0.5)
        return pin.union(left_head).union(right_head)

    def _member_shape(self, cq: Any) -> tuple[Any, list[tuple[object, Origin]]]:
        spec = self.spec
        lug_radius = float(spec.lug_diameter) * 0.5
        member_width = float(spec.member_width)
        pin_radius = float(spec.pin_diameter) * 0.5
        bore_radius = pin_radius + float(spec.radial_clearance)
        neck_length = float(spec.member_neck_length)
        neck_thickness = float(spec.member_neck_thickness)

        lug = _x_cylinder(cq, lug_radius, member_width)
        if spec.style != "trunnion":
            lug = lug.cut(_x_cylinder(cq, bore_radius, member_width + 0.004))
        visual_embed = min(lug_radius * 0.2, neck_length * 0.1)
        neck = _box(
            cq,
            (member_width, neck_length + visual_embed, neck_thickness),
            (0.0, lug_radius + (neck_length - visual_embed) * 0.5, 0.0),
        )
        shape = lug.union(neck)
        collision_primitives: list[tuple[object, Origin]] = [
            (
                Cylinder(radius=lug_radius, length=member_width),
                Origin(rpy=(0.0, math.pi * 0.5, 0.0)),
            ),
            (
                Box((member_width, neck_length, neck_thickness)),
                Origin(xyz=(0.0, lug_radius + neck_length * 0.5, 0.0)),
            ),
        ]

        if spec.style == "trunnion":
            overall = spec.overall_width
            side_length = (overall - member_width) * 0.5
            shaft = _x_cylinder(cq, pin_radius, overall)
            shape = shape.union(shaft)
            for sign in (-1.0, 1.0):
                collision_primitives.append(
                    (
                        Cylinder(radius=pin_radius, length=side_length),
                        Origin(
                            xyz=(sign * (member_width * 0.5 + side_length * 0.5), 0.0, 0.0),
                            rpy=(0.0, math.pi * 0.5, 0.0),
                        ),
                    )
                )
        return shape, collision_primitives
