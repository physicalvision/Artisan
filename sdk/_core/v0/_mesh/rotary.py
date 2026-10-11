from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Tuple

from sdk._dependencies import require_cadquery

from ..types import Box, Cylinder, Origin
from .cadquery_helpers import _mesh_geometry_from_cadquery_model
from .common import _set_collision_primitives, _set_required_collision_cavity
from .primitives import MeshGeometry, _adopt_mesh_geometry

Vec3 = Tuple[float, float, float]
RotaryMountMode = Literal["free_on_axle", "rigid_to_shaft"]
ShaftProfile = Literal["round", "keyed", "d_shaft", "double_d", "splined", "hex"]
RotaryRole = Literal["stationary", "rotor"]
PrincipalAxis = Literal["x", "y", "z"]

__all__ = ["RotaryConnectorSpec", "RotaryConnectorGeometry"]


@dataclass(frozen=True)
class RotaryConnectorSpec:

    mount_mode: RotaryMountMode = "free_on_axle"
    shaft_diameter: float = 0.010
    bearing_length: float = 0.020
    housing_outer_diameter: float = 0.024
    radial_clearance: float = 5e-4
    axial_clearance: float = 5e-4
    mount_length: float = 0.012
    mount_fit_clearance: float = 1e-4
    shaft_profile: ShaftProfile = "round"

    def __post_init__(self) -> None:
        if self.mount_mode not in {"free_on_axle", "rigid_to_shaft"}:
            raise ValueError("mount_mode must be 'free_on_axle' or 'rigid_to_shaft'")
        if self.shaft_profile not in {
            "round",
            "keyed",
            "d_shaft",
            "double_d",
            "splined",
            "hex",
        }:
            raise ValueError("unsupported shaft_profile")
        if self.mount_mode == "free_on_axle" and self.shaft_profile != "round":
            raise ValueError("free_on_axle requires shaft_profile='round'")
        positive = {
            "shaft_diameter": self.shaft_diameter,
            "bearing_length": self.bearing_length,
            "housing_outer_diameter": self.housing_outer_diameter,
            "radial_clearance": self.radial_clearance,
            "axial_clearance": self.axial_clearance,
            "mount_length": self.mount_length,
            "mount_fit_clearance": self.mount_fit_clearance,
        }
        for name, value in positive.items():
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.bearing_bore_diameter >= float(self.housing_outer_diameter):
            raise ValueError("housing_outer_diameter leaves no bearing wall")

    @property
    def shaft_radius(self) -> float:
        return float(self.shaft_diameter) * 0.5

    @property
    def bearing_bore_diameter(self) -> float:
        return float(self.shaft_diameter) + 2.0 * float(self.radial_clearance)

    @property
    def consumer_bore_diameter(self) -> float:
        if self.mount_mode == "free_on_axle":
            return self.bearing_bore_diameter
        return float(self.shaft_diameter) + 2.0 * float(self.mount_fit_clearance)

    def consumer_origin(
        self,
        *,
        axis: PrincipalAxis,
        axial_length: float,
        side: Literal[-1, 1] = 1,
    ) -> Origin:
        if float(axial_length) <= 0.0:
            raise ValueError("axial_length must be positive")
        offset = 0.0
        if self.mount_mode == "rigid_to_shaft":
            offset = float(side) * (
                float(self.bearing_length) * 0.5
                + float(self.axial_clearance)
                + float(axial_length) * 0.5
            )
        xyz = [0.0, 0.0, 0.0]
        xyz[{"x": 0, "y": 1, "z": 2}[axis]] = offset
        return Origin(xyz=(xyz[0], xyz[1], xyz[2]))


def _axis_index(axis: PrincipalAxis) -> int:
    return {"x": 0, "y": 1, "z": 2}[axis]


def _axis_vector(axis: PrincipalAxis) -> Vec3:
    return {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}[axis]


def _axis_rpy(axis: PrincipalAxis) -> Vec3:
    if axis == "x":
        return (0.0, math.pi * 0.5, 0.0)
    if axis == "y":
        return (-math.pi * 0.5, 0.0, 0.0)
    return (0.0, 0.0, 0.0)


def _orient_shape(shape: Any, axis: PrincipalAxis) -> Any:
    if axis == "x":
        return shape.rotate((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 90.0)
    if axis == "y":
        return shape.rotate((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), -90.0)
    return shape


def _orient_point(point: Vec3, axis: PrincipalAxis) -> Vec3:
    x, y, z = point
    if axis == "x":
        return (z, y, -x)
    if axis == "y":
        return (x, z, -y)
    return point


def _rpy_matrix(rpy: Vec3) -> tuple[tuple[float, float, float], ...]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _mat3_mul(
    left: tuple[tuple[float, float, float], ...],
    right: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(sum(left[row][k] * right[k][column] for k in range(3)) for column in range(3))
        for row in range(3)
    )


def _matrix_rpy(matrix: tuple[tuple[float, float, float], ...]) -> Vec3:
    sp = max(-1.0, min(1.0, -matrix[2][0]))
    pitch = math.asin(sp)
    if abs(sp) < 0.999999:
        roll = math.atan2(matrix[2][1], matrix[2][2])
        yaw = math.atan2(matrix[1][0], matrix[0][0])
    else:
        roll = math.atan2(-matrix[1][2], matrix[1][1])
        yaw = 0.0
    return (roll, pitch, yaw)


def _orient_origin(origin: Origin, axis: PrincipalAxis) -> Origin:
    if axis == "z":
        return origin
    rotation = _mat3_mul(_rpy_matrix(_axis_rpy(axis)), _rpy_matrix(origin.rpy))
    return Origin(xyz=_orient_point(origin.xyz, axis), rpy=_matrix_rpy(rotation))


def _z_cylinder(cq: Any, radius: float, length: float, center_z: float = 0.0) -> Any:
    return (
        cq.Workplane("XY")
        .circle(float(radius))
        .extrude(float(length) * 0.5, both=True)
        .translate((0.0, 0.0, float(center_z)))
    )


def _z_annulus(cq: Any, outer_radius: float, inner_radius: float, length: float) -> Any:
    return (
        cq.Workplane("XY")
        .circle(float(outer_radius))
        .circle(float(inner_radius))
        .extrude(float(length) * 0.5, both=True)
    )


def _z_shaft_profile(
    cq: Any,
    profile: ShaftProfile,
    diameter: float,
    length: float,
    center_z: float = 0.0,
) -> Any:
    diameter = float(diameter)
    length = float(length)
    radius = diameter * 0.5
    if profile == "round":
        shape = _z_cylinder(cq, radius, length)
    elif profile == "hex":
        shape = cq.Workplane("XY").polygon(6, diameter).extrude(length * 0.5, both=True)
    elif profile in {"d_shaft", "double_d"}:
        flat_x = radius - diameter * 0.16
        circle_points = [
            (radius * math.cos(theta), radius * math.sin(theta))
            for theta in (2.0 * math.pi * index / 64.0 for index in range(64))
        ]
        if profile == "d_shaft":
            points = [(min(x, flat_x), y) for x, y in circle_points]
        else:
            points = [(max(min(x, flat_x), -flat_x), y) for x, y in circle_points]
        shape = cq.Workplane("XY").polyline(points).close().extrude(length * 0.5, both=True)
    elif profile == "splined":
        points = []
        inner_radius = radius * 0.94
        for index in range(16):
            theta = math.pi * index / 8.0
            point_radius = radius if index % 2 == 0 else inner_radius
            points.append((point_radius * math.cos(theta), point_radius * math.sin(theta)))
        shape = cq.Workplane("XY").polyline(points).close().extrude(length * 0.5, both=True)
    else:
        key_width = diameter * 0.32
        key_inset = diameter * 0.08
        key_projection = diameter * 0.12
        key = (
            cq.Workplane("XY")
            .box(key_inset + key_projection, key_width, length)
            .translate((radius + (key_projection - key_inset) * 0.5, 0.0, 0.0))
        )
        shape = _z_cylinder(cq, radius, length).union(key)
    return shape.translate((0.0, 0.0, float(center_z)))


def _profile_inscribed_radius(spec: RotaryConnectorSpec) -> float:
    radius = spec.shaft_radius
    if spec.shaft_profile in {"d_shaft", "double_d"}:
        return radius * 0.68
    if spec.shaft_profile == "splined":
        return radius * 0.94
    if spec.shaft_profile == "hex":
        return radius * math.cos(math.pi / 6.0)
    return radius


def _annular_box_primitives(
    *,
    axis: PrincipalAxis,
    outer_radius: float,
    cavity_radius: float,
    axial_length: float,
    axial_center: float = 0.0,
) -> list[tuple[object, Origin]]:
    segments = 32
    outer = float(outer_radius)
    cavity = float(cavity_radius)
    if cavity >= outer:
        raise ValueError("cavity is too large for annular primitive fallback")
    radial_span = outer - cavity
    tangent_span = 2.0 * outer * math.sin(math.pi / segments)
    center_radius = cavity + radial_span * 0.5
    primitives: list[tuple[object, Origin]] = []
    for index in range(segments):
        theta = 2.0 * math.pi * index / segments
        origin = Origin(
            xyz=(
                center_radius * math.cos(theta),
                center_radius * math.sin(theta),
                axial_center,
            ),
            rpy=(0.0, 0.0, theta),
        )
        primitives.append(
            (
                Box((radial_span, tangent_span, axial_length)),
                _orient_origin(origin, axis),
            )
        )
    return primitives


def _configure_rotary_consumer_collision(
    geometry: MeshGeometry,
    spec: RotaryConnectorSpec,
    *,
    native_axis: PrincipalAxis,
) -> None:
    if spec.mount_mode != "free_on_axle":
        return
    axis_index = _axis_index(native_axis)
    mins = [min(vertex[index] for vertex in geometry.vertices) for index in range(3)]
    maxs = [max(vertex[index] for vertex in geometry.vertices) for index in range(3)]
    axial_length = maxs[axis_index] - mins[axis_index]
    axial_center = (mins[axis_index] + maxs[axis_index]) * 0.5
    center = [0.0, 0.0, 0.0]
    center[axis_index] = axial_center
    cross_axes = [index for index in range(3) if index != axis_index]
    outer_radius = max(
        math.hypot(vertex[cross_axes[0]], vertex[cross_axes[1]]) for vertex in geometry.vertices
    )
    _set_required_collision_cavity(
        geometry,
        origin=(center[0], center[1], center[2]),
        axis=_axis_vector(native_axis),
        radius=float(spec.consumer_bore_diameter) * 0.5,
        length=max(
            axial_length,
            float(spec.bearing_length) + 2.0 * float(spec.axial_clearance),
        ),
        fallback_primitives=_annular_box_primitives(
            axis=native_axis,
            outer_radius=outer_radius,
            cavity_radius=float(spec.consumer_bore_diameter) * 0.5,
            axial_length=axial_length,
            axial_center=axial_center,
        ),
    )


class RotaryConnectorGeometry(MeshGeometry):

    def __init__(
        self,
        spec: RotaryConnectorSpec,
        *,
        role: RotaryRole,
        axis: PrincipalAxis = "z",
    ):
        super().__init__()
        if role not in {"stationary", "rotor"}:
            raise ValueError("role must be 'stationary' or 'rotor'")
        if axis not in {"x", "y", "z"}:
            raise ValueError("axis must be 'x', 'y', or 'z'")
        self.spec = spec
        self.role = role
        self.axis = axis
        cq = require_cadquery(feature="RotaryConnectorGeometry")
        shape, collision_primitives = self._canonical_shape(cq)
        _adopt_mesh_geometry(self, _mesh_geometry_from_cadquery_model(_orient_shape(shape, axis)))
        _set_collision_primitives(self, self._orient_primitives(collision_primitives))

    def joint_axis_local(self) -> Vec3:
        return _axis_vector(self.axis)

    def mount_local(self) -> Origin:
        sign = -1.0 if self.role == "stationary" else 1.0
        clearance = (
            float(self.spec.axial_clearance)
            if self.spec.mount_mode == "free_on_axle" or self.role == "rotor"
            else 0.0
        )
        offset = sign * (
            float(self.spec.bearing_length) * 0.5 + clearance + float(self.spec.mount_length) * 0.5
        )
        xyz = [0.0, 0.0, 0.0]
        xyz[_axis_index(self.axis)] = offset
        return Origin(xyz=(xyz[0], xyz[1], xyz[2]))

    def _canonical_shape(self, cq: Any) -> tuple[Any, list[tuple[object, Origin]]]:
        spec = self.spec
        shaft_radius = spec.shaft_radius
        outer_radius = float(spec.housing_outer_diameter) * 0.5
        bearing_length = float(spec.bearing_length)
        mount_length = float(spec.mount_length)
        bore_radius = float(spec.bearing_bore_diameter) * 0.5
        mount_center = -(bearing_length + mount_length) * 0.5

        if spec.mount_mode == "free_on_axle" and self.role == "stationary":
            axle_length = bearing_length + 2.0 * float(spec.axial_clearance)
            axle = _z_cylinder(cq, shaft_radius, axle_length)
            mount_center -= float(spec.axial_clearance)
            mount = _z_cylinder(cq, outer_radius, mount_length, mount_center)
            return axle.union(mount), [
                (Cylinder(radius=shaft_radius, length=axle_length), Origin()),
                (
                    Cylinder(radius=outer_radius, length=mount_length),
                    Origin(xyz=(0.0, 0.0, mount_center)),
                ),
            ]

        if spec.mount_mode == "free_on_axle":
            sleeve = _z_annulus(cq, outer_radius, bore_radius, bearing_length)
            return sleeve, _annular_box_primitives(
                axis="z",
                outer_radius=outer_radius,
                cavity_radius=bore_radius,
                axial_length=bearing_length,
            )

        if self.role == "stationary":
            housing = _z_annulus(cq, outer_radius, bore_radius, bearing_length)
            mount = _z_annulus(cq, outer_radius * 1.25, bore_radius, mount_length).translate(
                (0.0, 0.0, mount_center)
            )
            primitives = _annular_box_primitives(
                axis="z",
                outer_radius=outer_radius,
                cavity_radius=bore_radius,
                axial_length=bearing_length,
            )
            primitives.extend(
                _annular_box_primitives(
                    axis="z",
                    outer_radius=outer_radius * 1.25,
                    cavity_radius=bore_radius,
                    axial_length=mount_length,
                    axial_center=mount_center,
                )
            )
            return housing.union(mount), primitives

        journal_length = bearing_length
        profile_length = float(spec.axial_clearance) + mount_length
        mount_center = bearing_length * 0.5 + profile_length * 0.5
        journal = _z_cylinder(cq, shaft_radius, journal_length)
        mount = _z_shaft_profile(
            cq,
            spec.shaft_profile,
            float(spec.shaft_diameter),
            profile_length,
            mount_center,
        )
        return journal.union(mount), [
            (Cylinder(radius=shaft_radius, length=journal_length), Origin()),
            (
                Cylinder(radius=_profile_inscribed_radius(spec), length=profile_length),
                Origin(xyz=(0.0, 0.0, mount_center)),
            ),
        ]

    def _orient_primitives(
        self, primitives: list[tuple[object, Origin]]
    ) -> list[tuple[object, Origin]]:
        if self.axis == "z":
            return primitives
        oriented: list[tuple[object, Origin]] = []
        for primitive, origin in primitives:
            oriented.append((primitive, _orient_origin(origin, self.axis)))
        return oriented
