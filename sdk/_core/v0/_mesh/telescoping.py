from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence, Tuple

from sdk._dependencies import require_cadquery

from ..types import Box, Cylinder, Origin
from .cadquery_helpers import _mesh_geometry_from_cadquery_model
from .common import _set_collision_primitives
from .primitives import MeshGeometry, _adopt_mesh_geometry
from .rotary import _annular_box_primitives

Vec3 = Tuple[float, float, float]
TelescopingProfile = Literal["rect_tube", "round_tube", "captured_c_channel"]
EndCap = Literal["none", "base", "tip"]
CouplingMode = Literal["independent", "proportional"]

__all__ = [
    "TelescopingMemberSpec",
    "TelescopingInterfaceSpec",
    "TelescopingSlideSpec",
    "TelescopingMemberGeometry",
]


@dataclass(frozen=True)
class TelescopingMemberSpec:

    profile: TelescopingProfile
    outer_width: float
    outer_height: float
    wall: float
    length: float
    lip: float = 0.0
    solid: bool = False
    end_cap: EndCap = "none"

    def __post_init__(self) -> None:
        if self.profile not in {"rect_tube", "round_tube", "captured_c_channel"}:
            raise ValueError("profile must be 'rect_tube', 'round_tube', or 'captured_c_channel'")
        if self.end_cap not in {"none", "base", "tip"}:
            raise ValueError("end_cap must be 'none', 'base', or 'tip'")
        for name in ("outer_width", "outer_height", "wall", "length"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not self.solid and 2.0 * float(self.wall) >= min(
            float(self.outer_width), float(self.outer_height)
        ):
            raise ValueError("wall leaves no inner cavity")
        if self.solid and self.end_cap != "none":
            raise ValueError("solid members do not use end_cap")
        if self.profile in {"rect_tube", "round_tube"} and float(self.lip) != 0.0:
            raise ValueError(f"{self.profile} requires lip=0")
        if self.profile == "round_tube" and not math.isclose(
            float(self.outer_width),
            float(self.outer_height),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "round_tube requires outer_width == outer_height; both values are the diameter"
            )
        if self.profile == "captured_c_channel" and not self.solid:
            if float(self.lip) <= 0.0:
                raise ValueError("captured_c_channel requires a positive lip")
            if 2.0 * float(self.lip) >= self.inner_width:
                raise ValueError("captured_c_channel lips close the channel mouth")

    @property
    def inner_width(self) -> float:
        return float(self.outer_width) - 2.0 * float(self.wall)

    @property
    def inner_height(self) -> float:
        return float(self.outer_height) - 2.0 * float(self.wall)

    @property
    def mouth_width(self) -> float:
        if self.profile == "captured_c_channel":
            return self.inner_width - 2.0 * float(self.lip)
        return self.inner_width

    @property
    def outer_diameter(self) -> float:
        if self.profile != "round_tube":
            raise ValueError("outer_diameter is only defined for round_tube")
        return float(self.outer_width)

    @property
    def inner_diameter(self) -> float:
        if self.profile != "round_tube":
            raise ValueError("inner_diameter is only defined for round_tube")
        return self.inner_width


@dataclass(frozen=True)
class TelescopingInterfaceSpec:

    parent_index: int
    child_index: int
    clearance: float
    home_offset: float
    travel: float
    retained_insertion: float
    stop_mode: Literal["joint_limit"] = "joint_limit"

    def __post_init__(self) -> None:
        if int(self.parent_index) < 0 or int(self.child_index) < 0:
            raise ValueError("parent_index and child_index must be non-negative")
        if int(self.child_index) != int(self.parent_index) + 1:
            raise ValueError("interfaces must connect adjacent parent/child member indices")
        if float(self.clearance) <= 0.0:
            raise ValueError("clearance must be positive")
        if float(self.travel) <= 0.0:
            raise ValueError("travel must be positive")
        if float(self.retained_insertion) <= 0.0:
            raise ValueError("retained_insertion must be positive")
        if self.stop_mode != "joint_limit":
            raise ValueError("V1 supports stop_mode='joint_limit' only")


@dataclass(frozen=True)
class TelescopingSlideSpec:

    members: Sequence[TelescopingMemberSpec]
    interfaces: Sequence[TelescopingInterfaceSpec]
    coupling_mode: CouplingMode = "independent"

    def __post_init__(self) -> None:
        members = tuple(self.members)
        interfaces = tuple(self.interfaces)
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "interfaces", interfaces)
        if len(members) < 2:
            raise ValueError("TelescopingSlideSpec requires at least two members")
        if len(interfaces) != len(members) - 1:
            raise ValueError("interface_count must equal member_count - 1")
        if self.coupling_mode not in {"independent", "proportional"}:
            raise ValueError("coupling_mode must be 'independent' or 'proportional'")
        profiles = {member.profile for member in members}
        if len(profiles) != 1:
            raise ValueError("V1 requires all telescoping members to use the same profile")

        for index, member in enumerate(members):
            if member.solid and index != len(members) - 1:
                raise ValueError("only the innermost telescoping member may be solid")
            if index == 0 and member.end_cap not in {"none", "base"}:
                raise ValueError("the outer member may only use a base end cap")
            if 0 < index < len(members) - 1 and member.end_cap != "none":
                raise ValueError("middle members must stay open at both axial ends")
            if index == len(members) - 1 and member.end_cap not in {"none", "tip"}:
                raise ValueError("the innermost member may only use a tip end cap")

        for expected_parent, interface in enumerate(interfaces):
            if interface.parent_index != expected_parent:
                raise ValueError("interfaces must be ordered from outermost to innermost")
            self._validate_interface(interface)

    @property
    def member_count(self) -> int:
        return len(self.members)

    @property
    def interface_count(self) -> int:
        return len(self.interfaces)

    @property
    def joint_axis_local(self) -> Vec3:
        return (1.0, 0.0, 0.0)

    def joint_origin(self, interface_index: int) -> Origin:
        interface = self.interfaces[interface_index]
        return Origin(xyz=(float(interface.home_offset), 0.0, 0.0))

    def joint_range(self, interface_index: int) -> tuple[float, float]:
        return (0.0, float(self.interfaces[interface_index].travel))

    def insertion_at(self, interface_index: int, q: float) -> float:
        interface = self.interfaces[interface_index]
        parent = self.members[interface.parent_index]
        child = self.members[interface.child_index]
        child_start = float(interface.home_offset) + float(q)
        child_end = child_start + float(child.length)
        return max(0.0, min(float(parent.length), child_end) - max(0.0, child_start))

    def proportional_multipliers(self) -> tuple[float, ...]:
        root_travel = float(self.interfaces[0].travel)
        return tuple(float(interface.travel) / root_travel for interface in self.interfaces)

    def _validate_interface(self, interface: TelescopingInterfaceSpec) -> None:
        parent = self.members[interface.parent_index]
        child = self.members[interface.child_index]
        clearance = float(interface.clearance)
        if float(child.outer_width) + 2.0 * clearance > parent.inner_width + 1e-12:
            raise ValueError("child outer_width does not fit parent cavity with clearance")
        if float(child.outer_height) + 2.0 * clearance > parent.inner_height + 1e-12:
            raise ValueError("child outer_height does not fit parent cavity with clearance")
        if parent.profile == "captured_c_channel" and (
            float(child.outer_width) <= parent.mouth_width + 2.0 * clearance
        ):
            raise ValueError("child is narrow enough to escape the captured channel mouth")
        max_insertion = min(float(parent.length), float(child.length))
        if float(interface.retained_insertion) > max_insertion:
            raise ValueError("retained_insertion exceeds the shorter member length")
        for q, label in ((0.0, "home"), (float(interface.travel), "upper")):
            insertion = self.insertion_at(interface.parent_index, q)
            if insertion + 1e-12 < float(interface.retained_insertion):
                raise ValueError(
                    f"{label} pose insertion {insertion:g} is below retained_insertion "
                    f"{float(interface.retained_insertion):g}"
                )
        if parent.end_cap == "base" and (
            float(interface.home_offset) < float(parent.wall) + clearance
        ):
            raise ValueError("home_offset collides with the parent's base end cap")


class TelescopingMemberGeometry(MeshGeometry):

    def __init__(self, slide: TelescopingSlideSpec, *, member_index: int):
        super().__init__()
        if member_index < 0 or member_index >= slide.member_count:
            raise ValueError("member_index is out of range")
        self.slide = slide
        self.member_index = int(member_index)
        self.member = slide.members[member_index]
        cq = require_cadquery(feature="TelescopingMemberGeometry")
        if self.member.profile == "round_tube":
            shape, primitives = self._round_geometry(cq)
        else:
            boxes = self._boxes()
            shape = None
            primitives = []
            for size, center in boxes:
                solid = cq.Workplane("XY").box(*size).translate(center)
                shape = solid if shape is None else shape.union(solid)
                primitives.append((Box(size), Origin(xyz=center)))
        if shape is None:
            raise ValueError("TelescopingMemberGeometry produced no geometry")
        _adopt_mesh_geometry(self, _mesh_geometry_from_cadquery_model(shape))
        _set_collision_primitives(self, primitives)

    def mount_local(self, end: Literal["base", "tip"] = "tip") -> Origin:
        if end not in {"base", "tip"}:
            raise ValueError("end must be 'base' or 'tip'")
        x = 0.0 if end == "base" else float(self.member.length)
        return Origin(xyz=(x, 0.0, 0.0))

    def mount_size(self) -> tuple[float, float]:
        return (float(self.member.outer_width), float(self.member.outer_height))

    def _round_geometry(self, cq: Any) -> tuple[Any, list[tuple[object, Origin]]]:
        member = self.member
        length = float(member.length)
        wall = float(member.wall)
        outer_radius = member.outer_diameter * 0.5
        inner_radius = member.inner_diameter * 0.5
        cylinder_rpy = (0.0, math.pi * 0.5, 0.0)

        if member.solid:
            shape = cq.Workplane("YZ").circle(outer_radius).extrude(length)
            return shape, [
                (
                    Cylinder(radius=outer_radius, length=length),
                    Origin(xyz=(length * 0.5, 0.0, 0.0), rpy=cylinder_rpy),
                )
            ]

        wall_start = 0.0
        wall_length = length
        cap_start: float | None = None
        if member.end_cap == "base":
            wall_start = wall
            wall_length = length - wall
            cap_start = 0.0
        elif member.end_cap == "tip":
            wall_length = length - wall
            cap_start = length - wall
        if wall_length <= 0.0:
            raise ValueError("end cap leaves no axial wall length")

        shape = (
            cq.Workplane("YZ")
            .circle(outer_radius)
            .circle(inner_radius)
            .extrude(wall_length)
            .translate((wall_start, 0.0, 0.0))
        )
        primitives = _annular_box_primitives(
            axis="x",
            outer_radius=outer_radius,
            cavity_radius=inner_radius,
            axial_length=wall_length,
            axial_center=wall_start + wall_length * 0.5,
        )
        if cap_start is not None:
            cap = (
                cq.Workplane("YZ")
                .circle(outer_radius)
                .extrude(wall)
                .translate((cap_start, 0.0, 0.0))
            )
            shape = shape.union(cap)
            primitives.append(
                (
                    Cylinder(radius=outer_radius, length=wall),
                    Origin(xyz=(cap_start + wall * 0.5, 0.0, 0.0), rpy=cylinder_rpy),
                )
            )
        return shape, primitives

    def _boxes(self) -> list[tuple[Vec3, Vec3]]:
        member = self.member
        length = float(member.length)
        width = float(member.outer_width)
        height = float(member.outer_height)
        wall = float(member.wall)
        if member.solid:
            return [((length, width, height), (length * 0.5, 0.0, 0.0))]

        wall_start = 0.0
        wall_length = length
        cap: tuple[Vec3, Vec3] | None = None
        if member.end_cap == "base":
            wall_start = wall
            wall_length = length - wall
            cap = ((wall, width, height), (wall * 0.5, 0.0, 0.0))
        elif member.end_cap == "tip":
            wall_length = length - wall
            cap = ((wall, width, height), (length - wall * 0.5, 0.0, 0.0))
        if wall_length <= 0.0:
            raise ValueError("end cap leaves no axial wall length")
        x_center = wall_start + wall_length * 0.5

        if member.profile == "rect_tube":
            boxes = [
                (
                    (wall_length, width, wall),
                    (x_center, 0.0, -(height - wall) * 0.5),
                ),
                (
                    (wall_length, width, wall),
                    (x_center, 0.0, (height - wall) * 0.5),
                ),
                (
                    (wall_length, wall, height - 2.0 * wall),
                    (x_center, -(width - wall) * 0.5, 0.0),
                ),
                (
                    (wall_length, wall, height - 2.0 * wall),
                    (x_center, (width - wall) * 0.5, 0.0),
                ),
            ]
        else:
            lip = float(member.lip)
            boxes = [
                (
                    (wall_length, width, wall),
                    (x_center, 0.0, -(height - wall) * 0.5),
                ),
                (
                    (wall_length, wall, height - wall),
                    (x_center, -(width - wall) * 0.5, wall * 0.5),
                ),
                (
                    (wall_length, wall, height - wall),
                    (x_center, (width - wall) * 0.5, wall * 0.5),
                ),
                (
                    (wall_length, lip, wall),
                    (x_center, -(width * 0.5 - wall - lip * 0.5), (height - wall) * 0.5),
                ),
                (
                    (wall_length, lip, wall),
                    (x_center, (width * 0.5 - wall - lip * 0.5), (height - wall) * 0.5),
                ),
            ]
        if cap is not None:
            boxes.append(cap)
        return boxes
