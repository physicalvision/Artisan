# Multi-Member Telescoping Slides

## Contract

A telescoping slide with `M` physical members has `M-1` interfaces and `M-1`
`PRISMATIC` joints. Use:

- `TelescopingMemberSpec`: one physical wall shell or innermost solid member.
- `TelescopingInterfaceSpec`: parent/child indices, fit clearance, home offset,
  travel, and retained insertion.
- `TelescopingSlideSpec`: validates the complete nested stack.
- `TelescopingMemberGeometry(slide, member_index=...)`: one geometry per Part.

Supported profiles are:

- `rect_tube`: closed rectangular wall shell.
- `round_tube`: closed circular wall shell.
- `captured_c_channel`: open rectilinear channel with inward retaining lips.

## Three-Member Rectangular Tube

```python
from sdk import (
    ArticulationType,
    MotionLimits,
    Origin,
    TelescopingInterfaceSpec,
    TelescopingMemberGeometry,
    TelescopingMemberSpec,
    TelescopingSlideSpec,
    mesh_from_geometry,
)

slide = TelescopingSlideSpec(
    members=(
        TelescopingMemberSpec(
            "rect_tube", 0.060, 0.040, 0.004, 0.300, end_cap="base"
        ),
        TelescopingMemberSpec("rect_tube", 0.050, 0.030, 0.003, 0.280),
        TelescopingMemberSpec(
            "rect_tube", 0.040, 0.020, 0.002, 0.260, solid=True
        ),
    ),
    interfaces=(
        TelescopingInterfaceSpec(
            parent_index=0, child_index=1, clearance=5e-4,
            home_offset=0.020, travel=0.200, retained_insertion=0.080,
        ),
        TelescopingInterfaceSpec(
            parent_index=1, child_index=2, clearance=5e-4,
            home_offset=0.020, travel=0.180, retained_insertion=0.080,
        ),
    ),
)

# Every pose below comes from the shared spec: each member is built anchor-relative
# (hence the identity Origin) and each joint frame is slide.joint_origin(index).
# There is no assembly coordinate to hand-tune here. If you mount this assembly ONTO
# a host — a column face, a carcass underside — that mounting pose DOES follow the
# host, so author it relationally; see 50_placement.md.
names = ("outer", "middle", "inner")
for index, name in enumerate(names):
    model.part(name).visual(
        mesh_from_geometry(
            TelescopingMemberGeometry(slide, member_index=index), f"{name}_member"
        ),
        origin=Origin(),
        material=steel,
        name=f"{name}_member",
    )

for index, (parent, child) in enumerate(zip(names, names[1:])):
    lower, upper = slide.joint_range(index)
    model.articulation(
        f"{parent}_to_{child}",
        ArticulationType.PRISMATIC,
        parent=parent,
        child=child,
        origin=slide.joint_origin(index),
        axis=slide.joint_axis_local,
        motion_limits=MotionLimits(
            lower=lower, upper=upper, effort=80.0, velocity=0.5
        ),
    )
```

Geometry is canonical local `+X`. Member `i+1` starts at
`home_offset + q` in member `i`; both its home and upper-limit insertion are
validated against `retained_insertion`.

## Hollow Structure And Caps

- Outer and middle members are real wall shells, open along the axis.
- A middle member is one wall set: its outside fits its parent while the inside is
  its child cavity. Do not stack a runner geometry and another channel geometry on
  the same middle Part.
- Only the outer member may use `end_cap="base"`.
- Middle members must stay open at both ends.
- Only the innermost member may use `end_cap="tip"`; it may instead be `solid=True`.
- V1 `stop_mode="joint_limit"` is explicit: the motion limit is authoritative. The
  helper does not claim a physical tab/shoulder stop or over-travel proof.

`captured_c_channel` adds inward lips. The child must fit the inner cavity with
clearance but remain wider than the mouth, so it cannot lift out through the opening.

## Round Telescoping Tube

For `round_tube`, pass the same outer diameter as both `outer_width` and
`outer_height`. The equality is validated, and `outer_diameter`/`inner_diameter`
properties expose the circular dimensions explicitly:

```python
round_slide = TelescopingSlideSpec(
    members=(
        TelescopingMemberSpec(
            "round_tube", 0.060, 0.060, 0.004, 0.300, end_cap="base"
        ),
        TelescopingMemberSpec("round_tube", 0.050, 0.050, 0.003, 0.280),
        TelescopingMemberSpec(
            "round_tube", 0.040, 0.040, 0.002, 0.260, solid=True
        ),
    ),
    interfaces=(
        TelescopingInterfaceSpec(0, 1, 5e-4, 0.020, 0.200, 0.080),
        TelescopingInterfaceSpec(1, 2, 5e-4, 0.020, 0.180, 0.080),
    ),
)
```

Round outer and middle members are true annular tubes, not solid cylinders. Their
analytic collision uses segmented annular walls so the inner cavity remains open;
an optional end cap is a separate axial cylinder. The innermost member may be solid.

## Independent Versus Proportional Motion

Default `coupling_mode="independent"` means both joints are independently poseable.
P4 checks `{lower, mid, upper}^2` for each adjacent PRISMATIC pair and forwards those
same poses to staged MJCF validation.

For proportional extension, declare all followers directly from the root driver:

```python
model.couple("outer_to_middle", "middle_to_inner", mapping="full")
```

The follower is then mimic-driven and is never tested at follower-only unreachable
poses. Sequential/dead-zone extension is outside V1; do not imitate it with an equal
ratio coupling.

`TelescopingMemberGeometry` carries analytic wall colliders. Do not mutate it with
low-level vertex/face edits or mesh transforms; place each member with `Visual.origin`.
