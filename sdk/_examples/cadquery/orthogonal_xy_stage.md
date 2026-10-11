---
example_kind: complete_model
title: 'Orthogonal XY Stage'
description: "Stacked X and Y carriages with rail-derived joint frames."
tags:
  - cadquery
  - examples
  - articulation
  - prismatic
  - xy
  - cartesian
---
# Orthogonal XY Stage


A stacked stage is the classic place where joint heights get typed in by hand, and
the classic place where one thickness change silently strands a carriage in mid-air.
Each rail set seats on the slab below it, and each prismatic joint frame is the top
face of the rail set its carriage rides on. Nothing in the Z stack is a literal, so
changing the base plate or the X carriage thickness moves everything above it.

The two rail seats are visual-to-visual inside one part, so both anchors of each
`attach(...)` take `FrameRef.part_local(<the owning part>)`. Leave that off and
`attach()` rejects the pair — it compares owner identity, and each Visual defaults to
its own frame.

```python
import cadquery as cq

from sdk import (
    AnchorSelector,
    ArticulatedObject,
    ArticulationType,
    Box,
    FrameRef,
    Inertial,
    MotionLimits,
    Origin,
    PlacementTarget,
    RigidFrame,
    TestContext,
    TestReport,
    anchor,
    attach,
    cadquery_anchor,
    explicit_anchor,
    mesh_from_cadquery,
)

BASE_L = 0.30
BASE_W = 0.30
BASE_T = 0.030
X_RAIL_L = 0.28
X_RAIL_W = 0.020
X_RAIL_H = 0.014
X_CARRIAGE_L = 0.20
X_CARRIAGE_W = 0.22
X_CARRIAGE_T = 0.018
Y_RAIL_L = 0.020
Y_RAIL_W = 0.20
Y_RAIL_H = 0.014
Y_PLATE_L = 0.16
Y_PLATE_W = 0.16
Y_PLATE_T = 0.014


def _slab(length: float, width: float, thickness: float, fillet: float) -> cq.Workplane:
    return cq.Workplane("XY").box(length, width, thickness).edges("|Z").fillet(fillet)


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="precision_xy_stage")

    anodized = model.material("anodized_black", rgba=(0.16, 0.17, 0.19, 1.0), material_class="anodized_metal")
    ground = model.material("ground_steel", rgba=(0.72, 0.74, 0.78, 1.0), material_class="steel")
    machined = model.material("machined_gray", rgba=(0.60, 0.62, 0.66, 1.0), material_class="aluminum")
    plate_blue = model.material("plate_blue", rgba=(0.41, 0.47, 0.60, 1.0), material_class="painted_metal")

    base_solid = _slab(BASE_L, BASE_W, BASE_T, 0.010)
    base = model.part("base", est_mass_kg=4.2)
    base_body = base.visual(mesh_from_cadquery(base_solid, "stage_base"), material=anodized, name="base_body")
    x_rails = base.visual(Box((X_RAIL_L, X_RAIL_W, X_RAIL_H)), material=ground, name="x_rails")
    base.inertial = Inertial.from_geometry(Box((BASE_L, BASE_W, BASE_T)), mass=4.2)

    x_solid = _slab(X_CARRIAGE_L, X_CARRIAGE_W, X_CARRIAGE_T, 0.006)
    x_stage = model.part("x_stage", est_mass_kg=2.576498)
    # Own half-thickness offset: the carriage underside sits on its part frame, which
    # is the X rail line. Internal geometry, so a plain Origin.
    x_carriage = x_stage.visual(
        mesh_from_cadquery(x_solid, "x_carriage"),
        origin=Origin(xyz=(0.0, 0.0, X_CARRIAGE_T / 2.0)),
        material=machined,
        name="x_carriage",
    )
    y_rails = x_stage.visual(Box((Y_RAIL_L, Y_RAIL_W, Y_RAIL_H)), material=ground, name="y_rails")

    y_stage = model.part("y_stage", est_mass_kg=2.792145)
    y_stage.visual(
        mesh_from_cadquery(_slab(Y_PLATE_L, Y_PLATE_W, Y_PLATE_T, 0.006), "y_plate"),
        origin=Origin(xyz=(0.0, 0.0, Y_PLATE_T / 2.0)),
        material=plate_blue,
        name="y_plate",
    )

    model.articulation(
        "base_to_x",
        ArticulationType.PRISMATIC,
        parent=base,
        child=x_stage,
        axis=(1.0, 0.0, 0.0),
        motion_limits=MotionLimits(lower=-0.060, upper=0.060, effort=120.0, velocity=0.25),
    )
    model.articulation(
        "x_to_y",
        ArticulationType.PRISMATIC,
        parent=x_stage,
        child=y_stage,
        axis=(0.0, 1.0, 0.0),
        motion_limits=MotionLimits(lower=-0.035, upper=0.035, effort=90.0, velocity=0.25),
    )

    # Each rail set seats on the slab below it. Two visuals in ONE part -> ONE frame,
    # the owning Part's, on BOTH anchors. Mating faces are back-to-back: axis_map.
    base_frame = FrameRef.part_local(base)
    model.bind_placement(
        PlacementTarget.visual(x_rails),
        attach(
            anchor(x_rails, AnchorSelector.face("-z"), frame=base_frame),
            cadquery_anchor(base_solid, AnchorSelector.face("+z"), frame=base_frame, subject=base_body),
            axis_map={"+z": "-z", "+x": "+x"},
        ),
    )
    x_frame = FrameRef.part_local(x_stage)
    model.bind_placement(
        PlacementTarget.visual(y_rails),
        attach(
            anchor(y_rails, AnchorSelector.face("-z"), frame=x_frame),
            cadquery_anchor(x_solid, AnchorSelector.face("+z"), frame=x_frame, subject=x_carriage),
            axis_map={"+z": "-z", "+x": "+x"},
        ),
    )

    # Each carriage rides ON its rails, so the prismatic frame is that rail top face.
    # A Part's own frame origin is the subject of a joint placement.
    ride_x = FrameRef.parent_link_local(base)
    model.bind_placement(
        PlacementTarget.joint(model.get_joint("base_to_x")),
        attach(
            explicit_anchor(x_stage, RigidFrame(), frame=ride_x, kind="point"),
            anchor(x_rails, AnchorSelector.face("+z"), frame=ride_x),
        ),
    )
    ride_y = FrameRef.parent_link_local(x_stage)
    model.bind_placement(
        PlacementTarget.joint(model.get_joint("x_to_y")),
        attach(
            explicit_anchor(y_stage, RigidFrame(), frame=ride_y, kind="point"),
            anchor(y_rails, AnchorSelector.face("+z"), frame=ride_y),
        ),
    )

    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    return TestContext(object_model).report()
```

The Z stack resolves to `base_to_x` at `BASE_T / 2 + X_RAIL_H` = 0.029 and `x_to_y`
at `X_CARRIAGE_T + Y_RAIL_H` = 0.032 — the same poses the hand-written `X_JOINT_Z`
and `Y_JOINT_Z` carried, now expressed as the relationships that produced them.
