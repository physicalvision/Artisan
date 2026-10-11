---
example_kind: complete_model
title: 'Single Continuous Rotary Shaft'
description: "Continuous spindle with a fixed wheelhead and geometry-bound placements."
tags:
  - cadquery
  - examples
  - articulation
  - continuous
  - rotary
  - spindle
---
# Single Continuous Rotary Shaft


A vertical stack is where hand-typed heights accumulate: the spindle sits on the
base collar, the coupling lug sits on the spindle, and the wheelhead sits on the
lug. Each of those three is a dependency, so none of them is typed. The collar top
comes from the base's own BRep face, and the other two from the analytic faces of
the visuals below them. Change the cabinet height, the collar height, the spindle
length, or the lug thickness and the whole stack follows.

What stays a plain `Origin(...)`: each visual's own half-height offset, which
builds that one visual's geometry in its own frame and depends on nothing else.

```python
import cadquery as cq

from sdk import (
    AnchorSelector,
    ArticulatedObject,
    ArticulationType,
    Box,
    Cylinder,
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

BASE_L = 0.44
BASE_W = 0.32
BASE_H = 0.28
COLLAR_RADIUS = 0.038
COLLAR_HEIGHT = 0.030
SHAFT_RADIUS = 0.016
SHAFT_HEIGHT = 0.100
COUPLING_LUG_SIZE = (0.030, 0.030, 0.012)
WHEELHEAD_RADIUS = 0.140
WHEELHEAD_THICKNESS = 0.018


def _base_solid() -> cq.Workplane:
    # The cabinet plus the raised bearing collar the spindle turns in. The collar is
    # part of the base solid, so its top face is a real queryable feature.
    cabinet = cq.Workplane("XY").box(BASE_L, BASE_W, BASE_H, centered=(True, True, False))
    return cabinet.faces(">Z").workplane().circle(COLLAR_RADIUS).extrude(COLLAR_HEIGHT)


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="pottery_wheel")

    body_gray = model.material("body_gray", rgba=(0.26, 0.28, 0.31, 1.0), material_class="painted_metal")
    steel = model.material("steel", rgba=(0.72, 0.74, 0.77, 1.0), material_class="steel")
    head_finish = model.material("wheelhead", rgba=(0.80, 0.82, 0.84, 1.0), material_class="aluminum")

    base_solid = _base_solid()
    base = model.part("base", est_mass_kg=18.0)
    cabinet = base.visual(mesh_from_cadquery(base_solid, "pottery_base"), material=body_gray, name="cabinet")
    base.inertial = Inertial.from_geometry(
        Box((BASE_L, BASE_W, BASE_H)),
        mass=18.0,
        origin=Origin(xyz=(0.0, 0.0, BASE_H / 2.0)),
    )

    shaft = model.part("shaft", est_mass_kg=0.6)
    # The spindle's own half-length offset puts its BOTTOM on the part frame, which is
    # the collar line. Internal geometry: a plain Origin.
    spindle = shaft.visual(
        Cylinder(radius=SHAFT_RADIUS, length=SHAFT_HEIGHT),
        origin=Origin(xyz=(0.0, 0.0, SHAFT_HEIGHT / 2.0)),
        material=steel,
        name="spindle",
    )
    # The drive lug carries NO coordinate: it follows the spindle top (below).
    coupling_lug = shaft.visual(Box(COUPLING_LUG_SIZE), material=steel, name="coupling_lug")
    shaft.inertial = Inertial.from_geometry(
        Cylinder(radius=SHAFT_RADIUS, length=SHAFT_HEIGHT),
        mass=0.6,
        origin=Origin(xyz=(0.0, 0.0, SHAFT_HEIGHT / 2.0)),
    )

    wheelhead = model.part("wheelhead", est_mass_kg=4.8)
    wheelhead.visual(
        Cylinder(radius=WHEELHEAD_RADIUS, length=WHEELHEAD_THICKNESS),
        origin=Origin(xyz=(0.0, 0.0, WHEELHEAD_THICKNESS / 2.0)),
        material=head_finish,
        name="platter",
    )
    wheelhead.inertial = Inertial.from_geometry(
        Cylinder(radius=WHEELHEAD_RADIUS, length=WHEELHEAD_THICKNESS),
        mass=4.8,
        origin=Origin(xyz=(0.0, 0.0, WHEELHEAD_THICKNESS / 2.0)),
    )

    model.articulation(
        "base_to_shaft",
        ArticulationType.CONTINUOUS,
        parent="base",
        child="shaft",
        axis=(0.0, 0.0, 1.0),
        motion_limits=MotionLimits(effort=25.0, velocity=12.0),
    )
    model.articulation(
        "shaft_to_wheelhead",
        ArticulationType.FIXED,
        parent="shaft",
        child="wheelhead",
    )

    # 1. The lug seats on the spindle top — two visuals in ONE part, so BOTH anchors
    #    take that part's frame. Mating faces are back-to-back, hence the axis_map.
    shaft_frame = FrameRef.part_local(shaft)
    model.bind_placement(
        PlacementTarget.visual(coupling_lug),
        attach(
            anchor(coupling_lug, AnchorSelector.face("-z"), frame=shaft_frame),
            anchor(spindle, AnchorSelector.face("+z"), frame=shaft_frame),
            axis_map={"+z": "-z", "+x": "+x"},
        ),
    )

    # 2. The spindle turns in the collar, so its joint frame IS the collar's top face.
    collar_frame = FrameRef.parent_link_local(base)
    model.bind_placement(
        PlacementTarget.joint(model.get_joint("base_to_shaft")),
        attach(
            # A Part's own frame origin is the subject of a joint placement.
            explicit_anchor(shaft, RigidFrame(), frame=collar_frame, kind="point"),
            cadquery_anchor(base_solid, AnchorSelector.face("+z"), frame=collar_frame, subject=cabinet),
        ),
    )

    # 3. The wheelhead bolts onto the lug, so it follows the lug that follows the
    #    spindle. Three links of one chain, no height typed anywhere in it.
    head_frame = FrameRef.parent_link_local(shaft)
    model.bind_placement(
        PlacementTarget.joint(model.get_joint("shaft_to_wheelhead")),
        attach(
            explicit_anchor(wheelhead, RigidFrame(), frame=head_frame, kind="point"),
            anchor(coupling_lug, AnchorSelector.face("+z"), frame=head_frame),
        ),
    )

    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    return TestContext(object_model).report()
```

The stack resolves to `base_to_shaft` at `BASE_H + COLLAR_HEIGHT` = 0.310, the lug
at `SHAFT_HEIGHT + COUPLING_LUG_SIZE[2] / 2` = 0.106, and `shaft_to_wheelhead` at
the lug top = 0.112 — the same poses the hand-written heights produced, now
expressed as the relationships that produced them.
