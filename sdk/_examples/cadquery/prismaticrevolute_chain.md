---
example_kind: complete_model
title: 'Prismatic-revolute Chain'
description: "A linear carriage carrying a hinged end effector."
tags:
  - cadquery
  - examples
  - articulation
  - prismatic
  - revolute
  - fixture
---
# Prismatic-revolute Chain


Neither joint frame is typed. A prismatic joint frame belongs to the RAIL its
carriage rides on, and a revolute joint frame belongs to the HINGE LINE it turns
about — so each is resolved from the element that owns it: the rail's top face, and
the hinge ear's outer top edge. The rail and the ear are themselves seated against
their hosts, so the whole fixture rescales from the plate and carriage dimensions
alone.

Watch the frame argument on the two same-part seats: `rail` on `plate` and
`hinge_ear` on `carriage_body` are two visuals inside ONE part, so both anchors take
`FrameRef.part_local(<the owning part>)`. Passing each Visual's own default frame is
the first thing to try and it is rejected — `attach()` compares owner identity.

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

BASE_L = 0.52
BASE_W = 0.30
BASE_T = 0.024
RAIL_Y = 0.060
RAIL_W = 0.030
RAIL_H = 0.018
CARRIAGE_L = 0.14
CARRIAGE_W = 0.12
CARRIAGE_H = 0.040
EAR_SIZE = (0.024, 0.014, 0.030)
FLAP_L = 0.13
FLAP_W = 0.10
FLAP_T = 0.006
PRISMATIC_LOWER = -0.16
PRISMATIC_UPPER = 0.16
FLAP_UPPER = 1.4


def _plate_solid() -> cq.Workplane:
    return cq.Workplane("XY").box(BASE_L, BASE_W, BASE_T).edges("|Z").fillet(0.012)


def _carriage_solid() -> cq.Workplane:
    return cq.Workplane("XY").box(CARRIAGE_L, CARRIAGE_W, CARRIAGE_H).edges("|Z").fillet(0.006)


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="inspection_fixture")

    base_finish = model.material("fixture_base", rgba=(0.42, 0.44, 0.48, 1.0), material_class="anodized_metal")
    carriage_finish = model.material("fixture_carriage", rgba=(0.78, 0.46, 0.14, 1.0), material_class="painted_metal")
    flap_finish = model.material("fixture_flap", rgba=(0.92, 0.77, 0.16, 1.0), material_class="painted_metal")

    plate_solid = _plate_solid()
    base = model.part("base", est_mass_kg=11.0)
    plate = base.visual(mesh_from_cadquery(plate_solid, "fixture_base"), material=base_finish, name="plate")
    # The rail carries no coordinate: it follows the plate top (below).
    rail = base.visual(Box((BASE_L, RAIL_W, RAIL_H)), material=base_finish, name="guide_rail")
    base.inertial = Inertial.from_geometry(Box((BASE_L, BASE_W, BASE_T)), mass=11.0)

    carriage_solid = _carriage_solid()
    carriage = model.part("carriage", est_mass_kg=5.310582)
    # The body's own half-height offset puts its underside on the part frame, which is
    # the rail line. Internal geometry: a plain Origin.
    carriage_body = carriage.visual(
        mesh_from_cadquery(carriage_solid, "carriage"),
        origin=Origin(xyz=(0.0, 0.0, CARRIAGE_H / 2.0)),
        material=carriage_finish,
        name="carriage_body",
    )
    hinge_ear = carriage.visual(Box(EAR_SIZE), material=carriage_finish, name="hinge_ear")

    flap = model.part("flap", est_mass_kg=0.6084)
    # The panel hangs off the hinge line along -Y; that offset is the flap's own
    # internal geometry, expressed in its own frame.
    flap.visual(
        mesh_from_cadquery(cq.Workplane("XY").box(FLAP_L, FLAP_W, FLAP_T), "inspection_flap"),
        origin=Origin(xyz=(0.0, -FLAP_W / 2.0, 0.0)),
        material=flap_finish,
        name="flap_panel",
    )

    model.articulation(
        "base_to_carriage",
        ArticulationType.PRISMATIC,
        parent="base",
        child="carriage",
        axis=(1.0, 0.0, 0.0),
        motion_limits=MotionLimits(
            lower=PRISMATIC_LOWER,
            upper=PRISMATIC_UPPER,
            effort=120.0,
            velocity=0.25,
        ),
    )
    model.articulation(
        "carriage_to_flap",
        ArticulationType.REVOLUTE,
        parent="carriage",
        child="flap",
        axis=(1.0, 0.0, 0.0),
        motion_limits=MotionLimits(lower=0.0, upper=FLAP_UPPER, effort=10.0, velocity=1.5),
    )

    # 1. The rail runs along the plate top, offset across it by RAIL_Y. `uv_offset` is
    #    an in-plane move in the TARGET Anchor's (+x, +y) — the plate face's own axes.
    base_frame = FrameRef.part_local(base)
    model.bind_placement(
        PlacementTarget.visual(rail),
        attach(
            anchor(rail, AnchorSelector.face("-z"), frame=base_frame),
            cadquery_anchor(plate_solid, AnchorSelector.face("+z"), frame=base_frame, subject=plate),
            axis_map={"+z": "-z", "+x": "+x"},
            uv_offset=(0.0, RAIL_Y),
        ),
    )

    # 2. The hinge ear stands on the carriage's far face, opposite the rail.
    carriage_frame = FrameRef.part_local(carriage)
    model.bind_placement(
        PlacementTarget.visual(hinge_ear),
        attach(
            anchor(hinge_ear, AnchorSelector.face("+y"), frame=carriage_frame),
            cadquery_anchor(carriage_solid, AnchorSelector.face("-y"), frame=carriage_frame, subject=carriage_body),
            axis_map={"+z": "-z", "+x": "+x"},
        ),
    )

    # 3. The carriage rides ON the rail, so the prismatic frame is the rail's top face.
    #    A Part's own frame origin is the subject of a joint placement.
    ride_frame = FrameRef.parent_link_local(base)
    model.bind_placement(
        PlacementTarget.joint(model.get_joint("base_to_carriage")),
        attach(
            explicit_anchor(carriage, RigidFrame(), frame=ride_frame, kind="point"),
            anchor(rail, AnchorSelector.face("+z"), frame=ride_frame),
        ),
    )

    # 4. THE HINGE LINE — the ear's outer top edge. An edge Anchor's +z runs along the
    #    line, which is exactly what a revolute frame needs.
    hinge_frame = FrameRef.parent_link_local(carriage)
    model.bind_placement(
        PlacementTarget.joint(model.get_joint("carriage_to_flap")),
        attach(
            explicit_anchor(flap, RigidFrame(), frame=hinge_frame, kind="point"),
            anchor(hinge_ear, AnchorSelector.edge("-y", "+z"), frame=hinge_frame),
        ),
    )

    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    return TestContext(object_model).report()
```

The bound frames resolve to `base_to_carriage` at
`(0, RAIL_Y, BASE_T / 2 + RAIL_H)` = `(0, 0.060, 0.030)` and `carriage_to_flap` at
the ear's outer top edge `(0, -0.074, 0.035)` — the same poses the hand-written
coordinates produced, now expressed as the relationships that produced them.
