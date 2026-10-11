---
example_kind: complete_model
title: 'Scooter Wheel with Road Tire'
description: 'Wheel-and-tire pair whose tire is MOUNTED on the wheel via explicit_anchor + attach + bind_placement, not a matched-constant identity pose.'
tags:
  - sdk
  - base sdk
  - wheel
  - tire
  - scooter wheel
  - road tire
  - wheelgeometry
  - tiregeometry
  - spokes
  - bolt pattern
  - mesh geometry
  - relational placement
  - anchor
  - attach
  - bind_placement
  - explicit_anchor
  - mesh anchor
---
# Scooter Wheel with Road Tire

A paired `WheelGeometry` / `TireGeometry` reference, and the smallest example of
a MOUNTED pair. The tire does not happen to sit where the wheel is — it is
fitted onto it, so its pose is a query, never a coordinate:

    fitted = attach(
        explicit_anchor(tire, AXLE, frame=frame, kind="axis"),
        explicit_anchor(wheel, AXLE, frame=frame, kind="axis"),
    )
    model.bind_placement(PlacementTarget.visual(tire), fitted)

Both Anchors take one `frame=FrameRef.part_local(assembly)` — the owning PART,
not either Visual. The bore is derived the same way, from the rim it seats on:
`TIRE_INNER_RADIUS = WHEEL_OUTER_RADIUS + BEAD_FIT`.

```python
from __future__ import annotations

import math

from sdk import (
    ArticulatedObject,
    BoltPattern,
    Box,
    FrameRef,
    Inertial,
    Origin,
    PlacementTarget,
    TestContext,
    TestReport,
    TireGeometry,
    TireGroove,
    TireSidewall,
    TireTread,
    WheelBore,
    WheelFace,
    WheelGeometry,
    WheelHub,
    WheelRim,
    WheelSpokes,
    attach,
    explicit_anchor,
    mesh_from_geometry,
)

WHEEL_OUTER_RADIUS = 0.090
WHEEL_WIDTH = 0.036
BEAD_FIT = 0.001
# The bore rides on the rim, so it is derived from the rim, never re-typed.
TIRE_INNER_RADIUS = WHEEL_OUTER_RADIUS + BEAD_FIT
TIRE_OUTER_RADIUS = 0.108
TIRE_WIDTH = 0.044

# Both generators lathe about their own local origin with the axle along local
# +X. That is a fabrication fact each mesh can declare about itself; an Anchor's
# +Z is its axis, so rotating +Z onto +X names the axle.
AXLE = Origin(rpy=(0.0, math.pi / 2.0, 0.0))


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="scooter_wheel")
    wheel_finish = model.material("scooter_wheel_silver", rgba=(0.78, 0.80, 0.82, 1.0), material_class="aluminum")
    tire_finish = model.material("scooter_tire_black", rgba=(0.10, 0.10, 0.11, 1.0), material_class="rubber")

    assembly = model.part("wheel_assembly", est_mass_kg=0.34)

    # --- Pass 1: FABRICATE. Each carcass in its own frame, seated against nothing. ---
    wheel = assembly.visual(
        mesh_from_geometry(
            WheelGeometry(
                WHEEL_OUTER_RADIUS,
                WHEEL_WIDTH,
                rim=WheelRim(inner_radius=0.060, flange_height=0.008, flange_thickness=0.003),
                hub=WheelHub(
                    radius=0.022,
                    width=0.026,
                    cap_style="domed",
                    bolt_pattern=BoltPattern(count=5, circle_diameter=0.028, hole_diameter=0.0035),
                ),
                face=WheelFace(dish_depth=0.004, front_inset=0.002),
                spokes=WheelSpokes(style="straight", count=6, thickness=0.003, window_radius=0.008),
                bore=WheelBore(style="round", diameter=0.010),
            ),
            "scooter_wheel",
        ),
        material=wheel_finish,
        name="wheel",
    )
    # The mounted part carries no coordinate at all — the relation supplies it.
    tire = assembly.visual(
        mesh_from_geometry(
            TireGeometry(
                TIRE_OUTER_RADIUS,
                TIRE_WIDTH,
                inner_radius=TIRE_INNER_RADIUS,
                tread=TireTread(style="circumferential", depth=0.0035, count=3),
                grooves=(TireGroove(center_offset=0.0, width=0.004, depth=0.002),),
                sidewall=TireSidewall(style="rounded", bulge=0.05),
            ),
            "scooter_tire",
        ),
        material=tire_finish,
        name="tire",
    )

    # --- Pass 2: ASSEMBLE. The tire is fitted to the wheel, so its pose is a query. ---
    # ONE frame for both Anchors: the PART that owns them, not either Visual.
    frame = FrameRef.part_local(assembly)
    fitted = attach(
        explicit_anchor(tire, AXLE, frame=frame, kind="axis"),
        explicit_anchor(wheel, AXLE, frame=frame, kind="axis"),
    )
    model.bind_placement(PlacementTarget.visual(tire), fitted)

    assembly.inertial = Inertial.from_geometry(
        Box((TIRE_WIDTH, 2.0 * TIRE_OUTER_RADIUS, 2.0 * TIRE_OUTER_RADIUS)),
        mass=0.34,
    )
    return model


def run_tests() -> TestReport:
    return TestContext(object_model).report()


object_model = build_object_model()
```

## Why a declared axis, and not the mesh's bounds

`bounds_anchor(...)` would be the wrong tool for this particular mount. The
wheel's domed hub cap pushes its bounding box 2.3 mm past the rim face on `+X`,
so the mesh's AABB centre sits 1.1 mm off the wheel's mid-plane and is **not**
the axle; centring the tire on it would seat the tire crooked on the rim. The
axle is a fact each generator owns — both lathe about their local origin along
local `+X` — so it is declared once with `explicit_anchor(..., kind="axis")` and
consumed by both sides. (`anchor(...)` itself is unavailable here: an
unannotated Mesh has no analytic face/edge/corner features.)

Two visuals living on one part changes nothing about any of this. A second
visual depends on its host exactly as a door depends on a cabinet; leaving both
at an implicit identity pose looks the same in the render and encodes no
dependency at all.

The `attach(...)` above resolves to the identity pose, which is the point: the
pose the old version left implicit is now the compiler's output. Move the wheel
anywhere within the part and the tire rides with it; change
`WHEEL_OUTER_RADIUS` and the bore resizes to keep the same bead fit. Neither
edit touches a coordinate.
