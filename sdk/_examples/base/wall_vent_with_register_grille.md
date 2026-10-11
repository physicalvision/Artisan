---
example_kind: complete_model
title: 'Wall Vent with Register Grille'
description: 'Base SDK example showing a wall vent with a true open grille shell, angled slats, shallow rear sleeve, and visible mounting screws using VentGrilleGeometry.'
tags:
  - sdk
  - base sdk
  - wall vent
  - vent grille
  - register
  - hvac vent
  - grille
  - slotted face
  - ventgrillegeometry
  - mesh geometry
---
# Wall Vent with Register Grille

This base-SDK example is a clean reference for `VentGrilleGeometry` used as a
full vent shell rather than a decorative face plate. It keeps the grille open,
adds a shallow rear sleeve, and layers in simple mounting screws.

```python
from __future__ import annotations

from sdk import (
    ArticulatedObject,
    Box,
    Cylinder,
    Inertial,
    Origin,
    TestContext,
    TestReport,
    VentGrilleGeometry,
    mesh_from_geometry,
)

WIDTH = 0.18
HEIGHT = 0.10
FACE_THICKNESS = 0.004
DUCT_DEPTH = 0.026
SCREW_RADIUS = 0.002
SCREW_LENGTH = 0.002
SCREW_OFFSET_X = WIDTH / 2.0 - 0.006
SCREW_OFFSET_Y = HEIGHT / 2.0 - 0.006
TOTAL_DEPTH = 0.032


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="wall_vent")

    plastic_white = model.material("plastic_white", color=(0.9, 0.9, 0.9), material_class="matte_plastic")
    metal_silver = model.material("metal_silver", color=(0.7, 0.7, 0.7), material_class="aluminum")

    vent_body = model.part("vent_body", est_mass_kg=0.20)
    vent_body.visual(
        mesh_from_geometry(
            VentGrilleGeometry(
                (WIDTH, HEIGHT),
                frame=0.012,
                face_thickness=FACE_THICKNESS,
                duct_depth=DUCT_DEPTH,
                duct_wall=0.003,
                slat_pitch=0.018,
                slat_width=0.009,
                slat_angle_deg=35.0,
                corner_radius=0.006,
            ),
            "vent_shell",
        ),
        origin=Origin(),
        material=plastic_white,
        name="vent_shell",
    )

    for i, (sx, sy) in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
        vent_body.visual(
            Cylinder(radius=SCREW_RADIUS, length=SCREW_LENGTH),
            origin=Origin(xyz=(sx * SCREW_OFFSET_X, sy * SCREW_OFFSET_Y, 0.004)),
            material=metal_silver,
            name=f"screw_{i}",
        )

    vent_body.inertial = Inertial.from_geometry(
        Box((WIDTH, HEIGHT, TOTAL_DEPTH)),
        mass=0.20,
        origin=Origin(xyz=(0.0, 0.0, -0.011)),
    )
    return model


def run_tests() -> TestReport:
    return TestContext(object_model).report()


object_model = build_object_model()
```
