---
example_kind: complete_model
title: 'Axial Fan Rotor'
description: 'Base SDK example showing a compact axial rotor with pitched blades and a central hub using FanRotorGeometry.'
tags:
  - sdk
  - base sdk
  - fan
  - axial fan
  - fan rotor
  - impeller
  - cooling fan
  - fanrotorgeometry
  - mesh geometry
---
# Axial Fan Rotor

This base-SDK example is a minimal reference for `FanRotorGeometry`. It is
useful for compact cooling fans, appliance rotors, and any exposed axial
impeller where the blades should read as real geometry rather than embossed
texture.

```python
from __future__ import annotations

from sdk import (
    ArticulatedObject,
    Box,
    FanRotorGeometry,
    Inertial,
    TestContext,
    TestReport,
    mesh_from_geometry,
)

OUTER_RADIUS = 0.070
THICKNESS = 0.010


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="axial_fan_rotor")
    finish = model.material("fan_black", rgba=(0.08, 0.08, 0.09, 1.0), material_class="abs_plastic")

    fan_rotor = model.part("fan_rotor", est_mass_kg=0.12)
    fan_rotor.visual(
        mesh_from_geometry(
            FanRotorGeometry(
                OUTER_RADIUS,
                0.020,
                5,
                thickness=THICKNESS,
                blade_pitch_deg=24.0,
                blade_sweep_deg=14.0,
            ),
            "fan_rotor",
        ),
        material=finish,
        name="fan_rotor",
    )
    fan_rotor.inertial = Inertial.from_geometry(Box((0.14, 0.14, 0.012)), mass=0.12)
    return model


def run_tests() -> TestReport:
    return TestContext(object_model).report()


object_model = build_object_model()
```
