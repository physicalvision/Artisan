---
example_kind: complete_model
title: 'Asymmetrical Panel Bezel with Mount Bosses'
description: 'Base SDK example showing an offset panel bezel with asymmetric wall thickness, rear bosses, and edge cutout details using BezelGeometry.'
tags:
  - sdk
  - base sdk
  - bezel
  - panel bezel
  - control panel
  - instrument bezel
  - mount bosses
  - recessed frame
  - trim frame
  - mesh geometry
---
# Asymmetrical Panel Bezel with Mount Bosses

This base-SDK example is a strong reference for `BezelGeometry` when the trim
is not perfectly symmetric. It shows asymmetric wall thickness, rear mounting
bosses, a side cutout, and a bottom notch feature in one compact example.

```python
from __future__ import annotations

from sdk import (
    ArticulatedObject,
    BezelCutout,
    BezelEdgeFeature,
    BezelGeometry,
    BezelMounts,
    Box,
    Inertial,
    TestContext,
    TestReport,
    mesh_from_geometry,
)

OUTER_SIZE = (0.112, 0.082)
DEPTH = 0.012


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="asym_panel_bezel")
    finish = model.material("panel_bezel_blue", rgba=(0.20, 0.28, 0.36, 1.0), material_class="abs_plastic")

    bezel = model.part("bezel", est_mass_kg=0.12)
    bezel.visual(
        mesh_from_geometry(
            BezelGeometry(
                (0.070, 0.044),
                OUTER_SIZE,
                DEPTH,
                outer_corner_radius=0.008,
                wall=(0.012, 0.020, 0.010, 0.014),
                mounts=BezelMounts(
                    style="bosses",
                    hole_count=4,
                    hole_diameter=0.003,
                    boss_diameter=0.007,
                    setback=0.004,
                ),
                cutouts=(BezelCutout(edge="right", width=0.016, depth=0.004),),
                edge_features=(
                    BezelEdgeFeature(
                        style="notch",
                        edge="bottom",
                        size=0.004,
                        extent=0.016,
                    ),
                ),
            ),
            "asym_panel_bezel",
        ),
        material=finish,
        name="bezel_shell",
    )
    bezel.inertial = Inertial.from_geometry(Box((OUTER_SIZE[0], OUTER_SIZE[1], 0.016)), mass=0.12)
    return model


def run_tests() -> TestReport:
    return TestContext(object_model).report()


object_model = build_object_model()
```
