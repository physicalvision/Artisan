---
example_kind: geometry_pattern
title: Relational placement with analytic Anchors
description: Derive an edge-mounted support, a span connector, and one shared hinge pin without retuning coordinates when hosts change.
tags:
  - placement
  - anchors
  - hinge
  - connectivity
---

# Relational placement with analytic Anchors

Placement pattern; `model`, `left_riser`, `right_bridge`, and `cleat` are existing host objects.

Use this pattern when a placement is defined by neighbouring geometry rather
than a fixed coordinate. The calls run while `build_object_model()` constructs
the ordinary SDK object; `compile_model` remains the final QC authority.

```python
from sdk import (
    AnchorSelector,
    Box,
    FrameRef,
    HingePin,
    Origin,
    PlacementTarget,
    anchor,
    attach,
    center_between,
    gap,
)


body = model.part("body", est_mass_kg=12.0)
deck = body.visual(
    Box((1.04, 0.66, 0.055)),
    origin=Origin(xyz=(0.0, 0.0, 0.930)),
    name="patient_deck_base",
)
support = body.visual(
    Box((1.04, 0.035, 0.085)),
    origin=Origin(xyz=(0.0, 0.0, 1.0)),
    name="front_hinge_support",
)
frame = FrameRef.part_local(body)

# The support follows the deck's front and top when deck depth/height changes.
support_result = attach(
    anchor(support, AnchorSelector.edge("+y", "-z"), frame=frame),
    anchor(deck, AnchorSelector.edge("+y", "+z"), frame=frame),
    axis_map={"+x": "+x", "+y": "-y"},
)
model.bind_placement(PlacementTarget.visual(support), support_result)

# A connector derives both its length and center from two host faces.
start = anchor(left_riser, AnchorSelector.face("+y"), frame=frame)
end = anchor(right_bridge, AnchorSelector.face("-y"), frame=frame)
neck = body.visual(
    Box((0.080, gap(start, end), 0.025)),
    origin=Origin(xyz=(0.0, 0.0, 0.245)),
    name="derived_neck",
)
neck_result = center_between(neck, start, end, axis=(0.0, 1.0, 0.0), frame=frame)
model.bind_placement(PlacementTarget.visual(neck), neck_result)

# A frozen shared HingePin is constructed once, then both halves and the joint
# consume this same instance.
mount = anchor(cleat, AnchorSelector.face("+y"), frame=frame).offset(
    translate=(0.003, 0.007, 0.0)
)
pin = HingePin.from_anchor(
    mount,
    axis=mount.pose.axis("-x"),
    length=0.05,
    knuckle_count=5,
    pin_diameter=0.006,
    knuckle_diameter=0.010,
    clearance=3e-4,
)
```

For a controlled Blender Mesh, declare placement-relevant semantic Anchors in
the same `run_blender` script and resolve them with `semantic_anchor(...)`. For an
external OBJ, use `surface_anchor(...)`, strict `mesh_feature_anchor(...)`, or an
explicit `RigidFrame` fallback; never rename triangle indices as semantic faces.
