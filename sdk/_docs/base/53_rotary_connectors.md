# Functional Rotary Connectors

## Choose The Motion Topology First

`RotaryConnectorSpec.mount_mode` separates two mechanisms that can look similar:

| Mode | Moving Part | Stationary Part | Consumer bore |
| --- | --- | --- | --- |
| `free_on_axle` | wheel/knob/fan rotates around the axle | axle and its mount | round, with radial clearance |
| `rigid_to_shaft` | consumer and shaft are one moving Part | bearing housing/frame | round, keyed, D, double-D, spline, or hex as supported by the consumer |

Both modes use one main `REVOLUTE` or `CONTINUOUS` joint. In rigid mode do not add
a second joint between the consumer and shaft; they belong to the same Part.
The rigid rotor has a round bearing journal inside the stationary housing and a
profile-matched mount segment outside it. `axial_clearance` separates the consumer
face from the housing; it is not merely extra shaft length.

## Free Wheel On A Fixed Axle

```python
from sdk import (
    ArticulationType,
    MotionLimits,
    RotaryConnectorGeometry,
    RotaryConnectorSpec,
    WheelGeometry,
    mesh_from_geometry,
)

fit = RotaryConnectorSpec(
    mount_mode="free_on_axle",
    shaft_diameter=0.010,
    bearing_length=0.020,
    radial_clearance=5e-4,
)

axle.visual(
    mesh_from_geometry(
        RotaryConnectorGeometry(fit, role="stationary", axis="x"), "wheel_axle"
    ),
    material=steel,
    name="wheel_axle",
)
wheel.visual(
    mesh_from_geometry(
        WheelGeometry(0.06, 0.020, rotary=fit), "wheel"
    ),
    material=polymer,
    name="wheel",
)
model.articulation(
    "wheel_spin",
    ArticulationType.CONTINUOUS,
    parent=axle,
    child=wheel,
    axis=(1.0, 0.0, 0.0),
    motion_limits=MotionLimits(effort=3.0, velocity=12.0),
)
```

Passing `rotary=fit` is opt-in. It owns the consumer bore, so do not also pass
`WheelBore`, `KnobBore`, or `FanRotorHub.bore_diameter`.

Functional rotary consumers require `center=True`, so the consumer arrives centred
on its own frame and its pose is written to `Visual.origin`. Two different poses end
up in that one field, and they are authored differently:

- **Along the connector's own axis** — the consumer's seat on the shaft — comes from
  the spec: `origin=fit.consumer_origin(axis=..., axial_length=...)`. Do not
  hand-compute it.
- **Where the connector sits on the host** — the axle mount on a chassis rail, the
  bearing housing on a fascia — follows the host geometry. Author it as a relation
  and bind the result:

```python
from sdk import AnchorSelector, FrameRef, PlacementTarget, anchor, attach, surface_anchor

frame = FrameRef.part_local(chassis)
model.bind_placement(
    PlacementTarget.visual(axle_visual),
    attach(
        # Connector geometry is an unannotated Mesh, so analytic face/edge selectors
        # are rejected — query a verified surface point instead. Map ONLY the normal:
        # a surface Anchor's in-plane axes are arbitrary tangents, and pinning to them
        # would twist the connector.
        surface_anchor(axle_visual, direction=(0.0, 0.0, -1.0), frame=frame),
        anchor(chassis_rail, AnchorSelector.face("+z", uv=(0.5, 0.2)), frame=frame),
        axis_map={"+z": "-z"},
    ),
)
```

A hand-typed assembly `Origin(...)` on a connector visual goes stale the moment the
host moves; the spec-derived `consumer_origin(...)` does not, and neither does a
bound relation. See `50_placement.md`.

## Knob Rigidly Mounted To A Shaft

```python
fit = RotaryConnectorSpec(
    mount_mode="rigid_to_shaft",
    shaft_profile="d_shaft",
    shaft_diameter=0.008,
    bearing_length=0.018,
)

frame.visual(
    mesh_from_geometry(
        RotaryConnectorGeometry(fit, role="stationary", axis="z"), "bearing_housing"
    ),
    material=steel,
)
shaft_and_knob.visual(
    mesh_from_geometry(
        RotaryConnectorGeometry(fit, role="rotor", axis="z"), "shaft"
    ),
    material=steel,
)
shaft_and_knob.visual(
    mesh_from_geometry(
        KnobGeometry(0.040, 0.020, rotary=fit), "selector_knob"
    ),
    origin=fit.consumer_origin(axis="z", axial_length=0.020),
    material=polymer,
)
model.articulation(
    "selector",
    ArticulationType.REVOLUTE,
    parent=frame,
    child=shaft_and_knob,
    axis=(0.0, 0.0, 1.0),
    motion_limits=MotionLimits(lower=-1.2, upper=1.2, effort=2.0, velocity=2.0),
)
```

## Consumer Support

- `WheelGeometry`: `round`, `keyed`, `hex`.
- `KnobGeometry`: `round`, `d_shaft`, `double_d`, `splined`, `hex`.
- `FanRotorGeometry`: `round` only.
- Free-on-axle always requires `round`.

The fit does not choose `REVOLUTE` versus `CONTINUOUS`. A selector knob is usually
bounded; an encoder, wheel, or fan may be continuous. The blueprint owns that choice.

## Cam lock recipe

Use a bounded rotary driver and a short prismatic latch follower. Build both with
normal clearance-fit connector geometry, then let the existing nonlinear cam
template derive the follower mapping:

```python
knob_joint = model.articulation(
    "lock_turn",
    ArticulationType.REVOLUTE,
    parent=frame,
    child=shaft_and_knob,
    axis=(0, 0, 1),
    motion_limits=MotionLimits(lower=0.0, upper=1.2, effort=2.0, velocity=2.0),
)
latch_joint = model.articulation(
    "latch_slide",
    ArticulationType.PRISMATIC,
    parent=frame,
    child=latch,
    axis=(1, 0, 0),
    motion_limits=MotionLimits(lower=0.0, upper=0.012, effort=8.0, velocity=0.1),
)
model.couple(
    knob_joint,
    latch_joint,
    kind="cam",
    lift=0.012,
    rise_angle=1.0,
    dwell_angle=0.2,
)
```

The latch must still ride in a real guide with positive clearance. The coupling
maps motion; it does not authorize a solid latch to pass through the keeper.

## Collision Contract

The consumer visual contains a real boolean bore. In free mode the mesh also carries
a required collision-decomposition contract and a full-clearance-bore cavity probe. Physics
enrichment first runs CoACD. It blocks if CoACD is disabled or unavailable, and checks
the resulting convex pieces against the probe. If CoACD fails or fills the bore, the
declared annular primitive fallback is used. This is an explicit measured fallback,
not silent convex-hull behavior.

The fallback is a 32-segment conservative annulus: it preserves the bore and reaches
the measured outer radius, but may fill openings such as spoke gaps. It is used only
after required CoACD fails or consumes the cavity, never as the normal inertia or
collision path. Default consumer behavior is unchanged when `rotary` is omitted.

Do not call `.rotate()`, `.translate()`, `.scale()`, or `.merge()` on functional
geometry. Use `Visual.origin`; mesh-level edits are rejected because they would make
the cavity probe or primitive transforms stale.
