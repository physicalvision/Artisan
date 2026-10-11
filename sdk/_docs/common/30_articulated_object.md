# ArticulatedObject

## Purpose

`ArticulatedObject` is the root authored assembly. Use it to create parts,
articulations, materials, and model-level metadata.

## Import

```python
from sdk import ArticulatedObject
```

## Recommended Surface

- `ArticulatedObject(...)`
- `model.part(...)`
- `model.articulation(...)`
- `model.material(...)`
- `model.get_part(...)`
- `model.get_articulation(...)`
- `model.get_joint(...)`
- `model.bind_placement(...)`
- `model.root_parts()`
- `model.validate(strict=True)`

## Construction

```python
ArticulatedObject(
    name: str,
    parts: list[Part] = [],
    articulations: list[Articulation] = [],
    materials: list[Material] = [],
    meta: dict[str, object] = {},
    assets=None,
)
```

Important fields:

- `name`: model name.
- `parts`: authored parts.
- `articulations`: authored articulations.
- `materials`: registered materials.
- `meta`: optional model-level metadata.
- `assets`: optional asset owner or root for mesh-backed authoring.
- Mesh assets are runtime-managed. Author mesh-backed visuals through helpers
  such as `mesh_from_geometry(...)`, `mesh_from_input(...)`, and
  `mesh_from_cadquery(...)`.

## Authoring Helpers

### `model.part(...)`

```python
model.part(
    name: str,
    *,
    visuals: Iterable[Visual] | None = None,
    inertial: Inertial | None = None,
    meta: dict[str, object] | None = None,
) -> Part
```

- Creates a `Part`, appends it to `model.parts`, and returns it.
- The returned part is the normal place to call `part.visual(...)`.

### `model.articulation(...)`

```python
model.articulation(
    name: str,
    articulation_type: ArticulationType | str,
    parent: str | Part,
    child: str | Part,
    *,
    origin: Origin | None = None,
    axis: tuple[float, float, float] | None = None,
    motion_limits: MotionLimits | None = None,
    motion_properties: MotionProperties | None = None,
    motion_feel: MotionFeel | dict[str, object] | None = None,
    mimic: Mimic | None = None,
    meta: dict[str, object] | None = None,
) -> Articulation
```

- `parent`, `child`: either part objects or part names.
- `origin`: transform from the parent part frame into the articulation frame.
  Defaults to `Origin()`.
- `axis`: motion axis expressed in the articulation frame. Defaults to
  `(0.0, 0.0, 1.0)`.
- `motion_limits`: joint limits where required.
- `motion_properties`: optional damping and friction.
- `motion_feel`: normalized passive hand-feel intent. Prefer
  `MotionFeel.from_intent(...)`; do not combine it with explicit damping/friction.
- `mimic`: optional derived-motion relationship to another scalar articulation.

Use `mimic` for simple linear couplings such as opposing gripper fingers,
belt-linked sliders, or other mechanisms where one scalar joint should follow
another as `q_follower = multiplier * q_source + offset`. Do not use it to
represent non-linear couplings or true closed-loop mechanics.

**Prefer `model.couple(driver, follower, sense=..., mapping=...)`** over hand-writing
`mimic=Mimic(...)`: it computes the multiplier and any unit conversion
deterministically from the two joints' limits, so you never compute a ratio or
radians yourself. `mapping="full"` makes the driver's full motion sweep the
follower's full travel (one revolution for a continuous driver); a number is an
explicit ratio. `sense="same"`/`"opposite"` sets the direction. The offset is
always 0 — both joints' rest pose (q=0) is the as-built pose and must correspond.

For a specific mechanism, `couple()` also fits a **nonlinear template** to a
degree-4 polynomial over the template's single-valued (monotone) window measured
from rest q=0: `kind="crank_slider"` (`crank_radius`, `rod_length`; one-sided
`[0, π]`), `kind="cam"` (`lift`, `rise_angle`; rise-and-hold, one-sided
`[0, rise_angle]`), or `kind="scotch_yoke"` (`radius`; symmetric `[-π/2, π/2]`).
A bounded driver's limits must lie inside that window (else the build is
rejected); a continuous driver uses the whole window. The full polynomial is
carried by MJCF (`<equality polycoef>`); URDF, USD, and the viewer use a single
SECANT-slope linear approximation that reaches the true peak excursion. Internally
the follower's `Mimic` carries `poly=(a0=0, a1, .., a4)` plus that secant.

`couple()` also expresses **dynamic behaviors** (the follower moves under its own
force law after a trigger, realized by a physics engine): `kind="latch_release"`
(`trigger_at`, `follower_goes`, `strength`), `kind="self_closer"` (`closes_to`,
`threshold`, `strength`; set driver == follower), `kind="momentary"`
(`follower_direction`, `speed`). These set the follower's `behavior`
(`ArticulationBehavior`) plus a joint spring (`MotionProperties.spring_stiffness`/
`spring_ref`) or velocity drive. MJCF emits the joint spring / velocity actuator;
USD authors a DriveAPI target + `physics:behavior:*` attributes; URDF cannot
express the trigger. A joint may have a mimic OR a behavior, not both.

Plain movable joints may instead declare passive hand feel:

```python
from sdk import MotionFeel

model.joint(
    "door_hinge",
    "revolute",
    "cabinet",
    "door",
    origin=Origin(xyz=(0.0, 0.0, 0.8)),
    axis=(0.0, 0.0, 1.0),
    limit=MotionLimits(lower=0.0, upper=1.6),
    motion_feel=MotionFeel.from_intent(
        effort="firm",
        motion="damped",
        hold="holds_anywhere",
        note="The door should remain where the user leaves it.",
    ),
)
```

`MotionFeel` is mutually exclusive with mimic and dynamic behavior ownership.
The export resolver derives physical damping/friction from the enriched subtree
mass, full inertia tensor, joint axis, and gravity loading. Coding should copy the
blueprint semantics, never tune raw physics values.

`couple()` (and `mimic`) may be **cross-domain**: a rotary driver
(revolute/continuous) can drive a linear follower (prismatic) — for example a focus
knob raising a focus carriage, or a lever releasing a gas-lift column. The
relationship stays affine; the `multiplier` carries the unit conversion (metres per
radian for a rack-and-pinion or lead-screw drive). The driver and follower must each
be a scalar joint (revolute/continuous/prismatic). **Chains are not allowed** (a
follower may not itself be a driver — couple each follower to its root driver and
compose the ratios), the source must exist, and the mimic graph must be acyclic.
When the driver has finite limits, the follower's reached range must stay within its
own limits. A continuous driver of a bounded follower is itself bounded (it stops
when the follower jams) and exports as a limited joint. The follower is derived, so
you do not pose it directly — drive its source.

## Frame And Direction Conventions

Articulations use a URDF-style joint frame:

1. `origin` places the articulation frame relative to the parent part frame.
2. `axis` is written in that articulation frame, not in world space.
3. At `q=0`, the child part frame is coincident with the articulation frame.
4. Positive `REVOLUTE` and `CONTINUOUS` motion follows the right-hand rule
   around `axis`.
5. Positive `PRISMATIC` motion translates the child along `+axis`.

In practice, the usual mistake is choosing the correct hinge line but the wrong
axis sign. If increasing the joint value makes the child close into the parent,
negate `axis` instead of swapping `lower` and `upper`.

### Example: Lid That Should Open Upward

```python
# Closed lid geometry extends along local +X from the hinge line.
# Using -Y makes positive q lift the free edge toward +Z.
model.articulation(
    "body_to_lid",
    ArticulationType.REVOLUTE,
    parent=body,
    child=lid,
    origin=Origin(xyz=(-0.09, 0.0, 0.05)),
    axis=(0.0, -1.0, 0.0),
    motion_limits=MotionLimits(effort=5.0, velocity=3.0, lower=0.0, upper=1.2),
)
```

### Example: Mirrored Lid With The Same "Positive Means Open" Convention

```python
# If the closed panel extends along local -X from the hinge line instead,
# flip the axis sign so positive q still opens upward/outward.
model.articulation(
    "body_to_mirrored_lid",
    ArticulationType.REVOLUTE,
    parent=body,
    child=mirrored_lid,
    origin=Origin(xyz=(0.09, 0.0, 0.05)),
    axis=(0.0, 1.0, 0.0),
    motion_limits=MotionLimits(effort=5.0, velocity=3.0, lower=0.0, upper=1.2),
)
```

### Example: Drawer That Should Extend Outward

```python
# Positive prismatic q moves the drawer out along +X.
model.articulation(
    "cabinet_to_drawer",
    ArticulationType.PRISMATIC,
    parent=cabinet,
    child=drawer,
    origin=Origin(xyz=(0.0, 0.0, 0.12)),
    axis=(1.0, 0.0, 0.0),
    motion_limits=MotionLimits(effort=40.0, velocity=0.25, lower=0.0, upper=0.28),
)
```

### Retained Insertion For Prismatic And Nested Stages

For sleeves, telescoping poles, nested rails, and similar slide-in-slide
assemblies, size the moving member for the fully extended pose, not just the
collapsed silhouette. If one part slides out of another, model the sliding
member with enough hidden length that it still remains engaged at the joint
upper limit.

For rectilinear nested members, use `TelescopingSlideSpec`: it validates the actual
parent/child axial interval at both home and upper travel. See
`geometry/telescoping-slides.md`. Do not use nested solid primitive cylinders or
boxes as sleeve proxies; their collision volumes fill the cavity.

```python
slide = TelescopingSlideSpec(
    members=(
        TelescopingMemberSpec("rect_tube", 0.060, 0.040, 0.004, 0.300),
        TelescopingMemberSpec("rect_tube", 0.050, 0.030, 0.003, 0.280),
    ),
    interfaces=(
        TelescopingInterfaceSpec(
            0, 1, clearance=5e-4, home_offset=0.020,
            travel=0.200, retained_insertion=0.080,
        ),
    ),
)
outer_sleeve = model.part("outer_sleeve")
inner_mast = model.part("inner_mast")
outer_sleeve.visual(
    mesh_from_geometry(TelescopingMemberGeometry(slide, member_index=0), "outer_sleeve")
)
inner_mast.visual(
    mesh_from_geometry(TelescopingMemberGeometry(slide, member_index=1), "inner_mast")
)
model.articulation(
    "sleeve_to_mast",
    ArticulationType.PRISMATIC,
    parent=outer_sleeve,
    child=inner_mast,
    origin=slide.joint_origin(0),
    axis=slide.joint_axis_local,
    motion_limits=MotionLimits(lower=0.0, upper=0.200, effort=80.0, velocity=0.20),
)
```

Round telescoping poles are outside this helper's V1 scope. If the asset requires
one, build a true hollow tube collision representation and validate its cavity; do
not substitute two overlapping solid `Cylinder` primitives.

### `model.material(...)`

```python
model.material(
    name: str,
    *,
    rgba: Sequence[float] | None = None,
    color: Sequence[float] | None = None,
    texture: str | None = None,
    material_class: str | None = None,
    metallic: float | None = None,
    roughness: float | None = None,
    emissive: float | None = None,
    texture_set: str | None = None,
    transmission: float | None = None,
    ior: float | None = None,
    attenuation_color: Sequence[float] | None = None,
    attenuation_distance: float | None = None,
) -> Material
```

- `name`: required material name. Registers the material on `model.materials`.
- `rgba`: 3 or 4 floats; a 3-tuple is expanded to `(r, g, b, 1.0)`.
- `color`: compatibility alias for `rgba`. Use either `rgba=...` or `color=...`,
  not both.
- `texture`: optional texture path.
- `material_class`: semantic finish (e.g. `steel`, `matte_plastic`, `glass`,
  `wood`, `emissive`) that maps to real PBR at render time. See
  `20_core_types.md` for the full enumerated list of known classes; an unknown
  class raises.
- `metallic`: optional `0..1` override of the class default.
- `roughness`: optional `0..1` override of the class default.
- `emissive`: optional emission strength (non-negative; `>0` for lit parts).
- `texture_set`: optional CC0 PBR texture-set name.
- `transmission`: optional `0..1` glass/refraction amount (rendered in Cycles).
- `ior`: optional index of refraction for glass (default `1.45`).
- `attenuation_color`: optional tinted-glass absorption color.
- `attenuation_distance`: optional tinted-glass absorption distance.

For the parameter meanings, value ranges, and the full enumerated set of
`material_class` finishes, see the `Material` entry in `20_core_types.md`.

## Lookup Helpers

### `model.get_part(name) -> Part`

```python
model.get_part(name: str | Part) -> Part
```

Returns the named part. Raises `ValidationError` if it does not exist.

### `model.get_articulation(name) -> Articulation`

```python
model.get_articulation(name: str | Articulation) -> Articulation
```

Returns the named articulation. Raises `ValidationError` if it does not exist.

### `model.get_joint(name) -> Articulation`

```python
model.get_joint(name: str | Articulation) -> Articulation
```

Alias of `model.get_articulation(...)`. Use it to fetch the joint you are about to
bind a placement result to.

### `model.root_parts() -> list[Part]`

Returns all parts that are not the child of another articulation.

## Placement Binding

### `model.bind_placement(target, result) -> None`

```python
model.bind_placement(
    target: PlacementTarget,
    result: PlacementResult,
) -> None
```

Applies one pure placement result to a model-owned pose field. This is the only
supported way to write a computed pose back into the model.

- `PlacementTarget.visual(visual)` writes `visual.origin`. The result subject must
  be that Visual and its frame must be `FrameRef.part_local(<owning part>)`.
- `PlacementTarget.joint(joint)` writes `joint.origin`. The result subject may be
  the joint or its child `Part`, and its frame must be
  `FrameRef.parent_link_local(<parent part>)`.

A mismatched subject or frame raises `ValidationError` rather than writing a
silently wrong pose. See `50_placement.md` for the Anchor and relation API.

### Example: A `FIXED` Mount Is Still A Dependent Pose

A fixture bolted onto a host — a monitor on a hood, a bracket on a panel — has no
motion, but its joint frame still follows the host's mounting face. Author it as a
relation so it tracks the face instead of going stale when the host changes.

```python
from sdk import (
    AnchorSelector,
    ArticulatedObject,
    ArticulationType,
    Box,
    FrameRef,
    Origin,
    PlacementTarget,
    RigidFrame,
    anchor,
    attach,
    explicit_anchor,
)

model = ArticulatedObject(name="hooded_monitor")

hood = model.part("hood")
roof = hood.visual(Box((0.34, 0.28, 0.012)), origin=Origin(xyz=(0.0, 0.0, 0.30)), name="hood_roof")

monitor = model.part("monitor")
# The monitor's own internal offset: the shell rises from its own mount point.
monitor.visual(Box((0.30, 0.02, 0.20)), origin=Origin(xyz=(0.0, 0.0, 0.10)), name="monitor_shell")

model.articulation("monitor_mount", ArticulationType.FIXED, parent=hood, child=monitor)

# The dependent pose: the mount frame sits ON the hood roof, 35% back along it.
mount_frame = FrameRef.parent_link_local(hood)
model.bind_placement(
    PlacementTarget.joint(model.get_joint("monitor_mount")),
    attach(
        # A Part's own frame origin is the subject of a joint placement.
        explicit_anchor(monitor, RigidFrame(), frame=mount_frame, kind="point"),
        anchor(roof, AnchorSelector.face("+z", uv=(0.5, 0.35)), frame=mount_frame),
    ),
)

mount_origin = model.get_joint("monitor_mount").origin  # Origin(xyz=(0.0, -0.042, 0.306))
```

## Validation

### `model.validate(...)`

```python
model.validate(
    *,
    strict: bool = True,
    strict_mesh_paths: bool = False,
) -> None
```

Validation includes:

- at least one part exists
- part names are unique
- articulation names are unique
- material names are unique
- articulations reference existing parent and child parts
- each part has at most one parent articulation
- geometry and materials are valid
- in strict mode, the articulation graph is connected from at least one root

## Joint Rules

Use these rules when calling `model.articulation(...)`:

- `ArticulationType.REVOLUTE` and `ArticulationType.PRISMATIC` require
  `MotionLimits(effort=..., velocity=..., lower=..., upper=...)`.
- `ArticulationType.CONTINUOUS` requires
  `MotionLimits(effort=..., velocity=...)` and must not set `lower` or `upper`.
- `ArticulationType.FIXED` and `ArticulationType.FLOATING` must not use
  `motion_limits`.
- Coupling (`model.couple(...)`, or `mimic=Mimic(...)`) is allowed only between
  scalar articulations (`REVOLUTE`, `CONTINUOUS`, `PRISMATIC`). Cross-domain
  (rotary<->linear) is allowed; chains (a follower that is itself a driver) and
  cycles are invalid; and a bounded driver must not drive its follower past the
  follower's own limits.
- In strict mode, articulated motion axes must be non-zero 3-vectors.

## Example

```python
model = ArticulatedObject(name="desk_lamp")

base = model.part("base")
arm = model.part("arm")

model.articulation(
    "base_to_arm",
    ArticulationType.REVOLUTE,
    parent=base,
    child=arm,
    origin=Origin(xyz=(0.0, 0.0, 0.08)),
    # Arm geometry extends along local +X from the shoulder, so -Y makes
    # positive q pitch the arm upward.
    axis=(0.0, -1.0, 0.0),
    motion_limits=MotionLimits(effort=5.0, velocity=2.0, lower=-0.8, upper=0.8),
)
```

## See Also

- `20_core_types.md` for `Part`, `Articulation`, `MotionLimits`, and materials
- `80_testing.md` for geometry and articulation QC

## Clarifications for agent usage

- `origin` defines the joint frame in the parent part frame. The child part then moves relative to that joint frame.
- `axis` is expressed in the articulation frame after applying `origin.rpy`, not in world space. Author it as a unit vector.
- `FLOATING` exists in the lower-level SDK but is outside the three-agent Pipeline authoring contract. Pipeline `model.py` files must use `REVOLUTE`, `CONTINUOUS`, `PRISMATIC`, or `FIXED`.
- If increasing a scalar joint value moves the child in the wrong direction, negate `axis`; do not swap semantic open/close meaning by flipping limits.
