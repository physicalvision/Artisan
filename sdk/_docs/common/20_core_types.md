# Core Types

## Purpose

This page documents the core data types used across the base `sdk`:
transforms, geometry descriptors, materials, inertials, parts, and
articulations.

## Import

```python
from sdk import (
    Origin,
    Box,
    Cylinder,
    Sphere,
    Mesh,
    Material,
    Visual,
    Inertia,
    Inertial,
    MotionLimits,
    MotionProperties,
    MotionFeel,
    Mimic,
    Part,
    Articulation,
    ArticulationType,
    scale_geometry_to_size,
)
```

## Units

- Distances are meters.
- Rotations are `Origin.rpy = (roll, pitch, yaw)` in radians.
- `Origin.rpy` uses the URDF-compatible composition `R = Rz(yaw) * Ry(pitch) * Rx(roll)`.
- Revolute and continuous joint positions are radians; prismatic joint
  positions are meters measured along the configured joint axis.
- URDF cylinders are aligned with local `+Z`.

## Recommended Surface

- `Origin`
- `Box`, `Cylinder`, `Sphere`, `Mesh`
- `Material`, `Visual`
- `Inertia`, `Inertial.from_geometry(...)`
- `MotionLimits`, `MotionProperties`, `MotionFeel`, `Mimic`
- `Part.visual(...)`, `Part.get_visual(...)`
- `Articulation`, `ArticulationType`
- `scale_geometry_to_size(...)`

## Transforms

### `Origin`

```python
Origin(
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
)
```

- `xyz`: translation in meters.
- `rpy`: rotation in radians, composed as `R = Rz(yaw) * Ry(pitch) * Rx(roll)`.
- Both fields must contain exactly 3 numbers.

### When A Coordinate Belongs In `Origin`, And When It Does Not

`Origin` stores every pose — including the ones a relation computes. The question
is not where the number is *stored*, it is where it *comes from*. The boundary is
**semantic dependency**, not structural nesting:

- The element's **own** coordinate — a design number that stands on its own, such
  as where a fascia sits on its drawer or how thick a shell wall is offset — is a
  hand-written `Origin(...)`. It answers to nothing else.
- A pose that **follows another element** — seated on it, flush with it, aligned
  to its edge, clearing it by a gap, spanning between two of them — is relational.
  Author it with `attach`/`gap`/`center_between`/`snap` and bind the result. See
  `50_placement.md`.

Part membership does **not** decide this. Two visuals in the **same part** are the
most common dependent pair there is: a display backing that must sit on a fascia
depends on that fascia exactly as much as a door depends on a cabinet. If a wrong
number there would penetrate or float, it depends on another element and belongs
in a relation. If it is pure styling proportion, keep the literal.

### Seating One Visual Against Another In The Same Part

```python
from sdk import (
    AnchorSelector,
    ArticulatedObject,
    Box,
    FrameRef,
    Origin,
    PlacementTarget,
    anchor,
    attach,
)

model = ArticulatedObject(name="control_drawer")
part = model.part("drawer")

# Own coordinate: hand-written Origin.
fascia = part.visual(
    Box((0.36, 0.018, 0.11)),
    origin=Origin(xyz=(0.0, 0.12, 0.055)),
    name="fascia",
)
# Dependent coordinate: no literal, the relation computes it.
display = part.visual(Box((0.14, 0.006, 0.05)), name="display_backing")

# Both Anchors take the OWNING PART's frame — the anchor() default would resolve
# to each Visual and attach() would reject the pair. See 50_placement.md.
frame = FrameRef.part_local(part)
seat = attach(
    anchor(display, AnchorSelector.face("-y"), frame=frame),
    anchor(fascia, AnchorSelector.face("+y"), frame=frame),
    axis_map={"+z": "-z", "+x": "+x"},
)
model.bind_placement(PlacementTarget.visual(display), seat)

seated_origin = display.origin  # Origin(xyz=(0.0, 0.132, 0.055))
```

## Geometry Descriptors

Geometry descriptors describe shape only. Put transforms on the containing
`Visual` or `Collision` via `origin=Origin(...)`.

### `Box`

```python
Box(size: tuple[float, float, float])
```

- `size`: full extents `(sx, sy, sz)` in meters.

### `Cylinder`

```python
Cylinder(radius: float, length: float)
```

- `radius`: cylinder radius in meters.
- `length`: full cylinder length along local `+Z`.
- `height`: accepted as a compatibility alias for `length`, but prefer `length`
  in new code.

### `Sphere`

```python
Sphere(radius: float)
```

- `radius`: sphere radius in meters.

### `Mesh`

```python
Mesh(
    filename: str | os.PathLike[str],
    scale: tuple[float, float, float] | None = None,
    source_geometry: Box | Cylinder | Sphere | None = None,
    source_transform: tuple[tuple[float, float, float, float], ...] | None = None,
)
```

- `filename`: absolute or asset-relative mesh path.
- `scale`: optional per-axis scale.
- `source_geometry`: optional primitive provenance for meshes that were
  generated from a primitive.
- `source_transform`: optional provenance transform. This requires
  `source_geometry`.

`filename` is required and must be non-empty.

## Materials and Visuals

### `Material`

```python
Material(
    name: str,
    rgba: tuple[float, float, float, float] | None = None,
    texture: str | None = None,
    *,
    color: tuple[float, float, float] | tuple[float, float, float, float] | None = None,
    material_class: str | None = None,   # semantic finish -> real PBR at render time
    metallic: float | None = None,       # 0..1 override of the class default
    roughness: float | None = None,      # 0..1 override of the class default
    emissive: float | None = None,       # emission strength (>0 for lit parts)
    texture_set: str | None = None,      # optional CC0 PBR texture-set name
    transmission: float | None = None,   # 0..1; glass/refraction (rendered in Cycles)
    ior: float | None = None,            # index of refraction (glass), default 1.45
    attenuation_color: tuple[float, float, float] | None = None,   # tinted glass
    attenuation_distance: float | None = None,
)
```

- `name`: required material name.
- `rgba`: 3 or 4 floats. A 3-tuple is expanded to `(r, g, b, 1.0)`.
- `color`: compatibility alias for `rgba`. Use either `rgba` or `color`, not both.
- `texture`: optional texture path.
- `material_class`: a semantic finish that maps to real PBR (metallic/roughness)
  at render time, so metal looks metallic and plastic looks like plastic. Known
  classes — metals: `steel`, `stainless_steel`, `aluminum`, `brushed_aluminum`,
  `brass`, `copper`, `chrome`, `gold`, `iron`, `anodized_metal`, `painted_metal`;
  plastics: `matte_plastic`, `glossy_plastic`, `abs_plastic`, `translucent_plastic`,
  `rubber`, `silicone`; `wood`, `painted_wood`, `ceramic`, `porcelain`, `concrete`,
  `stone`, `marble`, `fabric`, `leather`, `paper`, `cardboard`; glass: `glass`,
  `frosted_glass`, `tinted_glass`; `emissive`. An unknown class raises.
- `metallic`/`roughness`/`emissive`: optional explicit overrides of the class
  defaults (clamped to `[0, 1]`; emissive is a non-negative strength).
- `transmission`/`ior`/`attenuation_*`: glass/refraction. Set `material_class="glass"`
  (which supplies transmission=1.0, ior=1.45) or pass `transmission`/`ior` directly.
  Numeric transmission/IOR/attenuation are emitted in the articulated and posed GLBs
  through standard Khronos material extensions and rendered by Cycles;
  `attenuation_color`/`attenuation_distance` describe tinted (colour-absorbing) glass.
  The articulated GLB automatically unwraps visuals whose resolved texture set is
  available, bakes the Box Projection into shared Base Color, Metallic-Roughness,
  and Normal atlases, and embeds those images in the GLB. Materials without an
  available texture set remain numeric PBR. Static pose GLBs retain the legacy
  renderer-side Box Projection sidecar.
- Material assignment is per visual (sub-part): to give one part several finishes,
  add multiple `visual(...)` calls, each with its own Material.

### `Visual`

```python
Visual(
    geometry: Box | Cylinder | Sphere | Mesh,
    origin: Origin = Origin(),
    material: Material | str | None = None,
    name: str | None = None,
)
```

- `geometry`: visible geometry descriptor.
- `origin`: local transform of this visual on its part.
- `material`: either a `Material` object or a registered material name.
- Material assignment is per visual. Mesh subregions do not receive separate SDK
  materials unless they are authored as separate visuals.
- `color`: when adding visuals through `Part.visual(...)`, accepted as a
  compatibility alias for `material`; prefer `material` in new code.
- `name`: optional visual name. Use this when tests or probe snippets need to
  target a local feature.

## Inertials

### `Inertia`

```python
Inertia(
    ixx: float,
    ixy: float,
    ixz: float,
    iyy: float,
    iyz: float,
    izz: float,
)
```

Stores the inertia tensor components for one part.

### `Inertial`

```python
Inertial(
    mass: float,
    inertia: Inertia,
    origin: Origin = Origin(),
)
```

### `Inertial.from_geometry(...)`

```python
Inertial.from_geometry(
    geometry: Box | Cylinder | Sphere | Mesh,
    mass: float,
    *,
    origin: Origin | None = None,
) -> Inertial
```

- `mass`: required positive mass.
- `origin`: optional inertial origin.
- Supports `Box`, `Cylinder`, and `Sphere`.
- Raises `ValidationError` for mesh geometry.

## Motion

### `MotionLimits`

```python
MotionLimits(
    effort: float = 1.0,
    velocity: float = 1.0,
    lower: float | None = None,
    upper: float | None = None,
)
```

- `effort`: positive effort limit.
- `velocity`: positive velocity limit.
- `lower`, `upper`: optional joint-position bounds.

### `MotionProperties`

```python
MotionProperties(
    damping: float | None = None,
    friction: float | None = None,
)
```

Both fields are optional.

### `MotionFeel`

```python
MotionFeel.from_intent(
    effort: "light" | "medium" | "firm" | "stiff" | float | None = None,
    motion: "coasting" | "smooth" | "damped" | float | None = None,
    hold: "free" | "holds_anywhere" | None = None,
    note: str = "",
) -> MotionFeel
```

`MotionFeel` is the semantic, portable way to describe passive hand feel. Use it
only on a plain revolute, continuous, or prismatic joint. It is rejected on mimic
followers, behavior-owned joints, fixed/floating joints, and joints that also set
explicit damping or friction.

- `effort` controls normalized breakaway resistance.
- `motion` controls passive release decay.
- `hold="holds_anywhere"` requests a passive gravity-hold floor in MuJoCo.
- `note` preserves design context but does not change the numeric mapping.

The physics resolver owns damping and friction numbers. Do not derive or copy
`MotionProperties` values by hand. MuJoCo is the exact behavioral authority. The
Isaac Sim 4.5 USD profile is purely passive: it maps `effort` to dimensionless
`physxJoint:jointFriction`, uses only a zero-stiffness damping Drive, and records
`holds_anywhere` as `unsupported-passive-isaac45` because passive 4.5 friction
cannot guarantee static hold at every pose.

### `Mimic`

```python
Mimic(
    joint: str,
    multiplier: float = 1.0,
    offset: float = 0.0,
)
```

- `joint`: source articulation name.
- `multiplier`: linear scale applied to the source pose.
- `offset`: additive offset applied after scaling.

### `ArticulationType`

```python
ArticulationType.REVOLUTE
ArticulationType.CONTINUOUS
ArticulationType.PRISMATIC
ArticulationType.FIXED
ArticulationType.FLOATING
```

Use these enum values when creating articulations.

Pipeline authors must not create `FLOATING` articulations. The three-agent
Pipeline authoring subset is `REVOLUTE`, `CONTINUOUS`, `PRISMATIC`, and `FIXED`;
`FLOATING` remains listed only because it exists in the lower-level SDK API.

## Parts and Articulations

### `Part`

```python
Part(
    name: str,
    visuals: list[Visual] = [],
    collisions: list[Collision] = [],
    inertial: Inertial | None = None,
    meta: dict[str, object] = {},
)
```

Recommended part authoring entry points:

```python
part.visual(
    geometry,
    *,
    origin: Origin | None = None,
    material: Material | str | None = None,
    name: str | None = None,
) -> Visual

part.get_visual(name: str) -> Visual
```

- Use `part.visual(...)` to append one named or unnamed visual.
- Use `part.get_visual(...)` when tests or probe tooling need a specific local
  feature.

### `Articulation`

```python
Articulation(
    name: str,
    articulation_type: ArticulationType | str,
    parent: str,
    child: str,
    origin: Origin = Origin(),
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    motion_limits: MotionLimits | None = None,
    motion_properties: MotionProperties | None = None,
    mimic: Mimic | None = None,
    meta: dict[str, object] = {},
)
```

- `parent`, `child`: part names.
- `origin`: transform from the parent part frame into the articulation frame.
- `axis`: motion axis for revolute, continuous, and prismatic joints,
  expressed in the articulation frame.
- `origin.rpy` rotates the articulation frame relative to the parent, so it
  also rotates the meaning of `axis`.
- `mimic`: optional derived-motion relationship to another scalar articulation.
- Direction conventions and joint-limit rules live in `30_articulated_object.md`.

For authored models, prefer `model.articulation(...)` rather than constructing
`Articulation` objects manually.

## Supported Public Fields On Resolved Objects

Objects returned by `model.get_part(...)`, `model.get_articulation(...)`, and
`part.get_visual(...)` are public SDK objects. In tests, probe snippets, and
other authored code, reading the following fields and aliases is supported.

### `Part`

- `part.name`
- `part.visuals`
- `part.collisions`
- `part.inertial`
- `part.meta`
- `part.assets`

### `Visual`

- `visual.name`
- `visual.geometry`
- `visual.origin`
- `visual.material`

### `Articulation`

- `joint.name`
- `joint.articulation_type`
- `joint.joint_type`
- `joint.parent`
- `joint.child`
- `joint.origin`
- `joint.axis`
- `joint.motion_limits`
- `joint.limit`
- `joint.motion_properties`
- `joint.dynamics`
- `joint.mimic`
- `joint.meta`

### `MotionLimits`

- `limits.effort`
- `limits.velocity`
- `limits.lower`
- `limits.upper`

Example:

```python
hinge = object_model.get_articulation("lid_hinge")
limits = hinge.motion_limits
if limits is not None and limits.lower is not None and limits.upper is not None:
    lower = limits.lower
    upper = limits.upper
```

## Resizing Helper

### `scale_geometry_to_size(...)`

```python
scale_geometry_to_size(
    geometry,
    target_size,
    *,
    mode="stretch",
    asset_root=None,
    filename=None,
)
```

- `geometry`: primitive geometry or `Mesh`.
- `target_size`: target extents `(x, y, z)`. Each axis may be a positive float
  or `None`.
- `mode`: `"stretch"` or `"uniform"`.
- `asset_root`: mesh root used when resizing mesh geometry.
- `filename`: required when a resized primitive can no longer remain a primitive
  and must be emitted as a mesh.

### Advice

- Use `mode="stretch"` when exact target extents matter more than preserving
  aspect ratio.
- Use `mode="uniform"` when the shape should stay proportional.
- Primitive resizing stays primitive only when the result is still representable
  as that primitive. For example, stretching a sphere unevenly requires
  `filename=...` so the helper can emit a mesh-backed result.

## Examples

```python
badge = Material(name="badge", rgba=(0.8, 0.2, 0.1, 1.0))

panel = Part(name="panel")
panel.visual(
    Box((0.10, 0.16, 0.003)),
    origin=Origin(xyz=(0.0, 0.0, 0.0015)),
    material=badge,
    name="panel_shell",
)
```

```python
scaled = scale_geometry_to_size(
    Sphere(radius=0.02),
    (0.04, 0.06, 0.01),
    filename="assets/meshes/badge.obj",
)
```

## See Also

- `30_articulated_object.md` for model authoring helpers
- `50_placement.md` for mounting and surface wrapping helpers
- `80_testing.md` for test-time use of parts and named visuals

## Clarifications for agent usage

- `Origin` is always a local rigid transform: `xyz` is translation in meters and `rpy` is roll, pitch, yaw in radians using the right-hand rule and the URDF-compatible composition `R = Rz(yaw) * Ry(pitch) * Rx(roll)`.
- A part visual or collision is placed in its part-local frame with `origin=Origin(...)`; an articulation `origin` defines the child joint frame relative to the parent part frame.
- `Mesh(...)` requires either `filename` or `name`. Use `filename` for a concrete mesh asset path and `name` for managed/materialized mesh assets that will be resolved later.
- `Part.collisions` is not part of the authored source-model path for agent use. Author visuals; derived collision geometry is handled by validation/materialization.
- `ArticulationType.FLOATING` exists in the lower-level SDK but is outside the three-agent Pipeline authoring contract. Do not use it in `model.py`.
- For `REVOLUTE`, `CONTINUOUS`, and `PRISMATIC`, author `axis` as a unit vector in the articulation frame. Positive revolute motion follows the right-hand rule around `+axis`; positive prismatic motion translates along `+axis`.
- `Mimic(...)` is supported for scalar articulations. Mimic followers are derived from their source joint as `q_follower = multiplier * q_source + offset` and should not be posed directly in testing or probe code.
