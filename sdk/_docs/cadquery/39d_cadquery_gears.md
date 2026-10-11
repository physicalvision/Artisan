# CadQuery Gears

## Purpose

Use this page when you need generated gears rather than hand-authored tooth
profiles. `sdk` vendors a CadQuery-compatible gear surface adapted from
`cq_gears` and re-exports it from top-level `sdk`.

The output is regular CadQuery geometry, so you can:

- call `.build()` and attach the result with `mesh_from_cadquery(...)`
- call `.assemble()` on pair/gearset helpers when available
- use the preserved `cadquery.Workplane.gear(...)` and `.addGear(...)` plugin
  workflow

## Import

```python
import cadquery as cq

from sdk import (
    BevelGear,
    BevelGearPair,
    CrossedGearPair,
    CrossedHelicalGear,
    GearBase,
    HerringboneGear,
    HerringbonePlanetaryGearset,
    HerringboneRackGear,
    HerringboneRingGear,
    HyperbolicGear,
    HyperbolicGearPair,
    PlanetaryGearset,
    RackGear,
    RingGear,
    SpurGear,
    Worm,
    addGear,
    gear,
)
```

## Dependency

This surface requires `cadquery` to be installed in the current environment.
In this repository, the expected setup command is:

```bash
uv sync --group dev
```

## Recommended Surface

- Single gears: `SpurGear`, `HerringboneGear`, `RingGear`,
  `HerringboneRingGear`, `BevelGear`, `RackGear`, `HerringboneRackGear`,
  `Worm`, `CrossedHelicalGear`, `HyperbolicGear`
- Assemblies and paired gears: `PlanetaryGearset`,
  `HerringbonePlanetaryGearset`, `BevelGearPair`, `CrossedGearPair`,
  `HyperbolicGearPair`
- Workplane plugin helpers: `gear(...)`, `addGear(...)`

## Shared Contract

### `GearBase.build(...)`

```python
gear.build(**kv_params) -> cadquery.Shape
```

- Single-gear classes return a CadQuery solid or compound.
- Pair and gearset helpers return a CadQuery compound from `.build(...)`.
- Constructor `**build_params` become default keyword arguments for later
  `.build(...)` or `.assemble(...)` calls.

## Single-Gear Families

### `SpurGear(...)`

```python
SpurGear(
    module,
    teeth_number,
    width,
    pressure_angle=20.0,
    helix_angle=0.0,
    clearance=0.0,
    backlash=0.0,
    addendum_coeff=None,
    dedendum_coeff=None,
    **build_params,
)
```

- `module`: tooth module in your current CadQuery units.
- `teeth_number`: number of teeth.
- `width`: body width along the gear axis.
- `pressure_angle`: tooth pressure angle in degrees.
- `helix_angle`: helical twist angle in degrees. `0.0` gives a spur gear.
- `clearance`: extra dedendum clearance.
- `backlash`: tooth backlash value.
- `addendum_coeff`: optional addendum coefficient override.
- `dedendum_coeff`: optional dedendum coefficient override.
- `**build_params`: default build-time overrides.

`build(...)` accepts:

```python
bore_d=None,
missing_teeth=None,
hub_d=None,
hub_length=None,
recess_d=None,
recess=None,
bottom_recess=None,
bottom_recess_d=None,
bottom_hub_d=None,
n_spokes=None,
spoke_width=None,
spoke_fillet=None,
spokes_id=None,
spokes_od=None,
chamfer=None,
chamfer_top=None,
chamfer_bottom=None,
```

- `bore_d`: center bore diameter.
- `missing_teeth`: one `(start, end)` tooth range or a list of ranges to remove.
- `hub_d`, `hub_length`: optional hub diameter and axial length.
- `recess_d`, `recess`: top-face recess diameter and depth.
- `bottom_recess`, `bottom_recess_d`, `bottom_hub_d`: bottom-face recess and
  hub overrides.
- `n_spokes`, `spoke_width`, `spoke_fillet`, `spokes_id`, `spokes_od`: spoke
  cutout controls.
- `chamfer`, `chamfer_top`, `chamfer_bottom`: tooth-edge chamfer controls.

### `HerringboneGear(...)`

Same constructor as `SpurGear(...)`.

- Builds a double-helical herringbone body.
- Uses the same `build(...)` parameters as `SpurGear(...)`.

### `RingGear(...)`

```python
RingGear(
    module,
    teeth_number,
    width,
    rim_width,
    pressure_angle=20.0,
    helix_angle=0.0,
    clearance=0.0,
    backlash=0.0,
    **build_params,
)
```

- `rim_width`: extra radial rim thickness outside the internal teeth.
- Remaining parameters match the meaning of `SpurGear(...)`.

`build(...)` accepts:

```python
chamfer=None,
chamfer_top=None,
chamfer_bottom=None,
```

### `HerringboneRingGear(...)`

Same constructor as `RingGear(...)`.

- Builds a double-helical internal ring gear.
- Uses the same `build(...)` parameters as `RingGear(...)`.

### `BevelGear(...)`

```python
BevelGear(
    module,
    teeth_number,
    cone_angle,
    face_width,
    pressure_angle=20.0,
    helix_angle=0.0,
    clearance=0.0,
    backlash=0.0,
    **build_params,
)
```

- `cone_angle`: pitch-cone angle in degrees.
- `face_width`: tooth face width along the cone.
- Remaining parameters match the meaning of `SpurGear(...)`.

`build(...)` accepts:

```python
bore_d=None,
trim_bottom=True,
trim_top=True,
```

- `bore_d`: center bore diameter.
- `trim_bottom`, `trim_top`: whether to trim the outer cone limits.

### `RackGear(...)`

```python
RackGear(
    module,
    length,
    width,
    height,
    pressure_angle=20.0,
    helix_angle=0.0,
    clearance=0.0,
    backlash=0.0,
    **build_params,
)
```

- `length`: total rack length along X.
- `width`: rack width along Z in the generated local frame.
- `height`: backing height below the tooth root.
- Remaining parameters match the meaning of `SpurGear(...)`.

`build()` takes no additional public parameters.

### `HerringboneRackGear(...)`

Same constructor as `RackGear(...)`.

- Builds a double-helical rack body.
- Uses the same `build()` contract as `RackGear(...)`.

### `Worm(...)`

```python
Worm(
    module,
    lead_angle,
    n_threads,
    length,
    pressure_angle=20.0,
    clearance=0.0,
    backlash=0.0,
    **build_params,
)
```

- `lead_angle`: thread lead angle in degrees.
- `n_threads`: number of starts/threads.
- `length`: worm length along its axis.

`build(...)` accepts:

```python
bore_d=None
```

### `CrossedHelicalGear(...)`

```python
CrossedHelicalGear(
    module,
    teeth_number,
    width,
    pressure_angle=20.0,
    helix_angle=0.0,
    clearance=0.0,
    backlash=0.0,
    **build_params,
)
```

- Use this when you need one skew-axis helical member directly rather than a
  pre-arranged pair.
- `build(...)` uses the same parameters as `SpurGear(...)`.

### `HyperbolicGear(...)`

```python
HyperbolicGear(
    module,
    teeth_number,
    width,
    twist_angle,
    pressure_angle=20.0,
    clearance=0.0,
    backlash=0.0,
    **build_params,
)
```

- `twist_angle`: total hyperbolic twist angle in degrees.
- `build(...)` uses the same parameters as `SpurGear(...)`.

## Pair And Gearset Helpers

### `PlanetaryGearset(...)`

```python
PlanetaryGearset(
    module,
    sun_teeth_number,
    planet_teeth_number,
    width,
    rim_width,
    n_planets,
    pressure_angle=20.0,
    helix_angle=0.0,
    clearance=0.0,
    backlash=0.0,
    **build_params,
)
```

- Builds a sun, equally spaced planets, and a ring gear.
- `build()` returns a compound.

`assemble(...)` accepts:

```python
build_sun=True,
build_planets=True,
build_ring=True,
sun_build_args={},
planet_build_args={},
ring_build_args={},
```

- `build_sun`, `build_ring`: enable or disable those members.
- `build_planets`: `True` for all planets, `False` for none, or a per-planet
  boolean list.
- `sun_build_args`, `planet_build_args`, `ring_build_args`: forwarded to the
  underlying gear `.build(...)` calls.

### `HerringbonePlanetaryGearset(...)`

Same constructor and `assemble(...)` contract as `PlanetaryGearset(...)`.

- Uses herringbone gears for the sun, planets, and ring.

### `BevelGearPair(...)`

```python
BevelGearPair(
    module,
    gear_teeth,
    pinion_teeth,
    face_width,
    axis_angle=90.0,
    pressure_angle=20.0,
    helix_angle=0.0,
    clearance=0.0,
    backlash=0.0,
    **build_params,
)
```

- `gear_teeth`: tooth count for the larger gear side.
- `pinion_teeth`: tooth count for the mating pinion.
- `face_width`: tooth face width.
- `axis_angle`: shaft angle in degrees.

`assemble(...)` accepts:

```python
build_gear=True,
build_pinion=True,
transform_pinion=True,
gear_build_args={},
pinion_build_args={},
```

- `transform_pinion`: place the pinion into the meshing pose automatically.
- `gear_build_args`, `pinion_build_args`: forwarded to the underlying
  `BevelGear.build(...)` calls.

### `CrossedGearPair(...)`

```python
CrossedGearPair(
    module,
    gear1_teeth_number,
    gear2_teeth_number,
    gear1_width,
    gear2_width,
    pressure_angle=20.0,
    shaft_angle=90.0,
    gear1_helix_angle=None,
    clearance=0.0,
    backlash=0.0,
    **build_params,
)
```

- `shaft_angle`: skew-axis angle in degrees.
- `gear1_helix_angle`: optional explicit helix for gear 1. If omitted, both
  members split the shaft angle evenly.

`assemble(...)` accepts:

```python
build_gear1=True,
build_gear2=True,
transform_gear2=True,
gear1_build_args={},
gear2_build_args={},
```

### `HyperbolicGearPair(...)`

```python
HyperbolicGearPair(
    module,
    gear1_teeth_number,
    width,
    shaft_angle,
    gear2_teeth_number=None,
    pressure_angle=20.0,
    clearance=0.0,
    backlash=0.0,
    **build_params,
)
```

- `width`: shared face width for both members.
- `shaft_angle`: skew-axis angle in degrees.
- `gear2_teeth_number`: optional second tooth count. Defaults to the first.

`assemble(...)` accepts:

```python
build_gear1=True,
build_gear2=True,
transform_gear2=True,
gear1_build_args={},
gear2_build_args={},
```

## Gear Train Helpers

Build a whole meshing spur-gear train in one call — correct centre distances, meshing
phases (baked into each returned shape), and coupling ratios. Phases are solved
numerically and cached by layout. Mount the result with ONE uniform transform; never
rotate a gear individually (that breaks the mesh).

### `planar_gear_train(nodes, module, *, width, backlash=0.0, pressure_angle=20.0, phase_steps=48)`

- `nodes`: list of `GearNode(teeth, parent=None, angle_deg=None, target_xy=None, build_args=None)`.
  Exactly one root (`parent=None`); every other gear meshes its `parent` along `angle_deg`
  (line-of-centres direction, 0=+X, 90=+Y) **or** toward `target_xy` (snapped to the mesh
  circle) — one or the other.
- Returns `list[ChainGear]`: `index`, `teeth`, `shape` (phase baked in, centred at origin,
  spin +Z), `center=(x, y, 0)`, `phase_deg`, `ratio` (= z_root/z_gear), `sense`
  ("same"/"opposite" vs the root by mesh parity; `None` for the root).
- SCOPE: planar **trees** (chains, branches, idlers). Closed loops raise. Compound /
  planetary are different mechanisms (`PlanetaryGearset`).

### `spur_gear_chain(teeth, module, width, *, backlash=0.0, pressure_angle=20.0, build_args=None, phases=None, phase_steps=48)`

- The **colinear** special case: a straight row along +X, each gear meshing the next.
  Returns the same `ChainGear` list. For any non-straight layout use `planar_gear_train`.
- `phases`: optional pre-resolved phases (degrees) to skip the search (freeze a chain).

### `solve_gear_chain_phases(teeth, module, *, width, backlash=0.0, pressure_angle=20.0, phase_steps=48)`

- Returns just the colinear meshing phases (degrees) — e.g. to bake as constants.

## Workplane Plugin Helpers

### `gear(...)`

```python
gear(
    workplane,
    gear_,
    *build_args,
    **build_kv_args,
) -> cadquery.Workplane
```

- Builds the given gear and places it at each point on the current workplane.
- `gear_`: any gear object with a `.build(...)` method, typically one of the
  classes documented above.
- `*build_args`, `**build_kv_args`: forwarded to `gear_.build(...)`.

This helper is also installed as:

```python
cq.Workplane.gear(...)
```

### `addGear(...)`

```python
addGear(
    workplane,
    gear_,
    *build_args,
    **build_kv_args,
) -> cadquery.Workplane
```

- Builds the gear and unions it into the current workplane solids.
- This helper is also installed as:

```python
cq.Workplane.addGear(...)
```

## Recommended Pattern

```python
import cadquery as cq

from sdk import SpurGear, mesh_from_cadquery

spur = SpurGear(module=0.001, teeth_number=20, width=0.006, bore_d=0.004)
body = cq.Workplane("XY").gear(spur).val()

mesh = mesh_from_cadquery(body, "spur_gear")
```

## Meshing Spur-Gear Trains — use `spur_gear_chain`

For a row of intermeshing spur gears that must ROTATE together (a gear train), do
NOT place and phase the gears by hand. A working train needs three things right at
once, and the last two are easy to get wrong:

- centre distance between neighbours = sum of their pitch radii
  (`0.5 * module * (z_a + z_b)`);
- the meshing PHASE of each gear, which is *chained* — gear `i`'s phase depends on
  gear `i-1`'s actual phase, NOT a fixed half-tooth offset (an independent half-tooth
  meshes the first pair but interpenetrates every pair after it);
- the EXACT `z0/zi` coupling ratio with alternating sense (a rounded decimal drifts
  over a revolution until the teeth collide).

```python
from sdk import spur_gear_chain  # returns one ChainGear per tooth count

chain = spur_gear_chain(
    [24, 16, 30, 18],        # tooth counts in chain order
    module=0.006,
    width=0.018,
    backlash=0.006 * 0.07,   # small positive backlash
    build_args=dict(bore_d=0.012),  # or a list, one .gear(...) kwargs dict per gear
)
# each g: g.shape (phase BAKED in, centred at origin, spin axis +Z), g.center (along +X),
#         g.phase_deg, g.ratio (= z0/zi), g.sense ("same"/"opposite"; None for gear 0)
# You: mesh g.shape, place at g.center, add a CONTINUOUS articulation about the spin
#      axis, and for followers: model.couple("gear_0_spin", f"gear_{g.index}_spin",
#      sense=g.sense, mapping=g.ratio).  No allow_overlap on the teeth.
```

The gears come back in the helper frame (XY plane, spin axis +Z, laid along +X); apply
ONE uniform transform to the whole chain to reorient it. SCOPE: simple chains only
(each gear meshes its two neighbours) — not planetary / compound / one-gear-meshing-3+.
ReviewAgent should confirm the finished chain in `qa_tests.py` with
`ctx.fail_if_parts_overlap_in_sampled_poses(...)`; CodingAgent must keep
`run_tests()` allowances-only.

## Notes

- The gear builders use the current CadQuery unit story. When exporting into
  `sdk`, keep the final geometry in meters or apply `unit_scale` at export time.
- Pair and gearset helpers expose `.assemble(...)` when you want the CadQuery
  assembly structure, and `.build(...)` when you want one compound shape.
- Importing any of these helpers through top-level `sdk` is sufficient to make
  the `cq.Workplane.gear` and `cq.Workplane.addGear` plugin methods available.
