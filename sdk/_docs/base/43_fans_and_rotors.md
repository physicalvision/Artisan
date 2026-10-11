# Fans and Rotors

## Purpose

Use these helpers for exposed axial rotors and blower-wheel style fan parts.
They stay on the base `sdk` surface and export through the normal
`mesh_from_geometry(...)` path.

> **Static by default.** Without `rotary=`, this helper only builds shape. For a
> functional free bearing or rigid shaft assembly, share one `RotaryConnectorSpec`
> with `FanRotorGeometry` and `RotaryConnectorGeometry`, then add one rotary joint.

## Import

```python
from sdk import (
    FanRotorGeometry,
    FanRotorBlade,
    FanRotorHub,
    FanRotorShroud,
    BlowerWheelGeometry,
    RotaryConnectorSpec,
    mesh_from_geometry,
)
```

## Recommended Surface

- `FanRotorGeometry` for axial fan rotors that spin about local `Z`
- `FanRotorBlade`, `FanRotorHub`, and `FanRotorShroud` for blade, hub, and
  tip-ring customization
- `BlowerWheelGeometry` for squirrel-cage blower wheels

### `FanRotorGeometry`

```python
FanRotorGeometry(
    outer_radius,
    hub_radius,
    blade_count,
    *,
    thickness,
    blade_pitch_deg: float = 28.0,
    blade_sweep_deg: float = 20.0,
    blade_root_chord=None,
    blade_tip_chord=None,
    blade: FanRotorBlade | None = None,
    hub: FanRotorHub | None = None,
    shroud: FanRotorShroud | None = None,
    rotary: RotaryConnectorSpec | None = None,
    center: bool = True,
)
```

- Builds an axial fan rotor that spins about local `Z`.
- `outer_radius` is the overall rotor envelope radius.
- `thickness` controls hub depth and the blade section thickness scale.
- `blade_pitch_deg`, `blade_sweep_deg`, and `FanRotorBlade.tip_pitch_deg` use
  degrees.
- `center=False` places the rear-most face on `z=0`.
- `rotary=` opts into the functional connector contract, owns the round hub bore,
  and requires `center=True`. Fan rotors support only `shaft_profile="round"`.
- Use `FanRotorBlade` for blade planform and tip-pitch tuning, `FanRotorHub`
  for hub style and bore control, and `FanRotorShroud` for an attached tip
  ring.

### `FanRotorBlade`

```python
FanRotorBlade(
    shape: Literal["straight", "scimitar", "broad", "narrow"] = "straight",
    tip_pitch_deg: float | None = None,
    camber: float = 0.0,
    tip_clearance: float = 0.0,
)
```

- `shape` changes the blade planform without changing the rest of the rotor API.
- `tip_pitch_deg` lets the tip run flatter or steeper than the blade root.
- `camber` adds visible section curvature to the lofted blade profile.
- `tip_clearance` trims the unshrouded blade tips back from `outer_radius`.

### `FanRotorHub`

```python
FanRotorHub(
    style: Literal["flat", "domed", "capped", "spinner"] = "domed",
    rear_collar_height: float | None = None,
    rear_collar_radius: float | None = None,
    bore_diameter: float | None = None,
)
```

- `style` controls the front hub silhouette.
- `rear_collar_height` and `rear_collar_radius` tune the back-side collar.
- `bore_diameter` cuts a centered through-bore for shafted rotors.
- Do not set `bore_diameter` when `FanRotorGeometry(rotary=...)` owns the fit.

See `53_rotary_connectors.md` for stationary/rotor halves, placement, clearance,
collision decomposition, and joint topology.

### `FanRotorShroud`

```python
FanRotorShroud(
    thickness,
    depth: float | None = None,
    clearance: float = 0.0,
    lip_depth: float = 0.0,
)
```

- Adds an attached tip ring around the rotor.
- `thickness` is the radial wall thickness of the ring.
- `depth` controls the axial depth of the ring.
- `clearance` leaves radial margin between the ring outer wall and the overall
  `outer_radius`.
- `lip_depth` adds a front-side axial extension of the ring.

### `BlowerWheelGeometry`

```python
BlowerWheelGeometry(
    outer_radius,
    inner_radius,
    width,
    blade_count,
    *,
    blade_thickness,
    blade_sweep_deg: float = 35.0,
    backplate: bool = True,
    shroud: bool = True,
    center: bool = True,
)
```

- Builds a squirrel-cage blower wheel that spins about local `Z`.
- `width` is the axial width along local `Z`.
- `blade_thickness` controls the vane thickness in plan view.
- The vane passages open into the interior cavity instead of terminating on a
  flat inner wall.
- `center=False` places the rear-most face on `z=0`.

## Examples

```python
fan_rotor = FanRotorGeometry(
    0.070,
    0.020,
    7,
    thickness=0.010,
    blade_pitch_deg=31.0,
    blade_sweep_deg=24.0,
    blade=FanRotorBlade(
        shape="scimitar",
        tip_pitch_deg=12.0,
        camber=0.16,
    ),
    hub=FanRotorHub(
        style="spinner",
        bore_diameter=0.005,
    ),
    shroud=FanRotorShroud(
        thickness=0.004,
        depth=0.012,
        clearance=0.0015,
        lip_depth=0.002,
    ),
)
mesh = mesh_from_geometry(fan_rotor, "fan_rotor")
```

```python
blower_wheel = BlowerWheelGeometry(
    0.080,
    0.040,
    0.050,
    18,
    blade_thickness=0.004,
    blade_sweep_deg=25.0,
)
mesh = mesh_from_geometry(blower_wheel, "blower_wheel")
```

## Related examples

- Search the base SDK examples for axial fan rotor and squirrel-cage blower
  wheel worked scripts.

## Clarifications for agent usage

- Use `FanRotorGeometry` for the rotating impeller itself. Do not use it to
  author the surrounding fan frame, grille, or housing.
- `FanRotorShroud` adds an attached tip ring; it does not build a separate
  stationary duct or housing.
- When you need an axial rotor with a shaft bore, prefer `FanRotorHub` over
  manual boolean cuts.
