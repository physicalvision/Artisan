# Wheels and Tires

## Purpose

Use these helpers for wheel-and-tire visuals that should read as real rolling
hardware, not as generic cylinders.

> **Static by default.** Without `rotary=`, this helper only builds shape. For a
> functional free bearing or rigid shaft assembly, share one `RotaryConnectorSpec`
> with `WheelGeometry` and `RotaryConnectorGeometry`, then add one rotary joint.

## Import

```python
from sdk import (
    WheelGeometry,
    WheelRim,
    WheelHub,
    WheelFace,
    WheelSpokes,
    WheelBore,
    WheelFlange,
    RotaryConnectorSpec,
    BoltPattern,
    TireGeometry,
    TireCarcass,
    TireTread,
    TireGroove,
    TireSidewall,
    TireShoulder,
    mesh_from_geometry,
)
```

## Recommended APIs

| Shape intent | Helper |
| --- | --- |
| Rim, hub, face, spokes, and mounting bore | `WheelGeometry` |
| Tire carcass, tread, grooves, sidewall, shoulder | `TireGeometry` |
| Wheel structural detail groups | `WheelRim`, `WheelHub`, `WheelFace`, `WheelSpokes`, `WheelBore`, `WheelFlange`, `BoltPattern` |
| Tire detail groups | `TireCarcass`, `TireTread`, `TireGroove`, `TireSidewall`, `TireShoulder` |

## API Reference

### `WheelGeometry`

```python
WheelGeometry(
    radius,
    width,
    *,
    rim: WheelRim | None = None,
    hub: WheelHub | None = None,
    face: WheelFace | None = None,
    spokes: WheelSpokes | None = None,
    bore: WheelBore | None = None,
    flange: WheelFlange | None = None,
    rotary: RotaryConnectorSpec | None = None,
    center: bool = True,
)
```

- Builds a wheel, rim, and hub visual that spins about local `X`.
- Width spans local `X`; the radial wheel profile lies in `YZ`.
- `center=False` places the inner mounting side on `x=0`.
- `rotary=` owns the bore and opts into a functional fit. It supports `round`,
  `keyed`, and `hex`, and requires `center=True`.
- Use the wheel detail dataclasses to cover solid wheels, slotted wheels,
  spoke wheels, deep-dish faces, and push-rim style variants.

### `TireGeometry`

```python
TireGeometry(
    outer_radius,
    width,
    *,
    inner_radius: float | None = None,
    carcass: TireCarcass | None = None,
    tread: TireTread | None = None,
    grooves: Sequence[TireGroove] = (),
    sidewall: TireSidewall | None = None,
    shoulder: TireShoulder | None = None,
    center: bool = True,
)
```

- Builds a tire aligned to local `X` so it composes directly with
  `WheelGeometry`.
- `inner_radius` controls the wheel-seat cavity; `outer_radius` controls the
  outside rolling radius.
- `center=False` places the inner side face on `x=0`.
- Use the tire detail dataclasses to cover smooth rollers, road tires, square
  utility tires, ribbed tires, chevron tires, and block/lug tires.

## Advice

- Compose wheel and tire as separate visuals when materials differ.
- Use the wheel helper for real face structure instead of engraving shallow
  spoke lines onto a disc.
- Use `WheelBore` for standalone/manual geometry. Use `rotary=` for a generated
  collision-aware axle/shaft fit; do not pass both. See `53_rotary_connectors.md`.
- Use the tire helper for real sidewall and tread shape instead of a plain
  torus or cylinder.

## Examples

### Wheel and tire pair

```python
wheel = WheelGeometry(
    0.120,
    0.040,
    rim=WheelRim(
        inner_radius=0.082,
        flange_height=0.010,
        flange_thickness=0.004,
        bead_seat_depth=0.004,
    ),
    hub=WheelHub(
        radius=0.028,
        width=0.030,
        cap_style="domed",
        bolt_pattern=BoltPattern(
            count=5,
            circle_diameter=0.034,
            hole_diameter=0.004,
        ),
    ),
    face=WheelFace(dish_depth=0.006, front_inset=0.003, rear_inset=0.002),
    spokes=WheelSpokes(style="split_y", count=5, thickness=0.003, window_radius=0.010),
    bore=WheelBore(style="round", diameter=0.012),
)
wheel_mesh = mesh_from_geometry(wheel, "wheel")

tire = TireGeometry(
    0.145,
    0.052,
    inner_radius=0.110,
    carcass=TireCarcass(belt_width_ratio=0.66, sidewall_bulge=0.08),
    tread=TireTread(style="chevron", depth=0.006, count=18, angle_deg=26.0, land_ratio=0.58),
    grooves=(TireGroove(center_offset=0.0, width=0.006, depth=0.003),),
    sidewall=TireSidewall(style="rounded", bulge=0.06),
    shoulder=TireShoulder(width=0.006, radius=0.004),
)
tire_mesh = mesh_from_geometry(tire, "tire")
```

### Utility tire

```python
utility_tire = TireGeometry(
    0.180,
    0.080,
    inner_radius=0.132,
    tread=TireTread(style="block", depth=0.010, count=20, land_ratio=0.55),
    sidewall=TireSidewall(style="square", bulge=0.02),
    shoulder=TireShoulder(width=0.010, radius=0.003),
)
mesh = mesh_from_geometry(utility_tire, "utility_tire")
```

## See Also

- `40_mesh_geometry.md` for lower-level revolved and shell geometry helpers
- For a fuller worked script, see the base SDK scooter wheel with road tire
  example.
