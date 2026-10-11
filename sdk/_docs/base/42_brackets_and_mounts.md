# Brackets and Mounts

## Purpose

The original helpers on this page build standalone bracket visuals. They do not
define the mating half, clearance contract, collision primitives, or joint frame.
For a working clevis, pivot fork, or trunnion joint, use `PinnedJointSpec` plus
the two `PinnedJointGeometry` halves from `52_pinned_joints.md`.

## Import

```python
from sdk import (
    ClevisBracketGeometry,
    PinnedJointGeometry,
    PinnedJointSpec,
    PivotForkGeometry,
    TrunnionYokeGeometry,
    mesh_from_geometry,
)
```

## Recommended APIs

| Shape intent | Helper |
| --- | --- |
| Functional two-part clevis/fork/trunnion joint | `PinnedJointSpec` + `PinnedJointGeometry` |
| Legacy standalone clevis visual | `ClevisBracketGeometry` |
| Legacy standalone open-fork visual | `PivotForkGeometry` |
| Legacy standalone yoke visual | `TrunnionYokeGeometry` |

Do not combine a legacy bracket helper with an independent pin Part or contact
exclusion to simulate a joint. The functional connector uses two Parts, one
`REVOLUTE` joint, and no independent pin Part.

## API Reference

### `ClevisBracketGeometry`

```python
ClevisBracketGeometry(
    overall_size,
    *,
    gap_width,
    bore_diameter,
    bore_center_z,
    base_thickness,
    corner_radius: float = 0.0,
    center: bool = True,
)
```

- Builds a U-shaped clevis with a bottom base and a transverse bore.
- `overall_size`: `(width_x, depth_y, height_z)`.
- `gap_width` is the clear spacing between the cheeks.
- `bore_center_z` is measured upward from the bottom face.
- `center=False` places the bottom mounting face on `z=0`.

### `PivotForkGeometry`

```python
PivotForkGeometry(
    overall_size,
    *,
    gap_width,
    bore_diameter,
    bore_center_z,
    bridge_thickness,
    corner_radius: float = 0.0,
    center: bool = True,
)
```

- Builds an open-front fork with two tines and a rear bridge.
- `overall_size`: `(width_x, depth_y, height_z)`.
- `gap_width` is the clear spacing between the tines.
- `bridge_thickness` is the rear connecting bridge thickness along local `Y`.
- `bore_center_z` is measured upward from the bottom face.

### `TrunnionYokeGeometry`

```python
TrunnionYokeGeometry(
    overall_size,
    *,
    span_width,
    trunnion_diameter,
    trunnion_center_z,
    base_thickness,
    corner_radius: float = 0.0,
    center: bool = True,
)
```

- Builds a trunnion support yoke with cheek-mounted trunnion bores.
- `overall_size`: `(width_x, depth_y, height_z)`.
- `span_width` is the clear opening between the cheeks.
- `trunnion_center_z` is measured upward from the bottom face.
- `center=False` places the base on `z=0`.

## Legacy Visual-Only Advice

- Use these helpers only when a bracket is a standalone visual/mount and no
  mating functional connector is required.
- Preserve actual insertion and swing clearance. Do not close a fork or yoke
  just because a box is easier to author.
- If the feature is really a flat bent plate without a cheeked opening, use the
  lower-level mesh tools instead.

## Examples

```python
clevis = ClevisBracketGeometry(
    (0.08, 0.04, 0.06),
    gap_width=0.032,
    bore_diameter=0.012,
    bore_center_z=0.038,
    base_thickness=0.012,
)
clevis_mesh = mesh_from_geometry(clevis, "clevis")
```

```python
pivot_fork = PivotForkGeometry(
    (0.08, 0.05, 0.05),
    gap_width=0.034,
    bore_diameter=0.010,
    bore_center_z=0.028,
    bridge_thickness=0.012,
)
pivot_fork_mesh = mesh_from_geometry(pivot_fork, "pivot_fork")
```

```python
trunnion_yoke = TrunnionYokeGeometry(
    (0.12, 0.05, 0.08),
    span_width=0.060,
    trunnion_diameter=0.016,
    trunnion_center_z=0.050,
    base_thickness=0.014,
)
trunnion_yoke_mesh = mesh_from_geometry(trunnion_yoke, "trunnion_yoke")
```

## See Also

- `52_pinned_joints.md` for functional two-part pinned connectors
- `40_mesh_geometry.md` for lower-level extrudes and profile helpers
- `50_placement.md` for mounting bracket-backed parts onto parent bodies
