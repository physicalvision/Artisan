# Probe Tooling

## Purpose

`probe_model` is an inspection-only tool for running a short Python snippet
against the current `object_model`. Use it to measure the current model, learn
what a pose or relationship looks like, and then encode the lasting invariant
in ReviewAgent's `qa_tests.py::qa_tests(ctx)` when the invariant is part of the
binding acceptance criteria. CodingAgent's `run_tests()` remains allowances-only.

## Call Contract

Tool parameters:

- `code: str`
- `timeout_ms: int = 600000`
- `include_stdout: bool = False`

Execution rules:

- Write normal Python code and call `emit(value)` exactly once.
- `value` must be JSON-serializable.
- Do not mutate files, spawn subprocesses, or use the network.

## Preloaded Names

- `object_model`
- `ctx`
- `emit(value)`
- `catalog()`

## Lookup and Pose Helpers

### `pose(mapping: dict[str, float | Origin] | None = None, **kwargs: float | Origin) -> ContextManager`

Temporarily apply articulation positions.

- Inputs: articulation names mapped to scalar joint positions or `Origin(...)` for `FLOATING`.
- Revolute and continuous positions are radians; prismatic positions are
  meters. Floating positions use `Origin.xyz` in meters and `Origin.rpy` in radians.
- Positive scalar values follow the joint convention encoded by `axis`. Floating articulations do not use `axis`.
- Return: context manager for `with pose(...):`.

### `part(name: str) -> object`

Return the named part.

### `joint(name: str) -> object`

Return the named articulation.

### `visual(part_name: str, visual_name: str) -> object`

Return a named visual from a named part.

### `parts() -> list[object]`

Return all parts.

### `joints() -> list[object]`

Return all articulations.

### `visuals(part_or_name: object) -> list[object]`

Return all visuals for the given part.

- Accepted inputs: part object, part name, or visual whose owning part should be
  resolved.

### `name(obj: object) -> str`

Return a readable name for a part, articulation, or visual.

## Measurement Helpers

Accepted target types are part objects, visual objects, or compatible lookup
results unless a helper says otherwise.

### `aabb(obj: object) -> dict[str, list[float]] | None`

Return world-space AABB:

```python
{"min": [x, y, z], "max": [x, y, z]}
```

Also accepts the whole `object_model`, in which case it returns the union
(overall) AABB across all parts.

### `dims(obj: object) -> list[float] | None`

Return exact projection-derived size as `[dx, dy, dz]`.

Also accepts the whole `object_model`, in which case it returns the overall
extent of the assembled object. Use this to compare the built object against
the blueprint `target_dimensions` instead of summing parts by hand.

### `center(obj: object) -> list[float] | None`

Return exact projection-derived center as `[x, y, z]`.

Also accepts the whole `object_model`, in which case it returns the overall
center across all parts.

### `overall_aabb() -> dict[str, list[float]] | None` / `overall_dims() -> list[float] | None` / `overall_center() -> list[float] | None`

Explicit whole-object measurements (the union of exact part geometry). These are
equivalent to passing `object_model` into `aabb` / `dims` / `center`. Parts that
do not yield exact geometry are skipped; the result is `None` only when no part
has exact geometry.

### `position(obj: object) -> list[float] | None`

Return representative world position.

- Part: part world position
- Visual: visual center
- Articulation: articulation origin when available

### `projection(obj: object, axis_or_axes: str | Sequence[str]) -> dict[str, object]`

Return exact geometry projection intervals along one or more axes.

### `summary(obj: object) -> dict[str, object]`

Return a compact structured summary.

- Part summaries include name, position, center, dims, AABB, and visual names.
- Articulation summaries include name, type, axis, and motion-limit fields.

## Pair and Relationship Reports

These helpers are intended for direct `emit(...)` use.

### `pair_report(a, b, elem_a=None, elem_b=None) -> dict[str, object]`

Return exact pairwise distance, overlap, and projection details. The
authoritative collision verdict is the exact FCL pair result exposed as
`collided` (bool) and `min_distance` (meters); the per-axis `overlap_*` /
`gap_*` values are coarse projection numbers and can read as overlapping even
when the bodies do not actually intersect.

### `gap_report(positive, negative, axis, positive_elem=None, negative_elem=None) -> dict[str, object]`

Return signed directional gap along one axis.

- Positive values mean separation.
- Negative values mean penetration.

### `overlap_report(a, b, axes="xy", elem_a=None, elem_b=None) -> dict[str, object]`

Return projection overlap along one or more axes plus `min_overlap`. The
per-axis `overlap` map and `min_overlap` are COARSE projection numbers: a
positive overlap on every requested axis does not prove the bodies intersect.
The report now also carries the authoritative exact FCL verdict as `collided`
(bool) and `min_distance` (meters) — the same values `pair_report(...)`
returns. Decide actual collision from `collided` / `min_distance`, not from the
projection overlap.

### `within_report(inner, outer, axes="xy", inner_elem=None, outer_elem=None) -> dict[str, object]`

Return per-axis containment margins plus overall `within`.

### `contact_report(a, b, elem_a=None, elem_b=None, contact_tol=1e-6) -> dict[str, object]`

Return contact decision, `min_distance`, and collision status.

### `mount_report(child, parent, elem_a=None, elem_b=None) -> dict[str, object]`

Return a mounting-oriented review combining pair, XY containment, and Z gap.

### `containment_report(inner, outer, axes="xy") -> dict[str, object]`

Containment-oriented alias for `within_report(...)`.

### `alignment_report(a, b) -> dict[str, object]`

Return signed and absolute world-space center delta.

## Review Helpers

### `sample_poses(max_samples: int = 32, seed: int = 0) -> list[dict[str, float | Origin]]`

Return articulation pose samples.

### `nearest_neighbors(obj, candidates=None, limit: int = 5) -> list[dict[str, object]]`

Return the closest candidate parts relative to the target.

### `find_clearance_risks(limit: int = 10, parts=None) -> list[dict[str, object]]`

Return likely collision or clearance issues.

### `find_floating_parts(limit: int = 10, parts=None) -> list[dict[str, object]]`

Return parts whose nearest-neighbor relationship suggests they may read as
floating.

### `geometry_connectivity_report(part_or_name, contact_tol: float = 1e-6) -> dict[str, object]`

Return a part-level connectivity diagnosis that compares:

- raw connected components inside mesh-backed visuals or collisions
- compiled exact-collision entry count
- the current SDK connectivity-QC finding, if any

Use this when a part visibly contains floating islands but
the automated compile warning did not fire.

### `layout_report(items, axis: str = "x") -> dict[str, object]`

Return repeated-spacing review along one axis.

### `grid_report(items, axes="xy") -> dict[str, object]`

Return approximate 2D grid structure.

### `symmetry_report(items, axis: str = "x") -> dict[str, object]`

Return approximate bilateral symmetry report.

## Advice

- Use probe snippets to learn what the current model is doing.
- After Review finds the right invariant, move it into `qa_tests.py` with the
  corresponding `expect_*` assertion so it persists across repair turns.
- Prefer object-first snippets: resolve parts, articulations, and visuals into
  locals once, then pass those objects into the report helpers.
- If compile feedback suggests floating geometry, overlap risk, or a suspicious
  support relationship and the cause is not obvious, probe the current geometry
  before editing.
- Prefer a short focused probe that answers one spatial question over a large
  diagnostic script that mixes many checks at once.

## Full-collision motion signals (PENETRATION / SIM-JAM)

The build and review feedback name two deterministic full-collision signals. Both mean
a moving part's BODY does not clear a neighbour under full-collision checks (`filterparent` disabled),
which every rest-baselined screen misses. Recognise them and probe them:

- **PENETRATION `a↔b depth≈N mm along <axis>`** — the exact source geometry has `a` and
  `b` interpenetrating by `N` mm (measured against ZERO, at rest or in motion). `CONSTANT
  from rest` means it is already there when closed. It is a geometry defect that needs
  positive clearance; `<axis>` is the deepest-contact normal (a locating hint, NOT a
  prescribed fix direction). Which fix — resize/reposition the mover, widen the opening,
  add/swap a connector — is yours to judge from the whole design; NEVER clamp joint limits
  or add `allow_overlap`. Confirm and measure with `pair_report(a, b)` (→ `min_distance`
  < 0) and `gap_report(a, b, axis="<axis>")` at the named pose.
- **SIM-JAM `joint J reaches only N% ... pair a↔b`** — MuJoCo drove joint `J` under full
  collision and it tracked only `N%` of its contact-free travel because `a↔b` resisted (a
  real dynamic jam in the shipped scene). Same fix: clearance at the named pair. Probe the
  pair with `pair_report`/`gap_report` across the joint's travel (`pose({J: ...})`).

`find_clearance_risks(...)` surfaces the same body-vs-neighbour risks proactively; Review
can encode a blueprint-grounded confirmed gap as an `expect_*` in `qa_tests.py`.

## Common Probe Use Cases

- Overlap or collision classification: start with `pair_report(...)`,
  `overlap_report(...)`, or `mount_report(...)`.
- Floating-looking or weakly supported parts: start with
  `find_floating_parts(...)`, `nearest_neighbors(...)`, or `mount_report(...)`.
- Fit, containment, or directional clearance ambiguity: start with
  `within_report(...)` or `gap_report(...)`.
- A PENETRATION or SIM-JAM signal: confirm + size the fix with `pair_report(...)` and
  `gap_report(..., axis=<the named axis>)` at the reported pose.
- Suspicious disconnected mesh islands inside one part: start with
  `geometry_connectivity_report(...)`.

## Examples

Mounted feature review:

```python
panel = part("panel")
knob = visual("panel", "knob")
emit(mount_report(knob, panel))
```

Pose-aware contact check:

```python
with pose(lid_hinge=1.0):
    emit(contact_report(part("lid"), part("frame")))
```

Overlap classification for a reported pair:

```python
emit(pair_report(part("spindle_head"), part("table"), elem_a="quill", elem_b="table_disk"))
```

Floating/support-path review for a suspicious part:

```python
handle = part("handle")
emit({"floating": find_floating_parts(parts=[handle], limit=1), "neighbors": nearest_neighbors(handle, limit=3)})
```

Repeated layout review:

```python
keys = [visual("keyboard", name) for name in ("key_1", "key_2", "key_3", "key_4")]
emit(layout_report(keys, axis="x"))
```

Disconnected-island diagnosis for a suspicious mesh part:

```python
report = geometry_connectivity_report("left_frame")
emit(
    {
        "part": report["part"],
        "has_raw_disconnected_components": report["has_raw_disconnected_components"],
        "disconnected_items": report["disconnected_items"],
        "compiled_collision_count": report["compiled_collision_count"],
        "qc_detected_disconnected_islands": report["qc_detected_disconnected_islands"],
        "blind_spot_suspected": report["blind_spot_suspected"],
    }
)
```

## See Also

- `80_testing.md` for the persistent test API

## Floating pose notes

- `FLOATING` is outside the three-agent Pipeline authoring contract. The notes
  below document the lower-level SDK API only; do not create one in `model.py`.
- `pose(...)` accepts mixed values: scalar `float` for scalar joints, `Origin(...)` for `FLOATING`.
- `sample_poses(...)` returns the same mixed pose-map shape. Floating joints sample only `Origin()` by default unless the articulation metadata provides explicit `qc_samples` made of `Origin(...)` values.
- Probe `summary(joint)` reports floating joints as rigid 6-DOF motion with `Origin` pose values rather than an axis-driven scalar motion model.
