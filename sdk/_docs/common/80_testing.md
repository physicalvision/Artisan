# Testing

Import from top-level `sdk`:

```python
from sdk import AllowedOverlap, TestContext, TestFailure, TestReport
```

`TestContext` records blocking failures, non-blocking warnings, and explicit
allowances. In the three-agent Pipeline it has two distinct owners:

- CodingAgent's required `model.py::run_tests()` declares intentional baseline-QC
  allowances only, or returns an empty report.
- ReviewAgent authors `qa_tests.py::qa_tests(ctx)` and owns prompt-specific,
  functional, kinematic, contact, and acceptance assertions.

The complete API remains documented here because ReviewAgent reads this page
before authoring QA.

## Public Types

### `TestFailure`

```python
TestFailure(name: str, details: str)
```

- `name`: recorded check name.
- `details`: failure detail string stored in the report.

### `AllowedOverlap`

```python
AllowedOverlap(
    link_a: str,
    link_b: str,
    reason: str,
    elem_a: str | None = None,
    elem_b: str | None = None,
)
```

- Structured overlap allowance recorded by `allow_overlap(...)`.

### `TestReport`

```python
TestReport(
    passed: bool,
    checks_run: int,
    checks: tuple[str, ...],
    failures: tuple[TestFailure, ...],
    warnings: tuple[str, ...] = (),
    allowances: tuple[str, ...] = (),
    allowed_isolated_parts: tuple[str, ...] = (),
    allowed_overlaps: tuple[AllowedOverlap, ...] = (),
)
```

- `passed`: `True` when no blocking failures were recorded.
- `checks_run`: number of recorded checks.
- `checks`: ordered check names.
- `failures`: blocking failures.
- `warnings`: warning messages and failed warning-tier checks.
- `allowances`: human-readable entries recorded by `allow_*`.
- `allowed_isolated_parts`: names of parts explicitly allowed by `allow_isolated_part(...)`.
- `allowed_overlaps`: structured overlap allowances recorded by `allow_overlap(...)`.

## Construction

```python
ctx = TestContext(model, seed=0)
```

- `model`: the `ArticulatedObject` under test.
- `seed`: deterministic sampling seed for pose-sampled checks.

CodingAgent models should end allowances-only `run_tests()` with:

```python
return ctx.report()
```

`compile_model` automatically runs the baseline sanity/QC pass. You do NOT need
to call these from `run_tests()` — they always run:

- model validity (`check_model_valid`)
- exactly one root part
- mesh asset readiness (`check_mesh_assets_ready`)
- floating disconnected-part-group detection (equivalent to
  `fail_if_isolated_parts(...)`)
- disconnected geometry-island detection within a part (equivalent to
  `fail_if_part_contains_disconnected_geometry_islands(...)`)
- current-pose real 3D overlap detection (equivalent to
  `fail_if_parts_overlap_in_current_pose(...)`)

Through-motion penetration is NOT part of the baseline pass. When the blueprint
makes this a binding acceptance requirement, ReviewAgent calls
`fail_if_parts_overlap_in_sampled_poses(...)` from `qa_tests(ctx)`.

ReviewAgent uses `qa_tests(ctx)` for prompt-specific exact assertions such as
`expect_gap(...)`, `expect_overlap(...)`, `expect_contact(...)`, and
`expect_within(...)`.

Keep pose-specific checks lean. Add articulated-pose assertions only when a
prompt-critical mechanism remains ambiguous after exact rest-pose checks.

## High-Signal Testing Habits

- Prefer exact part or element relationship checks over broad heuristics when
  the mounting path, retained insertion, or clearance is what matters.
- If a part has multiple visual regions, use exact contact/support checks for
  the critical mounts instead of broad whole-part approximations.
- Delay brittle numeric thresholds and exact `elem_*` checks until the geometry
  representation is stable.
- Once `qa_tests.py` references a visual by exact `elem_*` name, treat that
  name as a contract. Preserve it or update every dependent check in the same
  edit.
- Do not add broad lower/upper pose sweeps by default. Use one or two decisive
  articulated poses when a prompt-critical mechanism still needs proof.
- When support, floating status, or overlap meaning is ambiguous, probe first,
  then encode a blueprint-grounded lasting invariant in `qa_tests.py`.

## Parameter Conventions

These rules apply across most of the API. Method signatures below are
authoritative when a helper differs.

- `link_*`, `part`, `positive_link`, `negative_link`, `inner_link`, and
  `outer_link` accept either a `Part` object or a part name string. The API uses
  both `link` and `part` terminology; both refer to authored parts.
- `elem_*`, `positive_elem`, `negative_elem`, `inner_elem`, and `outer_elem`
  accept either a named `Visual` object or a visual name string belonging to the
  referenced part.
- `axis` must be `"x"`, `"y"`, or `"z"` and always refers to the positive world
  axis direction.
- `axes` accepts `"x"`, `"y"`, `"z"`, combinations such as `"xy"`, or a sequence
  such as `("x", "y")`.
- Distances and tolerances are meters.
- `name` overrides the recorded check name in the final report.
- `max_pose_samples` controls how many sampled articulation poses are checked.
- `contact_tol=None`, `overlap_tol=None`, and `overlap_volume_tol=None` mean
  “use the SDK default”.

## Reporting Helpers

### `report() -> TestReport`

Returns the final report for the checks recorded so far.

### `check(name: str, ok: bool, details: str = "") -> bool`

Records a custom blocking check.

- `name`: report name.
- `ok`: pass/fail result.
- `details`: stored when `ok` is `False`.

### `fail(name: str, details: str) -> bool`

Convenience wrapper for `check(name, False, details)`.

### `warn(text: str) -> None`

Appends a non-blocking warning string to the report.

## Allowances

### `allow_overlap(link_a, link_b, *, reason, elem_a=None, elem_b=None) -> None`

Records an intentional real 3D interpenetration so overlap checks do not fail on
that case. Do not use this just because one part sits inside another part's
footprint or cavity.

- `reason`: required justification string.
- `elem_a`, `elem_b`: optional element-level scope. If omitted, the allowance
  applies to the full part pair.

### Acceptable Overlap Patterns

`allow_overlap(...)` is a LAST RESORT for the FEW contacts that genuinely cannot be
modeled with clearance — NOT the default tool for connections. If a real connector
exists, BUILD IT and there is nothing to declare:

- A pin/shaft/axle turning in a hole → drill a clearance BORE (the hinge helpers
  already do this; the pin is visual-only). No overlap, no allowance.
- A clevis/fork/trunnion pivot → `PinnedJointSpec` / `PinnedJointGeometry`
  (`52_pinned_joints.md`). No independent pin Part and no allowance.
- A wheel/knob/fan on an axle or shaft → `RotaryConnectorSpec` and the consumer's
  opt-in `rotary=` adapter (`53_rotary_connectors.md`).
- A drawer/basket/tray sliding → use `SlideRail` / `SlideRailGeometry` (`51_slides.md`):
  the runner rides in the channel with a fit clearance. No overlap, no allowance.
- A hinge → `HingePin` / `HingeHalfGeometry` (`49_hinges.md`): interleaved knuckles
  with clearance. No overlap, no allowance.

Reserve `allow_overlap(...)` for the genuinely-unmodelable, and scope it to the specific
named elements (never a broad whole-part allowance):

- Geometric transmission contact that must interpenetrate to transmit: meshing gear
  teeth. Prefer `spur_gear_chain(...)` (real involute teeth at the correct centre
  distance mesh with clearance — no tooth allowance); declare a tooth overlap only if a
  simplified tooth proxy truly cannot be avoided.
- True containment that reads as overlap only against a solid proxy: a bulb in a shade's
  hollow cavity, contents in a bin. (A properly modeled hollow shell has a real cavity,
  so the exact mesh reads no overlap and even this is unnecessary.)
- Seated trim / compliant compression: a bezel, cap, plug, seal, or bumper that needs a
  tiny local embed to read as seated. Pair with `expect_contact(...)` or
  `expect_gap(..., max_penetration=...)` on the seating interface.

A declared `allow_overlap` on a MOVING/jointed pair is audited by Review at mid-range
poses and cross-checked by the connection-cleanliness gate; a runner buried in a wall or
a member run through another's solid body is a geometry DEFECT no reason legitimizes.

### `allow_isolated_part(part, *, reason) -> None`

Records that a named part is allowed to remain isolated in the compiler-owned
floating/disconnected-part-group pass.

If the intentional floating assembly is a multi-part group, allow each authored
part in that group.

### `allow_coplanar_surfaces(link_a, link_b, *, reason, elem_a=None, elem_b=None) -> None`

Records an intentional coplanar-surface relationship for
`warn_if_coplanar_surfaces(...)`.

### Nested Sliders And Telescoping Fits

Build these with the connector, not an allowance. For a drawer/basket/tray/rail,
use `SlideRail` / `SlideRailGeometry` (`51_slides.md`): the runner rides in the channel
with a fit clearance, so there is NO interpenetration and NOTHING to declare — and it
stays clean under full collision (the runner cannot jam in the trough, unlike a
hand-rolled solid-proxy sleeve which the solver convex-hull-fills).

For a multi-member rectilinear slide, use `TelescopingSlideSpec` and
`TelescopingMemberGeometry` (`54_telescoping_slides.md`). Its middle member is one
hollow wall set and each interface validates `home_offset`, travel, and retained
insertion. Independent adjacent joints are checked on a 3x3 combination grid.

The wrong default — building a solid-proxy sleeve, ramming a runner into it, and hiding
the clip with `ctx.allow_overlap("cabinet", "drawer", ...)` — is exactly the
drawer-rammed-into-the-wall defect the connection-cleanliness gate flags. Do not do it.

Prove the fit with exact checks (no allowance needed):

- `expect_within(...)` on the non-motion axes to prove the runner stays centered
- `with ctx.pose({slide_joint: slide_upper})` showing the child actually moves in the
  intended direction and stays retained on the rail (travel ≤ the retained bite so it
  cannot pull fully off)

Round tubes are outside the V1 telescoping helper. Build a true hollow tube collider
and validate its cavity if one is required; a solid proxy plus `allow_overlap` is not
an acceptable substitute.

## Pose and World-Space Queries

### `pose(joint_positions: dict[object, float | Origin] | None = None, **kwargs: float | Origin) -> Iterator[None]`

Temporary pose override context manager.

```python
with ctx.pose({hinge: 0.5}):
    ...

with ctx.pose(hinge=0.5):
    ...
```

- `joint_positions`: mapping of articulation object or joint name to position. Scalar joints use `float`; floating joints use `Origin(...)`.
- `**kwargs`: joint-name shorthand.
- Revolute and continuous positions are radians; prismatic positions are meters. Floating poses use `Origin.xyz` in meters and `Origin.rpy` in radians.
- Positive scalar values follow the configured joint convention: right-hand rule for revolute/continuous, translation along `+axis` for prismatic. Floating articulations do not use `axis`.
- Mimic followers are derived automatically from their source articulation. Pose the source joint, not the mimic follower.
- Restores the previous pose on exit.

When debugging a reversed hinge or slider, compare the closed pose and an
opened/extended pose with `part_world_position(...)`, `part_world_aabb(...)`,
or prompt-specific `expect_*` checks. A good default is to confirm that the
upper-limit pose moves outward/upward in the intended direction, not just that
it avoids overlap.

### `part_world_position(part) -> tuple[float, float, float] | None`

Returns the part origin position in world coordinates for the current pose.

### `link_world_position(link) -> tuple[float, float, float] | None`

Alias for `part_world_position(...)`.

### `part_world_aabb(part) -> tuple[Vec3, Vec3] | None`

Returns the world-space AABB of the full part in the current pose.

### `link_world_aabb(link) -> tuple[Vec3, Vec3] | None`

Alias for `part_world_aabb(...)`.

### `part_element_world_aabb(part, *, elem) -> tuple[Vec3, Vec3] | None`

Returns the world-space AABB of one named visual element on a part.

## Structural Checks

All methods in this section record a named check and return `True` on pass,
`False` on fail.

The first three below ALSO run automatically in the `compile_model` baseline pass
(see the baseline list above) — call them explicitly only to override a tolerance.
The through-motion overlap checks do NOT run in the baseline; call them yourself.

### `fail_if_isolated_parts(*, max_pose_samples=1, contact_tol=None, name=None) -> bool`

Fails when a part group is support-disconnected (floats free of the grounded
body). `contact_tol` defaults to the env/scale-aware contact tolerance; honors
`allow_isolated_part(...)`. Runs in the baseline pass at `max_pose_samples=1`.

### `fail_if_part_contains_disconnected_geometry_islands(*, tol=1e-6, name=None) -> bool`

Fails when a single part's mesh splits into disconnected islands (e.g. a
named-plane extrude bug left orphaned geometry). `warn_if_part_contains_disconnected_geometry_islands(*, tol=1e-6, name=None)`
is the warning-tier variant. Runs in the baseline pass.

### `fail_if_parts_overlap_in_current_pose(*, overlap_tol=None, overlap_volume_tol=None, name=None) -> bool`

Fails on real 3D part interpenetration in the CURRENT (rest) pose. Honors declared
`allow_overlap(...)` allowances. Tolerances default to the env/scale-aware values.
Runs in the baseline pass.

### `fail_if_parts_overlap_in_sampled_poses(*, max_pose_samples=128, overlap_tol=None, overlap_volume_tol=None, ignore_adjacent=False, ignore_fixed=True, name=None) -> bool`

The through-motion penetration assertion: samples up to `max_pose_samples` joint
poses across the articulation ranges and fails if any two parts interpenetrate at
any sampled pose. NOT part of the baseline pass — Review calls it in `qa_tests(ctx)`
to prove parts stay clear as joints move. `ignore_adjacent` skips directly-jointed
neighbors; `ignore_fixed` skips fixed-joint pairs. Honors `allow_overlap(...)`.

### `fail_if_articulation_overlaps(*, max_pose_samples=128, overlap_tol=None, overlap_volume_tol=None, name=None) -> bool`

Like the sampled-poses check but scoped to articulated (moving) pairs, with
adjacency/fixed pairs included in that scope. `warn_if_articulation_overlaps(...)`
is the warning-tier variant.

### `warn_if_overlaps(*, max_pose_samples=128, overlap_tol=None, overlap_volume_tol=None, ignore_adjacent=False, ignore_fixed=True, name=None) -> bool`

Broad warning-tier sampled-pose overlap sensor. Prefer prompt-specific exact
`expect_*` checks first; add this only when a broad overlap sweep answers a
specific uncertainty.

### `fail_if_articulation_origin_far_from_geometry(*, tol=0.015, reason=None, name=None) -> bool`

Fails when an articulation origin is farther than `tol` from nearby geometry.

- `tol`: non-negative absolute tolerance in meters.
- `reason`: optional note recorded as a warning when using a relaxed tolerance.

Available, but not part of the recommended default stack because the tolerance
is absolute rather than scale-aware.

### `warn_if_articulation_origin_far_from_geometry(*, tol=0.015, reason=None, name=None) -> bool`

Warning-tier version of the same check.

### `warn_if_coplanar_surfaces(*, max_pose_samples=32, plane_tol=0.001, min_overlap=0.02, min_overlap_ratio=0.35, ignore_adjacent=True, ignore_fixed=True, name=None) -> bool`

Warning-tier heuristic for suspicious coplanar or nearly coplanar surfaces.

- `plane_tol`: maximum plane separation.
- `min_overlap`: minimum in-plane overlap distance on both in-plane axes.
- `min_overlap_ratio`: minimum overlap area ratio, in `[0, 1]`.
- `ignore_adjacent`, `ignore_fixed`: same meaning as the sampled overlap checks.

Use only when this specific heuristic answers a real uncertainty.

## Exact Assertions

All methods in this section record a named check and return `True` on pass,
`False` on fail.

### `expect_origin_distance(link_a, link_b, *, axes="xy", min_dist=0.0, max_dist=None, name=None) -> bool`

Checks the distance between part origins along the requested axes.

- `min_dist`: inclusive lower bound.
- `max_dist`: optional inclusive upper bound.

### `expect_origin_gap(positive_link, negative_link, *, axis, min_gap=0.0, max_gap=None, name=None) -> bool`

Checks the signed origin-to-origin gap along one positive world axis.

- `positive_link`: object expected on the positive side of the axis.
- `negative_link`: object expected on the negative side of the axis.
- `min_gap`: inclusive lower bound.
- `max_gap`: optional inclusive upper bound.

### `expect_contact(link_a, link_b, *, contact_tol=1e-6, elem_a=None, elem_b=None, name=None) -> bool`

Checks exact minimum distance between two parts or two named elements.

- `contact_tol`: maximum allowed separation to still count as contact.
- `elem_a`, `elem_b`: optional named element scope.

### `expect_gap(positive_link, negative_link, *, axis, min_gap=None, max_gap=None, max_penetration=None, positive_elem=None, negative_elem=None, elem_a=None, elem_b=None, name=None) -> bool`

Checks signed exact-geometry gap along a positive world axis.

- The measured gap is `positive.min[axis] - negative.max[axis]`.
- `min_gap`: inclusive lower bound. If omitted, it is derived from
  `max_penetration`.
- `max_gap`: optional inclusive upper bound.
- `max_penetration`: convenience way to express the allowed overlap depth;
  equivalent to a lower bound of `-max_penetration`.
- `positive_elem`, `negative_elem`: optional named element scope.
- `elem_a`, `elem_b`: accepted as compatibility aliases for
  `positive_elem`/`negative_elem`.

This is the main exact clearance and seating helper.

### `expect_gate_witness_clear(link_a, link_b, *, witness_pose, expected_policy_id, expected_penetration_depth_tol_m, expected_collision_mode, name=None) -> bool`

Replays an access-gate witness using the active full-collision FCL/EPA policy. It
checks the deepest contact across the entire target part pair and passes only when
that depth is strictly below the active threshold (default `0.0002 m`).

- `witness_pose`: the complete gate-open pose from `gate_evaluation.pose`.
- `expected_policy_id`, `expected_penetration_depth_tol_m`, and
  `expected_collision_mode`: provenance copied from the handoff. They must match
  the active policy and cannot override or loosen it. A mismatch raises
  `ValidationError` as a tester/configuration error; it is not a model failure.

Use this numeric assertion before rendering a `[gated-precondition]` witness. A
passing assertion proves collision clearance; rendering only judges whether the
gate-open pose is a sensible real operating configuration.

### `expect_overlap(link_a, link_b, *, axes="xy", min_overlap=0.0, elem_a=None, elem_b=None, name=None) -> bool`

Checks exact projected overlap between two parts or named elements. This is a
footprint/projection check, not a collision/contact check.

- `min_overlap`: required overlap on every requested axis.
- `elem_a`, `elem_b`: optional named element scope.

### `expect_within(inner_link, outer_link, *, axes="xy", margin=0.0, inner_elem=None, outer_elem=None, elem_a=None, elem_b=None, name=None) -> bool`

Checks that one part or named element stays within another on the requested
axes.

- `margin`: allowed slack outside the outer bounds.
- `inner_elem`, `outer_elem`: optional named element scope.
- `elem_a`, `elem_b`: accepted as compatibility aliases for
  `inner_elem`/`outer_elem`.

For nested sliders, use `expect_within(...)` on the non-motion axes and pair it
with `expect_overlap(...)` or `expect_gap(...)` on the slide axis. Do not use
`expect_within(...)` by itself as proof that the moving member still remains
inserted at full extension.

## Coupled-Motion Assertion

### `expect_coupled_motion(follower, driver, *, sense="same", min_ratio=0.0, q0=None, q1=None, name=None) -> bool`

Verifies a declared coupling: driving `driver` from `q0` to `q1` must move
the mimic-driven `follower` in the expected `sense`. Fails when the follower does
not move as its driver moves — exactly the "should be coupled but isn't" defect.
This is the right assertion for meshing gear trains and any driver/follower
linkage (rack-and-pinion, crank-slider, geared shafts).

- `follower`, `driver`: the two coupled articulations (a `Joint`/articulation
  object or its name). Drive the DRIVER; never pose the mimic-driven follower
  directly.
- `sense`: `"same"` or `"opposite"` — the expected relative motion direction.
- `min_ratio`: inclusive lower bound on `|Δfollower / Δdriver|`, expressed in
  FOLLOWER units per DRIVER unit (metres-per-radian for a rotary→linear drive,
  dimensionless for rotary→rotary). Leave at `0.0` to assert only
  direction-and-motion; the exact ratio is already proven by the build-time
  coupling-conformance gate, so QA need only confirm the follower follows.
- `q0`, `q1`: optional explicit driver samples; default to the driver's limits
  (continuous → `0..pi/2`).

The follower must be wired to this driver (`model.couple(driver, follower, ...)`);
an unwired follower, a wrong mimic source, or a mimic-driven driver fails the
check. This confirms only that the follower moves in the declared direction, not
that the visible transmission is physically coherent.

## Floating pose notes

- `FLOATING` is outside the three-agent Pipeline authoring contract. These notes
  describe the lower-level SDK API only.
- `ctx.pose(...)` accepts mixed values: scalar `float` for scalar articulations and `Origin(...)` for `FLOATING`. Passing the wrong value kind is a validation error.
- For floating articulations, `Origin.xyz` is relative translation in the articulation frame and `Origin.rpy` is relative rotation in that same frame. Zero floating pose is `Origin()`.
- Mimic followers cannot be overridden directly in `ctx.pose(...)`; drive the source articulation instead.
- `sample_poses(...)` and pose-sampled QC use only `Origin()` for floating joints by default. To test additional floating poses, provide `joint.meta["qc_samples"] = [Origin(...), ...]`.
