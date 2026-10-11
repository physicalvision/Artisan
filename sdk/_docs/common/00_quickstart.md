# SDK Quickstart

## Purpose

Use this page to start a new Artisan SDK script. It defines the required
script contract, the authoring workspace rules, and one minimal end-to-end
example.
Read the reference pages listed below with `read_file("<filename>")`.

## Authoring Workspace

- CodingAgent authors `model.py` and supporting geometry inputs.
- `model_contract.md` defines the complete pipeline authoring contract.
- SDK reference documents and examples are read-only resources.
- ReviewAgent independently authors functional checks in `qa_tests.py`.
- Import from `sdk` in `model.py`.
- Use `read_file(path=...)` to load exact reference text only when needed. For a
  focused source region, pass a 1-based `offset` line and a `limit` line count;
  omit both for the full file.

## Import Contract

Authoring helpers are exposed as top-level public imports from `sdk`.
The reference docs are grouped by topic for reading convenience; those filenames
do not imply matching Python submodules.

```python
# Correct
from sdk import ArticulatedObject, MotionLimits, place_on_face

# Wrong
from sdk.placement import place_on_face
from sdk.core_types import MotionLimits
```

## Reference Library

Read these pages by filename:

- `00_quickstart.md`: script contract, workspace rules, minimal example, workflow, and the
  full reference inventory.
- `10_errors.md`: common compile and authoring failures, plus how to interpret them.
- `20_core_types.md`: geometry, material, articulation, and test-related core
  types, plus optional inertial helpers.
- `30_articulated_object.md`: object, part, and articulation authoring helpers and lookup
  patterns.
- `40_assets.md`: explicit asset-root helpers for standalone scripts and tests.
- `50_placement.md`: placement helpers for mounting, offsets, wrapping, and alignment.
- `70_probe_tooling.md`: `probe_model` helper catalog and inspection workflow.
- `80_testing.md`: `TestContext`, `expect_*` assertions, and test authoring patterns.
- `35_cadquery.md`: when and why to use CadQuery-style geometry in Artisan.
- `36_cadquery_primer.md`: CadQuery mental model and core shape-building workflow.
- `37_cadquery_workplane.md`: workplane-based modeling patterns and common operations.
- `38_cadquery_sketch.md`: sketch-driven 2D profiles and profile construction tools.
- `39_cadquery_assembly.md`: CadQuery assembly helpers and composition patterns.
- `39d_cadquery_gears.md`: vendored gear builders and the preserved `Workplane.gear()`
  plugin workflow.
- `39b_cadquery_free_function.md`: free-function geometry helpers and utility builders.
- `39c_cadquery_api_ref.md`: compact CadQuery API reference and signatures.

Additional geometry references:

- `40_mesh_geometry.md`: mesh generation flow, managed meshes, and mesh-based
  low-level geometry helpers.
- `41_panels_and_grilles.md`: perforated panels, slotted panels, and full vent
  grilles.
- `42_brackets_and_mounts.md`: clevises, forks, and yoke-style support members.
- `43_fans_and_rotors.md`: axial fan rotors and blower wheels.
- `44_knobs_and_controls.md`: knobs, dial caps, grip details, and shaft bores.
- `45_wires.md`: wire and path construction helpers.
- `46_section_lofts.md`: section lofts, repairs, and section-driven geometry.
- `47_bezels_and_frames.md`: bezels, framed openings, recesses, and trim
  surrounds.
- `48_wheels_and_tires.md`: wheel structure, tire carcasses, tread, and sidewalls.
- `49_hinges.md`: working hinges — interleaved knuckles on a shared pin
  (`HingePin` + `HingeHalfGeometry`) that keep a door/lid attached through its swing.
- `51_slides.md`: captured two-part drawer and tray rails.
- `52_pinned_joints.md`: functional clevis, pivot-fork, and trunnion connectors.
- `53_rotary_connectors.md`: free-axle versus rigid-shaft rotary topology and
  Wheel/Knob/Fan adapters.
- `54_telescoping_slides.md`: hollow multi-member rectangular telescoping slides.

If a prompt clearly names a semantic part family such as a knob, bezel, wheel,
tire, vent grille, bracket, hinge, pinned joint, bearing, or telescoping slide, read that focused geometry page before
falling back to the low-level mesh page.

Read the exact document you need. Do not guess helper names or signatures from memory.

## Script Contract

Every generated script should define:

- `build_object_model() -> ArticulatedObject`
- `run_tests() -> TestReport`
- `object_model = build_object_model()`

`compile_model` compiles `object_model`, derives exact collisions from visuals,
runs tests, and exports the result. Do not emit URDF XML directly.

`compile_model` also owns the baseline sanity/QC pass. It automatically checks
model validity, exactly one root part, mesh assets, floating disconnected part
groups, disconnected geometry islands inside a part, and current-pose real 3D
overlaps.

## Recommended Imports

```python
from sdk import (
    ArticulatedObject,
    ArticulationType,
    Box,
    MotionLimits,
    Origin,
    TestContext,
    TestReport,
)
```

## Managed Mesh Pattern

Use logical mesh names. Artisan manages the materialized OBJ asset paths.

- Generate procedural meshes with `mesh_from_geometry(..., "part_name")`.
- Import existing OBJ inputs with `mesh_from_input("mesh_name")`.
- Use `TestContext(object_model)`; do not wire asset roots manually.
- If you are writing a standalone local script and need a stable asset root,
  read `40_assets.md`.

## Minimal Example

```python
from sdk import (
    AnchorSelector,
    ArticulatedObject,
    ArticulationType,
    Box,
    FrameRef,
    MotionLimits,
    Origin,
    PlacementTarget,
    RigidFrame,
    TestContext,
    TestReport,
    anchor,
    attach,
    explicit_anchor,
)


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="example_box_lid")

    base = model.part("base")
    base_shell = base.visual(
        Box((0.20, 0.20, 0.05)),
        origin=Origin(xyz=(0.0, 0.0, 0.025)),
        name="base_shell",
    )

    lid = model.part("lid")
    lid.visual(
        Box((0.18, 0.18, 0.02)),
        # The lid part frame sits on the hinge line; the panel extends along +X.
        # This is the lid's OWN internal offset, so it stays a hand-written Origin.
        origin=Origin(xyz=(0.09, 0.0, 0.01)),
        name="lid_shell",
    )

    model.articulation(
        "base_to_lid",
        ArticulationType.REVOLUTE,
        parent=base,
        child=lid,
        # Positive q should open the lid upward, not into the base.
        # Because the closed lid extends along local +X from the hinge,
        # choose -Y so positive rotation lifts the free edge toward +Z.
        axis=(0.0, -1.0, 0.0),
        motion_limits=MotionLimits(effort=5.0, velocity=3.0, lower=0.0, upper=1.2),
    )

    # The hinge line DEPENDS on the base: it lies on the base's top face, 5% in from
    # the -X edge. Seed it from that face instead of typing (-0.09, 0.0, 0.05), so it
    # follows the base when the base is resized.
    hinge_frame = FrameRef.parent_link_local(base)
    model.bind_placement(
        PlacementTarget.joint(model.get_joint("base_to_lid")),
        attach(
            # A Part's own frame origin is the subject of a joint placement.
            explicit_anchor(lid, RigidFrame(), frame=hinge_frame, kind="point"),
            anchor(base_shell, AnchorSelector.face("+z", uv=(0.05, 0.5)), frame=hinge_frame),
        ),
    )

    return model


def run_tests() -> TestReport:
    # CodingAgent declares only intentional baseline-QC allowances here.
    return TestContext(object_model).report()


object_model = build_object_model()
```

This example uses a hinge-line part frame for the lid. At `q=0`, the child
frame coincides with the articulation frame at the left edge of the lid. Since
the closed lid panel extends along local `+X` from that hinge, `axis=(0, -1, 0)`
makes positive angles open upward.

## Recommended Workflow

1. Build parts with `model.part(...)`.
2. Add visuals with `part.visual(...)`.
3. Add motion with `model.articulation(...)`.
4. Keep `run_tests()` limited to intentional baseline-QC allowances. ReviewAgent
   owns prompt-specific acceptance assertions in `qa_tests.py::qa_tests(ctx)`.
5. Use `allow_overlap(...)` and `allow_isolated_part(...)` only when the
   intended mechanism genuinely requires those exceptions — build a moving
   connection to ride with clearance instead (a hinge with `HingePin`, a
   drawer/slide with `SlideRail`/`SlideRailGeometry`), so there is nothing to
   declare. Reserve overlap declarations for meshing gears and true containment.

Set `est_mass_kg=<kg>` on every `model.part(...)` — it is the physics-export
mass source (enrich scales the inertia tensor from geometry, so mass magnitude
comes from your estimate, not from a solid-density guess). `part.inertial` stays
optional: omit it and let mass flow from `est_mass_kg`.

## Authoring Habits

- Model visible openings, cavities, and hollow bodies explicitly. Do not cap a
  visible opening with a solid placeholder.
- A CadQuery boolean can return without an exception even when a complex
  cutter degenerates. For critical openings or cavities built from lofted,
  thin-wall, tangent, or compound geometry, prefer sequential `body.cut(...)`
  calls over `body.cut(a.union(b).union(c))`; see `37_cadquery_workplane.md`.
- If the object is layered or nested, model those layers with clear visual
  separation instead of collapsing them into one solid mass.
- Preserve the visible construction logic of major faces and covers. If a
  visible surface should read as one continuous piece, keep it connected and
  cut openings into it rather than replacing it with floating fragments.
- Hidden supports and internal structure can stay simple as long as the visible
  form reads correctly.
- Within one part, avoid disconnected visual islands. If a feature should read
  as mounted, give it a real rib, bracket, pin, collar, stem, or wall
  connection, or split it into a separate part.
- Use `attach(...)` with structured `AnchorSelector` face/edge/corner Anchors
  when a mounted child pose should follow neighbouring geometry;
  use `gap(...)`, `center_between(...)`, or `snap(...)` for spans and one-line
  placement. Read `50_placement.md` before guessing signatures.
- Apply pure results with `model.bind_placement(PlacementTarget.visual(...), result)`
  or the matching joint target. Existing `place_on_surface(...)` and
  `place_on_face(_uv)` remain useful compatibility shortcuts for simple mounts.
- An explicit design-intent `Origin(...)` is valid when no unique feature exists;
  do not turn `no-feature`/`ambiguity` into a guessed Mesh feature.
- Use restrained real-world materials and colors rather than placeholder
  defaults.

## Reference Routing

- If you need type or helper signatures, read `20_core_types.md`.
- If you need part/object construction patterns, read
  `30_articulated_object.md`.
- If you need explicit asset-root control, read `40_assets.md`.
- If you need placement logic, read `50_placement.md`.
- If you need compile/debug interpretation, read `10_errors.md`.
- If you need probe helper details, read `70_probe_tooling.md`.
- If you need testing details, read `80_testing.md`.
- If you need lower-level CadQuery geometry, read the relevant
  CadQuery reference document.
- If you need semantic geometry families, read the relevant
  geometry-family reference family page first.
- If you need low-level mesh, wire, or loft helpers, read the relevant
  geometry-family reference document.
