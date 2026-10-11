# `model.py` contract (SDK carrier)

You author **one** Python file, `model.py`, that is the source of truth for a
single articulated 3D object. It is compiled by Artisan's own compiler
(`compile_model`), which exports URDF + meshes and runs the baseline QC gate.

## SDK reference library (read on demand)
This contract is your entrypoint. The full SDK reference is readable via
`read_file("<name>")` — read the exact page before guessing a helper's name or
signature. Available pages:

- `20_core_types.md` — geometry / material / articulation / inertial core types.
- `30_articulated_object.md` — object/part/articulation authoring + lookups.
- `50_placement.md` — `AnchorSelector`, `attach`, `gap`, `center_between`, `snap`,
  Mesh/CadQuery Anchor sources, and compatibility mounting helpers. Use
  `read_file("50_placement")` before authoring a relation instead of guessing.
- `80_testing.md` — Coding allowances plus the full `TestContext` API used by Review.
- `10_errors.md` — common compile/authoring failures and how to read them.
- `70_probe_tooling.md` — `probe_model` inspection catalog.
- CadQuery: `35_cadquery.md` (overview), `36_cadquery_primer.md`,
  `37_cadquery_workplane.md`, `38_cadquery_sketch.md`, `39_cadquery_assembly.md`,
  `39c_cadquery_api_ref.md`, `39d_cadquery_gears.md`.
- Geometry families (read the matching one before the low-level mesh page):
  `40_mesh_geometry.md`, `41_panels_and_grilles.md`, `42_brackets_and_mounts.md`,
  `43_fans_and_rotors.md`, `44_knobs_and_controls.md`, `45_wires.md`,
  `46_section_lofts.md`, `47_bezels_and_frames.md`, `48_wheels_and_tires.md`,
  `49_hinges.md`.
- `50_blender_freeform.md` — author genuinely organic/soft parts (upholstered
  cushions, irregular outlines, draped covers) in headless Blender via
  `run_blender` + `mesh_from_input`, when primitives/lofts cannot express the
  shape. Read this before building any soft/organic piece.

Plus ~90 worked examples: `find_examples(query, joint_type=...)` searches the
full `sdk/_examples/` corpus, and you can `read_file("<example>.md")` by name.

## Required top-level symbols

```python
from __future__ import annotations

from sdk import (
    AnchorSelector,
    ArticulatedObject,
    ArticulationType,
    Box,
    Cylinder,
    FrameRef,
    HingeHalfGeometry,
    HingePin,
    MotionLimits,
    Origin,
    PlacementTarget,
    RigidFrame,
    TestContext,
    TestReport,
    anchor,
    attach,
    explicit_anchor,
    mesh_from_geometry,
    semantic_anchor,
)


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="wall_cabinet")

    CARCASS_WIDTH = 0.60
    CARCASS_DEPTH = 0.30
    CARCASS_HEIGHT = 0.80
    CAP_OVERHANG = 0.01
    DOOR_EDGE_GAP = 0.01
    DOOR_THICKNESS = 0.02
    CARCASS_SIZE = (CARCASS_WIDTH, CARCASS_DEPTH, CARCASS_HEIGHT)
    TOP_CAP_SIZE = (
        CARCASS_WIDTH + 2.0 * CAP_OVERHANG,
        CARCASS_DEPTH + 2.0 * CAP_OVERHANG,
        0.008,
    )
    DOOR_SIZE = (
        CARCASS_WIDTH - 2.0 * DOOR_EDGE_GAP,
        DOOR_THICKNESS,
        CARCASS_HEIGHT - 2.0 * DOOR_EDGE_GAP,
    )

    # materials: semantic name + rgba + a material_class so the renderer shows
    # the real finish (metal looks metallic, glass is see-through). Pick the class
    # from the validated set listed in your system prompt (Quality bars) — glass
    # uses material_class="glass" (or transmission=). Material is per-visual:
    # multiple finishes on one part = multiple visual()s.
    case_finish = model.material("case_finish", rgba=(0.92, 0.93, 0.95, 1.0), material_class="painted_metal")
    trim_finish = model.material("trim_finish", rgba=(0.16, 0.17, 0.19, 1.0), material_class="anodized_metal")

    # ---- Pass 1 — FABRICATE: each part's own geometry, in its own frame. ----
    body = model.part("cabinet_body", est_mass_kg=16.0)
    carcass = body.visual(Box(CARCASS_SIZE), material=case_finish, name="carcass")
    # A plain internal offset: the shelf's height inside its own carcass answers to
    # nothing outside this part, so a literal Origin is exactly the right tool.
    body.visual(
        Box((0.57, 0.28, 0.016)),
        origin=Origin(xyz=(0.0, 0.0, 0.10)),
        material=case_finish,
        name="shelf",
    )

    # ---- Pass 2 — ASSEMBLE: every pose that depends on another element. ----
    # That includes two visuals inside one part, as all three seats below are.
    # Writing the number such a relation would produce is a contract violation, so
    # read_file("50_placement") before you author your first one.
    frame = FrameRef.part_local(body)  # the OWNING PART, not either Visual

    # SEATED — the cap rides on the carcass top face. Mating faces meet back-to-back,
    # which is what axis_map says. No z is typed, so a taller carcass carries it up.
    top_cap = body.visual(Box(TOP_CAP_SIZE), material=trim_finish, name="top_cap")
    model.bind_placement(
        PlacementTarget.visual(top_cap),
        attach(
            anchor(top_cap, AnchorSelector.face("-z"), frame=frame),
            anchor(carcass, AnchorSelector.face("+z"), frame=frame),
            axis_map={"+z": "-z", "+x": "+x"},
        ),
    )

    # FLUSH — the fascia's outer face is coplanar with the cabinet front: normals
    # facing the SAME way, clearance 0. uv_offset walks the host's own face plane.
    fascia = body.visual(Box((0.16, 0.012, 0.06)), material=trim_finish, name="control_fascia")
    model.bind_placement(
        PlacementTarget.visual(fascia),
        attach(
            anchor(fascia, AnchorSelector.face("+y"), frame=frame),
            anchor(carcass, AnchorSelector.face("+y"), frame=frame),
            axis_map={"+z": "+z", "+x": "+x"},
            uv_offset=(0.0, -0.30),
            clearance=0.0,
        ),
    )

    # PROUD — the same call with a real stand-off: the button's face stands 4 mm off
    # the fascia and its shank recesses back through the opening. Its host is itself
    # a placed element, so relations chain: move the fascia and the button follows.
    button = body.visual(Cylinder(radius=0.010, length=0.028), material=trim_finish, name="latch_button")
    model.bind_placement(
        PlacementTarget.visual(button),
        attach(
            anchor(button, AnchorSelector.face("+z"), frame=frame),
            anchor(fascia, AnchorSelector.face("+y"), frame=frame),
            axis_map={"+z": "+z", "+x": "+x"},
            uv_offset=(0.05, 0.0),
            clearance=0.004,
        ),
    )

    # A JOINT LINE IS A HOST FEATURE TOO — the hinge runs along the carcass's front
    # vertical edge. Query that edge, seed ONE shared HingePin from it, and let the
    # pin supply both the axis and, through bind_placement, the joint origin. An edge
    # Anchor's +z IS the edge line; taking -z here is the open-DIRECTION decision
    # (positive q swings the door out toward +Y) — confirm the sign with a render.
    joint_frame = FrameRef.parent_link_local(body)
    hinge_edge = anchor(carcass, AnchorSelector.edge("+x", "+y"), frame=joint_frame)
    pin = HingePin.from_edge(
        hinge_edge,
        axis=hinge_edge.pose.axis("-z"),
        length=0.06,
        knuckle_count=5,
        pin_diameter=0.006,
        knuckle_diameter=0.010,
        clearance=3e-4,
    )
    # This ordinary hinge is a short physical connector, not a full-height barrel.
    # The carcass reaches the pin line, so no separate leaf is needed: the parent's
    # knuckles fuse directly into the carcass at its front edge.
    body.visual(
        mesh_from_geometry(
            HingeHalfGeometry(pin, role="outer", include_pin=True),
            "cabinet_hinge_outer",
        ),
        origin=Origin(xyz=pin.anchor),
        material=trim_finish,
        name="cabinet_hinge_outer",
    )

    door = model.part("door", est_mass_kg=4.5)
    # Fabrication again: both terms are the panel's OWN half-extents, so it hangs
    # off its own hinge frame. Its knuckles ride the pin — read_file("49_hinges").
    door.visual(
        Box(DOOR_SIZE),
        origin=Origin(xyz=(-DOOR_SIZE[0] / 2.0, DOOR_SIZE[1] / 2.0, 0.0)),
        material=trim_finish,
        name="door_panel",
    )
    # The door panel also reaches the pin line, so its compact complementary half
    # needs no leaf. The child frame is placed on the pin by the joint binding below.
    door.visual(
        mesh_from_geometry(HingeHalfGeometry(pin, role="inner"), "door_hinge_inner"),
        material=trim_finish,
        name="door_hinge_inner",
    )

    # articulation: parent -> child, UNIT axis (child-local — identical here, since
    # the bound joint frame is unrotated), ORDERED limits (lower < upper)
    model.articulation(
        "body_to_door",
        ArticulationType.REVOLUTE,
        parent="cabinet_body",
        child="door",
        axis=pin.axis,
        motion_limits=MotionLimits(lower=0.0, upper=1.92, effort=10.0, velocity=1.5),
    )
    model.bind_placement(
        PlacementTarget.joint(model.get_joint("body_to_door")),
        # A Part's own frame origin is the subject of a joint placement.
        attach(
            explicit_anchor(door, RigidFrame(), frame=joint_frame, kind="point"),
            semantic_anchor(pin, "pin_axis", frame=joint_frame),
        ),
    )
    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    return TestContext(object_model).report()
```

`object_model` (top-level) and `build_object_model()` and `run_tests()` are all
required. `run_tests()` must return `ctx.report()`.

## Articulation types
Pipeline models may author only `ArticulationType.REVOLUTE`, `PRISMATIC`,
`CONTINUOUS`, and `FIXED`.
- REVOLUTE/PRISMATIC: require `MotionLimits(lower=..., upper=...)` with lower < upper.
- CONTINUOUS: no finite limits.
- axis must be **unit length** and expressed in the **child-local frame**
  (the #1 failure mode is an un-normalized or wrong-frame axis).

## Geometry
- Prefer SDK primitives for genuinely SOLID parts (a knob, a pin, a flat panel,
  a strut): `Box((x,y,z))`, `Cylinder(radius=, length=)`, `Sphere(radius=)`.
- For complex shapes use the higher-level generators or cadquery via
  `mesh_from_cadquery(...)`. Find archetypes with `find_examples(query, joint_type=...)`.
- `est_mass_kg=<kg>` is REQUIRED on every `model.part(...)` — the compile gate fails
  a part without it. It is the physics-export mass source (enrich scales inertia from
  geometry). Estimate the REAL mass from size + material + hollow-vs-solid; the
  `<mass_audit>` block after `compile_model` reports each part's implied solidity for a
  self-check (Quality bar 1g). `Inertial` itself stays OPTIONAL — omit it; mass comes
  from `est_mass_kg`.

### CRITICAL: shells / enclosures / containers MUST be hollow
A part that, in reality, is a hollow shell — a **lampshade**, a cabinet/box
carcass, a casing, a cup/bowl, a dome, a housing — must be modeled HOLLOW
(thin-walled, with the real opening), NOT as a stack of solid primitives.
Stacking solid `Cylinder`/`Box` for a container is wrong: it looks like a solid
drum, hides whatever belongs inside, and creates spurious overlaps with the
contents. Build hollow via (pick what fits, prefer cadquery for shells):

- **cadquery `.shell()`** — the canonical way. Make the solid, open one face,
  shell to a wall thickness, e.g. a lampshade:
  ```python
  import cadquery as cq
  from sdk import mesh_from_cadquery
  shade_solid = cq.Workplane("XY").circle(0.075).workplane(offset=0.11).circle(0.045).loft()
  shade = shade_solid.faces("<Z or >Z").shell(-0.003)  # open bottom, 3mm wall
  shade_mesh = mesh_from_cadquery(shade)
  ```
- **`boolean_difference(outer, inner)`** — subtract a slightly smaller inner
  shape to leave thin walls + a cavity.
- **`LatheGeometry`** with a thin two-sided profile — natural for a conical
  shade (surface of revolution).
- **`cut_opening_on_face(shell, face=..., opening_profile=..., depth=...)`** —
  add an opening throat to an already shell-like mesh.

The thing that belongs inside (a **bulb** in a shade, items in a cabinet) is a
separate part placed in the **cavity** — with the shade hollow it will not
overlap the walls. Study `find_examples("enclosure")` / `find_examples("shell")`
(a_parametric_enclosure, shelling_to_create_thin_features, the_classic_occ_bottle,
lego_brick) before building any container/shade.

## What the compile gate (deterministic, all objects) checks
single root + acyclic tree, unique part/articulation names, axis unit +
child-local, ordered limits, mesh assets present, no isolated/floating parts,
no rest-pose overlap, geometry scale sanity, and your `run_tests()`.

## Scope of `run_tests()` (allowances only, NOT where you write tests)
`run_tests()` is a REQUIRED top-level symbol (the compiler errors without it and
reads your allowances out of it). In this three-agent Pipeline its job is to
DECLARE intended exceptions to the baseline QC — `allow_overlap(...)` /
`allow_isolated_part(...)` (see below). With nothing to declare it can be just
`return TestContext(object_model).report()`. Do NOT encode the object's
functional spec here: object-specific functional acceptance tests (e.g. "a
fridge door opens to ≤ ~120°", "a drawer cannot be pulled fully out") are
authored independently by ReviewAgent in `qa_tests.py::qa_tests(ctx)`. Do not
weaken your geometry to pre-satisfy them.

## CRITICAL: declare INTENTIONAL contacts/nesting as allowances
The baseline QC fails on ANY rest-pose overlap and on any part not in contact
with the grounded body. Real assemblies legitimately have **intended** contacts
and nesting — a bulb sits INSIDE a shade, a shade seats ON the neck, a spring
touches the arm, two flush panels meet. If you try to physically *remove* these
overlaps you will fight the design forever and burn turns. Instead **declare
them** in `run_tests()` so the baseline skips them:

```python
def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    # intentional nesting/contact — these SHOULD overlap/touch:
    ctx.allow_overlap("bulb", "lamp_shade", reason="bulb sits inside the shade")
    ctx.allow_overlap("lamp_shade", "head_neck", reason="shade seats on the neck")
    ctx.allow_isolated_part("decorative_spring", reason="rests on arm, no joint")
    return ctx.report()
```

Rule of thumb: if `compile_model` reports `QC_REAL_OVERLAP` or `QC_ISOLATED_PART`
for a pair/part that is **supposed** to touch/nest by design, add an
`allow_overlap(...)` / `allow_isolated_part(...)` with a clear reason — do NOT
delete or move the geometry. Only fix overlaps that are genuine mistakes
(parts colliding that should be apart).

But allow_overlap is for CONTAINMENT/seating (a bulb in a cavity, a cap seated),
NOT for CONNECTIONS. A moving connection should ride with CLEARANCE, so there is
nothing to declare: build a pin/shaft with a drilled BORE (the `HingePin` helpers),
a drawer/basket/tray with `SlideRail`/`SlideRailGeometry` (a runner in a channel with
a fit gap — see `51_slides.md`), a hinge with `HingePin`/`HingeHalfGeometry`. Declaring
`allow_overlap("basket", "cabinet", ...)` to hide a runner clipped into a wall is the
drawer-rammed-into-the-wall defect the motion self-check flags as a RAM — build the
clearanced connector instead. Reserve overlap declarations for the genuinely-unmodelable
that MUST interpenetrate (meshing gear teeth) plus true containment.

Recommended (not required): when the intended contact is between two SPECIFIC
named features, scope the allowance to those ELEMENTS rather than the whole link
pair. `allow_overlap` defaults to the link-pair level but also takes
`elem_a`/`elem_b` (the `name=` you gave the touching `part.visual(...)`):

```python
ctx.allow_overlap("upper_arm", "shade_head", elem_a="hinge_eye", elem_b="trunnion_boss",
                  reason="distal arm hinge eye engages the shade trunnion at the head pivot")
```

Why prefer it for a JOINTED/moving pair: a whole-link allowance ALSO silences any
OTHER overlap between those two links, so if the shade BODY later swings into the
arm at some mid-travel angle, a link-level blanket hides that real penetration —
an element-scoped allowance keeps the rest of the pair honest and still catches
it. A whole-link allowance is fine when the parts genuinely only ever meet at one
place; for a pair that moves, scoping to the touching elements is the safer call.

Access gates are NOT yours to author. When the blueprint declares a part behind a
cover (a drawer behind a door) via a part's `motion_gate`, the motion screen
respects it automatically — it checks that gated joint's travel with the gate held
OPEN and treats a door-closed-vs-drawer-out contact as an unreachable-pose phantom,
not a defect. You do NOT write gates anywhere in `model.py`, and you must NOT clamp
a gated joint's `MotionLimits` or distort geometry to dodge such a phantom.

## Iterate
`write_file` / `replace` → `probe_model` (inspect structure) → `compile_model`
(fix every BLOCKING signal) → stop only on COMPILE SUCCESS with no later edits.

When compile feedback reports suppressed blockers, read
`reports/compile_findings.json` for the complete blocker and placement evidence.
Read complete exact-candidate warning strings only from
`reports/physics_validation.json`. In either case, compare the report's
`model_sha256` with the current `model.py` before relying on it.
