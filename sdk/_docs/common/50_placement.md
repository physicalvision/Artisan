# Relational Placement

## Purpose

Use relational placement when a pose should follow another element's face,
edge, surface, axis, or span. The API evaluates deterministic geometry at model
build time and returns an absolute pose; it is not a runtime constraint solver.

Prefer a relation that expresses the real design intent. A hand-written
`Origin(...)` remains first-class when the intended coordinate is explicit or no
unique feature exists. Do not replace a clear design number with an ambiguous
query.

The most common case is **two visuals inside one part** — a display seated on a
fascia, a button proud of a panel, a grip flush with a roof. That is not a
special case of assembly, it is the bulk of it. Read
[Seating One Visual Against Another In The Same Part](#seating-one-visual-against-another-in-the-same-part)
first; it also documents the one frame argument that decides whether the call
works at all.

## Core Imports

```python
from sdk import (
    AnchorSelector,
    FrameRef,
    PlacementTarget,
    RigidFrame,
    anchor,
    attach,
    center_between,
    explicit_anchor,
    gap,
    snap,
)
```

All distances are meters. `RigidFrame.basis` stores orthonormal `(+x, +y, +z)`
axes as columns. Euler angles appear only at the `Origin` boundary.

## Mental Model

- `FrameRef` names the coordinate system that will consume the result.
- `Anchor` is a point, plane, or axis expressed in that frame.
- `AnchorSource.source_kind` records which verified feature or aggregate selector
  produced it; the independent `owner_kind` records whether its owner is a Visual,
  Part, joint, supported standalone object, or unregistered object.
- `PlacementResult` is the subject's absolute candidate pose plus its immutable,
  ordered solve-operation provenance in that frame.
- `PlacementTarget` applies the result to one mutable model field.
- Placement operations are pure: they do not mutate parts or run global
  collision checks.

Every operation accepts an explicit `from_placement=` when it follows an earlier
operation. An unchained call starts from the subject's authored pose and is
independent. Later operations write only their controlled subspace: for example,
`center_between` writes one translation line and preserves orthogonal translation
and all rotation from the previous result.

## Frames And Binding

Use the frame belonging to the field that receives the pose:

```python
part_frame = FrameRef.part_local(part)                # Visual.origin
joint_frame = FrameRef.parent_link_local(parent_part) # articulation origin
```

Apply a final result through the model-owned binding API:

```python
model.bind_placement(PlacementTarget.visual(support), result)
model.bind_placement(PlacementTarget.joint(model.get_joint("drawer_slide")), result)
```

Visual binding requires the result subject to be that Visual. Joint binding
accepts the joint or its child Part as the result subject and requires the
parent-link-local frame. A Visual may have `visual_uid is None` immediately after
standalone construction or `Part.visual()`; the next model-owned validation,
binding, or collision compile assigns a monotonic write-once UID. Binding retains
evidence records for the final pose and every chained solve step. Rebinding marks
the earlier record superseded; active element and ancestor-joint records can then
be retained as compile-time overlap evidence. The default compile text does not
expand those records or print Stable Visual UIDs. It may show one neutral
`placement_evidence` line only for one exact direct Visual-to-Visual match; complete
UID, bind, provenance, and orientation facts remain in
`reports/compile_findings.json`.

## Analytic Anchors

Structured selectors avoid concatenated magic names:

```python
face = anchor(panel, AnchorSelector.face("+z", uv=(0.25, 0.75)), frame=part_frame)
edge = anchor(support, AnchorSelector.edge("+y", "-z", t=0.5), frame=part_frame)
corner = anchor(block, AnchorSelector.corner("+x", "-y", "+z"), frame=part_frame)
```

- `uv` and `t` are strict normalized values in `[0, 1]`; unlike the legacy
  `place_on_face_uv`, they are not clamped.
- Face `+z` is the outward normal. Edge `+z` is its line direction.
- Box selectors are exact. Cylinder/Sphere support only features the analytic
  shape actually owns. Unsupported or underdetermined features fail locally.
- A Part selector is an explicit bounds selector over the union of its elements;
  use a Visual when the relation belongs to one exact element.

Anchor source kinds preserve that distinction. Primitive/Visual selectors are
`analytic_face/edge/corner`; Part selectors are
`part_bounds_face/edge/corner`; CadQuery, verified surface, Mesh feature,
semantic, bounds, explicit, and internal mount producers carry their corresponding
typed source. A plane-like Anchor is not automatically an analytic face.
Geometry eligibility uses `source_kind`; stable identity matching uses
`owner_kind` plus the complete canonical `owner_ref`. These fields are orthogonal.

When both sides of `attach` come from analytic or CadQuery faces and their resolved
mating normals have dot product greater than `0.999`, the provenance step records
both normals, the dot, commanded `axis_map`, and resolved relative basis/RPY. This
is measurement evidence only. Antiparallel faces, Part bounds, surfaces, Mesh
patches, semantic/bounds/explicit anchors, and internal planes do not emit it.

### Seating One Visual Against Another In The Same Part

Two visuals on one part still have a real dependency: the display backing follows
the fascia it is mounted on. Express it as a relation, so moving the fascia moves
the display with it.

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

# The host carries its own design coordinate: where the fascia sits on the drawer.
fascia = part.visual(
    Box((0.36, 0.018, 0.11)),
    origin=Origin(xyz=(0.0, 0.12, 0.055)),
    name="fascia",
)
# The dependent carries no coordinate at all — the relation supplies it.
display = part.visual(Box((0.14, 0.006, 0.05)), name="display_backing")

# ONE frame for both Anchors: the PART that owns them. See "Frame Identity" below.
frame = FrameRef.part_local(part)
seat = attach(
    anchor(display, AnchorSelector.face("-y"), frame=frame),
    anchor(fascia, AnchorSelector.face("+y"), frame=frame),
    # Mating faces meet back-to-back: the subject's outward normal (+z of a face
    # Anchor) runs OPPOSITE the host's. "+x": "+x" then pins the remaining twist.
    axis_map={"+z": "-z", "+x": "+x"},
    uv_offset=(0.0, -0.012),  # 12 mm along the host face, in the TARGET Anchor's +y
)
model.bind_placement(PlacementTarget.visual(display), seat)

seated_origin = display.origin  # Origin(xyz=(0.0, 0.132, 0.067))
```

Two rules this example encodes, both of which reject a first correct-looking
attempt:

**Frame identity.** `anchor(...)` defaults to `frame=FrameRef.part_local(subject)`
and `FrameRef` compares **owner identity**, so two visuals on one part default to
two different frames and `attach(...)` refuses them:

```text
ValidationError: subject and target Anchors must use the same consumption frame
```

Pass `frame=FrameRef.part_local(part)` — the owning Part, not either Visual — to
**both** `anchor(...)` calls. That single argument is the difference between the
relation working and falling back to a literal.

**Mating normals.** Omitting `axis_map` performs full frame-to-frame alignment,
which turns the subject 180 degrees and seats it *inside* the host (here
`y=0.126` instead of `0.132`). Two mating faces are back-to-back, so map the
subject's `+z` to the target's `-z`.

### Building The Host And Seating The Subject In One Pass

The recipe above reads as if the host were already there. Your real task builds
both in one file: the host is fabricated a few lines earlier, so at seating time
its numbers are still on screen and the dependent coordinate is one addition
away. That addition is the relation. Author it as one.

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

model = ArticulatedObject(name="incubator")

# --- Pass 1: FABRICATE. The host's own geometry, in the part's own frame. ---
body = model.part("body", est_mass_kg=9.0)
body.visual(Box((0.62, 0.42, 0.26)), name="shell")
deck = body.visual(
    Box((0.58, 0.38, 0.014)),
    origin=Origin(xyz=(0.0, 0.0, 0.083)),  # the deck's own place inside the body
    name="deck",
)

# --- Pass 2: ASSEMBLE. Every dependent pose is a query against what Pass 1 built. ---
frame = FrameRef.part_local(body)  # the owning PART, on BOTH anchors

cover = body.visual(Box((0.10, 0.07, 0.008)), name="port_cover")
seat = attach(
    anchor(cover, AnchorSelector.face("-z"), frame=frame),
    anchor(deck, AnchorSelector.face("+z"), frame=frame),
    axis_map={"+z": "-z", "+x": "+x"},
    uv_offset=(0.17, 0.0),  # along the deck's own +x, measured from its face center
    clearance=0.001,  # a real gasket line, not a fudge
)
model.bind_placement(PlacementTarget.visual(cover), seat)

seated_origin = cover.origin  # Origin(xyz=(0.17, 0.0, 0.095))
```

`0.095` is `0.083 + 0.007 + 0.001 + 0.004`: the deck's height, half its
thickness, the gasket line, half the cover. Every term was typed on a line
above, which is exactly what makes the sum a relation rather than a design
decision. Written as a literal it decouples: change the deck to 0.018 thick and
the cover sinks into it, silently, with no compile error to catch it.

So fabricate first, assemble second:

- **Pass 1 — fabricate.** Each part's own geometry in its own frame. An
  `Origin(...)` for a part's internal offset — the deck inside the body — is
  correct here and stays.
- **Pass 2 — assemble.** Every pose that follows another element is an Anchor
  query against the geometry Pass 1 built, including two visuals in one part.

Both passes live in the same script; the split is derivation order, not file
count. The frame rule from the previous section still governs:
`frame=FrameRef.part_local(body)` — the owning Part — on BOTH anchors.

### Edge-To-Edge Attach

This incubator support follows both deck depth and deck height. No coordinate is
retuned when the deck changes:

```python
frame = FrameRef.part_local(body)
result = attach(
    anchor(support, AnchorSelector.edge("+y", "-z"), frame=frame),
    anchor(deck, AnchorSelector.edge("+y", "+z"), frame=frame),
    axis_map={"+x": "+x", "+y": "-y"},
)
model.bind_placement(PlacementTarget.visual(support), result)
```

`axis_map` maps signed subject-Anchor axes to signed target-Anchor axes. Two
non-collinear mappings determine full rotation. One mapping aligns only that axis
and inherits twist from `from_placement`. Omitting `axis_map` maps corresponding
axes and performs full frame-to-frame alignment. Contradictory or collinear maps
fail at the call site.

Offsets have separate meanings:

- `translate=(x, y, z)`: target-Anchor-local translation.
- `clearance=d`: target Anchor `+z` stand-off.
- `uv_offset=(u, v)`: target Anchor `(+x, +y)` in-plane offset.
- `Anchor.offset(translate=..., clearance=..., uv=...)`: move an Anchor itself.

## Span And Composition

`gap(...)` is a query and writes no pose. `center_between(...)` writes one
translation line. The Dolly connector derives both length and center:

```python
frame = FrameRef.part_local(undercarriage)
start = anchor(central_riser, AnchorSelector.face("+y", uv=(0.5, 0.3125)), frame=frame)
end = anchor(support_bridge, AnchorSelector.face("-y"), frame=frame)

neck = undercarriage.visual(
    Box((0.080, gap(start, end), 0.025)),
    # center_between writes ONLY the y line, so the seed must supply x and z.
    # Seed them from the host feature the connector starts at, not a magic triple.
    origin=Origin(xyz=start.pose.position),
    name="central_rail_neck",
)
result = center_between(neck, start, end, axis=(0.0, 1.0, 0.0), frame=frame)
model.bind_placement(PlacementTarget.visual(neck), result)
```

Chain orthogonal operations explicitly:

```python
r1 = snap(
    drawer,
    against=[back_panel],
    direction=(0.0, -1.0, 0.0),
    gap=0.001,
    frame=joint_frame,
)
r2 = center_between(
    drawer,
    left_inner,
    right_inner,
    axis=(1.0, 0.0, 0.0),
    frame=joint_frame,
    from_placement=r1,
)
```

`snap(...)` searches only the explicit `against=` set and moves on one normalized
line. Equal nearest targets raise `ambiguity`. If `max_travel` is insufficient,
the error reports `blocked_by`, achieved travel, and deficit. It does not inspect
the rest of a partially built model.

## Derived Dimensions

Sizes follow the same dependency rule as poses: write one determinant and derive
every mating dimension from it. Blueprint values fix the outer envelope; as-built
host geometry fixes interior and mating features. Common families are:

- A feature that **fits inside** another: `insert_size = host_inner_size - 2 * fit_clearance`.
- A shell that **wraps around** another: `shell_outer = wrapped_outer + 2 * wall_thickness`.
- A member that **spans between** features: `member_length = gap(start, end) - 2 * end_clearance`.
- A bore that another feature **passes through**: `bore_diameter = shaft_diameter + 2 * radial_clearance`.
- Repeated features that **patterns onto** a host: `pattern_pitch = host_span / (count - 1)`.
- A guide sized for the **travel it must cover**: `guide_length = travel + carriage_length + end_margin`.

Use shared constants when both dimensions are authored together. Use `anchor(...)`
and `gap(...)` when the determinant is as-built geometry. `probe_model` verifies
the determinant; its printed value is not a new literal source.

## Shared Generator Specs

Frozen shared specs use immutable constructors, not model binding. One returned
`HingePin` still drives both hinge halves and the joint:

```python
mount = anchor(cleat, AnchorSelector.face("+y"), frame=part_frame).offset(
    translate=(0.003, 0.007, 0.0)
)
pin = HingePin.from_anchor(
    mount,
    axis=mount.pose.axis("-x"),
    length=0.05,
    knuckle_count=5,
    pin_diameter=0.006,
    knuckle_diameter=0.010,
    clearance=3e-4,
)
pin_axis = semantic_anchor(pin, "pin_axis")
```

`HingePin.from_edge(edge_anchor, ...)` is a convenience over the same kernel and
accepts only an axis Anchor. `semantic_anchor(rail, "runner_axis")` exposes a
`SlideRail` axis through the same `Anchor` type.

## Mesh Source Rules

Anchor consumers are source-independent, but feature extraction is not.

### Verified Surface

```python
mount = surface_anchor(mesh_visual, direction=(0.0, 0.0, 1.0), frame=part_frame)
```

`surface_anchor(...)` queries in source-local coordinates and transforms the
completed frame afterward. `point_hint` and `direction` are mutually exclusive.
This route works for unannotated Meshes when the requested surface point is
unique.

### Bounds Are Not Contact Features

```python
diagnostic_corner = bounds_anchor(mesh, AnchorSelector.corner("+x", "+y", "+z"))
```

`bounds_anchor(...)` explicitly returns a synthetic AABB feature. It is useful for
measurement/layout, but a rounded or hollow Mesh's bounds face/edge/corner may be
far from real material. Do not silently use it as a mount contact.

### Strict Geometric Features

```python
patch = mesh_feature_anchor(mesh, "planar_patch", direction=(0.0, 0.0, 1.0))
edge = mesh_feature_anchor(mesh, "feature_edge", point_hint=(0.2, 0.1, 0.0))
corner = mesh_feature_anchor(mesh, "feature_corner", point_hint=(0.2, 0.1, 0.3))
```

These selectors use planar facets or sharp-feature topology, not raw triangle
indices. Rounded meshes normally return `no-feature`; multiple equally valid
features return `ambiguity`; a planar patch whose centroid lies in a hole is
rejected. On either failure, use a verified surface point or explicit fallback.

### Controlled Blender Semantic Anchors

A Mesh authored by the current task through `run_blender` may declare stable
semantic frames in the same script. After modifiers and decimation, the driver
validates and stages the sidecar. Resolve it with:

```python
mount = semantic_anchor(mesh_visual, "mount_face", frame=part_frame)
axis = semantic_anchor(mesh_visual, "pin_axis", frame=part_frame)
```

An external or legacy OBJ without that controlled sidecar cannot invent semantic
names from triangle indices.

### Explicit Fallback

```python
mount = explicit_anchor(
    mesh_visual,
    RigidFrame(position=(0.12, -0.03, 0.08)),
    frame=part_frame,
    kind="point",
)
```

Use `explicit_anchor(...)` or a hand-written `Origin(...)` when design intent is
known but automatic feature selection is `no-feature` or `ambiguity`.

## CadQuery BRep Anchors

CadQuery source models retain exact topology before tessellation:

```python
mount = cadquery_anchor(
    cq_model,
    AnchorSelector.face("+z", uv=(0.5, 0.5)),
    frame=part_frame,
    subject=visual,
)
```

`cadquery_anchor(...)` selects planar faces by outward normal/support location
and resolves edge/corner adjacency from BRep topology, never transient face
indices. Multiple support faces and UV points in holes fail explicitly.

## Compatibility APIs

Existing code remains valid:

- `surface_frame(...)` is the compatibility view of `surface_anchor(...)`.
- `place_on_surface(...)`, `place_on_face(...)`, and `place_on_face_uv(...)`
  retain their signatures and now use the shared Anchor/rigid-frame math.
- `wrap_mesh_onto_surface(...)` and `wrap_profile_onto_surface(...)` retain their
  specialized deformation behavior over the shared surface query.
- `align_centers(...)` retains its selected-axis AABB contract.
- `proud_for_flush_mount(...)` is deprecated. New code should attach an explicit
  child support Anchor with `clearance=`; the old helper remains as a v0 shim.
- Legacy `place_on_face_uv(...)` still clamps UV and margin values for
  compatibility. New `AnchorSelector` values are strict.

## Final Validation

Placement calls validate only local facts: frame/subject mismatch, invalid or
ambiguous selectors, contradictory axis maps, and blocked declared snaps. The
complete object model must still go through `compile_model`, whose existing
collision, island, motion, and related QC reports remain authoritative. The
compiler does not attribute collisions to placement commands and the relational
layer does not suppress or rewrite those findings.

## See Also

- `20_core_types.md` for `Origin`, `Visual`, and `Part`
- `50_blender_freeform.md` for same-script semantic Anchor declarations
- `35_cadquery.md` for CadQuery authoring
- `49_hinges.md` and `51_slides.md` for shared connector specs
