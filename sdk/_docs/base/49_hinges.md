# Hinges

## Purpose

A hinge OPENS, and opening is ARTICULATION (a `REVOLUTE` joint on the pin line),
NOT geometry. A part held to the body only by its joint floats away when the joint
turns; a REAL hinge keeps the two parts attached through the whole swing with a
physical CONNECTOR — interleaved knuckles around a shared pin. These helpers build
that connector, split into one HALF per part:

- `HingePin` — a shared spec that defines the pin axis line + knuckle layout ONCE,
  so both halves are coaxial and interleave by construction.
- `HingeHalfGeometry(pin, role=...)` — one part's share of the knuckles
  (`role="outer"` takes the end segments, `role="inner"` the gaps), auto-aligned to
  `pin.axis`; plus an optional `leaf` and an optional through-`pin`.
- `HingeLeaf` — a flat leaf plate, when the part bodies do not themselves meet at
  the pin (a cabinet door on a face frame).

Because the knuckles sit ON the pin axis, rotating the joint never moves them off
it — the parts stay connected (never float) at EVERY angle. This is what the
open-pose support screen checks; build the connector and it passes.

For ordinary hardware, keep this connector localized and unobtrusive: use the
shortest credible knuckle row and the smallest credible pin/knuckle diameters.
Reserve a full-span barrel for a real piano hinge. A leaf is conditional, not a
default: omit it when each part body already reaches its own knuckles; otherwise
use only enough leaf to join the knuckles to that body.

## Make a hinge (the usual case: the part bodies are the leaves)

A box lid on a box wall — no separate leaf plates needed; the lid panel and the
base box ARE the two leaves.

```python
from sdk import (
    AnchorSelector, ArticulatedObject, ArticulationType, Box, FrameRef, MotionLimits, Origin,
    HingePin, HingeHalfGeometry, anchor, mesh_from_geometry, TestContext, TestReport,
)


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="hinged_box")
    shell = model.material("shell", rgba=(0.88, 0.89, 0.91, 1.0), material_class="matte_plastic")
    steel = model.material("hinge_steel", rgba=(0.18, 0.20, 0.24, 1.0), material_class="steel")

    base = model.part("base")
    base_box = base.visual(Box((0.20, 0.20, 0.10)), origin=Origin(xyz=(0.0, 0.0, 0.05)), material=shell, name="base_box")

    # ONE shared pin: the axis line + knuckle layout. anchor is a point ON the pin —
    # use it for BOTH the geometry and the joint origin so they share the pin line.
    # SEED THE PIN FROM THE HOST, not a typed triple: the hinge line IS the box's
    # top-back edge, so it follows the box when the box is resized.
    hinge_line = anchor(base_box, AnchorSelector.edge("+y", "+z"), frame=FrameRef.part_local(base))
    pin = HingePin.from_edge(
        # An edge Anchor's +z IS the edge line (+X here). Take -z so positive q lifts
        # the lid; the SIGN is the open-direction decision, verify it with a render.
        hinge_line,
        axis=hinge_line.pose.axis("-z"),
        length=0.05,
        knuckle_count=5,
        pin_diameter=0.006,
        knuckle_diameter=0.010,
        clearance=3e-4,
    )

    # Parent half: the root part's frame is the world, so place its hinge at pin.anchor.
    base.visual(mesh_from_geometry(HingeHalfGeometry(pin, role="outer", include_pin=True), "hinge_base"),
                origin=Origin(xyz=pin.anchor), material=steel, name="hinge_base")

    lid = model.part("lid")
    lid.visual(Box((0.19, 0.19, 0.02)), origin=Origin(xyz=(0.0, -0.095, 0.01)), material=shell, name="lid_panel")
    # Child half: the child's frame sits ON the pin (its joint origin == pin.anchor),
    # so its hinge geometry is anchor-relative -> place it at the default Origin().
    lid.visual(mesh_from_geometry(HingeHalfGeometry(pin, role="inner"), "hinge_lid"),
               material=steel, name="hinge_lid")

    # THE JOINT — geometry and kinematics share the pin: origin = pin.anchor, axis = pin.axis.
    # The axis SIGN sets which way positive q swings: here -X lifts the lid UP (open).
    # ALWAYS confirm with a side render — a wrong sign swings the lid DOWN INTO the body.
    model.articulation(
        "lid_hinge", ArticulationType.REVOLUTE, parent=base, child=lid,
        origin=Origin(xyz=pin.anchor), axis=pin.axis,
        motion_limits=MotionLimits(effort=5.0, velocity=3.0, lower=0.0, upper=1.9),
    )
    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    # Nothing to declare — the interleaved knuckles are clearance-separated and the pin
    # is VISUAL-ONLY (not a collider), so a hinge built with these helpers has no overlap.
    return TestContext(object_model).report()
```

## Cabinet door on a face frame (leaves needed)

When the part bodies do NOT meet at the pin (a wide door beside a frame post),
give each half a `HingeLeaf` so a flat leaf bridges each body to the pin line.

```python
from sdk import (
    AnchorSelector, FrameRef, HingePin, HingeHalfGeometry, HingeLeaf,
    anchor, mesh_from_geometry, Origin,
)

# Same seeding rule as above: the pin line is the post stile's front-outer vertical
# edge. edge("+x", "+y") runs along +Z (the edge direction is the cross product of
# the two face normals); `t` picks the point along it that the hinge centers on.
hinge_line = anchor(
    post_stile, AnchorSelector.edge("+x", "+y", t=0.5), frame=FrameRef.part_local(post)
)
pin = HingePin.from_edge(
    hinge_line,
    axis=hinge_line.pose.axis("+z"),
    length=0.06,
    knuckle_count=5,
    pin_diameter=0.005,
    knuckle_diameter=0.009,
    clearance=3e-4,
    part_span=2.0,
)
leaf = HingeLeaf(width=0.014, thickness=0.002, toward=(0, 1, 0), length=0.06)
post.visual(mesh_from_geometry(HingeHalfGeometry(pin, role="outer", include_pin=True, leaf=leaf), "hinge_base"),
            origin=Origin(xyz=pin.anchor), material=steel, name="hinge_base")
door.visual(mesh_from_geometry(HingeHalfGeometry(pin, role="inner", leaf=leaf), "hinge_lid"),
            material=steel, name="hinge_lid")
# ... REVOLUTE on origin=Origin(xyz=pin.anchor), axis=pin.axis as above ...
```

Repeat this short module at the other supported location for a tall door. Do not
turn it into one full-height barrel unless the design is specifically a piano hinge.

## Continuous piano hinge

A piano hinge is one long barrel row, not one long knuckle per role. Keep each
knuckle segment stocky: segment_length / knuckle_diameter <= 2.5. For a 1.6 m
row with 20 mm knuckles, use about 33 interleaved segments:

    piano = HingePin(
        axis=(0, 0, 1),
        anchor=(0.0, 0.0, 0.85),
        length=1.60,
        knuckle_count=33,
        pin_diameter=0.012,
        knuckle_diameter=0.020,
        part_span=2.0,
    )

part_span is optional lint context. A hinge longer than about 60% of the part
span emits a non-blocking warning: for a true continuous piano hinge that can be
intentional; for ordinary cabinet hardware, prefer two or three shorter
HingePin assemblies at supported locations.

## Rules that keep it clean

- **Seed the pin from the host geometry, not a typed triple.** The hinge line is a
  real feature of the part it mounts on — a box edge, a stile face. Build it with
  `HingePin.from_edge(...)`/`from_anchor(...)` over an `AnchorSelector` Anchor so the
  pin follows that feature when the host is resized. A hand-typed `anchor=(x, y, z)`
  is the one coordinate in this whole recipe that goes stale.
- **One `HingePin`, passed to BOTH halves AND the joint.** The two halves take the
  SAME pin, so their knuckles are coaxial and interleave; the joint uses
  `origin=Origin(xyz=pin.anchor), axis=pin.axis`, so geometry and kinematics share
  the pin line. Do NOT hand-type the joint origin/axis separately.
- **Place the child half at `Origin()`, the (root) parent half at
  `Origin(xyz=pin.anchor)`.** The geometry is built anchor-relative; the child's
  frame already sits on the pin (its joint origin), the root parent's frame is the
  world. **This is NOT a double-placement**, and the parent half is NOT at the world
  origin: the root parent's PART frame IS the world, so giving its hinge VISUAL the
  offset `Origin(xyz=pin.anchor)` places the knuckles ON the pin line exactly once
  (part frame = world, applied once; visual offset = anchor, applied once — same anchor,
  not stacked). The child half needs no offset because its own joint origin is already
  `pin.anchor`, so its local frame sits on the pin. (If the hinge's parent is itself
  articulated — not the root — offset the parent half by the anchor expressed in the
  parent's frame instead of the world.)
- **`role`: one half `"outer"`, the other `"inner"`.** They take complementary,
  interleaving segments. `knuckle_count` is odd by convention (the outer half gets
  the two ends).
- **Keep ordinary connectors local.** Start with a short module and compact
  pin/knuckle diameters. Omit `HingeLeaf` when both part bodies reach the barrel;
  when a leaf is necessary, use the minimum width that actually fuses each half to
  its own body. Do not enlarge a leaf or bridge merely to silence a floating finding;
  verify the seat and full-travel clearance instead. Full-span barrels are for real
  piano hinges.
- **`axis` may point ANY direction — knuckles, pin and `leaf` all follow it.** A
  sloping pin (a tapered case's cover, a slanted lid) is supported: give `toward` any
  direction not parallel to the pin and the leaf is built in the pin's own frame, with
  its along-pin component projected out. Never hand-roll a leaf for a sloping hinge —
  a hand-rolled plate gets none of the interleave notching, and it will intersect the
  other half through the swing. The axis SIGN sets the OPEN direction — verify with a side render; a wrong sign
  swings the part DOWN/INTO the body instead of open.
- **Bearing contact: nothing to declare, ever.** The interleaved knuckles are
  clearance-separated, and `include_pin=True` adds a solid through-pin to ONE half.
  The OTHER half automatically **bores its knuckles** (`bore = not include_pin`) so the
  pin passes with radial clearance, and each leaf is **notched** where the opposite
  half's knuckles sit — so BOTH the exact (visual) mesh AND the colliders are
  overlap-free (nothing to declare). Pass the pin to exactly ONE half. A hinge built
  with these helpers is clearance-clean under full collision with NO `allow_overlap`.
  (`pin.declare_bearings(...)` / `pin.bearing_pairs(...)` are deprecated no-ops kept only
  so old code does not break — do not add allowances.)
- Geometry only makes the SHAPE; the `REVOLUTE` joint makes it MOVE.

## Physics collision (why a hinge does not jam in MuJoCo)

A solver (MuJoCo/PhysX) collides a `mesh` geom as its **convex hull** unless the export
convex-decomposes it first. Decomposition runs for every mesh with measurable concavity —
only a mesh that is convex within numerical tolerance is left whole, because only then is
the hull lossless. So a panel with functional through-bores keeps them as real holes in
collision; you do not have to split it into separate solids to defend them.

Decomposition is an approximation, not a guarantee: CoACD may still fill part of a small
hole in a large panel. Treat a bore as reliable clearance only when it is comfortably
larger than the thing passing through it, and confirm with `gap_report` rather than
assuming. What you must NOT do is design around the old failure — carving material away
so a concavity "survives". See the pin-clearance note below.

A hinge half
is several *disconnected* knuckle cylinders — its hull fills the inter-knuckle gaps,
so the two coaxial halves' hulls overlap ~100 % at the pin and shove the parts apart
the moment the same-joint parent↔child contact is enabled ("full collision"). The
SDK avoids this automatically: a `HingeHalfGeometry` carries its **exact analytic
colliders** (one cylinder per knuckle, the leaf as a box), and the physics export emits
those primitives INSTEAD of the fused mesh — so the interleave + clearance are preserved
and the halves clear at the closed pose. The through-`pin` (with `include_pin=True`) is
VISUAL-ONLY — never a collider. The EXACT (visual) mesh is kept clean the same way real
hardware is: the half WITHOUT the pin bores its knuckles (`bore = not include_pin`) so
the pin passes with clearance, and each leaf is notched around the opposite half's
knuckles — so compile-time overlap checks (which read the exact mesh) also pass. You do
not configure any of this; it rides on the same `HingePin`/`role`. Two consequences to know:

- The two leaves are auto-stacked along their thickness normal BY ROLE (`outer` on
  one side, `inner` on the other), so the closed pose never co-locates them. The
  visual leaf is fused into the barrel (one watertight solid); the collider leaf
  stays tangent so it never reaches across the pin into the other half's knuckles.
- This makes the HINGE robust under full collision. The part BODIES are still your
  job: under full collision (`filterparent` disabled — the target the object must
  satisfy) the door/lid body also collides the cabinet/frame it swings against. It must
  clear that body — and any toe-grille / frame lip / adjacent panel — by a positive gap
  (aim ≥ 1 mm) across the WHOLE swing, not just look flush at the closed pose. A panel
  modeled overlapping the frame is the residual hinge defect the PENETRATION and SIM-JAM
  screens flag (see `70_probe_tooling.md`); probe `gap_report`/`pair_report` between the
  body and the frame at the closed AND open poses to confirm the gap holds.

### The pin must sit CLEAR of the host wall — lengthen the leaf, do not hollow the wall

The knuckles are coaxial with the joint axis, so **rotating the hinge never moves them**
(see the note at the top of this file). A knuckle that overlaps the host wall therefore
overlaps it identically at EVERY angle: that is a **static placement error**, not a
motion-clearance requirement. Do not carve a swing envelope for it — there is no swing.

Fix it at the pin, not at the wall. Push the pin just outside the wall face and let the
leaf span the difference — that is exactly what the leaf is for, and `HingeLeaf.width`
(or a slightly deeper part body acting as the leaf) is the knob to turn. The overlap is
usually well under a millimetre.

```text
WRONG                                   RIGHT
 wall ──┬──────────  knuckle grazes      wall ──┬────────   pin fully clear of the wall
        │ ⊙  the wall by ~0.5 mm, so            │
        │    the model cuts a full-height       │   ⊙       leaf reaches out to the pin
        │    window through the wall            └───────    (a ~1 mm move, no window)
```

If, after the pin is clear, some geometry genuinely must pass through the wall, cut the
**smallest local opening** that admits it — sized to the feature and placed at the hinge
line — never a full-height slot, and never separate solids. A rice-cooker build carved
three 24 mm x 218 mm windows through its rear wall to resolve a 0.5 mm knuckle overlap;
the windows are visible in the final render, and Review flagged the body silhouette as a
major defect. Collision geometry is derived from your visuals, so **every collision
workaround is something the user sees** — prefer moving a part 1 mm over removing material.

## API Reference

### `HingePin`

```python
HingePin(
    axis,            # pin direction (any unit-ish vector)
    anchor,          # a point ON the pin line == the REVOLUTE joint origin
    length,          # knuckle-row length along the pin
    knuckle_count=5, # interleaved segments (odd by convention)
    pin_diameter=0.006,
    knuckle_diameter=0.010,
    clearance=3e-4,
    pin_style=None,  # HingePinStyle: pin end / exposed_end detail
    part_span=None,  # optional parent/child span for soft proportion linting
)
```

```python
HingePin.from_anchor(mount, *, length, axis=None, knuckle_count=5, ...)
HingePin.from_edge(edge_anchor, *, length, axis=None, knuckle_count=5, ...)
```

Seed the pin from the host instead of typing `anchor=(x, y, z)`: both take an
`Anchor` (`from_edge` requires an axis Anchor, i.e. an `AnchorSelector.edge(...)`)
and read `anchor` from its position. `axis` defaults to the Anchor's `+z` — the
edge line for an edge Anchor, the outward normal for a face — so pass
`mount.pose.axis("-z")` when the open direction is the other way. The resulting
pin is an ordinary frozen `HingePin`; everything downstream is unchanged.

- `pin.axis`, `pin.anchor` — pass to the joint: `origin=Origin(xyz=pin.anchor), axis=pin.axis`.
- `pin.bearing_pairs(...)`, `pin.declare_bearings(...)` — DEPRECATED no-ops (the pin is
  visual-only now, so there is no bearing overlap to declare). Do not call them.

- Keep the knuckle segment length / knuckle diameter ratio at or below 2.5;
  the constructor emits a non-blocking warning above that ratio.

### `HingeLeaf`

```python
HingeLeaf(width, thickness, toward, length=None, holes=None)
```

A flat leaf; hinge edge tangent to the knuckle barrel, extending `width` along
`toward` (perpendicular to the pin), spanning `length` along the pin (default =
pin length). `holes` (`HingeHolePattern`) is reserved for leaf screw patterns.

### `HingeHalfGeometry`

```python
HingeHalfGeometry(pin, *, role, leaf=None, include_pin=False, bore=None)
```

One part's half: its interleaved knuckles (`role="outer"|"inner"`), an optional
`leaf`, and an optional through-`pin`. Wrap with `mesh_from_geometry(geom, name)`
to add as a visual. `include_pin=True` on exactly ONE half adds the solid pin;
`bore` defaults to `not include_pin` (the pin-less half hollows its knuckles so the
pin clears) — leave it unset unless you need a pin-less hinge (`bore=False`).

`HingeHalfGeometry` carries functional collision primitives. Do not mutate it with
`add_vertex`, `add_face`, `merge`, `translate`, `scale`, or `rotate`; place each mesh
with `Visual.origin` so visual and collision frames remain identical.
