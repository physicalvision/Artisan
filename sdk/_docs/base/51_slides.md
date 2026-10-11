# Slides (drawers, baskets, trays)

**Use `SlideRail` + `SlideRailGeometry`. Do NOT hand-roll box rails, shanks or
guide bores and then declare `allow_overlap` to hide the runner clipping the wall:
the helpers carry their own fit clearance, so a runner inside a channel is
clearance-clean under full collision with no allowance at all.**

## Purpose

A drawer/basket/tray SLIDES, and sliding is ARTICULATION (a `PRISMATIC` joint on
the slide axis), NOT geometry. A part held to the body only by its joint floats away
when it extends; a REAL slide keeps the two parts mated through the whole travel with
a physical CONNECTOR — a runner captured inside a channel. These helpers build that
connector, split into one HALF per part, exactly like the hinge helpers:

- `SlideRail` — a shared spec that defines the slide axis + channel cross-section
  ONCE, so the channel and the runner interlock by construction.
- `SlideRailGeometry(rail, role=...)` — one part's share: `role="channel"` builds the
  fixed C/U channel (floor + two side walls + optional capture lips), `role="runner"`
  a T — a captured crossbar riding inside the channel with a fit clearance PLUS a stem
  that rises through the mouth/slot to a body-mount face (mirroring the hinge's leaf).

The runner's stem is what bridges the moving BODY (drawer/basket) to the rail. Seat the
body so it overlaps `rail.runner_mount_local()` — it is the SAME part, so the overlap
just fuses body↔runner. Do NOT leave the runner as a lone bar at the rail line and then
add a bracket to reach the body: that bracket is the thing that ends up clipping the
channel wall. The stem removes the need for it.

Because the runner rides in the channel with clearance on every captured face, it
stays connected (never floats, never jams) at EVERY slide position under full
collision — no `allow_overlap` needed. This is the RIGHT way to build a drawer/basket
rail: do NOT hand-roll box rails and declare `allow_overlap("basket", "cabinet", ...)`
to hide the runner clipping into the wall — that is the drawer-rammed-into-the-wall
defect the connection-cleanliness gate now flags.

## Make a slide (a basket that pulls out of a cabinet)

One rail on each side wall is typical; here one rail on the left wall.

```python
from sdk import (
    AnchorSelector, ArticulatedObject, ArticulationType, Box, FrameRef, MotionLimits, Origin,
    SlideRail, SlideRailGeometry, anchor, mesh_from_geometry, TestContext, TestReport,
)


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="pull_out_basket")
    shell = model.material("shell", rgba=(0.88, 0.89, 0.91, 1.0), material_class="matte_plastic")
    steel = model.material("rail_steel", rgba=(0.18, 0.20, 0.24, 1.0), material_class="steel")

    cabinet = model.part("cabinet", est_mass_kg=22.0)
    left_wall = cabinet.visual(Box((0.02, 0.42, 0.40)), origin=Origin(xyz=(-0.19, 0.0, 0.20)), material=shell, name="left_wall")

    # ONE shared rail: the slide axis + channel cross-section. anchor is a point ON the
    # slide line — use it for BOTH the geometry and the joint origin so they share it.
    # SEED THE RAIL FROM THE HOST, not a typed triple: the rail line lies on the wall's
    # INNER face, a quarter of the way up, so it follows the wall when the wall moves.
    rail_seat = anchor(
        left_wall, AnchorSelector.face("+x", uv=(0.5, 0.25)), frame=FrameRef.part_local(cabinet)
    )
    # width/height are the INNER cavity the runner rides in; the runner auto-shrinks by
    # `clearance` on every captured face. length >= runner_length + travel keeps the
    # runner on the channel through the whole pull.
    # up = the mount direction: the runner stem reaches +up out of the channel to the
    # basket. Here the rail is on the left wall and `up` points inward (+X) to the basket.
    rail = SlideRail(
        slide_axis=(0, 1, 0), up=(1, 0, 0), anchor=rail_seat.pose.position,
        length=0.42, width=0.014, height=0.020, runner_length=0.40,
        travel=0.34,
        mount_offset=0.05,   # stem reach from the wall rail inward to the basket side
    )

    # Parent half: the root part's frame is the world, so place its channel at rail.anchor.
    cabinet.visual(mesh_from_geometry(SlideRailGeometry(rail, role="channel"), "rail_channel"),
                   origin=Origin(xyz=rail.anchor), material=steel, name="rail_channel")

    basket = model.part("basket", est_mass_kg=1.4)
    # Child half: the child's frame sits ON the slide (its joint origin == rail.anchor),
    # so its runner is anchor-relative -> place it at the default Origin().
    basket.visual(mesh_from_geometry(SlideRailGeometry(rail, role="runner"), "rail_runner"),
                  material=steel, name="rail_runner")
    # Seat the body so it OVERLAPS the runner's mount face (same part -> fuses, no island,
    # no bracket). runner_mount_local() = up * mount_offset in the child frame.
    sx, sy, sz = rail.runner_mount_local()
    basket.visual(Box((0.30, 0.30, 0.18)), origin=Origin(xyz=(sx + 0.14, sy, sz)), material=shell, name="basket_body")

    # THE JOINT — geometry and kinematics share the slide: origin = rail.anchor, axis = rail.slide_axis.
    # `upper` is the pull-out travel; keep it <= length - runner_length + a retained bite so the
    # basket cannot pull fully off the rail (Review checks a drawer can't be pulled out).
    model.articulation(
        "basket_slide", ArticulationType.PRISMATIC, parent=cabinet, child=basket,
        origin=Origin(xyz=rail.anchor), axis=rail.slide_axis,
        motion_limits=MotionLimits(effort=40.0, velocity=0.5, lower=0.0, upper=0.34),
    )
    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    # Nothing to declare — the runner rides in the channel with clearance (no overlap).
    return TestContext(object_model).report()
```

## Rules that keep it clean

- **Seed the rail from the host geometry, not a typed triple.** The slide line lies on
  a real feature — the carcass wall's inner face, an under-mount shelf underside.
  Resolve it with an `AnchorSelector` Anchor and pass `anchor=<anchor>.pose.position`,
  so the rail follows that face when the carcass is resized. A hand-typed
  `anchor=(x, y, z)` is the one coordinate in this recipe that goes stale.
- **One `SlideRail`, passed to BOTH halves AND the joint.** Both halves take the SAME
  rail, so the runner and channel interlock; the joint uses
  `origin=Origin(xyz=rail.anchor), axis=rail.slide_axis`, so geometry and kinematics
  share the slide line. Do NOT hand-type the joint origin/axis separately.
- **Place the child (runner) half at `Origin()`, the (root) parent (channel) half at
  `Origin(xyz=rail.anchor)`.** The geometry is built anchor-relative; the child's frame
  already sits on the slide (its joint origin), the root parent's frame is the world.
- **`slide_axis` may point ANY direction; `up` only has to be non-parallel to it**
  (e.g. `slide_axis=(0,1,0)`, `up=(0,0,1)`, or a sloping rail on an inclined worktop).
  `up`'s along-slide component is projected out and the third axis is the lateral
  width. The channel
  mouth faces `+up`, and the runner stem rises `+up` to the body — so point `up` at where
  your body is (inward for a side-wall rail, upward for an under-mount rail).
- **Seat the body on `rail.runner_mount_local()`, do NOT add a bracket.** Set
  `mount_offset` to the gap from the rail line to your body's mounting face, then place
  the body so it overlaps `runner_mount_local()` (same part -> the overlap fuses them).
  Leaving the runner as a lone bar and bridging it with a separate bracket is the #1 slide
  defect — the bracket clips the channel; the connectivity gate flags the island otherwise.
- **`capture=True` (default)** adds top lips so the runner is held on all four
  cross-section faces (a C-channel — the runner slides in from the end). `capture=False`
  leaves the top open (a U-channel — the runner drops in and is held by the joint), for
  a basket that lifts out.
- **`runner_length >= travel + retained_margin`.** Pass the PRISMATIC travel to
  `SlideRail(travel=...)`; the soft lint uses a default retained margin of
  `max(0.01, 2*wall)` unless `retained_margin` is explicit. This keeps a useful
  runner section engaged through the full pull. Also keep the fixed channel long
  enough to support that runner at both endpoints.
- **No `allow_overlap`.** The runner rides with a fit clearance, so there is no
  interpenetration to declare. If you find yourself wanting to declare a slide overlap,
  build the connector instead.
- Geometry only makes the SHAPE; the `PRISMATIC` joint makes it MOVE.

## Physics collision (why a slide does not jam in MuJoCo)

A solver (MuJoCo/PhysX) collides a `mesh` geom as its **convex hull**. A C/U channel is
concave — its hull fills the trough, so a runner modeled inside it sits in solid
material and the slide jams the moment the same-joint parent↔child contact is enabled
("full collision"). The SDK avoids this automatically: a `SlideRailGeometry` channel
carries its **exact analytic colliders** (floor + side walls + capture lips as separate
boxes) and the runner is a crossbar box + a stem box, all separated by the fit clearance;
the physics export emits those primitives INSTEAD of the fused mesh — so the runner rides
in the channel with clearance and the slide moves freely. The stem is narrower than the
capture slot, so it clears the lips over the whole travel. You do not configure this; it
rides on the same `SlideRail`/`role`.

**Only the RUNNER is guaranteed clean — the mounted BODY is your job.** `SlideRail`
guarantees the runner rides its own channel with clearance; it does NOT know about the
cabinet interior. Under full collision (`filterparent` disabled — the target the object
must satisfy) the drawer/basket BODY you mount on the runner also collides the carcass
walls, dividers, and the central mullion. If the body is too wide or sits too deep, it
jams against those the moment collision is on, even though the runner is perfect. So:
**the mounted body must clear the cabinet interior — every surrounding wall/divider/
mullion — by a positive gap (aim ≥ 1 mm) over the WHOLE travel, not just at rest.** Size
the body to the opening minus clearance on each side, not to the cavity. A body pinched
into its channel or grazing the mullion is the residual slide defect the PENETRATION and
SIM-JAM screens flag (see `70_probe_tooling.md`); probe `gap_report`/`pair_report` between
the body and each neighbour at the open AND closed poses to confirm the gap holds.

## API Reference

### `SlideRail`

```python
SlideRail(
    slide_axis,        # travel direction (any direction)
    up,                # channel-opening direction (any direction not parallel to it)
    anchor,            # a point ON the slide line == the PRISMATIC joint origin
    length,            # channel span along slide_axis
    width,             # INNER cavity width (lateral) the runner rides in
    height,            # INNER cavity height (along up)
    runner_length=None,# runner span along slide (default = length)
    wall=0.006,        # channel wall thickness
    clearance=5e-4,    # fit gap the runner is shrunk by on each captured face
    lip=0.004,         # capture-lip inward overhang (capture=True only)
    capture=True,      # True = C-channel (held on 4 faces); False = open U-channel
    mount_offset=None, # runner-stem reach +up to the body face (default: just past channel)
    stem_width=None,   # stem lateral width (default: the capture slot, so it passes the lips)
    travel=None,       # optional PRISMATIC span for runner-retention soft linting
    retained_margin=None, # optional retained runner length (default max(0.01, 2*wall))
)
```

- `anchor` — seed it from the host rather than typing it:
  `anchor=anchor(wall, AnchorSelector.face("+x", uv=(0.5, 0.25)), frame=part_frame).pose.position`.
  `SlideRail` has no `from_anchor` constructor; it takes the Anchor's position
  directly, and everything downstream is unchanged.
- `rail.slide_axis`, `rail.anchor` — pass to the joint:
  `origin=Origin(xyz=rail.anchor), axis=rail.slide_axis`.
- `rail.runner_mount_local()` — the runner stem's body-mount point in the child frame
  (`= up * mount_offset`). Seat the drawer/basket body so it overlaps this point.

### `SlideRailGeometry`

```python
SlideRailGeometry(rail, *, role)  # role="channel" | "runner"
```

One part's half of the slide. Wrap with `mesh_from_geometry(geom, name)` to add as a
visual. Build the fixed channel on the housing/carcass part (`role="channel"`) and the
runner on the moving drawer/basket part (`role="runner"`).

`SlideRailGeometry` carries functional collision primitives. Do not mutate it with
low-level vertex/face edits or mesh transforms; use `Visual.origin` for placement.
