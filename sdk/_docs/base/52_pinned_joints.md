# Functional Pinned Joints

**Use `PinnedJointSpec` + `PinnedJointGeometry`. Do NOT hand-roll bores, journals
or bearing shells: the spec carries `radial_clearance` and `axial_clearance`, so a
shaft inside its housing is clearance-clean under full collision and needs no
`allow_overlap`.**

## Purpose

Use `PinnedJointSpec` and `PinnedJointGeometry` when two parts rotate on one
physical pin or trunnion. The connector is split between the two articulated
Parts and requires exactly one `REVOLUTE` joint.

| Style | Physical topology | Assembly |
| --- | --- | --- |
| `clevis` | rounded two-cheek support + retained through-pin + bored member lug | permanent or removable retained pin |
| `pivot_fork` | open-front rectangular fork + retained through-pin + bored member lug | permanent or removable retained pin |
| `trunnion` | bored support yoke + member-owned integral journals | permanent fabricated-after-insertion assembly |

The old `ClevisBracketGeometry`, `PivotForkGeometry`, and
`TrunnionYokeGeometry` remain visual bracket helpers. Use the new two-half
connector when the joint must work under full collision.

## Recipe

```python
from sdk import (
    AnchorSelector,
    ArticulationType,
    Box,
    FrameRef,
    MotionLimits,
    Origin,
    PinnedJointGeometry,
    PinnedJointSpec,
    anchor,
    mesh_from_geometry,
)

support_body = support.visual(Box((0.05, 0.03, 0.22)), material=steel, name="support_arm")

# SEED THE PIN LINE FROM THE HOST, not a typed triple. The pin sits on the support
# arm's front face, three quarters of the way up, standing 12 mm proud of it — so it
# follows the arm when the arm is resized or moved instead of drifting off it.
pin_seat = anchor(
    support_body, AnchorSelector.face("+y", uv=(0.5, 0.75)), frame=FrameRef.part_local(support)
).offset(clearance=0.012)

fit = PinnedJointSpec(
    style="clevis",
    anchor=pin_seat.pose.position,   # parent-local pin center, derived from the host
    axis=(0.0, 0.0, 1.0),           # parent-local pin direction
    opening_direction=(0.0, 1.0, 0.0),
    pin_diameter=0.010,
    lug_diameter=0.024,
    member_width=0.014,
    cheek_thickness=0.007,
    radial_clearance=5e-4,
    axial_clearance=5e-4,
    retention="permanent",
)

support.visual(
    mesh_from_geometry(PinnedJointGeometry(fit, role="support"), "pivot_support"),
    origin=fit.joint_origin(),
    material=steel,
    name="pivot_support",
)
member.visual(
    mesh_from_geometry(PinnedJointGeometry(fit, role="member"), "pivot_member"),
    origin=Origin(),
    material=steel,
    name="pivot_member",
)
model.articulation(
    "pivot",
    ArticulationType.REVOLUTE,
    parent=support,
    child=member,
    origin=fit.joint_origin(),
    axis=fit.joint_axis_local,
    motion_limits=MotionLimits(lower=-0.6, upper=0.6, effort=8.0, velocity=1.5),
)
```

`axis + opening_direction + anchor` determine the complete joint frame.
`joint_origin()` includes both translation and rotation, so this recipe also works
when `support` is not the root Part. Everything downstream of `anchor=` is derived
from the spec and stays correct by construction — the seed is the one value that
used to be hand-authored, and `50_placement.md` covers the other Anchor sources
(`cadquery_anchor`, `surface_anchor`, `semantic_anchor`) when the support is a mesh. Do not mutate the returned geometry with
`add_vertex`, `add_face`, `merge`, `translate`, `scale`, or `rotate`; place it with
`Visual.origin` as shown.

## Rules

- Build exactly two connector visuals: one `support`, one `member`.
- Do not create an independent pin Part or a second FIXED joint.
- The through-pin belongs to the support visual and is not a collider. Both exact
  meshes contain real clearance bores, while the support/member analytic colliders
  preserve the axial gap.
- A trunnion has no through-pin. Its journals belong to the member and fit the
  segmented hollow support colliders. It requires `retention="permanent"`.
- `radial_clearance` controls pin/journal-to-bore fit. `axial_clearance` is the
  per-side gap between the central member and support cheeks. Do not collapse them
  into one generic clearance.
- Each half exposes `mount_local()` and `mount_size()`. The owning Part body must
  actually meet that mounting face; a joint alone does not connect an isolated mesh.
- The connector does not choose joint limits. A clevis base, fork bridge, member
  neck, or mounted body can limit the sweep. Use bounded `REVOLUTE` unless the entire
  assembly has passed a full 360-degree collision sweep.
- No `allow_overlap` is needed.

## Seating your body on the mount datum

`mount_local()` publishes where the owning Part's body must sit; nothing places the
body for you. Seat it with one point Anchor on the connector visual:

```python
mount = PinnedJointGeometry(fit, role="member").mount_local()
model.bind_placement(
    PlacementTarget.visual(member_body),
    attach(
        anchor(member_body, AnchorSelector.face("-y"), frame=member_frame),
        explicit_anchor(
            pivot_member, RigidFrame(position=mount.xyz),
            frame=member_frame, kind="point",
        ),
    ),
)
```

- **Pass `mount.xyz` RAW.** The datum is connector-canonical, so its numbers usually
  differ from part-local coordinates — that is expected, not an error to correct. It
  resolves against the connector visual, which already carries the joint rotation in
  its own `origin`, so one call is right for every `axis`/`opening_direction` pairing.
  Pre-rotating it into the part frame applies the rotation twice and silently misses
  the mount on any joint whose `joint_origin().rpy` is not identity.
- **Use `kind="point"`** so the body keeps its own orientation and only translates.
- **Never use `surface_anchor(...)` on a connector half for this.** A half is an
  unannotated Mesh, so its in-plane tangents are arbitrary and `attach` can twist a
  long member 90 degrees with no error raised.
- The same call seats the support side with `role="support"` and the support visual.
  Its standoff — how far the pin line must clear the body's end face for the support's
  mounting face to land flush — is `-mount_local().xyz[1]`, usable directly as
  `anchor(...).offset(clearance=...)`. Both halves mount along the connector-canonical
  Y axis whatever the joint's real orientation is, so that index is always the one.

A chained example in CadQuery, where one part owns the member half of one joint and
the support half of the next, is `find_examples("chained pinned arm")`.

## API

```python
PinnedJointSpec(
    style="clevis",                 # clevis | pivot_fork | trunnion
    axis=(1, 0, 0),
    anchor=(0, 0, 0),
    opening_direction=None,         # set explicitly for intentional fork orientation
    pin_diameter=0.010,
    lug_diameter=0.024,
    member_width=0.014,
    cheek_thickness=0.007,
    radial_clearance=5e-4,
    axial_clearance=5e-4,
    support_arm_length=0.040,
    support_mount_thickness=0.008,
    member_neck_length=0.030,
    member_neck_thickness=0.010,
    retention="permanent",
)

PinnedJointGeometry(spec, role="support" | "member")
```
