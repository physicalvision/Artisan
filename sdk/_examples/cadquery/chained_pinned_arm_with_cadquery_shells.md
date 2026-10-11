---
example_kind: complete_model
title: 'Chained Pinned Arm with CadQuery Shells'
description: 'Two PinnedJointSpec connectors in series on CadQuery-authored segments. The middle part is the MEMBER of the first joint and the SUPPORT of the second, and each hand-built shell is seated on the datum its connector half publishes via mount_local().'
tags:
  - cadquery
  - sdk
  - PinnedJointSpec
  - PinnedJointGeometry
  - clevis
  - pivot fork
  - fork
  - yoke
  - through-pin
  - through pin
  - pin joint
  - pinned joint
  - lug
  - bore
  - cheek
  - revolute
  - revolute joint
  - revolute chain
  - joint chain
  - chained joints
  - two joints in series
  - multi-joint arm
  - arm
  - articulated arm
  - support arm
  - forearm
  - monitor arm
  - monitor mount
  - display mount
  - elbow
  - elbow joint
  - wrist
  - wrist joint
  - tilt
  - tilt joint
  - pivot
  - pivot joint
  - bracket
  - mount bracket
  - folding arm
  - hollow beam
  - beam
  - segment
  - joint clearance
  - radial clearance
  - axial clearance
  - mount_local
  - full collision joint
  - collision-safe joint
  - interlocking parts
  - part straddles another part
  - mesh from cadquery
  - cadquery anchor
---
# Chained Pinned Arm with CadQuery Shells

A three-segment arm whose two revolute joints are both real `PinnedJointSpec`
connectors. It shows the two things a single-joint demo cannot:

1. **A part that owns both roles.** `forearm` carries the MEMBER half of
   `elbow_pitch` and the SUPPORT half of `wrist_pitch`. Chaining connectors needs
   no special API — each half is just another visual on the part.
2. **How a hand-built CadQuery shell meets a connector.** Each half publishes
   `mount_local()`, the connector-local datum where the owning body must sit. Seat
   the shell on that datum instead of guessing a coordinate; a joint alone does not
   connect an isolated mesh.

## The mount datum is subject-relative — pass it raw

`mount_local()` is expressed in the connector's own canonical frame, so its numbers
generally do NOT equal part-local coordinates. That does not matter and you must not
"fix" it: hand the raw value to `explicit_anchor` with the connector VISUAL as the
subject, and the visual's own `origin` supplies the rotation.

    mount = PinnedJointGeometry(fit, role="member").mount_local()
    host = explicit_anchor(member_visual, RigidFrame(position=mount.xyz),
                           frame=part_frame, kind="point")

This is orientation-independent: verified to seat the body correctly for every
`axis`/`opening_direction` pairing. Pre-rotating `mount_local()` into the part frame
is the failure mode — it silently misses the mount on any joint whose
`joint_origin().rpy` is not identity, with no exception raised.

The standoffs likewise come from the connector, not from a guess:

    SUPPORT_STANDOFF = -PinnedJointGeometry(TEMPLATE, role="support").mount_local().xyz[1]

so `.offset(clearance=SUPPORT_STANDOFF)` lands the clevis base flush on the
segment's end face. Re-query the seat for the second joint AFTER the first body is
bound — an anchor resolves against the as-built pose, so the wrist pin follows the
forearm wherever the elbow placement put it.

```python
from __future__ import annotations

import cadquery as cq

from sdk import (
    AnchorSelector,
    ArticulatedObject,
    ArticulationType,
    FrameRef,
    MotionLimits,
    Origin,
    PinnedJointGeometry,
    PinnedJointSpec,
    PlacementTarget,
    RigidFrame,
    TestContext,
    TestReport,
    attach,
    cadquery_anchor,
    explicit_anchor,
    mesh_from_cadquery,
    mesh_from_geometry,
)

SEG_WIDTH = 0.034
SEG_THICKNESS = 0.028

# One shared fit for every joint in the chain. Real objects vary these per joint;
# sharing them here keeps the topology lesson in front.
FIT_DIMS = dict(
    axis=(1.0, 0.0, 0.0),
    opening_direction=(0.0, 1.0, 0.0),
    pin_diameter=0.009,
    lug_diameter=0.026,
    member_width=0.022,
    cheek_thickness=0.006,
    radial_clearance=4.5e-4,
    axial_clearance=5.5e-4,
    support_arm_length=0.022,
    support_mount_thickness=0.005,
    member_neck_length=0.024,
    member_neck_thickness=0.012,
    retention="permanent",
)
TEMPLATE = PinnedJointSpec(style="clevis", anchor=(0.0, 0.0, 0.0), **FIT_DIMS)
# How far the pin line must stand off the body face so the connector's own mounting
# face lands flush on it. Read it from the connector; never guess a gap. The datum
# is CONNECTOR-canonical, where both halves mount along the local Y axis whatever
# the joint's real orientation is, so index [1] is the standoff for any pairing.
SUPPORT_STANDOFF = -PinnedJointGeometry(TEMPLATE, role="support").mount_local().xyz[1]


def _segment(length: float) -> cq.Workplane:
    return (
        cq.Workplane("XY")
        .box(SEG_WIDTH, length, SEG_THICKNESS)
        .edges("|Y")
        .fillet(0.006)
    )


def _pin_seat(solid, visual, frame):
    """Stand the pin line off the segment's +Y end face by the connector's own datum."""
    return cadquery_anchor(
        solid, AnchorSelector.face("+y", uv=(0.5, 0.5)), frame=frame, subject=visual
    ).offset(clearance=SUPPORT_STANDOFF)


def _seat_body_on_member(model, fit, member_visual, solid, visual, frame) -> None:
    """Butt a CadQuery shell against the datum the MEMBER half publishes.

    The datum is a point anchor: the shell keeps its own axis-aligned orientation
    and only translates onto the mount. A `surface_anchor` on the connector mesh
    would work too, but its in-plane tangents are arbitrary and can twist a long
    body 90 degrees.

    Pass `mount.xyz` RAW. It is resolved against `member_visual`, which already
    carries the joint's rotation in its own origin, so this is correct for every
    joint orientation. Pre-rotating it into the part frame silently misses.
    """
    mount = PinnedJointGeometry(fit, role="member").mount_local()
    model.bind_placement(
        PlacementTarget.visual(visual),
        attach(
            cadquery_anchor(
                solid,
                AnchorSelector.face("-y", uv=(0.5, 0.5)),
                frame=frame,
                subject=visual,
            ),
            explicit_anchor(
                member_visual, RigidFrame(position=mount.xyz), frame=frame, kind="point"
            ),
        ),
    )


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="chained_pinned_arm")
    steel = model.material("steel", rgba=(0.22, 0.23, 0.25, 1.0), material_class="steel")

    # --- base: SUPPORT of joint 1 -------------------------------------------
    base = model.part("base", est_mass_kg=1.2)
    base_solid = _segment(0.150)
    base_visual = base.visual(
        mesh_from_cadquery(base_solid, "base_body"), material=steel, name="base_body"
    )
    base_frame = FrameRef.part_local(base)

    elbow_fit = PinnedJointSpec(
        style="clevis",
        anchor=_pin_seat(base_solid, base_visual, base_frame).pose.position,
        **FIT_DIMS,
    )
    base.visual(
        mesh_from_geometry(PinnedJointGeometry(elbow_fit, role="support"), "elbow_clevis"),
        origin=elbow_fit.joint_origin(),
        material=steel,
        name="elbow_clevis",
    )

    # --- forearm: MEMBER of joint 1 AND SUPPORT of joint 2 -------------------
    forearm = model.part("forearm", est_mass_kg=0.8)
    elbow_member = forearm.visual(
        mesh_from_geometry(PinnedJointGeometry(elbow_fit, role="member"), "elbow_lug"),
        origin=Origin(rpy=elbow_fit.joint_origin().rpy),
        material=steel,
        name="elbow_lug",
    )
    forearm_solid = _segment(0.150)
    forearm_visual = forearm.visual(
        mesh_from_cadquery(forearm_solid, "forearm_body"), material=steel, name="forearm_body"
    )
    forearm_frame = FrameRef.part_local(forearm)
    _seat_body_on_member(
        model, elbow_fit, elbow_member, forearm_solid, forearm_visual, forearm_frame
    )

    # Queried AFTER the forearm shell is bound, so the wrist pin follows it.
    wrist_fit = PinnedJointSpec(
        style="pivot_fork",
        anchor=_pin_seat(forearm_solid, forearm_visual, forearm_frame).pose.position,
        **FIT_DIMS,
    )
    forearm.visual(
        mesh_from_geometry(PinnedJointGeometry(wrist_fit, role="support"), "wrist_fork"),
        origin=wrist_fit.joint_origin(),
        material=steel,
        name="wrist_fork",
    )

    # --- head: MEMBER of joint 2 --------------------------------------------
    head = model.part("head", est_mass_kg=0.5)
    wrist_member = head.visual(
        mesh_from_geometry(PinnedJointGeometry(wrist_fit, role="member"), "wrist_lug"),
        origin=Origin(rpy=wrist_fit.joint_origin().rpy),
        material=steel,
        name="wrist_lug",
    )
    head_solid = _segment(0.110)
    head_visual = head.visual(
        mesh_from_cadquery(head_solid, "head_body"), material=steel, name="head_body"
    )
    head_frame = FrameRef.part_local(head)
    _seat_body_on_member(
        model, wrist_fit, wrist_member, head_solid, head_visual, head_frame
    )

    # --- joints: origin and axis both come from the connector ----------------
    model.articulation(
        "elbow_pitch",
        ArticulationType.REVOLUTE,
        parent=base,
        child=forearm,
        origin=elbow_fit.joint_origin(),
        axis=elbow_fit.joint_axis_local,
        motion_limits=MotionLimits(lower=-0.7, upper=0.7, effort=12.0, velocity=1.0),
    )
    model.articulation(
        "wrist_pitch",
        ArticulationType.REVOLUTE,
        parent=forearm,
        child=head,
        origin=wrist_fit.joint_origin(),
        axis=wrist_fit.joint_axis_local,
        motion_limits=MotionLimits(lower=-0.5, upper=0.5, effort=8.0, velocity=1.0),
    )
    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    return TestContext(object_model).report()
```

## Rules this example follows

- Two connector visuals per joint, one `support` and one `member`. No independent
  pin Part, no second FIXED joint.
- The connector geometry is never mutated. It is placed with `Visual.origin`:
  `joint_origin()` on the support side, `Origin(rpy=joint_origin().rpy)` on the
  member side, whose part-local origin IS the pin line.
- Joint `origin` and `axis` come from `joint_origin()` / `joint_axis_local`, so
  the kinematic frame and the physical pin can never drift apart.
- Limits are bounded. The connector does not choose them: a clevis base, fork
  bridge, or member neck can stop the sweep before the limit does. Use bounded
  `REVOLUTE` unless the whole assembly has passed a 360-degree collision sweep.
- No `allow_overlap` anywhere. The cheeks straddling the lug is the intended
  interlock, and both halves carry real clearance bores.
