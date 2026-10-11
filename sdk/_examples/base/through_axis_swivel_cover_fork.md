---
example_kind: geometry_pattern
title: 'Through-Axis Swivel Cover (Clevis Fork straddling a Fixed Body)'
description: 'Topology demo for a collision-aware two-part swivel cover using PinnedJointSpec(style="pivot_fork"). It uses one support half, one member half, and one REVOLUTE joint; the retained pin is visual geometry inside the support half, not an independent Part.'
tags:
  - sdk
  - base sdk
  - swivel
  - swivel cover
  - swivel cap
  - clevis
  - clevis fork
  - pivot fork
  - PinnedJointGeometry
  - fork
  - yoke
  - straddle
  - straddling cover
  - through-pin
  - through pin
  - through axis
  - axis through the body
  - revolute
  - revolute joint
  - cover
  - cap
  - lid
  - hinge
  - flip cover
  - swing cover
  - articulation
  - kinematic topology
  - rotation axis placement
  - usb cover
  - port cover
  - protective cover
  - mesh from geometry
  - topology demo
  # Object-level vocabulary: an agent describes the OBJECT ("monitor arm", "tilt
  # bracket"), not the connector's trade name, so the connector-only tags above
  # never matched the queries that most need this page.
  - arm
  - articulated arm
  - support arm
  - monitor arm
  - monitor mount
  - display mount
  - tilt
  - tilt joint
  - tilt bracket
  - pivot
  - pivot joint
  - swivel arm
  - bracket
  - mount bracket
  - elbow
  - elbow joint
  - lug
  - bore
  - cheek
  - pin joint
  - pinned joint
  - joint clearance
  - radial clearance
  - interlocking parts
  - part straddles another part
  - full collision joint
  - collision-safe joint
---
# Through-Axis Swivel Cover (Clevis Fork straddling a Fixed Body)

> **⚠️ THIS IS A TOPOLOGY / PRINCIPLE DEMO — NOT A PRODUCTION PART. DO NOT COPY THE GEOMETRY.**
> The shapes here are deliberately minimal: plain `Box` stand-ins plus the
> SDK's functional `PinnedJointGeometry` halves. **The ONE thing to copy is the kinematic
> *topology*, never the numbers.** Concretely, copy only this principle: the revolute
> axis must pass **THROUGH the stationary part**, and the moving cover **straddles** that
> part as a **fork / clevis** seated on a retained through-pin. Do **NOT** create an
> independent pin Part, a pin FIXED joint, or a contact exclusion. Do not copy the literal
> dimensions, proportions, or these stand-in box shapes into a real model. A real swivel
> cover needs proper wall detail, real proportions, and a body sized to the actual object
> (a USB stick, a port, a hinge boss, …). Treat the dimensions below as placeholders.

## Why this example exists (the topology lesson)

There are two ways to attach a swinging cover/cap, and only one of them rotates without
self-collision:

- **WRONG — "cap on a top boss" (offset axis).** A revolute axis placed *above* the body
  (e.g. `axis=(0,0,1)` through a boss on the top face) makes the cover **orbit** the body.
  As it swings, the cover arc sweeps *into* the body and into any neighbour — it collides.
  This is the topology that failed in production.
- **RIGHT — functional through-axis fork (this demo).** Use one
  `PinnedJointSpec(style="pivot_fork")` for both halves. The fixed Part owns the support
  half and retained-pin visual; the moving Part owns the bored member half. One REVOLUTE
  joint uses the same frame and axis.

Joint limits still depend on the mounted bodies. This demo caps travel at `2.4 rad`
after an exact full-collision sweep; do not infer `0..π` clearance from the connector alone.

## Honest caveats you MUST carry into a real part

These are the load-bearing lessons (the geometry is throwaway, *these* are not):

1. **A closed-bridge fork sweeps a DISK, so keep the body compact (or change the cover).**
   `PinnedJointGeometry(FIT, role="support")` pivot-fork half has a rear bridge, so the
   cover sweeps a *disk* of radius `~= fork_depth/2` about the pin. **No fixed feature may
   protrude past the body's tip into that swept disk**, or it collides near `180°`. That is
   why the demo body is kept short. A long body would need one of: capped travel (`~140°`),
   a much larger fork, or a **bridgeless open-channel** cover. Size the fork depth to the
   body's reach, not the other way around.
2. **Keep radial and axial clearances independent.** Radial clearance controls the
   pin/bore fit; axial clearance controls the member-to-cheek gaps.
3. **Do not hide connector collisions.** Functional connector halves carry analytic
   collision primitives and require no `allow_overlap` or MuJoCo contact exclusion.

## The build

Two parts: fixed `body` owns the support half, moving `cover` owns the member half, and
`cover_swivel` is the only joint. `PinnedJointSpec` supplies the visual placement,
articulation origin, and local joint axis from one shared frame.

```python
# ---------------------------------------------------------------------------
# TOPOLOGY / PRINCIPLE DEMO — NOT A PRODUCTION PART. DO NOT COPY THE GEOMETRY.
#
# Copy ONLY the kinematic TOPOLOGY, never the numbers:
#   * the revolute axis passes THROUGH the stationary part (here world +X
#     through the body at y=0, z=0), and
#   * the moving cover uses the member half of one functional pinned connector.
# The boxes and ALL dimensions below are deliberately minimal stand-ins.
# A real swivel cover needs proper proportions, wall detail, and a body sized to
# the actual object. Do NOT paste these literal shapes/numbers into a real model.
# ---------------------------------------------------------------------------
from __future__ import annotations

import math

from sdk import (
    AnchorSelector,
    ArticulatedObject,
    ArticulationType,
    Box,
    FrameRef,
    MotionLimits,
    PinnedJointGeometry,
    PinnedJointSpec,
    PlacementTarget,
    anchor,
    attach,
    mesh_from_geometry,
    surface_anchor,
)

FIT = PinnedJointSpec(
    style="pivot_fork",
    axis=(1.0, 0.0, 0.0),
    anchor=(0.0, 0.0, 0.0),
    opening_direction=(0.0, 1.0, 0.0),
    pin_diameter=0.005,
    lug_diameter=0.020,
    member_width=0.014,
    cheek_thickness=0.006,
    radial_clearance=4e-4,
    axial_clearance=8e-4,
    support_arm_length=0.040,
    member_neck_length=0.030,
)


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="through_axis_swivel_cover")

    steel = model.material(
        "stainless", rgba=(0.72, 0.72, 0.68, 1.0), material_class="stainless_steel"
    )
    cover_mat = model.material(
        "blackened_steel_cover", rgba=(0.10, 0.11, 0.12, 1.0), material_class="stainless_steel"
    )

    body = model.part("body")
    support_half = body.visual(
        mesh_from_geometry(PinnedJointGeometry(FIT, role="support"), "support_half"),
        origin=FIT.joint_origin(),
        material=steel,
        name="support_half",
    )
    # The body slab FOLLOWS the support arm — it is not placed at a chosen coordinate.
    # A connector half is an unannotated Mesh, so use a verified surface query rather
    # than bounds. The negative clearance drives 2 mm INTO the arm end so the two
    # solids fuse (same part, so the overlap is intended and declares nothing).
    fixed_body = body.visual(
        material=steel,
        geometry=Box((FIT.overall_width, 0.024, 0.028)),
        name="fixed_body",
    )
    body_frame = FrameRef.part_local(body)
    model.bind_placement(
        PlacementTarget.visual(fixed_body),
        attach(
            anchor(fixed_body, AnchorSelector.face("+y"), frame=body_frame),
            surface_anchor(support_half, direction=(0.0, -1.0, 0.0), frame=body_frame),
            # Normal-only map: a Mesh surface Anchor's in-plane axes are arbitrary
            # tangents, so pinning "+x" to them would twist the box. One mapping
            # aligns the normal and inherits twist from the subject's own pose.
            axis_map={"+z": "-z"},
            clearance=-0.002,
        ),
    )

    cover = model.part("cover")
    member_half = cover.visual(
        mesh_from_geometry(PinnedJointGeometry(FIT, role="member"), "member_half"),
        material=cover_mat,
        name="member_half",
    )
    # Same relationship on the moving side: the panel butts against the neck's far end.
    cover_panel = cover.visual(
        Box((FIT.member_width, 0.040, 0.026)),
        material=cover_mat,
        name="cover_panel",
    )
    cover_frame = FrameRef.part_local(cover)
    model.bind_placement(
        PlacementTarget.visual(cover_panel),
        attach(
            anchor(cover_panel, AnchorSelector.face("-y"), frame=cover_frame),
            surface_anchor(member_half, direction=(0.0, 1.0, 0.0), frame=cover_frame),
            axis_map={"+z": "-z"},
        ),
    )

    model.articulation(
        "cover_swivel",
        ArticulationType.REVOLUTE,
        parent="body",
        child="cover",
        origin=FIT.joint_origin(),
        axis=FIT.joint_axis_local,
        motion_limits=MotionLimits(lower=0.0, upper=2.4, effort=2.0, velocity=3.0),
    )

    return model




```

## Recap — what to take away (topology only)

1. **One spec owns the connector frame.** Place the support half and REVOLUTE origin from
   `FIT.joint_origin()` and use `FIT.joint_axis_local`; do not hand-align a separate pin.
2. **Match the fork swing radius to the body's reach** (caveat 1): a closed-bridge fork
   sweeps a disk, so keep fixed features inside `~fork_depth/2`, or switch to an open channel.
3. **The exact FCL and staged full-MJCF sweeps are the proof.** Keep positive clearances;
   do not add `allow_overlap` or a MuJoCo contact exclusion for connector internals.

Again: the boxes and every number above are **placeholders for demonstration only**. Reuse
the topology, re-author the geometry for your actual object.
