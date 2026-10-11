---
example_kind: geometry_pattern
title: 'Caster Wheel with a Curved Skin-Modifier Fork (Blender freeform)'
description: 'Technique demo: author a slender CURVED tubular member (a caster yoke / fork) in headless Blender with the Skin modifier instead of stacking boxy primitives, then attach it via mesh_from_input and wire the swivel + spin joints. The Skin skeleton->subsurf->bore->voxel-remesh recipe generalizes to any curved tubular part — fork, yoke, handle, hook, curved frame, spoke, arm.'
tags:
  - blender freeform
  - run_blender
  - mesh_from_input
  - skin modifier
  - curved member
  - tubular part
  - caster
  - swivel caster
  - caster fork
  - caster yoke
  - wheel
  - handle
  - hook
  - bracket arm
  - frame
  - continuous joint
  - voxel remesh
  - watertight collider
---
# Caster Wheel with a Curved Skin-Modifier Fork

**This is a TECHNIQUE demonstration, not a fixed shape.** It shows how to author a
slender *curved* tubular member in headless Blender with the **Skin modifier** — the
right tool when a part is a flowing, tapered, molded member that the SDK primitives
(or `PivotForkGeometry`'s filleted *rectangular* bracket) can only approximate as a
boxy box-stack. The illustrated part is a twin-cheek **caster fork**; to build a
handle, hook, curved frame, spoke, or any other curved member, **change the skeleton
points + radii** and keep the same five steps. Full rules: `read_file("50_blender_freeform.md")`.

The Skin recipe: an **edge skeleton** (the centreline, which may branch) → `SKIN`
(a tapered tube, per-vertex radius) → `SUBSURF` (smooth) → boolean-`DIFFERENCE` any
bore/axle hole → `REMESH` `VOXEL` (one **watertight** manifold — Skin branches leave
separate overlapping tubes otherwise). A clean watertight mesh CoACD-decomposes
tightly (pieces hug the shape, the wheel gap stays open instead of being filled).

## Step 1 — the bpy script (`blender/caster_fork.py`)

```python
# blender/caster_fork.py — a curved twin-cheek caster yoke via the Skin technique.
# Adapt the skeleton points/radii for any curved member; the 5 steps are the same.
import bpy
import math

verts, edges, radii = [], [], []

# THE axle point (in the fork's local frame): the bore is cut here AND the wheel_spin
# joint origin in model.py MUST be this SAME point — including the Y caster trail.
# (If they differ, the wheel sits off the bore: it overlaps the cheeks and its axle
# misses the hole. The e2e mistake was bore y=0.025 but joint y=0.)
AXLE = (0.0, 0.014, 0.0)


def v(p, r):
    verts.append(p)
    radii.append(r)
    return len(verts) - 1


# 1) edge skeleton (centreline). Stem (kingpin post) -> crown -> branch into 2 cheeks.
top = v((0.0, 0.0, 0.120), 0.011)
s2 = v((0.0, 0.0, 0.098), 0.011)
stm = v((0.0, 0.0, 0.074), 0.012)
crn = v((0.0, 0.0, 0.050), 0.015)
edges += [(top, s2), (s2, stm), (stm, crn)]
for sx in (-1.0, 1.0):  # two mirrored curved cheeks, separated along X (the axle)
    chain = [
        ((0.016 * sx, 0.000, 0.050), 0.013),
        ((0.034 * sx, 0.001, 0.040), 0.012),
        ((0.044 * sx, 0.010, 0.012), 0.012),
        ((0.041 * sx, AXLE[1], AXLE[2]), 0.016),  # FAT boss at the axle (thick bore wall)
    ]
    prev = crn
    for p, r in chain:
        i = v(p, r)
        edges.append((prev, i))
        prev = i

me = bpy.data.meshes.new("fork")
me.from_pydata(verts, edges, [])
me.update()
obj = bpy.data.objects.new("caster_fork", me)
bpy.context.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj
obj.select_set(True)

# 2) Skin = tapered tube around the skeleton; 3) Subsurf = smooth.
obj.modifiers.new("skin", "SKIN")
sv = obj.data.skin_vertices[0].data
for i, r in enumerate(radii):
    sv[i].radius = (r, r)
sv[crn].use_root = True
sub = obj.modifiers.new("sub", "SUBSURF")
sub.levels = 3
bpy.ops.object.modifier_apply(modifier="skin")
bpy.ops.object.modifier_apply(modifier="sub")

# 4) axle bore (a cylinder along X at AXLE, through both cheeks). Cut it BEFORE the
#    remesh: a boolean on the clean subsurf mesh is reliable; a boolean on a dense
#    remeshed mesh can shatter it into shards.
bore_r = 0.0085
bpy.ops.mesh.primitive_cylinder_add(
    radius=bore_r, depth=0.13, location=AXLE, rotation=(0.0, math.radians(90), 0.0), vertices=48
)
bore = bpy.context.active_object
bd = obj.modifiers.new("bore", "BOOLEAN")
bd.operation = "DIFFERENCE"
bd.object = bore
bd.solver = "EXACT"
bpy.context.view_layer.objects.active = obj
bpy.ops.object.modifier_apply(modifier="bore")
bpy.data.objects.remove(bore, do_unlink=True)

# 5) one watertight manifold (unions the Skin-branch overlaps + keeps the bore).
#    voxel_size must be a few × SMALLER than the bore wall (here the fat boss leaves
#    ~7 mm of wall around an 8.5 mm bore), or the remesh webs the hole shut / jags
#    its rim. Finer voxel = more faces, but the driver decimates to budget anyway.
rm = obj.modifiers.new("rm", "REMESH")
rm.mode = "VOXEL"
rm.voxel_size = 0.0012
bpy.ops.object.modifier_apply(modifier="rm")
for p in obj.data.polygons:
    p.use_smooth = True

# DECLARE the bore as a semantic Anchor in the SAME script that cut it. This mesh
# owns a placement-relevant interface, so model.py resolves the axle instead of
# retyping AXLE — the two can no longer drift apart.
export_result(
    obj,
    anchors=[
        {
            "name": "axle",
            "kind": "axis",
            "position": AXLE,
            "axis": (1.0, 0.0, 0.0),   # the bore centreline; becomes Anchor +z
            "up": (0.0, 0.0, 1.0),
            "surface_bound": False,     # a bore centre is in air, not on material
            "tolerance": 0.0005,
        },
    ],
)
```

Run it once: `run_blender("blender/caster_fork.py", "caster_fork")`. LOOK at the
returned preview (curved cheeks, stem post, OPEN round bore, gap clear), check
`watertight=True`. The axle hub fills the bore in the assembly, so a slightly
imperfect rim is hidden — but a webbed/closed hole (too-coarse voxel) is not.

## Step 2 — wire it into `model.py`

The fork is the Blender mesh; the **wheel** is an ordinary SDK `TireGeometry` + axle
hub that sits in the gap. Two `CONTINUOUS` joints give the caster its motion: a
vertical **swivel** (the stem is the kingpin) and a horizontal **spin** (the axle).
The spin-joint origin is not retyped — it is READ from the `axle` Anchor the bpy
script declared, so the bore and the joint are one fact with one source.

```python
from __future__ import annotations

import math

from sdk import (
    ArticulatedObject,
    ArticulationType,
    Cylinder,
    FrameRef,
    MotionLimits,
    Origin,
    TestContext,
    TestReport,
    TireGeometry,
    TireGroove,
    mesh_from_geometry,
    mesh_from_input,
    semantic_anchor,
)


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="swivel_caster")
    black = model.material("black", rgba=(0.04, 0.04, 0.04, 1.0), material_class="matte_plastic")
    rubber = model.material("rubber", rgba=(0.03, 0.03, 0.03, 1.0), material_class="rubber")

    # grounded stub leg the caster hangs from (a real asset uses the chair/table leg).
    leg = model.part("leg")
    leg.visual(Cylinder(radius=0.012, length=0.06), origin=Origin(xyz=(0.0, 0.0, 0.155)), material=black)

    # the curved Blender yoke — ONE visual, the whole fork.
    fork = model.part("caster_fork")
    fork_yoke = fork.visual(mesh_from_input("caster_fork"), material=black, name="fork_yoke")

    # the wheel: an SDK tire + an axle hub along X (fits the ~0.06 m cheek gap).
    wheel = model.part("caster_wheel")
    wheel.visual(
        mesh_from_geometry(
            TireGeometry(0.026, 0.044, inner_radius=0.014, grooves=(TireGroove(0.0, 0.006, 0.003),)),
            "tire",
        ),
        material=rubber,
    )
    wheel.visual(
        Cylinder(radius=0.013, length=0.050),
        origin=Origin(rpy=(0.0, math.pi / 2.0, 0.0)),
        material=black,
    )

    # swivel: the fork yaws around vertical Z (the stem is the kingpin in the socket).
    model.articulation(
        "caster_yaw",
        ArticulationType.CONTINUOUS,
        parent="leg",
        child="caster_fork",
        origin=Origin(xyz=(0.0, 0.0, 0.120)),
        axis=(0.0, 0.0, 1.0),
        motion_limits=MotionLimits(effort=2.0, velocity=8.0),
    )
    # spin: the wheel rolls around the axle (X). The origin is the SAME point the bpy
    # bored, and it is READ from the Anchor that script declared rather than retyped —
    # including the Y caster trail. Retyping it is where this goes wrong (the common
    # mistake: bore at y=0.025 but joint at y=0), leaving the wheel overlapping the
    # fork with its axle off the hole. The wheel's own axle hub is centred here too,
    # so it fills the bore.
    axle = semantic_anchor(fork_yoke, "axle", frame=FrameRef.part_local(fork))
    model.articulation(
        "wheel_spin",
        ArticulationType.CONTINUOUS,
        parent="caster_fork",
        child="caster_wheel",
        # Position only: the joint keeps its own axis convention (axis=+X below), so
        # take the Anchor's point and do not re-orient the frame onto its basis.
        origin=Origin(xyz=axle.pose.position),
        axis=(1.0, 0.0, 0.0),
        motion_limits=MotionLimits(effort=1.0, velocity=30.0),
    )
    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    # the wheel sits in the fork gap on the axle — an intended contact, not a clash.
    ctx.allow_overlap("caster_fork", "caster_wheel", reason="wheel rides on the axle inside the fork gap")
    return ctx.report()
```

Notes that transfer to any curved member:
- The Blender mesh is built in the part's own local frame (Z-up); place it with the
  visual/joint `origin`, never inside Blender.
- **A bored hole and the joint that uses it are the SAME point — so give it ONE
  source.** Declare the bore as a semantic Anchor in the script that cut it and read
  it back with `semantic_anchor(...)`; the bpy bore centre, the `wheel_spin` origin,
  and the wheel's axle-hub centre then cannot disagree. Retyping the coordinate in
  `model.py` is the frequent failure (matching Z but not the trail Y), and it leaves
  the wheel overlapping the cheeks with its axle off the hole.
- **A bore wants a fat boss + a fine voxel.** Put a thick boss radius at the bored
  vertex and keep `voxel_size` a few × smaller than the resulting wall, or the voxel
  remesh webs the hole shut and jags its rim. The mating part (the axle hub) fills
  the bore, so the rim need not be perfect — but it must be OPEN and roughly round.
- Match the cheek gap width to the wheel/tire width (widen the cheek `sx` offsets for
  a wider wheel) so the wheel sits in the gap without poking through the cheeks.
- Colour is set on the SDK side (`material=`), not in the bpy.
