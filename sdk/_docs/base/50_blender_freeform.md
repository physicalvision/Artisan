# Blender freeform authoring (`run_blender` + `mesh_from_input`)

The CAD SDK (primitives, lofts, CadQuery) is right for prismatic / engineered /
revolved / swept parts. Reach for **Blender** for two kinds of shape it does badly:

1. **Genuinely organic / soft** geometry — an upholstered cushion with sag and
   tufting, soft padding, an irregular organic outline (shield / teardrop / D-shaped
   pad), a molded or draped cover.
2. **Slender, curved, molded HARD parts** — the kind you can only approximate in the
   SDK by stacking boxes/cylinders, so it comes out blocky and razor-edged even
   though the real part is smooth, tapered, and sculpted: a caster fork/yoke, a
   sculpted handle or hook, a curved bracket arm, a frame, a spoke. (`PivotForkGeometry`
   etc. give a filleted *rectangular* bracket; when the part is genuinely a flowing
   curved member, Blender's Skin technique below is the right tool.)

Author the piece in headless Blender and attach the result as one of a part's
visuals. The SDK stays the source of truth for the skeleton, joints, and frames;
Blender only produces that one piece's mesh.

If geometry authored by this task owns an interface that later placement consumes
(`mount_face`, `pin_axis`, `runner_axis`, `channel_entry`, and similar), declare
that semantic Anchor in the same script. Decorative Meshes with no relational
consumer do not need unused Anchors.

## When to reach for it

- The shape is soft / organic / irregular and a loft of cross-sections would look
  stiff or symmetric (a real chair-back cushion, a sofa seat, a saddle, a soft
  grip). Prismatic and revolved parts stay in the SDK.
- Freeform is usually only **part of a part**: a backrest = a `Box`/extrude frame
  visual **+** a Blender cushion visual on the SAME part. Blender meshes coexist
  with primitive visuals — you do not convert the whole part.

## Slender curved / branching parts → the Skin technique

For a flowing tubular member — a fork/yoke, handle, hook, curved arm, frame, spoke —
do NOT stack primitives (boxy) and do NOT fight a CadQuery sweep (its profile
orientation is fiddly). Drive a tapered tube along an **edge skeleton** with the
**SKIN** modifier: you author the centreline, Blender wraps the surface. The same
five steps build ANY curved tubular member — change the skeleton points + radii to
your part, do not treat the snippet as one fixed shape:

1. Build a mesh of just **vertices + edges** = the part's centreline. Branch freely
   (one vertex can fan out to several edges — e.g. a stem that splits into two cheeks).
2. Add a **SKIN** modifier; set a per-vertex radius via
   `obj.data.skin_vertices[0].data[i].radius` (fuller where it carries load, slim
   where it tapers) and mark ONE vertex `use_root = True`.
3. Add **SUBSURF** for smoothness.
4. Boolean-DIFFERENCE a cylinder for any **bore / pin / axle hole**
   (`BOOLEAN`, `solver="EXACT"`), BEFORE the remesh (a boolean on the clean subsurf
   mesh is reliable; a boolean on a dense remeshed mesh can shatter it). Give the
   bored vertex a **FAT boss radius** so there is thick wall around the hole.
5. **Guarantee watertight**: a Skin branch leaves the limbs as separate overlapping
   tubes (not welded) → `watertight=False`. A `REMESH` modifier in `VOXEL` mode
   rebuilds the VOLUME as ONE closed manifold — it unions the overlaps AND keeps the
   bore. This is the reliable "make it watertight" step. Pick `voxel_size` a few ×
   SMALLER than the bore wall (≈1 mm for a ~7 mm wall) — too coarse and the remesh
   webs the hole shut / jags its rim. Finer = more faces, but the driver decimates.
6. **A hole and the joint that uses it are ONE point.** If the part is bored for an
   axle/pin that a child joint rotates on, the bore centre, that joint's `origin`,
   and the mating part's centre must be the SAME coordinate in ALL axes (e.g. a
   caster's bore == its `wheel_spin` origin == the wheel hub, trail offset and all).
   A mismatch leaves the mating part overlapping and its axis off the hole.

```python
# blender/<name>.py — skeleton -> skin -> subsurf -> (bore) -> voxel remesh.
import bpy

# 1) centreline as verts+edges (here: a stem branching into two arms — adapt freely).
verts = [(0, 0, 0.10), (0, 0, 0.05), (-0.04, 0, 0.0), (0.04, 0, 0.0)]
edges = [(0, 1), (1, 2), (1, 3)]
rad = [0.011, 0.014, 0.012, 0.012]            # per-vertex radius (the taper)
me = bpy.data.meshes.new("m"); me.from_pydata(verts, edges, []); me.update()
obj = bpy.data.objects.new("part", me); bpy.context.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj; obj.select_set(True)

obj.modifiers.new("skin", "SKIN")             # 2) tapered tube around the skeleton
sv = obj.data.skin_vertices[0].data
for i, r in enumerate(rad): sv[i].radius = (r, r)
sv[1].use_root = True
s = obj.modifiers.new("sub", "SUBSURF"); s.levels = 3   # 3) smooth
bpy.ops.object.modifier_apply(modifier="skin"); bpy.ops.object.modifier_apply(modifier="sub")
# (4) boolean-cut any bore here, then …
rm = obj.modifiers.new("rm", "REMESH"); rm.mode = "VOXEL"; rm.voxel_size = 0.0016  # 5)
bpy.ops.object.modifier_apply(modifier="rm")
for p in obj.data.polygons: p.use_smooth = True

export_result(obj)
```

## The contract

1. `write_file("blender/<name>.py", <bpy script>)` — a normal Python file you can
   re-edit with `replace(path="blender/<name>.py", ...)`. In that script `bpy`,
   `bmesh`, `mathutils` are already in scope.
2. The script builds geometry and calls **`export_result(obj)`** exactly once with
   the final **MESH** object. You do NOT export, decimate, or set materials — the
   driver does. Build the mesh in the **part's own local frame, Z-up**, centred on
   the local origin (you place it in the assembly with the visual `origin`).
3. `run_blender("blender/<name>.py", "<name>")` runs it and stages
   `inputs/<name>.obj`. It returns a report: `visual_tris`, `watertight`,
   `n_components`, `aabb`, `dims` — read it to verify size/placement — PLUS a
   clay-shaded **preview image of the mesh** attached to the result. LOOK at the
   preview: confirm the silhouette, curvature, and proportions match the intended
   part; if the shape is wrong, edit the bpy and re-run before you attach it.
   `stderr` is your error feedback if the script raised. Re-edit and re-run to
   iterate.
4. In `model.py`, attach it: `part.visual(mesh_from_input("<name>"), origin=...,
   color=(r,g,b))`. **Geometry only** — set the colour on the SDK `Visual`, never
   in Blender (the renderer bakes a flat colour; normal/bump maps are not carried).

## Placement-relevant semantic Anchors

Use the optional `anchors=` argument only for interfaces consumed by later
placement. Declarations are in the exported Mesh's source-local frame:

```python
export_result(obj, anchors=[
    {
        "name": "mount_face",
        "kind": "plane",
        "position": (0.0, 0.0, 0.08),
        "normal": (0.0, 0.0, 1.0),
        "up": (0.0, 1.0, 0.0),
        "surface_bound": True,
        "tolerance": 0.0005,
    },
    {
        "name": "pin_axis",
        "kind": "axis",
        "position": (0.0, -0.03, 0.06),
        "axis": (1.0, 0.0, 0.0),
        "up": (0.0, 0.0, 1.0),
        "surface_bound": False,
        "tolerance": 0.0005,
    },
])
```

- Names are stable `snake_case` nouns and must be unique.
- `plane` and surface-bound `point` declarations are checked against the final
  evaluated and decimated OBJ. An off-surface point or mismatched plane normal
  fails `run_blender` instead of producing stale placement metadata.
- `axis` represents a line such as a bore centerline and need not lie on material;
  it cannot be `surface_bound=True`.
- `normal`/`axis` becomes Anchor `+z`; `up` becomes `+y`; the driver constructs and
  validates the right-handed basis.
- The OBJ and `<name>.obj.anchors.json` sidecar are cached and staged together.
  Changing only Anchor metadata invalidates the managed Mesh identity.

Resolve a declaration in `model.py` through the same placement kernel used by
primitives and CadQuery:

```python
mesh_visual = part.visual(mesh_from_input("bracket"), name="bracket")
mount = semantic_anchor(mesh_visual, "mount_face", frame=FrameRef.part_local(part))
```

This authoring permission applies only where the Blender source is controlled by
the current task. For an external or legacy OBJ with no same-script sidecar, do
not infer semantic face/edge/corner names from triangle indices. Use
`surface_anchor(...)`, a strict `mesh_feature_anchor(...)`, `explicit_anchor(...)`,
or a hand-written `Origin(...)`; accept `no-feature`/`ambiguity` instead of
guessing. A third-party OBJ becomes eligible only if it is deliberately re-authored
through the controlled `run_blender` flow.

## Rules that keep it cheap and valid

- **Shape with modifiers, not brute subdivision.** Use `SUBSURF` for smoothness,
  `BEVEL` for soft rounded edges, `DISPLACE` for fabric relief, `SOLIDIFY` for
  thickness. The driver **decimates the result to a triangle budget** (~12k) — that
  decimated mesh is also the review-time collider, so a bloated mesh would slow the
  motion-sweep. Do not chase sub-millimetre detail in geometry; keep it to the
  shape that matters at centimetre scale.
- **Soft parts need ROUNDED edges — do not leave a sharp straight-sidewall seam
  unless the real part genuinely has one.** A cushion / pad / pillow / grip / any
  upholstered or molded-soft piece reads as soft because its PERIMETER rolls
  outward (a bulged, piped, beveled edge) — NOT because the front face is domed.
  The common failure is to build a crowned front + a flatter back joined by a thin
  straight side wall: that side wall is a hard rim that makes the whole thing look
  rigid and boxy even with smooth shading. Round the perimeter (a `BEVEL` with
  several segments, a `SUBSURF` over the whole body, or a rolled edge loop) so the
  front blends into the back. Keep an edge sharp ONLY where the real object has a
  crisp edge (a sewn welt, a hard plastic lip) — and then it is intentional, not an
  artifact. The clay preview that returns is exactly for catching a too-hard edge.
- **One watertight, single-component mesh.** `watertight=False` or
  `n_components>1` in the report are warnings: the export collider (CoACD) and the
  review collision check both want a closed single body. Weld stray verts / fill
  holes / join loose pieces — or, when modifiers (Skin, Boolean) leave overlapping
  or open shells, a `REMESH` in `VOXEL` mode rebuilds the whole thing as one closed
  manifold (the reliable fix; see the Skin section). A clean watertight mesh also
  gives a TIGHT CoACD collider (pieces hug the shape, ~1.0–1.1× the true volume, and
  a real gap/hollow stays open instead of being filled).
- **Look up the API — don't guess.** `lookup_blender_api("<intent>")` searches the
  installed Blender for the exact operator / modifier / `bmesh.ops` name and its
  args (e.g. `lookup_blender_api("solidify thickness")`,
  `lookup_blender_api("bmesh bridge loops")`); add `full=true` for one entry's
  complete doc.

## Minimal example (build any watertight mesh → hand it off)

```python
# blender/<name>.py — a soft rounded pad; replace the shape with whatever you need.
import bpy

bpy.ops.mesh.primitive_cube_add(size=1.0)
obj = bpy.context.active_object
obj.scale = (0.19, 0.045, 0.18)            # half-extents in the part's local frame (m)
bpy.ops.object.transform_apply(scale=True)

bev = obj.modifiers.new("bevel", "BEVEL")  # soft rounded perimeter
bev.width, bev.segments, bev.limit_method = 0.02, 3, "ANGLE"
sub = obj.modifiers.new("subsurf", "SUBSURF")  # smooth padded surface
sub.levels = sub.render_levels = 3
for poly in obj.data.polygons:
    poly.use_smooth = True

export_result(obj)                          # hand the final MESH to the driver
```

Then in `model.py`:

```python
back = model.part("backrest")
back.visual(Box(size=(0.40, 0.04, 0.36)), origin=Origin(xyz=(0, -0.02, 0.18)), name="frame")
back.visual(mesh_from_input("back_cushion"), origin=Origin(xyz=(0, 0.02, 0.18)),
            name="cushion", color=(0.10, 0.22, 0.82))
```
