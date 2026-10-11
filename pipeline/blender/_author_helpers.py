from __future__ import annotations

import bmesh  # type: ignore[import-not-found]
import bpy  # type: ignore[import-not-found]

_DECIMATE_MOD = "__artisan_budget_decimate__"


def object_mode() -> None:
    if bpy.context.mode != "OBJECT":
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:  # noqa: BLE001 — no active object / already object mode
            pass


def _eval_bmesh(obj) -> "bmesh.types.BMesh":
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    oe = obj.evaluated_get(dg)
    me = oe.to_mesh()
    bm = bmesh.new()
    bm.from_mesh(me)
    oe.to_mesh_clear()
    return bm


def _tri_count(bm) -> int:
    return sum(max(0, len(f.verts) - 2) for f in bm.faces)


def decimate_to_budget(obj, max_faces: int) -> int:
    bm = _eval_bmesh(obj)
    nt = _tri_count(bm)
    bm.free()
    if max_faces and nt > max_faces:
        d = obj.modifiers.new(_DECIMATE_MOD, "DECIMATE")
        d.decimate_type = "COLLAPSE"
        d.ratio = max(0.0, min(1.0, float(max_faces) / float(nt)))
    return nt


def _components(bm) -> int:
    bm.verts.index_update()
    n = len(bm.verts)
    if not n:
        return 0
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for e in bm.edges:
        ra, rb = find(e.verts[0].index), find(e.verts[1].index)
        if ra != rb:
            parent[ra] = rb
    return len({find(i) for i in range(n)})


def mesh_stats(obj) -> dict:
    bm = _eval_bmesh(obj)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    n_faces = len(bm.faces)
    n_verts = len(bm.verts)
    boundary = sum(1 for e in bm.edges if len(e.link_faces) != 2)
    watertight = boundary == 0 and n_faces > 0
    n_comp = _components(bm)
    if bm.verts:
        xs = [v.co.x for v in bm.verts]
        ys = [v.co.y for v in bm.verts]
        zs = [v.co.z for v in bm.verts]
        aabb = [[min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]]
        dims = [round(aabb[1][i] - aabb[0][i], 5) for i in range(3)]
        aabb = [[round(c, 5) for c in aabb[0]], [round(c, 5) for c in aabb[1]]]
    else:
        aabb, dims = [[0, 0, 0], [0, 0, 0]], [0, 0, 0]
    bm.free()
    return {
        "visual_tris": n_faces,
        "n_verts": n_verts,
        "watertight": watertight,
        "n_components": n_comp,
        "aabb": aabb,
        "dims": dims,
    }


def export_obj_zup(obj, path) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.obj_export(
        filepath=str(path),
        export_selected_objects=True,
        apply_modifiers=True,
        up_axis="Z",
        forward_axis="Y",
        export_materials=False,
    )
