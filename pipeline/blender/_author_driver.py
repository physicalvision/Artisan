from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

import bpy  # type: ignore[import-not-found]

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _author_helpers as H  # type: ignore  # noqa: E402
from _anchor_sidecar import normalize_anchor_declarations  # type: ignore  # noqa: E402


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--script", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--preview", default="")
    p.add_argument("--visual-budget", type=int, default=12000)
    return p.parse_args(argv)


def _fallback_result():
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    return meshes[-1] if meshes else None


def _render_preview(obj, path: str) -> None:
    import mathutils  # type: ignore[import-not-found]

    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.render.resolution_x = 480
    sc.render.resolution_y = 480
    sc.render.image_settings.file_format = "PNG"
    sc.display.shading.light = "STUDIO"
    sc.display.shading.color_type = "SINGLE"
    sc.display.shading.single_color = (0.10, 0.30, 0.85)

    corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    center = sum(corners, mathutils.Vector()) / 8.0
    size = max((max(c[i] for c in corners) - min(c[i] for c in corners)) for i in range(3))
    size = max(size, 1e-3)

    target = bpy.data.objects.new("CamTarget", None)
    bpy.context.collection.objects.link(target)
    target.location = center
    cam_d = bpy.data.cameras.new("Cam")
    cam_d.lens = 50
    cam = bpy.data.objects.new("Cam", cam_d)
    bpy.context.collection.objects.link(cam)
    con = cam.constraints.new("TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    sc.camera = cam
    az, el, d = math.radians(45), math.radians(20), size * 2.6
    cam.location = (
        center.x + d * math.cos(el) * math.sin(az),
        center.y - d * math.cos(el) * math.cos(az),
        center.z + d * math.sin(el),
    )
    sc.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def main() -> None:
    args = _parse_args()
    report: dict = {"ok": False, "stderr": ""}
    try:
        bpy.ops.wm.read_factory_settings(use_empty=True)

        result = {"obj": None, "anchors": None}

        def export_result(obj, *, anchors=None) -> None:
            if getattr(obj, "type", None) != "MESH":
                raise ValueError("export_result(obj): obj must be a Blender MESH object")
            result["obj"] = obj
            result["anchors"] = anchors

        import bmesh  # type: ignore[import-not-found]
        import mathutils  # type: ignore[import-not-found]

        ns = {
            "__name__": "__artisan_authoring__",
            "bpy": bpy,
            "bmesh": bmesh,
            "mathutils": mathutils,
            "export_result": export_result,
            "H": H,
        }
        code = Path(args.script).read_text(encoding="utf-8")
        exec(compile(code, str(args.script), "exec"), ns)  # noqa: S102 — trusted agent script

        obj = result["obj"] or _fallback_result()
        if obj is None:
            raise RuntimeError(
                "no mesh produced; the script must call export_result(obj) with the "
                "final MESH object"
            )

        H.object_mode()
        report["pre_decimate_tris"] = H.decimate_to_budget(obj, args.visual_budget)
        report.update(H.mesh_stats(obj))
        H.export_obj_zup(obj, args.out)
        anchor_payload = normalize_anchor_declarations(result["anchors"])
        if anchor_payload["anchors"]:
            anchor_path = Path(f"{args.out}.anchors.json")
            anchor_path.write_text(json.dumps(anchor_payload, indent=2), encoding="utf-8")
            report["anchors"] = str(anchor_path)
        report["anchor_count"] = len(anchor_payload["anchors"])
        report["ok"] = True
        report["out"] = str(args.out)

        if args.preview:
            try:
                _render_preview(obj, args.preview)
                report["preview"] = str(args.preview)
            except Exception as exc:  # noqa: BLE001 — preview is best-effort
                report["preview_err"] = f"{type(exc).__name__}: {exc}"
    except Exception:  # noqa: BLE001 — report the failure for the agent's repair loop
        report["stderr"] = traceback.format_exc()[-4000:]

    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
