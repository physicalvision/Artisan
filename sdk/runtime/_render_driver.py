from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import bpy  # type: ignore[import-not-found]
from mathutils import Vector  # type: ignore[import-not-found]

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _render_lighting as lighting  # type: ignore[import-not-found]  # noqa: E402
from _render_device import RenderDevicePolicy, configure_cycles_device  # noqa: E402
from _render_protocol import input_identity, verify_input, write_report  # noqa: E402

_VIEW_DIRS = {
    "front": Vector((0, 1, 0)),
    "side": Vector((1, 0, 0)),
    "top": Vector((0, 0, 1)),
    "perspective": Vector((1, 1, 0.7)),
}

_FOCUS_MARKER_PREFIX = "__artisan_focus__"
_REVIEW_RESOLUTION = 1024
_FOCUS_RESOLUTION = 512
_FOCUS_BOUNDS = None
_RENDER_MODE = "review"
_DEVICE_INFO = {}
_VIEW_TIMINGS = []


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--glb", required=True)
    p.add_argument("--views", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--camera", default="")
    p.add_argument("--mode", default="review")
    p.add_argument("--focus-bounds", default="")
    p.add_argument("--request-file", default="")
    return p.parse_args(argv)


def _free_dir(az_deg: float, el_deg: float) -> Vector:
    az, el = math.radians(az_deg), math.radians(el_deg)
    return Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))


def _reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _shade_auto_smooth_all(angle_deg: float = 35.0) -> None:
    for obj in bpy.data.objects:
        if obj.type != "MESH" or _is_ground(obj):
            continue
        try:
            for poly in obj.data.polygons:
                poly.use_smooth = True
            mod = obj.modifiers.new(name="auto_smooth_edgesplit", type="EDGE_SPLIT")
            mod.use_edge_angle = True
            mod.use_edge_sharp = False
            mod.split_angle = math.radians(angle_deg)
        except (AttributeError, RuntimeError, TypeError):
            pass


def _apply_material_sidecar(glb_path: str) -> None:
    glb = Path(glb_path)
    sidecar = glb.with_name(f"{glb.stem}.materials.json")
    if not sidecar.is_file():
        return
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import _blender_materials  # type: ignore  # noqa: PLC0415

        _blender_materials.apply_materials(str(sidecar))
    except Exception:  # noqa: BLE001
        traceback.print_exc()


def _extract_focus_marker() -> None:
    global _FOCUS_BOUNDS
    _FOCUS_BOUNDS = None
    marker = next(
        (
            o
            for o in bpy.data.objects
            if o.type == "MESH" and o.name.startswith(_FOCUS_MARKER_PREFIX)
        ),
        None,
    )
    if marker is None:
        return
    mins, maxs = [1e9] * 3, [-1e9] * 3
    for c in marker.bound_box:
        w = marker.matrix_world @ Vector(c)
        for i in range(3):
            mins[i] = min(mins[i], w[i])
            maxs[i] = max(maxs[i], w[i])
    center = Vector([(mins[i] + maxs[i]) / 2 for i in range(3)])
    radius = (max((maxs[i] - mins[i]) for i in range(3)) / 2.0) or 1.0
    _FOCUS_BOUNDS = (center, radius)
    bpy.data.objects.remove(marker, do_unlink=True)


def _override_focus_bounds(spec: str) -> None:
    global _FOCUS_BOUNDS
    if not spec:
        return
    values = [float(value) for value in spec.split(",")]
    if len(values) != 4 or values[3] <= 0.0:
        raise ValueError("--focus-bounds must be x,y,z,radius with radius > 0")
    _FOCUS_BOUNDS = (Vector(values[:3]), float(values[3]))


def _import_glb(glb_path: str) -> None:
    bpy.ops.import_scene.gltf(filepath=glb_path)
    _extract_focus_marker()
    _shade_auto_smooth_all()
    _apply_material_sidecar(glb_path)


def _is_ground(o) -> bool:
    return getattr(o, "is_shadow_catcher", False) or o.name.startswith("showcase_ground")


def _scene_bounds():
    mins, maxs = [1e9] * 3, [-1e9] * 3
    found = False
    for o in bpy.data.objects:
        if o.type != "MESH" or _is_ground(o):
            continue
        found = True
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            for i in range(3):
                mins[i] = min(mins[i], w[i])
                maxs[i] = max(maxs[i], w[i])
    if not found:
        return Vector((0, 0, 0)), 1.0
    center = Vector([(mins[i] + maxs[i]) / 2 for i in range(3)])
    radius = max((maxs[i] - mins[i]) for i in range(3)) or 1.0
    return center, radius


def _scene_min_z() -> float:
    zmin = 1e9
    found = False
    for o in bpy.data.objects:
        if o.type != "MESH" or _is_ground(o):
            continue
        for c in o.bound_box:
            found = True
            zmin = min(zmin, (o.matrix_world @ Vector(c)).z)
    return zmin if found else 0.0


def _setup_review_rig() -> None:
    world = bpy.data.worlds.new("review_world")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = lighting.REVIEW_WORLD_COLOR
        bg.inputs[1].default_value = lighting.REVIEW_WORLD_STRENGTH
    bpy.context.scene.world = world


def _refresh_review_lights() -> None:

    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT" and obj.name.startswith(lighting.REVIEW_LIGHT_PREFIX):
            bpy.data.objects.remove(obj, do_unlink=True)
    for data in list(bpy.data.lights):
        if data.users == 0 and data.name.startswith(lighting.REVIEW_LIGHT_PREFIX):
            bpy.data.lights.remove(data)

    center, radius = _scene_bounds()
    specs = lighting.review_area_light_specs(tuple(center), radius)
    for spec in specs:
        bpy.ops.object.light_add(type="AREA", location=spec.location)
        light = bpy.context.active_object
        light.name = spec.name
        light.data.name = f"{spec.name}_data"
        light.data.energy = spec.energy
        light.data.shape = "DISK"
        light.data.size = spec.size
        direction = (center - light.location).normalized()
        light.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _place_camera(view: str, free=None, focus=None):
    center, radius = focus if focus is not None else _scene_bounds()
    if free is not None:
        d, dist_mult = free
        d = d.normalized()
    else:
        base = view.split("_")[0]
        d = _VIEW_DIRS.get(base, _VIEW_DIRS["perspective"]).normalized()
        dist_mult = 3.0
    cam_data = bpy.data.cameras.new(f"cam_{view}")
    cam = bpy.data.objects.new(f"cam_{view}", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = center + d * (radius * dist_mult)
    direction = (center - cam.location).normalized()
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    return cam


def _configure_engine() -> None:
    global _DEVICE_INFO
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    _DEVICE_INFO = configure_cycles_device(bpy, RenderDevicePolicy.from_environment())
    sc.render.resolution_x = _REVIEW_RESOLUTION
    sc.render.resolution_y = _REVIEW_RESOLUTION


def _configure_review_world() -> None:
    sc = bpy.context.scene
    sc.cycles.samples = 24
    try:
        sc.view_settings.view_transform = lighting.REVIEW_VIEW_TRANSFORM
        sc.view_settings.look = "None"
        sc.view_settings.exposure = 0.0
        sc.view_settings.gamma = 1.0
    except (AttributeError, TypeError):
        pass
    _setup_review_rig()


def _configure_showcase_world() -> None:
    sc = bpy.context.scene
    sc.cycles.samples = 48
    try:
        sc.cycles.use_denoising = True
    except (AttributeError, TypeError):
        pass
    try:
        sc.view_settings.view_transform = "AgX"
    except (AttributeError, TypeError):
        pass
    world = bpy.data.worlds.new("showcase_world")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.70, 0.72, 0.76, 1.0)
        bg.inputs[1].default_value = 1.4
    sc.world = world

    center, radius = _scene_bounds()
    r2 = radius * radius

    def _area(offset, energy_per_r2, size_mult):
        loc = center + Vector(offset)
        bpy.ops.object.light_add(type="AREA", location=loc)
        lt = bpy.context.active_object
        lt.data.energy = energy_per_r2 * r2
        lt.data.size = max(radius * size_mult, 1e-3)
        d = (center - lt.location).normalized()
        lt.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        return lt

    r = radius
    _area((2.0 * r, -2.0 * r, 3.0 * r), 220.0, 2.0)
    _area((-3.0 * r, -1.0 * r, 1.5 * r), 90.0, 3.0)
    _area((0.0, 3.0 * r, 2.5 * r), 130.0, 2.0)

    zmin = _scene_min_z()
    bpy.ops.mesh.primitive_plane_add(size=radius * 30.0, location=(center.x, center.y, zmin))
    plane = bpy.context.active_object
    plane.name = "showcase_ground"
    try:
        plane.is_shadow_catcher = True
    except (AttributeError, TypeError):
        pass


def _configure_scene(mode: str = "review") -> None:
    global _RENDER_MODE
    _RENDER_MODE = mode
    _configure_engine()
    if mode == "showcase":
        _configure_showcase_world()
    else:
        _configure_review_world()


def _clear_imported() -> None:
    for obj in list(bpy.data.objects):
        if obj.type in {"MESH", "CAMERA", "EMPTY"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    for coll in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.cameras,
        bpy.data.node_groups,
    ):
        for blk in list(coll):
            if blk.users == 0:
                coll.remove(blk)


def _render_views_only(views: list[str], out_dir: Path, camera_spec: str = "") -> dict:
    global _VIEW_TIMINGS
    _VIEW_TIMINGS = []
    out_dir.mkdir(parents=True, exist_ok=True)
    if _RENDER_MODE == "review":
        _refresh_review_lights()
    sc = bpy.context.scene
    free = None
    if camera_spec:
        try:
            az, el, dist = (float(x) for x in camera_spec.split(","))
            free = (_free_dir(az, el), dist)
        except (ValueError, TypeError):
            free = None
    focus = _FOCUS_BOUNDS
    prev_res = None
    if focus is not None:
        prev_res = (sc.render.resolution_x, sc.render.resolution_y)
        sc.render.resolution_x = _FOCUS_RESOLUTION
        sc.render.resolution_y = _FOCUS_RESOLUTION
    out: dict[str, str] = {}
    try:
        for view in views:
            started = time.monotonic()
            _place_camera(view, free=free, focus=focus)
            path = out_dir / f"{view}.png"
            sc.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            _VIEW_TIMINGS.append({"view": view, "wall_s": time.monotonic() - started})
            out[view] = str(path)
    finally:
        if prev_res is not None:
            sc.render.resolution_x, sc.render.resolution_y = prev_res
    return out


def _render(views: list[str], out_dir: Path, camera_spec: str = "", mode: str = "review") -> dict:
    _configure_scene(mode)
    return _render_views_only(views, out_dir, camera_spec)


def main() -> None:
    args = _parse_args()
    report = {"glb": args.glb, "renders": {}, "errors": []}
    try:
        request = json.loads(Path(args.request_file).read_text()) if args.request_file else {}
        expected = request.get("input_identity") or input_identity(Path(args.glb))
        report.update(request_id=request.get("request_id"), input_identity=expected)
        verify_input(Path(args.glb), expected)
        _reset_scene()
        started = time.monotonic()
        _import_glb(args.glb)
        report["import_wall_s"] = time.monotonic() - started
        _override_focus_bounds(args.focus_bounds)
        views = [v for v in args.views.split(",") if v]
        report["renders"] = _render(
            views, Path(args.out_dir), camera_spec=args.camera, mode=args.mode
        )
        verify_input(Path(args.glb), expected)
    except Exception as exc:  # noqa: BLE001
        report["error_kind"] = type(exc).__name__
        report["errors"].append(traceback.format_exc())
    report["device"] = _DEVICE_INFO
    report["view_timings"] = _VIEW_TIMINGS
    write_report(Path(args.report), report)


if __name__ == "__main__":
    main()
