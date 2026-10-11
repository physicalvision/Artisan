from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Vector

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).absolute().parent))
from protocol_config import assert_blender_version  # noqa: E402

assert_blender_version(bpy)

runtime = (
    Path(os.environ.get("ARTISAN_RUNTIME_ROOT", str(Path(__file__).absolute().parents[3])))
    / "sdk/runtime"
)
if not runtime.is_dir():
    import artisan

    runtime = Path(artisan.__file__).parent.parent / "sdk/runtime"
sys.path.insert(0, str(runtime))

VIEWS = ("front", "iso", "top")


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_model(path: str) -> None:
    bpy.ops.import_scene.gltf(filepath=str(Path(path).absolute()))
    for obj in bpy.context.scene.objects:
        if obj.type in {"LIGHT", "CAMERA"}:
            obj.hide_render = True


def scene_corners() -> list[Vector]:
    corners = [
        obj.matrix_world @ Vector(corner)
        for obj in bpy.context.scene.objects
        if obj.type == "MESH"
        for corner in obj.bound_box
    ]
    if not corners or not all(math.isfinite(v) for corner in corners for v in corner):
        raise ValueError("GLB has no finite mesh bounds")
    return corners


def look_at(obj: Any, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def view_position(center: Vector, front: Vector, view: str) -> Vector:
    direction = front - center
    if view == "front":
        return front
    if view == "iso":
        return center + Vector((-direction.x, direction.y, direction.z))
    horizontal = Vector((direction.x, direction.y, 0.0))
    if horizontal.length < 1e-8:
        horizontal = Vector((0.0, -1.0, 0.0))
    horizontal.normalize()
    elevation = math.radians(68.0)
    radius = direction.length
    return (
        center
        + horizontal * radius * math.cos(elevation)
        + Vector((0.0, 0.0, radius * math.sin(elevation)))
    )


def add_camera(center: Vector, position: Vector) -> None:
    data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", data)
    bpy.context.scene.collection.objects.link(camera)
    camera.location = position
    look_at(camera, center)
    data.type = "PERSP"
    data.lens = 24.0
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = 20.955
    data.clip_start = 0.001
    data.clip_end = 100000.0
    bpy.context.scene.camera = camera


def setup_scene(config: dict[str, Any], center: Vector) -> dict[str, Any]:
    from _render_device import RenderDevicePolicy, configure_cycles_device

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    policy = RenderDevicePolicy.from_environment({"ARTISAN_RENDER_DEVICE": "gpu", **os.environ})
    if policy.mode == "auto":
        policy = RenderDevicePolicy(mode="gpu", backend=policy.backend, gpu_index=policy.gpu_index)
    try:
        device = configure_cycles_device(bpy, policy)
    except (RuntimeError, TypeError) as exc:
        raise RuntimeError(
            f"Fidelity {policy.backend} rendering is unavailable: {exc}. "
            "Set ARTISAN_RENDER_DEVICE=cpu to opt into CPU rendering."
        ) from exc
    scene.cycles.samples = config["samples"]
    scene.cycles.seed = config["seed"]
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 6
    scene.render.threads_mode = "FIXED"
    scene.render.threads = config["threads"]
    scene.render.resolution_x = config["resolution"]
    scene.render.resolution_y = config["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.82, 0.81, 0.82, 1.0)
    background.inputs["Strength"].default_value = 1.0
    data = bpy.data.lights.new("Sun", "SUN")
    light = bpy.data.objects.new("Sun", data)
    scene.collection.objects.link(light)
    light.location = center + Vector((1.0, -1.0, 2.0))
    light.rotation_euler = tuple(math.radians(value) for value in (-45.0, 0.0, 35.0))
    data.energy = 1.5
    data.angle = 0.08
    return device


def setup_coverage() -> None:
    material = bpy.data.materials.new("Coverage")
    material.use_nodes = True
    material.node_tree.nodes.clear()
    emission = material.node_tree.nodes.new("ShaderNodeEmission")
    emission.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    output = material.node_tree.nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(emission.outputs[0], output.inputs[0])
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.data.materials.clear()
            obj.data.materials.append(material)
    scene = bpy.context.scene
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = "RGBA"


def camera_spec(config: dict[str, Any]) -> tuple[Vector, dict[str, Vector], str]:
    camera = config.get("camera")
    if camera is None:
        reset_scene()
        import_model(config["models"]["gt_close"])
        corners = scene_corners()
        lower = Vector([min(corner[i] for corner in corners) for i in range(3)])
        upper = Vector([max(corner[i] for corner in corners) for i in range(3)])
        center = (lower + upper) / 2
        diagonal = (upper - lower).length
        front = center + Vector((0.4, -0.8, 0.32)).normalized() * diagonal * 1.05
        mask_mode = "geometry"
    else:
        center, front = Vector(camera["center"]), Vector(camera["position"])
        if "normalization" in camera:
            norm = camera["normalization"]
            x, y, z = norm["center"]
            origin = Vector((x, -z, y))
            center = (center - origin) * norm["scale"]
            front = (front - origin) * norm["scale"]
        mask_mode = "geometry"
    if len(center) != 3 or len(front) != 3 or not all(math.isfinite(v) for v in (*center, *front)):
        raise ValueError("Camera vectors must have three finite coordinates")
    if (front - center).length < 1e-8:
        raise ValueError("Invalid camera position")
    positions = {view: view_position(center, front, view) for view in config["views"]}
    return center, positions, mask_mode


def render(config: dict[str, Any]) -> dict[str, Any]:
    expected = {"gt_close", "gt_open", "pred_close", "pred_open"}
    if set(config["models"]) != expected:
        raise ValueError(f"models must contain exactly {sorted(expected)}")
    for name in ("resolution", "samples", "threads"):
        if not isinstance(config[name], int) or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config["resolution"] < 8:
        raise ValueError("resolution must be at least 8")
    views = config["views"]
    if not views or len(set(views)) != len(views) or set(views) - set(VIEWS):
        raise ValueError(f"views must select distinct names from {VIEWS}")
    center, positions, mask_mode = camera_spec(config)
    output = Path(config["output"]).absolute()
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, list[str]] = {name: [] for name in sorted(expected)}
    masks: dict[str, list[str]] = {name: [] for name in sorted(expected)}
    devices = {}
    for name in sorted(expected):
        reset_scene()
        raw_models = config.get("raw_gt_models", {}) if config.get("camera") else {}
        model = raw_models.get(name, config["models"][name])
        import_model(model)
        object_center, object_positions = center, positions
        if name in raw_models:
            raw_camera = {
                key: value for key, value in config["camera"].items() if key != "normalization"
            }
            object_center, object_positions, _ = camera_spec({**config, "camera": raw_camera})
        devices[name] = setup_scene(config, object_center)
        for view in views:
            add_camera(object_center, object_positions[view])
            path = output / f"{name}_{view}.png"
            bpy.context.scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            paths[name].append(str(path))
        setup_coverage()
        for view in views:
            add_camera(object_center, object_positions[view])
            path = output / f"{name}_{view}_mask.png"
            bpy.context.scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            masks[name].append(str(path))
    result = {
        "images": paths,
        "masks": masks,
        "camera": {
            "center": list(center),
            "positions": {v: list(p) for v, p in positions.items()},
            "focal_mm": 24.0,
            "sensor_width_mm": 20.955,
        },
        "mask_mode": mask_mode,
        "resolution": config["resolution"],
        "samples": config["samples"],
        "seed": config["seed"],
        "devices": devices,
        "blender_version": bpy.app.version_string,
    }
    (output / "render_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    render(json.loads(args.manifest.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
