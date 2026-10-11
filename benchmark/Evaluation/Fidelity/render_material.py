from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from mathutils import Vector

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).absolute().parent))

from render_blender import VIEWS, add_camera, camera_spec, import_model, reset_scene  # noqa: E402

CHANNELS = {"base_color": "Base Color", "metallic": "Metallic", "roughness": "Roughness"}
RESOLUTION = 512


def setup_scene(threads: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 0
    scene.cycles.diffuse_bounces = 0
    scene.cycles.glossy_bounces = 0
    scene.cycles.transmission_bounces = 0
    scene.cycles.transparent_max_bounces = 0
    scene.render.threads_mode = "FIXED"
    scene.render.threads = threads
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.exr_codec = "ZIP"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.color = (0.0, 0.0, 0.0)


def attribute_material(source: Any, channel: str) -> Any:
    material = source.copy() if source else bpy.data.materials.new(channel)
    material.use_nodes = True
    tree = material.node_tree
    outputs = [
        node for node in tree.nodes if node.type == "OUTPUT_MATERIAL" and node.is_active_output
    ]
    output = outputs[0] if outputs else tree.nodes.new("ShaderNodeOutputMaterial")
    surface = output.inputs["Surface"]
    principled = surface.links[0].from_node if surface.is_linked else None
    if principled is None or principled.type != "BSDF_PRINCIPLED":
        principled = next((node for node in tree.nodes if node.type == "BSDF_PRINCIPLED"), None)
    if principled is None:
        principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
        principled.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        principled.inputs["Metallic"].default_value = 0.0
        principled.inputs["Roughness"].default_value = 0.5
    for link in list(surface.links):
        tree.links.remove(link)
    output.is_active_output = True
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    socket = principled.inputs[CHANNELS[channel]]
    color = emission.inputs["Color"]
    if socket.is_linked:
        tree.links.new(socket.links[0].from_socket, color)
    elif channel == "base_color":
        color.default_value = socket.default_value
    else:
        value = float(socket.default_value)
        color.default_value = (value, value, value, 1.0)
    tree.links.new(emission.outputs["Emission"], surface)
    return material


def material_variants() -> tuple[list[tuple[Any, list[Any]]], dict[str, dict[Any, Any]]]:
    meshes = {obj.data for obj in bpy.context.scene.objects if obj.type == "MESH"}
    if not meshes:
        raise ValueError("GLB has no mesh objects")
    originals = []
    for mesh in meshes:
        if not mesh.materials:
            mesh.materials.append(None)
        originals.append((mesh, list(mesh.materials)))
    sources = {material for _, materials in originals for material in materials}
    return originals, {
        channel: {source: attribute_material(source, channel) for source in sources}
        for channel in CHANNELS
    }


def render_buffers(output: Path, role: str, camera: dict[str, Any]) -> None:
    center, position = Vector(camera["center"]), Vector(camera["position"])
    if (
        len(center) != 3
        or len(position) != 3
        or not all(math.isfinite(v) for v in (*center, *position))
    ):
        raise ValueError("Camera vectors must have three finite coordinates")
    if (position - center).length < 1e-8:
        raise ValueError("Camera position must differ from its center")
    if camera.get("projection", "perspective") != "perspective":
        raise ValueError("Material evaluation requires a perspective camera")
    add_camera(center, position)
    data = bpy.context.scene.camera.data
    data.lens = float(camera.get("lens_mm", 24.0))
    data.sensor_width = float(camera.get("sensor_width_mm", 20.955))
    originals, variants = material_variants()
    buffers = {}
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output, ignore_cleanup_errors=True) as temporary:
        for channel in CHANNELS:
            for mesh, materials in originals:
                for index, source in enumerate(materials):
                    mesh.materials[index] = variants[channel][source]
            path = Path(temporary) / f"{channel}.exr"
            bpy.context.scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            image = bpy.data.images.load(str(path), check_existing=False)
            try:
                pixels = np.asarray(image.pixels[:], dtype=np.float32)
                if pixels.size != RESOLUTION * RESOLUTION * 4:
                    raise ValueError("Unexpected material buffer dimensions")
                rgba = np.flipud(pixels.reshape(RESOLUTION, RESOLUTION, 4)).copy()
            finally:
                bpy.data.images.remove(image)
            buffers[channel] = rgba[..., :3] if channel == "base_color" else rgba[..., 0]
            if channel == "base_color":
                buffers["mask"] = rgba[..., 3].copy()
    np.savez_compressed(output / f"{role}_buffers.npz", **buffers)


def render(config: dict[str, Any]) -> dict[str, Any]:
    expected = {"gt_close", "gt_open", "pred_close", "pred_open"}
    if set(config["models"]) != expected:
        raise ValueError(f"models must contain exactly {sorted(expected)}")
    threads = config.get("threads", 1)
    if not isinstance(threads, int) or threads <= 0:
        raise ValueError("threads must be a positive integer")
    views = config.get("views", list(VIEWS))
    if not views or len(set(views)) != len(views) or set(views) - set(VIEWS):
        raise ValueError(f"views must select distinct names from {VIEWS}")
    models = config["models"]
    config = dict(config)
    config["models"] = {**config["models"], **config.get("raw_gt_models", {})}
    explicit = config.get("cameras")
    if explicit is None:
        center, positions, _ = camera_spec(config | {"views": views, "models": models})
        raw_camera = config.get("camera")
        if raw_camera:
            raw_camera = {key: value for key, value in raw_camera.items() if key != "normalization"}
        gt_center, gt_positions, _ = camera_spec(config | {"views": views, "camera": raw_camera})
    output = Path(config["output"]).absolute()
    correspondence = config.get("correspondence")
    if config.get("transfer"):
        from material_transfer import refine_correspondence, transfer_materials

        if correspondence is None:
            correspondence = {
                "pred_part_names": config["pred_part_names"],
                "gt_part_names": config["gt_part_names"],
            }
        correspondence["link_to_part"] = config.get("link_to_part")
        if "pairs" not in correspondence:
            correspondence = refine_correspondence(
                config["models"],
                correspondence["pred_part_names"],
                correspondence["gt_part_names"],
                config.get("link_to_part"),
            )
    cameras: dict[str, dict[str, Any]] = {}
    for state in ("close", "open"):
        cameras[state] = {}
        for view in views:
            cameras[state][view] = {}
            for role in ("pred", "gt"):
                camera_role = "gt" if config.get("transfer") else role
                camera = (
                    explicit[state][view][camera_role]
                    if explicit
                    else {
                        "center": list(gt_center if camera_role == "gt" else center),
                        "position": list(
                            gt_positions[view] if camera_role == "gt" else positions[view]
                        ),
                        "projection": "perspective",
                        "lens_mm": 24.0,
                        "sensor_width_mm": 20.955,
                    }
                )
                reset_scene()
                if config.get("transfer") and role == "pred":
                    transfer_materials(
                        Path(config["models"][f"pred_{state}"]),
                        Path(config["models"][f"gt_{state}"]),
                        correspondence,
                        Path(config["native_glb"]) if config.get("native_glb") else None,
                    )
                elif role == "pred" and config.get("native_glb"):
                    from material_assets import import_pose

                    import_pose(Path(config["native_glb"]), Path(config["models"][f"pred_{state}"]))
                else:
                    import_model(config["models"][f"{role}_{state}"])
                setup_scene(threads)
                render_buffers(output / state / view, role, camera)
                cameras[state][view][role] = camera
    result = {
        "cameras": cameras,
        "resolution": RESOLUTION,
        "blender_version": bpy.app.version_string,
    }
    if config.get("transfer"):
        result["correspondence"] = correspondence
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
