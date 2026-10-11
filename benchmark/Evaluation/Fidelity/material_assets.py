from __future__ import annotations

import re
from pathlib import Path

import bpy
import numpy as np


def mesh_key(name: str) -> str:
    parts = name.split("::")
    if len(parts) >= 3 and parts[0] == "visual":
        name = f"{parts[1]}_{parts[2]}"
    return re.sub("_+", "_", re.sub("[^a-z0-9]+", "_", name.lower())).strip("_")


def import_pose(native: Path | None, pose: Path) -> None:
    if native is None or native.resolve() == pose.resolve():
        bpy.ops.import_scene.gltf(filepath=str(pose))
        return
    bpy.ops.import_scene.gltf(filepath=str(native))
    sources = {}
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        key = mesh_key(obj.name)
        if key in sources:
            raise ValueError(f"Duplicate native material mesh key: {key}")
        mesh = obj.data
        indices = np.empty(len(mesh.polygons), dtype=np.int32)
        mesh.polygons.foreach_get("material_index", indices)
        layers = []
        for layer in mesh.uv_layers:
            values = np.empty(len(layer.data) * 2, dtype=np.float32)
            layer.data.foreach_get("uv", values)
            layers.append((layer.name, values))
        sources[key] = (
            [slot.material for slot in obj.material_slots],
            indices,
            layers,
            len(mesh.polygons),
            len(mesh.loops),
        )
    if not sources:
        raise ValueError("Native GLB has no material meshes")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(pose))
    matched = set()
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        key = mesh_key(obj.name)
        if key in matched:
            raise ValueError(f"Duplicate pose material mesh key: {key}")
        if key not in sources:
            raise ValueError(f"Pose mesh has no native material source: {key}")
        materials, indices, layers, polygons, loops = sources[key]
        mesh = obj.data
        if (len(mesh.polygons), len(mesh.loops)) != (polygons, loops):
            raise ValueError(f"Native and pose mesh topology differs: {key}")
        if any(len(values) != loops * 2 for _, values in layers):
            raise ValueError(f"Native UV loop count differs: {key}")
        mesh.materials.clear()
        for material in materials:
            mesh.materials.append(material)
        mesh.polygons.foreach_set("material_index", indices)
        while mesh.uv_layers:
            mesh.uv_layers.remove(mesh.uv_layers[0])
        for name, values in layers:
            layer = mesh.uv_layers.new(name=name)
            layer.data.foreach_set("uv", values)
        mesh.update()
        matched.add(key)
    if matched != sources.keys():
        raise ValueError(f"Native meshes missing from pose: {sorted(sources.keys() - matched)}")
