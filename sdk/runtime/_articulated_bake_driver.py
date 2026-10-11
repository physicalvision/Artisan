from __future__ import annotations

import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import bpy  # type: ignore[import-not-found]


def _load_spec() -> tuple[dict[str, Any], Path]:
    args = sys.argv[sys.argv.index("--") + 1 :]
    if len(args) != 1:
        raise ValueError("expected one bake-spec path after --")
    spec_path = Path(args[0])
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "artisan-articulated-bake-spec-v1":
        raise ValueError("unsupported articulated bake spec")
    return payload, Path(payload["result_json"])


def _principled(material: Any) -> Any:
    if not material.use_nodes:
        material.use_nodes = True
    node = next(
        (
            candidate
            for candidate in material.node_tree.nodes
            if candidate.type == "BSDF_PRINCIPLED"
        ),
        None,
    )
    if node is None:
        node = material.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
    return node


def _material_output(material: Any) -> Any:
    node = next(
        (
            candidate
            for candidate in material.node_tree.nodes
            if candidate.type == "OUTPUT_MATERIAL"
        ),
        None,
    )
    if node is None:
        node = material.node_tree.nodes.new("ShaderNodeOutputMaterial")
    return node


def _set_socket(inputs: Any, names: tuple[str, ...], value: Any) -> None:
    for name in names:
        if name in inputs:
            inputs[name].default_value = value
            return


def _load_image(path: Path, *, non_color: bool) -> Any:
    image = bpy.data.images.load(str(path), check_existing=True)
    if non_color:
        image.colorspace_settings.name = "Non-Color"
    return image


def _texture_node(
    material: Any,
    mapping: Any,
    path: Path,
    *,
    non_color: bool,
) -> Any:
    tree = material.node_tree
    node = tree.nodes.new("ShaderNodeTexImage")
    node.image = _load_image(path, non_color=non_color)
    node.projection = "BOX"
    node.projection_blend = 0.3
    tree.links.new(mapping.outputs["Vector"], node.inputs["Vector"])
    return node


def _apply_source_texture(
    material: Any,
    spec: dict[str, Any],
    scale: float,
    reference_object: Any,
) -> dict[str, Any]:
    tree = material.node_tree
    bsdf = _principled(material)
    output = _material_output(material)
    if not output.inputs["Surface"].is_linked:
        tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    coord = tree.nodes.new("ShaderNodeTexCoord")
    coord.object = reference_object
    mapping = tree.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (scale, scale, scale)
    tree.links.new(coord.outputs["Object"], mapping.inputs["Vector"])
    maps = {name: Path(path) for name, path in dict(spec.get("maps") or {}).items()}
    sockets: dict[str, Any] = {}
    color = _texture_node(material, mapping, maps["color"], non_color=False)
    tree.links.new(color.outputs["Color"], bsdf.inputs["Base Color"])
    sockets["base_color"] = color.outputs["Color"]
    if "roughness" in maps:
        roughness = _texture_node(material, mapping, maps["roughness"], non_color=True)
        tree.links.new(roughness.outputs["Color"], bsdf.inputs["Roughness"])
        sockets["roughness"] = roughness.outputs["Color"]
    if "metalness" in maps:
        metalness = _texture_node(material, mapping, maps["metalness"], non_color=True)
        tree.links.new(metalness.outputs["Color"], bsdf.inputs["Metallic"])
        sockets["metallic"] = metalness.outputs["Color"]
    if "normal" in maps:
        normal = _texture_node(material, mapping, maps["normal"], non_color=True)
        normal_map = tree.nodes.new("ShaderNodeNormalMap")
        tree.links.new(normal.outputs["Color"], normal_map.inputs["Color"])
        tree.links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
        sockets["normal"] = normal.outputs["Color"]
    return sockets


def _object_area(obj: Any) -> float:
    mesh = obj.data
    mesh.calc_loop_triangles()
    matrix = obj.matrix_world
    total = 0.0
    for triangle in mesh.loop_triangles:
        a, b, c = (matrix @ mesh.vertices[index].co for index in triangle.vertices)
        total += 0.5 * (b - a).cross(c - a).length
    return max(float(total), 1e-12)


def _next_power_of_two(value: float) -> int:
    return 1 << max(0, math.ceil(math.log2(max(1.0, value))))


def _atlas_groups(objects: list[Any], material_names: set[str]) -> list[list[Any]]:
    parent = {obj.name: obj.name for obj in objects}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    anchors: dict[str, str] = {}
    for obj in objects:
        for slot in obj.material_slots:
            if slot.material is None or slot.material.name not in material_names:
                continue
            union(anchors.setdefault(slot.material.name, obj.name), obj.name)
    groups: dict[str, list[Any]] = {}
    for obj in objects:
        groups.setdefault(find(obj.name), []).append(obj)
    return [groups[key] for key in sorted(groups)]


def _partition_pages(
    objects: list[Any],
    *,
    material_names: set[str],
    texels_per_meter: int,
    min_size: int,
    max_size: int,
    efficiency: float,
) -> tuple[int, list[list[Any]], dict[str, float]]:
    areas = {obj.name: _object_area(obj) for obj in objects}
    total_area = sum(areas.values())
    required_side = math.sqrt(total_area / efficiency) * texels_per_meter
    atlas_size = min(max_size, max(min_size, _next_power_of_two(required_side)))
    page_capacity = max(efficiency * (atlas_size / texels_per_meter) ** 2, 1e-12)
    page_count = max(1, math.ceil(total_area / page_capacity))
    pages: list[list[Any]] = [[] for _ in range(page_count)]
    loads = [0.0] * page_count
    groups = _atlas_groups(objects, material_names)
    group_areas = {
        index: sum(areas[obj.name] for obj in group) for index, group in enumerate(groups)
    }
    order = sorted(
        range(len(groups)), key=lambda index: (-group_areas[index], groups[index][0].name)
    )
    for index in order:
        page_index = min(range(page_count), key=lambda page: (loads[page], page))
        pages[page_index].extend(groups[index])
        loads[page_index] += group_areas[index]
    pages = [page for page in pages if page]
    return atlas_size, pages, areas


def _select(objects: list[Any]) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]


def _unwrap_page(objects: list[Any], margin: float) -> None:
    for obj in objects:
        uv = obj.data.uv_layers.new(name="ArtisanBakeUV")
        obj.data.uv_layers.active = uv
        uv.active_render = True
    _select(objects)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(
        angle_limit=1.151917,
        island_margin=0.0,
        area_weight=0.0,
        correct_aspect=True,
        scale_to_bounds=False,
    )
    bpy.ops.uv.pack_islands(
        rotate=True,
        scale=True,
        merge_overlap=False,
        margin_method="ADD",
        margin=margin,
    )
    bpy.ops.object.mode_set(mode="OBJECT")


def _joined_bake_copy(objects: list[Any], page_index: int) -> Any:
    duplicates: list[Any] = []
    for obj in objects:
        duplicate = obj.copy()
        duplicate.data = obj.data.copy()
        bpy.context.scene.collection.objects.link(duplicate)
        world = obj.matrix_world.copy()
        duplicate.parent = None
        duplicate.matrix_world = world
        duplicates.append(duplicate)
    _select(duplicates)
    bpy.context.view_layer.objects.active = duplicates[0]
    bpy.ops.object.join()
    joined = bpy.context.view_layer.objects.active
    joined.name = f"__artisan_bake_page_{page_index}__"
    return joined


def _new_image(
    name: str,
    path: Path,
    size: int,
    *,
    color_space: str,
    background: tuple[float, float, float, float],
) -> Any:
    image = bpy.data.images.new(name, width=size, height=size, alpha=True)
    image.generated_color = background
    image.colorspace_settings.name = color_space
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    return image


def _set_bake_targets(materials: list[Any], image: Any) -> None:
    for material in materials:
        nodes = material.node_tree.nodes
        target = nodes.get("ArtisanBakeTarget")
        if target is None:
            target = nodes.new("ShaderNodeTexImage")
            target.name = "ArtisanBakeTarget"
        target.image = image
        target.select = True
        nodes.active = target


def _bake_channel(
    objects: list[Any],
    material_sources: dict[str, dict[str, Any]],
    channel: str,
    image: Any,
    padding: int,
) -> bool:
    selected = [
        obj
        for obj in objects
        if any(
            slot.material is not None and channel in material_sources.get(slot.material.name, {})
            for slot in obj.material_slots
        )
    ]
    if not selected:
        return False
    materials = sorted(
        {
            slot.material
            for obj in selected
            for slot in obj.material_slots
            if slot.material is not None and channel in material_sources.get(slot.material.name, {})
        },
        key=lambda material: material.name,
    )
    restore: list[tuple[Any, Any, Any, Any]] = []
    for material in materials:
        tree = material.node_tree
        output = _material_output(material)
        if not output.inputs["Surface"].is_linked:
            raise ValueError(f"material {material.name!r} has no surface shader")
        old_socket = output.inputs["Surface"].links[0].from_socket
        emission = tree.nodes.new("ShaderNodeEmission")
        tree.links.new(material_sources[material.name][channel], emission.inputs["Color"])
        tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
        restore.append((tree, output, old_socket, emission))
    _set_bake_targets(materials, image)
    _select(selected)
    bpy.ops.object.bake(type="EMIT", margin=padding, use_clear=False)
    for tree, output, old_socket, emission in restore:
        tree.links.new(old_socket, output.inputs["Surface"])
        tree.nodes.remove(emission)
    image.save()
    return True


def _image_node(material: Any, image: Any, *, non_color: bool) -> Any:
    node = material.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = image
    if non_color:
        node.image.colorspace_settings.name = "Non-Color"
    return node


def _rebuild_material(
    material: Any,
    source_spec: dict[str, Any],
    channels: set[str],
    images: dict[str, Any],
) -> None:
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
    tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    base = list(source_spec.get("base_color_factor") or [0.8, 0.8, 0.8, 1.0])
    _set_socket(bsdf.inputs, ("Base Color",), (*base[:3], 1.0))
    _set_socket(bsdf.inputs, ("Alpha",), float(base[3] if len(base) > 3 else 1.0))
    _set_socket(bsdf.inputs, ("Metallic",), float(source_spec.get("metallic_factor", 0.0)))
    _set_socket(bsdf.inputs, ("Roughness",), float(source_spec.get("roughness_factor", 0.8)))
    if "base_color" in channels:
        image = _image_node(material, images["base_color"], non_color=False)
        tree.links.new(image.outputs["Color"], bsdf.inputs["Base Color"])
    if "roughness" in channels:
        image = _image_node(material, images["roughness"], non_color=True)
        tree.links.new(image.outputs["Color"], bsdf.inputs["Roughness"])
    if "metallic" in channels:
        image = _image_node(material, images["metallic"], non_color=True)
        tree.links.new(image.outputs["Color"], bsdf.inputs["Metallic"])
    if "normal" in channels:
        image = _image_node(material, images["normal"], non_color=True)
        normal_map = tree.nodes.new("ShaderNodeNormalMap")
        normal_map.space = "TANGENT"
        tree.links.new(image.outputs["Color"], normal_map.inputs["Color"])
        tree.links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
    transmission = source_spec.get("transmission")
    if transmission is not None:
        _set_socket(
            bsdf.inputs,
            ("Transmission Weight", "Transmission"),
            float(transmission),
        )
        _set_socket(bsdf.inputs, ("Alpha",), 1.0)
    if source_spec.get("ior") is not None:
        _set_socket(bsdf.inputs, ("IOR",), float(source_spec["ior"]))
    emission = float(source_spec.get("emissive_strength", 0.0))
    if emission > 0.0:
        _set_socket(bsdf.inputs, ("Emission Color", "Emission"), (*base[:3], 1.0))
        _set_socket(bsdf.inputs, ("Emission Strength",), emission)


def _export_glb(path: Path) -> None:
    bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        export_yup=True,
        export_extras=True,
        export_tangents=True,
        export_texcoords=True,
        export_normals=True,
        export_cameras=False,
        export_lights=False,
        export_animations=False,
        export_materials="EXPORT",
        export_image_format="AUTO",
        export_unused_images=False,
        export_unused_textures=False,
    )


def _run(spec: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    source_glb = Path(spec["source_glb"])
    output_glb = Path(spec["output_glb"])
    atlas_dir = Path(spec["atlas_dir"])
    policy = dict(spec["policy"])
    material_specs = dict(spec["materials"])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source_glb))
    materials: dict[str, Any] = {}
    for name, material_spec in sorted(material_specs.items()):
        material = bpy.data.materials.get(name)
        if material is None:
            raise ValueError(f"articulated GLB is missing material {name!r}")
        materials[name] = material
    objects = [
        obj
        for obj in bpy.data.objects
        if obj.type == "MESH"
        and any(
            slot.material is not None and slot.material.name in materials
            for slot in obj.material_slots
        )
    ]
    if not objects:
        raise ValueError("no textured visual objects matched the bake specification")
    objects.sort(key=lambda obj: obj.name)
    material_users: dict[str, list[Any]] = {name: [] for name in materials}
    for obj in objects:
        for slot in obj.material_slots:
            if slot.material is not None and slot.material.name in material_users:
                material_users[slot.material.name].append(obj)
    reference_objects: list[Any] = []
    material_sources: dict[str, dict[str, Any]] = {}
    material_user_counts: dict[str, int] = {}
    for name, material in sorted(materials.items()):
        users = material_users[name]
        if not users:
            raise ValueError(f"textured material {name!r} is not used by any visual")
        material_user_counts[name] = len(users)
        reference = bpy.data.objects.new(f"__artisan_texcoord_{name}__", None)
        bpy.context.scene.collection.objects.link(reference)
        reference.matrix_world = users[0].matrix_world.copy()
        reference_objects.append(reference)
        material_sources[name] = _apply_source_texture(
            material,
            material_specs[name],
            float(policy["projection_scale"]),
            reference,
        )
    requested_density = int(policy["texels_per_meter"])
    fallback_density = int(policy["fallback_texels_per_meter"])
    minimum_density = int(policy["minimum_texels_per_meter"])
    density_ladder: list[int] = []
    for density in (requested_density, fallback_density, minimum_density):
        if density not in density_ladder:
            density_ladder.append(density)
    atlas_size = 0
    pages: list[list[Any]] = []
    areas: dict[str, float] = {}
    used_density = requested_density
    for density in density_ladder:
        atlas_size, pages, areas = _partition_pages(
            objects,
            material_names=set(materials),
            texels_per_meter=density,
            min_size=int(policy["min_atlas_size"]),
            max_size=int(policy["max_atlas_size"]),
            efficiency=float(policy["packing_efficiency"]),
        )
        used_density = density
        if len(pages) <= 1:
            break
    used_preflight_fallback = used_density != requested_density
    padding = max(2, round(int(policy["padding_2k"]) * atlas_size / 2048))
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = "CPU"
    bpy.context.scene.cycles.samples = 1
    material_pages: dict[str, int] = {}
    baked_channels: dict[str, list[str]] = {}
    atlas_rows: list[dict[str, Any]] = []
    for page_index, page_objects in enumerate(pages):
        _unwrap_page(page_objects, padding / atlas_size)
        bake_object = _joined_bake_copy(page_objects, page_index)
        page_materials = {
            slot.material.name
            for obj in page_objects
            for slot in obj.material_slots
            if slot.material is not None and slot.material.name in materials
        }
        images = {
            "base_color": _new_image(
                f"ArtisanBaseColor{page_index}",
                atlas_dir / f"base_color_{page_index}.png",
                atlas_size,
                color_space="sRGB",
                background=(0.0, 0.0, 0.0, 1.0),
            ),
            "roughness": _new_image(
                f"ArtisanRoughness{page_index}",
                atlas_dir / f"roughness_{page_index}.png",
                atlas_size,
                color_space="Non-Color",
                background=(1.0, 1.0, 1.0, 1.0),
            ),
            "metallic": _new_image(
                f"ArtisanMetallic{page_index}",
                atlas_dir / f"metallic_{page_index}.png",
                atlas_size,
                color_space="Non-Color",
                background=(1.0, 1.0, 1.0, 1.0),
            ),
            "normal": _new_image(
                f"ArtisanNormal{page_index}",
                atlas_dir / f"normal_{page_index}.png",
                atlas_size,
                color_space="Non-Color",
                background=(0.5, 0.5, 1.0, 1.0),
            ),
        }
        page_channels: set[str] = set()
        for channel, image in images.items():
            if _bake_channel([bake_object], material_sources, channel, image, padding):
                page_channels.add(channel)
        for material_name in page_materials:
            if material_name in material_pages and material_pages[material_name] != page_index:
                raise ValueError(f"material {material_name!r} spans multiple atlas pages")
            material_pages[material_name] = page_index
            channels = set(material_sources[material_name]) & page_channels
            baked_channels[material_name] = sorted(channels)
            _rebuild_material(
                materials[material_name],
                material_specs[material_name],
                channels,
                images,
            )
        atlas_rows.append(
            {
                "page": page_index,
                "size": atlas_size,
                "padding": padding,
                "object_count": len(page_objects),
                "surface_area_m2": sum(areas[obj.name] for obj in page_objects),
                "channels": sorted(page_channels),
            }
        )
        bake_mesh = bake_object.data
        bpy.data.objects.remove(bake_object, do_unlink=True)
        if bake_mesh.users == 0:
            bpy.data.meshes.remove(bake_mesh)
    for reference in reference_objects:
        bpy.data.objects.remove(reference, do_unlink=True)
    _export_glb(output_glb)
    return {
        "status": "ok",
        "schema_version": "artisan-articulated-bake-result-v1",
        "policy_version": "articulated-pbr-bake-v1",
        "blender_version": bpy.app.version_string,
        "texture_mode": "hybrid-embedded",
        "requested_texels_per_meter": requested_density,
        "texels_per_meter": used_density,
        "used_preflight_fallback": used_preflight_fallback,
        "atlas_size": atlas_size,
        "atlas_pages": atlas_rows,
        "textured_material_count": len(materials),
        "textured_visual_count": len(objects),
        "textured_material_users": material_user_counts,
        "material_pages": material_pages,
        "baked_channels": baked_channels,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def main() -> None:
    result_path: Path | None = None
    try:
        spec, result_path = _load_spec()
        result = _run(spec)
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        if result_path is not None:
            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        traceback.print_exc()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
