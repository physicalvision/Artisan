from __future__ import annotations

import math
import re
from pathlib import Path

import bpy
import numpy as np
from material_assets import import_pose
from mathutils import Vector

SMALL_COMPONENT_AREA_FRACTION = 0.02


def transfer_materials(
    pred_path: Path, gt_path: Path, correspondence: dict, native_glb: Path | None = None
) -> None:
    identity = pred_path.resolve() == gt_path.resolve() and (
        native_glb is None or native_glb.resolve() == gt_path.resolve()
    )
    clear_scene()
    import_pose(native_glb, pred_path)
    donors, retained = collect_part_donors(
        correspondence["pred_part_names"], correspondence.get("link_to_part")
    )
    donors = {part: normalized_object_descriptors(items) for part, items in donors.items()}
    import_reference_geometry(gt_path)
    mapping = {
        int(pair["gt_part_id"]): int(pair["pred_part_id"]) for pair in correspondence["pairs"]
    }
    targets: dict[int, list[dict]] = {
        index: [] for index in range(len(correspondence["gt_part_names"]))
    }
    for obj in imported_meshes():
        part = part_index_for_object(obj.name, correspondence["gt_part_names"], "gt")
        lo, hi = object_world_bounds(obj)
        targets[part].append(
            {
                "object_ref": obj,
                "lo": lo,
                "hi": hi,
                "center": (lo + hi) * 0.5,
                "extent": np.maximum(hi - lo, 1e-8),
                "area": max(object_world_area(obj), 1e-12),
            }
        )
    material_cache: dict[tuple[int, str], bpy.types.Material] = {}
    average_cache: dict[int, tuple[float, float, float, float]] = {}
    try:
        for gt_part, items in targets.items():
            pred_part = mapping.get(gt_part)
            part_donors = donors.get(pred_part, []) if pred_part is not None else []
            if pred_part is not None and not part_donors:
                raise ValueError(f"Matched prediction part {pred_part} has no donor meshes")
            items = normalized_object_descriptors(items)
            matches = match_components(items, part_donors)
            for index, item in enumerate(items):
                obj = item["object_ref"]
                obj.data.materials.clear()
                donor = part_donors[matches[index]] if index in matches else None
                if donor is None or donor["material"] is None:
                    continue
                material = donor["material"]
                if donor["textured"] and not identity:
                    material = flatten_image_textures(
                        material,
                        donor["texture_surface_means"],
                        (int(material.as_pointer()), donor["object"]),
                        material_cache,
                        average_cache,
                    )
                obj.data.materials.append(material)
                for polygon in obj.data.polygons:
                    polygon.material_index = 0
                if donor["textured"] and identity:
                    ensure_uv_map(obj)
    finally:
        for material in retained:
            material.use_fake_user = False


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def imported_meshes() -> list[bpy.types.Object]:
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def object_key(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", name.lower())).strip("_")


def material_mesh_key(name: str) -> str:

    parts = name.split("::")
    if len(parts) >= 3 and parts[0] == "visual":
        return object_key(f"{parts[1]}_{parts[2]}")
    return object_key(name)


def part_index_for_object(
    name: str, part_names: list[str], role: str, link_to_part: dict[str, int] | None = None
) -> int:
    normalized = material_mesh_key(name)
    if role == "ours" and link_to_part is not None:
        tokens = name.split("::")
        if len(tokens) >= 3 and tokens[0] == "visual":
            return link_to_part[tokens[1]]
        matches = [
            (len(object_key(link)), part)
            for link, part in link_to_part.items()
            if normalized == object_key(link) or normalized.startswith(f"{object_key(link)}_")
        ]
        if not matches:
            raise ValueError(f"Prediction mesh has no URDF link identity: {name!r}")
        return max(matches)[1]
    matches = []
    for index, part_name in enumerate(part_names):
        if role == "ours" and index == 0 and object_key(part_name) == "root":
            continue
        key = object_key(part_name)
        if normalized == key or normalized.startswith(f"{key}_"):
            matches.append((len(key), index))
    if matches:
        return max(matches)[1]

    return 0


def object_world_bounds(obj: bpy.types.Object) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        [tuple(obj.matrix_world @ Vector(corner)) for corner in obj.bound_box], dtype=np.float64
    )
    return points.min(axis=0), points.max(axis=0)


def object_world_area(obj: bpy.types.Object) -> float:
    vertices = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    area = 0.0
    for polygon in obj.data.polygons:
        if len(polygon.vertices) < 3:
            continue
        origin = vertices[polygon.vertices[0]]
        for offset in range(1, len(polygon.vertices) - 1):
            edge_a = vertices[polygon.vertices[offset]] - origin
            edge_b = vertices[polygon.vertices[offset + 1]] - origin
            area += 0.5 * edge_a.cross(edge_b).length
    return float(area)


def import_reference_geometry(path: Path) -> None:
    clear_scene()
    bpy.ops.import_scene.gltf(filepath=str(path))


def raw_part_geometry(
    part_names: list[str], role: str, link_to_part: dict[str, int] | None = None
) -> tuple[dict[int, dict], np.ndarray, float, float]:
    grouped: dict[int, list[dict]] = {index: [] for index in range(len(part_names))}
    all_lows = []
    all_highs = []
    total_area = 0.0
    for obj in imported_meshes():
        part_id = part_index_for_object(obj.name, part_names, role, link_to_part)
        lo, hi = object_world_bounds(obj)
        area = max(object_world_area(obj), 1e-12)
        grouped[part_id].append({"lo": lo, "hi": hi, "area": area, "object": obj.name})
        all_lows.append(lo)
        all_highs.append(hi)
        total_area += area
    if not all_lows:
        raise ValueError("Aligned reference contains no mesh geometry")
    overall_lo = np.min(np.stack(all_lows), axis=0)
    overall_hi = np.max(np.stack(all_highs), axis=0)
    overall_center = (overall_lo + overall_hi) * 0.5
    overall_scale = max(float(np.max(overall_hi - overall_lo)), 1e-8)
    parts = {}
    for part_id, items in grouped.items():
        if not items:
            continue
        lo = np.min(np.stack([item["lo"] for item in items]), axis=0)
        hi = np.max(np.stack([item["hi"] for item in items]), axis=0)
        parts[part_id] = {
            "lo": lo,
            "hi": hi,
            "center": (lo + hi) * 0.5,
            "extent": np.maximum(hi - lo, 1e-8),
            "area": float(sum(item["area"] for item in items)),
            "component_count": len(items),
            "objects": [item["object"] for item in items],
        }
    return parts, overall_center, overall_scale, max(total_area, 1e-12)


def reference_part_descriptors(
    close_path: Path,
    open_path: Path,
    part_names: list[str],
    role: str,
    link_to_part: dict[str, int] | None = None,
) -> dict[int, dict]:
    import_reference_geometry(close_path)
    close_parts, center, scale, total_area = raw_part_geometry(part_names, role, link_to_part)
    import_reference_geometry(open_path)
    open_parts, _, _, _ = raw_part_geometry(part_names, role, link_to_part)
    descriptors = {}
    for part_id, close in close_parts.items():
        opened = open_parts.get(part_id, close)
        extent = close["extent"] / scale
        longest = max(float(np.max(extent)), 1e-12)
        descriptors[part_id] = {
            "center": (close["center"] - center) / scale,
            "extent": extent,
            "shape": np.sort(extent / longest),
            "relative_area": float(close["area"] / total_area),
            "motion": (opened["center"] - close["center"]) / scale,
            "component_count": close["component_count"],
            "objects": close["objects"],
        }
    clear_scene()
    return descriptors


def part_geometry_cost(pred: dict, gt: dict) -> float:
    center_cost = float(np.linalg.norm(pred["center"] - gt["center"]))
    extent_cost = float(np.linalg.norm(pred["extent"] - gt["extent"]))
    shape_cost = float(
        np.linalg.norm(
            np.log(np.maximum(pred["shape"], 1e-4)) - np.log(np.maximum(gt["shape"], 1e-4))
        )
    )
    area_cost = abs(math.log(max(pred["relative_area"], 1e-12) / max(gt["relative_area"], 1e-12)))
    pred_motion = pred["motion"]
    gt_motion = gt["motion"]
    motion_magnitude_cost = abs(
        float(np.linalg.norm(pred_motion)) - float(np.linalg.norm(gt_motion))
    )
    motion_vector_cost = float(np.linalg.norm(pred_motion - gt_motion))
    return float(
        center_cost
        + 0.45 * extent_cost
        + 0.20 * shape_cost
        + 0.08 * area_cost
        + 0.80 * motion_magnitude_cost
        + 0.35 * motion_vector_cost
    )


def dominant_material(obj: bpy.types.Object) -> bpy.types.Material | None:
    if not obj.material_slots:
        return None
    areas = np.zeros(len(obj.material_slots), dtype=np.float64)
    for polygon in obj.data.polygons:
        if 0 <= polygon.material_index < len(areas):
            areas[polygon.material_index] += float(polygon.area)
    index = int(np.argmax(areas)) if areas.size else 0
    return obj.material_slots[index].material


def material_uses_image(material: bpy.types.Material | None) -> bool:
    return bool(
        material
        and material.use_nodes
        and material.node_tree
        and any(
            node.type == "TEX_IMAGE" and node.image is not None for node in material.node_tree.nodes
        )
    )


def material_image_nodes(material: bpy.types.Material | None) -> list[bpy.types.Node]:
    if not material or not material.use_nodes or not material.node_tree:
        return []
    return [
        node
        for node in material.node_tree.nodes
        if node.type == "TEX_IMAGE" and node.image is not None
    ]


def image_pixel_array(image: bpy.types.Image) -> np.ndarray:
    values = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(values)
    channels = max(int(image.channels), 1)
    return values.reshape(int(image.size[1]), int(image.size[0]), channels)


def sample_image_at_uv(pixels: np.ndarray, uv: np.ndarray, extension: str) -> np.ndarray | None:
    u, v = float(uv[0]), float(uv[1])
    if extension == "REPEAT":
        u %= 1.0
        v %= 1.0
    elif extension == "EXTEND":
        u = min(max(u, 0.0), 1.0 - 1e-8)
        v = min(max(v, 0.0), 1.0 - 1e-8)
    elif extension == "MIRROR":
        u = 1.0 - abs((u % 2.0) - 1.0)
        v = 1.0 - abs((v % 2.0) - 1.0)
        u = min(u, 1.0 - 1e-8)
        v = min(v, 1.0 - 1e-8)
    elif not (0.0 <= u < 1.0 and 0.0 <= v < 1.0):
        return None
    height, width = pixels.shape[:2]
    return pixels[min(int(v * height), height - 1), min(int(u * width), width - 1)]


def surface_sampled_image_mean(
    obj: bpy.types.Object,
    material: bpy.types.Material,
    node: bpy.types.Node,
    pixels: np.ndarray,
) -> tuple[float, float, float, float]:
    layer = obj.data.uv_layers.active
    vector_input = node.inputs.get("Vector")

    if layer is None or (vector_input is not None and vector_input.is_linked):
        mean = pixels.reshape(-1, pixels.shape[-1]).mean(axis=0)
        return rgba_tuple(mean)
    slot_ids = {
        index
        for index, slot in enumerate(obj.material_slots)
        if slot.material is not None and slot.material.as_pointer() == material.as_pointer()
    }
    weighted = np.zeros(pixels.shape[-1], dtype=np.float64)
    total_weight = 0.0
    world_vertices = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    for polygon in obj.data.polygons:
        if polygon.material_index not in slot_ids or len(polygon.vertices) < 3:
            continue
        loop_indices = list(polygon.loop_indices)
        origin_vertex = world_vertices[polygon.vertices[0]]
        origin_uv = np.asarray(layer.data[loop_indices[0]].uv, dtype=np.float64)
        for offset in range(1, len(polygon.vertices) - 1):
            vertex_a = world_vertices[polygon.vertices[offset]]
            vertex_b = world_vertices[polygon.vertices[offset + 1]]
            weight = 0.5 * (vertex_a - origin_vertex).cross(vertex_b - origin_vertex).length
            if weight <= 1e-16:
                continue
            uv_a = np.asarray(layer.data[loop_indices[offset]].uv, dtype=np.float64)
            uv_b = np.asarray(layer.data[loop_indices[offset + 1]].uv, dtype=np.float64)
            value = sample_image_at_uv(pixels, (origin_uv + uv_a + uv_b) / 3.0, node.extension)
            if value is None:
                continue
            weighted += value * weight
            total_weight += weight
    if total_weight <= 1e-16:
        return rgba_tuple(pixels.reshape(-1, pixels.shape[-1]).mean(axis=0))
    return rgba_tuple(weighted / total_weight)


def rgba_tuple(values: np.ndarray) -> tuple[float, float, float, float]:
    rgba = np.ones(4, dtype=np.float32)
    rgba[: min(len(values), 4)] = values[:4]
    if len(values) == 1:
        rgba[:3] = values[0]
    return tuple(float(value) for value in rgba)


def ensure_uv_map(obj: bpy.types.Object) -> bool:
    if obj.data.uv_layers:
        return False
    previous_active = bpy.context.view_layer.objects.active
    previous_mode = bpy.context.mode
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    try:
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=1.15192, island_margin=0.02)
        bpy.ops.object.mode_set(mode="OBJECT")
    finally:
        if bpy.context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        obj.select_set(False)
        if previous_active and previous_active.name in bpy.context.scene.objects:
            bpy.context.view_layer.objects.active = previous_active
        if previous_mode != "OBJECT" and previous_active:
            try:
                bpy.ops.object.mode_set(mode=previous_mode)
            except RuntimeError:
                pass
    return True


def collect_part_donors(
    part_names: list[str],
    link_to_part: dict[str, int] | None = None,
) -> tuple[dict[int, list[dict]], list[bpy.types.Material]]:
    donors: dict[int, list[dict]] = {index: [] for index in range(len(part_names))}
    retained_materials = []
    image_users: dict[int, dict] = {}
    for obj in imported_meshes():
        part_id = part_index_for_object(obj.name, part_names, "ours", link_to_part)
        lo, hi = object_world_bounds(obj)
        material = dominant_material(obj)
        if material is not None:
            material.use_fake_user = True
            retained_materials.append(material)
        donor = {
            "object": obj.name,
            "source_object": str(obj.get("material_eval_source_object", obj.name)),
            "lo": lo,
            "hi": hi,
            "center": (lo + hi) * 0.5,
            "extent": np.maximum(hi - lo, 1e-8),
            "area": max(object_world_area(obj), 1e-12),
            "material": material,
            "textured": material_uses_image(material),
            "has_uv": bool(obj.data.uv_layers),
            "texture_surface_means": {},
        }
        donors[part_id].append(donor)
        for node in material_image_nodes(material):
            key = int(node.image.as_pointer())
            image_users.setdefault(key, {"image": node.image, "users": []})["users"].append(
                (donor, obj, material, node)
            )

    for image_user in image_users.values():
        pixels = image_pixel_array(image_user["image"])
        for donor, obj, material, node in image_user["users"]:
            donor["texture_surface_means"][node.name] = surface_sampled_image_mean(
                obj, material, node, pixels
            )
        del pixels
    return donors, retained_materials


def normalized_object_descriptors(items: list[dict]) -> list[dict]:
    if not items:
        return []
    part_lo = np.min(np.stack([item["lo"] for item in items]), axis=0)
    part_hi = np.max(np.stack([item["hi"] for item in items]), axis=0)
    part_extent = np.maximum(part_hi - part_lo, 1e-8)
    part_area = max(sum(item["area"] for item in items), 1e-12)
    part_volume = max(float(np.prod(part_extent)), 1e-12)
    output = []
    for item in items:
        descriptor = dict(item)
        descriptor["normalized_center"] = (item["center"] - part_lo) / part_extent
        descriptor["normalized_extent"] = item["extent"] / part_extent
        descriptor["relative_area"] = float(item["area"] / part_area)
        descriptor["relative_volume"] = float(np.prod(item["extent"]) / part_volume)
        longest = max(float(np.max(item["extent"])), 1e-12)
        descriptor["shape"] = np.sort(item["extent"] / longest)
        output.append(descriptor)
    return output


def linear_sum_assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    rows, columns = cost.shape
    if rows == 0 or columns == 0:
        return []
    if rows > columns:
        return [(column, row) for row, column in linear_sum_assignment(cost.T)]
    u = np.zeros(rows + 1, dtype=np.float64)
    v = np.zeros(columns + 1, dtype=np.float64)
    matched_row = np.zeros(columns + 1, dtype=np.int32)
    predecessor = np.zeros(columns + 1, dtype=np.int32)
    for row in range(1, rows + 1):
        matched_row[0] = row
        minimum = np.full(columns + 1, np.inf, dtype=np.float64)
        used = np.zeros(columns + 1, dtype=bool)
        column = 0
        while True:
            used[column] = True
            active_row = matched_row[column]
            delta = np.inf
            next_column = 0
            for candidate in range(1, columns + 1):
                if used[candidate]:
                    continue
                reduced = cost[active_row - 1, candidate - 1] - u[active_row] - v[candidate]
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    predecessor[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(columns + 1):
                if used[candidate]:
                    u[matched_row[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = predecessor[column]
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break
    return sorted(
        (int(matched_row[column] - 1), column - 1)
        for column in range(1, columns + 1)
        if matched_row[column]
    )


def component_cost(target: dict, donor: dict, reflection: np.ndarray) -> float:
    donor_center = np.where(
        reflection, 1.0 - donor["normalized_center"], donor["normalized_center"]
    )
    center_cost = float(np.linalg.norm(target["normalized_center"] - donor_center))
    extent_cost = float(np.linalg.norm(target["normalized_extent"] - donor["normalized_extent"]))
    shape_cost = float(
        np.linalg.norm(
            np.log(np.maximum(target["shape"], 1e-4)) - np.log(np.maximum(donor["shape"], 1e-4))
        )
    )
    area_cost = abs(
        math.log(max(target["relative_area"], 1e-12) / max(donor["relative_area"], 1e-12))
    )
    volume_cost = abs(
        math.log(max(target["relative_volume"], 1e-12) / max(donor["relative_volume"], 1e-12))
    )
    if target["relative_area"] <= SMALL_COMPONENT_AREA_FRACTION:
        weights = (0.45, 0.75, 0.28, 0.16, 0.06)
    else:
        weights = (0.80, 0.45, 0.12, 0.08, 0.03)
    return float(
        sum(
            weight * value
            for weight, value in zip(
                weights, (center_cost, extent_cost, shape_cost, area_cost, volume_cost)
            )
        )
    )


def match_components(targets: list[dict], donors: list[dict]) -> dict:
    if not targets or not donors:
        return {}

    if len(targets) == 1 or len(targets) > 8:
        cost = np.asarray(
            [[legacy_component_cost(target, donor) for donor in donors] for target in targets],
            dtype=np.float64,
        )
        target_to_donor = {row: int(np.argmin(cost[row])) for row in range(len(targets))}
        return target_to_donor
    best = None
    for mask in range(8):
        reflection = np.asarray([(mask >> axis) & 1 for axis in range(3)], dtype=bool)
        cost = np.asarray(
            [[component_cost(target, donor, reflection) for donor in donors] for target in targets],
            dtype=np.float64,
        )
        pairs = linear_sum_assignment(cost)
        score = float(sum(cost[row, column] for row, column in pairs) / max(len(pairs), 1))

        score += float(reflection.sum()) * 1e-8
        if best is None or score < best[0]:
            best = (score, reflection, cost, pairs)
    _, reflection, cost, pairs = best
    target_to_donor = {row: column for row, column in pairs}

    for row in range(len(targets)):
        if row not in target_to_donor:
            column = int(np.argmin(cost[row]))
            target_to_donor[row] = column
    return target_to_donor


def image_average(
    image: bpy.types.Image, cache: dict[int, tuple[float, float, float, float]]
) -> tuple[float, float, float, float]:
    key = int(image.as_pointer())
    if key not in cache:
        values = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(values)
        channels = max(int(image.channels), 1)
        pixels = values.reshape(-1, channels)
        mean = pixels.mean(axis=0) if pixels.size else np.ones(channels, dtype=np.float32)
        image_average_value = rgba_tuple(mean)
        cache[key] = image_average_value
    return cache[key]


def flatten_image_textures(
    material: bpy.types.Material,
    surface_means: dict[str, tuple[float, float, float, float]],
    cache_key: tuple[int, str],
    material_cache: dict[tuple[int, str], bpy.types.Material],
    average_cache: dict[int, tuple[float, float, float, float]],
) -> bpy.types.Material:
    if cache_key in material_cache:
        return material_cache[cache_key]
    flattened = material.copy()
    flattened.name = f"{material.name}__component_mean"
    tree = flattened.node_tree
    if tree is not None:
        for node in list(tree.nodes):
            if node.type != "TEX_IMAGE" or node.image is None:
                continue
            mean = surface_means.get(node.name)
            if mean is None:
                mean = image_average(node.image, average_cache)
            rgb = tree.nodes.new("ShaderNodeRGB")
            rgb.outputs["Color"].default_value = mean
            alpha = tree.nodes.new("ShaderNodeValue")
            alpha.outputs["Value"].default_value = mean[3]
            for output in node.outputs:
                replacement = (
                    alpha.outputs["Value"] if output.name == "Alpha" else rgb.outputs["Color"]
                )
                destinations = [link.to_socket for link in output.links]
                for link in list(output.links):
                    tree.links.remove(link)
                for destination in destinations:
                    tree.links.new(replacement, destination)
            tree.nodes.remove(node)
    material_cache[cache_key] = flattened
    return flattened


def refine_correspondence(
    models: dict[str, str],
    pred_names: list[str],
    gt_names: list[str],
    link_to_part: dict[str, int] | None = None,
) -> dict:
    pred = reference_part_descriptors(
        Path(models["pred_close"]), Path(models["pred_open"]), pred_names, "ours", link_to_part
    )
    gt = reference_part_descriptors(
        Path(models["gt_close"]), Path(models["gt_open"]), gt_names, "gt"
    )
    pairs = []
    matched_pred = set()
    matched_gt = set()

    if 0 in pred and 0 in gt:
        pairs.append(
            {
                "pred_part_id": 0,
                "gt_part_id": 0,
            }
        )
        matched_pred.add(0)
        matched_gt.add(0)
    pred_ids = sorted(set(pred) - matched_pred)
    gt_ids = sorted(set(gt) - matched_gt)
    if pred_ids and gt_ids:
        costs = np.asarray(
            [
                [part_geometry_cost(pred[pred_id], gt[gt_id]) for gt_id in gt_ids]
                for pred_id in pred_ids
            ],
            dtype=np.float64,
        )
        for pred_row, gt_column in linear_sum_assignment(costs):
            pred_id = pred_ids[pred_row]
            gt_id = gt_ids[gt_column]
            pairs.append(
                {
                    "pred_part_id": pred_id,
                    "gt_part_id": gt_id,
                }
            )
            matched_pred.add(pred_id)
            matched_gt.add(gt_id)
    pairs.sort(key=lambda pair: pair["pred_part_id"])
    return {
        "pred_part_names": pred_names,
        "link_to_part": link_to_part,
        "gt_part_names": gt_names,
        "pairs": pairs,
    }


def legacy_component_cost(target: dict, donor: dict) -> float:
    center_cost = np.linalg.norm(target["normalized_center"] - donor["normalized_center"])
    extent_cost = np.linalg.norm(target["normalized_extent"] - donor["normalized_extent"])
    area_cost = abs(math.log(max(target["area"], 1e-12) / max(donor["area"], 1e-12)))
    return float(center_cost + 0.35 * extent_cost + 0.05 * area_cost)
