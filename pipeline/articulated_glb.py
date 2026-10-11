from __future__ import annotations

import json
import struct
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ARTICULATED_GLB_FILENAME = "object.articulated.glb"
ARTICULATED_APPEARANCE_FILENAME = "object.articulated.appearance.json"
ARTICULATED_ROOT_NODE = "__artisan_urdf_z_up__"
LINK_NODE_PREFIX = "link::"
VISUAL_NODE_PREFIX = "visual::"

_GLB_HEADER = struct.Struct("<4sII")
_GLB_CHUNK_HEADER = struct.Struct("<II")
_GLB_MAGIC = b"glTF"
_GLB_VERSION = 2
_JSON_CHUNK = 0x4E4F534A


class ArticulatedGlbError(ValueError):
    pass


def _chunks(data: bytes) -> list[tuple[int, bytes]]:
    if len(data) < _GLB_HEADER.size:
        raise ArticulatedGlbError("GLB is shorter than its header")
    magic, version, declared_length = _GLB_HEADER.unpack_from(data, 0)
    if magic != _GLB_MAGIC or version != _GLB_VERSION:
        raise ArticulatedGlbError("expected a glTF 2.0 binary container")
    if declared_length != len(data):
        raise ArticulatedGlbError(
            f"GLB length mismatch: header={declared_length}, actual={len(data)}"
        )
    offset = _GLB_HEADER.size
    parsed: list[tuple[int, bytes]] = []
    while offset < len(data):
        if offset + _GLB_CHUNK_HEADER.size > len(data):
            raise ArticulatedGlbError("GLB has a truncated chunk header")
        length, chunk_type = _GLB_CHUNK_HEADER.unpack_from(data, offset)
        offset += _GLB_CHUNK_HEADER.size
        end = offset + length
        if end > len(data):
            raise ArticulatedGlbError("GLB has a truncated chunk payload")
        parsed.append((chunk_type, data[offset:end]))
        offset = end
    return parsed


def read_glb_json(path: Path) -> dict[str, Any]:
    json_chunks = [
        payload for kind, payload in _chunks(Path(path).read_bytes()) if kind == _JSON_CHUNK
    ]
    if len(json_chunks) != 1:
        raise ArticulatedGlbError(f"expected exactly one GLB JSON chunk, got {len(json_chunks)}")
    payload = json_chunks[0].rstrip(b" \t\r\n\0")
    document = json.loads(payload.decode("utf-8"))
    if not isinstance(document, dict):
        raise ArticulatedGlbError("GLB JSON root must be an object")
    return document


def rewrite_glb_json(
    source_path: Path,
    destination_path: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> Path:
    source_path = Path(source_path)
    destination_path = Path(destination_path)
    chunks = _chunks(source_path.read_bytes())
    json_count = sum(kind == _JSON_CHUNK for kind, _payload in chunks)
    if json_count != 1:
        raise ArticulatedGlbError(f"expected exactly one GLB JSON chunk, got {json_count}")

    rewritten: list[tuple[int, bytes]] = []
    for kind, payload in chunks:
        if kind != _JSON_CHUNK:
            rewritten.append((kind, payload))
            continue
        document = json.loads(payload.rstrip(b" \t\r\n\0").decode("utf-8"))
        if not isinstance(document, dict):
            raise ArticulatedGlbError("GLB JSON root must be an object")
        mutate(document)
        encoded = json.dumps(
            document,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        encoded += b" " * ((-len(encoded)) % 4)
        rewritten.append((kind, encoded))

    total_length = _GLB_HEADER.size + sum(_GLB_CHUNK_HEADER.size + len(p) for _k, p in rewritten)
    output = bytearray(_GLB_HEADER.pack(_GLB_MAGIC, _GLB_VERSION, total_length))
    for kind, payload in rewritten:
        output.extend(_GLB_CHUNK_HEADER.pack(len(payload), kind))
        output.extend(payload)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_bytes(bytes(output))
    return destination_path


def _use_extension(document: dict[str, Any], extension: str) -> None:
    used = {str(value) for value in document.get("extensionsUsed") or []}
    used.add(extension)
    document["extensionsUsed"] = sorted(used)


def apply_standard_material_extensions(
    glb_path: Path,
    material_specs: Mapping[str, Mapping[str, Any]],
) -> Path:

    def _mutate(document: dict[str, Any]) -> None:
        for material in document.get("materials") or []:
            if not isinstance(material, dict):
                continue
            name = str(material.get("name") or "")
            spec = material_specs.get(name) or {}
            pbr = material.get("pbrMetallicRoughness")
            if not isinstance(pbr, dict):
                pbr = {}
                material["pbrMetallicRoughness"] = pbr

            transmission = float(spec.get("transmission") or 0.0)
            base = pbr.get("baseColorFactor")
            if isinstance(base, list) and len(base) == 4:
                if transmission > 0.0:
                    base[3] = 1.0
                    material.pop("alphaMode", None)
                    material.pop("alphaCutoff", None)
                elif float(base[3]) < 1.0:
                    material["alphaMode"] = "BLEND"

            if transmission > 0.0:
                extensions = material.setdefault("extensions", {})
                extensions["KHR_materials_transmission"] = {
                    "transmissionFactor": transmission,
                }
                _use_extension(document, "KHR_materials_transmission")
                if spec.get("ior") is not None:
                    extensions["KHR_materials_ior"] = {"ior": float(spec["ior"])}
                    _use_extension(document, "KHR_materials_ior")

                attenuation_color = spec.get("attenuation_color")
                attenuation_distance = spec.get("attenuation_distance")
                if attenuation_color is not None or attenuation_distance is not None:
                    volume: dict[str, Any] = {"thicknessFactor": 1.0}
                    if attenuation_color is not None:
                        volume["attenuationColor"] = [
                            float(value) for value in list(attenuation_color)[:3]
                        ]
                    if attenuation_distance is not None:
                        volume["attenuationDistance"] = float(attenuation_distance)
                    extensions["KHR_materials_volume"] = volume
                    _use_extension(document, "KHR_materials_volume")

            emission = spec.get("emission")
            if isinstance(emission, Mapping):
                strength = float(emission.get("strength") or 0.0)
                color = [float(value) for value in list(emission.get("color") or [1, 1, 1])[:3]]
                if strength > 1.0:
                    material["emissiveFactor"] = color
                    extensions = material.setdefault("extensions", {})
                    extensions["KHR_materials_emissive_strength"] = {
                        "emissiveStrength": strength,
                    }
                    _use_extension(document, "KHR_materials_emissive_strength")

    return rewrite_glb_json(glb_path, glb_path, _mutate)


def add_link_node_metadata(
    glb_path: Path,
    link_metadata: Mapping[str, Mapping[str, Any]],
) -> Path:

    def _mutate(document: dict[str, Any]) -> None:
        found: set[str] = set()
        for node in document.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            name = str(node.get("name") or "")
            if not name.startswith(LINK_NODE_PREFIX):
                continue
            link_name = name.removeprefix(LINK_NODE_PREFIX)
            metadata = link_metadata.get(link_name)
            if metadata is None:
                continue
            extras = node.setdefault("extras", {})
            extras["articulation"] = dict(metadata)
            found.add(link_name)
        missing = sorted(set(link_metadata) - found)
        if missing:
            raise ArticulatedGlbError(f"articulated GLB is missing link nodes: {missing}")

    return rewrite_glb_json(glb_path, glb_path, _mutate)


def _column_major(matrix: Sequence[Sequence[float]]) -> list[float]:
    rows = [[float(value) for value in row] for row in matrix]
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ArticulatedGlbError("link transform must be a 4x4 matrix")
    return [rows[row][column] for column in range(4) for row in range(4)]


def pose_articulated_glb(
    source_path: Path,
    destination_path: Path,
    world_transforms: Mapping[str, Sequence[Sequence[float]]],
) -> Path:
    import numpy as np

    world = {
        str(link): np.asarray(matrix, dtype=float) for link, matrix in world_transforms.items()
    }

    def _mutate(document: dict[str, Any]) -> None:
        link_nodes: dict[str, dict[str, Any]] = {}
        parent_by_link: dict[str, str | None] = {}
        for node in document.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            extras = node.get("extras")
            metadata = (
                extras.get("articulation")
                if isinstance(extras, dict)
                else None
            )
            if not isinstance(metadata, dict) or metadata.get("kind") != "link":
                continue
            link = str(metadata.get("link") or "")
            if not link:
                continue
            link_nodes[link] = node
            parent = metadata.get("parent_link")
            parent_by_link[link] = str(parent) if parent else None

        missing = sorted(set(world) - set(link_nodes))
        if missing:
            raise ArticulatedGlbError(f"articulated GLB is missing pose link nodes: {missing}")
        for link, node in link_nodes.items():
            if link not in world:
                continue
            parent = parent_by_link.get(link)
            local = world[link]
            if parent is not None:
                if parent not in world:
                    raise ArticulatedGlbError(
                        f"pose is missing parent link {parent!r} required by {link!r}"
                    )
                local = np.linalg.inv(world[parent]) @ local
            node["matrix"] = _column_major(local.tolist())
            node.pop("translation", None)
            node.pop("rotation", None)
            node.pop("scale", None)

    return rewrite_glb_json(source_path, destination_path, _mutate)
