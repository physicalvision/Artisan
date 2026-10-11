from __future__ import annotations

import json
import re
from pathlib import Path

import bpy  # type: ignore[import-not-found]

_DUP_SUFFIX = re.compile(r"\.\d{3}$")


def _set(inputs, *names, value) -> bool:
    for n in names:
        if n in inputs:
            inputs[n].default_value = value
            return True
    return False


def _principled(mat):
    if not mat.use_nodes:
        mat.use_nodes = True
    for node in mat.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    return None


def _apply_glass(mat, spec: dict) -> None:
    bsdf = _principled(mat)
    if bsdf is None:
        return
    i = bsdf.inputs
    _set(i, "Transmission Weight", "Transmission", value=float(spec.get("transmission", 1.0)))
    if "ior" in spec:
        _set(i, "IOR", value=float(spec["ior"]))
    _set(i, "Alpha", value=1.0)
    try:
        mat.blend_method = "OPAQUE"
    except (AttributeError, TypeError):
        pass
    try:
        mat.use_screen_refraction = True
    except (AttributeError, TypeError):
        pass
    ac = spec.get("attenuation_color")
    ad = spec.get("attenuation_distance")
    if ac is not None and ad:
        _apply_volume_absorption(mat, ac, ad)


def _apply_volume_absorption(mat, color, distance) -> None:
    nt = mat.node_tree
    out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
    if out is None or "Volume" not in out.inputs:
        return
    try:
        rgb = tuple(float(c) for c in list(color)[:3])
        dist = float(distance)
    except (TypeError, ValueError):
        return
    if dist <= 0.0:
        return
    vol = nt.nodes.new("ShaderNodeVolumeAbsorption")
    vol.inputs["Color"].default_value = (*rgb, 1.0)
    vol.inputs["Density"].default_value = 1.0 / dist
    nt.links.new(vol.outputs["Volume"], out.inputs["Volume"])


def _apply_emission(mat, spec: dict) -> None:
    bsdf = _principled(mat)
    if bsdf is None:
        return
    em = spec.get("emission") or {}
    try:
        strength = float(em.get("strength", 0.0))
    except (TypeError, ValueError):
        return
    if strength <= 0.0:
        return
    try:
        rgb = tuple(float(c) for c in list(em.get("color") or [1.0, 1.0, 1.0])[:3])
    except (TypeError, ValueError):
        rgb = (1.0, 1.0, 1.0)
    i = bsdf.inputs
    _set(i, "Emission Color", "Emission", value=(*rgb, 1.0))
    _set(i, "Emission Strength", value=strength)


def _load_image(path: str, non_color: bool):
    img = bpy.data.images.load(path, check_existing=True)
    if non_color:
        try:
            img.colorspace_settings.name = "Non-Color"
        except (AttributeError, TypeError):
            pass
    return img


def _apply_texture(mat, texture_dir: str) -> None:
    d = Path(texture_dir)
    color = d / "color.png"
    if not color.is_file():
        return
    bsdf = _principled(mat)
    if bsdf is None:
        return
    nt = mat.node_tree
    coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (3.0, 3.0, 3.0)
    nt.links.new(coord.outputs["Object"], mapping.inputs["Vector"])

    def _tex(path: str, non_color: bool):
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = _load_image(path, non_color)
        node.projection = "BOX"
        try:
            node.projection_blend = 0.3
        except (AttributeError, TypeError):
            pass
        nt.links.new(mapping.outputs["Vector"], node.inputs["Vector"])
        return node

    nt.links.new(_tex(str(color), False).outputs["Color"], bsdf.inputs["Base Color"])
    rough = d / "roughness.png"
    if rough.is_file():
        nt.links.new(_tex(str(rough), True).outputs["Color"], bsdf.inputs["Roughness"])
    metal = d / "metalness.png"
    if metal.is_file():
        nt.links.new(_tex(str(metal), True).outputs["Color"], bsdf.inputs["Metallic"])
    normal = d / "normal.png"
    if normal.is_file():
        nmap = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(_tex(str(normal), True).outputs["Color"], nmap.inputs["Color"])
        if "Normal" in bsdf.inputs:
            nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])


def apply_materials(spec_path: str) -> int:
    p = Path(spec_path)
    if not p.is_file():
        return 0
    try:
        specs = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    if not isinstance(specs, dict):
        return 0
    touched = 0
    for mat in list(bpy.data.materials):
        base = _DUP_SUFFIX.sub("", mat.name)
        spec = specs.get(mat.name) or specs.get(base)
        if not spec:
            continue
        if spec.get("texture_dir"):
            _apply_texture(mat, str(spec["texture_dir"]))
        if spec.get("transmission"):
            _apply_glass(mat, spec)
        if spec.get("emission"):
            _apply_emission(mat, spec)
        touched += 1
    return touched
