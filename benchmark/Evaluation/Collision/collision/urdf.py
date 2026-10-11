from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Origin:
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class MaterialSpec:
    source_name: str
    rgba: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class GeometrySpec:
    kind: str
    attrs: dict[str, str]


@dataclass(frozen=True)
class VisualSpec:
    name: str
    origin: Origin
    geometry: GeometrySpec
    material: MaterialSpec | None


@dataclass(frozen=True)
class CollisionSpec:
    name: str
    origin: Origin
    geometry: GeometrySpec


@dataclass(frozen=True)
class InertialSpec:
    origin: Origin
    mass: float
    inertia: dict[str, float]


@dataclass(frozen=True)
class LinkSpec:
    name: str
    visuals: list[VisualSpec] = field(default_factory=list)
    collisions: list[CollisionSpec] = field(default_factory=list)
    inertial: InertialSpec | None = None


@dataclass(frozen=True)
class MimicSpec:
    joint: str
    multiplier: float = 1.0
    offset: float = 0.0


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: Origin
    axis: tuple[float, float, float]
    limit: dict[str, float]
    dynamics: dict[str, float]
    mimic: MimicSpec | None = None


@dataclass(frozen=True)
class RobotSpec:
    name: str
    links: dict[str, LinkSpec]
    joints: list[JointSpec]
    materials: dict[str, MaterialSpec]


def _resolve_mesh_source(filename: str, *, urdf_dir: Path) -> Path:
    if "://" in filename:
        if filename.startswith("file://"):
            candidate = Path(filename.removeprefix("file://"))
        else:
            raise ValueError(f"Unsupported mesh URI in URDF: {filename}")
    else:
        candidate = Path(filename)
    if not candidate.is_absolute():
        candidate = urdf_dir / candidate
    source = candidate.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"URDF mesh file does not exist: {filename} -> {source}")
    return source


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(element) if _tag(child) == name]


def _child(element: ET.Element, name: str) -> ET.Element | None:
    matches = _children(element, name)
    return matches[0] if matches else None


def _float_attr(element: ET.Element | None, name: str, default: float) -> float:
    if element is None:
        return default
    value = element.attrib.get(name)
    if value is None:
        return default
    return float(value)


def _float_tuple(text: str | None, *, length: int, default: tuple[float, ...]) -> tuple[float, ...]:
    if text is None:
        return default
    values = tuple((float(part) for part in text.split()))
    if len(values) != length:
        raise ValueError(f"Expected {length} values, got {len(values)} in {text!r}")
    return values


def _parse_origin(element: ET.Element | None) -> Origin:
    if element is None:
        return Origin()
    xyz = _float_tuple(element.attrib.get("xyz"), length=3, default=(0.0, 0.0, 0.0))
    rpy = _float_tuple(element.attrib.get("rpy"), length=3, default=(0.0, 0.0, 0.0))
    return Origin(
        xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
        rpy=(float(rpy[0]), float(rpy[1]), float(rpy[2])),
    )


def _parse_material_element(element: ET.Element | None) -> MaterialSpec | None:
    if element is None:
        return None
    name = element.attrib.get("name", "material")
    color = _child(element, "color")
    rgba: tuple[float, float, float, float] | None = None
    if color is not None:
        raw_rgba = _float_tuple(color.attrib.get("rgba"), length=4, default=(0.8, 0.8, 0.8, 1.0))
        rgba = (
            float(raw_rgba[0]),
            float(raw_rgba[1]),
            float(raw_rgba[2]),
            float(raw_rgba[3]),
        )
    return MaterialSpec(source_name=name, rgba=rgba)


def _parse_geometry(element: ET.Element) -> GeometrySpec:
    geometry = _child(element, "geometry")
    if geometry is None:
        raise ValueError("Missing <geometry> element")
    box = _child(geometry, "box")
    if box is not None:
        return GeometrySpec(kind="box", attrs=dict(box.attrib))
    sphere = _child(geometry, "sphere")
    if sphere is not None:
        return GeometrySpec(kind="sphere", attrs=dict(sphere.attrib))
    cylinder = _child(geometry, "cylinder")
    if cylinder is not None:
        return GeometrySpec(kind="cylinder", attrs=dict(cylinder.attrib))
    mesh = _child(geometry, "mesh")
    if mesh is not None:
        return GeometrySpec(kind="mesh", attrs=dict(mesh.attrib))
    raise ValueError("Unsupported URDF geometry")


def _parse_visual(
    element: ET.Element, *, index: int, materials: dict[str, MaterialSpec]
) -> VisualSpec:
    material = _parse_material_element(_child(element, "material"))
    if material is not None and material.rgba is None and (material.source_name in materials):
        material = materials[material.source_name]
    return VisualSpec(
        name=element.attrib.get("name", f"visual_{index}"),
        origin=_parse_origin(_child(element, "origin")),
        geometry=_parse_geometry(element),
        material=material,
    )


def _parse_collision(element: ET.Element, *, index: int) -> CollisionSpec:
    return CollisionSpec(
        name=element.attrib.get("name", f"collision_{index}"),
        origin=_parse_origin(_child(element, "origin")),
        geometry=_parse_geometry(element),
    )


def _parse_inertial(element: ET.Element | None) -> InertialSpec | None:
    if element is None:
        return None
    mass_element = _child(element, "mass")
    inertia_element = _child(element, "inertia")
    if mass_element is None or inertia_element is None:
        return None
    inertia = {
        key: _float_attr(inertia_element, key, 0.0)
        for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
    }
    return InertialSpec(
        origin=_parse_origin(_child(element, "origin")),
        mass=_float_attr(mass_element, "value", 0.0),
        inertia=inertia,
    )


def _parse_link(element: ET.Element, materials: dict[str, MaterialSpec]) -> LinkSpec:
    name = element.attrib["name"]
    visuals = [
        _parse_visual(visual, index=index, materials=materials)
        for index, visual in enumerate(_children(element, "visual"), start=1)
    ]
    collisions = [
        _parse_collision(collision, index=index)
        for index, collision in enumerate(_children(element, "collision"), start=1)
    ]
    return LinkSpec(
        name=name,
        visuals=visuals,
        collisions=collisions,
        inertial=_parse_inertial(_child(element, "inertial")),
    )


def _parse_mimic(element: ET.Element | None) -> MimicSpec | None:
    if element is None:
        return None
    joint = element.attrib.get("joint")
    if not joint:
        return None
    return MimicSpec(
        joint=joint,
        multiplier=float(element.attrib.get("multiplier", "1.0")),
        offset=float(element.attrib.get("offset", "0.0")),
    )


def _parse_joint(element: ET.Element) -> JointSpec:
    parent = _child(element, "parent")
    child = _child(element, "child")
    if parent is None or child is None:
        raise ValueError(f"Joint {element.attrib.get('name', '<unnamed>')} is missing parent/child")
    axis_element = _child(element, "axis")
    limit_element = _child(element, "limit")
    dynamics_element = _child(element, "dynamics")
    limit = {}
    if limit_element is not None:
        for key in ("lower", "upper", "effort", "velocity"):
            if key in limit_element.attrib:
                limit[key] = float(limit_element.attrib[key])
    dynamics = {}
    if dynamics_element is not None:
        for key in ("damping", "friction"):
            if key in dynamics_element.attrib:
                dynamics[key] = float(dynamics_element.attrib[key])
    axis = _float_tuple(
        axis_element.attrib.get("xyz") if axis_element is not None else None,
        length=3,
        default=(1.0, 0.0, 0.0),
    )
    return JointSpec(
        name=element.attrib["name"],
        joint_type=element.attrib.get("type", "fixed"),
        parent=parent.attrib["link"],
        child=child.attrib["link"],
        origin=_parse_origin(_child(element, "origin")),
        axis=(float(axis[0]), float(axis[1]), float(axis[2])),
        limit=limit,
        dynamics=dynamics,
        mimic=_parse_mimic(_child(element, "mimic")),
    )


def parse_urdf(urdf_path: Path) -> RobotSpec:
    root = ET.parse(urdf_path).getroot()
    if _tag(root) != "robot":
        raise ValueError(f"Expected URDF <robot>, got <{_tag(root)}>")
    materials = {
        material.source_name: material
        for element in _children(root, "material")
        if (material := _parse_material_element(element)) is not None
    }
    links = {
        link.name: link
        for element in _children(root, "link")
        if (link := _parse_link(element, materials)) is not None
    }
    joints = [_parse_joint(element) for element in _children(root, "joint")]
    return RobotSpec(
        name=root.attrib.get("name", urdf_path.stem),
        links=links,
        joints=joints,
        materials=materials,
    )
