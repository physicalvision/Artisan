from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Origin:
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class GeometrySpec:
    kind: str
    attrs: dict[str, str]


@dataclass(frozen=True)
class VisualSpec:
    origin: Origin
    geometry: GeometrySpec


@dataclass(frozen=True)
class LinkSpec:
    name: str
    visuals: list[VisualSpec] = field(default_factory=list)


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
    mimic: MimicSpec | None = None


@dataclass(frozen=True)
class RobotSpec:
    links: dict[str, LinkSpec]
    joints: list[JointSpec]


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


def _float_tuple(text: str | None, *, length: int, default: tuple[float, ...]) -> tuple[float, ...]:
    if text is None:
        return default
    values = tuple(float(part) for part in text.split())
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


def _parse_visual(element: ET.Element) -> VisualSpec:
    return VisualSpec(
        origin=_parse_origin(_child(element, "origin")),
        geometry=_parse_geometry(element),
    )


def _parse_link(element: ET.Element) -> LinkSpec:
    name = element.attrib["name"]
    visuals = [_parse_visual(visual) for visual in _children(element, "visual")]
    return LinkSpec(name=name, visuals=visuals)


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
    limit = {}
    if limit_element is not None:
        for key in ("lower", "upper"):
            if key in limit_element.attrib:
                limit[key] = float(limit_element.attrib[key])
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
        mimic=_parse_mimic(_child(element, "mimic")),
    )


def parse_urdf(urdf_path: Path) -> RobotSpec:
    root = ET.parse(urdf_path).getroot()
    if _tag(root) != "robot":
        raise ValueError(f"Expected URDF <robot>, got <{_tag(root)}>")
    links = {
        link.name: link
        for element in _children(root, "link")
        if (link := _parse_link(element)) is not None
    }
    joints = [_parse_joint(element) for element in _children(root, "joint")]
    return RobotSpec(
        links=links,
        joints=joints,
    )
