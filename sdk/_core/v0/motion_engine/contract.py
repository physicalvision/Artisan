from __future__ import annotations

import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .._coupling import driver_effective_ranges
from ..types import ArticulationType, Box, Cylinder, Mesh, Origin, Sphere

SIMULATOR_CONTRACT_SCHEMA_VERSION = "simulator-collision-contract-v3"

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class ContractUnsupported(RuntimeError):

    xml_path: str
    element_path: str
    offending: str
    reason: str

    def __str__(self) -> str:
        return (
            f"unsupported collision contract at {self.xml_path}:{self.element_path}: "
            f"{self.offending}: {self.reason}"
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "xml_path": self.xml_path,
            "element_path": self.element_path,
            "offending": self.offending,
            "reason": self.reason,
        }


def canonical_payload(payload: object) -> object:

    if isinstance(payload, Mapping):
        return {str(key): canonical_payload(payload[key]) for key in sorted(payload, key=str)}
    if isinstance(payload, (list, tuple)):
        return [canonical_payload(value) for value in payload]
    if isinstance(payload, float):
        if not math.isfinite(payload):
            raise ValueError("collision contract contains a non-finite float")
        return 0.0 if payload == 0.0 else payload
    return payload


def canonical_contract_payload(payload: object) -> object:

    if isinstance(payload, Mapping):
        return {
            str(key): canonical_contract_payload(payload[key]) for key in sorted(payload, key=str)
        }
    if isinstance(payload, (list, tuple)):
        return [canonical_contract_payload(value) for value in payload]
    if isinstance(payload, float):
        if not math.isfinite(payload):
            raise ValueError("collision contract contains a non-finite float")
        exported = float(f"{payload:.6g}")
        return 0.0 if exported == 0.0 else exported
    return payload


def canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        canonical_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def contract_digest(contract: Mapping[str, object]) -> str:
    encoded = json.dumps(
        canonical_contract_payload(contract),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def first_contract_difference(expected: object, actual: object, path: str = "$") -> str | None:

    expected = canonical_contract_payload(expected)
    actual = canonical_contract_payload(actual)
    if type(expected) is not type(actual):
        return f"{path}: type {type(expected).__name__} != {type(actual).__name__}"
    if isinstance(expected, dict):
        expected_keys = list(expected)
        actual_keys = list(actual)  # type: ignore[arg-type]
        if expected_keys != actual_keys:
            return f"{path}: keys {expected_keys!r} != {actual_keys!r}"
        for key in expected_keys:
            difference = first_contract_difference(expected[key], actual[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):  # type: ignore[arg-type]
            return f"{path}: length {len(expected)} != {len(actual)}"  # type: ignore[arg-type]
        for index, value in enumerate(expected):
            difference = first_contract_difference(value, actual[index], f"{path}[{index}]")  # type: ignore[index]
            if difference is not None:
                return difference
        return None
    if expected != actual:
        return f"{path}: {expected!r} != {actual!r}"
    return None


def _safe_name(value: object, *, fallback: str) -> str:
    normalized = _SAFE_NAME_RE.sub("_", str(value or "").strip()).strip("_")
    return normalized or fallback


def source_element_id(part_name: str, element: object, index: int) -> str:
    existing = getattr(element, "source_element_id", None)
    if isinstance(existing, str) and existing:
        return existing
    element_name = _safe_name(getattr(element, "name", None), fallback=f"element_{index:03d}")
    return f"{part_name}/{element_name}"


def collision_geom_name(part_name: str, element: object, index: int) -> str:
    existing = getattr(element, "resolved_piece_id", None)
    if isinstance(existing, str) and existing:
        return existing
    part = _safe_name(part_name, fallback="part")
    raw_name = getattr(element, "name", None)
    if isinstance(raw_name, str) and raw_name.strip():
        return f"collision_{part}_{_safe_name(raw_name, fallback='element')}"
    return f"collision_{part}_element_{index:03d}"


def mesh_asset_identity(geometry: Mesh) -> tuple[str, tuple[float, float, float]]:
    filename = str(geometry.filename or "").replace("\\", "/")
    scale = tuple(float(value) for value in (geometry.scale or (1.0, 1.0, 1.0)))
    return filename, scale


def mesh_asset_name(filename: str, scale: Sequence[float]) -> str:
    payload = f"{filename}|{','.join(repr(float(value)) for value in scale)}".encode("utf-8")
    return f"mesh_{hashlib.sha256(payload).hexdigest()[:20]}"


def _parse_floats(value: str | None, *, count: int, default: Sequence[float]) -> tuple[float, ...]:
    if value is None:
        return tuple(float(item) for item in default)
    values = tuple(float(item) for item in value.split())
    if len(values) != count:
        raise ValueError(f"expected {count} floats, got {len(values)}")
    return values


def _normalize_vector(values: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = (float(values[0]), float(values[1]), float(values[2]))
    magnitude = math.sqrt(x * x + y * y + z * z)
    if magnitude <= 1e-12:
        raise ValueError("axis must be non-zero")
    return (x / magnitude, y / magnitude, z / magnitude)


def _quat_from_rpy(rpy: Sequence[float]) -> tuple[float, float, float, float]:
    from .._mjcf_export import _fmt_vec
    from .._mjcf_export import _quat_from_rpy as _exported_quat_from_rpy

    exported = _exported_quat_from_rpy(tuple(float(value) for value in rpy))
    return _canonical_quaternion(
        _parse_floats(_fmt_vec(exported), count=4, default=(1.0, 0.0, 0.0, 0.0))
    )


def complete_joint_equalities(
    contract: Mapping[str, object],
    pose: Mapping[str, float],
) -> dict[str, float]:

    completed = {str(name): float(value) for name, value in pose.items()}
    kinematic = contract.get("kinematic")
    if not isinstance(kinematic, Mapping):
        return completed
    equalities = kinematic.get("equalities")
    if not isinstance(equalities, Sequence) or isinstance(equalities, (str, bytes)):
        return completed
    for equality in equalities:
        if not isinstance(equality, Mapping):
            continue
        follower = equality.get("joint1")
        driver = equality.get("joint2")
        coefficients = equality.get("polycoef")
        if (
            not isinstance(follower, str)
            or not isinstance(driver, str)
            or follower in completed
            or driver not in completed
            or not isinstance(coefficients, Sequence)
            or isinstance(coefficients, (str, bytes))
        ):
            continue
        driver_value = completed[driver]
        completed[follower] = sum(
            float(coefficient) * driver_value**power
            for power, coefficient in enumerate(coefficients)
        )
    return completed


def _canonical_quaternion(values: Sequence[float]) -> tuple[float, float, float, float]:
    w, x, y, z = (float(value) for value in values)
    magnitude = math.sqrt(w * w + x * x + y * y + z * z)
    if magnitude <= 1e-12:
        raise ValueError("quaternion must be non-zero")
    normalized = [w / magnitude, x / magnitude, y / magnitude, z / magnitude]
    sign_value = normalized[0]
    if sign_value == 0.0:
        sign_value = next((value for value in normalized[1:] if value != 0.0), 1.0)
    if sign_value < 0.0:
        normalized = [-value for value in normalized]
    return tuple(0.0 if value == 0.0 else value for value in normalized)  # type: ignore[return-value]


def _origin_record(origin: Origin | None) -> dict[str, list[float]]:
    resolved = origin or Origin()
    return {
        "pos": [float(value) for value in resolved.xyz],
        "quat": list(_quat_from_rpy(resolved.rpy)),
    }


def _geometry_source_path(geometry: Mesh, asset_root: Path) -> Path:
    materialized = getattr(geometry, "materialized_path", None)
    path = Path(materialized) if materialized else asset_root / str(geometry.filename or "")
    if not path.is_file():
        raise FileNotFoundError(f"collision mesh is unavailable: {path}")
    return path


def _collider_record_from_model(
    part_name: str,
    element: object,
    index: int,
    *,
    asset_root: Path,
) -> dict[str, object]:
    geometry = getattr(element, "geometry", None)
    record: dict[str, object] = {
        "owning_body": part_name,
        "resolved_piece_id": collision_geom_name(part_name, element, index),
        **_origin_record(getattr(element, "origin", None)),
    }
    if isinstance(geometry, Box):
        record.update(
            geom_type="box",
            size=[float(value) * 0.5 for value in geometry.size],
            scale_xyz=[1.0, 1.0, 1.0],
        )
    elif isinstance(geometry, Cylinder):
        record.update(
            geom_type="cylinder",
            size=[float(geometry.radius), float(geometry.length) * 0.5],
            scale_xyz=[1.0, 1.0, 1.0],
        )
    elif isinstance(geometry, Sphere):
        record.update(
            geom_type="sphere",
            size=[float(geometry.radius)],
            scale_xyz=[1.0, 1.0, 1.0],
        )
    elif isinstance(geometry, Mesh):
        mesh_path = _geometry_source_path(geometry, asset_root)
        record.update(
            geom_type="mesh",
            mesh_sha256=hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
            scale_xyz=[float(value) for value in (geometry.scale or (1.0, 1.0, 1.0))],
        )
    else:
        raise TypeError(f"unsupported resolved collider: {type(geometry).__name__}")
    return record


def _articulation_type(value: object) -> ArticulationType:
    return value if isinstance(value, ArticulationType) else ArticulationType(str(value))


def _model_kinematics(model: object) -> dict[str, object]:
    articulations = list(getattr(model, "articulations", []) or [])
    by_child = {str(art.child): art for art in articulations}
    bodies = []
    for part in getattr(model, "parts", []) or []:
        incoming = by_child.get(str(part.name))
        bodies.append(
            {
                "name": str(part.name),
                "parent": str(incoming.parent) if incoming is not None else None,
                **_origin_record(getattr(incoming, "origin", None)),
            }
        )
    effective_ranges = driver_effective_ranges(model)
    joints: list[dict[str, object]] = []
    for articulation in articulations:
        joint_type = _articulation_type(articulation.articulation_type)
        if joint_type is ArticulationType.FIXED:
            continue
        if joint_type is ArticulationType.FLOATING:
            joints.append(
                {
                    "name": str(articulation.name),
                    "body": str(articulation.child),
                    "type": "free",
                }
            )
            continue
        baked_range = effective_ranges.get(str(articulation.name))
        limits = getattr(articulation, "motion_limits", None)
        joint_range: list[float] | None = None
        limited = False
        if baked_range is not None:
            joint_range = [float(baked_range[0]), float(baked_range[1])]
            limited = True
        elif (
            joint_type in {ArticulationType.REVOLUTE, ArticulationType.PRISMATIC}
            and limits is not None
            and limits.lower is not None
            and limits.upper is not None
        ):
            joint_range = [float(limits.lower), float(limits.upper)]
            limited = True
        joints.append(
            {
                "name": str(articulation.name),
                "body": str(articulation.child),
                "type": "slide" if joint_type is ArticulationType.PRISMATIC else "hinge",
                "axis": list(_normalize_vector(articulation.axis)),
                "pos": [0.0, 0.0, 0.0],
                "ref": 0.0,
                "limited": limited,
                "range": joint_range,
            }
        )
    equalities: list[dict[str, object]] = []
    for articulation in articulations:
        mimic = getattr(articulation, "mimic", None)
        if mimic is None:
            continue
        coefficients = list(mimic.poly) if mimic.poly else [mimic.offset, mimic.multiplier]
        coefficients = (coefficients + [0.0] * 5)[:5]
        equalities.append(
            {
                "joint1": str(articulation.name),
                "joint2": str(mimic.joint),
                "polycoef": [float(value) for value in coefficients],
            }
        )
    return {
        "bodies": sorted(bodies, key=lambda value: str(value["name"])),
        "joints": sorted(joints, key=lambda value: str(value["name"])),
        "equalities": sorted(
            equalities, key=lambda value: (str(value["joint1"]), str(value["joint2"]))
        ),
    }


def contract_from_model(
    model: object,
    *,
    asset_root: Path,
    contact_excludes: Iterable[tuple[str, str]] = (),
    filterparent: str = "disable",
) -> dict[str, object]:
    colliders: list[dict[str, object]] = []
    filters: list[dict[str, object]] = []
    for part in getattr(model, "parts", []) or []:
        for index, element in enumerate(getattr(part, "collisions", []) or []):
            collider = _collider_record_from_model(
                str(part.name), element, index, asset_root=Path(asset_root)
            )
            colliders.append(collider)
            filters.append(
                {
                    "resolved_piece_id": collider["resolved_piece_id"],
                    "contype": 1,
                    "conaffinity": 1,
                    "margin": 0.0,
                    "gap": 0.0,
                }
            )
    excludes = sorted(tuple(sorted((str(a), str(b)))) for a, b in contact_excludes)
    return canonical_contract_payload(
        {
            "simulator_contract_schema_version": SIMULATOR_CONTRACT_SCHEMA_VERSION,
            "colliders": sorted(colliders, key=lambda value: str(value["resolved_piece_id"])),
            "kinematic": _model_kinematics(model),
            "filter": {
                "filterparent": str(filterparent),
                "geoms": sorted(filters, key=lambda value: str(value["resolved_piece_id"])),
                "contact_excludes": [list(value) for value in excludes],
            },
        }
    )  # type: ignore[return-value]


def _element_path(element: ET.Element, parents: Mapping[int, ET.Element]) -> str:
    values = []
    current: ET.Element | None = element
    while current is not None:
        name = current.get("name")
        values.append(f"{current.tag}[@name='{name}']" if name else current.tag)
        current = parents.get(id(current))
    return "/" + "/".join(reversed(values))


def _unsupported(
    xml_path: Path,
    element: ET.Element,
    parents: Mapping[int, ET.Element],
    offending: str,
    reason: str,
) -> ContractUnsupported:
    return ContractUnsupported(
        xml_path=str(xml_path),
        element_path=_element_path(element, parents),
        offending=offending,
        reason=reason,
    )


def _check_allowed_attributes(
    xml_path: Path,
    element: ET.Element,
    parents: Mapping[int, ET.Element],
    allowed: set[str],
) -> None:
    unexpected = sorted(set(element.attrib) - allowed)
    if unexpected:
        raise _unsupported(
            xml_path,
            element,
            parents,
            f"attribute {unexpected[0]!r}",
            "attribute is outside the collision-contract MJCF dialect",
        )


def _xml_default_geom(root: ET.Element, class_name: str) -> dict[str, str]:
    for default in root.findall("default"):
        for nested in default.findall("default"):
            if nested.get("class") == class_name:
                geom = nested.find("geom")
                return dict(geom.attrib) if geom is not None else {}
    return {}


def _validate_default_dialect(
    root: ET.Element,
    *,
    xml_path: Path,
    parents: Mapping[int, ET.Element],
) -> None:
    for default in root.findall("default"):
        _check_allowed_attributes(xml_path, default, parents, set())
        for child in default:
            if child.tag != "default":
                raise _unsupported(
                    xml_path,
                    child,
                    parents,
                    f"default {child.tag}",
                    "global body/joint/geom defaults are outside the exporter dialect",
                )
            _check_allowed_attributes(xml_path, child, parents, {"class"})
            class_name = child.get("class")
            if class_name not in {"visual", "collision"}:
                raise _unsupported(
                    xml_path,
                    child,
                    parents,
                    f"default class {class_name!r}",
                    "only exporter visual/collision default classes are modeled",
                )
            for nested in child:
                if nested.tag != "geom":
                    raise _unsupported(
                        xml_path,
                        nested,
                        parents,
                        f"default {nested.tag}",
                        "joint/body defaults can alter collision kinematics",
                    )
                allowed = (
                    {"group", "contype", "conaffinity"}
                    if class_name == "visual"
                    else {"group", "contype", "conaffinity", "margin", "gap"}
                )
                _check_allowed_attributes(xml_path, nested, parents, allowed)


def _xml_mesh_assets(
    root: ET.Element,
    *,
    xml_path: Path,
    parents: Mapping[int, ET.Element],
) -> dict[str, dict[str, object]]:
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    result: dict[str, dict[str, object]] = {}
    asset = root.find("asset")
    if asset is None:
        return result
    for mesh in asset.findall("mesh"):
        _check_allowed_attributes(xml_path, mesh, parents, {"name", "file", "scale", "inertia"})
        if mesh.get("inertia", "shell") != "shell":
            raise _unsupported(
                xml_path,
                mesh,
                parents,
                "mesh inertia",
                "only inertia='shell' is in the exporter dialect",
            )
        if mesh.get("refpos") is not None or mesh.get("refquat") is not None:
            raise _unsupported(
                xml_path,
                mesh,
                parents,
                "mesh reference transform",
                "mesh refpos/refquat is not modeled",
            )
        name = mesh.get("name")
        filename = mesh.get("file")
        if not name or not filename:
            raise _unsupported(xml_path, mesh, parents, "mesh", "mesh name and file are required")
        mesh_path = (xml_path.parent / meshdir / filename).resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(f"staged collision mesh is unavailable: {mesh_path}")
        result[name] = {
            "mesh_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
            "scale_xyz": list(_parse_floats(mesh.get("scale"), count=3, default=(1, 1, 1))),
        }
    return result


def contract_from_mjcf(
    xml_path: Path,
    *,
    strip_scene_rig: bool = False,
) -> dict[str, object]:

    xml_path = Path(xml_path)
    root = ET.parse(xml_path).getroot()
    parents = {id(child): parent for parent in root.iter() for child in parent}
    compiler = root.find("compiler")
    if compiler is not None:
        _check_allowed_attributes(
            xml_path,
            compiler,
            parents,
            {"angle", "meshdir", "autolimits", "balanceinertia"},
        )
        if compiler.get("angle", "degree") != "radian":
            raise _unsupported(
                xml_path,
                compiler,
                parents,
                "compiler angle",
                "the exporter collision dialect requires radians",
            )
    _validate_default_dialect(root, xml_path=xml_path, parents=parents)
    contact = root.find("contact")
    if contact is not None and contact.find("pair") is not None:
        pair = contact.find("pair")
        assert pair is not None
        raise _unsupported(
            xml_path,
            pair,
            parents,
            "contact pair",
            "explicit contact pairs are not modeled",
        )
    mesh_assets = _xml_mesh_assets(root, xml_path=xml_path, parents=parents)
    collision_defaults = _xml_default_geom(root, "collision")
    colliders: list[dict[str, object]] = []
    filters: list[dict[str, object]] = []
    bodies: list[dict[str, object]] = []
    joints: list[dict[str, object]] = []

    def visit_body(body: ET.Element, parent_name: str | None) -> None:
        _check_allowed_attributes(xml_path, body, parents, {"name", "pos", "quat"})
        for child in body:
            if child.tag not in {"body", "freejoint", "geom", "inertial", "joint"}:
                raise _unsupported(
                    xml_path,
                    child,
                    parents,
                    f"body child {child.tag}",
                    "body child is outside the collision-contract MJCF dialect",
                )
        body_name = body.get("name")
        if not body_name:
            raise _unsupported(xml_path, body, parents, "body name", "body names are required")
        bodies.append(
            {
                "name": body_name,
                "parent": parent_name,
                "pos": list(_parse_floats(body.get("pos"), count=3, default=(0, 0, 0))),
                "quat": list(
                    _canonical_quaternion(
                        _parse_floats(body.get("quat"), count=4, default=(1, 0, 0, 0))
                    )
                ),
            }
        )
        for freejoint in body.findall("freejoint"):
            if (
                strip_scene_rig
                and parent_name is None
                and freejoint.get("name") == f"{body_name}_free"
            ):
                continue
            _check_allowed_attributes(xml_path, freejoint, parents, {"name"})
            joints.append(
                {"name": str(freejoint.get("name") or ""), "body": body_name, "type": "free"}
            )
        for joint in body.findall("joint"):
            _check_allowed_attributes(
                xml_path,
                joint,
                parents,
                {
                    "name",
                    "type",
                    "axis",
                    "pos",
                    "ref",
                    "range",
                    "limited",
                    "damping",
                    "frictionloss",
                    "solreffriction",
                    "solimpfriction",
                    "stiffness",
                    "springref",
                },
            )
            name = joint.get("name")
            if not name:
                raise _unsupported(
                    xml_path, joint, parents, "joint name", "joint names are required"
                )
            range_value = joint.get("range")
            limited = joint.get("limited") != "false" and range_value is not None
            joints.append(
                {
                    "name": name,
                    "body": body_name,
                    "type": joint.get("type", "hinge"),
                    "axis": list(
                        _normalize_vector(
                            _parse_floats(joint.get("axis"), count=3, default=(0, 0, 1))
                        )
                    ),
                    "pos": list(_parse_floats(joint.get("pos"), count=3, default=(0, 0, 0))),
                    "ref": float(joint.get("ref", "0")),
                    "limited": limited,
                    "range": (
                        list(_parse_floats(range_value, count=2, default=(0, 0)))
                        if range_value is not None
                        else None
                    ),
                }
            )
        for geom in body.findall("geom"):
            geom_class = geom.get("class")
            if geom.get("name") == "scene_ground" and strip_scene_rig:
                continue
            if geom_class == "visual":
                continue
            if geom_class != "collision":
                raise _unsupported(
                    xml_path,
                    geom,
                    parents,
                    "geom class",
                    "non-rig object geoms must use the collision or visual class",
                )
            _check_allowed_attributes(
                xml_path,
                geom,
                parents,
                {
                    "class",
                    "name",
                    "type",
                    "size",
                    "mesh",
                    "pos",
                    "quat",
                    "contype",
                    "conaffinity",
                    "margin",
                    "gap",
                },
            )
            piece_id = geom.get("name")
            if not piece_id:
                raise _unsupported(
                    xml_path, geom, parents, "geom name", "collision geom names are required"
                )
            geom_type = geom.get("type", "sphere")
            record: dict[str, object] = {
                "owning_body": body_name,
                "resolved_piece_id": piece_id,
                "geom_type": geom_type,
                "pos": list(_parse_floats(geom.get("pos"), count=3, default=(0, 0, 0))),
                "quat": list(
                    _canonical_quaternion(
                        _parse_floats(geom.get("quat"), count=4, default=(1, 0, 0, 0))
                    )
                ),
            }
            if geom_type == "mesh":
                asset_name = geom.get("mesh")
                if not asset_name or asset_name not in mesh_assets:
                    raise _unsupported(
                        xml_path, geom, parents, "mesh reference", "collision mesh asset is missing"
                    )
                record.update(mesh_assets[asset_name])
            else:
                size_count = {"box": 3, "cylinder": 2, "sphere": 1}.get(geom_type)
                if size_count is None:
                    raise _unsupported(
                        xml_path,
                        geom,
                        parents,
                        f"geom type {geom_type!r}",
                        "geom type is outside the collision-contract dialect",
                    )
                record["size"] = list(
                    _parse_floats(geom.get("size"), count=size_count, default=(0,) * size_count)
                )
                record["scale_xyz"] = [1.0, 1.0, 1.0]
            effective = {**collision_defaults, **geom.attrib}
            filters.append(
                {
                    "resolved_piece_id": piece_id,
                    "contype": int(effective.get("contype", "1")),
                    "conaffinity": int(effective.get("conaffinity", "1")),
                    "margin": float(effective.get("margin", "0")),
                    "gap": float(effective.get("gap", "0")),
                }
            )
            colliders.append(record)
        for child in body.findall("body"):
            visit_body(child, body_name)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"MJCF has no worldbody: {xml_path}")
    for body in worldbody.findall("body"):
        visit_body(body, None)

    equalities: list[dict[str, object]] = []
    equality = root.find("equality")
    if equality is not None:
        for child in equality:
            if child.tag != "joint":
                raise _unsupported(
                    xml_path,
                    child,
                    parents,
                    f"equality {child.tag}",
                    "only joint equality constraints are modeled",
                )
            _check_allowed_attributes(
                xml_path, child, parents, {"joint1", "joint2", "polycoef", "active"}
            )
            if child.get("active", "true") == "false":
                raise _unsupported(
                    xml_path,
                    child,
                    parents,
                    "inactive equality",
                    "inactive equality constraints are not emitted by the exporter",
                )
            equalities.append(
                {
                    "joint1": str(child.get("joint1") or ""),
                    "joint2": str(child.get("joint2") or ""),
                    "polycoef": list(
                        _parse_floats(child.get("polycoef"), count=5, default=(0, 1, 0, 0, 0))
                    ),
                }
            )

    filterparent = "enable"
    flag = root.find("option/flag")
    if flag is not None:
        _check_allowed_attributes(xml_path, flag, parents, {"filterparent"})
        filterparent = flag.get("filterparent", "enable")
    excludes: list[list[str]] = []
    if contact is not None:
        for child in contact:
            if child.tag != "exclude":
                raise _unsupported(
                    xml_path,
                    child,
                    parents,
                    f"contact {child.tag}",
                    "contact node is outside the collision-contract dialect",
                )
            _check_allowed_attributes(xml_path, child, parents, {"body1", "body2"})
            excludes.append(sorted((str(child.get("body1") or ""), str(child.get("body2") or ""))))

    return canonical_contract_payload(
        {
            "simulator_contract_schema_version": SIMULATOR_CONTRACT_SCHEMA_VERSION,
            "colliders": sorted(colliders, key=lambda value: str(value["resolved_piece_id"])),
            "kinematic": {
                "bodies": sorted(bodies, key=lambda value: str(value["name"])),
                "joints": sorted(joints, key=lambda value: str(value["name"])),
                "equalities": sorted(
                    equalities,
                    key=lambda value: (str(value["joint1"]), str(value["joint2"])),
                ),
            },
            "filter": {
                "filterparent": filterparent,
                "geoms": sorted(filters, key=lambda value: str(value["resolved_piece_id"])),
                "contact_excludes": sorted(excludes),
            },
        }
    )  # type: ignore[return-value]
