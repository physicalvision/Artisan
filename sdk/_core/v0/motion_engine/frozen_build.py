
from __future__ import annotations

import dataclasses
import json
from enum import Enum
from pathlib import Path
from typing import Any

from .. import types
from ..articulated_object import ArticulatedObject
from ..assets import AssetContext


def _model_types() -> dict[str, type]:
    return {
        cls.__name__: cls
        for cls in (ArticulatedObject, AssetContext, *vars(types).values())
        if isinstance(cls, type) and dataclasses.is_dataclass(cls)
    }


def _encode(value: Any) -> Any:
    if isinstance(value, AssetContext):
        return {"type": "AssetContext"}
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        if type(value) not in _model_types().values():
            raise TypeError(f"unsupported frozen SDK value: {type(value).__name__}")
        return {
            "type": type(value).__name__,
            "allow_explicit_collisions": bool(
                getattr(value, "_sdk_allow_explicit_collisions", False)
            ),
            "fields": {
                field.name: _encode(getattr(value, field.name))
                for field in dataclasses.fields(value)
                if field.name not in {"_part_index", "_articulation_index", "materialized_path"}
            },
        }
    if isinstance(value, Path):
        return {"type": "Path", "value": str(value)}
    if isinstance(value, dict):
        return {"type": "dict", "items": [[_encode(k), _encode(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "items": [_encode(item) for item in value]}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported frozen SDK value: {type(value).__name__}")


def _decode(value: Any, root: Path) -> Any:
    if not isinstance(value, dict):
        return value
    kind = value["type"]
    if kind == "AssetContext":
        return AssetContext(root)
    if kind == "Path":
        return Path(value["value"])
    if kind == "dict":
        return {_decode(k, root): _decode(v, root) for k, v in value["items"]}
    if kind in {"list", "tuple"}:
        items = [_decode(item, root) for item in value["items"]]
        return tuple(items) if kind == "tuple" else items
    cls = _model_types()[kind]
    instance = object.__new__(cls)
    for field in dataclasses.fields(cls):
        if field.name in value["fields"]:
            restored = _decode(value["fields"][field.name], root)
        elif field.default is not dataclasses.MISSING:
            restored = field.default
        elif field.default_factory is not dataclasses.MISSING:
            restored = field.default_factory()
        else:
            raise ValueError(f"missing frozen field: {kind}.{field.name}")
        object.__setattr__(instance, field.name, restored)
    if isinstance(instance, types.Mesh):
        object.__setattr__(instance, "materialized_path", str(root / instance.filename))
    if isinstance(instance, ArticulatedObject):
        instance._part_index = {part.name: part for part in instance.parts}
        instance._articulation_index = {joint.name: joint for joint in instance.articulations}
        instance._sdk_allow_explicit_collisions = bool(value.get("allow_explicit_collisions"))
    if isinstance(instance, types.Articulation):
        instance.articulation_type = types.ArticulationType(instance.articulation_type)
    return instance


def write_frozen_build(path: Path, raw_model: object, resolved_model: object) -> None:
    payload = {"raw_model": _encode(raw_model), "resolved_model": _encode(resolved_model)}
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def read_frozen_build(path: Path) -> tuple[object, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _decode(payload["raw_model"], path.parent), _decode(
        payload["resolved_model"], path.parent
    )
