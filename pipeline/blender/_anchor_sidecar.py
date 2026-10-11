from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence

_SCHEMA_VERSION = "artisan-mesh-anchors-v1"
_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_EPS = 1e-9


def _vec3(value: object, *, name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{name} must contain 3 numeric values")
    result = (float(value[0]), float(value[1]), float(value[2]))
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} must be finite")
    return result


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum(left[index] * right[index] for index in range(3))


def _cross(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _normalize(value: tuple[float, float, float], *, name: str) -> tuple[float, float, float]:
    magnitude = math.sqrt(_dot(value, value))
    if magnitude <= _EPS:
        raise ValueError(f"{name} must be non-zero")
    return tuple(component / magnitude for component in value)  # type: ignore[return-value]


def _basis_from_normal(
    normal: object,
    up: object,
) -> tuple[tuple[float, float, float], ...]:
    z_axis = _normalize(_vec3(normal, name="normal/axis"), name="normal/axis")
    up_axis = _normalize(_vec3(up, name="up"), name="up")
    projected_up = tuple(
        up_axis[index] - z_axis[index] * _dot(up_axis, z_axis) for index in range(3)
    )
    y_axis = _normalize(projected_up, name="up projected onto Anchor plane")  # type: ignore[arg-type]
    x_axis = _normalize(_cross(y_axis, z_axis), name="Anchor x axis")
    return (x_axis, y_axis, z_axis)


def _validate_basis(value: object) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError("basis must contain 3 axes")
    axes = tuple(_vec3(axis, name=f"basis[{index}]") for index, axis in enumerate(value))
    for index, axis in enumerate(axes):
        if abs(math.sqrt(_dot(axis, axis)) - 1.0) > 1e-7:
            raise ValueError(f"basis[{index}] must be unit length")
    if any(abs(_dot(axes[left], axes[right])) > 1e-7 for left, right in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("basis axes must be orthogonal")
    if _dot(_cross(axes[0], axes[1]), axes[2]) < 1.0 - 1e-7:
        raise ValueError("basis must be right-handed")
    return axes


def normalize_anchor_declarations(
    declarations: Iterable[Mapping[str, object]] | None,
) -> dict[str, object]:

    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, declaration in enumerate(declarations or ()):
        if not isinstance(declaration, Mapping):
            raise ValueError(f"anchors[{index}] must be a mapping")
        name = str(declaration.get("name", "")).strip()
        if not _NAME_RE.fullmatch(name):
            raise ValueError(f"anchors[{index}].name must be stable snake_case")
        if name in seen:
            raise ValueError(f"duplicate semantic Anchor name: {name}")
        seen.add(name)
        kind = str(declaration.get("kind", "")).strip().lower()
        if kind not in {"point", "plane", "axis"}:
            raise ValueError(f"Anchor {name!r} kind must be point, plane, or axis")
        position = _vec3(declaration.get("position"), name=f"Anchor {name!r} position")
        if "basis" in declaration:
            basis = _validate_basis(declaration["basis"])
        else:
            normal_key = "axis" if kind == "axis" else "normal"
            basis = _basis_from_normal(
                declaration.get(normal_key),
                declaration.get("up", (0.0, 1.0, 0.0)),
            )
        surface_bound = bool(declaration.get("surface_bound", kind == "plane"))
        if kind == "axis" and surface_bound:
            raise ValueError(f"axis Anchor {name!r} cannot be surface_bound")
        tolerance = float(declaration.get("tolerance", 1e-4))
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError(f"Anchor {name!r} tolerance must be positive and finite")
        normalized.append(
            {
                "name": name,
                "kind": kind,
                "position": list(position),
                "basis": [list(axis) for axis in basis],
                "surface_bound": surface_bound,
                "tolerance": tolerance,
            }
        )
    return {"schema_version": _SCHEMA_VERSION, "anchors": normalized}


__all__ = ["normalize_anchor_declarations"]
