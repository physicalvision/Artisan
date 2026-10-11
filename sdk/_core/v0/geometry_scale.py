from __future__ import annotations

import math
import os
from typing import Literal, Sequence

from .assets import AssetContextLike, resolve_asset_root
from .errors import ValidationError
from .geometry_qc import _geometry_local_aabb
from .mesh import CylinderGeometry, SphereGeometry, mesh_from_geometry
from .types import Box, Cylinder, Geometry, Mesh, Sphere

ResizableGeometry = Box | Cylinder | Sphere | Mesh
ScaleMode = Literal["stretch", "uniform"]
TargetSize = tuple[float | None, float | None, float | None]

_FLOAT_TOL = 1e-6
_AXIS_NAMES = ("x", "y", "z")


def scale_geometry_to_size(
    geometry: Geometry,
    target_size: Sequence[float | None],
    *,
    mode: ScaleMode = "stretch",
    asset_root: AssetContextLike | None = None,
    mesh_name: str | os.PathLike[str] | None = None,
    filename: str | os.PathLike[str] | None = None,
) -> Geometry:
    supported = _require_supported_geometry(geometry)
    resolved_mode = _normalize_mode(mode)
    requested_dims = _coerce_target_size(target_size)
    current_dims = _geometry_local_size(supported, asset_root=asset_root)
    final_dims = _resolve_final_dims(current_dims, requested_dims, mode=resolved_mode)

    if isinstance(supported, Box):
        return Box(final_dims)

    if isinstance(supported, Cylinder):
        if _approx_equal(final_dims[0], final_dims[1]):
            radius = (float(final_dims[0]) + float(final_dims[1])) * 0.25
            return Cylinder(radius=radius, length=float(final_dims[2]))
        return _convert_primitive_to_mesh(
            supported,
            current_dims=current_dims,
            final_dims=final_dims,
            mesh_name=mesh_name if mesh_name is not None else filename,
        )

    if isinstance(supported, Sphere):
        if _approx_equal(final_dims[0], final_dims[1]) and _approx_equal(
            final_dims[1], final_dims[2]
        ):
            radius = (float(final_dims[0]) + float(final_dims[1]) + float(final_dims[2])) / 6.0
            return Sphere(radius=radius)
        return _convert_primitive_to_mesh(
            supported,
            current_dims=current_dims,
            final_dims=final_dims,
            mesh_name=mesh_name if mesh_name is not None else filename,
        )

    return _resize_mesh_reference(
        supported,
        current_dims=current_dims,
        final_dims=final_dims,
    )


def _require_supported_geometry(geometry: Geometry) -> ResizableGeometry:
    if not isinstance(geometry, (Box, Cylinder, Sphere, Mesh)):
        raise ValidationError(
            "scale_geometry_to_size only supports Box, Cylinder, Sphere, and Mesh geometry"
        )
    return geometry


def _normalize_mode(mode: str) -> ScaleMode:
    normalized = str(mode or "").strip().lower()
    if normalized not in {"stretch", "uniform"}:
        raise ValidationError("mode must be 'stretch' or 'uniform'")
    return normalized  # type: ignore[return-value]


def _coerce_target_size(target_size: Sequence[float | None]) -> TargetSize:
    if len(target_size) != 3:
        raise ValidationError("target_size must have exactly 3 elements")

    values: list[float | None] = []
    for idx, raw in enumerate(target_size):
        if raw is None:
            values.append(None)
            continue
        value = float(raw)
        if not math.isfinite(value) or value <= 0.0:
            raise ValidationError(
                f"target_size.{_AXIS_NAMES[idx]} must be a positive finite value or None"
            )
        values.append(value)

    if all(value is None for value in values):
        raise ValidationError("target_size must specify at least one axis")
    return (values[0], values[1], values[2])


def _geometry_local_size(
    geometry: ResizableGeometry,
    *,
    asset_root: AssetContextLike | None,
) -> tuple[float, float, float]:
    if isinstance(geometry, Mesh) and geometry.scale is not None:
        if any(float(component) <= 0.0 for component in geometry.scale):
            raise ValidationError(
                "scale_geometry_to_size does not support non-positive mesh.scale components"
            )

    root = resolve_asset_root(asset_root)
    local_min, local_max = _geometry_local_aabb(geometry, asset_root=root, _obj_cache={})
    dims = tuple(float(local_max[idx]) - float(local_min[idx]) for idx in range(3))
    for idx, value in enumerate(dims):
        if not math.isfinite(value):
            raise ValidationError(f"geometry has a non-finite local span along {_AXIS_NAMES[idx]}")
        if value < -_FLOAT_TOL:
            raise ValidationError(f"geometry has a negative local span along {_AXIS_NAMES[idx]}")
    return tuple(0.0 if abs(value) <= _FLOAT_TOL else float(value) for value in dims)  # type: ignore[return-value]


def _resolve_final_dims(
    current_dims: tuple[float, float, float],
    requested_dims: TargetSize,
    *,
    mode: ScaleMode,
) -> tuple[float, float, float]:
    if mode == "stretch":
        return tuple(
            float(requested if requested is not None else current)
            for current, requested in zip(current_dims, requested_dims, strict=True)
        )

    ratios: list[float] = []
    for idx, requested in enumerate(requested_dims):
        if requested is None:
            continue
        current = float(current_dims[idx])
        if current <= _FLOAT_TOL:
            raise ValidationError(
                f"uniform resize requires a positive current span along {_AXIS_NAMES[idx]}"
            )
        ratios.append(float(requested) / current)

    if not ratios:
        raise ValidationError("uniform resize requires at least one specified axis")

    ratio = ratios[0]
    for candidate in ratios[1:]:
        if not _approx_equal(candidate, ratio):
            raise ValidationError(
                "uniform resize requested inconsistent target ratios across specified axes"
            )

    return (
        float(current_dims[0]) * ratio,
        float(current_dims[1]) * ratio,
        float(current_dims[2]) * ratio,
    )


def _scale_factors_for_dims(
    current_dims: tuple[float, float, float],
    final_dims: tuple[float, float, float],
) -> tuple[float, float, float]:
    values: list[float] = []
    for idx, (current, final) in enumerate(zip(current_dims, final_dims, strict=True)):
        current_f = float(current)
        final_f = float(final)
        if current_f <= _FLOAT_TOL:
            if abs(final_f) <= _FLOAT_TOL:
                values.append(1.0)
                continue
            raise ValidationError(
                f"cannot resize geometry along {_AXIS_NAMES[idx]} because the current span is zero"
            )
        values.append(final_f / current_f)
    return (values[0], values[1], values[2])


def _resize_mesh_reference(
    geometry: Mesh,
    *,
    current_dims: tuple[float, float, float],
    final_dims: tuple[float, float, float],
) -> Mesh:
    if geometry.collision_primitives or geometry.collision_decomposition == "required":
        raise ValidationError(
            "cannot resize Mesh geometry with functional collision provenance; "
            "regenerate the source geometry at the intended dimensions"
        )
    current_scale = geometry.scale if geometry.scale is not None else (1.0, 1.0, 1.0)
    factors = _scale_factors_for_dims(current_dims, final_dims)
    new_scale = tuple(float(current_scale[idx]) * float(factors[idx]) for idx in range(3))
    scale_value = (
        None if all(_approx_equal(component, 1.0) for component in new_scale) else new_scale
    )
    return Mesh(
        filename=geometry.filename,
        scale=scale_value,
        source_geometry=geometry.source_geometry,
        source_transform=geometry.source_transform,
        materialized_path=geometry.materialized_path,
    )


def _convert_primitive_to_mesh(
    geometry: Cylinder | Sphere,
    *,
    current_dims: tuple[float, float, float],
    final_dims: tuple[float, float, float],
    mesh_name: str | os.PathLike[str] | None,
) -> Mesh:
    if mesh_name is None:
        raise ValidationError(
            "mesh_name is required when resizing this primitive would convert it into Mesh geometry"
        )

    if isinstance(geometry, Cylinder):
        mesh_geometry = CylinderGeometry(
            radius=float(geometry.radius),
            height=float(geometry.length),
        )
    else:
        mesh_geometry = SphereGeometry(radius=float(geometry.radius))

    mesh_geometry.scale(*_scale_factors_for_dims(current_dims, final_dims))
    return mesh_from_geometry(mesh_geometry, mesh_name)


def _approx_equal(a: float, b: float, *, tol: float = _FLOAT_TOL) -> bool:
    return abs(float(a) - float(b)) <= float(tol) * max(1.0, abs(float(a)), abs(float(b)))


__all__ = ["scale_geometry_to_size"]
