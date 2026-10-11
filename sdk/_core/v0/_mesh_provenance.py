from __future__ import annotations

import math

from .types import Box, Cylinder, Mat4, Mesh, Sphere, Vec3

PrimitiveGeometry = Box | Cylinder | Sphere
Mat3 = tuple[Vec3, Vec3, Vec3]
_AFFINE_TOL = 1e-6


def decompose_mesh_primitive_provenance(
    geometry: Mesh,
) -> tuple[PrimitiveGeometry, Mat4] | None:
    source = geometry.source_geometry
    if not isinstance(source, (Box, Cylinder, Sphere)):
        return None

    affine = _mesh_source_affine(geometry)
    return _decompose_primitive_affine(source, affine)


def _mesh_source_affine(geometry: Mesh) -> Mat4:
    base = geometry.source_transform or _identity4()
    if geometry.scale is None:
        return base

    sx, sy, sz = (float(v) for v in geometry.scale)
    scale_tf: Mat4 = (
        (sx, 0.0, 0.0, 0.0),
        (0.0, sy, 0.0, 0.0),
        (0.0, 0.0, sz, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    return _mat4_mul(scale_tf, base)


def _decompose_primitive_affine(
    primitive: PrimitiveGeometry,
    affine: Mat4,
) -> tuple[PrimitiveGeometry, Mat4] | None:
    linear = tuple(tuple(float(affine[row][col]) for col in range(3)) for row in range(3))
    translation = (
        float(affine[0][3]),
        float(affine[1][3]),
        float(affine[2][3]),
    )
    columns = [(linear[0][col], linear[1][col], linear[2][col]) for col in range(3)]
    scales = [math.sqrt(sum(component * component for component in col)) for col in columns]
    if any(scale <= _AFFINE_TOL for scale in scales):
        return None

    for i in range(3):
        for j in range(i + 1, 3):
            dot = sum(columns[i][k] * columns[j][k] for k in range(3))
            limit = _AFFINE_TOL * max(1.0, scales[i] * scales[j])
            if abs(dot) > limit:
                return None

    rot_cols = [tuple(component / scales[idx] for component in columns[idx]) for idx in range(3)]
    rotation = tuple(tuple(rot_cols[col][row] for col in range(3)) for row in range(3))
    det = _mat3_det(rotation)
    if abs(det) <= _AFFINE_TOL:
        return None
    if det < 0.0:
        rot_cols[0] = tuple(-component for component in rot_cols[0])
        rotation = tuple(tuple(rot_cols[col][row] for col in range(3)) for row in range(3))

    sx, sy, sz = (float(scale) for scale in scales)
    if isinstance(primitive, Box):
        collision_geometry: PrimitiveGeometry = Box(
            (
                float(primitive.size[0]) * sx,
                float(primitive.size[1]) * sy,
                float(primitive.size[2]) * sz,
            )
        )
    elif isinstance(primitive, Cylinder):
        if not _approx_equal(sx, sy):
            return None
        collision_geometry = Cylinder(
            radius=float(primitive.radius) * ((sx + sy) * 0.5),
            length=float(primitive.length) * sz,
        )
    else:
        if not (_approx_equal(sx, sy) and _approx_equal(sx, sz)):
            return None
        collision_geometry = Sphere(radius=float(primitive.radius) * ((sx + sy + sz) / 3.0))

    rigid_tf: Mat4 = (
        (rotation[0][0], rotation[0][1], rotation[0][2], translation[0]),
        (rotation[1][0], rotation[1][1], rotation[1][2], translation[1]),
        (rotation[2][0], rotation[2][1], rotation[2][2], translation[2]),
        (0.0, 0.0, 0.0, 1.0),
    )
    return collision_geometry, rigid_tf


def _approx_equal(a: float, b: float, *, tol: float = _AFFINE_TOL) -> bool:
    return abs(float(a) - float(b)) <= float(tol) * max(1.0, abs(float(a)), abs(float(b)))


def _mat3_det(mat: Mat3) -> float:
    return (
        mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1])
        - mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0])
        + mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0])
    )


def _identity4() -> Mat4:
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _mat4_mul(a: Mat4, b: Mat4) -> Mat4:
    out: list[list[float]] = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            out[i][j] = (
                float(a[i][0]) * float(b[0][j])
                + float(a[i][1]) * float(b[1][j])
                + float(a[i][2]) * float(b[2][j])
                + float(a[i][3]) * float(b[3][j])
            )
    return (
        (out[0][0], out[0][1], out[0][2], out[0][3]),
        (out[1][0], out[1][1], out[1][2], out[1][3]),
        (out[2][0], out[2][1], out[2][2], out[2][3]),
        (out[3][0], out[3][1], out[3][2], out[3][3]),
    )
