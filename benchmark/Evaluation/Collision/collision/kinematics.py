from __future__ import annotations

import math

import numpy as np

from collision.urdf import JointSpec, Origin


def _rpy_to_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = (float(v) for v in rpy)
    cr, sr = (math.cos(roll), math.sin(roll))
    cp, sp = (math.cos(pitch), math.sin(pitch))
    cy, sy = (math.cos(yaw), math.sin(yaw))
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _origin_to_matrix(origin: Origin) -> np.ndarray:
    mat = np.eye(4)
    mat[:3, :3] = _rpy_to_matrix(origin.rpy)
    mat[:3, 3] = np.asarray(origin.xyz, dtype=float)
    return mat


def _normalize_axis(axis: tuple[float, float, float]) -> np.ndarray:
    vec = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return vec / norm


def _axis_angle_matrix(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    x, y, z = _normalize_axis(axis)
    c, s = (math.cos(angle), math.sin(angle))
    cc = 1.0 - c
    return np.array(
        [
            [c + x * x * cc, x * y * cc - z * s, x * z * cc + y * s],
            [y * x * cc + z * s, c + y * y * cc, y * z * cc - x * s],
            [z * x * cc - y * s, z * y * cc + x * s, c + z * z * cc],
        ]
    )


def _motion_transform(joint: JointSpec, q: float) -> np.ndarray:
    mat = np.eye(4)
    if joint.joint_type in {"revolute", "continuous"}:
        mat[:3, :3] = _axis_angle_matrix(joint.axis, float(q))
    elif joint.joint_type == "prismatic":
        mat[:3, 3] = float(q) * _normalize_axis(joint.axis)
    return mat
