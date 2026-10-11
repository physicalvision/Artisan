from __future__ import annotations

import math

import numpy as np
from protocol_loader import Origin


def _rpy_to_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = (float(v) for v in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _origin_to_matrix(origin: Origin) -> np.ndarray:
    mat = np.eye(4)
    mat[:3, :3] = _rpy_to_matrix(origin.rpy)
    mat[:3, 3] = np.asarray(origin.xyz, dtype=float)
    return mat
