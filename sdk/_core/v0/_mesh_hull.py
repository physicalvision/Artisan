
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

MUJOCO_MIN_MESH_VOLUME = 1e-15
RESIDUE_MAX_MIDDLE_EXTENT = 5e-5
MIN_SHEET_THICKNESS = 1e-4

HullKind = Literal["ok", "thin", "unhullable"]


@dataclass(frozen=True, slots=True)
class MeshHullCheck:

    kind: HullKind
    volume: float
    extent: tuple[float, float, float]
    centre: tuple[float, float, float]

    @property
    def is_residue(self) -> bool:
        return self.kind != "ok" and sorted(self.extent)[1] < RESIDUE_MAX_MIDDLE_EXTENT


def check_mesh_hull(vertices: NDArray[np.float64]) -> MeshHullCheck:
    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    lower, upper = points.min(axis=0), points.max(axis=0)
    extent = _vec3(upper - lower)
    centre = _vec3((upper + lower) / 2.0)
    if not all(extent):
        return MeshHullCheck("unhullable", 0.0, extent, centre)
    bound = _spanning_tetra_volume(points)
    if bound >= MUJOCO_MIN_MESH_VOLUME:
        return MeshHullCheck("ok", bound, extent, centre)
    if len(np.unique(points, axis=0)) < 4:
        return MeshHullCheck("unhullable", 0.0, extent, centre)
    volume = _hull_volume(points)
    if volume is None or not np.isfinite(volume) or volume <= 0.0:
        return MeshHullCheck("unhullable", 0.0, extent, centre)
    kind: HullKind = "ok" if volume >= MUJOCO_MIN_MESH_VOLUME else "thin"
    return MeshHullCheck(kind, volume, extent, centre)


def check_mesh_file_hull(
    path: str | Path, scale: Sequence[float] | None = None
) -> MeshHullCheck | None:
    vertices = load_mesh_vertices(path, scale)
    return None if vertices is None else check_mesh_hull(vertices)


def load_mesh_vertices(
    path: str | Path, scale: Sequence[float] | None = None
) -> NDArray[np.float64] | None:
    import trimesh

    try:
        mesh = trimesh.load(Path(path), force="mesh", process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
    except Exception:  # noqa: BLE001 - an unreadable mesh is left to its consumer
        return None
    if vertices.size == 0 or not np.isfinite(vertices).all():
        return None
    if scale is not None:
        vertices = vertices * np.asarray(tuple(float(value) for value in scale), dtype=np.float64)
    return vertices


def _vec3(values: NDArray[np.float64]) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def _spanning_tetra_volume(points: NDArray[np.float64]) -> float:
    a = points[int(np.argmin(points[:, 0]))]
    b = points[int(np.argmax(np.linalg.norm(points - a, axis=1)))]
    ab = b - a
    c = points[int(np.argmax(np.linalg.norm(np.cross(points - a, ab), axis=1)))]
    normal = np.cross(ab, c - a)
    heights = (points - a) @ normal
    return float(np.max(np.abs(heights)) / 6.0)


def _hull_volume(points: NDArray[np.float64]) -> float | None:
    from scipy.spatial import ConvexHull, QhullError

    try:
        hull = ConvexHull(points, qhull_options="QbB Qt")
    except (QhullError, ValueError):
        return None
    anchor = points[hull.vertices].mean(axis=0)
    corners = points[hull.simplices] - anchor
    return float(np.abs(np.linalg.det(corners)).sum() / 6.0)
