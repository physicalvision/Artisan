from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from protocol_fk import _origin_to_matrix
from protocol_loader import RobotSpec, parse_urdf
from protocol_lw_geometry import load_visual_mesh
from protocol_lw_parts import PartPartition, part_partition


def load_prediction(urdf_path: Path) -> tuple[RobotSpec, PartPartition]:
    robot = parse_urdf(urdf_path)
    return robot, part_partition(robot)


class LocalSample:
    __slots__ = ("points", "link_index", "link_names", "part_ids", "part_names")

    def __init__(
        self,
        points: np.ndarray,
        link_index: np.ndarray,
        link_names: tuple[str, ...],
        part_ids: np.ndarray,
        part_names: tuple[str, ...],
    ) -> None:
        self.points = points
        self.link_index = link_index
        self.link_names = link_names
        self.part_ids = part_ids
        self.part_names = part_names


def sample_local(
    robot: RobotSpec, partition: PartPartition, urdf_dir: Path, num_points: int, seed: int
) -> LocalSample:
    rng = np.random.default_rng(seed)
    entries: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for link_name, link in robot.links.items():
        part = partition.link_to_part.get(link_name)
        if part is None:
            continue
        for visual in link.visuals:
            mesh = load_visual_mesh(visual.geometry, urdf_dir)
            if mesh is None or len(mesh.faces) == 0:
                continue
            local = np.asarray(mesh.vertices)
            visual_tf = _origin_to_matrix(visual.origin)
            homogeneous = np.column_stack((local, np.ones(len(local))))
            placed = (visual_tf @ homogeneous.T).T[:, :3]
            entries.append((link_name, part, placed, np.asarray(mesh.faces)))
    if not entries:
        raise ValueError(f"no loadable visual geometry under {urdf_dir}")

    areas = np.asarray(
        [trimesh.triangles.area(verts[faces]).sum() for _, _, verts, faces in entries]
    )
    if areas.sum() <= 0:
        raise ValueError(f"zero total visual area under {urdf_dir}")
    counts = rng.multinomial(num_points, areas / areas.sum())

    link_names = tuple(dict.fromkeys(link for link, _, _, _ in entries))
    link_lookup = {name: index for index, name in enumerate(link_names)}
    part_names = _ordered_parts({part for _, part, _, _ in entries})
    part_lookup = {name: index for index, name in enumerate(part_names)}

    points, link_index, part_ids = [], [], []
    for (link_name, part, verts, faces), count in zip(entries, counts, strict=True):
        if count == 0:
            continue
        triangles = verts[faces]
        face_area = trimesh.triangles.area(triangles)
        total = face_area.sum()
        chosen = (
            rng.choice(len(faces), size=int(count), replace=True, p=face_area / total)
            if total > 0
            else rng.integers(0, len(faces), size=int(count))
        )
        weights = rng.random((int(count), 2))
        over = weights.sum(axis=1) > 1.0
        weights[over] = 1.0 - weights[over]
        bary = np.column_stack((weights[:, 0], weights[:, 1], 1.0 - weights.sum(axis=1)))
        points.append((verts[faces[chosen]] * bary[:, :, None]).sum(axis=1))
        link_index.append(np.full(int(count), link_lookup[link_name], dtype=np.int32))
        part_ids.append(np.full(int(count), part_lookup[part], dtype=np.int32))
    return LocalSample(
        np.concatenate(points),
        np.concatenate(link_index),
        link_names,
        np.concatenate(part_ids),
        part_names,
    )


def _ordered_parts(parts: set[str]) -> tuple[str, ...]:
    names = sorted(parts)
    if "part_root" in names:
        names.remove("part_root")
        return ("part_root", *names)
    return tuple(names)
