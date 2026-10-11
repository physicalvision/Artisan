from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from protocol_loader import GeometrySpec, _resolve_mesh_source


@dataclass(frozen=True)
class Piece:
    part: str
    faces: np.ndarray
    vertices_close: np.ndarray
    vertices_open: np.ndarray


def _primitive(geometry: GeometrySpec) -> trimesh.Trimesh | None:
    kind = geometry.kind
    attrs = geometry.attrs
    if kind == "box":
        size = [float(value) for value in attrs.get("size", "0 0 0").split()]
        return trimesh.creation.box(extents=size)
    if kind == "cylinder":
        return trimesh.creation.cylinder(
            radius=float(attrs.get("radius", 0.0)), height=float(attrs.get("length", 0.0))
        )
    if kind == "sphere":
        return trimesh.creation.icosphere(radius=float(attrs.get("radius", 0.0)))
    return None


def load_visual_mesh(geometry: GeometrySpec, urdf_dir: Path) -> trimesh.Trimesh | None:
    primitive = _primitive(geometry)
    if primitive is not None:
        return primitive
    if geometry.kind != "mesh":
        return None
    source = _resolve_mesh_source(geometry.attrs["filename"], urdf_dir=urdf_dir)
    loaded = trimesh.load(source, process=False, force="scene")
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            return None
        mesh = trimesh.util.concatenate(
            [
                geom.copy().apply_transform(loaded.graph.get(frame_to=node)[0])
                for node in loaded.graph.nodes_geometry
                for geom in [loaded.geometry[loaded.graph.get(frame_to=node)[1]]]
            ]
        )
    else:
        mesh = loaded
    scale = geometry.attrs.get("scale")
    if scale:
        factors = [float(value) for value in scale.split()]
        mesh = mesh.copy()
        mesh.vertices = mesh.vertices * np.asarray(factors)
    return mesh


def sample_pieces(pieces: list[Piece], num_points: int, seed: int) -> dict[str, np.ndarray]:
    if not pieces:
        raise ValueError("no visual pieces to sample")
    rng = np.random.default_rng(seed)
    areas = np.asarray(
        [trimesh.triangles.area(piece.vertices_close[piece.faces]).sum() for piece in pieces]
    )
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError("visual geometry has zero or non-finite total area")
    counts = rng.multinomial(num_points, areas / areas.sum())

    close, opened, part_ids = [], [], []
    part_names = _stable_part_names(pieces)
    index = {name: number for number, name in enumerate(part_names)}
    for piece, count in zip(pieces, counts, strict=True):
        if count == 0:
            continue
        points_close, points_open = _sample_piece(piece, int(count), rng)
        close.append(points_close)
        opened.append(points_open)
        part_ids.append(np.full(int(count), index[piece.part], dtype=np.int32))
    return {
        "close": np.concatenate(close),
        "open": np.concatenate(opened),
        "part_ids": np.concatenate(part_ids),
        "part_names": part_names,
    }


def _stable_part_names(pieces: list[Piece]) -> tuple[str, ...]:
    names = sorted({piece.part for piece in pieces})

    for root in ("part_root", "body_root"):
        if root in names:
            names.remove(root)
            return (root, *names)
    return tuple(names)


def _sample_piece(
    piece: Piece, count: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    triangles = piece.vertices_close[piece.faces]
    areas = trimesh.triangles.area(triangles)
    total = areas.sum()
    if total <= 0:
        faces = rng.integers(0, len(piece.faces), size=count)
    else:
        faces = rng.choice(len(piece.faces), size=count, replace=True, p=areas / total)
    weights = rng.random((count, 2))
    over = weights.sum(axis=1) > 1.0
    weights[over] = 1.0 - weights[over]
    bary = np.column_stack((weights[:, 0], weights[:, 1], 1.0 - weights.sum(axis=1)))
    close = (piece.vertices_close[piece.faces[faces]] * bary[:, :, None]).sum(axis=1)
    opened = (piece.vertices_open[piece.faces[faces]] * bary[:, :, None]).sum(axis=1)
    return close, opened
