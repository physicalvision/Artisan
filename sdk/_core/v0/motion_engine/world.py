from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..geometry_qc import (
    _collision_object_from_geometry,
    _geometry_local_aabb,
    _identity4,
    _origin_to_mat4,
    _transform_aabb,
    compute_part_world_transforms,
)
from ..types import Origin
from .resolve import ResolvedCollisionManifest

AABB = tuple[tuple[float, float, float], tuple[float, float, float]]

_CORNER_BITS: tuple[tuple[int, int, int], ...] = tuple(
    (x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)
)


@dataclass(slots=True)
class CollisionPiece:
    part_name: str
    source_element_id: str
    resolved_piece_id: str
    resolution_kind: str
    local_aabb: AABB
    element_transform: Any
    collision_object: Any
    world_aabb: AABB


@dataclass(slots=True)
class CollisionWorld:

    model: object
    manifest: ResolvedCollisionManifest
    pieces: tuple[CollisionPiece, ...]
    part_names: tuple[str, ...] = ()
    part_of_piece: Any = None
    element_matrices: Any = None
    local_corners: Any = None
    piece_lo: Any = None
    piece_hi: Any = None
    part_matrices: Any = None
    _applied: bool = field(default=False, repr=False)

    def invalidate_pose(self) -> None:
        self._applied = False

    def set_pose(self, qpos: dict[str, float]) -> dict[str, Any]:
        import fcl
        import numpy as np

        transforms = compute_part_world_transforms(self.model, qpos)
        identity = _identity4()
        part_matrices = np.empty((len(self.part_names), 4, 4), dtype=float)
        for slot, name in enumerate(self.part_names):
            part_matrices[slot] = transforms.get(name, identity)
        if self._applied:
            changed_parts = np.any(
                part_matrices.view(np.uint64) != self.part_matrices.view(np.uint64), axis=(1, 2)
            )
        else:
            changed_parts = np.ones(len(self.part_names), dtype=bool)

        moved = np.nonzero(changed_parts[self.part_of_piece])[0]
        self._applied = False
        if moved.size:
            left = part_matrices[self.part_of_piece[moved]]
            right = self.element_matrices[moved]
            world = np.empty((moved.size, 3, 4), dtype=float)
            for row in range(3):
                for col in range(4):
                    world[:, row, col] = (
                        left[:, row, 0] * right[:, 0, col]
                        + left[:, row, 1] * right[:, 1, col]
                        + left[:, row, 2] * right[:, 2, col]
                        + left[:, row, 3] * right[:, 3, col]
                    )

            corners = self.local_corners[moved]
            lo = np.empty((moved.size, 3), dtype=float)
            hi = np.empty((moved.size, 3), dtype=float)
            for axis in range(3):
                projected = (
                    world[:, axis, 0, None] * corners[:, :, 0]
                    + world[:, axis, 1, None] * corners[:, :, 1]
                    + world[:, axis, 2, None] * corners[:, :, 2]
                    + world[:, axis, 3, None]
                )
                low = np.full(moved.size, np.inf)
                high = np.full(moved.size, -np.inf)
                for corner in range(projected.shape[1]):
                    value = projected[:, corner]
                    low = np.where(value < low, value, low)
                    high = np.where(value > high, value, high)
                lo[:, axis] = low
                hi[:, axis] = high

            self.piece_lo[moved] = lo
            self.piece_hi[moved] = hi
            rotations = np.ascontiguousarray(world[:, :, :3])
            translations = np.ascontiguousarray(world[:, :, 3])
            pieces = self.pieces
            lo_rows = lo.tolist()
            hi_rows = hi.tolist()
            for index, piece_index in enumerate(moved.tolist()):
                piece = pieces[piece_index]
                piece.world_aabb = (tuple(lo_rows[index]), tuple(hi_rows[index]))
                piece.collision_object.setTransform(
                    fcl.Transform(rotations[index], translations[index])
                )

        self.part_matrices = part_matrices
        self._applied = True
        return transforms


def build_collision_world(
    resolved_model: object,
    manifest: ResolvedCollisionManifest,
    *,
    asset_root: Path,
) -> CollisionWorld:

    import numpy as np

    manifest_by_piece = {record.resolved_piece_id: record for record in manifest.colliders}
    mesh_cache: dict[object, object] = {}
    object_aabb_cache: dict[Path, AABB] = {}
    pieces: list[CollisionPiece] = []
    for part in getattr(resolved_model, "parts", []) or []:
        for index, collision in enumerate(getattr(part, "collisions", []) or []):
            piece_id = str(
                getattr(collision, "resolved_piece_id", None)
                or getattr(collision, "name", None)
                or f"collision_{part.name}_element_{index:03d}"
            )
            record = manifest_by_piece.get(piece_id)
            if record is None:
                raise ValueError(f"resolved collider is absent from manifest: {piece_id}")
            geometry = collision.geometry
            origin = getattr(collision, "origin", None) or Origin()
            element_transform = _origin_to_mat4(origin)
            local_aabb = _geometry_local_aabb(
                geometry,
                asset_root=asset_root,
                _obj_cache=object_aabb_cache,
            )
            collision_object = _collision_object_from_geometry(
                geometry,
                elem_tf=element_transform,
                asset_root=asset_root,
                mesh_cache=mesh_cache,
            )
            pieces.append(
                CollisionPiece(
                    part_name=str(part.name),
                    source_element_id=record.source_element_id,
                    resolved_piece_id=record.resolved_piece_id,
                    resolution_kind=record.resolution_kind,
                    local_aabb=local_aabb,
                    element_transform=element_transform,
                    collision_object=collision_object,
                    world_aabb=_transform_aabb(local_aabb, element_transform),
                )
            )
    pieces.sort(key=lambda value: value.resolved_piece_id)
    if len(pieces) != len(manifest.colliders):
        raise ValueError(
            f"resolved world/manifest piece count mismatch: {len(pieces)} != "
            f"{len(manifest.colliders)}"
        )

    count = len(pieces)
    part_names = tuple(dict.fromkeys(piece.part_name for piece in pieces))
    part_slot = {name: slot for slot, name in enumerate(part_names)}
    corners = np.empty((count, len(_CORNER_BITS), 3), dtype=float)
    element_matrices = np.empty((count, 4, 4), dtype=float)
    piece_lo = np.empty((count, 3), dtype=float)
    piece_hi = np.empty((count, 3), dtype=float)
    for index, piece in enumerate(pieces):
        element_matrices[index] = piece.element_transform
        piece_lo[index] = piece.world_aabb[0]
        piece_hi[index] = piece.world_aabb[1]
        for corner, bits in enumerate(_CORNER_BITS):
            for axis in range(3):
                corners[index, corner, axis] = piece.local_aabb[bits[axis]][axis]
    return CollisionWorld(
        model=resolved_model,
        manifest=manifest,
        pieces=tuple(pieces),
        part_names=part_names,
        part_of_piece=np.asarray(
            [part_slot[piece.part_name] for piece in pieces], dtype=np.intp
        ).reshape(count),
        element_matrices=element_matrices,
        local_corners=corners,
        piece_lo=piece_lo,
        piece_hi=piece_hi,
        part_matrices=None,
    )
