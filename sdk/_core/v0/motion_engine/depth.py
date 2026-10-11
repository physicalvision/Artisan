from __future__ import annotations

from dataclasses import dataclass

from ..geometry_qc import _aabb_intersection_depth, _aabbs_touch_or_overlap
from .world import CollisionPiece, CollisionWorld


@dataclass(frozen=True, slots=True)
class DepthResult:
    status: str
    colliding: bool
    depth_m: float | None
    method: str
    axis: str
    piece_pair: tuple[str, str] | None = None


def _axis_of_normal(normal: object) -> str:
    values = [abs(float(normal[index])) for index in range(3)]  # type: ignore[index]
    return ("x", "y", "z")[max(range(3), key=values.__getitem__)]


def _selected_pieces(
    world: CollisionWorld,
    *,
    part_pair: tuple[str, str] | None,
    element_pair: tuple[str, str] | None,
) -> tuple[list[CollisionPiece], list[CollisionPiece]]:
    if part_pair is None and element_pair is None:
        raise ValueError("part_pair or element_pair is required")
    if element_pair is not None:
        left = [piece for piece in world.pieces if piece.source_element_id == element_pair[0]]
        right = [piece for piece in world.pieces if piece.source_element_id == element_pair[1]]
    else:
        assert part_pair is not None
        left = [piece for piece in world.pieces if piece.part_name == part_pair[0]]
        right = [piece for piece in world.pieces if piece.part_name == part_pair[1]]
    return left, right


def evaluate_depth(
    world: CollisionWorld,
    *,
    qpos: dict[str, float],
    part_pair: tuple[str, str] | None = None,
    element_pair: tuple[str, str] | None = None,
    pose_is_set: bool = False,
) -> DepthResult:

    import fcl

    if not pose_is_set:
        world.set_pose(qpos)
    left, right = _selected_pieces(world, part_pair=part_pair, element_pair=element_pair)
    request = fcl.CollisionRequest(enable_contact=True, num_max_contacts=64)
    any_collision = False
    valid: list[tuple[float, str, tuple[str, str]]] = []
    first_pair: tuple[str, str] | None = None
    for piece_a in left:
        for piece_b in right:
            if not _aabbs_touch_or_overlap(piece_a.world_aabb, piece_b.world_aabb, tol=0.0):
                continue
            result = fcl.CollisionResult()
            if (
                int(
                    fcl.collide(piece_a.collision_object, piece_b.collision_object, request, result)
                )
                <= 0
            ):
                continue
            any_collision = True
            pair = (piece_a.resolved_piece_id, piece_b.resolved_piece_id)
            first_pair = first_pair or pair
            contacts = getattr(result, "contacts", None) or []
            if not contacts:
                continue
            overlap = _aabb_intersection_depth(piece_a.world_aabb, piece_b.world_aabb)
            cap = max(0.0, min(float(value) for value in overlap))
            for contact in contacts:
                depth = max(0.0, float(contact.penetration_depth))
                if depth > cap + max(1e-7, cap * 1e-3):
                    continue
                valid.append((depth, _axis_of_normal(contact.normal), pair))
    if not any_collision:
        return DepthResult("ok", False, 0.0, "epa-pieces", "x")
    if not valid:
        return DepthResult(
            "degenerate",
            True,
            None,
            "epa-pieces",
            "x",
            first_pair,
        )
    depth, axis, pair = max(valid, key=lambda value: value[0])
    return DepthResult("ok", True, depth, "epa-pieces", axis, pair)
