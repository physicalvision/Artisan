from __future__ import annotations

import os
import pickle
import shutil
import signal
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from itertools import compress
from pathlib import Path
from typing import Any, Callable, Iterable

from ..geometry_qc import _aabb_intersection_depth, _aabb_intersection_volume
from .pose_plan import PoseTable
from .world import AABB, CollisionPiece, CollisionWorld

_BROAD_PHASE_ROW_CHUNK = 256

_MEMO_PLAN_CACHE_LIMIT = 1024


@dataclass(frozen=True, slots=True)
class ContactRow:
    pose_id: int
    piece_a: str
    piece_b: str
    source_element_a: str
    source_element_b: str
    part_a: str
    part_b: str
    colliding: bool
    min_distance: float
    overlap_depth: tuple[float, float, float]
    overlap_volume: float
    aabb_a: AABB
    aabb_b: AABB


@dataclass(frozen=True, slots=True)
class SweepStats:
    n_poses: int
    n_pairs_broad: int
    n_pairs_narrow: int
    n_pairs_memoized: int = 0
    n_shards: int = 1


@dataclass(frozen=True, slots=True)
class ContactTable:
    rows: tuple[ContactRow, ...]
    stats: SweepStats

    def rows_for_pose(self, pose_id: int) -> tuple[ContactRow, ...]:
        return tuple(row for row in self.rows if row.pose_id == pose_id)


def _aabbs_within_margin(a: AABB, b: AABB, margin: float) -> bool:
    return all(
        a[0][axis] <= b[1][axis] + margin and b[0][axis] <= a[1][axis] + margin for axis in range(3)
    )


def _candidate_pair_keys(
    lo,
    hi,
    part_codes,
    margin: float,
    n_pieces: int,
    *,
    rows=None,
    moved=None,
    chunk: int = _BROAD_PHASE_ROW_CHUNK,
):
    import numpy as np

    empty = np.empty(0, dtype=np.int64)
    if n_pieces == 0:
        return empty, empty, empty
    if rows is None:
        rows = np.arange(n_pieces, dtype=np.intp)
    if moved is None:
        moved = np.ones(n_pieces, dtype=bool)
    if rows.size == 0:
        return empty, empty, empty

    hi_margin = hi + margin
    left_blocks: list[Any] = []
    right_blocks: list[Any] = []
    for start in range(0, rows.size, chunk):
        block = rows[start : start + chunk]
        within = np.all(
            (lo[block][:, None, :] <= hi_margin[None, :, :])
            & (lo[None, :, :] <= hi_margin[block][:, None, :]),
            axis=2,
        )
        within &= part_codes[block][:, None] != part_codes[None, :]
        block_rows, cols = np.nonzero(within)
        block_rows = block[block_rows]
        keep = (cols > block_rows) | ~moved[cols]
        block_rows = block_rows[keep]
        cols = cols[keep]
        left_blocks.append(np.minimum(block_rows, cols))
        right_blocks.append(np.maximum(block_rows, cols))

    left = np.concatenate(left_blocks)
    right = np.concatenate(right_blocks)
    keys = left.astype(np.int64) * n_pieces + right
    keys.sort()
    first = keys // n_pieces
    return keys, first, keys - first * n_pieces


@dataclass(frozen=True, slots=True)
class _MemoPlan:

    n_static_candidates: int
    row_keys: Any
    rows: tuple[ContactRow, ...]


@dataclass(slots=True)
class RestMemo:

    n_pieces: int = -1
    margin: float = float("nan")
    part_names: tuple[str, ...] = ()
    base_part_matrices: Any = None
    candidate_i: Any = None
    candidate_j: Any = None
    row_keys: Any = None
    row_i: Any = None
    row_j: Any = None
    rows: tuple[ContactRow, ...] = ()
    base_aabbs: tuple[AABB, ...] = ()
    plans: dict[bytes, _MemoPlan] = field(default_factory=dict, repr=False)

    @property
    def populated(self) -> bool:
        return self.base_part_matrices is not None

    def populate(
        self,
        world: CollisionWorld,
        margin: float,
        candidate_i,
        candidate_j,
        row_keys,
        row_i,
        row_j,
        rows: tuple[ContactRow, ...],
    ) -> None:
        self.n_pieces = len(world.pieces)
        self.margin = float(margin)
        self.part_names = tuple(world.part_names)
        self.base_part_matrices = world.part_matrices.copy()
        self.base_aabbs = tuple(piece.world_aabb for piece in world.pieces)
        self.candidate_i = candidate_i
        self.candidate_j = candidate_j
        self.row_keys = row_keys
        self.row_i = row_i
        self.row_j = row_j
        self.rows = rows
        self.plans.clear()

    def check(self, world: CollisionWorld, margin: float) -> None:
        if (
            self.n_pieces != len(world.pieces)
            or self.margin != float(margin)
            or self.part_names != tuple(world.part_names)
        ):
            raise ValueError(
                "rest memo does not belong to this sweep: "
                f"pieces {self.n_pieces} vs {len(world.pieces)}, "
                f"margin {self.margin} vs {margin}, "
                f"parts {self.part_names} vs {tuple(world.part_names)}"
            )

    def moved_parts(self, world: CollisionWorld):
        import numpy as np

        return np.any(
            world.part_matrices.view(np.uint64) != self.base_part_matrices.view(np.uint64),
            axis=(1, 2),
        )

    def plan(self, moved_parts, part_of_piece) -> _MemoPlan:
        import numpy as np

        cache_key = moved_parts.tobytes()
        cached = self.plans.get(cache_key)
        if cached is not None:
            return cached
        static = ~moved_parts[part_of_piece]
        n_static = int(np.count_nonzero(static[self.candidate_i] & static[self.candidate_j]))
        row_mask = static[self.row_i] & static[self.row_j]
        plan = _MemoPlan(
            n_static_candidates=n_static,
            row_keys=self.row_keys[row_mask],
            rows=tuple(compress(self.rows, row_mask.tolist())),
        )
        if len(self.plans) < _MEMO_PLAN_CACHE_LIMIT:
            self.plans[cache_key] = plan
        return plan


def _ordered_pair(a: CollisionPiece, b: CollisionPiece) -> tuple[CollisionPiece, CollisionPiece]:
    return (a, b) if a.resolved_piece_id <= b.resolved_piece_id else (b, a)


def _narrow_probe(
    left: CollisionPiece,
    right: CollisionPiece,
    margin: float,
    fcl,
    collision_request,
    distance_request,
) -> tuple[CollisionPiece, CollisionPiece, bool, float, tuple[float, ...], float] | None:
    result = fcl.CollisionResult()
    colliding = (
        int(fcl.collide(left.collision_object, right.collision_object, collision_request, result))
        > 0
    )
    distance = 0.0
    if not colliding:
        distance = float(
            fcl.distance(
                left.collision_object,
                right.collision_object,
                distance_request,
                fcl.DistanceResult(),
            )
        )
        if distance > margin:
            return None
    first, second = _ordered_pair(left, right)
    depth = _aabb_intersection_depth(first.world_aabb, second.world_aabb)
    return (
        first,
        second,
        colliding,
        distance,
        tuple(float(value) for value in depth),
        float(_aabb_intersection_volume(first.world_aabb, second.world_aabb)),
    )


def _narrow_row(
    left: CollisionPiece,
    right: CollisionPiece,
    pose_id: int,
    margin: float,
    fcl,
    collision_request,
    distance_request,
) -> ContactRow | None:
    probe = _narrow_probe(left, right, margin, fcl, collision_request, distance_request)
    if probe is None:
        return None
    first, second, colliding, distance, depth, volume = probe
    return ContactRow(
        pose_id=pose_id,
        piece_a=first.resolved_piece_id,
        piece_b=second.resolved_piece_id,
        source_element_a=first.source_element_id,
        source_element_b=second.source_element_id,
        part_a=first.part_name,
        part_b=second.part_name,
        colliding=colliding,
        min_distance=distance,
        overlap_depth=depth,
        overlap_volume=volume,
        aabb_a=first.world_aabb,
        aabb_b=second.world_aabb,
    )


def sweep_contacts(
    world: CollisionWorld,
    poses: PoseTable,
    *,
    broad_phase_margin: float,
    pose_ids: Iterable[int] | None = None,
    memo: RestMemo | None = None,
) -> ContactTable:
    import fcl
    import numpy as np

    rows: list[ContactRow] = []
    append = rows.append
    n_broad = 0
    n_narrow = 0
    n_memoized = 0
    collision_request = fcl.CollisionRequest()
    distance_request = fcl.DistanceRequest()
    selected = poses.rows if pose_ids is None else [poses.rows[index] for index in pose_ids]

    pieces = world.pieces
    n_pieces = len(pieces)
    part_codes = world.part_of_piece
    if memo is None:
        memo = RestMemo()
    elif memo.populated:
        memo.check(world, broad_phase_margin)

    for pose in selected:
        world.set_pose(pose.qpos)
        pose_id = pose.pose_id

        if not memo.populated:
            keys, left, right = _candidate_pair_keys(
                world.piece_lo, world.piece_hi, part_codes, broad_phase_margin, n_pieces
            )
            base_rows: list[ContactRow] = []
            base_keys: list[int] = []
            for key, left_index, right_index in zip(keys.tolist(), left.tolist(), right.tolist()):
                row = _narrow_row(
                    pieces[left_index],
                    pieces[right_index],
                    pose_id,
                    broad_phase_margin,
                    fcl,
                    collision_request,
                    distance_request,
                )
                if row is not None:
                    append(row)
                    base_rows.append(row)
                    base_keys.append(key)
            n_broad += int(keys.size)
            n_narrow += int(keys.size)
            row_keys = np.asarray(base_keys, dtype=np.int64)
            row_i = (row_keys // n_pieces) if n_pieces else row_keys
            memo.populate(
                world,
                broad_phase_margin,
                left,
                right,
                row_keys,
                row_i,
                row_keys - row_i * n_pieces,
                tuple(base_rows),
            )
            continue

        moved_parts = memo.moved_parts(world)
        plan = memo.plan(moved_parts, part_codes)
        static_rows = plan.rows
        n_static_rows = len(static_rows)
        moved_pieces = moved_parts[part_codes]
        moved_indices = np.nonzero(moved_pieces)[0]
        keys, left, right = _candidate_pair_keys(
            world.piece_lo,
            world.piece_hi,
            part_codes,
            broad_phase_margin,
            n_pieces,
            rows=moved_indices,
            moved=moved_pieces,
        )
        n_broad += int(keys.size) + plan.n_static_candidates
        n_narrow += int(keys.size)
        n_memoized += plan.n_static_candidates

        cursor = 0
        if keys.size:
            insert = np.searchsorted(plan.row_keys, keys)
            for position, left_index, right_index in zip(
                insert.tolist(), left.tolist(), right.tolist()
            ):
                while cursor < position:
                    base = static_rows[cursor]
                    append(
                        ContactRow(
                            pose_id,
                            base.piece_a,
                            base.piece_b,
                            base.source_element_a,
                            base.source_element_b,
                            base.part_a,
                            base.part_b,
                            base.colliding,
                            base.min_distance,
                            base.overlap_depth,
                            base.overlap_volume,
                            base.aabb_a,
                            base.aabb_b,
                        )
                    )
                    cursor += 1
                row = _narrow_row(
                    pieces[left_index],
                    pieces[right_index],
                    pose_id,
                    broad_phase_margin,
                    fcl,
                    collision_request,
                    distance_request,
                )
                if row is not None:
                    append(row)
        while cursor < n_static_rows:
            base = static_rows[cursor]
            append(
                ContactRow(
                    pose_id,
                    base.piece_a,
                    base.piece_b,
                    base.source_element_a,
                    base.source_element_b,
                    base.part_a,
                    base.part_b,
                    base.colliding,
                    base.min_distance,
                    base.overlap_depth,
                    base.overlap_volume,
                    base.aabb_a,
                    base.aabb_b,
                )
            )
            cursor += 1

    return ContactTable(
        rows=tuple(rows),
        stats=SweepStats(
            n_poses=len(selected),
            n_pairs_broad=n_broad,
            n_pairs_narrow=n_narrow,
            n_pairs_memoized=n_memoized,
        ),
    )


def merge_contact_tables(*tables: ContactTable) -> ContactTable:
    rows = tuple(row for table in tables for row in table.rows)
    return ContactTable(
        rows=rows,
        stats=SweepStats(
            n_poses=sum(table.stats.n_poses for table in tables),
            n_pairs_broad=sum(table.stats.n_pairs_broad for table in tables),
            n_pairs_narrow=sum(table.stats.n_pairs_narrow for table in tables),
            n_pairs_memoized=sum(table.stats.n_pairs_memoized for table in tables),
            n_shards=max((table.stats.n_shards for table in tables), default=1),
        ),
    )


DEFAULT_MOTION_SHARDS = 8
_SHARDS_ENV = "ARTISAN_MOTION_SHARDS"
MOTION_SHARD_PGID_ENV = "ARTISAN_MOTION_SHARD_PGID"
_MIN_POSES_PER_SHARD = 16
_SHARD_TERM_GRACE_S = 10.0
_SHARD_WAIT_FLOOR_S = 120.0
_SHARD_WAIT_PREPASS_FACTOR = 10.0


class _ShardFallback(RuntimeError):
    pass


def resolve_shard_count(requested: int | None = None) -> int:
    value = requested
    if value is None:
        raw = os.environ.get(_SHARDS_ENV)
        if raw:
            try:
                value = int(raw)
            except ValueError:
                value = None
    if value is None:
        value = DEFAULT_MOTION_SHARDS
    return max(1, min(int(value), os.cpu_count() or 1))


def motion_execution_config() -> dict[str, Any]:
    return {"shards": resolve_shard_count()}


def _shard_group_confirmed() -> bool:
    declared = os.environ.get(MOTION_SHARD_PGID_ENV)
    if not declared:
        return True
    try:
        return int(declared) == os.getpgrp()
    except (ValueError, OSError):
        return False


def _shard_scan(
    world: CollisionWorld,
    poses: PoseTable,
    indices: list[int],
    *,
    broad_phase_margin: float,
    memo: RestMemo,
) -> dict[str, Any]:
    import fcl
    import numpy as np

    world.invalidate_pose()
    pieces = world.pieces
    n_pieces = len(pieces)
    part_codes = world.part_of_piece
    collision_request = fcl.CollisionRequest()
    distance_request = fcl.DistanceRequest()

    rows_per_pose: list[int] = []
    row_key: list[int] = []
    row_colliding: list[bool] = []
    row_distance: list[float] = []
    row_depth: list[tuple[float, ...]] = []
    row_volume: list[float] = []
    n_broad = 0
    n_narrow = 0
    n_memoized = 0

    for index in indices:
        pose = poses.rows[index]
        world.set_pose(pose.qpos)
        moved_parts = memo.moved_parts(world)
        plan = memo.plan(moved_parts, part_codes)
        moved_pieces = moved_parts[part_codes]
        moved_indices = np.nonzero(moved_pieces)[0]
        keys, left, right = _candidate_pair_keys(
            world.piece_lo,
            world.piece_hi,
            part_codes,
            broad_phase_margin,
            n_pieces,
            rows=moved_indices,
            moved=moved_pieces,
        )
        n_broad += int(keys.size) + plan.n_static_candidates
        n_narrow += int(keys.size)
        n_memoized += plan.n_static_candidates
        emitted = 0
        for key, left_index, right_index in zip(keys.tolist(), left.tolist(), right.tolist()):
            probe = _narrow_probe(
                pieces[left_index],
                pieces[right_index],
                broad_phase_margin,
                fcl,
                collision_request,
                distance_request,
            )
            if probe is None:
                continue
            _first, _second, colliding, distance, depth, volume = probe
            row_key.append(key)
            row_colliding.append(colliding)
            row_distance.append(distance)
            row_depth.append(depth)
            row_volume.append(volume)
            emitted += 1
        rows_per_pose.append(emitted)

    return {
        "n_poses": len(indices),
        "n_pairs_broad": n_broad,
        "n_pairs_narrow": n_narrow,
        "n_pairs_memoized": n_memoized,
        "rows_per_pose": np.asarray(rows_per_pose, dtype=np.int64),
        "row_key": np.asarray(row_key, dtype=np.int64),
        "row_colliding": np.asarray(row_colliding, dtype=bool),
        "row_min_distance": np.asarray(row_distance, dtype=np.float64),
        "row_depth": np.asarray(row_depth, dtype=np.float64).reshape(len(row_key), 3),
        "row_volume": np.asarray(row_volume, dtype=np.float64),
    }


def _fork_shards(
    assignments: list[list[int]],
    run_shard: Callable[[int, list[int]], dict[str, Any]],
    ipc_root: Path,
    pids: dict[int, int],
) -> None:
    parent_pid = os.getpid()
    for shard, group in enumerate(assignments):
        try:
            pid = os.fork()
        except BaseException as error:
            if os.getpid() != parent_pid:
                os._exit(1)
            raise _ShardFallback(f"could not fork shard {shard}: {error!r}") from error
        if pid == 0:  # pragma: no cover - the child never returns to the parent's frames
            status = 1
            try:
                payload = run_shard(shard, group)
                temp = ipc_root / f"shard-{shard}.tmp"
                with temp.open("wb") as handle:
                    pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(temp, ipc_root / f"shard-{shard}.pickle")
                status = 0
            except BaseException:
                try:
                    (ipc_root / f"shard-{shard}.error").write_text(
                        traceback.format_exc(limit=20), encoding="utf-8"
                    )
                except OSError:
                    pass
            finally:
                os._exit(status)
        pids[shard] = pid


def _reap_shards(pids: dict[int, int], *, reaped: set[int]) -> list[int]:
    stragglers = [shard for shard in pids if shard not in reaped]
    if not stragglers:
        return []
    for shard in stragglers:
        try:
            os.kill(pids[shard], signal.SIGTERM)
        except OSError:
            pass
    deadline = time.monotonic() + _SHARD_TERM_GRACE_S
    for shard in stragglers:
        pid = pids[shard]
        while time.monotonic() < deadline:
            try:
                waited, _status = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                waited = pid
            if waited:
                reaped.add(shard)
                break
            time.sleep(0.02)
    for shard in stragglers:
        if shard in reaped:
            continue
        try:
            os.kill(pids[shard], signal.SIGKILL)
        except OSError:
            pass
        try:
            os.waitpid(pids[shard], 0)
        except ChildProcessError:
            pass
        reaped.add(shard)
    return stragglers


def sweep_contacts_sharded(
    world: CollisionWorld,
    poses: PoseTable,
    *,
    broad_phase_margin: float,
    pose_ids: Iterable[int] | None = None,
    memo: RestMemo | None = None,
    shards: int | None = None,
) -> ContactTable:
    import numpy as np

    indices = (
        list(range(len(poses.rows))) if pose_ids is None else [int(value) for value in pose_ids]
    )
    shard_count = resolve_shard_count(shards)
    if (
        shard_count <= 1
        or not hasattr(os, "fork")
        or len(indices) < _MIN_POSES_PER_SHARD * shard_count
        or not _shard_group_confirmed()
    ):
        return sweep_contacts(
            world, poses, broad_phase_margin=broad_phase_margin, pose_ids=indices, memo=memo
        )

    if memo is None:
        memo = RestMemo()
    elif memo.populated:
        memo.check(world, broad_phase_margin)

    head = indices[:1] if not memo.populated else []
    tail = indices[len(head) :]
    head_table = (
        sweep_contacts(
            world, poses, broad_phase_margin=broad_phase_margin, pose_ids=head, memo=memo
        )
        if head
        else None
    )

    pieces = world.pieces
    n_pieces = len(pieces)
    part_codes = world.part_of_piece
    base_aabb = memo.base_aabbs

    assignments = [tail[shard::shard_count] for shard in range(shard_count)]
    assignments = [group for group in assignments if group]

    def run_shard(_shard: int, group: list[int]) -> dict[str, Any]:
        return _shard_scan(world, poses, group, broad_phase_margin=broad_phase_margin, memo=memo)

    ipc_root = Path(tempfile.mkdtemp(prefix="artisan-motion-shards-"))
    pids: dict[int, int] = {}
    reaped: set[int] = set()
    prepass_started = time.monotonic()
    try:
        _fork_shards(assignments, run_shard, ipc_root, pids)

        static_rows: list[tuple[ContactRow, ...]] = []
        static_keys: list[Any] = []
        moved_aabb: list[dict[int, AABB]] = []
        for index in tail:
            pose = poses.rows[index]
            world.set_pose(pose.qpos)
            pose_id = pose.pose_id
            moved_parts = memo.moved_parts(world)
            plan = memo.plan(moved_parts, part_codes)
            static_rows.append(
                tuple(
                    ContactRow(
                        pose_id,
                        base.piece_a,
                        base.piece_b,
                        base.source_element_a,
                        base.source_element_b,
                        base.part_a,
                        base.part_b,
                        base.colliding,
                        base.min_distance,
                        base.overlap_depth,
                        base.overlap_volume,
                        base.aabb_a,
                        base.aabb_b,
                    )
                    for base in plan.rows
                )
            )
            static_keys.append(plan.row_keys)
            moved_aabb.append(
                {
                    int(piece_index): pieces[piece_index].world_aabb
                    for piece_index in np.nonzero(moved_parts[part_codes])[0].tolist()
                }
            )

        payloads = _collect_shard_payloads(
            assignments,
            ipc_root,
            pids,
            reaped,
            timeout_s=max(
                _SHARD_WAIT_FLOOR_S,
                _SHARD_WAIT_PREPASS_FACTOR * (time.monotonic() - prepass_started),
            ),
        )
    except _ShardFallback:
        payloads = None
    finally:
        _reap_shards(pids, reaped=reaped)
        shutil.rmtree(ipc_root, ignore_errors=True)

    if payloads is None:
        contacts = sweep_contacts(
            world, poses, broad_phase_margin=broad_phase_margin, pose_ids=tail, memo=memo
        )
        return contacts if head_table is None else merge_contact_tables(head_table, contacts)

    rows: list[ContactRow] = list(head_table.rows) if head_table is not None else []
    append = rows.append
    n_broad = head_table.stats.n_pairs_broad if head_table is not None else 0
    n_narrow = head_table.stats.n_pairs_narrow if head_table is not None else 0
    n_memoized = head_table.stats.n_pairs_memoized if head_table is not None else 0
    decoded = []
    for payload in payloads:
        n_broad += int(payload["n_pairs_broad"])
        n_narrow += int(payload["n_pairs_narrow"])
        n_memoized += int(payload["n_pairs_memoized"])
        keys = payload["row_key"]
        left = keys // n_pieces if n_pieces else keys
        decoded.append(
            {
                "cursor": 0,
                "pose": 0,
                "counts": payload["rows_per_pose"].tolist(),
                "keys_array": keys,
                "keys": keys.tolist(),
                "left": left.tolist(),
                "right": (keys - left * n_pieces).tolist(),
                "colliding": payload["row_colliding"].tolist(),
                "distance": payload["row_min_distance"].tolist(),
                "depth": payload["row_depth"].tolist(),
                "volume": payload["row_volume"].tolist(),
            }
        )
    pair_cache: dict[int, tuple[str, str, str, str, str, str, int, int]] = {}
    for position, index in enumerate(tail):
        shard = decoded[position % len(decoded)]
        offset = shard["pose"]
        shard["pose"] = offset + 1
        count = shard["counts"][offset]
        start = shard["cursor"]
        shard["cursor"] = stop = start + count
        pose_id = poses.rows[index].pose_id
        memo_rows = static_rows[position]
        n_memo_rows = len(memo_rows)
        cursor = 0
        if count:
            aabb_at = moved_aabb[position]
            insert = np.searchsorted(static_keys[position], shard["keys_array"][start:stop])
            keys = shard["keys"]
            lefts = shard["left"]
            rights = shard["right"]
            colliding = shard["colliding"]
            distance = shard["distance"]
            depth = shard["depth"]
            volume = shard["volume"]
            for step, at in enumerate(insert.tolist()):
                while cursor < at:
                    append(memo_rows[cursor])
                    cursor += 1
                row_index = start + step
                key = keys[row_index]
                identity = pair_cache.get(key)
                if identity is None:
                    first, second = _ordered_pair(
                        pieces[lefts[row_index]], pieces[rights[row_index]]
                    )
                    identity = (
                        first.resolved_piece_id,
                        second.resolved_piece_id,
                        first.source_element_id,
                        second.source_element_id,
                        first.part_name,
                        second.part_name,
                        lefts[row_index],
                        rights[row_index],
                    )
                    pair_cache[key] = identity
                append(
                    ContactRow(
                        pose_id,
                        identity[0],
                        identity[1],
                        identity[2],
                        identity[3],
                        identity[4],
                        identity[5],
                        colliding[row_index],
                        distance[row_index],
                        tuple(depth[row_index]),
                        volume[row_index],
                        aabb_at.get(identity[6], base_aabb[identity[6]]),
                        aabb_at.get(identity[7], base_aabb[identity[7]]),
                    )
                )
        while cursor < n_memo_rows:
            append(memo_rows[cursor])
            cursor += 1

    return ContactTable(
        rows=tuple(rows),
        stats=SweepStats(
            n_poses=len(indices),
            n_pairs_broad=n_broad,
            n_pairs_narrow=n_narrow,
            n_pairs_memoized=n_memoized,
            n_shards=len(assignments),
        ),
    )


def _wait_for_shard(pid: int, deadline: float) -> int:
    while True:
        waited, status = os.waitpid(pid, os.WNOHANG)
        if waited:
            return status
        if time.monotonic() >= deadline:
            raise _ShardFallback(f"shard pid {pid} did not finish before its deadline")
        time.sleep(0.01)


def _collect_shard_payloads(
    assignments: list[list[int]],
    ipc_root: Path,
    pids: dict[int, int],
    reaped: set[int],
    *,
    timeout_s: float,
    payload_failures: Callable[[int, dict[str, Any]], list[str]] | None = None,
) -> list[dict[str, Any]]:
    failures: list[str] = []
    deadline = time.monotonic() + timeout_s
    for shard, pid in pids.items():
        status = _wait_for_shard(pid, deadline)
        reaped.add(shard)
        if os.WIFSIGNALED(status):
            failures.append(f"shard {shard} died on signal {os.WTERMSIG(status)}")
        elif not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
            detail = ""
            error_path = ipc_root / f"shard-{shard}.error"
            if error_path.is_file():
                tail_line = error_path.read_text(encoding="utf-8").strip().splitlines()
                detail = f": {tail_line[-1]}" if tail_line else ""
            code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else status
            failures.append(f"shard {shard} exited {code}{detail}")
    payloads: list[dict[str, Any]] = []
    for shard, group in enumerate(assignments):
        path = ipc_root / f"shard-{shard}.pickle"
        if not path.is_file():
            failures.append(f"shard {shard} produced no result")
            continue
        payload = pickle.loads(path.read_bytes())
        if int(payload.get("n_poses", -1)) != len(group):
            failures.append(
                f"shard {shard} covered {payload.get('n_poses')} poses, expected {len(group)}"
            )
        failures.extend((payload_failures or _sweep_payload_failures)(shard, payload))
        payloads.append(payload)
    if failures:
        raise RuntimeError("motion sweep shard failure: " + "; ".join(failures))
    return payloads


def _sweep_payload_failures(shard: int, payload: dict[str, Any]) -> list[str]:
    if int(payload["rows_per_pose"].sum()) != int(payload["row_key"].size):
        return [f"shard {shard} row blocks do not add up to its row count"]
    return []


def _mapped_payload_failures(shard: int, payload: dict[str, Any]) -> list[str]:
    if len(payload["results"]) != int(payload["n_poses"]):
        return [f"shard {shard} returned {len(payload['results'])} results for its poses"]
    return []


def map_pose_shards(
    positions: list[int],
    scan: Callable[[list[int]], list[Any]],
    *,
    shards: int | None = None,
) -> tuple[list[Any], int]:
    shard_count = resolve_shard_count(shards)
    if (
        shard_count <= 1
        or not hasattr(os, "fork")
        or len(positions) < _MIN_POSES_PER_SHARD * shard_count
        or not _shard_group_confirmed()
    ):
        return list(scan(positions)), 1

    groups = [positions[shard::shard_count] for shard in range(shard_count)]
    groups = [group for group in groups if group]
    own, children = groups[0], groups[1:]

    def run_shard(_shard: int, group: list[int]) -> dict[str, Any]:
        return {"n_poses": len(group), "results": scan(group)}

    ipc_root = Path(tempfile.mkdtemp(prefix="artisan-motion-shards-"))
    pids: dict[int, int] = {}
    reaped: set[int] = set()
    try:
        _fork_shards(children, run_shard, ipc_root, pids)
        started = time.monotonic()
        own_results = list(scan(own))
        payloads: list[dict[str, Any]] | None = _collect_shard_payloads(
            children,
            ipc_root,
            pids,
            reaped,
            timeout_s=max(
                _SHARD_WAIT_FLOOR_S,
                _SHARD_WAIT_PREPASS_FACTOR * (time.monotonic() - started),
            ),
            payload_failures=_mapped_payload_failures,
        )
    except _ShardFallback:
        payloads = None
    finally:
        _reap_shards(pids, reaped=reaped)
        shutil.rmtree(ipc_root, ignore_errors=True)

    if payloads is None:
        return list(scan(positions)), 1
    by_position: dict[int, Any] = dict(zip(own, own_results, strict=True))
    for group, payload in zip(children, payloads, strict=True):
        by_position.update(zip(group, payload["results"], strict=True))
    return [by_position[position] for position in positions], len(groups)
