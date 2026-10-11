
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from artisan.physics_policy import validate_penetration_depth_tolerance

from ._contact_exemptions import (
    _REST_CONTACT_TOL,
    contact_exempt_pairs,
    contact_exemptions_detailed,
    coupling_contact_pairs,
)
from .exact_collisions import compile_object_model_with_exact_collisions
from .geometry_qc import (
    _collision_depth_evidence,
    _compiled_part_collision_entries,
    _find_collision_overlaps_fcl_for_poses,
    _identity4,
    compute_part_world_transforms,
    default_penetration_depth_tol_from_env,
)
from .self_collision import (
    _elem_key,
    _gate_value_grid,
    _moving_joints,
    _screen_range,
    compile_for_motion_screen,
    motion_screen_collision_mode,
    moving_subtree,
)


@dataclass
class DriverReachFinding:

    joint: str
    nominal: tuple[float, float]
    reachable: tuple[float, float]
    reachable_fraction: float
    blocked_by: Optional[tuple[str, str]]
    blocked_at: Optional[float]
    elements: Optional[tuple[str, str]]
    blocked_sides: list[dict] = field(default_factory=list)
    note: str = ""


def _outward_samples(
    lo: float,
    hi: float,
    *,
    per_side_min: int = 12,
    total: int = 25,
    max_step: float | None = None,
) -> tuple[list[float], list[float]]:
    rest = min(max(0.0, lo), hi)
    lower_span = rest - lo
    upper_span = hi - rest
    full = lower_span + upper_span
    if full <= 0:
        return [], []
    n_lower = per_side_min if lower_span > 0 else 0
    n_upper = per_side_min if upper_span > 0 else 0
    remaining = max(0, total - n_lower - n_upper)
    if remaining and full > 0:
        if lower_span > 0:
            n_lower += int(round(remaining * lower_span / full))
        if upper_span > 0:
            n_upper += int(round(remaining * upper_span / full))
    if max_step is not None and max_step > 0.0:
        if lower_span > 0:
            n_lower = max(n_lower, int(math.ceil(lower_span / max_step)))
        if upper_span > 0:
            n_upper = max(n_upper, int(math.ceil(upper_span / max_step)))
    lower = [rest - lower_span * k / n_lower for k in range(1, n_lower + 1)] if n_lower else []
    upper = [rest + upper_span * k / n_upper for k in range(1, n_upper + 1)] if n_upper else []
    return lower, upper


@dataclass
class FullCollisionPenetrationFinding:

    joint: str
    link_a: str
    link_b: str
    penetration_depth: float | None
    penetration_volume: float
    worst_pose: dict
    rest_depth: float | None
    constant: bool
    suggested_axis: str
    offending_elements: tuple[str, str]
    declared: bool
    blocking: bool
    depth_method: str = "unknown"
    depth_signed: bool = False
    depth_degenerate: bool = False


def _pair_fcl_penetration(compiled, parts, asset_root, mesh_cache, a, b, pose):
    if a not in parts or b not in parts:
        return _collision_depth_evidence(
            [], [], collision_mode=motion_screen_collision_mode(compiled)
        )
    tf = compute_part_world_transforms(compiled, pose)
    entries_a = _compiled_part_collision_entries(
        parts[a], asset_root=asset_root, mesh_cache=mesh_cache, part_tf=tf.get(a, _identity4())
    )
    entries_b = _compiled_part_collision_entries(
        parts[b], asset_root=asset_root, mesh_cache=mesh_cache, part_tf=tf.get(b, _identity4())
    )
    return _collision_depth_evidence(
        entries_a,
        entries_b,
        collision_mode=motion_screen_collision_mode(compiled),
    )


def _build_outward_pose_units(
    swept: list,
    ranges: dict[str, tuple[float, float]],
    gate_combos_of: dict[str, list[dict[str, float]]],
    samples_per_joint: int,
) -> tuple[
    list[dict[str, float]], list[tuple[str, float, str]], dict[str, dict[str, list[list[int]]]]
]:
    poses: list[dict[str, float]] = [{}]
    meta: list[tuple[str, float, str]] = [("", 0.0, "rest")]
    sides_of: dict[str, dict[str, list[list[int]]]] = {}
    for j in swept:
        lo, hi = ranges[j.name]
        lower, upper = _outward_samples(lo, hi, total=samples_per_joint)
        combos = gate_combos_of.get(j.name)
        sides_of[j.name] = {"lower": [], "upper": []}
        for side, qs in (("lower", lower), ("upper", upper)):
            for q in qs:
                unit: list[int] = []
                if combos:
                    for combo in combos:
                        unit.append(len(poses))
                        poses.append({j.name: q, **combo})
                        meta.append((j.name, q, side))
                else:
                    unit.append(len(poses))
                    poses.append({j.name: q})
                    meta.append((j.name, q, side))
                sides_of[j.name][side].append(unit)
    return poses, meta, sides_of


def _serial_prismatic_pose_grid(
    swept: list,
    ranges: dict[str, tuple[float, float]],
    gate_combos_of: dict[str, list[dict[str, float]]],
) -> list[dict[str, float]]:

    def _kind(joint: object) -> str:
        return str(getattr(getattr(joint, "articulation_type", None), "value", "")).lower()

    def _three_values(name: str) -> tuple[float, ...]:
        lo, hi = ranges[name]
        candidates = (float(lo), float((lo + hi) * 0.5), float(hi))
        return tuple(
            value
            for index, value in enumerate(candidates)
            if not any(abs(value - earlier) <= 1e-12 for earlier in candidates[:index])
        )

    gate_participants = set(gate_combos_of)
    for combos in gate_combos_of.values():
        for combo in combos:
            gate_participants.update(combo)

    poses: list[dict[str, float]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for index, first in enumerate(swept):
        if _kind(first) != "prismatic" or first.name in gate_participants:
            continue
        for second in swept[index + 1 :]:
            if _kind(second) != "prismatic" or second.name in gate_participants:
                continue
            adjacent = first.child == second.parent or second.child == first.parent
            if not adjacent:
                continue
            for first_q in _three_values(first.name):
                for second_q in _three_values(second.name):
                    pose = {first.name: first_q, second.name: second_q}
                    key = tuple(sorted(pose.items()))
                    if key not in seen:
                        seen.add(key)
                        poses.append(pose)
    return poses


def _combined_pose_attribution_joint(
    *,
    pair: tuple[str, str],
    pose: dict[str, float],
    swept: list[Any],
    movers: dict[str, set[str]],
) -> Optional[Any]:
    a, b = pair
    crossing = [
        joint
        for joint in swept
        if joint.name in pose and (a in movers[joint.name]) != (b in movers[joint.name])
    ]
    return min(crossing, key=lambda candidate: len(movers[candidate.name])) if crossing else None


def _penetration_findings(
    compiled,
    *,
    asset_root,
    overlaps,
    swept,
    movers,
    sides_of,
    combined_pose_indices,
    gear_pairs,
    declared_pairs,
    depth_tol,
) -> list[FullCollisionPenetrationFinding]:

    def _pk(a, b):
        return (a, b) if a <= b else (b, a)

    raw: dict[tuple[int, tuple[str, str]], object] = {}
    for ov in overlaps:
        pair = _pk(ov.link_a, ov.link_b)
        if pair in gear_pairs:
            continue
        key = (ov.pose_index, pair)
        cur = raw.get(key)
        if cur is None or ov.overlap_volume > cur.overlap_volume:
            raw[key] = ov
    if not raw:
        return []

    pairs_at_pose: dict[int, set[tuple[str, str]]] = {}
    for pose_index, pair in raw:
        pairs_at_pose.setdefault(pose_index, set()).add(pair)

    reps: list[tuple[str, tuple[str, str], object]] = []
    for j in swept:
        jname = j.name
        sub = movers[jname]
        units = sides_of[jname]["lower"] + sides_of[jname]["upper"]
        cand_pairs: set[tuple[str, str]] = set()
        for unit in units:
            for idx in unit:
                cand_pairs |= pairs_at_pose.get(idx, set())
        for pair in cand_pairs:
            a, b = pair
            if (a in sub) == (b in sub):
                continue
            best = None
            for unit in units:
                if unit and all((idx, pair) in raw for idx in unit):
                    rep_idx = min(unit, key=lambda i: raw[(i, pair)].overlap_volume)
                    rep = raw[(rep_idx, pair)]
                    if best is None or rep.overlap_volume > best.overlap_volume:
                        best = rep
            if best is not None:
                reps.append((jname, pair, best))
    for pose_index in combined_pose_indices:
        for pair in pairs_at_pose.get(pose_index, set()):
            overlap = raw[(pose_index, pair)]
            joint = _combined_pose_attribution_joint(
                pair=pair,
                pose=overlap.pose,
                swept=swept,
                movers=movers,
            )
            if joint is None:
                continue
            reps.append((joint.name, pair, overlap))
    if not reps:
        return []

    parts = {getattr(p, "name", None): p for p in getattr(compiled, "parts", None) or []}
    mesh_cache: dict = {}
    depth_cache: dict[tuple[int, tuple[str, str]], object] = {}

    def _depth(pose_index: int, pose: dict, pair: tuple[str, str]):
        key = (pose_index, pair)
        if key not in depth_cache:
            depth_cache[key] = _pair_fcl_penetration(
                compiled, parts, asset_root, mesh_cache, pair[0], pair[1], pose
            )
        return depth_cache[key]

    raw_findings: list[FullCollisionPenetrationFinding] = []
    for jname, pair, rep in reps:
        measured = _depth(rep.pose_index, rep.pose, pair)
        depth = measured.depth_m
        if not measured.colliding:
            continue
        if depth is not None and depth < depth_tol:
            continue
        rest_measured = _depth(0, {}, pair)
        rest_depth = rest_measured.depth_m if rest_measured.colliding else 0.0
        declared = pair in declared_pairs
        resolved_depth = (
            max(depth, rest_depth if rest_depth is not None else 0.0) if depth is not None else None
        )
        raw_findings.append(
            FullCollisionPenetrationFinding(
                joint=jname,
                link_a=pair[0],
                link_b=pair[1],
                penetration_depth=resolved_depth,
                penetration_volume=float(getattr(rep, "overlap_volume", 0.0)),
                worst_pose=dict(rep.pose),
                rest_depth=rest_depth,
                constant=bool(
                    depth is not None
                    and rest_depth is not None
                    and depth > 0
                    and rest_depth >= 0.8 * depth
                ),
                suggested_axis=measured.axis,
                offending_elements=(
                    (
                        f"{pair[0]}:{measured.elements[0]}",
                        f"{pair[1]}:{measured.elements[1]}",
                    )
                    if measured.elements
                    else _elem_key(rep)
                ),
                declared=declared,
                blocking=not declared,
                depth_method=measured.method,
                depth_signed=measured.signed,
                depth_degenerate=measured.degenerate,
            )
        )

    subtree_size = {j.name: len(movers[j.name]) for j in swept}

    def _rank(f: FullCollisionPenetrationFinding) -> tuple[float, int]:
        return (
            (round(f.penetration_depth, 9) if f.penetration_depth is not None else float("inf")),
            -subtree_size.get(f.joint, 1 << 30),
        )

    by_pair: dict[tuple[str, str], FullCollisionPenetrationFinding] = {}
    for f in raw_findings:
        key = (f.link_a, f.link_b)
        cur = by_pair.get(key)
        if cur is None or _rank(f) > _rank(cur):
            by_pair[key] = f
    return sorted(
        by_pair.values(),
        key=lambda f: (
            f.penetration_depth is not None,
            -(f.penetration_depth or 0.0),
            f.link_a,
            f.link_b,
        ),
    )


@dataclass
class DriverReachSweepEvidence:

    driver_reach_findings: list[DriverReachFinding]
    full_collision_penetrations: list[FullCollisionPenetrationFinding]
    sampled_poses: list[dict[str, float]]
    declared_gate_witnesses: list[dict[str, object]] = field(default_factory=list)


def find_driver_reach_sweep_evidence(
    model: object,
    *,
    asset_root: Path | None = None,
    declared_overlap_pairs: Optional[set[tuple[str, str]]] = None,
    gate_map: Optional[dict[str, list[str]]] = None,
    samples_per_joint: int = 25,
    min_reachable_fraction: float = 0.9,
    penetration_depth_tol: Optional[float] = None,
    penetration_volume_floor: float = 1e-9,
    validate_model: bool = False,
) -> DriverReachSweepEvidence:
    return _find_driver_reach_and_penetration(
        model,
        asset_root=asset_root,
        declared_overlap_pairs=declared_overlap_pairs,
        gate_map=gate_map,
        samples_per_joint=samples_per_joint,
        min_reachable_fraction=min_reachable_fraction,
        penetration_depth_tol=(
            default_penetration_depth_tol_from_env()
            if penetration_depth_tol is None
            else validate_penetration_depth_tolerance(
                penetration_depth_tol, name="penetration_depth_tol"
            )
        ),
        penetration_volume_floor=penetration_volume_floor,
        validate_model=validate_model,
    )


def find_driver_reach_and_penetration_findings(
    model: object,
    *,
    asset_root: Path | None = None,
    declared_overlap_pairs: Optional[set[tuple[str, str]]] = None,
    gate_map: Optional[dict[str, list[str]]] = None,
    samples_per_joint: int = 25,
    min_reachable_fraction: float = 0.9,
    penetration_depth_tol: Optional[float] = None,
    penetration_volume_floor: float = 1e-9,
    validate_model: bool = False,
) -> tuple[list[DriverReachFinding], list[FullCollisionPenetrationFinding]]:
    evidence = find_driver_reach_sweep_evidence(
        model,
        asset_root=asset_root,
        declared_overlap_pairs=declared_overlap_pairs,
        gate_map=gate_map,
        samples_per_joint=samples_per_joint,
        min_reachable_fraction=min_reachable_fraction,
        penetration_depth_tol=penetration_depth_tol,
        penetration_volume_floor=penetration_volume_floor,
        validate_model=validate_model,
    )
    return evidence.driver_reach_findings, evidence.full_collision_penetrations


def find_driver_reach_findings(
    model: object,
    *,
    asset_root: Path | None = None,
    declared_overlap_pairs: Optional[set[tuple[str, str]]] = None,
    gate_map: Optional[dict[str, list[str]]] = None,
    samples_per_joint: int = 25,
    min_reachable_fraction: float = 0.9,
    validate_model: bool = False,
) -> list[DriverReachFinding]:
    evidence = find_driver_reach_sweep_evidence(
        model,
        asset_root=asset_root,
        declared_overlap_pairs=declared_overlap_pairs,
        gate_map=gate_map,
        samples_per_joint=samples_per_joint,
        min_reachable_fraction=min_reachable_fraction,
        validate_model=validate_model,
    )
    return evidence.driver_reach_findings


def _find_driver_reach_and_penetration(
    model: object,
    *,
    asset_root: Path | None = None,
    declared_overlap_pairs: Optional[set[tuple[str, str]]] = None,
    gate_map: Optional[dict[str, list[str]]] = None,
    samples_per_joint: int = 25,
    min_reachable_fraction: float = 0.9,
    penetration_depth_tol: float = 2e-4,
    penetration_volume_floor: float = 1e-9,
    validate_model: bool = False,
) -> DriverReachSweepEvidence:
    joints = _moving_joints(model)
    if not joints:
        return DriverReachSweepEvidence([], [], [{}])

    ranges: dict[str, tuple[float, float]] = {}
    for j in joints:
        lo, hi = _screen_range(model, j)
        if hi > lo:
            ranges[j.name] = (float(lo), float(hi))
    swept = [j for j in joints if j.name in ranges]
    if not swept:
        return DriverReachSweepEvidence([], [], [{}])

    compiled = compile_for_motion_screen(model, asset_root=asset_root, validate=validate_model)

    exempt = set(
        contact_exempt_pairs(
            compiled, asset_root=asset_root, declared_overlap_pairs=declared_overlap_pairs
        )
    )

    movers = {j.name: moving_subtree(model, j) for j in swept}
    all_movers: set[str] = set()
    for s in movers.values():
        all_movers |= s

    gear_pairs = {
        (a, b) if a <= b else (b, a)
        for (a, b) in coupling_contact_pairs(compiled, asset_root=asset_root)
    }
    part_names = [getattr(p, "name", None) for p in getattr(compiled, "parts", [])]
    static = sorted(n for n in part_names if isinstance(n, str) and n not in all_movers)
    allowed: set[tuple[str, str]] = set(gear_pairs)
    for i in range(len(static)):
        for k in range(i + 1, len(static)):
            a, b = static[i], static[k]
            allowed.add((a, b) if a <= b else (b, a))

    gate_combos_of: dict[str, list[dict[str, float]]] = {}
    gate_links: set[str] = set()
    if gate_map:
        swept_by_name = {j.name: j for j in swept}
        for gated_name, gate_names in gate_map.items():
            if gated_name not in swept_by_name:
                continue
            gate_objs = [swept_by_name[g] for g in (gate_names or []) if g in swept_by_name]
            if gate_objs:
                gate_combos_of[gated_name] = _gate_value_grid(model, gate_objs)
                for g in gate_objs:
                    gate_links |= moving_subtree(model, g)

    poses, meta, sides_of = _build_outward_pose_units(
        swept, ranges, gate_combos_of, samples_per_joint
    )
    witness_rest_indices: dict[str, list[int]] = {}
    for joint_name, combos in gate_combos_of.items():
        lo, hi = ranges[joint_name]
        rest = min(max(0.0, lo), hi)
        witness_rest_indices[joint_name] = []
        for combo in combos:
            witness_rest_indices[joint_name].append(len(poses))
            poses.append({joint_name: rest, **combo})
            meta.append((joint_name, rest, "witness-rest"))
    combined_pose_indices: list[int] = []
    for pose in _serial_prismatic_pose_grid(swept, ranges, gate_combos_of):
        combined_pose_indices.append(len(poses))
        poses.append(pose)
        meta.append(("__serial_prismatic_pair__", 0.0, "combined"))
    if len(poses) <= 1:
        return DriverReachSweepEvidence([], [], poses)

    overlaps = _find_collision_overlaps_fcl_for_poses(
        compiled,
        asset_root=asset_root,
        source_model=model,
        poses=poses,
        overlap_tol=0.0,
        overlap_volume_tol=0.0,
        allowed_pairs=allowed,
        contact_distance_tol=None,
    )

    blocks_at: dict[int, list] = {}
    for ov in overlaps:
        jname, _q, _side = meta[ov.pose_index]
        if jname not in movers:
            continue
        sub = movers[jname]
        pair = (ov.link_a, ov.link_b) if ov.link_a <= ov.link_b else (ov.link_b, ov.link_a)
        if pair in exempt:
            continue
        if ov.link_a in sub or ov.link_b in sub:
            blocks_at.setdefault(ov.pose_index, []).append(ov)

    findings: list[DriverReachFinding] = []
    for j in swept:
        lo, hi = ranges[j.name]
        rest = min(max(0.0, lo), hi)
        sides = sides_of[j.name]

        def _reach(units: list[list[int]], bound: float) -> tuple[float, Optional[int]]:

            def _genuine_vol(i: int) -> float:
                return max(
                    (
                        o.overlap_volume
                        for o in blocks_at[i]
                        if o.link_a not in gate_links and o.link_b not in gate_links
                    ),
                    default=0.0,
                )

            def _any_vol(i: int) -> float:
                return max((o.overlap_volume for o in blocks_at[i]), default=0.0)

            prev = rest
            for unit in units:
                if unit and all(idx in blocks_at for idx in unit):
                    genuine = [i for i in unit if _genuine_vol(i) > 0.0]
                    blk_idx = max(genuine, key=_genuine_vol) if genuine else max(unit, key=_any_vol)
                    return prev, blk_idx
                prev = meta[unit[0]][1]
            return bound, None

        reach_lo, blk_lo = _reach(sides["lower"], lo)
        reach_hi, blk_hi = _reach(sides["upper"], hi)
        span = hi - lo
        frac = (reach_hi - reach_lo) / span if span > 0 else 1.0
        if frac >= min_reachable_fraction:
            continue

        blocked_sides: list[dict] = []
        for side, blk, reached in (("lower", blk_lo, reach_lo), ("upper", blk_hi, reach_hi)):
            if blk is None:
                continue
            blocking = blocks_at[blk]
            genuine = [
                o for o in blocking if o.link_a not in gate_links and o.link_b not in gate_links
            ]
            ov = max(genuine or blocking, key=lambda o: o.overlap_volume)
            pair = tuple(sorted((ov.link_a, ov.link_b)))
            blocked_sides.append(
                {
                    "side": side,
                    "at": round(meta[blk][1], 5),
                    "pair": list(pair),
                    "elements": list(_elem_key(ov)),
                }
            )
        dom = max(
            blocked_sides,
            key=lambda s: abs(s["at"] - rest),
            default=None,
        )
        findings.append(
            DriverReachFinding(
                joint=j.name,
                nominal=(lo, hi),
                reachable=(reach_lo, reach_hi),
                reachable_fraction=frac,
                blocked_by=tuple(dom["pair"]) if dom else None,
                blocked_at=dom["at"] if dom else None,
                elements=tuple(dom["elements"]) if dom else None,
                blocked_sides=blocked_sides,
                note=(
                    "exact-collision single-joint sweep; thin spears may pass "
                    "between samples (screening, not proof)"
                ),
            )
        )

    findings.sort(key=lambda f: f.reachable_fraction)

    declared_gate_witnesses: list[dict[str, object]] = []
    selected_combo_index: dict[str, int] = {}
    for joint_name, combos in sorted(gate_combos_of.items()):
        units = sides_of[joint_name]["lower"] + sides_of[joint_name]["upper"]
        candidates: list[tuple[int, list[int], list]] = []
        for combo_index, _combo in enumerate(combos):
            indices = [witness_rest_indices[joint_name][combo_index]]
            indices.extend(unit[combo_index] for unit in units)
            blocked = [index for index in indices if index in blocks_at]
            candidates.append((combo_index, indices, blocked))
        clear_candidates = [item for item in candidates if not item[2]]
        ranked = clear_candidates or sorted(candidates, key=lambda item: len(item[2]))
        combo_index, indices, blocked = max(
            ranked,
            key=lambda item: (
                -len(item[2]),
                sum(abs(value) for value in combos[item[0]].values()),
            ),
        )
        if not blocked:
            selected_combo_index[joint_name] = combo_index
        blocking_pairs = sorted(
            {
                tuple(sorted((overlap.link_a, overlap.link_b)))
                for index in blocked
                for overlap in blocks_at.get(index, [])
            }
        )
        declared_gate_witnesses.append(
            {
                "joint": joint_name,
                "gate_joints": sorted(combos[combo_index]),
                "gate_pose": {
                    name: float(value) for name, value in sorted(combos[combo_index].items())
                },
                "target_range": [float(ranges[joint_name][0]), float(ranges[joint_name][1])],
                "sample_count": len(indices),
                "clear": not blocked,
                "blocking_pairs": [list(pair) for pair in blocking_pairs],
                "blocked_pose": dict(poses[blocked[0]]) if blocked else None,
            }
        )

    accepted_poses: list[dict[str, float]] = [{}]
    for joint in swept:
        units = sides_of[joint.name]["lower"] + sides_of[joint.name]["upper"]
        fixed_combo_index = selected_combo_index.get(joint.name)
        for unit in units:
            if fixed_combo_index is not None:
                accepted_poses.append(dict(poses[unit[fixed_combo_index]]))
                continue
            clear = [pose_index for pose_index in unit if pose_index not in blocks_at]
            candidates = clear or unit
            if not candidates:
                continue
            selected = max(
                candidates,
                key=lambda pose_index: sum(
                    abs(value) for name, value in poses[pose_index].items() if name != joint.name
                ),
            )
            accepted_poses.append(dict(poses[selected]))
    accepted_poses.extend(dict(poses[index]) for index in combined_pose_indices)

    penetrations = _penetration_findings(
        compiled,
        asset_root=asset_root,
        overlaps=overlaps,
        swept=swept,
        movers=movers,
        sides_of=sides_of,
        combined_pose_indices=combined_pose_indices,
        gear_pairs=gear_pairs,
        declared_pairs={
            (a, b) if a <= b else (b, a) for (a, b) in (declared_overlap_pairs or set())
        },
        depth_tol=penetration_depth_tol,
    )
    return DriverReachSweepEvidence(
        findings,
        penetrations,
        accepted_poses,
        declared_gate_witnesses,
    )


@dataclass
class ContactExclusionEvidence:

    part_a: str
    part_b: str
    rest_distance: float
    motion_min_distance: float
    pose: dict


def find_contact_exclusion_evidence(
    model: object,
    *,
    asset_root: Path | None = None,
    declared_overlap_pairs: Optional[set[tuple[str, str]]] = None,
    max_pose_samples: int = 64,
    seed: int = 0,
    validate_model: bool = False,
) -> list["ContactExclusionEvidence"]:
    from .geometry_qc import generate_pose_samples, pair_min_distances_for_poses

    compiled = compile_object_model_with_exact_collisions(
        model, asset_root=asset_root, validate=validate_model
    )
    detail = contact_exemptions_detailed(
        compiled, asset_root=asset_root, declared_overlap_pairs=declared_overlap_pairs
    )
    source4_only = detail.source4_only
    if not source4_only:
        return []

    rest_pose = {
        name: 0.0
        for j in getattr(compiled, "articulations", []) or []
        if isinstance((name := getattr(j, "name", None)), str)
    }
    swept_poses = generate_pose_samples(compiled, max_samples=max_pose_samples, seed=seed)

    rest_d = pair_min_distances_for_poses(
        compiled, asset_root=asset_root, source_model=model, poses=[rest_pose], pairs=source4_only
    )
    swept_d = pair_min_distances_for_poses(
        compiled,
        asset_root=asset_root,
        source_model=model,
        poses=swept_poses,
        pairs=source4_only,
    )

    clear_tol = _REST_CONTACT_TOL * 0.5
    contact_tol = _REST_CONTACT_TOL * 0.1
    findings: list[ContactExclusionEvidence] = []
    for pair in sorted(source4_only):
        rest = rest_d.get(pair)
        swept = swept_d.get(pair)
        if rest is None or swept is None:
            continue
        d_rest = rest[0]
        d_motion, pose_index = swept
        if d_rest > clear_tol and d_motion <= contact_tol:
            pose = swept_poses[pose_index] if 0 <= pose_index < len(swept_poses) else {}
            findings.append(
                ContactExclusionEvidence(
                    part_a=pair[0],
                    part_b=pair[1],
                    rest_distance=d_rest,
                    motion_min_distance=d_motion,
                    pose={k: float(v) for k, v in dict(pose).items()},
                )
            )
    findings.sort(key=lambda f: f.motion_min_distance)
    return findings
