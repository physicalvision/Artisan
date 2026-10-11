from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from ..geometry_qc import (
    _aabb_intersection_depth,
    _aabb_intersection_volume,
    _identity4,
    _pair_key,
    _root_part_names,
    _transform_aabb,
    _unsupported_component_findings,
    compute_part_world_transforms,
    default_overlap_tol_from_env,
    default_overlap_volume_tol_from_env,
    default_penetration_depth_tol_from_env,
    strip_component_suffix,
    weld_group_of,
)
from ..self_collision import _active_relative_drivers, _moving_joints, moving_subtree
from .depth import DepthResult
from .pose_plan import PoseTable
from .resolve import ResolvedCollisionManifest
from .sweep import ContactRow, ContactTable

_RAM_ADVISORY_VOLUME_TOL = 5e-8
_RAM_BLOCK_VOLUME_TOL = 1e-6


@dataclass(frozen=True, slots=True)
class QueryBundle:
    evidence: dict[str, Any]
    required_detector_status: dict[str, dict[str, Any]]
    timings: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class ReachResolution:
    target_states: dict[str, str]
    driver_reach_findings: tuple[dict[str, Any], ...]
    declared_gate_witnesses: tuple[dict[str, Any], ...]
    accepted_replay_pose_ids: tuple[int, ...]
    selected_gate_combo_by_target: dict[str, int]
    owner_joint_by_pose_id: dict[int, str]


def _part_pair(row: ContactRow) -> tuple[str, str]:
    return _pair_key(row.part_a, row.part_b)


def _display_element(value: str) -> str:
    if "/" in value:
        part, name = value.split("/", 1)
        return f"{part}:{name}"
    return value


def _element_pair(row: ContactRow) -> tuple[str, str]:
    values = sorted(
        (_display_element(row.source_element_a), _display_element(row.source_element_b))
    )
    return values[0], values[1]


def _base_element_pair(row: ContactRow) -> tuple[str, str]:
    left, right = _element_pair(row)
    return tuple(sorted((strip_component_suffix(left), strip_component_suffix(right))))  # type: ignore[return-value]


def _rows_by_pose(contacts: ContactTable) -> dict[int, list[ContactRow]]:
    result: dict[int, list[ContactRow]] = defaultdict(list)
    for row in contacts.rows:
        result[row.pose_id].append(row)
    return result


def _contract_excluded_pairs(
    manifest: ResolvedCollisionManifest,
) -> set[tuple[str, str]]:
    return {
        _pair_key(str(pair[0]), str(pair[1]))
        for pair in manifest.simulator_contract.get("filter", {}).get("contact_excludes", [])
    }


def _colliding_rows(
    rows: Iterable[ContactRow],
    *,
    overlap_tol: float = 0.0,
    volume_tol: float = 0.0,
) -> list[ContactRow]:
    return [
        row
        for row in rows
        if row.colliding
        and all(value > overlap_tol for value in row.overlap_depth)
        and row.overlap_volume > volume_tol
    ]


def _rest_fingerprint(
    poses: PoseTable,
    rows_by_pose: dict[int, list[ContactRow]],
    *,
    contact_tol: float = 1e-3,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    selfcol_ids = list(poses.metadata.get("selfcol_pose_ids") or [])
    rest_id = selfcol_ids[0] if selfcol_ids else 0
    elements: set[tuple[str, str]] = set()
    parts: set[tuple[str, str]] = set()
    for row in rows_by_pose.get(rest_id, []):
        if row.colliding or row.min_distance <= contact_tol:
            elements.add(_element_pair(row))
            parts.add(_part_pair(row))
    return elements, parts


def _gate_evaluation(
    poses: PoseTable,
    rows_by_pose: dict[int, list[ContactRow]],
    reach_resolution: ReachResolution,
    *,
    source_pose_id: int,
    pair: tuple[str, str],
    depths: dict[tuple[int, tuple[str, str]], DepthResult],
) -> dict[str, Any] | None:
    witness_map = poses.metadata.get("gate_witnesses") or {}
    records = witness_map.get(f"{source_pose_id}|{pair[0]}|{pair[1]}") or []
    source_joints = set(poses.rows[source_pose_id].qpos)
    owners = {
        str(record.get("target_joint"))
        for record in records
        if isinstance(record, dict)
        and str(record.get("target_joint")) in source_joints
        and str(record.get("target_joint")) in reach_resolution.selected_gate_combo_by_target
    }
    if len(owners) != 1:
        return None
    owner_joint = next(iter(owners))
    combo_index = reach_resolution.selected_gate_combo_by_target[owner_joint]
    selected_record = next(
        (
            record
            for record in records
            if isinstance(record, dict)
            and record.get("target_joint") == owner_joint
            and int(record.get("combo_index", -1)) == combo_index
        ),
        None,
    )
    if selected_record is None:
        return None
    pose_id = int(selected_record["pose_id"])
    threshold = default_penetration_depth_tol_from_env()
    pair_rows = [
        row for row in _colliding_rows(rows_by_pose.get(pose_id, [])) if _part_pair(row) == pair
    ]
    depth = depths.get((pose_id, pair))
    clear = not pair_rows or (
        depth is not None
        and depth.status == "ok"
        and depth.depth_m is not None
        and depth.depth_m < threshold
    )
    return {
        "pose_id": pose_id,
        "pose": dict(poses.rows[pose_id].qpos),
        "target_joint": owner_joint,
        "selected_combo_index": combo_index,
        "policy_id": "full-collision-epa-v2",
        "collision_mode": "full-coacd",
        "penetration_depth_tol_m": threshold,
        "max_penetration_depth_m": depth.depth_m if depth is not None else 0.0,
        "clear": clear,
        "axis": depth.axis if depth is not None else "x",
        "elements": list(depth.piece_pair) if depth is not None and depth.piece_pair else None,
        "depth_method": depth.method if depth is not None else "none",
        "depth_signed": False,
        "depth_degenerate": bool(depth is not None and depth.status == "degenerate"),
    }


def _self_collision_query(
    poses: PoseTable,
    contacts: ContactTable,
    depths: dict[tuple[int, tuple[str, str]], DepthResult],
    reach_resolution: ReachResolution,
    *,
    declared_pairs: set[tuple[str, str]],
    declared_element_pairs: set[tuple[tuple[str, str], tuple[str, str]]],
) -> list[dict[str, Any]]:
    rows_by_pose = _rows_by_pose(contacts)
    rest_elements, _rest_parts = _rest_fingerprint(poses, rows_by_pose)
    overlap_tol = default_overlap_tol_from_env()
    volume_tol = default_overlap_volume_tol_from_env()
    pose_ids = list(poses.metadata.get("selfcol_pose_ids") or []) + list(
        poses.metadata.get("selfcol_refine_pose_ids") or []
    )
    reach_metadata = poses.metadata.get("reach")
    raw_movers = reach_metadata.get("movers") if isinstance(reach_metadata, dict) else {}
    movers = {
        str(name): {str(part) for part in parts}
        for name, parts in (raw_movers.items() if isinstance(raw_movers, dict) else ())
        if isinstance(parts, (list, tuple, set))
    }
    worst: dict[tuple[str, str], dict[str, Any]] = {}
    rank = {"CONDITIONAL": 0, "COORDINATION": 1, "SUSPICIOUS": 2, "DEFINITE": 3}
    for pose_id in pose_ids[1:]:
        for row in _colliding_rows(
            rows_by_pose.get(pose_id, []), overlap_tol=overlap_tol, volume_tol=volume_tol
        ):
            elements = _element_pair(row)
            if elements in rest_elements:
                continue
            pair = _part_pair(row)
            if (pair, _base_element_pair(row)) in declared_element_pairs:
                continue
            declared = pair in declared_pairs
            active_drivers = _active_relative_drivers(
                poses.rows[pose_id].qpos,
                pair,
                movers,
            )
            classification = (
                "SUSPICIOUS"
                if declared
                else ("COORDINATION" if len(active_drivers) >= 2 else "DEFINITE")
            )
            gate = None
            if not declared:
                gate = _gate_evaluation(
                    poses,
                    rows_by_pose,
                    reach_resolution,
                    source_pose_id=pose_id,
                    pair=pair,
                    depths=depths,
                )
                if gate is not None and gate["clear"]:
                    classification = "CONDITIONAL"
            finding_pose_id = int(gate["pose_id"]) if gate and not gate["clear"] else pose_id
            candidate = {
                "pair": list(pair),
                "elements": list(elements),
                "worst_pose": {
                    name: round(value, 5)
                    for name, value in poses.rows[finding_pose_id].qpos.items()
                },
                "clear_pose": (
                    {name: round(value, 5) for name, value in gate["pose"].items()}
                    if gate is not None and gate["clear"]
                    else None
                ),
                "gate_evaluation": gate,
                "classification": classification,
                "declared": declared,
                "overlap_volume": round(row.overlap_volume, 6),
                "note": (
                    "gate-clearable phantom"
                    if classification == "CONDITIONAL"
                    else (
                        "independent-driver coordination contact"
                        if classification == "COORDINATION"
                        else (
                            "gate-open witness still collides"
                            if gate
                            else "new element-pair under motion"
                        )
                    )
                ),
                "oracle_pose_ids": sorted({pose_id, finding_pose_id}),
                "sample_provenance": list(poses.rows[pose_id].provenance),
            }
            if classification == "COORDINATION":
                candidate["coordination_joints"] = list(active_drivers)
            current = worst.get(pair)
            if current is None or (rank[classification], row.overlap_volume) > (
                rank[str(current["classification"])],
                float(current["overlap_volume"]),
            ):
                worst[pair] = candidate
    return sorted(
        worst.values(),
        key=lambda value: (-rank[str(value["classification"])], -float(value["overlap_volume"])),
    )


def _connectivity_query(manifest: ResolvedCollisionManifest) -> list[dict[str, Any]]:
    entries_by_part: dict[str, list[dict[str, object]]] = defaultdict(list)
    for entry in manifest.exact_legality.entries:
        entries_by_part[str(entry["part_name"])].append(entry)
    edges = {tuple(sorted(edge)) for edge in manifest.exact_legality.adjacency_edges}
    findings: list[dict[str, Any]] = []
    for part_name, entries in sorted(entries_by_part.items()):
        if len(entries) <= 1:
            continue
        ids = [str(entry["legacy_entry_id"]) for entry in entries]
        remaining = set(ids)
        groups: list[set[str]] = []
        while remaining:
            group = {min(remaining)}
            queue = list(group)
            while queue:
                current = queue.pop()
                for candidate in ids:
                    if candidate in group:
                        continue
                    if tuple(sorted((current, candidate))) in edges:
                        group.add(candidate)
                        queue.append(candidate)
            remaining.difference_update(group)
            groups.append(group)
        largest = max(
            groups, key=lambda value: (len(value), -min(ids.index(item) for item in value))
        )
        if len(largest) == len(entries):
            continue
        disconnected = [
            f"{entry.get('name') or f'#{index}'}:{entry.get('geometry_name')}"
            for index, entry in enumerate(entries)
            if str(entry["legacy_entry_id"]) not in largest
        ]
        findings.append(
            {
                "part": part_name,
                "connected": len(largest),
                "total": len(entries),
                "disconnected": disconnected,
                "oracle_pose_ids": [],
                "sample_provenance": ["exact-legality-table"],
            }
        )
    return findings


def _support_query(model: object, poses: PoseTable, contacts: ContactTable) -> list[dict[str, Any]]:
    pose_ids = list(poses.metadata.get("support_pose_ids") or [])
    if not pose_ids:
        return []
    part_names = [str(part.name) for part in getattr(model, "parts", []) or []]
    joints = list(getattr(model, "articulations", []) or [])
    roots = _root_part_names(part_names, joints)
    rows_by_pose = _rows_by_pose(contacts)
    seen: set[tuple[str, ...]] = set()
    findings: list[dict[str, Any]] = []
    for pose_id in pose_ids:
        edges = {
            _part_pair(row)
            for row in rows_by_pose.get(pose_id, [])
            if row.colliding or row.min_distance <= 1e-3
        }
        for finding in _unsupported_component_findings(
            part_names,
            root_parts=roots,
            support_pairs=edges,
            pair_min_distances={},
            pose_index=pose_id,
            pose=poses.rows[pose_id].qpos,
            contact_tol=1e-3,
            backend="fcl-v2",
        ):
            parts = tuple(sorted(finding.parts or ((finding.part,) if finding.part else ())))
            if not parts or parts in seen:
                continue
            seen.add(parts)
            findings.append(
                {
                    "parts": list(parts),
                    "pose": {
                        name: round(value, 5) for name, value in poses.rows[pose_id].qpos.items()
                    },
                    "nearest_grounded_part": finding.nearest_part,
                    "approx_gap": finding.min_distance,
                    "note": "floats at a motion endpoint - add a connector holding it across travel",
                    "oracle_pose_ids": [pose_id],
                    "sample_provenance": list(poses.rows[pose_id].provenance),
                }
            )
    return findings


def _reach_query(
    model: object,
    manifest: ResolvedCollisionManifest,
    poses: PoseTable,
    contacts: ContactTable,
    *,
    declared_pairs: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    metadata = poses.metadata.get("reach") or {}
    rows_by_pose = _rows_by_pose(contacts)
    _rest_elements, rest_pairs = _rest_fingerprint(poses, rows_by_pose)
    adjacency = {
        _pair_key(str(joint.parent), str(joint.child))
        for joint in getattr(model, "articulations", []) or []
    }
    contract_excludes = _contract_excluded_pairs(manifest)
    exempt = rest_pairs | adjacency | contract_excludes | declared_pairs
    gate_links = set(metadata.get("gate_links") or [])
    findings: list[dict[str, Any]] = []
    ranges = metadata.get("ranges") or {}
    movers = {name: set(parts) for name, parts in (metadata.get("movers") or {}).items()}
    for joint_name in metadata.get("joint_order") or []:
        lo, hi = (float(value) for value in ranges[joint_name])
        rest = min(max(0.0, lo), hi)
        blocks: dict[int, list[ContactRow]] = {}
        for side_units in (metadata.get("sides") or {}).get(joint_name, {}).values():
            for unit in side_units:
                for pose_id in unit:
                    rows = [
                        row
                        for row in _colliding_rows(rows_by_pose.get(pose_id, []))
                        if _part_pair(row) not in exempt
                        and (
                            (row.part_a in movers[joint_name]) != (row.part_b in movers[joint_name])
                        )
                    ]
                    if rows:
                        blocks[pose_id] = rows

        def reach(units: list[list[int]], bound: float) -> tuple[float, int | None]:
            previous = rest
            for unit in units:
                if unit and all(pose_id in blocks for pose_id in unit):

                    def genuine_volume(pose_id: int) -> float:
                        return max(
                            (
                                row.overlap_volume
                                for row in blocks[pose_id]
                                if row.part_a not in gate_links and row.part_b not in gate_links
                            ),
                            default=0.0,
                        )

                    genuine = [pose_id for pose_id in unit if genuine_volume(pose_id) > 0.0]
                    selected = max(genuine or unit, key=genuine_volume)
                    return previous, selected
                if unit:
                    previous = poses.rows[unit[0]].qpos.get(joint_name, previous)
            return bound, None

        sides = (metadata.get("sides") or {}).get(joint_name, {})
        reach_lo, blocked_lo = reach(list(sides.get("lower") or []), lo)
        reach_hi, blocked_hi = reach(list(sides.get("upper") or []), hi)
        fraction = (reach_hi - reach_lo) / (hi - lo) if hi > lo else 1.0
        if fraction >= 0.9:
            continue
        blocked_sides: list[dict[str, Any]] = []
        for side, pose_id in (("lower", blocked_lo), ("upper", blocked_hi)):
            if pose_id is None:
                continue
            genuine = [
                row
                for row in blocks[pose_id]
                if row.part_a not in gate_links and row.part_b not in gate_links
            ]
            row = max(genuine or blocks[pose_id], key=lambda value: value.overlap_volume)
            blocked_sides.append(
                {
                    "side": side,
                    "at": round(poses.rows[pose_id].qpos.get(joint_name, rest), 5),
                    "pair": list(_part_pair(row)),
                    "elements": list(_element_pair(row)),
                    "pose_id": pose_id,
                }
            )
        dominant = max(
            blocked_sides,
            key=lambda value: abs(float(value["at"]) - rest),
            default=None,
        )
        findings.append(
            {
                "joint": joint_name,
                "nominal": [round(lo, 5), round(hi, 5)],
                "reachable": [round(reach_lo, 5), round(reach_hi, 5)],
                "reachable_fraction": round(fraction, 3),
                "blocked_by": dominant["pair"] if dominant else None,
                "blocked_at": dominant["at"] if dominant else None,
                "elements": dominant["elements"] if dominant else None,
                "blocked_sides": [
                    {key: value for key, value in side.items() if key != "pose_id"}
                    for side in blocked_sides
                ],
                "note": "resolved-collision outward sweep",
                "oracle_pose_ids": [int(side["pose_id"]) for side in blocked_sides],
                "sample_provenance": [f"reach:{joint_name}"],
            }
        )
    return sorted(findings, key=lambda value: float(value["reachable_fraction"]))


def _gate_opening_score(
    combo: dict[str, float],
    ranges: dict[str, object],
) -> float:
    score = 0.0
    for name, value in combo.items():
        raw_range = ranges.get(name)
        if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
            continue
        lo, hi = (float(item) for item in raw_range)
        rest = min(max(0.0, lo), hi)
        score += abs(float(value) - rest) / max(hi - lo, 1e-12)
    return score


def _pair_changes_under_joints(
    pair: tuple[str, str],
    joint_names: list[str],
    movers: dict[str, set[str]],
) -> bool:
    return any(
        (pair[0] in movers.get(name, set())) != (pair[1] in movers.get(name, set()))
        for name in joint_names
    )


def _pose_blocking_pairs(
    pose_id: int,
    *,
    target_joint: str,
    gate_joints: list[str],
    rows_by_pose: dict[int, list[ContactRow]],
    depths: dict[tuple[int, tuple[str, str]], DepthResult],
    movers: dict[str, set[str]],
    contract_excludes: set[tuple[str, str]],
    threshold: float,
) -> tuple[set[tuple[str, str]], float]:
    blocked: set[tuple[str, str]] = set()
    worst_depth = 0.0
    affected_joints = [target_joint, *gate_joints]
    for row in rows_by_pose.get(pose_id, []):
        if not row.colliding:
            continue
        pair = _part_pair(row)
        if pair in contract_excludes or not _pair_changes_under_joints(
            pair, affected_joints, movers
        ):
            continue
        depth = depths.get((pose_id, pair))
        if (
            depth is not None
            and depth.status == "ok"
            and depth.depth_m is not None
            and depth.depth_m < threshold
        ):
            continue
        blocked.add(pair)
        if depth is None or depth.status != "ok" or depth.depth_m is None:
            worst_depth = float("inf")
        elif worst_depth != float("inf"):
            worst_depth = max(worst_depth, float(depth.depth_m))
    return blocked, worst_depth


def _blocking_pair_evidence(
    pose_ids: list[int],
    *,
    target_joint: str,
    gate_joints: list[str],
    poses: PoseTable,
    rows_by_pose: dict[int, list[ContactRow]],
    depths: dict[tuple[int, tuple[str, str]], DepthResult],
    movers: dict[str, set[str]],
    contract_excludes: set[tuple[str, str]],
    threshold: float,
) -> list[dict[str, Any]]:

    worst_by_pair: dict[tuple[str, str], tuple[tuple[float, ...], dict[str, Any]]] = {}
    for pose_id in pose_ids:
        blocking_pairs, _worst_depth = _pose_blocking_pairs(
            pose_id,
            target_joint=target_joint,
            gate_joints=gate_joints,
            rows_by_pose=rows_by_pose,
            depths=depths,
            movers=movers,
            contract_excludes=contract_excludes,
            threshold=threshold,
        )
        for pair in sorted(blocking_pairs):
            depth = depths.get((pose_id, pair))
            pair_rows = [
                row
                for row in rows_by_pose.get(pose_id, [])
                if row.colliding and _part_pair(row) == pair
            ]
            source_row = None
            if depth is not None and depth.piece_pair is not None:
                source_row = next(
                    (
                        row
                        for row in pair_rows
                        if (row.piece_a, row.piece_b) == depth.piece_pair
                        or (row.piece_b, row.piece_a) == depth.piece_pair
                    ),
                    None,
                )
            if source_row is None and pair_rows:
                source_row = max(
                    pair_rows,
                    key=lambda row: (float(row.overlap_volume), _element_pair(row)),
                )

            measurable = bool(
                depth is not None and depth.status == "ok" and depth.depth_m is not None
            )
            rank = (
                0.0 if measurable else 1.0,
                float(depth.depth_m) if measurable and depth is not None else 0.0,
                float(source_row.overlap_volume) if source_row is not None else 0.0,
                -float(pose_id),
            )
            finding = {
                "pair": list(pair),
                "elements": list(_element_pair(source_row)) if source_row is not None else None,
                "penetration_depth_m": (
                    float(depth.depth_m) if measurable and depth is not None else None
                ),
                "depth_status": depth.status if depth is not None else "missing",
                "depth_method": depth.method if depth is not None else None,
                "axis": depth.axis if depth is not None else None,
                "overlap_volume_m3": (
                    float(source_row.overlap_volume) if source_row is not None else None
                ),
                "worst_pose": {
                    name: float(value) for name, value in poses.rows[pose_id].qpos.items()
                },
                "oracle_pose_id": int(pose_id),
            }
            current = worst_by_pair.get(pair)
            if current is None or rank > current[0]:
                worst_by_pair[pair] = (rank, finding)
    return [worst_by_pair[pair][1] for pair in sorted(worst_by_pair)]


def _resolve_reach(
    manifest: ResolvedCollisionManifest,
    poses: PoseTable,
    contacts: ContactTable,
    depths: dict[tuple[int, tuple[str, str]], DepthResult],
    *,
    driver_reach_findings: list[dict[str, Any]],
    gate_topology_issues: list[dict[str, Any]],
    unresolved_gates: list[dict[str, Any]],
) -> ReachResolution:
    metadata = poses.metadata.get("reach") or {}
    rows_by_pose = _rows_by_pose(contacts)
    ranges = dict(metadata.get("ranges") or {})
    movers = {name: set(parts) for name, parts in (metadata.get("movers") or {}).items()}
    gate_map = {
        str(name): [str(value) for value in values]
        for name, values in (metadata.get("gate_map") or {}).items()
    }
    gate_combos = {
        str(name): [
            {str(joint): float(value) for joint, value in combo.items()} for combo in combos
        ]
        for name, combos in (metadata.get("gate_combos") or {}).items()
    }
    sample_units = metadata.get("sample_units") or {}
    witness_rest = metadata.get("witness_rest_pose_ids") or {}
    contract_excludes = _contract_excluded_pairs(manifest)
    threshold = default_penetration_depth_tol_from_env()
    issue_targets = {
        str(issue.get("target_joint") or issue.get("target") or "")
        for issue in gate_topology_issues
        if isinstance(issue, dict)
    }
    target_states: dict[str, str] = {}
    selected: dict[str, int] = {}
    witnesses: list[dict[str, Any]] = []

    for target_joint, combos in sorted(gate_combos.items()):
        gate_joints = list(gate_map.get(target_joint, ()))
        units = [
            unit
            for side in ("lower", "upper")
            for unit in (sample_units.get(target_joint, {}) or {}).get(side, [])
            if isinstance(unit, dict)
        ]
        if target_joint in issue_targets:
            target_states[target_joint] = "unsupported_gate_topology"
            witnesses.append(
                {
                    "joint": target_joint,
                    "state": "unsupported_gate_topology",
                    "gate_joints": sorted(gate_joints),
                    "gate_pose": None,
                    "target_range": list(ranges.get(target_joint) or []),
                    "sample_count": 0,
                    "clear": False,
                    "blocking_pairs": [],
                    "blocking_pair_evidence": [],
                    "blocked_pose": None,
                    "worst_failed_pose": None,
                    "selected_combo_index": None,
                    "oracle_pose_ids": [],
                    "sample_provenance": [f"reach:{target_joint}:unsupported-topology"],
                }
            )
            continue

        candidates: list[dict[str, Any]] = []
        for combo_index, combo in enumerate(combos):
            rest_ids = list(witness_rest.get(target_joint) or [])
            pose_ids = [rest_ids[combo_index]] if combo_index < len(rest_ids) else []
            pose_ids.extend(
                int(unit["combo_pose_ids"][combo_index])
                for unit in units
                if combo_index < len(unit.get("combo_pose_ids") or [])
            )
            complete = len(pose_ids) == len(units) + 1
            failed_pose_ids: list[int] = []
            blocking_pairs: set[tuple[str, str]] = set()
            worst_depth = 0.0
            for pose_id in pose_ids:
                pose_pairs, pose_depth = _pose_blocking_pairs(
                    pose_id,
                    target_joint=target_joint,
                    gate_joints=gate_joints,
                    rows_by_pose=rows_by_pose,
                    depths=depths,
                    movers=movers,
                    contract_excludes=contract_excludes,
                    threshold=threshold,
                )
                if pose_pairs:
                    failed_pose_ids.append(pose_id)
                    blocking_pairs.update(pose_pairs)
                    worst_depth = max(worst_depth, pose_depth)
            candidates.append(
                {
                    "combo_index": combo_index,
                    "combo": combo,
                    "pose_ids": pose_ids,
                    "failed_pose_ids": failed_pose_ids,
                    "blocking_pairs": blocking_pairs,
                    "opening_score": _gate_opening_score(combo, ranges),
                    "worst_depth": worst_depth,
                    "complete": complete,
                }
            )

        if any(not bool(candidate["complete"]) for candidate in candidates):
            raise ValueError(
                f"gate search metadata is incomplete for target joint {target_joint!r}"
            )

        clear_candidates = [
            candidate
            for candidate in candidates
            if candidate["complete"] and not candidate["failed_pose_ids"]
        ]
        if clear_candidates:
            chosen = max(
                clear_candidates,
                key=lambda candidate: (
                    float(candidate["opening_score"]),
                    -int(candidate["combo_index"]),
                ),
            )
            state = "resolved"
            selected[target_joint] = int(chosen["combo_index"])
        elif candidates:
            chosen = min(
                candidates,
                key=lambda candidate: (
                    len(candidate["failed_pose_ids"]),
                    float(candidate["worst_depth"]),
                    -float(candidate["opening_score"]),
                    int(candidate["combo_index"]),
                ),
            )
            state = "no_fixed_witness"
        else:
            chosen = {
                "combo_index": None,
                "combo": {},
                "pose_ids": [],
                "failed_pose_ids": [],
                "blocking_pairs": set(),
                "opening_score": 0.0,
                "worst_depth": float("inf"),
                "complete": False,
            }
            state = "no_fixed_witness"
        target_states[target_joint] = state
        failed_pose_ids = [int(value) for value in chosen["failed_pose_ids"]]
        worst_pose_id = max(
            failed_pose_ids,
            key=lambda pose_id: _pose_blocking_pairs(
                pose_id,
                target_joint=target_joint,
                gate_joints=gate_joints,
                rows_by_pose=rows_by_pose,
                depths=depths,
                movers=movers,
                contract_excludes=contract_excludes,
                threshold=threshold,
            )[1],
            default=None,
        )
        witnesses.append(
            {
                "joint": target_joint,
                "state": state,
                "gate_joints": sorted(gate_joints),
                "gate_pose": {
                    name: float(value) for name, value in sorted(chosen["combo"].items())
                },
                "target_range": [float(value) for value in ranges.get(target_joint, ())],
                "sample_count": len(chosen["pose_ids"]),
                "clear": state == "resolved",
                "blocking_pairs": [list(pair) for pair in sorted(chosen["blocking_pairs"])],
                "blocking_pair_evidence": _blocking_pair_evidence(
                    failed_pose_ids,
                    target_joint=target_joint,
                    gate_joints=gate_joints,
                    poses=poses,
                    rows_by_pose=rows_by_pose,
                    depths=depths,
                    movers=movers,
                    contract_excludes=contract_excludes,
                    threshold=threshold,
                ),
                "blocked_pose": (
                    dict(poses.rows[failed_pose_ids[0]].qpos) if failed_pose_ids else None
                ),
                "worst_failed_pose": (
                    dict(poses.rows[worst_pose_id].qpos) if worst_pose_id is not None else None
                ),
                "selected_combo_index": chosen["combo_index"],
                "oracle_pose_ids": failed_pose_ids,
                "sample_provenance": [f"reach:{target_joint}:fixed-witness"],
            }
        )

    for target_joint in sorted(issue_targets - set(target_states)):
        target_states[target_joint] = "unsupported_gate_topology"
        witnesses.append(
            {
                "joint": target_joint,
                "state": "unsupported_gate_topology",
                "gate_joints": [],
                "gate_pose": None,
                "target_range": list(ranges.get(target_joint) or []),
                "sample_count": 0,
                "clear": False,
                "blocking_pairs": [],
                "blocking_pair_evidence": [],
                "blocked_pose": None,
                "worst_failed_pose": None,
                "selected_combo_index": None,
                "oracle_pose_ids": [],
                "sample_provenance": [f"reach:{target_joint}:unsupported-topology"],
            }
        )

    all_resolved = (
        not gate_topology_issues
        and not unresolved_gates
        and all(state == "resolved" for state in target_states.values())
    )
    accepted: list[int] = []
    owner_by_pose: dict[int, str] = {}
    outward_ids = list(metadata.get("outward_pose_ids") or [])
    if all_resolved and outward_ids:
        accepted.append(int(outward_ids[0]))
        for joint_name in metadata.get("joint_order") or []:
            for side in ("lower", "upper"):
                for unit in (sample_units.get(joint_name, {}) or {}).get(side, []):
                    pose_ids = [int(value) for value in unit.get("combo_pose_ids") or []]
                    if joint_name in gate_combos:
                        combo_index = selected[joint_name]
                        selected_ids = pose_ids[combo_index : combo_index + 1]
                    else:
                        selected_ids = pose_ids
                    for pose_id in selected_ids:
                        accepted.append(pose_id)
                        owner_by_pose.setdefault(pose_id, str(joint_name))
        for joint_name in metadata.get("joint_order") or []:
            if joint_name not in selected:
                continue
            rest_ids = [int(value) for value in witness_rest.get(joint_name) or []]
            combo_index = selected[joint_name]
            if combo_index < len(rest_ids):
                pose_id = rest_ids[combo_index]
                accepted.append(pose_id)
                owner_by_pose.setdefault(pose_id, str(joint_name))
        accepted.extend(int(value) for value in metadata.get("combined_pose_ids") or [])

    return ReachResolution(
        target_states=target_states,
        driver_reach_findings=tuple(driver_reach_findings),
        declared_gate_witnesses=tuple(witnesses),
        accepted_replay_pose_ids=tuple(dict.fromkeys(accepted)),
        selected_gate_combo_by_target=selected,
        owner_joint_by_pose_id=owner_by_pose,
    )


def resolve_reach_policy(
    model: object,
    manifest: ResolvedCollisionManifest,
    poses: PoseTable,
    contacts: ContactTable,
    depths: dict[tuple[int, tuple[str, str]], DepthResult],
    *,
    declared_pairs: set[tuple[str, str]],
    gate_topology_issues: list[dict[str, Any]],
    unresolved_gates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], ReachResolution]:

    reach = _reach_query(
        model,
        manifest,
        poses,
        contacts,
        declared_pairs=declared_pairs,
    )
    resolution = _resolve_reach(
        manifest,
        poses,
        contacts,
        depths,
        driver_reach_findings=reach,
        gate_topology_issues=gate_topology_issues,
        unresolved_gates=unresolved_gates,
    )
    return reach, resolution


def _penetration_query(
    model: object,
    manifest: ResolvedCollisionManifest,
    poses: PoseTable,
    contacts: ContactTable,
    depths: dict[tuple[int, tuple[str, str]], DepthResult],
    reach_resolution: ReachResolution,
    *,
    declared_pairs: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    reach_ids = set(reach_resolution.accepted_replay_pose_ids)
    contract_excludes = _contract_excluded_pairs(manifest)
    rows_by_pose = _rows_by_pose(contacts)
    movers = {joint.name: moving_subtree(model, joint) for joint in _moving_joints(model)}
    candidates: list[dict[str, Any]] = []
    threshold = default_penetration_depth_tol_from_env()
    for (pose_id, pair), depth in depths.items():
        if pose_id not in reach_ids or pair in contract_excludes or not depth.colliding:
            continue
        pair_rows = [row for row in rows_by_pose.get(pose_id, []) if _part_pair(row) == pair]
        volume = max((row.overlap_volume for row in pair_rows), default=0.0)
        if depth.status == "ok" and depth.depth_m is not None and depth.depth_m < threshold:
            continue
        posed_crossing = [
            name
            for name, subtree in movers.items()
            if name in poses.rows[pose_id].qpos and ((pair[0] in subtree) != (pair[1] in subtree))
        ]
        joint = reach_resolution.owner_joint_by_pose_id.get(pose_id)
        if joint is None:
            joint = (
                min(posed_crossing, key=lambda name: len(movers[name])) if posed_crossing else ""
            )
        declared = pair in declared_pairs
        candidates.append(
            {
                "joint": joint,
                "pair": list(pair),
                "elements": list(depth.piece_pair or pair),
                "penetration_depth": (
                    round(depth.depth_m, 6) if depth.depth_m is not None else None
                ),
                "penetration_volume": round(volume, 9),
                "worst_pose": {
                    name: round(value, 5) for name, value in poses.rows[pose_id].qpos.items()
                },
                "rest_depth": None,
                "constant": False,
                "axis": depth.axis,
                "declared": declared,
                "blocking": not declared,
                "depth_method": depth.method,
                "depth_signed": False,
                "depth_degenerate": depth.status == "degenerate",
                "finding_type": (
                    "unmeasurable_penetration"
                    if depth.status == "degenerate"
                    else "full_collision_penetration"
                ),
                "note": "resolved collision body penetration; add positive clearance",
                "oracle_pose_ids": [pose_id],
                "sample_provenance": list(poses.rows[pose_id].provenance),
            }
        )
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for finding in candidates:
        pair = tuple(finding["pair"])
        current = by_pair.get(pair)
        rank = (
            float("inf")
            if finding["penetration_depth"] is None
            else float(finding["penetration_depth"])
        )
        current_rank = (
            -1.0
            if current is None
            else (
                float("inf")
                if current["penetration_depth"] is None
                else float(current["penetration_depth"])
            )
        )
        if current is None or rank > current_rank:
            by_pair[pair] = finding
    return sorted(
        by_pair.values(),
        key=lambda value: (
            value["penetration_depth"] is not None,
            -(float(value["penetration_depth"] or 0.0)),
        ),
    )


def _ram_pose_obeys_fixed_witness(
    finding: dict[str, Any],
    poses: PoseTable,
    reach_resolution: ReachResolution,
) -> bool:
    pose = None
    oracle_pose_ids = finding.get("oracle_pose_ids")
    if isinstance(oracle_pose_ids, list) and oracle_pose_ids:
        pose_id = oracle_pose_ids[0]
        if isinstance(pose_id, int) and 0 <= pose_id < len(poses.rows):
            pose = poses.rows[pose_id].qpos
    if pose is None:
        pose = finding.get("worst_pose")
    if not isinstance(pose, dict):
        return True
    metadata = poses.metadata.get("reach") or {}
    ranges = metadata.get("ranges") or {}
    gate_combos = metadata.get("gate_combos") or {}
    for target, combo_index in reach_resolution.selected_gate_combo_by_target.items():
        if target not in pose:
            continue
        raw_range = ranges.get(target)
        if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
            continue
        lo, hi = (float(value) for value in raw_range)
        rest = min(max(0.0, lo), hi)
        if abs(float(pose[target]) - rest) <= 1e-12:
            continue
        combos = gate_combos.get(target) or []
        if combo_index >= len(combos):
            return False
        selected = combos[combo_index]
        if any(pose.get(name) != float(value) for name, value in selected.items()):
            return False
    return True


def _ram_query(
    model: object,
    manifest: ResolvedCollisionManifest,
    poses: PoseTable,
    contacts: ContactTable,
    *,
    declared_pairs: set[tuple[str, str]],
    declared_element_pairs: set[tuple[tuple[str, str], tuple[str, str]]],
) -> list[dict[str, Any]]:
    rows_by_pose = _rows_by_pose(contacts)
    rest_elements, _rest_pairs = _rest_fingerprint(poses, rows_by_pose)
    entries_by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for entry in manifest.exact_legality.entries:
        entries_by_source[str(entry["source_element_id"])].append(entry)
    moving_parts: set[str] = set()
    for joint in _moving_joints(model):
        moving_parts.update(moving_subtree(model, joint))
    welds = weld_group_of(model)
    worst: dict[tuple[str, str], dict[str, Any]] = {}
    pose_ids = list(poses.metadata.get("selfcol_pose_ids") or [])
    for pose_id in pose_ids:
        transforms = compute_part_world_transforms(model, poses.rows[pose_id].qpos)
        for row in _colliding_rows(rows_by_pose.get(pose_id, [])):
            elements = _element_pair(row)
            if elements not in rest_elements:
                continue
            pair = _part_pair(row)
            if moving_parts and pair[0] not in moving_parts and pair[1] not in moving_parts:
                continue
            if welds.get(pair[0]) is not None and welds.get(pair[0]) == welds.get(pair[1]):
                continue
            left_entries = entries_by_source.get(row.source_element_a, [])
            right_entries = entries_by_source.get(row.source_element_b, [])
            best_volume = 0.0
            best_span = 0.0
            for left in left_entries:
                left_aabb = _transform_aabb(
                    (tuple(left["aabb_min"]), tuple(left["aabb_max"])),  # type: ignore[arg-type]
                    transforms.get(str(left["part_name"]), _identity4()),
                )
                for right in right_entries:
                    right_aabb = _transform_aabb(
                        (tuple(right["aabb_min"]), tuple(right["aabb_max"])),  # type: ignore[arg-type]
                        transforms.get(str(right["part_name"]), _identity4()),
                    )
                    volume = _aabb_intersection_volume(left_aabb, right_aabb)
                    if volume > best_volume:
                        best_volume = volume
                        best_span = max(_aabb_intersection_depth(left_aabb, right_aabb))
            if best_volume < _RAM_ADVISORY_VOLUME_TOL:
                continue
            declared = (
                pair in declared_pairs or (pair, _base_element_pair(row)) in declared_element_pairs
            )
            candidate = {
                "pair": list(pair),
                "elements": list(elements),
                "worst_pose": {
                    name: round(value, 5) for name, value in poses.rows[pose_id].qpos.items()
                },
                "penetration_volume": round(best_volume, 8),
                "penetration_span": round(best_span, 5),
                "blocking": best_volume >= _RAM_BLOCK_VOLUME_TOL and not declared,
                "note": "a joint connection rams across its motion",
                "oracle_pose_ids": [pose_id],
                "sample_provenance": list(poses.rows[pose_id].provenance),
            }
            current = worst.get(pair)
            if current is None or best_volume > float(current["penetration_volume"]):
                worst[pair] = candidate
    return sorted(worst.values(), key=lambda value: -float(value["penetration_volume"]))


def _contact_exclusion_query(
    manifest: ResolvedCollisionManifest,
    poses: PoseTable,
    contacts: ContactTable,
) -> list[dict[str, Any]]:
    rows_by_pose = _rows_by_pose(contacts)
    exclusions = {
        _pair_key(str(pair[0]), str(pair[1]))
        for pair in manifest.simulator_contract.get("filter", {}).get("contact_excludes", [])
    }
    pose_ids = list(poses.metadata.get("exclusion_pose_ids") or [])
    if not pose_ids:
        return []
    rest_id = pose_ids[0]
    findings: list[dict[str, Any]] = []
    for pair in sorted(exclusions):
        rest_distance = min(
            (row.min_distance for row in rows_by_pose.get(rest_id, []) if _part_pair(row) == pair),
            default=float("inf"),
        )
        best = min(
            (
                (row.min_distance, pose_id)
                for pose_id in pose_ids
                for row in rows_by_pose.get(pose_id, [])
                if _part_pair(row) == pair
            ),
            default=None,
        )
        if best is None or best[0] >= rest_distance:
            continue
        findings.append(
            {
                "pair": list(pair),
                "rest_distance": round(rest_distance, 6),
                "motion_min_distance": round(best[0], 6),
                "pose": {name: round(value, 5) for name, value in poses.rows[best[1]].qpos.items()},
                "oracle_pose_ids": [rest_id, best[1]],
                "sample_provenance": ["contact-exclusion"],
            }
        )
    return findings


def run_queries(
    model: object,
    manifest: ResolvedCollisionManifest,
    poses: PoseTable,
    contacts: ContactTable,
    depths: dict[tuple[int, tuple[str, str]], DepthResult],
    *,
    declared_pairs: set[tuple[str, str]] | None = None,
    declared_element_pairs: set[tuple[tuple[str, str], tuple[str, str]]] | None = None,
    precomputed_ram_findings: list[dict[str, Any]] | None = None,
    precomputed_ram_error: str | None = None,
    precomputed_driver_reach_findings: list[dict[str, Any]] | None = None,
    precomputed_reach_resolution: ReachResolution | None = None,
    gate_topology_issues: list[dict[str, Any]] | None = None,
    unresolved_gates: list[dict[str, Any]] | None = None,
) -> QueryBundle:
    declared = {_pair_key(*pair) for pair in (declared_pairs or set())}
    declared_elements = declared_element_pairs or set()
    status: dict[str, dict[str, Any]] = {}
    timings: list[dict[str, Any]] = []

    def execute(
        name: str,
        callback: Callable[[], Iterable[dict[str, Any]]],
        *,
        advisory: bool = False,
    ) -> list[dict[str, Any]]:
        started = time.perf_counter()
        try:
            findings = list(callback())
        except Exception as exc:  # noqa: BLE001 - detector failures remain query-scoped
            timings.append(
                {
                    "kind": "gate_stage",
                    "stage": f"query:{name}",
                    "wall_s": time.perf_counter() - started,
                    "outcome": "failed",
                }
            )
            status[name] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "detail": str(exc),
                "advisory": advisory,
            }
            return []
        timings.append(
            {
                "kind": "gate_stage",
                "stage": f"query:{name}",
                "wall_s": time.perf_counter() - started,
                "outcome": "ok",
            }
        )
        status[name] = {
            "status": "complete",
            "finding_count": len(findings),
            "advisory": advisory,
        }
        return findings

    connectivity = execute("connectivity", lambda: _connectivity_query(manifest))
    reach = execute(
        "driver_reach",
        lambda: (
            precomputed_driver_reach_findings
            if precomputed_driver_reach_findings is not None
            else _reach_query(model, manifest, poses, contacts, declared_pairs=declared)
        ),
    )
    reach_resolution: ReachResolution | None = precomputed_reach_resolution

    def resolve_gate_witnesses() -> Iterable[dict[str, Any]]:
        nonlocal reach_resolution
        if reach_resolution is None:
            reach_resolution = _resolve_reach(
                manifest,
                poses,
                contacts,
                depths,
                driver_reach_findings=reach,
                gate_topology_issues=list(gate_topology_issues or []),
                unresolved_gates=list(unresolved_gates or []),
            )
        return reach_resolution.declared_gate_witnesses

    gate_witnesses = execute("gate_witness", resolve_gate_witnesses)
    if reach_resolution is None:
        reach_resolution = ReachResolution(
            target_states={},
            driver_reach_findings=tuple(reach),
            declared_gate_witnesses=(),
            accepted_replay_pose_ids=(),
            selected_gate_combo_by_target={},
            owner_joint_by_pose_id={},
        )
    self_collision = execute(
        "self_collision",
        lambda: _self_collision_query(
            poses,
            contacts,
            depths,
            reach_resolution,
            declared_pairs=declared,
            declared_element_pairs=declared_elements,
        ),
    )
    support = execute("support", lambda: _support_query(model, poses, contacts))
    penetration = execute(
        "penetration",
        lambda: _penetration_query(
            model,
            manifest,
            poses,
            contacts,
            depths,
            reach_resolution,
            declared_pairs=declared,
        ),
    )

    def connection_ram_query() -> Iterable[dict[str, Any]]:
        if precomputed_ram_error is not None:
            raise RuntimeError(precomputed_ram_error)
        findings = (
            precomputed_ram_findings
            if precomputed_ram_findings is not None
            else _ram_query(
                model,
                manifest,
                poses,
                contacts,
                declared_pairs=declared,
                declared_element_pairs=declared_elements,
            )
        )
        return [
            finding
            for finding in findings
            if _ram_pose_obeys_fixed_witness(finding, poses, reach_resolution)
        ]

    ram = execute("connection_ram", connection_ram_query)
    exclusion = execute(
        "contact_exclusion",
        lambda: _contact_exclusion_query(manifest, poses, contacts),
        advisory=True,
    )
    topology = [
        {
            "part": str(entry["part_name"]),
            "entry": str(entry["legacy_entry_id"]),
            "topology": dict(entry.get("topology") or {}),
        }
        for entry in manifest.exact_legality.entries
        if not bool((entry.get("topology") or {}).get("watertight", True))  # type: ignore[union-attr]
    ]
    status["source_topology"] = {
        "status": "complete",
        "finding_count": len(topology),
        "advisory": True,
    }
    timings.append(
        {
            "kind": "gate_stage",
            "stage": "query:source_topology",
            "wall_s": 0.0,
            "outcome": "ok",
        }
    )
    evidence = {
        "motion_contact_candidates": self_collision,
        "geometry_island_warnings": connectivity,
        "motion_support_findings": support,
        "driver_reach_findings": reach,
        "contact_exclusion_evidence": exclusion,
        "connection_cleanliness_findings": ram,
        "full_collision_penetrations": penetration,
        "accepted_fcl_poses": [
            dict(poses.rows[pose_id].qpos) for pose_id in reach_resolution.accepted_replay_pose_ids
        ],
        "declared_gate_witnesses": gate_witnesses,
        "gate_resolution": {
            "target_states": dict(reach_resolution.target_states),
            "selected_gate_combo_by_target": dict(reach_resolution.selected_gate_combo_by_target),
            "accepted_replay_pose_ids": list(reach_resolution.accepted_replay_pose_ids),
            "topology_issues": [dict(value) for value in gate_topology_issues or []],
            "unresolved_gates": [dict(value) for value in unresolved_gates or []],
        },
        "source_topology_diagnostics": topology,
    }
    return QueryBundle(
        evidence=evidence,
        required_detector_status=status,
        timings=tuple(timings),
    )
