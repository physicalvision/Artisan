from __future__ import annotations

import ast
import hashlib
import json
import resource
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from ..geometry_qc import (
    _pair_key,
    default_overlap_tol_from_env,
    default_overlap_volume_tol_from_env,
    default_penetration_depth_tol_from_env,
)
from ..self_collision import (
    _RAM_ADVISORY_VOLUME_TOL,
    _RAM_BLOCK_VOLUME_TOL,
    _joint_value_grid,
    _moving_joints,
    moving_subtree,
    prepare_connection_ram_scan,
)
from .depth import DepthResult, evaluate_depth
from .pose_plan import PosePlanPolicy, PoseTable, build_pose_table
from .queries import QueryBundle, ReachResolution, resolve_reach_policy, run_queries
from .resolve import ResolvedCollisionManifest, canonical_bytes, resolve_collision_geometry
from .sweep import (
    ContactTable,
    RestMemo,
    map_pose_shards,
    merge_contact_tables,
    sweep_contacts,
    sweep_contacts_sharded,
)
from .world import build_collision_world

MOTION_SCAN_ALGORITHM_VERSION = "motion-evidence-v14"
MOTION_EVIDENCE_SCHEMA_VERSION = "motion-evidence-envelope-v14"
DEPTH_POLICY_VERSION = "epa-pieces-v1"
DETECTOR_POLICY_VERSION = "detector-policy-v1"

RAM_CONNECTION_TOL = 1e-3


def effective_detector_policy() -> dict[str, Any]:
    return {
        "policy_version": DETECTOR_POLICY_VERSION,
        "overlap_tol": default_overlap_tol_from_env(),
        "overlap_volume_tol": default_overlap_volume_tol_from_env(),
        "penetration_depth_tol": default_penetration_depth_tol_from_env(),
        "ram_connection_tol": RAM_CONNECTION_TOL,
        "ram_advisory_volume_tol": _RAM_ADVISORY_VOLUME_TOL,
        "ram_advisory_comparison": ">=",
        "ram_block_volume_tol": _RAM_BLOCK_VOLUME_TOL,
        "ram_block_comparison": ">=",
    }


def detector_policy_digest() -> str:
    return hashlib.sha256(canonical_bytes(effective_detector_policy())).hexdigest()


def _timed(
    timings: list[dict[str, Any]],
    stage: str,
    callback: Callable[[], Any],
) -> Any:
    started = time.perf_counter()
    try:
        value = callback()
    except BaseException:
        timings.append(
            {
                "kind": "gate_stage",
                "stage": stage,
                "wall_s": time.perf_counter() - started,
                "outcome": "failed",
            }
        )
        raise
    timings.append(
        {
            "kind": "gate_stage",
            "stage": stage,
            "wall_s": time.perf_counter() - started,
            "outcome": "ok",
        }
    )
    return value


def _declared_allowances(
    source: str,
) -> tuple[set[tuple[str, str]], set[tuple[tuple[str, str], tuple[str, str]]]]:
    pairs: set[tuple[str, str]] = set()
    element_pairs: set[tuple[tuple[str, str], tuple[str, str]]] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return pairs, element_pairs
    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Attribute)
            or node.func.attr != "allow_overlap"
            or len(node.args) < 2
        ):
            continue
        try:
            left = ast.literal_eval(node.args[0])
            right = ast.literal_eval(node.args[1])
            keywords = {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in node.keywords
                if keyword.arg is not None
            }
        except (ValueError, SyntaxError):
            continue
        if not isinstance(left, str) or not isinstance(right, str):
            continue
        pair = _pair_key(left, right)
        pairs.add(pair)
        elem_a = keywords.get("elem_a")
        elem_b = keywords.get("elem_b")
        if isinstance(elem_a, str) and isinstance(elem_b, str):
            element_pairs.add((pair, tuple(sorted((f"{left}:{elem_a}", f"{right}:{elem_b}")))))
    return pairs, element_pairs


def resolve_gate_map(
    blueprint: dict[str, Any] | None,
    model: object,
) -> tuple[dict[str, list[str]], list[dict[str, str]], list[dict[str, Any]]]:
    moving_by_child = {
        str(joint.child): str(joint.name)
        for joint in _moving_joints(model)
        if isinstance(getattr(joint, "child", None), str)
    }
    gate_map: dict[str, list[str]] = {}
    unresolved: list[dict[str, str]] = []
    topology_issues: list[dict[str, Any]] = []
    for part in (blueprint or {}).get("parts", []) if isinstance(blueprint, dict) else []:
        if not isinstance(part, dict) or not isinstance(part.get("motion_gate"), dict):
            continue
        part_name = part.get("name")
        if not isinstance(part_name, str):
            continue
        target_joint = moving_by_child.get(part_name)
        if target_joint is None:
            unresolved.append(
                {"part": part_name, "role": "gated", "detail": "no movable joint in built model"}
            )
            continue
        gate_joints: list[str] = []
        for gate_part in part["motion_gate"].get("gates") or []:
            gate_joint = moving_by_child.get(str(gate_part))
            if gate_joint is None:
                unresolved.append(
                    {
                        "part": str(gate_part),
                        "role": "gate",
                        "detail": f"gate of '{part_name}' has no movable joint in built model",
                    }
                )
            else:
                if gate_joint == target_joint:
                    topology_issues.append(
                        {
                            "state": "unsupported_gate_topology",
                            "target_joint": target_joint,
                            "chain": [target_joint, gate_joint],
                            "detail": "target joint cannot gate itself",
                        }
                    )
                else:
                    gate_joints.append(gate_joint)
        if gate_joints:
            gate_map[target_joint] = list(dict.fromkeys(gate_joints))
    for target_joint, gate_joints in sorted(gate_map.items()):
        nested = [gate_joint for gate_joint in gate_joints if gate_joint in gate_map]
        if nested:
            topology_issues.append(
                {
                    "state": "unsupported_gate_topology",
                    "target_joint": target_joint,
                    "chain": [target_joint, nested[0], *gate_map[nested[0]]],
                    "detail": "recursive gate witness composition is unsupported",
                }
            )
    return gate_map, unresolved, topology_issues


def motion_request_key(
    manifest: ResolvedCollisionManifest,
    *,
    gate_identity: dict[str, list[str]],
    policy: PosePlanPolicy,
    seed: int,
) -> str:
    payload = {
        "resolution_request_key": manifest.resolution_request_key,
        "gate_map": {name: sorted(values) for name, values in sorted(gate_identity.items())},
        "engine_version": MOTION_SCAN_ALGORITHM_VERSION,
        "pose_plan_policy_sha256": policy.digest,
        "depth_policy": DEPTH_POLICY_VERSION,
        "detector_policy_sha256": detector_policy_digest(),
        "seed": int(seed),
    }
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def blueprint_gate_identity(blueprint: dict[str, Any] | None) -> dict[str, list[str]]:
    identity: dict[str, list[str]] = {}
    for part in (blueprint or {}).get("parts", []) if isinstance(blueprint, dict) else []:
        if not isinstance(part, dict) or not isinstance(part.get("motion_gate"), dict):
            continue
        name = part.get("name")
        gates = sorted(str(value) for value in part["motion_gate"].get("gates") or [])
        if isinstance(name, str) and gates:
            identity[name] = gates
    return identity


def motion_evidence_key(request_key: str, resolved_manifest_digest: str) -> str:
    return hashlib.sha256(
        canonical_bytes(
            {
                "motion_request_key": request_key,
                "resolved_manifest_digest": resolved_manifest_digest,
            }
        )
    ).hexdigest()


def _contact_rows_by_pose(contacts: ContactTable) -> dict[int, list[Any]]:
    rows: dict[int, list[Any]] = {}
    for row in contacts.rows:
        rows.setdefault(row.pose_id, []).append(row)
    return rows


def _rest_contact_pairs(poses: PoseTable, contacts: ContactTable) -> set[tuple[str, str]]:
    pose_ids = list(poses.metadata.get("selfcol_pose_ids") or [])
    rest_id = pose_ids[0] if pose_ids else 0
    return {
        _pair_key(row.part_a, row.part_b)
        for row in contacts.rows_for_pose(rest_id)
        if row.colliding or row.min_distance <= 1e-3
    }


def _rest_contact_elements(poses: PoseTable, contacts: ContactTable) -> set[tuple[str, str]]:
    pose_ids = list(poses.metadata.get("selfcol_pose_ids") or [])
    rest_id = pose_ids[0] if pose_ids else 0
    return {
        tuple(sorted((str(row.source_element_a), str(row.source_element_b))))
        for row in contacts.rows_for_pose(rest_id)
        if row.colliding or row.min_distance <= 1e-3
    }


def _refine_pose_table(
    world: Any,
    poses: PoseTable,
    contacts: ContactTable,
    model: object,
    policy: PosePlanPolicy,
    memo: RestMemo | None = None,
) -> ContactTable:
    names, value_map = _joint_value_grid(model, per_joint=policy.selfcol_per_joint)
    if not names:
        return contacts
    rest = {name: 0.0 for name in names}
    rest_pairs = _rest_contact_pairs(poses, contacts)
    rows_by_pose = _contact_rows_by_pose(contacts)

    def pose_id_for(name: str, value: float) -> int:
        qpos = dict(rest)
        qpos[name] = value
        return poses.add(
            qpos, tags=("selfcol_refine",), provenance=(f"refine-seed:{name}:{value!r}",)
        )

    def near(pose_id: int) -> bool:
        return any(
            _pair_key(row.part_a, row.part_b) not in rest_pairs
            and (row.colliding or row.min_distance <= policy.refine_proximity_tol)
            for row in rows_by_pose.get(pose_id, [])
        )

    near_at: dict[str, dict[float, bool]] = {name: {} for name in names}
    seed_count = 0
    for name in names:
        for value in value_map[name]:
            seed_count += 1
            near_at[name][value] = near(pose_id_for(name, value))
    budget_left = policy.refine_budget - seed_count
    active: list[tuple[str, float, float, bool, bool]] = []
    for name in names:
        values = value_map[name]
        for index in range(len(values) - 1):
            lower, upper = values[index], values[index + 1]
            lower_near, upper_near = near_at[name][lower], near_at[name][upper]
            if lower_near or upper_near:
                active.append((name, lower, upper, lower_near, upper_near))
    refined_ids: list[int] = []
    depth = 0
    while active and depth < policy.refine_max_depth:
        requested: list[tuple[int, float]] = []
        before_count = len(poses.rows)
        for interval_index, (name, lower, upper, _left, _right) in enumerate(active):
            for fraction in (0.25, 0.5, 0.75):
                value = lower + (upper - lower) * fraction
                qpos = dict(rest)
                qpos[name] = value
                pose_id = poses.add(
                    qpos,
                    tags=("selfcol", "selfcol_refine"),
                    provenance=(f"refine:{depth}:{interval_index}:{fraction}",),
                )
                requested.append((interval_index, value))
                refined_ids.append(pose_id)
        if len(requested) > budget_left:
            break
        budget_left -= len(requested)
        new_ids = list(range(before_count, len(poses.rows)))
        if new_ids:
            increment = sweep_contacts(
                world,
                poses,
                broad_phase_margin=policy.broad_phase_margin,
                pose_ids=new_ids,
                memo=memo,
            )
            contacts = merge_contact_tables(contacts, increment)
            rows_by_pose = _contact_rows_by_pose(contacts)
        interior: dict[int, list[tuple[float, bool]]] = {}
        for interval_index, value in requested:
            pose_id = pose_id_for(active[interval_index][0], value)
            interior.setdefault(interval_index, []).append((value, near(pose_id)))
        next_active: list[tuple[str, float, float, bool, bool]] = []
        for interval_index, (name, lower, upper, lower_near, upper_near) in enumerate(active):
            points = [
                (lower, lower_near),
                *sorted(interior.get(interval_index, [])),
                (upper, upper_near),
            ]
            for index in range(len(points) - 1):
                (left, left_near), (right, right_near) = points[index], points[index + 1]
                if (left_near or right_near) and right - left > 0.01:
                    next_active.append((name, left, right, left_near, right_near))
        active = next_active
        depth += 1
    poses.metadata["selfcol_refine_pose_ids"] = list(dict.fromkeys(refined_ids))
    return contacts


def _augment_gate_witnesses(
    world: Any,
    poses: PoseTable,
    contacts: ContactTable,
    model: object,
    gate_map: dict[str, list[str]],
    declared_pairs: set[tuple[str, str]],
    policy: PosePlanPolicy,
    memo: RestMemo | None = None,
) -> ContactTable:
    if not gate_map:
        return contacts
    moving = {joint.name: joint for joint in _moving_joints(model)}
    gate_combos = (poses.metadata.get("reach") or {}).get("gate_combos") or {}
    gated_specs = [
        (
            target,
            moving_subtree(model, moving[target]),
            [dict(combo) for combo in gate_combos.get(target, [])],
        )
        for target, gates in gate_map.items()
        if target in moving and any(name in moving for name in gates) and gate_combos.get(target)
    ]
    rest_elements = _rest_contact_elements(poses, contacts)
    selfcol_ids = set(poses.metadata.get("selfcol_pose_ids") or []) | set(
        poses.metadata.get("selfcol_refine_pose_ids") or []
    )
    rows = [
        row
        for row in contacts.rows
        if row.pose_id in selfcol_ids
        and row.colliding
        and tuple(sorted((str(row.source_element_a), str(row.source_element_b))))
        not in rest_elements
        and _pair_key(row.part_a, row.part_b) not in declared_pairs
        and all(value > default_overlap_tol_from_env() for value in row.overlap_depth)
        and row.overlap_volume > default_overlap_volume_tol_from_env()
    ]
    witness_map: dict[str, list[dict[str, Any]]] = {}
    before_count = len(poses.rows)
    for row in rows:
        pair = _pair_key(row.part_a, row.part_b)
        key = f"{row.pose_id}|{pair[0]}|{pair[1]}"
        source_pose = poses.rows[row.pose_id].qpos
        for target_joint, subtree, combos in gated_specs:
            if target_joint not in source_pose or (
                pair[0] not in subtree and pair[1] not in subtree
            ):
                continue
            for combo_index, combo in enumerate(combos):
                pose_id = poses.add(
                    {**source_pose, **combo},
                    tags=("gate_grid",),
                    provenance=(f"gate-witness:{key}:{target_joint}:{combo_index}",),
                )
                witness_map.setdefault(key, []).append(
                    {
                        "target_joint": target_joint,
                        "combo_index": combo_index,
                        "pose_id": pose_id,
                    }
                )
    poses.metadata["gate_witnesses"] = witness_map
    new_ids = list(range(before_count, len(poses.rows)))
    if not new_ids:
        return contacts
    return merge_contact_tables(
        contacts,
        sweep_contacts(
            world,
            poses,
            broad_phase_margin=policy.broad_phase_margin,
            pose_ids=new_ids,
            memo=memo,
        ),
    )


def _build_depth_table(
    world: Any,
    poses: PoseTable,
    contacts: ContactTable,
) -> dict[tuple[int, tuple[str, str]], DepthResult]:
    reach_ids = set((poses.metadata.get("reach") or {}).get("search_pose_ids") or [])
    gate_ids = {
        int(record["pose_id"])
        for records in (poses.metadata.get("gate_witnesses") or {}).values()
        for record in records
        if isinstance(record, dict) and isinstance(record.get("pose_id"), int)
    }
    candidates = {
        (row.pose_id, _pair_key(row.part_a, row.part_b))
        for row in contacts.rows
        if row.colliding and row.pose_id in reach_ids | gate_ids
    }
    result: dict[tuple[int, tuple[str, str]], DepthResult] = {}
    by_pose: dict[int, list[tuple[str, str]]] = {}
    for pose_id, pair in sorted(candidates):
        by_pose.setdefault(pose_id, []).append(pair)
    for pose_id, pairs in by_pose.items():
        qpos = poses.rows[pose_id].qpos
        world.set_pose(qpos)
        for pair in pairs:
            result[(pose_id, pair)] = evaluate_depth(
                world,
                qpos=qpos,
                part_pair=pair,
                pose_is_set=True,
            )
    return result


def _exact_ram_findings(
    raw_model: object,
    poses: PoseTable,
    contacts: ContactTable,
    *,
    asset_root: Path,
    declared_pairs: set[tuple[str, str]],
    declared_element_pairs: set[tuple[tuple[str, str], tuple[str, str]]],
    policy: PosePlanPolicy,
    pose_ids: list[int] | None = None,
    execution: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    resolved_pose_ids = list(
        dict.fromkeys(pose_ids or list(poses.metadata.get("selfcol_pose_ids") or []))
    )
    pose_values = [poses.rows[pose_id].qpos for pose_id in resolved_pose_ids]
    local_to_global = {
        tuple(sorted((str(name), float(value)) for name, value in pose.items())): pose_id
        for pose_id, pose in zip(resolved_pose_ids, pose_values, strict=True)
    }
    ram_pose_ids = set(local_to_global.values())
    reachable_pairs = {
        _pair_key(row.part_a, row.part_b)
        for row in contacts.rows
        if row.colliding and row.pose_id in ram_pose_ids
    }
    scan = prepare_connection_ram_scan(
        raw_model,
        asset_root=asset_root,
        declared_pairs=declared_pairs,
        declared_elem_pairs=declared_element_pairs,
        per_joint=policy.selfcol_per_joint,
        random_fill=policy.selfcol_random_fill,
        seed=policy.seed,
        pose_values=pose_values,
        link_pairs=reachable_pairs,
    )
    n_shards = 1
    findings = []
    if scan is not None:
        records, n_shards = map_pose_shards(list(range(len(scan.poses))), scan.scan)
        findings = scan.findings(records)
    if execution is not None:
        execution["n_shards"] = n_shards
    result: list[dict[str, Any]] = []
    for finding in findings:
        pose = {str(name): float(value) for name, value in finding.worst_pose.items()}
        pose_id = local_to_global.get(tuple(sorted(pose.items())))
        pair = _pair_key(finding.link_a, finding.link_b)
        if pose_id is None or not any(
            row.colliding and _pair_key(row.part_a, row.part_b) == pair
            for row in contacts.rows_for_pose(pose_id)
        ):
            continue
        result.append(
            {
                "pair": [finding.link_a, finding.link_b],
                "elements": list(finding.offending_elements),
                "worst_pose": {name: round(value, 5) for name, value in pose.items()},
                "penetration_volume": round(finding.motion_penetration_volume, 8),
                "penetration_span": round(finding.penetration_span, 5),
                "blocking": finding.blocking,
                "note": "a joint connection rams across its motion",
                "oracle_pose_ids": [] if pose_id is None else [pose_id],
                "sample_provenance": list(poses.rows[pose_id].provenance),
            }
        )
    return result


def _ram_exact_pose_ids(
    poses: PoseTable,
    reach_resolution: ReachResolution | None,
) -> list[int]:
    pose_ids = [int(value) for value in poses.metadata.get("selfcol_pose_ids") or []]
    if reach_resolution is None:
        return list(dict.fromkeys(pose_ids))
    gated_targets = set(reach_resolution.selected_gate_combo_by_target)
    pose_ids.extend(
        pose_id
        for pose_id, owner in reach_resolution.owner_joint_by_pose_id.items()
        if owner in gated_targets
    )
    return list(dict.fromkeys(pose_ids))


def _oracle_pose_table(evidence: dict[str, Any], poses: PoseTable) -> dict[str, dict[str, float]]:
    pose_ids: set[int] = set()
    for findings in evidence.values():
        if not isinstance(findings, list):
            continue
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            pose_ids.update(
                int(value)
                for value in finding.get("oracle_pose_ids", [])
                if isinstance(value, int) and 0 <= value < len(poses.rows)
            )
    return {str(pose_id): dict(poses.rows[pose_id].qpos) for pose_id in sorted(pose_ids)}


def build_scan_evidence(
    model_path: Path,
    manifest: ResolvedCollisionManifest,
    *,
    source: str,
    blueprint: dict[str, Any] | None = None,
    sdk_package: str = "sdk",
    seed: int = 0,
    policy: PosePlanPolicy | None = None,
) -> dict[str, Any]:
    resolved_policy = policy or PosePlanPolicy(seed=seed)
    timings: list[dict[str, Any]] = []
    started = time.perf_counter()

    def build_world_payload() -> tuple[Any, Any]:
        resolved_geometry = resolve_collision_geometry(
            Path(model_path), sdk_package=sdk_package, manifest=manifest
        )
        collision_world = build_collision_world(
            resolved_geometry.resolved_model,
            manifest,
            asset_root=manifest.root,
        )
        return resolved_geometry, collision_world

    resolved, world = _timed(timings, "world_build", build_world_payload)
    gate_map, unresolved, gate_topology_issues = resolve_gate_map(blueprint, resolved.raw_model)
    declared_pairs, declared_element_pairs = _declared_allowances(source)
    poses = _timed(
        timings,
        "pose_plan",
        lambda: build_pose_table(
            resolved.resolved_model, gate_map=gate_map, policy=resolved_policy
        ),
    )
    rest_memo = RestMemo()
    contacts = _timed(
        timings,
        "sweep",
        lambda: sweep_contacts_sharded(
            world,
            poses,
            broad_phase_margin=resolved_policy.broad_phase_margin,
            memo=rest_memo,
        ),
    )
    contacts = _timed(
        timings,
        "sweep_refine",
        lambda: _refine_pose_table(
            world, poses, contacts, resolved.resolved_model, resolved_policy, memo=rest_memo
        ),
    )
    contacts = _timed(
        timings,
        "sweep_gate",
        lambda: _augment_gate_witnesses(
            world,
            poses,
            contacts,
            resolved.resolved_model,
            gate_map,
            declared_pairs,
            resolved_policy,
            memo=rest_memo,
        ),
    )
    depths = _timed(timings, "depth", lambda: _build_depth_table(world, poses, contacts))
    precomputed_reach: list[dict[str, Any]] | None = None
    precomputed_resolution: ReachResolution | None = None
    try:
        precomputed_reach, precomputed_resolution = _timed(
            timings,
            "reach_policy",
            lambda: resolve_reach_policy(
                resolved.resolved_model,
                manifest,
                poses,
                contacts,
                depths,
                declared_pairs=declared_pairs,
                gate_topology_issues=gate_topology_issues,
                unresolved_gates=unresolved,
            ),
        )
    except Exception:  # noqa: BLE001 - run_queries records the scoped detector failure
        precomputed_reach = None
        precomputed_resolution = None
    ram_findings: list[dict[str, Any]] | None = None
    ram_error: str | None = None
    ram_execution: dict[str, Any] = {}
    try:
        ram_findings = _timed(
            timings,
            "ram_exact",
            lambda: _exact_ram_findings(
                resolved.raw_model,
                poses,
                contacts,
                asset_root=manifest.root,
                declared_pairs=declared_pairs,
                declared_element_pairs=declared_element_pairs,
                policy=resolved_policy,
                pose_ids=_ram_exact_pose_ids(poses, precomputed_resolution),
                execution=ram_execution,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - L4 exposes the scoped detector failure
        ram_error = f"{type(exc).__name__}: {exc}"
    query_bundle: QueryBundle = run_queries(
        resolved.resolved_model,
        manifest,
        poses,
        contacts,
        depths,
        declared_pairs=declared_pairs,
        declared_element_pairs=declared_element_pairs,
        precomputed_ram_findings=ram_findings,
        precomputed_ram_error=ram_error,
        precomputed_driver_reach_findings=precomputed_reach,
        precomputed_reach_resolution=precomputed_resolution,
        gate_topology_issues=gate_topology_issues,
        unresolved_gates=unresolved,
    )
    timings.extend(query_bundle.timings)
    query_bundle.evidence["oracle_pose_table"] = _oracle_pose_table(query_bundle.evidence, poses)
    reach_metadata = poses.metadata.get("reach")
    movers = reach_metadata.get("movers") if isinstance(reach_metadata, dict) else {}
    query_bundle.evidence["joint_movers"] = {
        str(name): [str(part) for part in parts]
        for name, parts in movers.items()
        if isinstance(parts, (list, tuple))
    }
    request_key = motion_request_key(
        manifest,
        gate_identity=blueprint_gate_identity(blueprint),
        policy=resolved_policy,
        seed=seed,
    )
    evidence_key = motion_evidence_key(request_key, manifest.resolved_manifest_digest)
    evidence = {
        **query_bundle.evidence,
        "access_gate_unresolved": unresolved,
        "motion_execution_status": {"status": "ok"},
        "required_detector_status": query_bundle.required_detector_status,
        "motion_identity": {
            "schema_version": MOTION_EVIDENCE_SCHEMA_VERSION,
            "motion_request_key": request_key,
            "motion_evidence_key": evidence_key,
            "resolved_manifest_digest": manifest.resolved_manifest_digest,
            "simulator_collision_digest": manifest.simulator_collision_digest,
            "resolution_request_key": manifest.resolution_request_key,
            "requested_collision_mode": "full-coacd",
            "actual_collision_mode": "full-coacd",
            "fallback_policy": None,
            "fallback_reason": None,
            "cache_hit": False,
        },
    }
    worker_wall = time.perf_counter() - started
    timed_wall = sum(float(row["wall_s"]) for row in timings)
    peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    return {
        "schema_version": MOTION_EVIDENCE_SCHEMA_VERSION,
        "motion_request_key": request_key,
        "motion_evidence_key": evidence_key,
        "resolved_manifest_digest": manifest.resolved_manifest_digest,
        "simulator_collision_digest": manifest.simulator_collision_digest,
        "evidence": evidence,
        "stage_timings": timings,
        "stats": {
            "n_elements": len(manifest.exact_legality.entries),
            "n_pieces": len(manifest.colliders),
            "n_poses": len(poses.rows),
            "n_pairs_broad": contacts.stats.n_pairs_broad,
            "n_pairs_narrow": contacts.stats.n_pairs_narrow,
            "n_pairs_memoized": contacts.stats.n_pairs_memoized,
            "n_shards": contacts.stats.n_shards,
            "n_ram_shards": ram_execution.get("n_shards", 0),
            "n_depth_pairs": len(depths),
            "peak_rss_mb": peak_rss,
            "worker_wall_s": worker_wall,
            "timing_coverage": timed_wall / worker_wall if worker_wall > 0.0 else 1.0,
        },
        "policy": asdict(resolved_policy),
    }


def write_envelope_atomic(path: Path, envelope: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(
        f".{path.name}.{hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:12]}.tmp"
    )
    try:
        temp.write_text(json.dumps(envelope, sort_keys=True), encoding="utf-8")
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()
