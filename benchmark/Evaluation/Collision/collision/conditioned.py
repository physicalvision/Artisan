from __future__ import annotations

from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any

from collision.backend import FCLBackend, NarrowPhaseResult
from collision.cache import GTCache
from collision.mask import (
    RolePartition,
    adjacent_body_pairs,
    apply_collision_mask,
    build_role_partition,
    parse_body_pair,
    project_gt_collision_mask,
)
from collision.matching import match_models
from collision.metrics import (
    EvidenceState,
    aggregate_cell,
    aggregate_tuple,
    classify_collision,
)
from collision.model import EvalModel, load_eval_model
from collision.policy import BenchPolicy
from collision.sweep import JointSweep, SweepEngine


def _scan_unit(
    engine: SweepEngine, model: EvalModel, level: str, keys: tuple[str, ...]
) -> JointSweep:
    joints = tuple(model.joint_by_key[key] for key in keys)
    if level == "L1":
        return engine.scan_joint(joints[0])
    if level == "L2":
        return engine.scan_l2(joints[0], joints[1])
    return engine.scan_lmulti(joints)


def _cell_state(
    gen_results: dict[str, NarrowPhaseResult],
    gt_results: dict[str, NarrowPhaseResult],
    gt_reference: dict[str, NarrowPhaseResult],
    gen_model: EvalModel,
    gt_model: EvalModel,
    gen_partition: RolePartition,
    gt_partition: RolePartition,
    gen_epsilon: float,
    gt_epsilon: float,
) -> EvidenceState:
    adjacent = adjacent_body_pairs(gen_model)
    rows: dict[tuple[str, str, bool], list[EvidenceState]] = {}
    for key, result in gen_results.items():
        pair = parse_body_pair(key)
        roles = gen_partition.role_pair(pair)
        parent_child = pair in adjacent
        state = classify_collision(result, epsilon=gen_epsilon)
        if not parent_child:
            decision = project_gt_collision_mask(
                role_pair=roles,
                gt_results=gt_results,
                gt_partition=gt_partition,
                gt_model=gt_model,
                epsilon=gt_epsilon,
                gt_reference=gt_reference,
            )
            state = apply_collision_mask(state, decision)
        rows.setdefault((*roles, parent_child), []).append(state)
    return aggregate_cell(aggregate_tuple(states) for states in rows.values())


def evaluate_pair(
    gt_urdf: Path,
    gen_urdf: Path,
    *,
    policy: BenchPolicy | None = None,
    gt_cache_dir: Path | None = None,
    policy_name: str = "rev10",
    refresh: bool = False,
    levels: tuple[str, ...] = ("L1", "L2", "Lmulti"),
) -> dict[str, Any]:
    requested = tuple(dict.fromkeys(levels))
    if not requested or set(requested) - {"L1", "L2", "Lmulti"}:
        raise ValueError("Select at least one of L1, L2 or Lmulti")
    policy = policy or BenchPolicy()
    gt_model = load_eval_model(gt_urdf, policy)
    gen_model = load_eval_model(gen_urdf, policy)
    matches = match_models(gt_model, gen_model)
    gt_roles = {record.gt_key: f"role:{record.gt_key}" for record in matches.records}
    gen_roles = {record.gen_key: gt_roles[record.gt_key] for record in matches.records}
    gt_partition = build_role_partition(gt_model, gt_roles)
    gen_partition = build_role_partition(gen_model, gen_roles)
    gt_engine = SweepEngine(gt_model, FCLBackend(policy), policy)
    gen_engine = SweepEngine(gen_model, FCLBackend(policy), policy)
    gt_epsilon = policy.effective_epsilon(gt_engine.scene_diagonal)
    gen_epsilon = policy.effective_epsilon(gen_engine.scene_diagonal)
    cache = (
        GTCache(gt_cache_dir, gt_urdf, policy_name, refresh=refresh)
        if gt_cache_dir is not None
        else None
    )
    gt_rest = cache.rest(gt_engine.rest_table) if cache else gt_engine.rest_table()
    records = matches.records
    groups = {
        "L1": tuple((record,) for record in records),
        "L2": tuple(combinations(records, 2)),
        "Lmulti": (records,) if len(records) >= 3 else (),
    }
    results = {}
    for level in requested:
        violating = 0
        denominator = 0
        for group in groups[level]:
            gt_keys = tuple(record.gt_key for record in group)
            gen_keys = tuple(record.gen_key for record in group)

            def scan_gt() -> JointSweep:
                return _scan_unit(gt_engine, gt_model, level, gt_keys)

            gt_sweep = cache.sweep(level, gt_keys, scan_gt) if cache else scan_gt()
            gen_sweep = _scan_unit(gen_engine, gen_model, level, gen_keys)
            for gt_cell, gen_cell in zip(gt_sweep.cells, gen_sweep.cells, strict=True):
                state = _cell_state(
                    gen_cell.body_pairs,
                    {**gt_rest, **gt_cell.body_pairs},
                    gt_rest,
                    gen_model,
                    gt_model,
                    gen_partition,
                    gt_partition,
                    gen_epsilon,
                    gt_epsilon,
                )
                violating += state is EvidenceState.VIOLATING
                denominator += 1
        results[level] = {"violating": violating, "denominator": denominator}
    return {
        "status": "scored",
        "matching": {
            "matched_count": len(records),
            "records": [asdict(record) for record in records],
            "unmatched_gt": list(matches.unmatched_gt),
            "unmatched_gen": list(matches.unmatched_gen),
        },
        "levels": results,
    }
