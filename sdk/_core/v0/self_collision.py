
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from ._coupling import effective_driver_range, follower_bound_range
from .assets import resolve_asset_root
from .exact_collisions import compile_object_model_with_exact_collisions
from .geometry_qc import (
    FullCollisionPairEvaluation,
    UnsupportedPartFinding,
    _aabb_intersection_depth,
    _aabb_intersection_volume,
    _build_exact_collision_world,
    _find_collision_overlaps_fcl_for_poses,
    _pair_key,
    _root_part_names,
    _unsupported_component_findings,
    default_overlap_tol_from_env,
    default_overlap_volume_tol_from_env,
    default_penetration_depth_tol_from_env,
    evaluate_full_collision_pair_poses,
    find_geometry_overlaps_in_poses,
    strip_component_suffix,
    weld_group_of,
)
from .types import ArticulationType

logger = logging.getLogger(__name__)

_MOVING = {ArticulationType.REVOLUTE, ArticulationType.PRISMATIC, ArticulationType.CONTINUOUS}


def _moving_joints(model: object) -> list:
    out = []
    for j in getattr(model, "articulations", None) or []:
        if getattr(j, "mimic", None) is not None:
            continue
        if getattr(j, "articulation_type", None) in _MOVING and isinstance(
            getattr(j, "name", None), str
        ):
            out.append(j)
    return out


def moving_subtree(model: object, joint: object) -> set[str]:
    arts = list(getattr(model, "articulations", None) or [])
    jname = getattr(joint, "name", None)

    def descendants(start: str) -> set[str]:
        seen: set[str] = {start}
        stack = [start]
        while stack:
            cur = stack.pop()
            for art in arts:
                child = getattr(art, "child", None)
                if (
                    getattr(art, "parent", None) == cur
                    and isinstance(child, str)
                    and child not in seen
                ):
                    seen.add(child)
                    stack.append(child)
        return seen

    roots: set[str] = set()
    child = getattr(joint, "child", None)
    if isinstance(child, str):
        roots.add(child)
    for art in arts:
        m = getattr(art, "mimic", None)
        if m is not None and str(getattr(m, "joint", "")) == str(jname):
            fchild = getattr(art, "child", None)
            if isinstance(fchild, str):
                roots.add(fchild)

    out: set[str] = set()
    for r in roots:
        out |= descendants(r)
    return out


def _joint_range(joint: object) -> tuple[float, float]:
    atype = getattr(joint, "articulation_type", None)
    ml = getattr(joint, "motion_limits", None)
    lo = getattr(ml, "lower", None) if ml is not None else None
    hi = getattr(ml, "upper", None) if ml is not None else None
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and hi > lo:
        return float(lo), float(hi)
    if atype in (ArticulationType.CONTINUOUS, ArticulationType.REVOLUTE):
        return -math.pi, math.pi
    return 0.0, 0.0


_follower_bound_range = follower_bound_range


def _screen_range(model: object, joint: object) -> tuple[float, float]:
    eff = effective_driver_range(model, joint)
    return eff if eff is not None else _joint_range(joint)


def _joint_value_grid(
    model: object, *, per_joint: int = 7
) -> tuple[list[str], dict[str, list[float]]]:
    joints = _moving_joints(model)
    names = [j.name for j in joints]
    ranges = {j.name: _screen_range(model, j) for j in joints}

    def vals(n: str) -> list[float]:
        lo, hi = ranges[n]
        if hi <= lo:
            return [0.0]
        if per_joint < 2:
            return [lo, hi]
        return [lo + (hi - lo) * k / (per_joint - 1) for k in range(per_joint)]

    return names, {n: vals(n) for n in names}


def generate_screening_poses(
    model: object,
    *,
    per_joint: int = 7,
    pairwise: bool = True,
    random_fill: int = 200,
    max_poses: int = 1500,
    seed: int = 0,
) -> list[dict[str, float]]:
    names, valmap = _joint_value_grid(model, per_joint=per_joint)

    rest = {n: 0.0 for n in names}
    poses = [dict(rest)]
    seen = {tuple(rest[n] for n in names)}

    def add(p: dict[str, float]) -> bool:
        if len(poses) >= max_poses:
            return False
        key = tuple(round(p[n], 6) for n in names)
        if key not in seen:
            seen.add(key)
            poses.append(p)
        return True

    for n in names:
        for v in valmap[n]:
            p = dict(rest)
            p[n] = v
            add(p)
    cap2 = max(0, max_poses - random_fill)
    if pairwise:
        for i in range(len(names)):
            if len(poses) >= cap2:
                break
            for j in range(i + 1, len(names)):
                if len(poses) >= cap2:
                    break
                for vi in valmap[names[i]]:
                    for vj in valmap[names[j]]:
                        p = dict(rest)
                        p[names[i]] = vi
                        p[names[j]] = vj
                        add(p)
    rng = random.Random(seed)
    for _ in range(random_fill):
        add({n: rng.choice(valmap[n]) for n in names})
    return poses


def element_id(link: str, name: str) -> str:
    return f"{link}:{name}"


def _elem_id(link: str, name: Optional[str], idx: int) -> str:
    return element_id(link, name if name else f"#{idx}")


def _elem_key(ov) -> tuple[str, str]:
    a = _elem_id(ov.link_a, ov.elem_a_name, ov.elem_a)
    b = _elem_id(ov.link_b, ov.elem_b_name, ov.elem_b)
    return (a, b) if a <= b else (b, a)


def _elem_base_key(ek: tuple[str, str]) -> tuple[str, str]:
    a, b = strip_component_suffix(ek[0]), strip_component_suffix(ek[1])
    return (a, b) if a <= b else (b, a)


def _active_relative_drivers(
    pose: dict[str, float],
    pair: tuple[str, str],
    movers: dict[str, set[str]],
    *,
    epsilon: float = 1e-9,
) -> tuple[str, ...]:

    return tuple(
        sorted(
            name
            for name, subtree in movers.items()
            if abs(float(pose.get(name, 0.0))) > epsilon
            and ((pair[0] in subtree) != (pair[1] in subtree))
        )
    )


_RANK = {"CONDITIONAL": 0, "COORDINATION": 1, "SUSPICIOUS": 2, "DEFINITE": 3}


@dataclass
class PairFinding:
    link_a: str
    link_b: str
    classification: str
    declared: bool
    worst_pose: dict[str, float]
    excess: float
    offending_elements: tuple[str, str]
    note: str = ""
    coordination_joints: tuple[str, ...] = ()
    clear_pose: dict[str, float] | None = None
    gate_evaluation: GateWitnessEvaluation | None = None


@dataclass
class GateWitnessEvaluation:
    pose: dict[str, float]
    policy_id: str
    collision_mode: str
    penetration_depth_tol_m: float
    max_penetration_depth_m: float | None
    clear: bool
    axis: str
    elements: tuple[str, str] | None
    depth_method: str = "unknown"
    depth_signed: bool = False
    depth_degenerate: bool = False


@dataclass
class _GateClearResult:
    evaluation: GateWitnessEvaluation | None = None


@dataclass
class SelfCollisionReport:
    findings: list[PairFinding]
    n_poses: int
    collision_mode: str = "full-exact"
    fallback_reason: str | None = None

    @property
    def definite(self) -> list[PairFinding]:
        return [f for f in self.findings if f.classification == "DEFINITE"]

    @property
    def suspicious(self) -> list[PairFinding]:
        return [f for f in self.findings if f.classification == "SUSPICIOUS"]

    @property
    def coordination(self) -> list[PairFinding]:
        return [f for f in self.findings if f.classification == "COORDINATION"]

    @property
    def conditional(self) -> list[PairFinding]:
        return [f for f in self.findings if f.classification == "CONDITIONAL"]


def _pose_has_near_contact(
    model: object,
    poses: list[dict[str, float]],
    *,
    asset_root: Optional[Path],
    tau: float,
    rest_contact_pairs: set[tuple[str, str]],
    validate_model: bool,
) -> list[bool]:
    if not poses:
        return []
    overlaps = find_geometry_overlaps_in_poses(
        model,
        poses=poses,
        asset_root=asset_root,
        overlap_tol=0.0,
        overlap_volume_tol=0.0,
        allowed_pairs=None,
        validate_model=validate_model,
        contact_distance_tol=tau,
    )
    near = [False] * len(poses)
    for ov in overlaps:
        if _pair_key(ov.link_a, ov.link_b) in rest_contact_pairs:
            continue
        if 0 <= ov.pose_index < len(near):
            near[ov.pose_index] = True
    return near


def _refine_near_contact_poses(
    model: object,
    *,
    names: list[str],
    valmap: dict[str, list[float]],
    rest_contact_pairs: set[tuple[str, str]],
    asset_root: Optional[Path],
    tau: float,
    eps_theta: float,
    max_depth: int,
    budget: int,
    validate_model: bool,
) -> list[dict[str, float]]:
    if not names:
        return []
    rest = {n: 0.0 for n in names}

    def single(n: str, a: float) -> dict[str, float]:
        p = dict(rest)
        p[n] = a
        return p

    budget_left = budget
    refined: list[dict[str, float]] = []

    seed_poses: list[dict[str, float]] = []
    seed_prov: list[tuple[str, float]] = []
    for n in names:
        for v in valmap[n]:
            seed_poses.append(single(n, v))
            seed_prov.append((n, v))
    seed_near = _pose_has_near_contact(
        model,
        seed_poses,
        asset_root=asset_root,
        tau=tau,
        rest_contact_pairs=rest_contact_pairs,
        validate_model=validate_model,
    )
    budget_left -= len(seed_poses)
    near_at: dict[str, dict[float, bool]] = {n: {} for n in names}
    for (n, v), f in zip(seed_prov, seed_near):
        near_at[n][v] = f

    active: list[tuple[str, float, float, bool, bool]] = []
    for n in names:
        vs = valmap[n]
        for k in range(len(vs) - 1):
            lo, hi = vs[k], vs[k + 1]
            nlo, nhi = near_at[n].get(lo, False), near_at[n].get(hi, False)
            if nlo or nhi:
                active.append((n, lo, hi, nlo, nhi))

    depth = 0
    detection_truncated = False
    while active and depth < max_depth:
        sub_poses: list[dict[str, float]] = []
        sub_prov: list[tuple[int, float]] = []
        for ii, (n, lo, hi, _nlo, _nhi) in enumerate(active):
            for t in (0.25, 0.5, 0.75):
                a = lo + (hi - lo) * t
                sub_poses.append(single(n, a))
                sub_prov.append((ii, a))
        if len(sub_poses) > budget_left:
            if depth == 0:
                detection_truncated = True
            break
        budget_left -= len(sub_poses)
        refined.extend(sub_poses)
        sub_near = _pose_has_near_contact(
            model,
            sub_poses,
            asset_root=asset_root,
            tau=tau,
            rest_contact_pairs=rest_contact_pairs,
            validate_model=validate_model,
        )
        interior_by_iv: dict[int, list[tuple[float, bool]]] = {}
        for (ii, a), f in zip(sub_prov, sub_near):
            interior_by_iv.setdefault(ii, []).append((a, f))

        next_active: list[tuple[str, float, float, bool, bool]] = []
        for ii, (n, lo, hi, nlo, nhi) in enumerate(active):
            pts = [(lo, nlo)] + sorted(interior_by_iv.get(ii, [])) + [(hi, nhi)]
            for k in range(len(pts) - 1):
                (a, na), (b, nb) = pts[k], pts[k + 1]
                if (na or nb) and (b - a) > eps_theta:
                    next_active.append((n, a, b, na, nb))
        active = next_active
        depth += 1

    if detection_truncated:
        logger.warning(
            "self-collision refinement budget (%d poses) too small for the first "
            "sub-sampling pass over %d joint(s); raise refine_budget or a narrow "
            "motion-collision window may be missed",
            budget,
            len(names),
        )

    seen: set[tuple[float, ...]] = set()
    out: list[dict[str, float]] = []
    for p in refined:
        key = tuple(round(p[n], 9) for n in names)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _gate_value_grid(
    model: object, gate_joints: list, *, per_gate: int = 7
) -> list[dict[str, float]]:
    if not gate_joints:
        return [{}]
    n_gates = len(gate_joints)

    def samples(j: object) -> list[float]:
        lo, hi = _screen_range(model, j)
        if hi <= lo:
            return [0.0]
        if n_gates >= 3:
            rest = min(max(0.0, lo), hi)
            return sorted({lo, hi, rest})
        if per_gate < 2:
            return [lo, hi]
        return [lo + (hi - lo) * k / (per_gate - 1) for k in range(per_gate)]

    combos: list[dict[str, float]] = [{}]
    for j in gate_joints:
        vals = samples(j)
        combos = [{**c, j.name: v} for c in combos for v in vals]
    return combos


def _gates_for_pair(pair: tuple[str, str], gated_specs: list[tuple[set[str], list]]) -> list:
    gates: list = []
    seen: set[str] = set()
    for subtree, gate_objs in gated_specs:
        if pair[0] in subtree or pair[1] in subtree:
            for g in gate_objs:
                if g.name not in seen:
                    seen.add(g.name)
                    gates.append(g)
    return gates


def _coacd_screen_enabled() -> bool:
    return False


_MOTION_SCREEN_COLLISION_MODE_ATTR = "_artisan_motion_screen_collision_mode"
_MOTION_SCREEN_FALLBACK_REASON_ATTR = "_artisan_motion_screen_fallback_reason"
LEGACY_MOTION_FALLBACK_POLICY = "retired-exact-screen"


def motion_screen_collision_mode(compiled_model: object | None = None) -> str:
    actual = (
        getattr(compiled_model, _MOTION_SCREEN_COLLISION_MODE_ATTR, None)
        if compiled_model is not None
        else None
    )
    if actual in {"full-exact", "full-coacd"}:
        return str(actual)
    return "full-coacd" if _coacd_screen_enabled() else "full-exact"


def motion_screen_fallback_reason(compiled_model: object | None = None) -> str | None:
    value = (
        getattr(compiled_model, _MOTION_SCREEN_FALLBACK_REASON_ATTR, None)
        if compiled_model is not None
        else None
    )
    return str(value) if isinstance(value, str) and value else None


def compile_for_motion_screen(
    model: object, *, asset_root: Optional[Path] = None, validate: bool = False
) -> object:
    compiled = compile_object_model_with_exact_collisions(
        model, asset_root=asset_root, validate=validate
    )
    setattr(compiled, _MOTION_SCREEN_COLLISION_MODE_ATTR, "full-exact")
    if not _coacd_screen_enabled():
        return compiled
    try:
        import copy

        from .physics_export import _coacd_cache_dir, _coacd_part_collisions

        cache_dir = _coacd_cache_dir(None)
        clone = copy.copy(compiled)
        clone.parts = []
        for part in getattr(compiled, "parts", []) or []:
            pclone = copy.copy(part)
            pclone.collisions = list(getattr(part, "collisions", []) or [])
            _coacd_part_collisions(pclone, asset_root=asset_root, cache_dir=cache_dir)
            clone.parts.append(pclone)
        setattr(clone, _MOTION_SCREEN_COLLISION_MODE_ATTR, "full-coacd")
        return clone
    except Exception as exc:  # noqa: BLE001 — never let a CoACD failure break the screen
        setattr(
            compiled,
            _MOTION_SCREEN_FALLBACK_REASON_ATTR,
            f"coacd_compile_failed:{type(exc).__name__}",
        )
        return compiled


def find_self_collisions_over_motion(
    model: object,
    *,
    asset_root: Optional[Path] = None,
    declared_pairs: Optional[set[tuple[str, str]]] = None,
    declared_elem_pairs: Optional[set[tuple[tuple[str, str], tuple[str, str]]]] = None,
    gate_map: Optional[dict[str, list[str]]] = None,
    per_joint: int = 7,
    random_fill: int = 200,
    seed: int = 0,
    overlap_tol: Optional[float] = None,
    overlap_volume_tol: Optional[float] = None,
    connection_tol: float = 1e-3,
    validate_model: bool = False,
    refine: bool = True,
    refine_proximity_tol: float = 0.02,
    refine_angle_eps: float = 0.01,
    refine_max_depth: int = 8,
    refine_budget: int = 400,
) -> SelfCollisionReport:
    declared_pairs = {_pair_key(*p) for p in (declared_pairs or set())}
    declared_elem_pairs = declared_elem_pairs or set()
    otol = default_overlap_tol_from_env() if overlap_tol is None else overlap_tol
    vtol = (
        default_overlap_volume_tol_from_env() if overlap_volume_tol is None else overlap_volume_tol
    )

    gated_specs: list[tuple[set[str], list]] = []
    if gate_map:
        by_name = {j.name: j for j in _moving_joints(model)}
        for gated_name, gate_names in gate_map.items():
            gj = by_name.get(gated_name)
            if gj is None:
                continue
            gates = [by_name[g] for g in (gate_names or []) if g in by_name]
            if not gates:
                continue
            gated_specs.append((moving_subtree(model, gj), gates))

    screen_compiled = compile_for_motion_screen(model, asset_root=asset_root, validate=False)

    poses = generate_screening_poses(model, per_joint=per_joint, random_fill=random_fill, seed=seed)

    rest_overlaps = find_geometry_overlaps_in_poses(
        model,
        poses=[poses[0]],
        asset_root=asset_root,
        overlap_tol=0.0,
        overlap_volume_tol=0.0,
        allowed_pairs=None,
        validate_model=False,
        contact_distance_tol=connection_tol,
        compiled_model=screen_compiled,
    )
    rest_elem_set: set[tuple[str, str]] = set()
    rest_contact_pairs: set[tuple[str, str]] = set()
    for ov in rest_overlaps:
        rest_elem_set.add(_elem_key(ov))
        rest_contact_pairs.add(_pair_key(ov.link_a, ov.link_b))

    if refine:
        names, valmap = _joint_value_grid(model, per_joint=per_joint)
        poses = poses + _refine_near_contact_poses(
            model,
            names=names,
            valmap=valmap,
            rest_contact_pairs=rest_contact_pairs,
            asset_root=asset_root,
            tau=refine_proximity_tol,
            eps_theta=refine_angle_eps,
            max_depth=refine_max_depth,
            budget=refine_budget,
            validate_model=False,
        )

    overlaps = find_geometry_overlaps_in_poses(
        model,
        poses=poses,
        asset_root=asset_root,
        overlap_tol=otol,
        overlap_volume_tol=vtol,
        allowed_pairs=None,
        validate_model=validate_model,
        compiled_model=screen_compiled,
    )

    subtree_of = {j.name: moving_subtree(model, j) for j in _moving_joints(model)}
    gate_results_by_idx: dict[int, _GateClearResult] = {}
    if gated_specs:
        groups: dict[tuple, dict] = {}
        for idx, ov in enumerate(overlaps):
            if ov.pose_index == 0:
                continue
            ek = _elem_key(ov)
            if ek in rest_elem_set:
                continue
            pair = _pair_key(ov.link_a, ov.link_b)
            if (pair, _elem_base_key(ek)) in declared_elem_pairs:
                continue
            if pair in declared_pairs:
                continue
            gates = _gates_for_pair(pair, gated_specs)
            if not gates:
                continue
            gate_names = {g.name for g in gates}
            key_joints = sorted(
                n
                for n, sub in subtree_of.items()
                if (pair[0] in sub or pair[1] in sub) and n not in gate_names
            )
            restricted = tuple((n, round(ov.pose.get(n, 0.0), 12)) for n in key_joints)
            gkey = (pair, restricted)
            grp = groups.get(gkey)
            if grp is None:
                groups[gkey] = grp = {
                    "combos": _gate_value_grid(model, gates),
                    "rep_pose": ov.pose,
                    "members": [],
                }
            grp["members"].append((idx, ek))

        big_pair_poses: list[tuple[tuple[str, str], dict[str, float]]] = []
        spans: dict[tuple, tuple[int, int]] = {}
        for gkey, grp in groups.items():
            start = len(big_pair_poses)
            combos = grp["combos"]
            for combo in combos:
                big_pair_poses.append((gkey[0], {**grp["rep_pose"], **combo}))
            spans[gkey] = (start, len(combos))
        gate_evaluations: list[FullCollisionPairEvaluation] = []
        if big_pair_poses:
            gate_evaluations = evaluate_full_collision_pair_poses(
                screen_compiled,
                asset_root=asset_root,
                source_model=model,
                pair_poses=big_pair_poses,
                penetration_depth_tol_m=default_penetration_depth_tol_from_env(),
            )
        collision_mode = motion_screen_collision_mode(screen_compiled)
        for gkey, grp in groups.items():
            start, ncombos = spans[gkey]
            candidates = gate_evaluations[start : start + ncombos]
            selected = next((candidate for candidate in candidates if candidate.clear), None)
            if selected is None and candidates:
                selected = min(
                    enumerate(candidates),
                    key=lambda item: (
                        (
                            item[1].max_penetration_depth_m
                            if item[1].max_penetration_depth_m is not None
                            else float("inf")
                        ),
                        -item[0],
                    ),
                )[1]
            result = _GateClearResult(
                evaluation=(
                    GateWitnessEvaluation(
                        pose=dict(selected.pose),
                        policy_id=selected.policy_id,
                        collision_mode=collision_mode,
                        penetration_depth_tol_m=selected.penetration_depth_tol_m,
                        max_penetration_depth_m=selected.max_penetration_depth_m,
                        clear=selected.clear,
                        axis=selected.axis,
                        elements=selected.elements,
                        depth_method=selected.depth_method,
                        depth_signed=selected.depth_signed,
                        depth_degenerate=selected.depth_degenerate,
                    )
                    if selected is not None
                    else None
                )
            )
            for idx, _ek in grp["members"]:
                gate_results_by_idx[idx] = result

    worst: dict[tuple[str, str], PairFinding] = {}
    for idx, ov in enumerate(overlaps):
        if ov.pose_index == 0:
            continue
        ek = _elem_key(ov)
        if ek in rest_elem_set:
            continue
        pair = _pair_key(ov.link_a, ov.link_b)
        if (pair, _elem_base_key(ek)) in declared_elem_pairs:
            continue
        declared = pair in declared_pairs
        active_drivers = _active_relative_drivers(ov.pose, pair, subtree_of)
        cls = (
            "SUSPICIOUS"
            if declared
            else ("COORDINATION" if len(active_drivers) >= 2 else "DEFINITE")
        )
        mag = float(ov.overlap_volume)
        cur = worst.get(pair)
        gate_result = gate_results_by_idx.get(idx)
        gate_evaluation = gate_result.evaluation if gate_result is not None else None
        gate_clear_pose = (
            dict(gate_evaluation.pose)
            if gate_evaluation is not None and gate_evaluation.clear
            else None
        )
        finding_pose = dict(ov.pose)
        if cls == "DEFINITE" and gate_evaluation is not None and not gate_evaluation.clear:
            finding_pose = dict(gate_evaluation.pose)
            if gate_evaluation.elements is not None:
                ek = (
                    f"{pair[0]}:{gate_evaluation.elements[0]}",
                    f"{pair[1]}:{gate_evaluation.elements[1]}",
                )
        if cls in {"DEFINITE", "COORDINATION"} and gated_specs and gate_clear_pose is not None:
            cls = "CONDITIONAL"
        cand = PairFinding(
            link_a=pair[0],
            link_b=pair[1],
            classification=cls,
            declared=declared,
            worst_pose=finding_pose,
            excess=mag,
            offending_elements=ek,
            note=(
                "gate-clearable phantom"
                if cls == "CONDITIONAL"
                else (
                    "independent-driver coordination contact"
                    if cls == "COORDINATION"
                    else (
                        "gate-open witness still collides"
                        if gate_evaluation is not None and not gate_evaluation.clear
                        else "new element-pair under motion"
                    )
                )
            ),
            coordination_joints=active_drivers if cls == "COORDINATION" else (),
            clear_pose=gate_clear_pose if cls == "CONDITIONAL" else None,
            gate_evaluation=gate_evaluation,
        )
        if cur is None or (_RANK[cls], mag) > (_RANK[cur.classification], cur.excess):
            worst[pair] = cand

    findings = sorted(
        worst.values(),
        key=lambda f: (-_RANK[f.classification], -f.excess),
    )
    return SelfCollisionReport(
        findings=findings,
        n_poses=len(poses),
        collision_mode=motion_screen_collision_mode(screen_compiled),
        fallback_reason=motion_screen_fallback_reason(screen_compiled),
    )


@dataclass
class ConnectionRamFinding:

    link_a: str
    link_b: str
    motion_penetration_volume: float
    penetration_span: float
    worst_pose: dict[str, float]
    offending_elements: tuple[str, str]
    blocking: bool


_RAM_ADVISORY_VOLUME_TOL = 5e-8
_RAM_BLOCK_VOLUME_TOL = 1e-6


def _ram_pair_filter(
    model: object, link_pairs: Optional[set[tuple[str, str]]]
) -> Callable[[tuple[str, str]], bool]:
    moving_parts: set[str] = set()
    for j in _moving_joints(model):
        moving_parts |= moving_subtree(model, j)
    welds = weld_group_of(model)

    def admissible(pair: tuple[str, str]) -> bool:
        if link_pairs is not None and pair not in link_pairs:
            return False
        if moving_parts and pair[0] not in moving_parts and pair[1] not in moving_parts:
            return False
        weld = welds.get(pair[0])
        return weld is None or weld != welds.get(pair[1])

    return admissible


@dataclass
class ConnectionRamScan:

    poses: list[dict[str, float]]
    world: object
    rows: object
    cols: object
    active: tuple[int, ...]
    element_ids: tuple[str, ...]
    declared_pairs: set[tuple[str, str]]
    declared_elem_pairs: set[tuple[tuple[str, str], tuple[str, str]]]
    overlap_tol: float
    overlap_volume_tol: float
    advisory_volume_tol: float
    block_volume_tol: float

    def scan(
        self, positions: Sequence[int]
    ) -> list[list[tuple[int, int, tuple[float, float, float], float]]]:
        import fcl

        world = self.world
        world_aabbs = world.world_aabbs
        objects = world.objects
        overlap_tol = self.overlap_tol
        request = fcl.CollisionRequest()
        out: list[list[tuple[int, int, tuple[float, float, float], float]]] = []
        for position in positions:
            world.set_pose_of(self.poses[position], self.active)
            lefts, rights = world.penetration_candidates_among(
                self.rows, self.cols, overlap_tol=overlap_tol
            )
            hits: list[tuple[int, int, tuple[float, float, float], float]] = []
            for left, right in zip(lefts.tolist(), rights.tolist()):
                aabb_a = world_aabbs[left]
                aabb_b = world_aabbs[right]
                depth = _aabb_intersection_depth(aabb_a, aabb_b)
                volume = _aabb_intersection_volume(aabb_a, aabb_b)
                if not (
                    depth[0] > overlap_tol and depth[1] > overlap_tol and depth[2] > overlap_tol
                ):
                    continue
                if volume <= self.overlap_volume_tol:
                    continue
                if float(volume) < self.advisory_volume_tol:
                    continue
                result = fcl.CollisionResult()
                if int(fcl.collide(objects[left], objects[right], request, result)) <= 0:
                    continue
                hits.append((left, right, depth, volume))
            out.append(hits)
        return out

    def findings(
        self,
        records: Sequence[Sequence[tuple[int, int, tuple[float, float, float], float]]],
    ) -> list[ConnectionRamFinding]:
        part_of = self.world.part_of
        worst: dict[tuple[str, str], ConnectionRamFinding] = {}
        for pose, hits in zip(self.poses, records, strict=True):
            for left, right, depth, volume in hits:
                a, b = self.element_ids[left], self.element_ids[right]
                ek = (a, b) if a <= b else (b, a)
                pair = _pair_key(part_of[left], part_of[right])
                if (pair, _elem_base_key(ek)) in self.declared_elem_pairs:
                    declared = True
                else:
                    declared = pair in self.declared_pairs
                vol = float(volume)
                if vol < self.advisory_volume_tol:
                    continue
                span = max(float(x) for x in depth)
                blocking = (vol >= self.block_volume_tol) and not declared
                cur = worst.get(pair)
                if cur is None or vol > cur.motion_penetration_volume:
                    worst[pair] = ConnectionRamFinding(
                        link_a=pair[0],
                        link_b=pair[1],
                        motion_penetration_volume=vol,
                        penetration_span=span,
                        worst_pose={k: float(v) for k, v in dict(pose).items()},
                        offending_elements=ek,
                        blocking=blocking,
                    )
        return sorted(worst.values(), key=lambda f: -f.motion_penetration_volume)


def prepare_connection_ram_scan(
    model: object,
    *,
    asset_root: Optional[Path] = None,
    declared_pairs: Optional[set[tuple[str, str]]] = None,
    declared_elem_pairs: Optional[set[tuple[tuple[str, str], tuple[str, str]]]] = None,
    per_joint: int = 7,
    random_fill: int = 200,
    seed: int = 0,
    connection_tol: float = 1e-3,
    advisory_volume_tol: float = _RAM_ADVISORY_VOLUME_TOL,
    block_volume_tol: float = _RAM_BLOCK_VOLUME_TOL,
    pose_values: Optional[list[dict[str, float]]] = None,
    link_pairs: Optional[set[tuple[str, str]]] = None,
) -> Optional[ConnectionRamScan]:
    import numpy as np

    declared_pairs = {_pair_key(*p) for p in (declared_pairs or set())}
    declared_elem_pairs = declared_elem_pairs or set()
    otol = default_overlap_tol_from_env()
    vtol = default_overlap_volume_tol_from_env()

    screen_compiled = compile_for_motion_screen(model, asset_root=asset_root, validate=False)
    poses = (
        [{str(name): float(value) for name, value in pose.items()} for pose in pose_values]
        if pose_values is not None
        else generate_screening_poses(
            model, per_joint=per_joint, random_fill=random_fill, seed=seed
        )
    )
    if not poses:
        return None

    admissible = _ram_pair_filter(model, link_pairs)
    world = _build_exact_collision_world(
        screen_compiled, asset_root=resolve_asset_root(asset_root, screen_compiled, model)
    )
    if world.size < 2:
        return None
    names = world.part_names
    excluded = {
        (a, b) for i, a in enumerate(names) for b in names[i + 1 :] if not admissible((a, b))
    }

    rest_overlaps = _find_collision_overlaps_fcl_for_poses(
        screen_compiled,
        asset_root=asset_root,
        source_model=model,
        poses=[poses[0]],
        overlap_tol=0.0,
        overlap_volume_tol=0.0,
        allowed_pairs=excluded,
        contact_distance_tol=connection_tol,
        world=world,
    )
    rest_elem: set[tuple[str, str]] = {_elem_key(ov) for ov in rest_overlaps}
    if not rest_elem:
        return None

    part_of = world.part_of
    local_of = world.local_of
    element_ids = tuple(
        _elem_id(part_of[index], element.name, local_of[index])
        for index, element in enumerate(world.elements)
    )
    indices_of: dict[str, list[int]] = {}
    for index, identity in enumerate(element_ids):
        indices_of.setdefault(identity, []).append(index)
    candidates: set[tuple[int, int]] = set()
    for a, b in rest_elem:
        for first in indices_of.get(a, ()):
            for second in indices_of.get(b, ()):
                if part_of[first] == part_of[second]:
                    continue
                pair = (first, second) if first < second else (second, first)
                if admissible(_pair_key(part_of[pair[0]], part_of[pair[1]])):
                    candidates.add(pair)
    if not candidates:
        return None
    codes = world._part_codes.tolist()
    ordered = sorted(
        candidates,
        key=lambda pair: (codes[pair[0]], codes[pair[1]], local_of[pair[0]], local_of[pair[1]]),
    )
    return ConnectionRamScan(
        poses=poses,
        world=world,
        rows=np.asarray([pair[0] for pair in ordered], dtype=np.int64),
        cols=np.asarray([pair[1] for pair in ordered], dtype=np.int64),
        active=tuple(sorted({index for pair in ordered for index in pair})),
        element_ids=element_ids,
        declared_pairs=declared_pairs,
        declared_elem_pairs=declared_elem_pairs,
        overlap_tol=otol,
        overlap_volume_tol=vtol,
        advisory_volume_tol=advisory_volume_tol,
        block_volume_tol=block_volume_tol,
    )


def find_connection_ram_findings(
    model: object,
    *,
    asset_root: Optional[Path] = None,
    declared_pairs: Optional[set[tuple[str, str]]] = None,
    declared_elem_pairs: Optional[set[tuple[tuple[str, str], tuple[str, str]]]] = None,
    per_joint: int = 7,
    random_fill: int = 200,
    seed: int = 0,
    connection_tol: float = 1e-3,
    advisory_volume_tol: float = _RAM_ADVISORY_VOLUME_TOL,
    block_volume_tol: float = _RAM_BLOCK_VOLUME_TOL,
    validate_model: bool = False,
    pose_values: Optional[list[dict[str, float]]] = None,
    raise_on_error: bool = False,
) -> list["ConnectionRamFinding"]:
    try:
        scan = prepare_connection_ram_scan(
            model,
            asset_root=asset_root,
            declared_pairs=declared_pairs,
            declared_elem_pairs=declared_elem_pairs,
            per_joint=per_joint,
            random_fill=random_fill,
            seed=seed,
            connection_tol=connection_tol,
            advisory_volume_tol=advisory_volume_tol,
            block_volume_tol=block_volume_tol,
            pose_values=pose_values,
        )
        if scan is None:
            return []
        return scan.findings(scan.scan(range(len(scan.poses))))
    except Exception:  # noqa: BLE001 — compatibility callers remain best-effort
        if raise_on_error:
            raise
        logger.debug("find_connection_ram_findings failed", exc_info=True)
        return []


def find_motion_support_findings(
    model: object,
    *,
    gate_map: Optional[dict[str, list[str]]] = None,
    asset_root: Optional[Path] = None,
    contact_tol: float = 1e-3,
    gate_per: int = 5,
    validate_model: bool = False,
) -> list[UnsupportedPartFinding]:
    compiled = compile_for_motion_screen(model, asset_root=asset_root, validate=validate_model)
    part_names = [
        n
        for p in (getattr(compiled, "parts", None) or [])
        if isinstance((n := getattr(p, "name", None)), str)
    ]
    joints = getattr(compiled, "articulations", None) or []
    if len(part_names) <= 1:
        return []
    root_parts = _root_part_names(part_names, joints)

    by_name = {j.name: j for j in _moving_joints(model)}
    gates_of: dict[str, list] = {}
    if gate_map:
        for gated_name, gate_names in gate_map.items():
            if gated_name not in by_name:
                continue
            gobjs = [by_name[g] for g in (gate_names or []) if g in by_name]
            if gobjs:
                gates_of[gated_name] = gobjs

    poses: list[dict[str, float]] = []
    for j in _moving_joints(model):
        lo, hi = _screen_range(model, j)
        ends = sorted({round(v, 9) for v in (lo, hi) if abs(v) > 1e-9})
        if not ends:
            continue
        combos = (
            _gate_value_grid(model, gates_of[j.name], per_gate=gate_per)
            if j.name in gates_of
            else [{}]
        )
        for e in ends:
            for combo in combos:
                poses.append({j.name: e, **combo})
    if not poses:
        return []

    overlaps = _find_collision_overlaps_fcl_for_poses(
        compiled,
        asset_root=asset_root,
        source_model=model,
        poses=poses,
        overlap_tol=0.0,
        overlap_volume_tol=0.0,
        allowed_pairs=None,
        contact_distance_tol=contact_tol,
    )
    edges_by_pose: dict[int, set[tuple[str, str]]] = {}
    for ov in overlaps:
        edges_by_pose.setdefault(ov.pose_index, set()).add(_pair_key(ov.link_a, ov.link_b))

    seen: set[tuple[str, ...]] = set()
    findings: list[UnsupportedPartFinding] = []
    for i, pose in enumerate(poses):
        for f in _unsupported_component_findings(
            part_names,
            root_parts=root_parts,
            support_pairs=edges_by_pose.get(i, set()),
            pair_min_distances={},
            pose_index=i,
            pose=pose,
            contact_tol=contact_tol,
            backend="fcl",
        ):
            key = tuple(sorted(f.parts or ((f.part,) if f.part else ())))
            if key and key not in seen:
                seen.add(key)
                findings.append(f)
    return findings
