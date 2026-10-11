from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from pipeline.policies import (
    DEFAULT_POLICIES,
    JOINT_PROP_AXIS,
    JOINT_PROP_LIMIT_LOWER,
    JOINT_PROP_LIMIT_UPPER,
    JOINT_PROP_TYPE,
    Policies,
)


@dataclass(slots=True)
class GateResult:
    name: str
    passed: bool
    summary: str
    failures: list[str] = field(default_factory=list)

    @classmethod
    def ok(cls, name: str, summary: str = "ok") -> "GateResult":
        return cls(name=name, passed=True, summary=summary)


def joint_reference_frame_check(
    joints: list[dict[str, Any]], policies: Policies = DEFAULT_POLICIES
) -> GateResult:
    failures: list[str] = []
    for j in joints:
        jid = j.get("name", "<unnamed>")
        jtype = j.get(JOINT_PROP_TYPE)
        if jtype not in {"revolute", "continuous", "prismatic", "fixed", "detachable"}:
            failures.append(f"{jid}: unknown joint_type {jtype!r}")
        if jtype in {"fixed", "detachable"}:
            continue

        axis = j.get(JOINT_PROP_AXIS)
        if not axis or len(axis) != 3 or any(not math.isfinite(c) for c in axis):
            failures.append(f"{jid}: axis missing/non-finite: {axis!r}")
        else:
            norm = math.sqrt(sum(c * c for c in axis))
            if abs(norm - 1.0) > 1e-3:
                failures.append(
                    f"{jid}: axis not unit (|{axis}|={norm:.4f}); likely wrong "
                    f"reference frame or un-normalized"
                )

        lo = j.get(JOINT_PROP_LIMIT_LOWER)
        hi = j.get(JOINT_PROP_LIMIT_UPPER)
        if jtype == "continuous":
            pass
        elif lo is None or hi is None:
            failures.append(f"{jid}: missing limits ({lo}, {hi})")
        elif not (lo < hi):
            failures.append(f"{jid}: limit lower>=upper ({lo} >= {hi})")
        elif not (lo <= 0 <= hi):
            failures.append(
                f"{jid}: rest pose q=0 unreachable — window [{lo}, {hi}] must include 0"
            )
        else:
            rng = (
                policies.revolute_limit_range_rad
                if jtype == "revolute"
                else policies.prismatic_limit_range_m
            )
            span = hi - lo
            if span <= 0 or span > (rng[1] - rng[0]):
                failures.append(
                    f"{jid}: implausible {jtype} limit span {span} (allowed <= {rng[1] - rng[0]})"
                )

    if failures:
        return GateResult(
            name="joint_reference_frame",
            passed=False,
            summary=f"{len(failures)} joint-frame violation(s)",
            failures=failures,
        )
    return GateResult.ok("joint_reference_frame", f"{len(joints)} joint(s) valid")


COUPLING_SCALAR_TYPES = frozenset({"revolute", "continuous", "prismatic"})
NONLINEAR_COUPLING_TYPES = frozenset({"crank_slider", "cam", "scotch_yoke"})
BEHAVIOR_COUPLING_TYPES = frozenset({"latch_release", "self_closer", "momentary"})
SUPPORTED_COUPLING_TYPES = (
    frozenset({"proportional"}) | NONLINEAR_COUPLING_TYPES | BEHAVIOR_COUPLING_TYPES
)

_BEHAVIOR_SPEC_FIELDS = {
    "latch_release": {
        "trigger_at": {"min", "max"},
        "follower_goes": {"min", "max"},
        "strength": {"soft", "firm", "strong"},
    },
    "self_closer": {
        "closes_to": {"min", "max"},
        "threshold": {"min", "max"},
        "strength": {"soft", "firm", "strong"},
    },
    "momentary": {"follower_direction": {"min", "max"}, "speed": {"slow", "normal", "fast"}},
}
_CONTINUOUS_REFERENCE = 2.0 * math.pi


def _coupling_cycle(edges: dict[str, str]) -> list[str] | None:
    for start in edges:
        seen: list[str] = []
        in_path: set[str] = set()
        node: str | None = start
        while node in edges:
            if node in in_path:
                i = seen.index(node)
                return seen[i:] + [node]
            in_path.add(node)
            seen.append(node)
            node = edges[node]
    return None


def _coupling_travel(jdict: dict[str, Any]) -> float | None:
    lo = jdict.get("limit_lower", jdict.get("lower"))
    hi = jdict.get("limit_upper", jdict.get("upper"))
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and float(hi) > float(lo):
        return float(hi) - float(lo)
    return None


def _expected_multiplier_magnitude(
    driver: dict[str, Any] | None,
    follower: dict[str, Any] | None,
    mapping: object,
    turns: object = None,
) -> float | None:
    if isinstance(turns, (int, float)) and not isinstance(turns, bool) and turns > 0:
        ftr = _coupling_travel(follower) if follower is not None else None
        return ftr / (float(turns) * _CONTINUOUS_REFERENCE) if ftr else None
    if isinstance(mapping, (int, float)) and not isinstance(mapping, bool):
        return abs(float(mapping)) if math.isfinite(mapping) and mapping != 0 else None
    if mapping != "full" or driver is None or follower is None:
        return None
    ftr = _coupling_travel(follower)
    if ftr is None:
        return None
    ref = _CONTINUOUS_REFERENCE if driver.get("type") == "continuous" else _coupling_travel(driver)
    if not ref:
        return None
    return ftr / ref


_ANGULAR_COUPLING_TYPES = frozenset({"revolute", "continuous"})


def _coupling_domain(jdict: dict[str, Any] | None) -> str | None:
    if jdict is None:
        return None
    t = jdict.get("type")
    if t in _ANGULAR_COUPLING_TYPES:
        return "angular"
    if t == "prismatic":
        return "linear"
    return None


_MIN_ANGULAR_SWEEP = math.radians(3.0)
_MAX_ANGULAR_SWEEP = 200.0 * 2.0 * math.pi
_MIN_LINEAR_SWEEP = 5e-4
_MAX_LINEAR_SWEEP = 10.0


def _check_coupling_operable(
    cid: object,
    driver: dict[str, Any] | None,
    follower: dict[str, Any] | None,
    spec: dict[str, Any],
    failures: list[str],
) -> None:
    if driver is None or follower is None:
        return
    ftr = _coupling_travel(follower)
    if ftr is None or ftr <= 0:
        return
    turns = spec.get("turns")
    mapping = spec.get("mapping")
    needed: float | None = None
    domain = _coupling_domain(driver)
    if isinstance(turns, (int, float)) and not isinstance(turns, bool) and turns > 0:
        needed, domain = float(turns) * 2.0 * math.pi, "angular"
    elif (
        isinstance(mapping, (int, float))
        and not isinstance(mapping, bool)
        and math.isfinite(mapping)
        and mapping > 0
    ):
        needed = ftr / float(mapping)
    if needed is None or domain is None:
        return
    lo, hi = (
        (_MIN_ANGULAR_SWEEP, _MAX_ANGULAR_SWEEP)
        if domain == "angular"
        else (_MIN_LINEAR_SWEEP, _MAX_LINEAR_SWEEP)
    )
    unit = "rad" if domain == "angular" else "m"
    if needed < lo or needed > hi:
        failures.append(
            f"coupling {cid}: the declared ratio implies an inoperable driver sweep of "
            f"{needed:.4g} {unit} to drive the follower across its travel — the control would be "
            "frozen or the ratio is physically wrong. Use turns=N (N driver revolutions sweep the "
            "follower's full travel) or mapping='full'."
        )


_REST_MIRROR = {"at_min": "at_max", "at_max": "at_min", "centered": "centered"}


def _check_rest_correspondence(
    cid: object,
    driver: dict[str, Any] | None,
    follower: dict[str, Any] | None,
    sense: object,
    failures: list[str],
) -> None:
    if driver is None or follower is None or driver.get("type") == "continuous":
        return
    d_rest = driver.get("rest")
    f_rest = follower.get("rest")
    if d_rest not in _REST_MIRROR or f_rest not in _REST_MIRROR:
        return
    want = _REST_MIRROR[d_rest] if sense == "opposite" else d_rest
    if f_rest != want:
        failures.append(
            f"coupling {cid}: mapping='full' rest mismatch — driver rest {d_rest!r} with sense "
            f"{sense!r} implies follower rest {want!r}, but the follower rest is {f_rest!r}"
        )


def _pos(spec: dict[str, Any], key: str) -> bool:
    v = spec.get(key)
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0


def _check_nonlinear_spec(
    cid: object, ctype: str, spec: dict[str, Any], failures: list[str]
) -> None:
    sense = spec.get("sense", "same")
    if sense not in {"same", "opposite"}:
        failures.append(f"coupling {cid}: spec.sense must be 'same' or 'opposite'")
    if ctype == "crank_slider":
        if not (_pos(spec, "crank_radius") and _pos(spec, "rod_length")):
            failures.append(
                f"coupling {cid}: crank_slider needs crank_radius > 0 and rod_length > 0"
            )
        elif spec["rod_length"] <= spec["crank_radius"]:
            failures.append(f"coupling {cid}: crank_slider needs rod_length > crank_radius")
    elif ctype == "cam":
        if not (_pos(spec, "lift") and _pos(spec, "rise_angle")):
            failures.append(f"coupling {cid}: cam needs lift > 0 and rise_angle > 0")
    elif ctype == "scotch_yoke":
        if not _pos(spec, "radius"):
            failures.append(f"coupling {cid}: scotch_yoke needs radius > 0")


_NONLINEAR_PARAM_KEYS = frozenset(
    {"crank_radius", "rod_length", "lift", "rise_angle", "dwell_angle", "radius"}
)

_NONLINEAR_FIT_TOL = 0.06


def _nonlinear_driver_window(
    ctype: str, spec: dict[str, Any], driver: dict[str, Any] | None
) -> tuple[float, float] | None:
    if driver is None:
        return None
    try:
        from sdk._core.v0._coupling import template_monotone_window

        params = {k: float(spec[k]) for k in _NONLINEAR_PARAM_KEYS if k in spec}
        mono_lo, mono_hi = template_monotone_window(ctype, **params)
    except Exception:  # noqa: BLE001 — bad params already fail elsewhere
        return None
    lo = driver.get("limit_lower", driver.get("lower"))
    hi = driver.get("limit_upper", driver.get("upper"))
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and float(hi) > float(lo):
        w_lo, w_hi = max(float(lo), mono_lo), min(float(hi), mono_hi)
        return (w_lo, w_hi) if w_hi > w_lo else None
    if driver.get("type") == "continuous":
        return mono_lo, mono_hi
    return None


def _check_nonlinear_sense(
    cid: object,
    ctype: str,
    spec: dict[str, Any],
    driver: dict[str, Any] | None,
    built_poly: list[Any],
    failures: list[str],
) -> None:
    window = _nonlinear_driver_window(ctype, spec, driver)
    if window is None:
        return
    sense = spec.get("sense", "same")
    params = {k: spec[k] for k in _NONLINEAR_PARAM_KEYS if k in spec}
    try:
        from sdk._core.v0._coupling import evaluate_poly, nonlinear_template_poly

        expected = nonlinear_template_poly(ctype, driver_window=window, sense=sense, **params)
    except Exception:  # noqa: BLE001 — bad params already rejected at declaration/build
        return
    built = tuple(float(x) for x in built_poly)
    lo, hi = window
    samples = [lo + (hi - lo) * k / 8.0 for k in range(1, 8)]
    theta = max(samples, key=lambda t: abs(evaluate_poly(expected, t)))
    exp_v = evaluate_poly(expected, theta)
    if abs(exp_v) < 1e-9:
        return
    built_v = evaluate_poly(built, theta)
    if (built_v < 0.0) != (exp_v < 0.0):
        failures.append(
            f"coupling {cid}: {ctype} built curve travels opposite to declared sense "
            f"{sense!r} (built {built_v:+.4g} vs expected {exp_v:+.4g} at theta={theta:.4g})"
        )


_BEHAVIOR_OPTIONAL = {"strength", "speed"}


def _check_behavior_spec(
    cid: object, ctype: str, spec: dict[str, Any], failures: list[str]
) -> None:
    for key, allowed in _BEHAVIOR_SPEC_FIELDS.get(ctype, {}).items():
        v = spec.get(key)
        if v is None:
            if key not in _BEHAVIOR_OPTIONAL:
                failures.append(f"coupling {cid}: {ctype} requires spec.{key}")
        elif v not in allowed:
            failures.append(f"coupling {cid}: {ctype} spec.{key} must be one of {sorted(allowed)}")


def joint_coupling_check(
    joints: list[dict[str, Any]], couplings: list[dict[str, Any]] | None
) -> GateResult:
    couplings = couplings or []
    if not couplings:
        return GateResult.ok("joint_coupling", "no couplings declared")

    by_name = {j.get("name"): j for j in joints if isinstance(j.get("name"), str)}
    failures: list[str] = []
    followers_seen: set[str] = set()
    edges: dict[str, str] = {}

    for idx, c in enumerate(couplings):
        if not isinstance(c, dict):
            failures.append(f"coupling #{idx}: not an object")
            continue
        driver = c.get("driver")
        follower = c.get("follower")
        ctype = c.get("type")
        cid = follower if isinstance(follower, str) else f"#{idx}"

        if not isinstance(driver, str) or driver not in by_name:
            failures.append(f"coupling {cid}: driver {driver!r} is not a declared joint")
        if not isinstance(follower, str) or follower not in by_name:
            failures.append(f"coupling {cid}: follower {follower!r} is not a declared joint")
        if (
            isinstance(driver, str)
            and isinstance(follower, str)
            and driver == follower
            and ctype != "self_closer"
        ):
            failures.append(f"coupling {cid}: driver and follower are the same joint")
        if ctype not in SUPPORTED_COUPLING_TYPES:
            failures.append(
                f"coupling {cid}: type {ctype!r} not supported "
                f"(use one of {sorted(SUPPORTED_COUPLING_TYPES)})"
            )

        for role, name in (("driver", driver), ("follower", follower)):
            if isinstance(name, str) and name in by_name:
                jt = by_name[name].get("type")
                if jt not in COUPLING_SCALAR_TYPES:
                    failures.append(
                        f"coupling {cid}: {role} {name!r} type {jt!r} cannot be coupled "
                        f"(need revolute/continuous/prismatic)"
                    )

        if isinstance(follower, str):
            if follower in followers_seen:
                failures.append(
                    f"coupling {cid}: follower {follower!r} driven by more than one coupling"
                )
            followers_seen.add(follower)
            if isinstance(driver, str) and ctype not in BEHAVIOR_COUPLING_TYPES:
                edges[follower] = driver

        if ctype == "proportional":
            spec = c.get("spec") or {}
            sense = spec.get("sense")
            mapping = spec.get("mapping")
            turns = spec.get("turns")
            driver_j = by_name.get(driver) if isinstance(driver, str) else None
            follower_j = by_name.get(follower) if isinstance(follower, str) else None
            if sense not in {"same", "opposite"}:
                failures.append(f"coupling {cid}: spec.sense must be 'same' or 'opposite'")
            has_turns = turns is not None
            numeric = (
                isinstance(mapping, (int, float))
                and not isinstance(mapping, bool)
                and math.isfinite(mapping)
                and mapping > 0
            )
            if has_turns:
                if mapping not in (None, "full"):
                    failures.append(f"coupling {cid}: declare spec.turns OR spec.mapping, not both")
                if not (
                    isinstance(turns, (int, float))
                    and not isinstance(turns, bool)
                    and math.isfinite(turns)
                    and turns > 0
                ):
                    failures.append(
                        f"coupling {cid}: spec.turns must be a POSITIVE number of revolutions"
                    )
                elif driver_j is not None and driver_j.get("type") not in _ANGULAR_COUPLING_TYPES:
                    failures.append(
                        f"coupling {cid}: spec.turns needs a rotary driver (revolute/continuous), "
                        f"got {driver_j.get('type')!r}"
                    )
            else:
                if not (mapping == "full" or numeric):
                    failures.append(
                        f"coupling {cid}: spec.mapping must be 'full', a POSITIVE finite number, or "
                        "use spec.turns=N (the sign comes from spec.sense, not the ratio)"
                    )
                if numeric:
                    dd, fd = _coupling_domain(driver_j), _coupling_domain(follower_j)
                    if dd is not None and fd is not None and dd != fd:
                        failures.append(
                            f"coupling {cid}: a numeric mapping is a dimensionless ratio for "
                            "SAME-DOMAIN drives only; this coupling is cross-domain "
                            "(rotary<->linear) — use spec.turns=N or mapping='full'"
                        )
                if mapping == "full":
                    if follower_j is not None and follower_j.get("type") == "continuous":
                        failures.append(
                            f"coupling {cid}: mapping='full' needs a bounded follower; a continuous "
                            "follower has no travel — use a numeric ratio (e.g. a gear ratio)"
                        )
                    else:
                        _check_rest_correspondence(cid, driver_j, follower_j, sense, failures)
            _check_coupling_operable(cid, driver_j, follower_j, spec, failures)
        elif ctype in NONLINEAR_COUPLING_TYPES:
            _check_nonlinear_spec(cid, ctype, c.get("spec") or {}, failures)
            driver_j = by_name.get(driver) if isinstance(driver, str) else None
            follower_j = by_name.get(follower) if isinstance(follower, str) else None
            if driver_j is not None and driver_j.get("type") not in {"revolute", "continuous"}:
                failures.append(
                    f"coupling {cid}: {ctype} driver {driver!r} must be rotary "
                    f"(revolute/continuous), got {driver_j.get('type')!r}"
                )
            if follower_j is not None and follower_j.get("type") != "prismatic":
                failures.append(
                    f"coupling {cid}: {ctype} follower {follower!r} must be prismatic (linear), "
                    f"got {follower_j.get('type')!r}"
                )
        elif ctype in BEHAVIOR_COUPLING_TYPES:
            _check_behavior_spec(cid, ctype, c.get("spec") or {}, failures)

    cycle = _coupling_cycle(edges)
    if cycle:
        failures.append(f"coupling cycle detected: {' -> '.join(cycle)}")
    chained = sorted(set(edges.values()) & set(edges.keys()))
    if chained:
        failures.append(
            f"coupling chain not supported: joint(s) {chained} are both a driver and a follower "
            "— couple each follower directly to its root driver (compose the ratios)"
        )

    if failures:
        return GateResult(
            name="joint_coupling",
            passed=False,
            summary=f"{len(failures)} coupling violation(s)",
            failures=failures,
        )
    return GateResult.ok("joint_coupling", f"{len(couplings)} coupling(s) valid")


_MOVING_JOINT_TYPES = {"revolute", "continuous", "prismatic"}


def _access_gate_cycle(graph: dict[str, list[str]]) -> list[str] | None:
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {}
    path: list[str] = []

    def visit(node: str) -> list[str] | None:
        color[node] = GRAY
        path.append(node)
        for nxt in graph.get(node, []):
            c = color.get(nxt, WHITE)
            if c == GRAY:
                return path[path.index(nxt) :] + [nxt]
            if c == WHITE and nxt in graph:
                r = visit(nxt)
                if r:
                    return r
        path.pop()
        color[node] = BLACK
        return None

    for n in graph:
        if color.get(n, WHITE) == WHITE:
            r = visit(n)
            if r:
                return r
    return None


def access_gate_check(parts: list[dict[str, Any]], joints: list[dict[str, Any]]) -> GateResult:
    parts = parts or []
    part_names = {
        p["name"] for p in parts if isinstance(p, dict) and isinstance(p.get("name"), str)
    }
    movable: set[str] = set()
    for j in joints or []:
        if not isinstance(j, dict):
            continue
        child = j.get("child")
        if isinstance(child, str) and str(j.get("type")) in _MOVING_JOINT_TYPES:
            movable.add(child)

    failures: list[str] = []
    graph: dict[str, list[str]] = {}
    n_decls = 0
    for p in parts:
        if not isinstance(p, dict):
            continue
        mg = p.get("motion_gate")
        if mg is None:
            continue
        name = p.get("name")
        pid = name if isinstance(name, str) else "<unnamed>"
        if not isinstance(mg, dict):
            failures.append(
                f'part {pid!r}: motion_gate must be an object {{"gates": [...], "reason": ...}}'
            )
            continue
        gates = mg.get("gates")
        if gates is None or (isinstance(gates, list) and len(gates) == 0):
            continue
        if not isinstance(gates, list) or not all(isinstance(g, str) for g in gates):
            failures.append(
                f"part {pid!r}: motion_gate.gates must be a list of declared part names"
            )
            continue
        n_decls += 1
        if not isinstance(name, str) or name not in movable:
            failures.append(
                f"part {pid!r}: declares motion_gate but is not movable — a gated part needs a "
                "revolute/continuous/prismatic parent joint"
            )
        if isinstance(name, str) and name in gates:
            failures.append(f"part {pid!r}: cannot gate itself")
        for g in gates:
            if g not in part_names:
                failures.append(f"part {pid!r}: gate {g!r} is not a declared part")
            elif g not in movable:
                failures.append(
                    f"part {pid!r}: gate {g!r} is not movable — a gate must be able to open "
                    "(needs a revolute/continuous/prismatic parent joint)"
                )
        if isinstance(name, str):
            graph[name] = [g for g in gates if g != name]

    cycle = _access_gate_cycle(graph)
    if cycle:
        failures.append(f"access-gate cycle (deadlock): {' -> '.join(cycle)}")

    for name, gates in sorted(graph.items()):
        for gate in gates:
            if graph.get(gate):
                failures.append(
                    f"part {name!r}: gate {gate!r} is itself gated by "
                    f"{graph[gate]!r} — nested access gates are unsupported; "
                    "gate on a directly openable part"
                )

    if failures:
        return GateResult(
            name="access_gate",
            passed=False,
            summary=f"{len(failures)} access-gate violation(s)",
            failures=failures,
        )
    if n_decls == 0:
        return GateResult.ok("access_gate", "no access gates declared")
    return GateResult.ok("access_gate", f"{n_decls} access gate(s) valid")


def _behavior_endpoint(joint: dict[str, Any] | None, which: object) -> float | None:
    if joint is None or which not in {"min", "max"}:
        return None
    key = "limit_upper" if which == "max" else "limit_lower"
    value = joint.get(key, joint.get("upper" if which == "max" else "lower"))
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _check_behavior_conformance(
    cid: object,
    ctype: str,
    spec: dict[str, Any],
    behavior: dict[str, Any],
    driver: dict[str, Any] | None,
    follower: dict[str, Any] | None,
    failures: list[str],
) -> None:
    built_params = behavior.get("params")
    if not isinstance(built_params, dict):
        failures.append(f"coupling {cid}: built {ctype} behavior has no authored params")
    else:
        for key, expected in spec.items():
            if built_params.get(key) != expected:
                failures.append(
                    f"coupling {cid}: built {ctype} param {key}={built_params.get(key)!r} "
                    f"!= declared {expected!r}"
                )

    if ctype == "latch_release":
        expected_target = _behavior_endpoint(follower, spec.get("follower_goes"))
        expected_threshold = _behavior_endpoint(driver, spec.get("trigger_at"))
    elif ctype == "self_closer":
        expected_target = _behavior_endpoint(follower, spec.get("closes_to"))
        expected_threshold = _behavior_endpoint(follower, spec.get("threshold"))
    else:
        expected_target = _behavior_endpoint(follower, spec.get("follower_direction"))
        expected_threshold = None

    for behavior_field, expected in (
        ("target", expected_target),
        ("threshold", expected_threshold),
    ):
        if expected is None:
            continue
        actual = behavior.get(behavior_field)
        if (
            not isinstance(actual, (int, float))
            or isinstance(actual, bool)
            or not math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-9)
        ):
            failures.append(
                f"coupling {cid}: built {ctype} {behavior_field}={actual!r} "
                "!= declared endpoint "
                f"{expected:.6g}"
            )

    if ctype == "momentary":
        expected_direction = 1.0 if spec.get("follower_direction") == "max" else -1.0
        actual_direction = behavior.get("direction")
        if not isinstance(actual_direction, (int, float)) or not math.isclose(
            float(actual_direction), expected_direction, rel_tol=0.0, abs_tol=1e-9
        ):
            failures.append(
                f"coupling {cid}: built momentary direction={actual_direction!r} "
                f"!= declared {expected_direction:+.0f}"
            )
    elif behavior.get("strength") != spec.get("strength", "firm"):
        failures.append(
            f"coupling {cid}: built {ctype} strength={behavior.get('strength')!r} "
            f"!= declared {spec.get('strength', 'firm')!r}"
        )


def coupling_conformance_check(
    probe_joints: list[dict[str, Any]], couplings: list[dict[str, Any]] | None
) -> GateResult:
    couplings = couplings or []
    by_name = {j.get("name"): j for j in probe_joints if isinstance(j.get("name"), str)}

    built: dict[str, dict[str, Any]] = {}
    built_behaviors: dict[str, dict[str, Any]] = {}
    for j in probe_joints:
        name = j.get("name")
        if not isinstance(name, str):
            continue
        mim = j.get("mimic")
        if isinstance(mim, dict) and mim.get("joint"):
            built[name] = mim
        beh = j.get("behavior")
        if isinstance(beh, dict) and beh.get("kind"):
            built_behaviors[name] = beh

    if not couplings:
        undeclared = [
            f"joint {n!r} has an undeclared mimic on {m.get('joint')!r}"
            for n, m in sorted(built.items())
        ] + [
            f"joint {n!r} has an undeclared {b.get('kind')} behavior"
            for n, b in sorted(built_behaviors.items())
        ]
        if undeclared:
            return GateResult(
                name="coupling_conformance",
                passed=False,
                summary=f"{len(undeclared)} undeclared coupling(s) in build",
                failures=undeclared,
            )
        return GateResult.ok("coupling_conformance", "no couplings declared")

    failures: list[str] = []
    declared_followers: set[str] = set()
    declared_behavior_followers: set[str] = set()
    for idx, c in enumerate(couplings):
        if not isinstance(c, dict):
            failures.append(f"coupling #{idx}: not an object")
            continue
        driver = c.get("driver")
        follower = c.get("follower")
        ctype = c.get("type")
        cid = follower if isinstance(follower, str) else f"#{idx}"
        if not isinstance(follower, str):
            failures.append(f"coupling {cid}: follower name missing")
            continue

        if ctype in BEHAVIOR_COUPLING_TYPES:
            declared_behavior_followers.add(follower)
            beh = built_behaviors.get(follower)
            if beh is None:
                failures.append(
                    f"coupling {cid}: {ctype} declared but the built follower has no behavior "
                    "(wire it with model.couple(..., kind=<behavior>, **params))"
                )
            elif beh.get("kind") != ctype:
                failures.append(
                    f"coupling {cid}: built behavior kind {beh.get('kind')!r} != declared {ctype!r}"
                )
            else:
                built_driver = beh.get("driver")
                want_driver = follower if ctype == "self_closer" else driver
                if isinstance(want_driver, str) and built_driver != want_driver:
                    failures.append(
                        f"coupling {cid}: {ctype} fires on {built_driver!r}, not the declared "
                        f"{'self (' + follower + ')' if ctype == 'self_closer' else repr(driver)}"
                    )
                _check_behavior_conformance(
                    cid,
                    str(ctype),
                    c.get("spec") or {},
                    beh,
                    by_name.get(driver) if isinstance(driver, str) else None,
                    by_name.get(follower),
                    failures,
                )
            continue

        declared_followers.add(follower)

        mim = built.get(follower)
        if mim is None:
            failures.append(
                f"coupling {cid}: follower {follower!r} has NO mimic in the built model "
                "(the declared coupling was not wired)"
            )
            continue
        if mim.get("joint") != driver:
            failures.append(
                f"coupling {cid}: follower {follower!r} is driven by {mim.get('joint')!r}, "
                f"not the declared driver {driver!r}"
            )
            continue

        if c.get("type") in NONLINEAR_COUPLING_TYPES:
            poly = mim.get("poly") or []
            if len(poly) < 2 or all(abs(float(x)) < 1e-12 for x in poly):
                failures.append(
                    f"coupling {cid}: {c.get('type')} declared but the built follower has no "
                    "fitted polynomial (wire it with model.couple(..., kind=<template>, **params))"
                )
                continue
            built_kind = mim.get("kind")
            if isinstance(built_kind, str) and built_kind != c.get("type"):
                failures.append(
                    f"coupling {cid}: built nonlinear template {built_kind!r} != declared "
                    f"{c.get('type')!r} (wire model.couple(..., kind='{c.get('type')}', ...))"
                )
                continue
            lin = mim.get("multiplier")
            if not isinstance(lin, (int, float)) or isinstance(lin, bool) or not math.isfinite(lin):
                failures.append(
                    f"coupling {cid}: nonlinear linear multiplier {lin!r} is not finite"
                )
                continue
            fit_error = mim.get("fit_error")
            if isinstance(fit_error, (int, float)) and fit_error > _NONLINEAR_FIT_TOL:
                failures.append(
                    f"coupling {cid}: {c.get('type')} degree-4 fit error "
                    f"{fit_error * 100:.1f}% exceeds {_NONLINEAR_FIT_TOL * 100:.0f}% of stroke "
                    "(the driver window is too wide for a faithful fit — tighten it)"
                )
            _check_nonlinear_sense(
                cid,
                str(c.get("type")),
                c.get("spec") or {},
                by_name.get(driver),
                poly,
                failures,
            )
            continue

        mult = mim.get("multiplier")
        if (
            not isinstance(mult, (int, float))
            or isinstance(mult, bool)
            or not math.isfinite(mult)
            or mult == 0
        ):
            failures.append(f"coupling {cid}: multiplier {mult!r} is not a non-zero finite number")
            continue
        off = mim.get("offset")
        if isinstance(off, (int, float)) and abs(off) > 1e-6:
            failures.append(
                f"coupling {cid}: offset {off!r} should be ~0 (coupled rest poses must correspond)"
            )

        spec = c.get("spec") or {}
        sense = spec.get("sense", "same")
        want_negative = sense == "opposite"
        if (mult < 0) != want_negative:
            failures.append(
                f"coupling {cid}: multiplier sign ({mult:+.4g}) does not match sense {sense!r}"
            )

        mapping = spec.get("mapping")
        turns = spec.get("turns")
        if mapping is None and turns is None:
            failures.append(
                f"coupling {cid}: proportional coupling declares no spec.mapping or spec.turns — "
                "the multiplier magnitude cannot be verified (declare mapping='full', a positive "
                "numeric ratio, or turns=N)"
            )
        else:
            expected = _expected_multiplier_magnitude(
                by_name.get(driver), by_name.get(follower), mapping, turns
            )
            if expected is not None and expected > 0:
                is_numeric = isinstance(mapping, (int, float)) and not isinstance(mapping, bool)
                tol = 0.05 if is_numeric else 0.10
                if abs(abs(mult) - expected) > tol * expected + 1e-9:
                    label = f"turns={turns!r}" if turns is not None else f"mapping={mapping!r}"
                    failures.append(
                        f"coupling {cid}: |multiplier|={abs(mult):.4g} differs from expected "
                        f"{expected:.4g} ({label}, tolerance {tol:.0%})"
                    )

    for n in sorted(built):
        if n not in declared_followers:
            failures.append(f"joint {n!r} has an undeclared mimic (invented coupling)")
    for n in sorted(built_behaviors):
        if n not in declared_behavior_followers:
            failures.append(f"joint {n!r} has an undeclared behavior (invented coupling)")

    if failures:
        return GateResult(
            name="coupling_conformance",
            passed=False,
            summary=f"{len(failures)} coupling-conformance violation(s)",
            failures=failures,
        )
    return GateResult.ok("coupling_conformance", f"{len(couplings)} coupling(s) match the build")


def dynamics_intent_conformance_check(
    probe_joints: list[dict[str, Any]], blueprint_joints: list[dict[str, Any]] | None
) -> GateResult:
    from sdk._core.v0.types import MotionFeel

    blueprint_joints = blueprint_joints or []
    declared = {
        str(joint.get("name")): joint.get("dynamics_intent")
        for joint in blueprint_joints
        if isinstance(joint, dict)
        and isinstance(joint.get("name"), str)
        and isinstance(joint.get("dynamics_intent"), dict)
    }
    built = {
        str(joint.get("name")): joint
        for joint in probe_joints
        if isinstance(joint, dict) and isinstance(joint.get("name"), str)
    }
    failures: list[str] = []

    for name, raw_intent in declared.items():
        joint = built.get(name)
        if joint is None:
            failures.append(f"joint {name!r}: declared dynamics_intent but joint is missing")
            continue
        expected = MotionFeel.from_intent(
            effort=raw_intent.get("effort"),
            motion=raw_intent.get("motion"),
            hold=raw_intent.get("hold"),
            note=str(raw_intent.get("feel", "")),
        )
        actual = joint.get("motion_feel")
        if not isinstance(actual, dict):
            failures.append(f"joint {name!r}: missing motion_feel for declared dynamics_intent")
            continue
        if joint.get("mimic"):
            failures.append(f"joint {name!r}: motion_feel is not allowed on a mimic follower")
        if joint.get("behavior"):
            failures.append(f"joint {name!r}: motion_feel is not allowed on a behavior-owned joint")
        for field_name in ("effort", "motion", "hold"):
            actual_value = actual.get(field_name)
            expected_value = getattr(expected, field_name)
            if actual_value != expected_value:
                failures.append(
                    f"joint {name!r}: motion_feel.{field_name}={actual_value!r} != "
                    f"normalized blueprint value {expected_value!r}"
                )

    if failures:
        return GateResult(
            name="dynamics_intent_conformance",
            passed=False,
            summary=f"{len(failures)} dynamics-intent conformance violation(s)",
            failures=failures,
        )
    return GateResult.ok(
        "dynamics_intent_conformance",
        f"{len(declared)} dynamics intent(s) match the build",
    )


def blueprint_presence_check(blueprint: dict[str, Any]) -> GateResult:
    required = ("blueprint_meta", "parts", "joints", "acceptance")
    missing = [k for k in required if k not in blueprint]
    if missing:
        return GateResult(
            name="blueprint_presence",
            passed=False,
            summary=f"missing keys: {missing}",
            failures=[f"missing top-level key: {k}" for k in missing],
        )
    return GateResult.ok("blueprint_presence")


_PROVENANCE_TARGET_TYPES = frozenset({"joint", "task", "interaction"})
_PROVENANCE_SOURCES = frozenset({"prompt_required", "visually_observed", "inferred_from_category"})
_PROVENANCE_CONFIDENCE = frozenset({"high", "medium", "low"})


def blueprint_decision_provenance_check(blueprint: dict[str, Any]) -> GateResult:
    raw = blueprint.get("decision_provenance")
    if raw is None:
        return GateResult.ok("blueprint_decision_provenance", "no provenance declared")
    if not isinstance(raw, list):
        return GateResult(
            name="blueprint_decision_provenance",
            passed=False,
            summary="decision_provenance must be a list",
            failures=["`decision_provenance` must be a list of provenance records"],
        )

    joint_names = {
        str(item.get("name"))
        for item in blueprint.get("joints", [])
        if isinstance(item, dict) and item.get("name")
    }
    task_names: set[str] = set()
    for item in blueprint.get("tasks") or []:
        if isinstance(item, str):
            instruction = item
        elif isinstance(item, dict):
            instruction = item.get("instruction")
        else:
            continue
        if isinstance(instruction, str) and instruction.strip():
            task_names.add(instruction)
    failures: list[str] = []
    seen: set[tuple[str, str]] = set()
    covered_joints: set[str] = set()
    covered_tasks: set[str] = set()
    for index, item in enumerate(raw):
        label = f"decision_provenance[{index}]"
        if not isinstance(item, dict):
            failures.append(f"{label} must be an object")
            continue
        target_type = str(item.get("target_type") or "")
        target = str(item.get("target") or "").strip()
        source = str(item.get("source") or "")
        confidence = str(item.get("confidence") or "")
        evidence = str(item.get("evidence") or "").strip()
        if target_type not in _PROVENANCE_TARGET_TYPES:
            failures.append(f"{label}.target_type must be joint, task, or interaction")
        if not target:
            failures.append(f"{label}.target must be non-empty")
        if source not in _PROVENANCE_SOURCES:
            failures.append(f"{label}.source must be one of {sorted(_PROVENANCE_SOURCES)}")
        if confidence not in _PROVENANCE_CONFIDENCE:
            failures.append(f"{label}.confidence must be one of {sorted(_PROVENANCE_CONFIDENCE)}")
        if not evidence:
            failures.append(f"{label}.evidence must be non-empty")
        key = (target_type, target)
        if target and key in seen:
            failures.append(f"duplicate provenance target {target_type}:{target}")
        seen.add(key)
        if target_type == "joint":
            covered_joints.add(target)
            if target not in joint_names:
                failures.append(f"{label}.target names undeclared joint {target!r}")
        elif target_type == "task":
            covered_tasks.add(target)
            if target not in task_names:
                failures.append(f"{label}.target names undeclared task {target!r}")
    for name in sorted(joint_names - covered_joints):
        failures.append(f"missing provenance for joint {name!r}")
    for name in sorted(task_names - covered_tasks):
        failures.append(f"missing provenance for task {name!r}")
    if failures:
        return GateResult(
            name="blueprint_decision_provenance",
            passed=False,
            summary=f"{len(failures)} decision-provenance violation(s)",
            failures=failures,
        )
    return GateResult.ok("blueprint_decision_provenance", f"{len(raw)} provenance record(s) valid")


MANIP_INTENTS = frozenset(
    {
        "grasp_pull_prismatic",
        "grasp_swing_revolute",
        "grasp_lift_lid",
        "grasp_turn_crank",
        "press_turn_switch",
        "press_button",
        "press_pedal_coupled",
        "push_prismatic",
        "push_whole_body",
    }
)

WHOLE_BODY_TARGET = "__whole_body__"
WHOLE_BODY_JOINT_TYPE = "whole_body"

_MANIP_INTENT_JOINT_TYPES: dict[str, frozenset[str]] = {
    "grasp_pull_prismatic": frozenset({"prismatic"}),
    "grasp_swing_revolute": frozenset({"revolute", "continuous"}),
    "grasp_lift_lid": frozenset({"revolute", "prismatic"}),
    "grasp_turn_crank": frozenset({"revolute", "continuous"}),
    "press_turn_switch": frozenset({"revolute", "continuous"}),
    "press_button": frozenset({"prismatic"}),
    "press_pedal_coupled": frozenset({"revolute", "prismatic"}),
    "push_prismatic": frozenset({"prismatic"}),
    "push_whole_body": frozenset({WHOLE_BODY_JOINT_TYPE}),
}


def _walk_to_actuated_joint(link: str | None, joints: list[dict[str, Any]]) -> str | None:
    if not isinstance(link, str):
        return None
    by_child: dict[str, dict[str, Any]] = {}
    for j in joints or []:
        if isinstance(j, dict) and isinstance(j.get("child"), str):
            by_child.setdefault(j["child"], j)
    seen: set[str] = set()
    cur: Any = link
    while isinstance(cur, str) and cur in by_child and cur not in seen:
        seen.add(cur)
        j = by_child[cur]
        jtype = str(j.get("type") or j.get(JOINT_PROP_TYPE) or "")
        if jtype in _MOVING_JOINT_TYPES:
            return j.get("name")
        cur = j.get("parent")
    return None


def _moving_ancestor_path(link: str, target: str, joints: list[dict[str, Any]]) -> list[str] | None:
    by_child = {}
    for joint in joints:
        if not isinstance(joint, dict):
            return None
        child = joint.get("child")
        if child in by_child:
            return None
        by_child[child] = joint
    path = []
    seen = set()
    current = link
    while current in by_child and current not in seen:
        seen.add(current)
        joint = by_child[current]
        kind = str(joint.get("type") or joint.get(JOINT_PROP_TYPE) or "")
        if kind in _MOVING_JOINT_TYPES:
            if not isinstance(joint.get("name"), str):
                return None
            path.append(joint.get("name"))
            if joint.get("name") == target:
                return path
        elif kind != "fixed":
            return None
        current = joint.get("parent")
    return None


def manipulation_tasks_check(blueprint: dict[str, Any]) -> GateResult:
    tasks = blueprint.get("tasks")
    if tasks is None:
        return GateResult.ok("manipulation_tasks", "no tasks declared")
    if not isinstance(tasks, list) or not tasks:
        return GateResult(
            name="manipulation_tasks",
            passed=False,
            summary="tasks must be a non-empty list",
            failures=["`tasks` must be a non-empty list of task sentences"],
        )
    part_names = {
        p["name"]
        for p in (blueprint.get("parts") or [])
        if isinstance(p, dict) and isinstance(p.get("name"), str)
    }
    failures: list[str] = []
    for i, t in enumerate(tasks):
        tid = f"tasks[{i}]"
        if isinstance(t, str):
            if not t.strip():
                failures.append(f"{tid}: empty task sentence")
        elif isinstance(t, dict):
            instr = t.get("instruction")
            if not isinstance(instr, str) or not instr.strip():
                failures.append(f"{tid}: object task needs a non-empty 'instruction'")
            operates = t.get("operates")
            if operates is not None:
                if not isinstance(operates, str) or not operates.strip():
                    failures.append(f"{tid}: operates must be a non-empty part name")
                elif operates not in part_names:
                    failures.append(f"{tid}: operates {operates!r} is not a declared part")
        else:
            failures.append(f"{tid}: must be a sentence or {{instruction, operates?}}")
    if failures:
        return GateResult(
            name="manipulation_tasks",
            passed=False,
            summary=f"{len(failures)} task violation(s)",
            failures=failures,
        )
    return GateResult.ok("manipulation_tasks", f"{len(tasks)} task(s) valid")


def manipulation_binding_check(
    tasks_json: Any,
    parts: list[str],
    joints: list[dict[str, Any]],
    *,
    visuals_by_part: dict[str, list[str | None]] | None = None,
    couplings: list[dict[str, Any]] | None = None,
) -> GateResult:
    if tasks_json is None:
        return GateResult.ok("manipulation_binding", "no tasks.json")
    if not isinstance(tasks_json, dict):
        return GateResult(
            name="manipulation_binding",
            passed=False,
            summary="tasks.json must be a JSON object",
            failures=["tasks.json must be a JSON object"],
        )
    failures: list[str] = []
    if tasks_json.get("schema_version") != "2.0":
        failures.append("tasks.json schema_version must equal '2.0'")
    tasks = tasks_json.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        failures.append("tasks.json tasks must be a non-empty list")
        return GateResult(
            name="manipulation_binding",
            passed=False,
            summary=f"{len(failures)} binding violation(s)",
            failures=failures,
        )
    link_names = set(parts or [])
    visual_inventory = visuals_by_part or {}
    joints = joints or []
    by_name = {j.get("name"): j for j in joints if isinstance(j, dict)}
    movable = {
        j.get("name")
        for j in joints
        if str(j.get("type") or j.get(JOINT_PROP_TYPE) or "") in _MOVING_JOINT_TYPES
    }
    seen_task_ids: set[str] = set()
    for i, t in enumerate(tasks):
        if not isinstance(t, dict):
            failures.append(f"tasks[{i}]: task must be an object")
            continue
        raw_task_id = t.get("task_id")
        tid = raw_task_id if isinstance(raw_task_id, str) and raw_task_id.strip() else f"tasks[{i}]"
        if not isinstance(raw_task_id, str) or not raw_task_id.strip():
            failures.append(f"tasks[{i}]: task_id must be a non-empty string")
        elif raw_task_id in seen_task_ids:
            failures.append(f"{tid}: duplicate task_id")
        else:
            seen_task_ids.add(raw_task_id)
        instruction = t.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            failures.append(f"{tid}: instruction must be a non-empty string")
        for legacy_field in (
            "contactable_part_link",
            "contactable_visual",
            "contactable_visuals",
        ):
            if legacy_field in t:
                failures.append(f"{tid}: legacy field {legacy_field!r} is not valid in schema v2")

        contact_target = t.get("contact_target")
        link: Any = None
        requested_visuals: Any = None
        if not isinstance(contact_target, dict):
            failures.append(f"{tid}: contact_target must be an object with link and visuals")
        else:
            if set(contact_target) != {"link", "visuals"}:
                failures.append(f"{tid}: contact_target must contain exactly link and visuals")
            link = contact_target.get("link")
            requested_visuals = contact_target.get("visuals")

        link_is_valid = isinstance(link, str) and bool(link.strip()) and link in link_names
        if not link_is_valid:
            failures.append(f"{tid}: contact_target.link {link!r} is not a built link")

        valid_visual_names: list[str] = []
        if not isinstance(requested_visuals, list) or not requested_visuals:
            failures.append(f"{tid}: contact_target.visuals must be a non-empty list")
        else:
            seen_visuals: set[str] = set()
            for visual in requested_visuals:
                if not isinstance(visual, str) or not visual.strip():
                    failures.append(
                        f"{tid}: contact_target.visuals entries must be non-empty strings"
                    )
                    continue
                if visual in seen_visuals:
                    failures.append(f"{tid}: duplicate contact visual {visual!r}")
                    continue
                seen_visuals.add(visual)
                valid_visual_names.append(visual)

        if link_is_valid:
            available_visuals = visual_inventory.get(link)
            if not isinstance(available_visuals, list):
                failures.append(f"{tid}: visual inventory is unavailable for link {link!r}")
            else:
                for visual in valid_visual_names:
                    matches = sum(candidate == visual for candidate in available_visuals)
                    if matches == 0:
                        failures.append(
                            f"{tid}: contact visual {visual!r} does not exist on link {link!r}"
                        )
                    elif matches > 1:
                        failures.append(
                            f"{tid}: contact visual {visual!r} is ambiguous on link {link!r} "
                            f"({matches} matches)"
                        )
        walked = _walk_to_actuated_joint(link, joints) if isinstance(link, str) else None
        raw_tj = t.get("target_joint")
        whole_body = raw_tj == WHOLE_BODY_TARGET
        tj = raw_tj if whole_body else (raw_tj or walked)
        if whole_body:
            if WHOLE_BODY_TARGET in by_name:
                failures.append(
                    f"{tid}: {WHOLE_BODY_TARGET!r} is reserved for whole-body tasks but the "
                    f"model builds a joint with that name; rename the joint"
                )
            elif walked is not None:
                failures.append(
                    f"{tid}: contact target link {link!r} reaches movable joint {walked!r} "
                    f"via fixed joints, so this task moves that joint, not the whole body"
                )
        elif not isinstance(tj, str) or tj not in by_name:
            hint = (
                f"; if this task moves the whole object rather than a joint, set "
                f"target_joint to {WHOLE_BODY_TARGET!r} with task_intent 'push_whole_body'"
                if link_is_valid and walked is None
                else ""
            )
            failures.append(f"{tid}: target_joint {tj!r} is not a built joint{hint}")
        elif tj not in movable:
            failures.append(f"{tid}: target_joint {tj!r} is not movable")
        elif "articulated_path" in t:
            declared = t["articulated_path"]
            resolved = _moving_ancestor_path(link, tj, joints) if link_is_valid else None
            if (
                not isinstance(declared, list)
                or len(declared) < 2
                or not all(isinstance(name, str) for name in declared)
                or declared != resolved
                or raw_tj != tj
            ):
                failures.append(
                    f"{tid}: articulated_path must explicitly match all movable joints "
                    f"from contact link to target_joint, nearest first; resolved={resolved!r}"
                )
        elif walked is not None and walked != tj:
            failures.append(
                f"{tid}: contact target link {link!r} reaches {walked!r} via fixed joints, "
                f"not target_joint {tj!r}"
            )
        if whole_body and "articulated_path" in t:
            failures.append(f"{tid}: articulated_path is not valid for whole-body tasks")
        succ = t.get("success")
        if not isinstance(succ, dict):
            failures.append(f"{tid}: success is required ({{joint, to_fraction}})")
        else:
            if succ.get("joint") != tj:
                failures.append(
                    f"{tid}: success.joint {succ.get('joint')!r} must equal target_joint {tj!r}"
                )
            joint = by_name.get(tj, {}) if tj in by_name else {}
            jtype = (
                WHOLE_BODY_JOINT_TYPE
                if whole_body
                else str(joint.get("type") or joint.get(JOINT_PROP_TYPE) or "")
            )
            if whole_body:
                to_value = succ.get("to_value")
                if (
                    not isinstance(to_value, (int, float))
                    or isinstance(to_value, bool)
                    or not math.isfinite(to_value)
                    or to_value <= 0.0
                ):
                    failures.append(
                        f"{tid}: whole-body success needs a positive finite success.to_value "
                        f"in metres of root displacement, not a to_fraction"
                    )
            elif jtype == "continuous":
                to_value = succ.get("to_value")
                if (
                    not isinstance(to_value, (int, float))
                    or isinstance(to_value, bool)
                    or not math.isfinite(to_value)
                ):
                    failures.append(
                        f"{tid}: continuous joint needs a finite numeric success.to_value"
                    )
            else:
                tf = succ.get("to_fraction")
                if (
                    not isinstance(tf, (int, float))
                    or isinstance(tf, bool)
                    or not math.isfinite(tf)
                    or not (0.0 < tf <= 1.0)
                ):
                    failures.append(f"{tid}: success.to_fraction must be in (0, 1]")
        intent = t.get("task_intent")
        if not isinstance(intent, str) or intent not in MANIP_INTENTS:
            failures.append(
                f"{tid}: task_intent {intent!r} not in the documented set {sorted(MANIP_INTENTS)}"
            )
        elif whole_body or (isinstance(tj, str) and tj in by_name):
            jtype = (
                WHOLE_BODY_JOINT_TYPE
                if whole_body
                else str(by_name[tj].get("type") or by_name[tj].get(JOINT_PROP_TYPE) or "")
            )
            allowed_types = _MANIP_INTENT_JOINT_TYPES[intent]
            if jtype not in allowed_types:
                failures.append(
                    f"{tid}: task_intent {intent!r} needs joint type "
                    f"{sorted(allowed_types)}, got {jtype!r} on {tj!r}"
                )
            if intent == "press_pedal_coupled" and couplings is not None and not whole_body:
                declared_drivers = {
                    coupling.get("driver") for coupling in couplings if isinstance(coupling, dict)
                }
                if tj not in declared_drivers:
                    failures.append(
                        f"{tid}: press_pedal_coupled target_joint {tj!r} is not a declared "
                        "coupling driver"
                    )
    if failures:
        return GateResult(
            name="manipulation_binding",
            passed=False,
            summary=f"{len(failures)} binding violation(s)",
            failures=failures,
        )
    return GateResult.ok("manipulation_binding", f"{len(tasks)} task binding(s) valid")
