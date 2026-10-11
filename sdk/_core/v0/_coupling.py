
from __future__ import annotations

import math

from .errors import ValidationError
from .types import ArticulationType

CONTINUOUS_REFERENCE = 2.0 * math.pi

POLY_DEGREE = 4
_FIT_SAMPLES = 96


def _window(joint: object) -> tuple[float, float] | None:
    ml = getattr(joint, "motion_limits", None)
    lo = getattr(ml, "lower", None) if ml is not None else None
    hi = getattr(ml, "upper", None) if ml is not None else None
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and float(hi) > float(lo):
        return float(lo), float(hi)
    return None


def driver_pose_for_follower(model: object, follower_name: str, value: float) -> float | None:
    follower = None
    for j in getattr(model, "articulations", None) or []:
        if getattr(j, "name", None) == follower_name:
            follower = j
            break
    if follower is None:
        return None
    mimic = getattr(follower, "mimic", None)
    if mimic is None:
        return None
    driver_name = str(getattr(mimic, "joint", "")).strip()
    poly = getattr(mimic, "poly", ()) or ()
    q_driver: float | None
    if poly:
        win = _coupling_meta_window(follower) or _driver_fit_window(model, driver_name)
        if win is None:
            return None
        lo, hi = win
        ptuple = tuple(float(c) for c in poly)
        flo, fhi = evaluate_poly(ptuple, lo), evaluate_poly(ptuple, hi)
        increasing = fhi >= flo
        target = min(max(float(value), min(flo, fhi)), max(flo, fhi))
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if (evaluate_poly(ptuple, mid) < target) == increasing:
                lo = mid
            else:
                hi = mid
        q_driver = 0.5 * (lo + hi)
    else:
        mult = float(getattr(mimic, "multiplier", 1.0) or 0.0)
        if mult == 0.0 or not math.isfinite(mult):
            return None
        off = float(getattr(mimic, "offset", 0.0) or 0.0)
        q_driver = (float(value) - off) / mult
    driver_art = next(
        (
            j
            for j in getattr(model, "articulations", None) or []
            if getattr(j, "name", None) == driver_name
        ),
        None,
    )
    if driver_art is not None:
        rng = effective_driver_range(model, driver_art)
        if rng is not None:
            q_driver = min(max(q_driver, rng[0]), rng[1])
    return q_driver


def driver_reference(joint: object) -> float | None:
    if getattr(joint, "articulation_type", None) == ArticulationType.CONTINUOUS:
        return CONTINUOUS_REFERENCE
    win = _window(joint)
    return None if win is None else win[1] - win[0]


def full_multiplier_magnitude(follower: object, driver: object) -> float | None:
    fwin = _window(follower)
    ref = driver_reference(driver)
    if fwin is None or ref is None or ref == 0.0:
        return None
    return (fwin[1] - fwin[0]) / ref


def turns_to_multiplier(follower: object, turns: float) -> float | None:
    fwin = _window(follower)
    if fwin is None or not (isinstance(turns, (int, float)) and turns > 0):
        return None
    return (fwin[1] - fwin[0]) / (float(turns) * CONTINUOUS_REFERENCE)


def _driver_fit_window(model: object, driver_name: str) -> tuple[float, float] | None:
    for j in getattr(model, "articulations", None) or []:
        if getattr(j, "name", None) != driver_name:
            continue
        w = _window(j)
        if w is not None:
            return w
        if getattr(j, "articulation_type", None) == ArticulationType.CONTINUOUS:
            return (0.0, CONTINUOUS_REFERENCE)
        return None
    return None


def _coupling_meta_window(follower: object) -> tuple[float, float] | None:
    meta = getattr(follower, "meta", None)
    if not isinstance(meta, dict):
        return None
    cp = meta.get("coupling")
    if not isinstance(cp, dict):
        return None
    win = cp.get("window")
    if (
        isinstance(win, (list, tuple))
        and len(win) == 2
        and all(isinstance(x, (int, float)) for x in win)
        and float(win[1]) > float(win[0])
    ):
        return float(win[0]), float(win[1])
    return None


def _poly_driver_bound(
    poly: tuple[float, ...],
    follower_window: tuple[float, float],
    driver_window: tuple[float, float],
) -> tuple[float, float] | None:
    f_lo, f_hi = follower_window
    d_lo, d_hi = driver_window
    tol = 1e-6 + 1e-3 * abs(f_hi - f_lo)

    def inside(q: float) -> bool:
        v = evaluate_poly(poly, q)
        return f_lo - tol <= v <= f_hi + tol

    rest = min(max(0.0, d_lo), d_hi)
    if not inside(rest):
        return None

    def crossing(inside_q: float, outside_q: float) -> float:
        for _ in range(40):
            mid = 0.5 * (inside_q + outside_q)
            if inside(mid):
                inside_q = mid
            else:
                outside_q = mid
            if abs(outside_q - inside_q) < 1e-4:
                break
        return inside_q

    def walk(target: float) -> float:
        if target == rest:
            return rest
        n = 64
        prev = rest
        for k in range(1, n + 1):
            q = rest + (target - rest) * k / n
            if not inside(q):
                return crossing(prev, q)
            prev = q
        return target

    upper = walk(d_hi)
    lower = walk(d_lo)
    if upper >= d_hi - 1e-9 and lower <= d_lo + 1e-9:
        return None
    if not (upper > lower):
        return None
    return lower, upper


def follower_bound_range(model: object, driver_name: str) -> tuple[float, float] | None:
    lo_bound = -math.inf
    hi_bound = math.inf
    found = False
    for j in getattr(model, "articulations", None) or []:
        mimic = getattr(j, "mimic", None)
        if mimic is None or str(getattr(mimic, "joint", "")) != driver_name:
            continue
        win = _window(j)
        if win is None:
            continue
        poly = getattr(mimic, "poly", ())
        if poly:
            dw = _coupling_meta_window(j) or _driver_fit_window(model, driver_name)
            if dw is None:
                continue
            bnd = _poly_driver_bound(tuple(float(c) for c in poly), win, dw)
            if bnd is None:
                continue
            d_lo, d_hi = bnd
        else:
            mult = float(getattr(mimic, "multiplier", 1.0) or 0.0)
            if mult == 0.0 or not math.isfinite(mult):
                continue
            off = float(getattr(mimic, "offset", 0.0) or 0.0)
            a, b = (win[0] - off) / mult, (win[1] - off) / mult
            d_lo, d_hi = (a, b) if a <= b else (b, a)
        lo_bound, hi_bound = max(lo_bound, d_lo), min(hi_bound, d_hi)
        found = True
    if not found or not (hi_bound > lo_bound):
        return None
    return lo_bound, hi_bound


_BACK_UNSET = object()


def effective_driver_range(
    model: object, joint: object, *, back: object = _BACK_UNSET
) -> tuple[float, float] | None:
    name = getattr(joint, "name", None)
    if back is _BACK_UNSET:
        back = follower_bound_range(model, name) if isinstance(name, str) else None
    own = _window(joint)
    if own is None and back is None:
        return None
    if own is None:
        return back
    if back is None:
        return own
    lo, hi = max(own[0], back[0]), min(own[1], back[1])
    return (lo, hi) if hi > lo else own


def driver_effective_ranges(model: object) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for art in getattr(model, "articulations", None) or []:
        name = getattr(art, "name", None)
        if not isinstance(name, str):
            continue
        back = follower_bound_range(model, name)
        if back is None:
            continue
        eff = effective_driver_range(model, art, back=back)
        if eff is not None:
            out[name] = eff
    return out


NONLINEAR_TEMPLATES = ("crank_slider", "cam", "scotch_yoke")


def evaluate_poly(poly: tuple[float, ...], q: float) -> float:
    out = 0.0
    p = 1.0
    for c in poly:
        out += c * p
        p *= q
    return out


def _fit_through_origin(thetas, values, degree: int = POLY_DEGREE) -> tuple[float, ...]:
    import numpy as np

    t = np.asarray(thetas, dtype=float)
    v = np.asarray(values, dtype=float)
    cols = np.vstack([t**k for k in range(1, degree + 1)]).T
    coef, *_ = np.linalg.lstsq(cols, v, rcond=None)
    poly = [0.0, *(float(c) for c in coef)]
    while len(poly) > 2 and abs(poly[-1]) < 1e-12:
        poly.pop()
    return tuple(poly)


def _scotch_yoke_disp(theta, radius: float):
    import numpy as np

    return radius * np.sin(np.asarray(theta, dtype=float))


def _crank_slider_disp(theta, crank_radius: float, rod_length: float):
    import numpy as np

    t = np.asarray(theta, dtype=float)
    s = crank_radius * np.cos(t) + np.sqrt(rod_length**2 - (crank_radius * np.sin(t)) ** 2)
    return s - (crank_radius + rod_length)


def _cam_disp(theta, lift: float, rise_angle: float):
    import numpy as np

    t = np.asarray(theta, dtype=float)
    out = np.empty_like(t)
    for i, th in np.ndenumerate(t):
        if th <= 0.0:
            out[i] = 0.0
        elif th < rise_angle:
            out[i] = lift * (
                th / rise_angle - math.sin(2 * math.pi * th / rise_angle) / (2 * math.pi)
            )
        else:
            out[i] = lift
    return out


def _template_disp(template: str, theta, **params: float):
    if template == "scotch_yoke":
        radius = float(params.get("radius", 0.0))
        if radius <= 0:
            raise ValidationError("scotch_yoke needs radius > 0")
        return _scotch_yoke_disp(theta, radius)
    if template == "crank_slider":
        r = float(params.get("crank_radius", 0.0))
        rod = float(params.get("rod_length", 0.0))
        if r <= 0 or rod <= 0:
            raise ValidationError("crank_slider needs crank_radius > 0 and rod_length > 0")
        if rod <= r:
            raise ValidationError(
                "crank_slider needs rod_length > crank_radius (else the rod binds)"
            )
        return _crank_slider_disp(theta, r, rod)
    if template == "cam":
        lift = float(params.get("lift", 0.0))
        rise = float(params.get("rise_angle", 0.0))
        if lift <= 0 or rise <= 0:
            raise ValidationError("cam needs lift > 0, rise_angle > 0")
        return _cam_disp(theta, lift, rise)
    raise ValidationError(
        f"unknown nonlinear template {template!r} (use one of {NONLINEAR_TEMPLATES})"
    )


def template_monotone_window(template: str, **params: float) -> tuple[float, float]:
    if template == "scotch_yoke":
        return (-math.pi / 2.0, math.pi / 2.0)
    if template == "crank_slider":
        return (0.0, math.pi)
    if template == "cam":
        rise = float(params.get("rise_angle", 0.0))
        if rise <= 0:
            raise ValidationError("cam needs rise_angle > 0")
        return (0.0, rise)
    raise ValidationError(
        f"unknown nonlinear template {template!r} (use one of {NONLINEAR_TEMPLATES})"
    )


def secant_multiplier(poly: tuple[float, ...], window: tuple[float, float]) -> float:
    import numpy as np

    lo, hi = float(window[0]), float(window[1])
    thetas = np.linspace(lo, hi, _FIT_SAMPLES)
    vals = np.array([evaluate_poly(poly, float(t)) for t in thetas])
    idx = int(np.argmax(np.abs(vals)))
    theta_star = float(thetas[idx])
    if abs(theta_star) < 1e-9:
        return poly[1] if len(poly) > 1 else 0.0
    return float(vals[idx]) / theta_star


def template_fit_error(
    template: str, window: tuple[float, float], poly: tuple[float, ...], **params: float
) -> float:
    import numpy as np

    lo, hi = float(window[0]), float(window[1])
    thetas = np.linspace(lo, hi, _FIT_SAMPLES)
    true = np.asarray(_template_disp(template, thetas, **params), dtype=float)
    approx = np.array([evaluate_poly(poly, float(t)) for t in thetas])
    span = float(np.max(true) - np.min(true))
    if span <= 1e-12:
        return 0.0
    return float(np.max(np.abs(true - approx)) / span)


def nonlinear_template_poly(
    template: str,
    *,
    driver_window: tuple[float, float],
    sense: str = "same",
    **params: float,
) -> tuple[float, ...]:
    import numpy as np

    lo, hi = float(driver_window[0]), float(driver_window[1])
    if not (hi > lo):
        raise ValidationError(
            f"nonlinear coupling needs a finite driver window, got {driver_window!r}"
        )
    if sense not in ("same", "opposite"):
        raise ValidationError(f"sense must be 'same' or 'opposite', got {sense!r}")
    thetas = np.linspace(lo, hi, _FIT_SAMPLES)
    values = _template_disp(template, thetas, **params)
    poly = _fit_through_origin(thetas, values)
    if sense == "opposite":
        poly = tuple(-c for c in poly)
    return poly
