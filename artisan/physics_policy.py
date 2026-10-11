from __future__ import annotations

import math
import os

DEFAULT_PENETRATION_DEPTH_TOL_M = 2e-4


def validate_penetration_depth_tolerance(
    value: float | str,
    *,
    name: str = "penetration_depth_tol_m",
) -> float:
    message = f"{name} must be a finite, non-negative number in metres"
    try:
        threshold = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError(message)
    return threshold


def penetration_depth_tolerance_from_env() -> float:
    name = "ARTISAN_PENETRATION_DEPTH_TOL"
    return validate_penetration_depth_tolerance(
        os.environ.get(name, str(DEFAULT_PENETRATION_DEPTH_TOL_M)),
        name=name,
    )
