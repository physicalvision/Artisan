from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StabilityPolicy:
    timestep_s: float = 0.002
    iterations: int = 100
    tolerance: float = 1e-08
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    ground_friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    asset_geom_friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    contact_solref: tuple[float, float] = (0.02, 1.0)
    contact_solimp: tuple[float, float, float, float, float] = (0.9, 0.95, 0.001, 0.5, 2.0)
    condim: int = 3
    contact_margin: float = 0.0
    contact_gap: float = 0.0
    geom_priority: int = 0
    geom_solmix: float = 1.0
    eq_solref: tuple[float, float] = (0.02, 1.0)
    eq_solimp: tuple[float, float, float, float, float] = (0.9, 0.95, 0.001, 0.5, 2.0)
    eq_tracking_tol_rad: float = 0.02
    eq_tracking_tol_m: float = 0.002
    max_abs_qvel: float = 100.0
    max_abs_qacc: float = 100000.0
    settle_seconds: float = 5.0
    test_seconds: float = 5.0
    max_pos_dev_m: float = 0.001
    max_rot_dev_rad: float = 0.01
    tipped_swing_deg: float = 30.0
    max_ground_penetration_m: float = 0.005
    limit_inset_fraction: float = 0.01
    placement_clearance_m: float = 0.001
