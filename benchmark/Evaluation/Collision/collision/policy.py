from __future__ import annotations

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class BenchPolicy:
    epsilon_m: float = 0.001
    tau_float_m: float = 0.02
    primitive_chord_tolerance_m: float = 0.0005
    local_window_ratio: float = 0.05
    separation_resolution_m: float = 0.00002
    separation_scan_steps: int = 16
    max_contacts: int = 4096
    containment_points: int = 5
    containment_winding_tolerance: float = 0.005
    containment_self_tolerance: float = 0.02
    containment_offset_ratio: float = 0.0001
    containment_segment_tolerance_m: float = 1e-9
    locked_range_tolerance: float = 1e-10
    motion_translation_tolerance_m: float = 1e-8
    motion_rotation_tolerance_rad: float = 1e-8
    l1_cells: int = 9
    l2_cells_per_axis: int = 9
    lmulti_sobol_cells: int = 200
    lmulti_diagonal_cells: int = 5
    random_seed: int = 20260802

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if not math.isfinite(value) or value < 0 or (value == 0 and item.name != "random_seed"):
                raise ValueError(f"Invalid finite positive value for {item.name}")
            if item.type == "int" and not isinstance(value, int):
                raise ValueError(f"{item.name} must be an integer")
        for name in (
            "separation_scan_steps",
            "l1_cells",
            "l2_cells_per_axis",
            "lmulti_diagonal_cells",
        ):
            if getattr(self, name) < 2:
                raise ValueError(f"{name} must be at least two")

    def effective_epsilon(self, scene_diagonal: float) -> float:
        return self.epsilon_m

    def effective_tau_float(self, scene_diagonal: float) -> float:
        return self.tau_float_m
