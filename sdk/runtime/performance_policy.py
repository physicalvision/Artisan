from __future__ import annotations

import os
from dataclasses import dataclass


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name, "1" if default else "0").strip().lower()
    if value not in {"0", "1", "false", "true"}:
        raise ValueError(f"{name} must be 0 or 1")
    return value in {"1", "true"}


@dataclass(frozen=True)
class PerformancePolicy:
    reuse_diagnostic_geometry: bool = True
    reuse_render_model: bool = True
    shared_compile_diagnostics: bool = False
    frozen_inspection: bool = False

    @classmethod
    def from_env(cls) -> PerformancePolicy:
        return cls(
            reuse_diagnostic_geometry=_flag("ARTISAN_REUSE_DIAGNOSTIC_GEOMETRY", True),
            reuse_render_model=_flag("ARTISAN_REUSE_RENDER_MODEL", True),
            shared_compile_diagnostics=_flag("ARTISAN_SHARED_COMPILE_DIAGNOSTICS", False),
            frozen_inspection=_flag("ARTISAN_FROZEN_INSPECTION", False),
        )
