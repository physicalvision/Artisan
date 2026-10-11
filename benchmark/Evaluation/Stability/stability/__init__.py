from __future__ import annotations

from pathlib import Path
from typing import Any

from .loader import load_urdf
from .probes import run_configurations


def evaluate_asset(path: Path) -> dict[str, Any]:
    cells = run_configurations(load_urdf(path))
    return {
        name: cell["status"] == "pass" if cell["status"] in {"pass", "fail"} else None
        for name, cell in cells.items()
    }
