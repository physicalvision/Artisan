from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def load_design_prompt(name: str) -> str:
    """Read a DesignAgent prompt asset by file name (e.g. "system.md",
    "blueprint.template.json"). Cached: these files are static per process."""
    path = _DIR / name
    if not path.exists():
        raise FileNotFoundError(f"design prompt asset not found: {name} ({path})")
    return path.read_text(encoding="utf-8")
