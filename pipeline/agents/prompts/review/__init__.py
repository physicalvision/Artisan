from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def load_review_prompt(name: str) -> str:
    """Read a prompt by filename and cache its static contents for this process."""
    path = _DIR / name
    if not path.exists():
        raise FileNotFoundError(f"review prompt asset not found: {name} ({path})")
    return path.read_text(encoding="utf-8")
