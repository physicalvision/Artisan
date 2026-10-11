from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

BLENDER_VERSION = (5, 0, 0)
BLENDER_BUILD_HASH = "a37564c4df7a"
BLENDER_BIN = "blender"
PROTOCOL_VERSION = "fidelity/3"


@lru_cache(maxsize=None)
def checked_blender(path: str) -> str:
    output = subprocess.check_output([path, "--version"], text=True)
    if (
        output.splitlines()[0] != "Blender 5.0.0"
        or f"build hash: {BLENDER_BUILD_HASH}" not in output
    ):
        raise RuntimeError(f"Fidelity requires Blender 5.0.0 build {BLENDER_BUILD_HASH}: {path}")
    return path


def blender_bin() -> str:
    return checked_blender(
        str(Path(os.environ.get("ARTISAN_BLENDER_BIN", BLENDER_BIN)).expanduser())
    )


def assert_blender_version(bpy) -> None:
    if bpy.app.version != BLENDER_VERSION or bpy.app.build_hash.decode() != BLENDER_BUILD_HASH:
        raise RuntimeError(f"Fidelity requires Blender 5.0.0 build {BLENDER_BUILD_HASH}")
