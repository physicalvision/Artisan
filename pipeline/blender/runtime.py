from __future__ import annotations

import os
import shutil
from pathlib import Path


def blender_bin() -> str:
    configured = os.environ.get("ARTISAN_BLENDER_BIN", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.absolute())
        raise FileNotFoundError("ARTISAN_BLENDER_BIN must name an executable Blender binary.")
    found = shutil.which("blender")
    if found:
        return found
    raise FileNotFoundError(
        "Blender was not found. Set ARTISAN_BLENDER_BIN or add Blender to PATH."
    )
