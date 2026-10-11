from __future__ import annotations

import os
import re
from pathlib import Path

from artisan.config import apply_runtime_config, load_config
from pipeline.llm.model_registry import DEFAULT_TEXT_MODEL


def dataset_cases(data_root: Path, dataset: str) -> list[str]:
    root = data_root / dataset
    if not root.is_dir():
        raise FileNotFoundError(root)
    if dataset == "Stability":
        names = [p.stem for p in root.glob("*.txt") if p.is_file() and not p.name.startswith(".")]
        return sorted(
            names, key=lambda name: (not name.isdecimal(), int(name) if name.isdecimal() else name)
        )
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


def default_method(model: str) -> str:
    words = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip(".-").split("-")
    return "Artisan_" + "-".join(
        word.upper() if word.lower() == "gpt" else word.capitalize() for word in words
    )


def results_directory(
    project: Path, dataset: str, method: str | None, outputs_root: Path | None
) -> Path:
    if method is None:
        configured = os.environ.get("ARTISAN_CONFIG", "").strip()
        path = Path(configured).expanduser() if configured else project / ".artisan/config.toml"
        config = load_config(path) if configured or path.is_file() else {}
        apply_runtime_config(config, {})
        model = os.environ.get("ARTISAN_PIPELINE_MODEL_CODING") or os.environ.get(
            "ARTISAN_PIPELINE_MODEL", DEFAULT_TEXT_MODEL
        )
        method = default_method(model)
    if (
        method in {"", ".", ".."}
        or method.startswith(".")
        or any(char in method for char in "/\\\n\r\0")
    ):
        raise ValueError("Method must be a non-hidden directory name without separators")
    return (outputs_root or project / "outputs").expanduser().absolute() / dataset / method
