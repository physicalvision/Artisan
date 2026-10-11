from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path
from typing import Any

_ENVIRONMENT_SETTINGS = {
    "provider": "ARTISAN_PIPELINE_PROVIDER",
    "openai_base_url": "OPENAI_BASE_URL",
    "anthropic_base_url": "ANTHROPIC_BASE_URL",
    "openrouter_base_url": "OPENROUTER_BASE_URL",
    "gemini_base_url": "GEMINI_BASE_URL",
    "image_base_url": "ARTISAN_IMAGE_BASE_URL",
    "model": "ARTISAN_PIPELINE_MODEL",
    "thinking": "ARTISAN_PIPELINE_THINKING",
    "image_model": "ARTISAN_IMAGE_MODEL",
    "image_tool_model": "ARTISAN_IMAGE_TOOL_MODEL",
    "compaction_model": "OPENAI_COMPACTION_MODEL",
    "compaction_trigger_tokens": "OPENAI_COMPACTION_COST_TRIGGER_TOKENS",
    "blender": "ARTISAN_BLENDER_BIN",
    "render_device": "ARTISAN_RENDER_DEVICE",
    "cache_dir": "ARTISAN_CACHE_DIR",
}
THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_PATH_SETTINGS = {"workspace", "cache_dir", "blender", "credentials_file"}
_CONFIG_SETTINGS = set(_ENVIRONMENT_SETTINGS) | {
    "workspace",
    "max_cost_usd",
    "credentials_file",
    "cpu_threads",
}


def default_config_path() -> Path:
    configured = os.environ.get("ARTISAN_CONFIG", "").strip()
    if configured:
        return Path(configured).expanduser().absolute()
    return Path.cwd() / ".artisan" / "config.toml"


def default_cost_limit(model: str) -> float:
    return 15.0 if model.rsplit("/", 1)[-1].lower() == "gpt-6-astra" else 10.0


def initialize_config(
    path: Path | None = None,
    *,
    workspace: Path | None = None,
    blender: str | None = None,
    credentials: Path | None = None,
) -> dict[str, Any]:
    destination = (path or default_config_path()).expanduser().absolute()
    if destination.exists():
        return {
            "config": str(destination),
            "created": False,
            "message": "Existing configuration preserved",
        }
    from pipeline.llm.model_registry import (
        DEFAULT_COMPACTION_MODEL,
        DEFAULT_IMAGE_TOOL_MODEL,
        DEFAULT_TEXT_MODEL,
    )

    workspace = (workspace or destination.parent / "workspace").expanduser().absolute()
    credentials = (credentials or destination.parent / "credentials.env").expanduser().absolute()
    credentials.parent.mkdir(parents=True, exist_ok=True)
    if not credentials.exists():
        fd = os.open(credentials, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write("OPENAI_API_KEY=\n")
    settings: dict[str, Any] = {
        "workspace": str(workspace),
        "credentials_file": str(credentials),
        "model": DEFAULT_TEXT_MODEL,
        "thinking": "high",
        "max_cost_usd": default_cost_limit(DEFAULT_TEXT_MODEL),
        "compaction_model": DEFAULT_COMPACTION_MODEL,
        "compaction_trigger_tokens": 110000,
        "image_tool_model": DEFAULT_IMAGE_TOOL_MODEL,
        "cpu_threads": 1,
        "render_device": "cpu",
        "cache_dir": str(destination.parent / "cache"),
    }
    if blender:
        settings["blender"] = str(Path(blender).expanduser().absolute())
    contents = "[artisan]\n" + "".join(
        f"{key} = {json.dumps(value)}\n" for key, value in settings.items()
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(contents)
    return {"config": str(destination), "created": True, "credentials_file": str(credentials)}


def cache_dir() -> Path:
    configured = os.environ.get("ARTISAN_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().absolute()
    return default_config_path().parent / "cache"


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        path = default_config_path()
        if not path.exists() and not os.environ.get("ARTISAN_CONFIG", "").strip():
            return {}
    config_path = path.expanduser().absolute()
    with config_path.open("rb") as stream:
        document = tomllib.load(stream)
    section = document.get("artisan", document)
    if not isinstance(section, dict):
        raise ValueError("Configuration must contain an artisan table")
    unknown = set(section) - _CONFIG_SETTINGS
    if unknown:
        raise ValueError("Unknown Artisan configuration settings: " + ", ".join(sorted(unknown)))
    result = dict(section)
    for key in _PATH_SETTINGS & result.keys():
        raw = result[key]
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"{key} must be a non-empty path string")
        value = Path(raw).expanduser()
        result[key] = str(value if value.is_absolute() else config_path.parent / value)
    for key, minimum in (("cpu_threads", 1), ("compaction_trigger_tokens", 0)):
        if key in result and (type(result[key]) is not int or result[key] < minimum):
            raise ValueError(f"{key} must be an integer greater than or equal to {minimum}")
    if "render_device" in result and (
        not isinstance(result["render_device"], str)
        or result["render_device"] not in {"cpu", "gpu", "auto"}
    ):
        raise ValueError("render_device must be cpu, gpu, or auto")
    return result


def apply_runtime_config(config: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, env_name in _ENVIRONMENT_SETTINGS.items():
        if key in config and config[key] is not None:
            os.environ.setdefault(env_name, str(config[key]))
        if overrides.get(key) is not None:
            os.environ[env_name] = str(overrides[key])
            if key in {"provider", "model", "thinking"}:
                for role in ("DESIGN", "CODING", "REVIEW"):
                    os.environ.pop(f"{env_name}_{role}", None)
    if "cpu_threads" in config:
        for name in THREAD_ENVIRONMENT_VARIABLES:
            os.environ.setdefault(name, str(config["cpu_threads"]))
    os.environ.setdefault("ARTISAN_PIPELINE_PROVIDER", "openai")
    os.environ.setdefault("OPENAI_COMPACTION_COST_TRIGGER_TOKENS", "110000")


def load_credentials(path: Path | None) -> None:
    if path is None:
        return
    from dotenv import dotenv_values

    allowed = {
        "OPENAI_API_KEY",
        "OPENAI_API_KEYS",
        "ARTISAN_IMAGE_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_API_KEYS",
        "OPENROUTER_API_KEY",
        "OPENROUTER_API_KEYS",
        "GEMINI_API_KEY",
        "GEMINI_API_KEYS",
        "GOOGLE_API_KEY",
        "GOOGLE_API_KEYS",
    }
    if not path.expanduser().is_file():
        raise FileNotFoundError(f"Credentials file does not exist: {path}")
    for key, value in dotenv_values(path.expanduser(), interpolate=False).items():
        if key in allowed and value and not os.environ.get(key, "").strip():
            os.environ[key] = value
