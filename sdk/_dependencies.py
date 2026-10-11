from __future__ import annotations

import importlib
from typing import Any


def _install_command_hint(*, extra: str | None = None) -> str:
    return "Install the dependencies listed in the Artisan installation guide, then retry."


def _missing_dependency_message(*, feature: str, dependency: str, extra: str | None = None) -> str:
    return (
        f"{feature} requires the `{dependency}` package, but it is not installed in the current "
        f"environment. {_install_command_hint(extra=extra)}"
    )


def require_cadquery(*, feature: str) -> Any:
    try:
        return importlib.import_module("cadquery")
    except Exception as exc:
        raise RuntimeError(
            _missing_dependency_message(feature=feature, dependency="cadquery")
        ) from exc


def _require_sim_dependency(*, feature: str, dependency: str, module: str) -> Any:
    try:
        return importlib.import_module(module)
    except Exception as exc:
        raise RuntimeError(
            _missing_dependency_message(feature=feature, dependency=dependency, extra="sim")
        ) from exc


def require_mujoco(*, feature: str) -> Any:
    return _require_sim_dependency(feature=feature, dependency="mujoco", module="mujoco")


def require_usd(*, feature: str) -> Any:
    return _require_sim_dependency(feature=feature, dependency="usd-core", module="pxr")


def require_coacd(*, feature: str) -> Any:
    return _require_sim_dependency(feature=feature, dependency="coacd", module="coacd")


def sim_extra_available(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False
