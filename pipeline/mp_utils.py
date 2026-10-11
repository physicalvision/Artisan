from __future__ import annotations

import multiprocessing as mp
import os
from typing import Sequence

_MP_START_METHOD_ENV = "ARTISAN_MP_START_METHOD"
_DEFAULT_START_METHODS = ("spawn", "forkserver")


def mp_start_method_env_var() -> str:
    return _MP_START_METHOD_ENV


def configured_mp_start_method_override() -> str | None:
    raw_override = os.environ.get(_MP_START_METHOD_ENV)
    if raw_override is None:
        return None
    normalized = raw_override.strip()
    return normalized or None


def resolve_mp_start_method(*, prefer_fork: bool = False) -> str:
    available = tuple(mp.get_all_start_methods())
    if not available:
        raise RuntimeError("No multiprocessing start methods are available on this platform.")

    raw_override = configured_mp_start_method_override()
    if raw_override is not None:
        start_method = raw_override.lower()
        if start_method not in available:
            supported = ", ".join(available)
            raise ValueError(
                f"Unsupported multiprocessing start method {raw_override!r} from "
                f"{_MP_START_METHOD_ENV}. Supported values: {supported}."
            )
        return start_method

    if prefer_fork and "fork" in available:
        return "fork"

    for start_method in _DEFAULT_START_METHODS:
        if start_method in available:
            return start_method
    return available[0]


def get_mp_context(
    *,
    prefer_fork: bool = False,
    forkserver_preload: Sequence[str] | None = None,
) -> mp.context.BaseContext:
    start_method = resolve_mp_start_method(prefer_fork=prefer_fork)
    if (
        start_method == "forkserver"
        and forkserver_preload
        and hasattr(mp, "set_forkserver_preload")
    ):
        modules = [str(name).strip() for name in forkserver_preload if str(name).strip()]
        if modules:
            mp.set_forkserver_preload(modules)
    return mp.get_context(start_method)
