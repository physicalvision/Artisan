
from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Iterator, ParamSpec, TypeVar


@dataclass(slots=True)
class OperationTimings:
    parent_span_id: str
    rows: list[dict[str, Any]] = field(default_factory=list)


_COLLECTOR: ContextVar[OperationTimings | None] = ContextVar("operation_timings", default=None)
_PARENT: ContextVar[str | None] = ContextVar("operation_parent", default=None)
_P = ParamSpec("_P")
_R = TypeVar("_R")


def active_operation_id() -> str | None:
    return _PARENT.get() if _COLLECTOR.get() is not None else None


@contextmanager
def collect_operation_timings(
    *, parent_span_id: str | None = None, enabled: bool = True
) -> Iterator[OperationTimings]:
    collection = OperationTimings(parent_span_id or uuid.uuid4().hex)
    collector_token = _COLLECTOR.set(collection if enabled else None)
    parent_token = _PARENT.set(collection.parent_span_id if enabled else None)
    try:
        yield collection
    finally:
        _PARENT.reset(parent_token)
        _COLLECTOR.reset(collector_token)


@contextmanager
def operation_span(name: str) -> Iterator[None]:
    collector = _COLLECTOR.get()
    if collector is None:
        yield
        return
    span_id = uuid.uuid4().hex
    parent = _PARENT.get()
    token = _PARENT.set(span_id)
    started = time.monotonic()
    started_cpu = time.process_time()
    started_at = time.time()
    status = "ok"
    try:
        yield
    except BaseException as exc:
        status = type(exc).__name__
        raise
    finally:
        collector.rows.append(
            {
                "span_id": span_id,
                "parent_span_id": parent,
                "name": name,
                "worker_pid": os.getpid(),
                "started_at": started_at,
                "wall_s": time.monotonic() - started,
                "worker_cpu_s": time.process_time() - started_cpu,
                "cpu_scope": "worker_process_inclusive",
                "outcome": status,
                "duration_scope": "inclusive",
            }
        )
        _PARENT.reset(token)


def timed_operation(name: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(function)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with operation_span(name):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def import_operation_timings(rows: list[dict[str, Any]]) -> None:
    collector = _COLLECTOR.get()
    if collector is not None:
        collector.rows.extend(rows)
