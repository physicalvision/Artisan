from __future__ import annotations

import json
import os
import resource
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from pipeline.operation_timing import collect_operation_timings
from pipeline.traces import rotate_path_aside


def _timing_enabled() -> bool:
    v = (os.environ.get("ARTISAN_TIMING") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _cpu(who: int) -> float:
    ru = resource.getrusage(who)
    return ru.ru_utime + ru.ru_stime


@dataclass(slots=True)
class Sample:

    wall_s: float = 0.0
    self_cpu_s: float = 0.0
    child_cpu_s: float = 0.0
    ok: bool = True
    started_at: float | None = None
    span_id: str | None = None
    operation_spans: list[dict[str, Any]] = field(default_factory=list)


@contextmanager
def _null_measure() -> Iterator[Sample]:
    s = Sample()
    w0 = time.monotonic()
    try:
        yield s
    finally:
        s.wall_s = time.monotonic() - w0


class TimingRecorder:

    __slots__ = ("path", "agent", "_phase", "_enabled", "_fp")

    def __init__(self, path: Path | str, *, agent: str = "") -> None:
        self.path = Path(path)
        self.agent = agent
        self._phase: list[str | None] = [None]
        self._enabled = _timing_enabled()
        self._fp: Any = None
        if self._enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            rotate_path_aside(self.path)
            self._fp = self.path.open("a", encoding="utf-8")

    def set_phase(self, phase: str | None) -> None:
        self._phase[0] = phase

    def for_agent(self, agent: str) -> "TimingRecorder":
        view = object.__new__(TimingRecorder)
        view.path = self.path
        view.agent = agent
        view._phase = self._phase
        view._enabled = self._enabled
        view._fp = self._fp
        return view

    @contextmanager
    def measure(self) -> Iterator[Sample]:
        s = Sample()
        s.started_at = time.time()
        w0 = time.monotonic()
        c0 = _cpu(resource.RUSAGE_SELF)
        k0 = _cpu(resource.RUSAGE_CHILDREN)
        with collect_operation_timings(enabled=self._enabled) as operations:
            s.span_id = operations.parent_span_id
            try:
                yield s
            finally:
                s.wall_s = time.monotonic() - w0
                s.self_cpu_s = _cpu(resource.RUSAGE_SELF) - c0
                s.child_cpu_s = _cpu(resource.RUSAGE_CHILDREN) - k0
                s.operation_spans = operations.rows

    def record(
        self,
        *,
        kind: str,
        name: str,
        sample: Sample,
        agent: str | None = None,
        tokens: dict | None = None,
        cost_usd: float | None = None,
        ok: bool | None = None,
        **extra: Any,
    ) -> None:
        if not self._enabled or self._fp is None:
            return
        rec: dict[str, Any] = {
            "ts": time.time(),
            "agent": agent or self.agent,
            "kind": kind,
            "name": name,
            "phase": self._phase[0],
            "wall_s": round(sample.wall_s, 4),
            "self_cpu_s": round(sample.self_cpu_s, 4),
            "child_cpu_s": round(sample.child_cpu_s, 4),
            "ok": sample.ok if ok is None else ok,
        }
        if tokens is not None:
            rec["tokens"] = tokens
        if cost_usd is not None:
            rec["cost_usd"] = round(cost_usd, 6)
        if sample.operation_spans:
            rec["span_id"] = sample.span_id
            rec["operation_spans"] = sample.operation_spans
        if sample.started_at is not None:
            rec["started_at"] = sample.started_at
        rec.update(extra)
        self._fp.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self._fp.flush()

    def close(self) -> None:
        if self._fp is not None:
            try:
                self._fp.close()
            finally:
                self._fp = None
