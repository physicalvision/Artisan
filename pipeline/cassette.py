from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pipeline.traces import rotate_path_aside


class CassetteError(RuntimeError):
    pass


class Cassette:
    def __init__(self, path: Path, mode: str = "off") -> None:
        self.path = Path(path)
        self.mode = mode if mode in ("off", "record", "replay") else "off"
        self._records: list[dict[str, Any]] = []
        self._cursor = 0
        self._fp = None
        if self.mode == "replay":
            if not self.path.exists():
                raise CassetteError(f"replay cassette not found: {self.path}")
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    self._records.append(json.loads(line))
        elif self.mode == "record":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            rotate_path_aside(self.path)
            self._fp = self.path.open("w", encoding="utf-8")

    @classmethod
    def from_env(cls, default_path: Path) -> "Cassette":
        mode = (os.environ.get("ARTISAN_PIPELINE_CASSETTE_MODE") or "off").strip().lower()
        path = os.environ.get("ARTISAN_PIPELINE_CASSETTE") or str(default_path)
        return cls(Path(path), mode=mode)

    @property
    def recording(self) -> bool:
        return self.mode == "record"

    @property
    def replaying(self) -> bool:
        return self.mode == "replay"

    def record(self, label: str, kind: str, data: dict[str, Any]) -> None:
        if self.mode != "record" or self._fp is None:
            return
        self._fp.write(
            json.dumps(
                {"label": label, "kind": kind, "data": data}, ensure_ascii=False, default=str
            )
            + "\n"
        )
        self._fp.flush()

    def replay(self, label: str, kind: str) -> dict[str, Any]:
        if self.mode != "replay":
            raise CassetteError("replay() called while not in replay mode")
        if self._cursor >= len(self._records):
            raise CassetteError(
                f"cassette exhausted at call {self._cursor} (expected {label}/{kind}); "
                "the code path made more LLM calls than were recorded"
            )
        rec = self._records[self._cursor]
        self._cursor += 1
        if rec.get("label") != label or rec.get("kind") != kind:
            raise CassetteError(
                f"cassette drift at call {self._cursor - 1}: recorded "
                f"{rec.get('label')}/{rec.get('kind')} but code expected {label}/{kind}"
            )
        return rec.get("data", {})

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None
