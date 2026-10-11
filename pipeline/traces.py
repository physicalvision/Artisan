from __future__ import annotations

import json
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from storage.trajectories import TRAJECTORY_FILENAME


def rotate_path_aside(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    n = 1
    while True:
        rotated = path.with_name(f"{path.stem}.r{n}{path.suffix}")
        if not rotated.exists():
            break
        n += 1
    path.rename(rotated)


@dataclass(slots=True)
class TraceWriter:

    trace_dir: Path
    trajectory_filename: str = TRAJECTORY_FILENAME
    conversation_path: Path = field(init=False)
    _conversation_file: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.conversation_path = self.trace_dir / self.trajectory_filename
        rotate_path_aside(self.conversation_path)
        self._conversation_file = self.conversation_path.open("w", encoding="utf-8")

    def write_message(self, message: dict[str, Any], **extra: Any) -> None:
        record = {"message": message, **extra}
        self.write_event("message", record)

    def write_event(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {
            "ts": time.time(),
            "type": event_type,
            **payload,
        }
        self._conversation_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._conversation_file.flush()

    def close(self) -> None:
        with suppress(Exception):
            self._conversation_file.close()
