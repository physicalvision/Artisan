from __future__ import annotations

import fcntl
import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from collision.backend import NarrowPhaseKind, NarrowPhaseResult
from collision.sweep import CellResult, JointSweep


def _pairs(rows: dict[str, Any]) -> dict[str, NarrowPhaseResult]:
    return {
        key: NarrowPhaseResult(**{**value, "kind": NarrowPhaseKind(value["kind"])})
        for key, value in rows.items()
    }


class GTCache:
    def __init__(
        self, root: Path, urdf: Path, policy_name: str = "rev10", *, refresh: bool = False
    ) -> None:
        if not policy_name or policy_name in {".", ".."}:
            raise ValueError("Select a nonempty policy name other than . or ..")
        path = urdf.expanduser().absolute()

        self.key = str(Path(quote(policy_name, safe=""), *path.parts[1:]))
        self.target = root / f"{self.key}.json"
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self._reset_pending = refresh

    def _get(self, unit: Any, compute: Callable[[], Any]) -> Any:
        key = json.dumps(unit)
        with self.target.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)

            values = (
                json.loads(self.target.read_text())
                if self.target.is_file() and not self._reset_pending
                else {}
            )
            if key in values:
                return values[key]
            value = compute()
            values[key] = value
            temporary = self.target.with_name(f".{self.target.name}.{uuid.uuid4().hex}.tmp")
            with temporary.open("x") as stream:
                json.dump(values, stream, allow_nan=False)
            temporary.replace(self.target)
            self._reset_pending = False
            return value

    def rest(
        self, compute: Callable[[], dict[str, NarrowPhaseResult]]
    ) -> dict[str, NarrowPhaseResult]:
        return _pairs(self._get(["rest"], lambda: {k: asdict(v) for k, v in compute().items()}))

    def sweep(
        self, level: str, keys: tuple[str, ...], compute: Callable[[], JointSweep]
    ) -> JointSweep:
        value = self._get([level, keys], lambda: asdict(compute()))
        return JointSweep(
            joint_key=value["joint_key"],
            swept_joint_keys=tuple(value["swept_joint_keys"]),
            dynamic_body_pairs=tuple(value["dynamic_body_pairs"]),
            cells=tuple(
                CellResult(**{**row, "body_pairs": _pairs(row["body_pairs"])})
                for row in value["cells"]
            ),
        )
