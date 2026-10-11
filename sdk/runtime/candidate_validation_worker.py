from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import signal
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CANDIDATE_BUDGET_S = 1200.0


@dataclass(frozen=True, slots=True)
class CandidateWorkerOutcome:
    status: str
    report: dict[str, Any] | None
    envelope: dict[str, Any]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def _entry(payload: dict[str, Any], result_path_raw: str) -> None:
    result_path = Path(result_path_raw)
    try:
        from sdk.runtime.release_candidate import (
            StagedReleaseCandidate,
            load_manifest,
        )
        from sdk.runtime.runtime import validate_staged_release_candidate

        root = Path(payload["root"])
        manifest_path = Path(payload["manifest_path"])
        staged = StagedReleaseCandidate(
            root=root,
            manifest_path=manifest_path,
            manifest=load_manifest(manifest_path),
            export_report=dict(payload["export_report"]),
        )
        report = validate_staged_release_candidate(
            staged,
            dict(payload["blueprint"]),
            accepted_fcl_poses=list(payload["accepted_fcl_poses"]),
        )
        _write_json_atomic(result_path, {"status": "ok", "report": report})
    except BaseException as exc:  # noqa: BLE001 - worker boundary serializes Python failures
        _write_json_atomic(
            result_path,
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "detail": str(exc),
                "traceback": traceback.format_exc(limit=20),
            },
        )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return dict(value) if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def validate_candidate_in_worker(
    staged: Any,
    blueprint: dict[str, Any],
    *,
    accepted_fcl_poses: list[dict[str, float]],
    deadline_s: float | None = None,
) -> CandidateWorkerOutcome:
    budget = float(
        deadline_s
        if deadline_s is not None
        else os.environ.get("ARTISAN_CANDIDATE_BUDGET_S", DEFAULT_CANDIDATE_BUDGET_S)
    )
    ipc_root = Path(tempfile.mkdtemp(prefix="artisan-candidate-validation-"))
    result_path = ipc_root / "result.json"
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_entry,
        args=(
            {
                "root": str(staged.root),
                "manifest_path": str(staged.manifest_path),
                "export_report": staged.export_report,
                "blueprint": blueprint,
                "accepted_fcl_poses": accepted_fcl_poses,
            },
            str(result_path),
        ),
        daemon=False,
    )
    try:
        process.start()
        process.join(max(0.0, budget))
        if process.is_alive():
            process.terminate()
            process.join(10.0)
            if process.is_alive():
                process.kill()
                process.join(5.0)
            return CandidateWorkerOutcome(
                "budget_exceeded",
                None,
                {"deadline_s": budget, "exitcode": process.exitcode},
            )
        result = _read_json(result_path)
        if process.exitcode != 0 and not result:
            signal_number = -process.exitcode if process.exitcode and process.exitcode < 0 else None
            return CandidateWorkerOutcome(
                "crashed",
                None,
                {
                    "exitcode": process.exitcode,
                    "signal": signal_number,
                    "signal_name": signal.Signals(signal_number).name if signal_number else None,
                },
            )
        if result.get("status") == "ok" and isinstance(result.get("report"), dict):
            return CandidateWorkerOutcome("ok", dict(result["report"]), {})
        return CandidateWorkerOutcome("crashed", None, result)
    finally:
        shutil.rmtree(ipc_root, ignore_errors=True)
