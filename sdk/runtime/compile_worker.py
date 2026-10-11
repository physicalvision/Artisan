
from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from typing import Any

from pipeline.models import CompileReport, CompileSignalBundle
from pipeline.mp_utils import get_mp_context
from pipeline.operation_timing import (
    active_operation_id,
    collect_operation_timings,
    import_operation_timings,
    operation_span,
)
from sdk.runtime._render_protocol import write_report

PROTOCOL_VERSION = "compile-feedback-v1"


def _core_inputs(model_path: Path, urdf_xml: str) -> str:
    paths = {model_path.absolute()}
    for name in ["assets/meshes", "inputs", "references/inputs"]:
        root = model_path.parent / name
        if root.exists():
            paths.update(path.absolute() for path in root.rglob("*") if path.is_file())
    for node in ET.fromstring(urdf_xml).iter("mesh"):
        filename = node.get("filename")
        if filename:
            path = Path(filename)
            paths.add(path if path.is_absolute() else (model_path.parent / path).absolute())
    rows = [
        (str(path), hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None)
        for path in sorted(paths)
    ]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _failure(exc: BaseException) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "error": f"{type(exc).__name__}: {exc}",
        "error_type": type(exc).__name__,
        "traceback": traceback.format_exc(),
    }
    for key in ["compiled_urdf_xml", "warnings"]:
        value = getattr(exc, key, None)
        if value is not None:
            payload[key] = value
    bundle = getattr(exc, "compile_signal_bundle", None)
    if isinstance(bundle, CompileSignalBundle):
        payload["signal_bundle"] = bundle.to_dict()
    return payload


def _worker(request: dict[str, Any], directory_raw: str) -> None:
    directory = Path(directory_raw)
    core_sent = False
    try:
        os.setsid()
        write_report(directory / "started.json", {"pid": os.getpid(), "pgid": os.getpgrp()})
        if request.get("schema_version") != PROTOCOL_VERSION:
            raise ValueError("Unsupported compile feedback request protocol")
        from pipeline.compiler import compile_urdf_report
        from sdk._core.v0.assets import activate_asset_session, asset_session_for_script
        from sdk._core.v0.exact_collisions import compile_object_model_with_exact_collisions
        from sdk.runtime import runtime as diagnostics_runtime

        model_path = Path(request["model_path"])
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != request["source_sha256"]:
            raise ValueError("Model changed before compile started")
        captured: list[dict[str, Any]] = []
        with collect_operation_timings(
            parent_span_id=request.get("timing_parent_id"),
            enabled=request.get("timing_parent_id") is not None,
        ) as core_timings:
            report = compile_urdf_report(
                model_path,
                sdk_package=request["sdk_package"],
                target=request["target"],
                _captured_globals=captured,
            )
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != request["source_sha256"]:
            raise ValueError("Model changed during compilation")
        payload = {
            "ok": True,
            "schema_version": PROTOCOL_VERSION,
            "urdf_xml": report.urdf_xml,
            "warnings": report.warnings,
            "signal_bundle": report.signal_bundle.to_dict(),
            "core_inputs": _core_inputs(model_path, report.urdf_xml),
            "operation_spans": core_timings.rows,
        }
        write_report(directory / "core.json", payload)
        core_sent = True
        with collect_operation_timings(
            parent_span_id=request.get("timing_parent_id"),
            enabled=request.get("timing_parent_id") is not None,
        ) as diagnostic_timings:
            with (
                operation_span("compile.diagnostic_geometry"),
                activate_asset_session(asset_session_for_script(model_path)),
            ):
                raw = captured[0]["object_model"]
                compiled = compile_object_model_with_exact_collisions(
                    raw,
                    asset_root=model_path.parent,
                    validate=False,
                )
            geometry = (compiled, model_path.parent)
            for name, function in [
                ("mass", diagnostics_runtime._render_mass_audit_block),
                ("topology", diagnostics_runtime._render_collision_topology_block),
            ]:
                try:
                    value = function(
                        model_path, sdk_package=request["sdk_package"], _geometry=geometry
                    )
                    result = {"text": value}
                except Exception as exc:
                    result = {"text": None, "error": f"{type(exc).__name__}: {exc}"}
                write_report(directory / f"{name}.json", result)
                write_report(
                    directory / "diagnostic_timings.json", {"rows": diagnostic_timings.rows}
                )
        write_report(directory / "done.json", {"status": "complete"})
    except BaseException as exc:
        write_report(
            directory / ("diagnostic_error.json" if core_sent else "core.json"), _failure(exc)
        )


def _read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _stop(process: Any, directory: Path) -> None:
    from sdk._core.v0.motion_engine.executor import _process_group_members

    started = _read(directory / "started.json") or {}
    owned_group = started.get("pid") == process.pid and started.get("pgid") == process.pid
    if owned_group:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        process.join(timeout=0.25)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.is_alive():
        process.kill()
    process.join(timeout=2)
    if process.is_alive():
        raise RuntimeError("Compile worker could not be reaped")
    if owned_group:
        deadline = time.monotonic() + 2
        while True:
            survivors = []
            for pid in _process_group_members(process.pid):
                try:
                    fields = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
                    if fields[0] not in {"Z", "X"}:
                        survivors.append(pid)
                except OSError:
                    pass
            if not survivors:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Compile descendants survived cancellation: {survivors}")
            time.sleep(0.02)


def _wait_for_results(
    process: Any,
    directory: Path,
    *,
    core_deadline: float,
    diagnostic_budget_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    core = None
    diagnostic_deadline = None
    status = "incomplete"
    while True:
        if core is None:
            core = _read(directory / "core.json")
            if core is not None:
                if core.get("ok") is not True:
                    return core, {"status": "not_run"}
                diagnostic_deadline = time.monotonic() + diagnostic_budget_s
        if core is not None:
            if (directory / "done.json").is_file():
                status = "complete"
                break
            if (directory / "diagnostic_error.json").is_file():
                status = "error"
                break
        if not process.is_alive():
            if core is None:
                core = _read(directory / "core.json")
            if core is None:
                raise RuntimeError(
                    f"Compile worker exited without a core result: {process.exitcode}"
                )
            if core.get("ok") is not True:
                return core, {"status": "not_run"}
            status = (
                "complete"
                if (directory / "done.json").is_file()
                else "error"
                if (directory / "diagnostic_error.json").is_file()
                else "worker_exit"
            )
            break
        deadline = core_deadline if core is None else diagnostic_deadline
        assert deadline is not None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if core is None:
                raise TimeoutError("URDF compile timed out before a core result was available")
            status = "timeout"
            break
        process.join(timeout=min(0.05, remaining))
    diagnostics = {"schema_version": PROTOCOL_VERSION, "status": status}
    for name in ["mass", "topology"]:
        try:
            result = _read(directory / f"{name}.json") or {}
            diagnostics[name] = result.get("text")
            if result.get("error"):
                diagnostics["status"] = "error"
        except (OSError, ValueError):
            diagnostics[name] = None
            diagnostics["status"] = "error"
    try:
        telemetry = _read(directory / "diagnostic_timings.json") or {}
    except (OSError, ValueError):
        telemetry = {}
    import_operation_timings(telemetry.get("rows", []))
    return core, diagnostics


def compile_with_diagnostics(model_path: Path, *, sdk_package: str, target: str) -> CompileReport:
    from pipeline.compiler import _compile_report_from_worker_payload, _env_float

    budget = _env_float("URDF_COMPILE_TIMEOUT_SECONDS", 300)
    diagnostic_budget = float(os.environ.get("ARTISAN_COMPILE_DIAGNOSTICS_TIMEOUT_SECONDS", "60"))
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("Shared compile worker requires a finite positive core timeout")
    if not math.isfinite(diagnostic_budget) or diagnostic_budget < 0:
        raise ValueError("Compile diagnostic timeout must be finite and nonnegative")
    model_path = model_path.absolute()
    request = {
        "schema_version": PROTOCOL_VERSION,
        "model_path": str(model_path),
        "sdk_package": sdk_package,
        "target": target,
        "source_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "timing_parent_id": active_operation_id(),
    }
    with tempfile.TemporaryDirectory(prefix="artisan-compile-feedback-") as raw_directory:
        directory = Path(raw_directory)
        process = get_mp_context().Process(
            target=_worker, args=(request, raw_directory), daemon=True
        )
        started = False
        try:
            process.start()
            started = True
            core, diagnostics = _wait_for_results(
                process,
                directory,
                core_deadline=time.monotonic() + budget,
                diagnostic_budget_s=diagnostic_budget,
            )
        finally:
            if started:
                _stop(process, directory)
        import_operation_timings(core.get("operation_spans", []))
        report = _compile_report_from_worker_payload(core)
        if core.get("schema_version") != PROTOCOL_VERSION:
            raise ValueError("Unsupported compile feedback response protocol")
        if _core_inputs(model_path, report.urdf_xml) != core.get("core_inputs"):
            raise ValueError("Core compile inputs changed before result publication")
        return replace(report, diagnostics=diagnostics)
