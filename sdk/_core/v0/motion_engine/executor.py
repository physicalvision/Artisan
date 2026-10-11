from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing
import os
import shutil
import signal
import tempfile
import time
import traceback
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .pose_plan import PosePlanPolicy
from .report import (
    MOTION_EVIDENCE_SCHEMA_VERSION,
    blueprint_gate_identity,
    build_scan_evidence,
    detector_policy_digest,
    motion_evidence_key,
    motion_request_key,
    write_envelope_atomic,
)
from .resolve import (
    StaleInputError,
    default_manifest_cache_dir,
    load_collision_manifest,
    load_or_create_collision_manifest,
    resolution_request_key,
    verify_frozen_inputs,
)
from .sweep import MOTION_SHARD_PGID_ENV, motion_execution_config

DEFAULT_MOTION_BUDGET_S = 1200.0
_UNAVAILABLE_SHARD_PGID = "unavailable"
REQUIRED_QUERIES = (
    "connectivity",
    "self_collision",
    "support",
    "driver_reach",
    "gate_witness",
    "penetration",
    "connection_ram",
)
MOTION_FINDING_KEYS = (
    "motion_contact_candidates",
    "geometry_island_warnings",
    "motion_support_findings",
    "driver_reach_findings",
    "contact_exclusion_evidence",
    "connection_cleanliness_findings",
    "full_collision_penetrations",
    "access_gate_unresolved",
    "declared_gate_witnesses",
)


@dataclass(frozen=True, slots=True)
class WorkerOutcome:
    status: str
    payload: dict[str, Any]
    exitcode: int | None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def _progress(path: Path, stage: str, **extra: Any) -> None:
    _write_json_atomic(path, {"stage": stage, "updated_at": time.time(), **extra})


def _worker_entry(
    kind: str,
    payload: dict[str, Any],
    result_path_raw: str,
    progress_path_raw: str,
) -> None:
    result_path = Path(result_path_raw)
    progress_path = Path(progress_path_raw)
    try:
        model_path = Path(payload["model_path"])
        _progress(progress_path, "source_hash")
        actual_hash = _sha256_file(model_path)
        if actual_hash != payload["expected_source_sha256"]:
            _write_json_atomic(
                result_path,
                {
                    "status": "stale_input",
                    "expected_source_sha256": payload["expected_source_sha256"],
                    "actual_source_sha256": actual_hash,
                },
            )
            return
        if kind == "resolution":
            attempt_log = progress_path.with_name("coacd-attempts.jsonl")
            repair_attempt_log = progress_path.with_name("repair-attempts.jsonl")
            os.environ["ARTISAN_COACD_ATTEMPT_LOG"] = str(attempt_log)
            os.environ["ARTISAN_REPAIR_ATTEMPT_LOG"] = str(repair_attempt_log)
            _progress(progress_path, "resolution")
            manifest = load_or_create_collision_manifest(
                model_path,
                sdk_package=str(payload["sdk_package"]),
                cache_dir=Path(payload["manifest_cache_dir"]),
            )
            verify_frozen_inputs(model_path, manifest, sdk_package=str(payload["sdk_package"]))
            _write_json_atomic(
                result_path,
                {
                    "status": "ok",
                    "resolution_request_key": manifest.resolution_request_key,
                    "resolved_manifest_digest": manifest.resolved_manifest_digest,
                    "simulator_collision_digest": manifest.simulator_collision_digest,
                    "manifest_published": True,
                },
            )
            return
        if kind == "scan":
            manifest = load_collision_manifest(
                Path(payload["manifest_cache_dir"]), str(payload["resolution_request_key"])
            )
            if (
                manifest is None
                or manifest.resolved_manifest_digest != payload["resolved_manifest_digest"]
            ):
                _write_json_atomic(
                    result_path,
                    {"status": "stale_input", "detail": "manifest changed before scan"},
                )
                return
            _progress(progress_path, "scan")
            envelope = build_scan_evidence(
                model_path,
                manifest,
                source=str(payload["source"]),
                blueprint=payload.get("blueprint"),
                sdk_package=str(payload["sdk_package"]),
                seed=int(payload["seed"]),
                policy=PosePlanPolicy(**dict(payload["policy"])),
            )
            verify_frozen_inputs(model_path, manifest, sdk_package=str(payload["sdk_package"]))
            _write_json_atomic(result_path, {"status": "ok", "envelope": envelope})
            return
        raise ValueError(f"unknown motion worker kind: {kind}")
    except StaleInputError as exc:
        _write_json_atomic(result_path, {"status": "stale_input", "detail": exc.details})
    except BaseException as exc:  # noqa: BLE001 - worker boundary serializes all failures
        _write_json_atomic(
            result_path,
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "detail": str(exc),
                "traceback": traceback.format_exc(limit=20),
            },
        )


def _leader_entry(
    kind: str,
    payload: dict[str, Any],
    result_path_raw: str,
    progress_path_raw: str,
    handshake_path_raw: str,
) -> None:
    try:
        os.setsid()
    except OSError:
        pgid = None
        os.environ[MOTION_SHARD_PGID_ENV] = _UNAVAILABLE_SHARD_PGID
    else:
        pgid = os.getpgrp()
        os.environ[MOTION_SHARD_PGID_ENV] = str(pgid)
    _write_json_atomic(Path(handshake_path_raw), {"pid": os.getpid(), "pgid": pgid})
    _worker_entry(kind, payload, result_path_raw, progress_path_raw)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return dict(payload) if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _handshake_pgid(handshake_path: Path) -> int | None:
    try:
        pgid = int(_read_json(handshake_path).get("pgid"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if pgid <= 0 or pgid == os.getpgrp():
        return None
    return pgid


def _process_group_members(pgid: int) -> list[int]:
    members: list[int] = []
    own = os.getpid()
    try:
        entries = os.listdir("/proc")
    except OSError:  # pragma: no cover - Linux-only deployment target
        return members
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == own:
            continue
        try:
            stat = Path("/proc", entry, "stat").read_text(encoding="utf-8")
        except OSError:
            continue
        fields = stat.rpartition(")")[2].split()
        if len(fields) < 3:
            continue
        try:
            if int(fields[2]) == pgid:
                members.append(pid)
        except ValueError:
            continue
    return members


def _await_group_exit(pgid: int, timeout_s: float) -> list[int]:
    deadline = time.monotonic() + timeout_s
    members = _process_group_members(pgid)
    while members and time.monotonic() < deadline:
        time.sleep(0.02)
        members = _process_group_members(pgid)
    return members


def _terminate_process_group(
    pgid: int | None, *, term_grace_s: float = 10.0, kill_grace_s: float = 5.0
) -> dict[str, Any]:
    if pgid is None:
        return {"pgid": None, "signalled": [], "survivors": []}
    members = _process_group_members(pgid)
    if not members:
        return {"pgid": pgid, "signalled": [], "survivors": []}
    try:
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        pass
    survivors = _await_group_exit(pgid, term_grace_s)
    if survivors:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass
        survivors = _await_group_exit(pgid, kill_grace_s)
    return {"pgid": pgid, "signalled": members, "survivors": survivors}


def _terminate_worker(process: Any, handshake_path: Path) -> dict[str, Any]:
    group = _terminate_process_group(_handshake_pgid(handshake_path))
    if process.is_alive():
        process.terminate()
        process.join(10.0)
        if process.is_alive():
            process.kill()
            process.join(5.0)
    return group


def _spawn_worker(
    kind: str,
    payload: dict[str, Any],
    *,
    deadline_s: float,
) -> WorkerOutcome:
    ipc_root = Path(tempfile.mkdtemp(prefix=f"artisan-motion-{kind}-"))
    result_path = ipc_root / "result.json"
    progress_path = ipc_root / "progress.json"
    handshake_path = ipc_root / "handshake.json"
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_leader_entry,
        args=(
            kind,
            payload,
            str(result_path),
            str(progress_path),
            str(handshake_path),
        ),
        daemon=False,
    )
    try:
        process.start()
        process.join(max(0.0, deadline_s))
        if process.is_alive():
            _terminate_worker(process, handshake_path)
            return WorkerOutcome(
                "budget_exceeded",
                {
                    "phase": kind,
                    "deadline_s": deadline_s,
                    "last_progress": _read_json(progress_path),
                    "cache_marker": _resolution_cache_marker(ipc_root),
                },
                process.exitcode,
            )
        payload_result = _read_json(result_path)
        if process.exitcode != 0 and not payload_result:
            crashed_signal = (
                -process.exitcode if process.exitcode and process.exitcode < 0 else None
            )
            return WorkerOutcome(
                "crashed",
                {
                    "phase": kind,
                    "signal": crashed_signal,
                    "signal_name": (
                        signal.Signals(crashed_signal).name if crashed_signal else None
                    ),
                    "last_progress": _read_json(progress_path),
                },
                process.exitcode,
            )
        return WorkerOutcome(
            str(payload_result.get("status") or "failed"), payload_result, process.exitcode
        )
    finally:
        _terminate_worker(process, handshake_path)
        shutil.rmtree(ipc_root, ignore_errors=True)


def _resolution_cache_marker(ipc_root: Path) -> dict[str, Any]:
    def events(path: Path) -> tuple[list[str], list[str]]:
        attempted: set[str] = set()
        published: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(event.get("key") or "")
            if event.get("action") == "attempted" and key:
                attempted.add(key)
            if event.get("action") == "published" and key:
                published.add(key)
        return sorted(attempted), sorted(published)

    attempted_coacd, published_coacd = events(ipc_root / "coacd-attempts.jsonl")
    attempted_repair, published_repair = events(ipc_root / "repair-attempts.jsonl")
    return {
        "attempted_repair_keys": attempted_repair,
        "published_repair_keys": published_repair,
        "attempted_coacd_keys": attempted_coacd,
        "published_coacd_keys": published_coacd,
        "manifest_published": False,
    }


def _failure_evidence(
    *,
    request_key: str | None,
    manifest_digest: str | None,
    simulator_digest: str | None,
    status: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    detector_status = {
        name: {"status": "failed", "failure": status, "detail": details}
        for name in REQUIRED_QUERIES
    }
    return {
        "motion_contact_candidates": [],
        "geometry_island_warnings": [],
        "motion_support_findings": [],
        "driver_reach_findings": [],
        "contact_exclusion_evidence": [],
        "connection_cleanliness_findings": [],
        "full_collision_penetrations": [],
        "accepted_fcl_poses": [],
        "access_gate_unresolved": [],
        "declared_gate_witnesses": [],
        "joint_movers": {},
        "motion_execution_status": {"status": status, **details},
        "required_detector_status": detector_status,
        "motion_identity": {
            "schema_version": MOTION_EVIDENCE_SCHEMA_VERSION,
            "motion_request_key": request_key,
            "motion_evidence_key": (
                motion_evidence_key(request_key, manifest_digest)
                if request_key and manifest_digest
                else None
            ),
            "resolved_manifest_digest": manifest_digest,
            "simulator_collision_digest": simulator_digest,
            "resolution_request_key": None,
            "cache_hit": False,
        },
    }


def _cached_envelope(
    cache_path: Path,
    *,
    request_key: str,
    manifest_digest: str,
    deadline_s: float,
) -> dict[str, Any] | None:
    cached = _read_json(cache_path)
    if not cached:
        return None
    if (
        cached.get("schema_version") != MOTION_EVIDENCE_SCHEMA_VERSION
        or cached.get("motion_request_key") != request_key
        or cached.get("resolved_manifest_digest") != manifest_digest
    ):
        return None
    execution = (cached.get("evidence") or {}).get("motion_execution_status") or {}
    if execution.get("status") != "ok":
        if execution.get("execution_config") != motion_execution_config():
            return None
    if execution.get("status") == "budget_exceeded":
        recorded = float(execution.get("deadline_s", 0.0))
        marker = execution.get("cache_marker") or {}
        warmed = bool(
            marker.get("published_repair_keys")
            or marker.get("published_coacd_keys")
            or marker.get("manifest_published")
        )
        if deadline_s > recorded or (
            execution.get("phase") == "resolution"
            and warmed
            and not execution.get("same_budget_retry_consumed")
        ):
            return None
    evidence = dict(cached.get("evidence") or {})
    identity = dict(evidence.get("motion_identity") or {})
    identity["cache_hit"] = True
    evidence["motion_identity"] = identity
    return evidence


def _resolution_failure_cache_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.name}.resolution-failure.json")


def _resolution_failure_record(path: Path, *, resolution_key: str) -> dict[str, Any]:
    cached = _read_json(path)
    if (
        cached.get("resolution_request_key") != resolution_key
        or cached.get("schema_version") != MOTION_EVIDENCE_SCHEMA_VERSION
        or cached.get("detector_policy_sha256") != detector_policy_digest()
    ):
        return {}
    return cached


def _cached_resolution_failure(
    path: Path,
    *,
    resolution_key: str,
    deadline_s: float,
) -> dict[str, Any] | None:
    cached = _resolution_failure_record(path, resolution_key=resolution_key)
    if not cached:
        return None
    evidence = cached.get("evidence")
    if not isinstance(evidence, dict):
        return None
    execution = evidence.get("motion_execution_status")
    if not isinstance(execution, dict):
        return None
    if execution.get("status") != "budget_exceeded":
        return dict(evidence)
    recorded = float(execution.get("deadline_s", 0.0))
    if deadline_s > recorded:
        return None
    marker = execution.get("cache_marker") or {}
    warmed = bool(
        marker.get("published_repair_keys")
        or marker.get("published_coacd_keys")
        or marker.get("manifest_published")
    )
    if warmed and not execution.get("same_budget_retry_consumed"):
        updated = dict(evidence)
        updated_execution = dict(execution)
        updated_execution["same_budget_retry_consumed"] = True
        updated["motion_execution_status"] = updated_execution
        _write_json_atomic(
            path,
            {
                **cached,
                "evidence": updated,
            },
        )
        return None
    result = dict(evidence)
    identity = dict(result.get("motion_identity") or {})
    identity["cache_hit"] = True
    result["motion_identity"] = identity
    return result


def _build_motion_evidence_unpublished(
    model_path: Path,
    *,
    source: str,
    cache_path: Path,
    blueprint: dict[str, Any] | None = None,
    sdk_package: str = "sdk",
    seed: int = 0,
    deadline_s: float | None = None,
    manifest_cache_dir: Path | None = None,
    measure: Callable[[str], AbstractContextManager[Any]] | None = None,
) -> dict[str, Any]:
    model_path = Path(model_path).resolve()
    cache_path = Path(cache_path)
    budget = float(
        deadline_s
        if deadline_s is not None
        else os.environ.get("ARTISAN_MOTION_BUDGET_S", DEFAULT_MOTION_BUDGET_S)
    )
    started = time.monotonic()
    expected_source = _sha256_file(model_path)
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != expected_source:
        return _failure_evidence(
            request_key=None,
            manifest_digest=None,
            simulator_digest=None,
            status="stale_input",
            details={"phase": "source", "detail": "source text does not match model.py"},
        )
    resolved_manifest_cache = Path(manifest_cache_dir or default_manifest_cache_dir())
    resolution_key = resolution_request_key(
        model_path, sdk_package=sdk_package, cache_dir=resolved_manifest_cache
    )
    manifest = load_collision_manifest(resolved_manifest_cache, resolution_key)
    if manifest is None:
        failure_path = _resolution_failure_cache_path(cache_path)
        same_budget_retry_consumed = False
        cached_failure = _cached_resolution_failure(
            failure_path,
            resolution_key=resolution_key,
            deadline_s=budget,
        )
        if cached_failure is not None:
            return cached_failure
        prior_failure = (
            _resolution_failure_record(failure_path, resolution_key=resolution_key).get("evidence")
            or {}
        )
        prior_execution = (
            prior_failure.get("motion_execution_status") if isinstance(prior_failure, dict) else {}
        )
        same_budget_retry_consumed = bool(
            isinstance(prior_execution, dict) and prior_execution.get("same_budget_retry_consumed")
        )
        resolution = _measured_spawn_worker(
            "resolution",
            {
                "model_path": str(model_path),
                "expected_source_sha256": expected_source,
                "sdk_package": sdk_package,
                "manifest_cache_dir": str(resolved_manifest_cache),
            },
            deadline_s=budget,
            measure=measure,
        )
        if resolution.status != "ok":
            details = {
                "phase": "resolution",
                **resolution.payload,
                "deadline_s": budget,
                "same_budget_retry_consumed": same_budget_retry_consumed,
            }
            evidence = _failure_evidence(
                request_key=None,
                manifest_digest=None,
                simulator_digest=None,
                status=resolution.status,
                details=details,
            )
            write_envelope_atomic(
                failure_path,
                {
                    "schema_version": MOTION_EVIDENCE_SCHEMA_VERSION,
                    "resolution_request_key": resolution_key,
                    "detector_policy_sha256": detector_policy_digest(),
                    "evidence": evidence,
                },
            )
            return evidence
        resolution_key = str(resolution.payload.get("resolution_request_key") or "")
        manifest = load_collision_manifest(resolved_manifest_cache, resolution_key)
        if manifest is None or (
            manifest.resolved_manifest_digest != resolution.payload.get("resolved_manifest_digest")
            or manifest.simulator_collision_digest
            != resolution.payload.get("simulator_collision_digest")
        ):
            return _failure_evidence(
                request_key=None,
                manifest_digest=None,
                simulator_digest=None,
                status="stale_input",
                details={
                    "phase": "resolution",
                    "detail": "worker final identity is unavailable or changed",
                },
            )
        if failure_path.exists():
            failure_path.unlink()
    try:
        verify_frozen_inputs(model_path, manifest, sdk_package=sdk_package)
        if manifest.source_sha256 != expected_source:
            raise StaleInputError({"reason": "source changed during resolution"})
    except (OSError, ValueError) as exc:
        return _failure_evidence(
            request_key=None,
            manifest_digest=manifest.resolved_manifest_digest,
            simulator_digest=manifest.simulator_collision_digest,
            status="stale_input",
            details={"phase": "resolution", "detail": str(exc)},
        )
    policy = PosePlanPolicy(seed=seed)
    request_key = motion_request_key(
        manifest,
        gate_identity=blueprint_gate_identity(blueprint),
        policy=policy,
        seed=seed,
    )
    cached = _cached_envelope(
        cache_path,
        request_key=request_key,
        manifest_digest=manifest.resolved_manifest_digest,
        deadline_s=budget,
    )
    if cached is not None:
        try:
            verify_frozen_inputs(model_path, manifest, sdk_package=sdk_package)
        except (OSError, ValueError) as exc:
            return _failure_evidence(
                request_key=request_key,
                manifest_digest=manifest.resolved_manifest_digest,
                simulator_digest=manifest.simulator_collision_digest,
                status="stale_input",
                details={"phase": "cache", "detail": str(exc)},
            )
        return cached
    remaining = max(0.0, budget - (time.monotonic() - started))
    scan = _measured_spawn_worker(
        "scan",
        {
            "model_path": str(model_path),
            "expected_source_sha256": expected_source,
            "sdk_package": sdk_package,
            "manifest_cache_dir": str(resolved_manifest_cache),
            "resolution_request_key": resolution_key,
            "resolved_manifest_digest": manifest.resolved_manifest_digest,
            "source": source,
            "blueprint": blueprint,
            "seed": seed,
            "policy": asdict(policy),
        },
        deadline_s=remaining,
        measure=measure,
    )
    if scan.status == "ok" and isinstance(scan.payload.get("envelope"), dict):
        envelope = dict(scan.payload["envelope"])
        try:
            verify_frozen_inputs(model_path, manifest, sdk_package=sdk_package)
            expected = {
                "schema_version": MOTION_EVIDENCE_SCHEMA_VERSION,
                "motion_request_key": request_key,
                "motion_evidence_key": motion_evidence_key(
                    request_key, manifest.resolved_manifest_digest
                ),
                "resolved_manifest_digest": manifest.resolved_manifest_digest,
                "simulator_collision_digest": manifest.simulator_collision_digest,
            }
            evidence = envelope.get("evidence")
            identity = evidence.get("motion_identity") if isinstance(evidence, dict) else None
            if not isinstance(identity, dict):
                raise StaleInputError({"reason": "scan envelope has no motion identity"})
            if any(envelope.get(key) != value for key, value in expected.items()) or (
                identity.get("resolution_request_key") != manifest.resolution_request_key
                or any(identity.get(key) != value for key, value in expected.items())
            ):
                raise StaleInputError(
                    {"reason": "scan envelope identity does not match frozen build"}
                )
        except (OSError, ValueError) as exc:
            scan = WorkerOutcome("stale_input", {"detail": str(exc)}, scan.exitcode)
        else:
            write_envelope_atomic(cache_path, envelope)
            return dict(envelope["evidence"])
    details = {
        "phase": "scan",
        **scan.payload,
        "deadline_s": budget,
        "worker_deadline_s": remaining,
        "execution_config": motion_execution_config(),
    }
    evidence = _failure_evidence(
        request_key=request_key,
        manifest_digest=manifest.resolved_manifest_digest,
        simulator_digest=manifest.simulator_collision_digest,
        status=scan.status,
        details=details,
    )
    write_envelope_atomic(
        cache_path,
        {
            "schema_version": MOTION_EVIDENCE_SCHEMA_VERSION,
            "motion_request_key": request_key,
            "resolved_manifest_digest": manifest.resolved_manifest_digest,
            "simulator_collision_digest": manifest.simulator_collision_digest,
            "evidence": evidence,
        },
    )
    return evidence


def _finding_pair(finding: dict[str, Any]) -> tuple[str, str] | None:
    raw = finding.get("pair") or finding.get("blocked_by")
    if (
        not isinstance(raw, (list, tuple))
        or len(raw) != 2
        or not all(isinstance(value, str) for value in raw)
    ):
        return None
    return str(raw[0]), str(raw[1])


def _annotate_relevant_joints(evidence: dict[str, Any]) -> dict[str, Any]:
    finalized = copy.deepcopy(evidence)
    raw_movers = finalized.get("joint_movers")
    mover_items = raw_movers.items() if isinstance(raw_movers, dict) else ()
    movers = {
        str(name): {str(part) for part in parts}
        for name, parts in mover_items
        if isinstance(parts, (list, tuple, set))
    }
    for key in MOTION_FINDING_KEYS:
        raw_findings = finalized.get(key)
        if not isinstance(raw_findings, list):
            finalized[key] = []
            continue
        for finding in raw_findings:
            if not isinstance(finding, dict):
                continue
            pair = _finding_pair(finding)
            if pair is None:
                continue
            finding["relevant_joints"] = [
                name
                for name, subtree in movers.items()
                if (pair[0] in subtree) != (pair[1] in subtree)
            ]
    return finalized


def _motion_findings_sidecar(evidence: dict[str, Any]) -> dict[str, Any]:
    identity = evidence.get("motion_identity")
    resolved_identity = identity if isinstance(identity, dict) else {}
    return {
        "schema_version": resolved_identity.get("schema_version"),
        "motion_request_key": resolved_identity.get("motion_request_key"),
        "motion_evidence_key": resolved_identity.get("motion_evidence_key"),
        "resolved_manifest_digest": resolved_identity.get("resolved_manifest_digest"),
        "motion_execution_status": evidence.get("motion_execution_status"),
        "findings": {
            key: copy.deepcopy(evidence.get(key)) if isinstance(evidence.get(key), list) else []
            for key in MOTION_FINDING_KEYS
        },
    }


def _write_motion_findings_atomic(model_path: Path, evidence: dict[str, Any]) -> None:
    sidecar_path = Path(model_path).resolve().parent / "reports" / "motion_findings.json"
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = sidecar_path.with_name(f".{sidecar_path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        temporary.write_text(
            json.dumps(_motion_findings_sidecar(evidence), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, sidecar_path)
    finally:
        temporary.unlink(missing_ok=True)


def _finalize_motion_evidence(
    model_path: Path,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    finalized = _annotate_relevant_joints(evidence)
    _write_motion_findings_atomic(model_path, finalized)
    return finalized


def _measured_spawn_worker(
    kind: str,
    payload: dict[str, Any],
    *,
    deadline_s: float,
    measure: Callable[[str], AbstractContextManager[Any]] | None,
) -> WorkerOutcome:
    with measure(f"motion.{kind}") if measure is not None else nullcontext():
        return _spawn_worker(kind, payload, deadline_s=deadline_s)


def build_motion_evidence(
    model_path: Path,
    *,
    source: str,
    cache_path: Path,
    blueprint: dict[str, Any] | None = None,
    sdk_package: str = "sdk",
    seed: int = 0,
    deadline_s: float | None = None,
    manifest_cache_dir: Path | None = None,
    measure: Callable[[str], AbstractContextManager[Any]] | None = None,
) -> dict[str, Any]:
    evidence = _build_motion_evidence_unpublished(
        model_path,
        source=source,
        cache_path=cache_path,
        blueprint=blueprint,
        sdk_package=sdk_package,
        seed=seed,
        deadline_s=deadline_s,
        manifest_cache_dir=manifest_cache_dir,
        measure=measure,
    )
    return _finalize_motion_evidence(model_path, evidence)
