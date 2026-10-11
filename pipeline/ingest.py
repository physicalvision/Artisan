from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import random
import shutil
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from pipeline.agent_meta import (
    build_pipeline_agent_meta,
    coding_turn_count,
    resolve_pipeline_provider_model,
)
from pipeline.articulated_glb import (
    ARTICULATED_APPEARANCE_FILENAME,
    ARTICULATED_GLB_FILENAME,
)
from pipeline.models import CompileSignalBundle
from pipeline.providers import pipeline_thinking
from storage.collections import CollectionStore
from storage.identifiers import (
    SLUG_RE,
    validate_record_id,
)
from storage.materialize import build_compile_fingerprint_from_inputs, sha256_file
from storage.models import (
    CompileReport as StorageCompileReport,
)
from storage.models import (
    CompileWarning,
    CreatorMetadata,
    DisplayMetadata,
    EnvironmentSettings,
    GenerationSettings,
    PromptingSettings,
    Provenance,
    Record,
    RecordArtifacts,
    RecordHashes,
    RunSummary,
    SdkSettings,
    SourceRef,
)
from storage.records import RecordStore

if TYPE_CHECKING:
    from pipeline.salvage import SalvageDecision
from storage.records_index import write_records_index
from storage.repo import StorageRepo
from storage.revisions import (
    INITIAL_REVISION_ID,
    build_revision_payload,
    revision_relative_path,
)
from storage.search import SearchIndex

logger = logging.getLogger(__name__)


MAX_SLUG_LEN = 60


def _physics_validation_provenance(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"status": "missing", "skipped_reason": "physics_validation.json missing"}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "corrupt",
            "skipped_reason": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(report, dict):
        return {"status": "corrupt", "skipped_reason": "top-level value is not an object"}
    dynamics = report.get("joint_dynamics")
    if not isinstance(dynamics, dict):
        dynamics = {}
    policy = dynamics.get("policy") if isinstance(dynamics.get("policy"), dict) else None
    policy_sha = (
        hashlib.sha256(
            json.dumps(policy, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if policy is not None
        else None
    )
    joints: list[dict[str, object]] = []
    hold_unsupported = False
    for joint in dynamics.get("joints", []) or []:
        if not isinstance(joint, dict):
            continue
        intent = joint.get("intent") if isinstance(joint.get("intent"), dict) else {}
        breakaway = joint.get("breakaway") if isinstance(joint.get("breakaway"), dict) else {}
        hold = joint.get("hold") if isinstance(joint.get("hold"), dict) else {}
        decay = joint.get("decay") if isinstance(joint.get("decay"), dict) else {}
        hold_unsupported = hold_unsupported or intent.get("hold") == "holds_anywhere"
        joints.append(
            {
                "joint": joint.get("joint"),
                "status": joint.get("status"),
                "breakaway": breakaway.get("status"),
                "hold": hold.get("status"),
                "decay": decay.get("status"),
            }
        )
    if dynamics.get("cache_hit"):
        cache = "validator-cache"
    else:
        cache = "exact-candidate"
    return {
        "status": report.get("status", dynamics.get("status", "unknown")),
        "candidate_id": report.get("candidate_id"),
        "validation_algorithm": report.get("algorithm_version"),
        "validation_policy_sha256": report.get("validation_policy_sha256"),
        "skipped_reason": dynamics.get("load_error"),
        "joint_dynamics_status": dynamics.get("status", "missing"),
        "algorithm": dynamics.get("algorithm"),
        "policy_sha256": policy_sha,
        "mjcf_sha256": dynamics.get("mjcf_sha256"),
        "cache": cache,
        "joints": joints,
        "usd45_hold_support": (
            "unsupported-passive-isaac45" if hold_unsupported else "not-requested"
        ),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _timestamp_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _slugify(text: str) -> str:
    cleaned: list[str] = []
    last = False
    for char in text.lower():
        if char.isalnum():
            cleaned.append(char)
            last = False
        elif not last:
            cleaned.append("_")
            last = True
    return "".join(cleaned).strip("_") or "pipeline"


def _category_slug_from_brief(brief: dict, blueprint: dict) -> str | None:
    raw = (
        str(brief.get("category") or "").strip()
        or str((blueprint.get("asset") or {}).get("category") or "").strip()
    )
    if not raw:
        return None
    slug = _slugify(raw)
    return slug if SLUG_RE.fullmatch(slug) else None


def _generated_record_id(prompt: str, *, run_id: str) -> str:
    base = _slugify(prompt)[:MAX_SLUG_LEN].rstrip("_") or "pipeline"
    digest = hashlib.sha1(f"{prompt}:{run_id}".encode("utf-8")).hexdigest()[:8]
    return validate_record_id(f"rec_{base}_{_timestamp_token()}_{digest}")


def _display_for_record(prompt: str, brief: dict) -> DisplayMetadata:
    title = str(brief.get("name") or "").strip()
    if not title:
        title = (prompt.splitlines()[0] if prompt else "Pipeline asset")[:120]
    preview = " ".join(prompt.split())
    if len(preview) > 160:
        preview = preview[:157].rstrip() + "..."
    return DisplayMetadata(title=title[:120], prompt_preview=preview)


def _pipeline_provenance(
    record_id: str,
    *,
    provider: str | None,
    model_id: str | None,
    thinking_level: str | None,
    turn_count: int,
    cost_usd: float | None,
    final_status: str = "success",
    prompt_manifest: dict[str, object] | None = None,
    prompt_manifest_sha256: str | None = None,
) -> Provenance:
    snapshots = prompt_manifest.get("snapshots") if prompt_manifest else None
    latest = snapshots[-1] if isinstance(snapshots, list) and snapshots else {}
    latest = latest if isinstance(latest, dict) else {}
    sdk_docs = latest.get("sdk_docs")
    sdk_docs = sdk_docs if isinstance(sdk_docs, dict) else {}
    git = latest.get("git")
    git = git if isinstance(git, dict) else {}
    system_prompts = latest.get("system_prompts")
    system_prompts = system_prompts if isinstance(system_prompts, dict) else {}
    primary_system_prompt = "pipeline/agents/prompts/coding/system.md"
    return Provenance(
        schema_version=2,
        record_id=record_id,
        generation=GenerationSettings(
            provider=provider,
            model_id=model_id,
            thinking_level=thinking_level,
            openai_transport=None,
            openai_reasoning_summary=None,
            max_turns=None,
            max_cost_usd=cost_usd,
        ),
        prompting=PromptingSettings(
            system_prompt_file=primary_system_prompt,
            system_prompt_sha256=(
                str(system_prompts[primary_system_prompt])
                if system_prompts.get(primary_system_prompt)
                else None
            ),
            prompt_manifest_file=(
                "traces/prompt_manifest.json" if prompt_manifest is not None else None
            ),
            prompt_manifest_sha256=prompt_manifest_sha256,
        ),
        sdk=SdkSettings(
            sdk_package="sdk",
            sdk_version="multi_agent_v1",
            sdk_fingerprint=str(sdk_docs["sha256"]) if sdk_docs.get("sha256") else None,
        ),
        environment=EnvironmentSettings(
            python_version=(
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            ),
            platform="sdk_python",
            git_commit=str(git["commit"]) if git.get("commit") else None,
            uv_lock_sha256=None,
        ),
        run_summary=RunSummary(
            turn_count=turn_count,
            tool_call_count=0,
            compile_attempt_count=0,
            final_status=final_status,
        ),
    )


def _write_materialization_cache(
    repo: StorageRepo,
    record_id: str,
    *,
    pipeline_exports: Path,
    model_py_sha256: str,
) -> dict[str, object]:
    mat_dir = repo.layout.record_materialization_dir(record_id)
    mat_dir.mkdir(parents=True, exist_ok=True)

    urdf_src = pipeline_exports / "object.urdf"
    if not urdf_src.is_file():
        raise FileNotFoundError(f"pipeline export missing urdf: {urdf_src}")
    shutil.copy2(urdf_src, repo.layout.record_materialization_urdf_path(record_id))

    meshes_dst = repo.layout.record_materialization_asset_meshes_dir(record_id)
    if meshes_dst.exists():
        shutil.rmtree(meshes_dst, ignore_errors=True)
    src_meshes = pipeline_exports / "assets" / "meshes"
    if src_meshes.is_dir():
        shutil.copytree(src_meshes, meshes_dst)

    articulated_src = pipeline_exports / ARTICULATED_GLB_FILENAME
    articulated_dir = repo.layout.record_materialization_asset_glb_dir(record_id)
    if articulated_src.is_file():
        articulated_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(articulated_src, articulated_dir / "articulated.glb")
    appearance_src = pipeline_exports / ARTICULATED_APPEARANCE_FILENAME
    if appearance_src.is_file():
        appearance = json.loads(appearance_src.read_text(encoding="utf-8"))
        appearance["glb"] = "assets/glb/articulated.glb"
        repo.write_json(mat_dir / ARTICULATED_APPEARANCE_FILENAME, appearance)

    fingerprint_inputs = {"model_py_sha256": model_py_sha256, "sdk_fingerprint": None}
    fingerprint = build_compile_fingerprint_from_inputs(fingerprint_inputs)
    metrics = {
        "compile_level": "full",
        "validation_level": "none",
        "fingerprint_inputs": fingerprint_inputs,
        "materialization_fingerprint": fingerprint,
    }
    report = StorageCompileReport(
        schema_version=1,
        record_id=record_id,
        status="success",
        urdf_path="model.urdf",
        warnings=[CompileWarning(code="info", message="ingested from pipeline run")],
        checks_run=["compile_sdk_ingest"],
        metrics=metrics,
        signal_bundle=CompileSignalBundle(status="success", summary="ingested").to_dict(),
    )
    repo.write_json(
        repo.layout.record_materialization_compile_report_path(record_id),
        report.to_dict(),
    )
    return {
        "materialization_dir": str(mat_dir),
        "fingerprint": fingerprint,
    }


_PUBLISH_LOCK_POLL_SECONDS = 0.25
_PUBLISH_LOCK_TIMEOUT_SECONDS = 900.0


@contextmanager
def _publish_lock(
    repo: StorageRepo, *, timeout_s: float = _PUBLISH_LOCK_TIMEOUT_SECONDS
) -> Iterator[None]:
    lock_path = repo.layout.cache_root / ".ingest.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    with open(lock_path, "w", encoding="utf-8") as handle:
        while True:
            try:
                fcntl.lockf(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"ingest publish lock {lock_path} still held after "
                        f"{timeout_s:g}s; another ingest is not releasing it"
                    ) from None
                time.sleep(_PUBLISH_LOCK_POLL_SECONDS * (0.5 + random.random()))
        try:
            yield
        finally:
            fcntl.lockf(handle.fileno(), fcntl.LOCK_UN)


def ingest_run(
    repo_root: Path,
    run_id: str,
    *,
    rating: int | None = None,
    collections: tuple[str, ...] = ("workbench",),
    salvage: "SalvageDecision | None" = None,
) -> dict[str, object]:
    repo = StorageRepo(Path(repo_root).resolve())
    workspace = repo.layout.run_dir(run_id) / "workspace"
    if not workspace.is_dir():
        raise FileNotFoundError(f"pipeline workspace missing for {run_id}: {workspace}")

    if salvage is not None:
        from pipeline.salvage import SALVAGED_EXPORTS_DIRNAME

        model_py = Path(salvage.model_path or (workspace / "model.py"))
        exports_dir = workspace / SALVAGED_EXPORTS_DIRNAME
    else:
        model_py = workspace / "model.py"
        exports_dir = workspace / "exports"
    blueprint_path = workspace / "blueprint.json"
    state_path = repo.layout.run_dir(run_id) / "pipeline_state.json"
    for required in (model_py, blueprint_path, state_path):
        if not required.is_file():
            raise FileNotFoundError(f"required pipeline artifact missing: {required}")

    blueprint = json.loads(blueprint_path.read_text(encoding="utf-8"))
    brief = blueprint.get("asset_brief", {})
    state = json.loads(state_path.read_text(encoding="utf-8"))
    prompt = str(state.get("prompt") or "").strip() or "Pipeline asset"
    cost_usd = float(state.get("cost_usd") or 0.0)
    category_slug = _category_slug_from_brief(brief, blueprint)

    record_id = _generated_record_id(prompt, run_id=run_id)
    if repo.layout.record_dir(record_id).exists():
        raise ValueError(f"record already exists: {record_id}")
    revision_id = INITIAL_REVISION_ID
    now = _utc_now()

    record_store = RecordStore(repo)
    record_store.ensure_record_dirs(record_id)
    rev_dir = repo.layout.record_revision_dir(record_id, revision_id)
    rev_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = repo.layout.record_revision_inputs_dir(record_id, revision_id)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = repo.layout.record_revision_traces_dir(record_id, revision_id)
    traces_dir.mkdir(parents=True, exist_ok=True)

    model_dst = rev_dir / "model.py"
    shutil.copy2(model_py, model_dst)
    model_py_sha256 = sha256_file(model_dst)

    prompt_path = rev_dir / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    provider, model_id = resolve_pipeline_provider_model(workspace)
    thinking_level = pipeline_thinking("CODING")
    agents, cost_total = build_pipeline_agent_meta(
        workspace, state, provider=provider, model=model_id
    )
    turn_count = coding_turn_count(agents)
    prompt_manifest_path = workspace / "trace" / "prompt_manifest.json"
    prompt_manifest: dict[str, object] | None = None
    prompt_manifest_sha256: str | None = None
    if prompt_manifest_path.is_file():
        loaded_manifest = json.loads(prompt_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_manifest, dict):
            raise ValueError(f"prompt manifest must be an object: {prompt_manifest_path}")
        snapshots = loaded_manifest.get("snapshots")
        if loaded_manifest.get("schema_version") != 1 or not isinstance(snapshots, list):
            raise ValueError(f"unsupported prompt manifest: {prompt_manifest_path}")
        prompt_manifest = loaded_manifest
        prompt_manifest_sha256 = sha256_file(prompt_manifest_path)
    provenance = _pipeline_provenance(
        record_id,
        provider=provider,
        model_id=model_id,
        thinking_level=thinking_level,
        turn_count=turn_count,
        cost_usd=cost_usd,
        final_status="salvaged" if salvage is not None else "success",
        prompt_manifest=prompt_manifest,
        prompt_manifest_sha256=prompt_manifest_sha256,
    )
    record_store.write_provenance(record_id, provenance, revision_id=revision_id)

    cost_payload = {
        "schema_version": 2,
        "record_id": record_id,
        "total": cost_total,
        "agents": agents,
        "run_kind": "pipeline",
        "total_usd": cost_usd,
        "source": "pipeline.run_summary.cost_by_agent",
    }
    repo.write_json(rev_dir / "cost.json", cost_payload)

    design_board = workspace / "references" / "design_board.png"
    if design_board.is_file():
        shutil.copy2(design_board, inputs_dir / "design_board.png")

    pipeline_sidecar = rev_dir / "pipeline"
    pipeline_sidecar.mkdir(parents=True, exist_ok=True)
    shutil.copy2(blueprint_path, pipeline_sidecar / "blueprint.json")
    manip_manifest_src = exports_dir / "manipulation.json"
    if manip_manifest_src.is_file():
        shutil.copy2(manip_manifest_src, pipeline_sidecar / "manipulation.json")
    tasks_json_src = workspace / "tasks.json"
    if tasks_json_src.is_file():
        shutil.copy2(tasks_json_src, pipeline_sidecar / "tasks.json")
    reports_src = workspace / "reports"
    if reports_src.is_dir():
        shutil.copytree(reports_src, pipeline_sidecar / "reports", dirs_exist_ok=True)
    export_manifest = exports_dir / "export_manifest.json"
    if export_manifest.is_file():
        shutil.copy2(export_manifest, pipeline_sidecar / "export_manifest.json")
    candidate_manifest = workspace / "candidate" / "manifest.json"
    if candidate_manifest.is_file():
        shutil.copy2(candidate_manifest, pipeline_sidecar / "candidate_manifest.json")
    trace_src = workspace / "trace"
    if trace_src.is_dir():
        for f in trace_src.iterdir():
            if f.is_file():
                shutil.copy2(f, traces_dir / f.name)
        review_trace = trace_src / "review" / "trajectory.jsonl"
        if review_trace.is_file():
            shutil.copy2(review_trace, traces_dir / "review_trajectory.jsonl")

    artifacts_payload = {
        "prompt_txt": revision_relative_path(revision_id, "prompt.txt"),
        "prompt_series_json": None,
        "model_py": revision_relative_path(revision_id, "model.py"),
        "provenance_json": revision_relative_path(revision_id, "provenance.json"),
        "cost_json": revision_relative_path(revision_id, "cost.json"),
        "inputs_dir": revision_relative_path(revision_id, "inputs"),
        "traces_dir": revision_relative_path(revision_id, "traces"),
    }
    hashes = {"prompt_sha256": prompt_sha256, "model_py_sha256": model_py_sha256}
    revision_payload = build_revision_payload(
        record_id=record_id,
        revision_id=revision_id,
        created_at=now,
        prompt_text=prompt,
        prompt_kind="single_prompt",
        source=SourceRef(run_id=run_id).to_dict(),
        generation=provenance.generation.to_dict(),
        artifacts=artifacts_payload,
        hashes=hashes,
        run_summary=provenance.run_summary.to_dict(),
    )
    revision_payload["physics_validation"] = _physics_validation_provenance(
        reports_src / "physics_validation.json"
    )
    repo.write_json(
        repo.layout.record_revision_metadata_path(record_id, revision_id),
        revision_payload,
    )

    record = Record(
        schema_version=3,
        record_id=record_id,
        created_at=now,
        updated_at=now,
        rating=rating,
        secondary_rating=None,
        kind="generated_model",
        prompt_kind="single_prompt",
        category_slug=category_slug,
        source=SourceRef(run_id=run_id),
        sdk_package="sdk",
        provider=provider,
        model_id=model_id,
        display=_display_for_record(prompt, brief),
        artifacts=RecordArtifacts(
            prompt_txt=artifacts_payload["prompt_txt"],
            prompt_series_json=None,
            model_py=artifacts_payload["model_py"],
            provenance_json=artifacts_payload["provenance_json"],
            cost_json=artifacts_payload["cost_json"],
            inputs_dir=artifacts_payload["inputs_dir"],
            traces_dir=artifacts_payload["traces_dir"],
        ),
        hashes=RecordHashes(
            prompt_sha256=prompt_sha256,
            model_py_sha256=model_py_sha256,
        ),
        collections=list(collections),  # type: ignore[arg-type]
        active_revision_id=revision_id,
        lineage=(
            {"origin": "multi_agent_pipeline", "run_id": run_id}
            if salvage is None
            else {
                "origin": "multi_agent_pipeline",
                "run_id": run_id,
                "salvage": salvage.as_metadata(),
            }
        ),
        creator=CreatorMetadata(
            mode="pipeline",
            agent=None,
            trace_available=True,
        ),
        author=None,
        rated_by=None,
        secondary_rated_by=None,
        build_backend="sdk_python",
        run_kind="pipeline",
        agents=agents,
    )
    mat_summary: dict[str, object] = {}
    with _publish_lock(repo):
        record_store.write_record(record)

        if "workbench" in collections:
            CollectionStore(repo).upsert_workbench_entry(
                record_id=record_id,
                added_at=now,
                label=record.display.title,
                tags=(
                    ["pipeline"]
                    if salvage is None
                    else ["pipeline", "salvaged", f"salvage:{salvage.grade}"]
                ),
            )

        if exports_dir.is_dir():
            mat_summary = _write_materialization_cache(
                repo,
                record_id,
                pipeline_exports=exports_dir,
                model_py_sha256=model_py_sha256,
            )
        else:
            logger.warning("pipeline %s has no exports/ — materialization cache empty", run_id)

        write_records_index(repo)
        SearchIndex(repo).rebuild()

    return {
        "record_id": record_id,
        "record_dir": str(repo.layout.record_dir(record_id)),
        "revision_id": revision_id,
        "category_slug": category_slug,
        **mat_summary,
    }
