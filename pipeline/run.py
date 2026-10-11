from __future__ import annotations

import os
import uuid
from pathlib import Path

from artisan.physics_policy import penetration_depth_tolerance_from_env
from pipeline import workspace
from pipeline.agents.coding import CodingAgent
from pipeline.agents.design import DesignAgent
from pipeline.agents.review import ReviewAgent
from pipeline.backends.sdk_build import SDK_CORPUS_ROOT, SdkBackend
from pipeline.llm.model_registry import validate_runtime_configuration
from pipeline.nodes import NodeBundle, build_nodes
from pipeline.orchestrator import PipelineContext, resume_pipeline, run_pipeline
from pipeline.policies import DEFAULT_POLICIES
from pipeline.prompt_manifest import capture_pipeline_prompt_manifest
from pipeline.state import MEDIA_KIND_IMAGE, PipelineState, load_checkpoint


def _bundle(state: PipelineState, max_cost_usd: float) -> tuple[NodeBundle, dict]:
    os.environ.setdefault("ARTISAN_REQUIRE_EST_MASS", "1")
    from pipeline.cassette import Cassette

    repo_root = Path(state.repo_root)
    paths = workspace.ensure(repo_root, state.run_id)
    state.references = workspace.stage_references(
        repo_root, state.run_id, state.references, prompt=state.prompt
    )
    _img = state.image_references()
    if _img:
        state.image_path = _img[0]
    trace_dir = paths["root"] / "trace"
    from pipeline.providers import (
        effective_pipeline_model,
        pipeline_provider,
        pipeline_thinking,
    )

    agent_settings = {
        role: {
            "provider": pipeline_provider(role),
            "model_id": effective_pipeline_model(role),
            "reasoning_level": pipeline_thinking(role),
        }
        for role in ("design", "coding", "review")
    }
    capture_pipeline_prompt_manifest(
        trace_dir / "prompt_manifest.json",
        source_root=Path(__file__).resolve().parents[1],
        phase=state.phase.value,
        agent_settings=agent_settings,
    )
    cassette = Cassette.from_env(paths["root"] / "cassette.jsonl")
    backend = SdkBackend()
    from pipeline.timing import TimingRecorder

    recorder = TimingRecorder(trace_dir / "timing.jsonl")
    bundle = NodeBundle(
        state=state,
        paths=paths,
        design=DesignAgent(
            blueprint_path=paths["blueprint"],
            session_path=paths["sessions"] / "design.json",
            selfcheck_report_path=paths["p1_selfcheck"],
            trace_path=trace_dir / "design.jsonl",
            max_cost_usd=max_cost_usd,
            cassette=cassette,
            recorder=recorder,
        ),
        review=ReviewAgent(
            paths["model"],
            backend=backend,
            session_path=paths["sessions"] / "review.json",
            max_cost_usd=max_cost_usd,
            trace_dir=trace_dir / "review",
            render_root=paths["renders"],
            qa_test_path=paths["root"] / "qa_tests.py",
            cassette=cassette,
            recorder=recorder,
        ),
        coding=CodingAgent(
            paths["model"],
            SDK_CORPUS_ROOT,
            paths["sessions"] / "coding.json",
            max_cost_usd=max_cost_usd,
            trace_dir=trace_dir,
            cassette=cassette,
            backend=backend,
            render_root=paths["renders"],
            recorder=recorder,
        ),
        backend=backend,
        recorder=recorder,
    )
    return bundle, build_nodes(bundle)


async def run(
    prompt: str,
    *,
    repo_root: Path,
    max_cost_usd: float = 10.0,
    run_id: str | None = None,
    image_paths: list[str] | None = None,
    image_path: str | None = None,
) -> PipelineState:
    validate_runtime_configuration()
    penetration_depth_tolerance_from_env()
    run_id = run_id or f"pl_{uuid.uuid4().hex[:12]}"
    paths = [p for p in (image_paths or ([image_path] if image_path else [])) if p]
    references = [{"path": p, "media_kind": MEDIA_KIND_IMAGE} for p in paths]
    state = PipelineState(
        run_id=run_id,
        prompt=prompt,
        record_id=run_id,
        repo_root=str(repo_root),
        image_path=paths[0] if paths else None,
        references=references,
    )
    bundle, nodes = _bundle(state, max_cost_usd)
    ctx = PipelineContext(
        repo_root=repo_root,
        backend=bundle.backend,
        policies=DEFAULT_POLICIES,
        max_cost_usd=max_cost_usd,
    )
    return await run_pipeline(state, ctx, nodes)


async def resume(run_id: str, *, repo_root: Path, max_cost_usd: float = 10.0) -> PipelineState:
    validate_runtime_configuration()
    penetration_depth_tolerance_from_env()
    state = load_checkpoint(repo_root, run_id)
    bundle, nodes = _bundle(state, max_cost_usd)
    ctx = PipelineContext(
        repo_root=repo_root,
        backend=bundle.backend,
        policies=DEFAULT_POLICIES,
        max_cost_usd=max_cost_usd,
    )
    return await resume_pipeline(repo_root, run_id, ctx, nodes, state=state)
