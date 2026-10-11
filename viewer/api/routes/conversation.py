from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from viewer.api.conversation import (
    ConversationBuilder,
    list_pipeline_runs,
    validate_run_id,
)
from viewer.api.dependencies import ViewerStoreDep
from viewer.api.media import file_cache_control, file_media_type
from viewer.api.schemas import ConversationResponse, PipelineRunSummary

router = APIRouter()


@router.get("/api/pipeline/runs", response_model=list[PipelineRunSummary])
async def pipeline_runs(store: ViewerStoreDep) -> list[PipelineRunSummary]:
    return await asyncio.to_thread(list_pipeline_runs, store.repo)


@router.get(
    "/api/pipeline/runs/{run_id}/conversation",
    response_model=ConversationResponse,
)
async def run_conversation(run_id: str, store: ViewerStoreDep) -> ConversationResponse:
    if not validate_run_id(run_id):
        raise HTTPException(status_code=400, detail="Invalid run ID format")

    def _build() -> ConversationResponse | None:
        builder = ConversationBuilder(store.repo, run_id)
        if not builder.exists():
            return None
        return builder.build()

    try:
        response = await asyncio.to_thread(_build)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Access denied") from exc
    if response is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return response


@router.get("/api/pipeline/runs/{run_id}/conversation/image")
async def run_conversation_image(
    run_id: str,
    store: ViewerStoreDep,
    src: str = Query(..., description="Opaque image token from the conversation payload"),
) -> FileResponse:
    if not validate_run_id(run_id):
        raise HTTPException(status_code=400, detail="Invalid run ID format")

    def _resolve() -> Path | None:
        builder = ConversationBuilder(store.repo, run_id)
        if not builder.exists():
            return None
        builder.build()
        return builder.allow.get(src)

    try:
        target = await asyncio.to_thread(_resolve)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Access denied") from exc
    if target is None or not target.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(
        target,
        media_type=file_media_type(target),
        headers={"Cache-Control": file_cache_control(immutable=False)},
    )
