from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Query

from storage.identifiers import validate_record_id
from viewer.api.dependencies import ViewerStoreDep
from viewer.api.schemas import RecordSummaryResponse

router = APIRouter()


@router.get("/api/records", response_model=list[RecordSummaryResponse])
async def records(
    store: ViewerStoreDep,
    q: str = "",
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[RecordSummaryResponse]:
    def load() -> list[RecordSummaryResponse]:
        summaries = []
        query = q.casefold().split()
        for record_dir in store.repo.layout.records_root.iterdir():
            if not record_dir.is_dir() or not record_dir.resolve().is_relative_to(store.repo.root):
                continue
            try:
                validate_record_id(record_dir.name)
            except ValueError:
                continue
            summary = store.records._record_summary(record_dir.name)
            if summary is None:
                continue
            searchable = f"{summary.title} {summary.prompt_preview} {summary.run_id or ''} {summary.record_id}".casefold()
            if all(term in searchable for term in query):
                summaries.append(summary)
        summaries.sort(
            key=lambda row: (row.updated_at or row.created_at or "", row.record_id), reverse=True
        )
        return summaries[offset : offset + limit]

    return await asyncio.to_thread(load)


@router.get("/api/records/{record_id}/summary", response_model=RecordSummaryResponse)
async def record_summary(record_id: str, store: ViewerStoreDep) -> RecordSummaryResponse:
    try:
        validate_record_id(record_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid record ID format") from exc
    if not store.repo.layout.record_dir(record_id).resolve().is_relative_to(store.repo.root):
        raise HTTPException(status_code=403, detail="Access denied")
    summary = await asyncio.to_thread(store.records._record_summary, record_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"Record not found: {record_id}")
    return summary
