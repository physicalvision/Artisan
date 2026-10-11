from __future__ import annotations

from typing import Any

from storage.revisions import (
    active_cost_path,
    active_provenance_path,
    active_traces_dir,
)
from viewer.api.schemas import (
    RecordAgentMeta,
    RecordSummaryResponse,
)
from viewer.api.store_components import ViewerStoreComponent
from viewer.api.store_values import (
    _coerce_float,
    _coerce_int,
    _coerce_rating,
    _coerce_string,
    _cost_totals,
    _effective_rating,
    _normalize_sdk_package_value,
    _thinking_level_from_provenance,
)


def _pipeline_agent_fields(
    record: dict[str, Any], cost: Any
) -> tuple[str | None, list[RecordAgentMeta], float | None, float | None]:
    run_kind = _coerce_string(record.get("run_kind"))
    if run_kind is None:
        lineage = record.get("lineage")
        if isinstance(lineage, dict) and lineage.get("origin") == "multi_agent_pipeline":
            run_kind = "pipeline"

    raw_agents = record.get("agents")
    if not isinstance(raw_agents, list) and isinstance(cost, dict):
        raw_agents = cost.get("agents")
    agents: list[RecordAgentMeta] = []
    if isinstance(raw_agents, list):
        for item in raw_agents:
            if isinstance(item, dict) and item.get("agent"):
                agents.append(RecordAgentMeta.model_validate(item))

    total_active_seconds: float | None = None
    elapsed_seconds: float | None = None
    if isinstance(cost, dict):
        total = cost.get("total")
        if isinstance(total, dict):
            total_active_seconds = _coerce_float(total.get("active_seconds"))
            elapsed_seconds = _coerce_float(total.get("elapsed_seconds"))

    return run_kind, agents, total_active_seconds, elapsed_seconds


class ViewerRecordsStore(ViewerStoreComponent):
    def _record_has_traces(self, record_id: str, record: dict[str, Any]) -> bool:
        creator = record.get("creator")
        if isinstance(creator, dict) and creator.get("trace_available") is False:
            return False
        traces_dir = active_traces_dir(self.repo, record_id, record=record)
        return traces_dir.is_dir() and any(traces_dir.iterdir())

    def _record_summary(
        self,
        record_id: str,
        summary_cache: dict[str, RecordSummaryResponse | None] | None = None,
    ) -> RecordSummaryResponse | None:
        if summary_cache is not None and record_id in summary_cache:
            return summary_cache[record_id]

        record_path = self.repo.layout.record_metadata_path(record_id)
        record = self.repo.read_json(record_path)
        if record is None:
            if summary_cache is not None:
                summary_cache[record_id] = None
            return None

        display = record.get("display") or {}
        source = record.get("source") or {}
        artifacts = record.get("artifacts") or {}
        record_dir = self.repo.layout.record_dir(record_id)

        compile_path = self.repo.layout.record_materialization_compile_report_path(record_id)
        provenance_path = active_provenance_path(self.repo, record_id, record=record)
        cost_name = artifacts.get("cost_json")
        provenance = self.repo.read_json(provenance_path)
        cost_path = (
            record_dir / str(cost_name)
            if cost_name
            else active_cost_path(self.repo, record_id, record=record)
        )
        cost = self.repo.read_json(cost_path) if cost_path.exists() else None
        materialization_status = self.materialization._materialization_status_for_record(record_id)
        has_traces = self._record_has_traces(record_id, record)
        lineage = record.get("lineage") if isinstance(record.get("lineage"), dict) else {}
        active_id = _coerce_string(record.get("active_revision_id"))
        revision_root = self.repo.layout.record_revisions_dir(record_id)
        revision_count = (
            len([path for path in revision_root.iterdir() if path.is_dir()])
            if revision_root.is_dir()
            else 0
        )

        turn_count: int | None = None
        thinking_level: str | None = None
        run_status: str | None = None
        run_message: str | None = None
        if isinstance(provenance, dict):
            run_summary = provenance.get("run_summary")
            if isinstance(run_summary, dict):
                turn_count = _coerce_int(run_summary.get("turn_count"))
                run_status = _coerce_string(run_summary.get("final_status"))
            thinking_level = _thinking_level_from_provenance(provenance)
        salvage = lineage.get("salvage")
        if isinstance(salvage, dict) and salvage.get("degraded") is True:
            run_status = "salvaged"
            run_message = _coerce_string(salvage.get("termination_reason"))

        total_cost_usd, input_tokens, output_tokens = _cost_totals(cost)
        run_kind, agents, total_active_seconds, elapsed_seconds = _pipeline_agent_fields(
            record, cost
        )

        run_id = _coerce_string(source.get("run_id")) if isinstance(source, dict) else None
        primary_rating = _coerce_rating(record.get("rating"))
        secondary_rating = _coerce_rating(record.get("secondary_rating"))
        summary = RecordSummaryResponse(
            record_id=record_id,
            title=str(display.get("title") or record_id),
            prompt_preview=str(display.get("prompt_preview") or ""),
            rating=primary_rating,
            secondary_rating=secondary_rating,
            effective_rating=_effective_rating(primary_rating, secondary_rating),
            author=_coerce_string(record.get("author")),
            rated_by=_coerce_string(record.get("rated_by")),
            secondary_rated_by=_coerce_string(record.get("secondary_rated_by")),
            created_at=record.get("created_at"),
            updated_at=record.get("updated_at"),
            viewer_asset_updated_at=self.materialization._viewer_asset_updated_at_for_record(
                record_id
            ),
            sdk_package=_normalize_sdk_package_value(record.get("sdk_package")),
            provider=record.get("provider"),
            model_id=record.get("model_id"),
            has_traces=has_traces,
            thinking_level=thinking_level,
            turn_count=turn_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_cost_usd=total_cost_usd,
            run_kind=run_kind,
            agents=agents,
            total_active_seconds=total_active_seconds,
            elapsed_seconds=elapsed_seconds,
            category_slug=record.get("category_slug"),
            run_id=run_id,
            run_status=run_status,
            run_message=run_message,
            active_revision_id=active_id,
            origin_record_id=_coerce_string(lineage.get("origin_record_id")),
            parent_record_id=_coerce_string(lineage.get("parent_record_id")),
            revision_count=revision_count,
            has_history=revision_count > 1
            or _coerce_string(lineage.get("parent_record_id")) is not None,
            collections=[str(item) for item in record.get("collections", [])],
            materialization_status=materialization_status,
            has_compile_report=compile_path.exists(),
            has_provenance=provenance_path.exists(),
            has_cost=cost_path.exists(),
        )
        if summary_cache is not None:
            summary_cache[record_id] = summary
        return summary
