from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from storage.repo import StorageRepo
from storage.revisions import active_cost_path, active_provenance_path, active_traces_dir

RECORDS_INDEX_SCHEMA_VERSION = 1


class RecordsIndexError(ValueError):
    pass


def load_records_index(repo: StorageRepo) -> list[dict[str, Any]]:
    path = repo.layout.records_index_path
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    seen_record_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise RecordsIndexError(
                    f"{path}: line {line_number}: invalid JSON: {exc.msg} at column {exc.colno}"
                ) from exc
            if not isinstance(parsed, dict):
                raise RecordsIndexError(f"{path}: line {line_number}: row must be a JSON object")
            if parsed.get("schema_version") != RECORDS_INDEX_SCHEMA_VERSION:
                raise RecordsIndexError(
                    f"{path}: line {line_number}: unsupported schema_version="
                    f"{parsed.get('schema_version')!r}"
                )
            record_id = _string_or_none(parsed.get("record_id"))
            if record_id is None:
                raise RecordsIndexError(f"{path}: line {line_number}: record_id is required")
            if record_id in seen_record_ids:
                raise RecordsIndexError(
                    f"{path}: line {line_number}: duplicate record_id={record_id}"
                )
            seen_record_ids.add(record_id)
            rows.append(parsed)
    return rows


def records_index_by_id(repo: StorageRepo) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("record_id")): row
        for row in load_records_index(repo)
        if isinstance(row.get("record_id"), str)
    }


def find_record_index_row(repo: StorageRepo, record_id: str) -> dict[str, Any] | None:
    row = records_index_by_id(repo).get(record_id)
    return row if isinstance(row, dict) else None


def build_records_index_rows(repo: StorageRepo) -> list[dict[str, Any]]:
    category_titles = _category_titles(repo)
    rows_by_id: dict[str, dict[str, Any]] = {
        str(row.get("record_id")): row
        for row in load_records_index(repo)
        if isinstance(row.get("record_id"), str)
    }
    records_root = repo.layout.records_root
    if not records_root.exists():
        return [rows_by_id[record_id] for record_id in sorted(rows_by_id)]

    for record_dir in sorted(path for path in records_root.iterdir() if path.is_dir()):
        row = _build_record_index_row(repo, record_dir, category_titles)
        if row is not None:
            rows_by_id[str(row["record_id"])] = row
    return [rows_by_id[record_id] for record_id in sorted(rows_by_id)]


def write_records_index(repo: StorageRepo) -> list[dict[str, Any]]:
    rows = build_records_index_rows(repo)
    text = "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows)
    repo.write_text(repo.layout.records_index_path, text)
    return rows


def remove_records_from_index(repo: StorageRepo, record_ids: list[str]) -> list[dict[str, Any]]:
    removed = set(record_ids)
    rows = [
        row for row in load_records_index(repo) if str(row.get("record_id") or "") not in removed
    ]
    text = "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows)
    repo.write_text(repo.layout.records_index_path, text)
    return rows


def _build_record_index_row(
    repo: StorageRepo,
    record_dir: Path,
    category_titles: dict[str, str],
) -> dict[str, Any] | None:
    record_path = record_dir / "record.json"
    record = repo.read_json(record_path, default=None)
    if not isinstance(record, dict):
        return None

    record_id = str(record.get("record_id") or record_dir.name)
    display = record.get("display") if isinstance(record.get("display"), dict) else {}
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    artifacts = record.get("artifacts") if isinstance(record.get("artifacts"), dict) else {}
    creator = record.get("creator") if isinstance(record.get("creator"), dict) else {}
    lineage = record.get("lineage") if isinstance(record.get("lineage"), dict) else {}
    collections = record.get("collections") if isinstance(record.get("collections"), list) else []

    provenance_path = active_provenance_path(repo, record_id, record=record)
    provenance = repo.read_json(provenance_path, default=None) if provenance_path.exists() else None
    cost_name = artifacts.get("cost_json")
    cost_path = (
        record_dir / str(cost_name)
        if cost_name
        else active_cost_path(repo, record_id, record=record)
    )
    cost = repo.read_json(cost_path, default=None) if cost_path.exists() else None
    total_cost_usd, input_tokens, output_tokens = _cost_totals(cost)

    generation = provenance.get("generation") if isinstance(provenance, dict) else {}
    run_summary = provenance.get("run_summary") if isinstance(provenance, dict) else {}
    salvage = lineage.get("salvage")
    degraded = isinstance(salvage, dict) and salvage.get("degraded") is True
    run_status = (
        "salvaged"
        if degraded
        else _string_or_none(run_summary.get("final_status"))
        if isinstance(run_summary, dict)
        else None
    )
    active_revision_id = _string_or_none(record.get("active_revision_id"))
    revision_root = repo.layout.record_revisions_dir(record_id)
    revision_count = (
        len([path for path in revision_root.iterdir() if path.is_dir()])
        if revision_root.is_dir()
        else 0
    )
    creator_mode = _string_or_none(creator.get("mode")) if isinstance(creator, dict) else None
    category_slug = _string_or_none(record.get("category_slug"))

    has_traces = not (isinstance(creator, dict) and creator.get("trace_available") is False)
    if has_traces:
        traces_dir = active_traces_dir(repo, record_id, record=record)
        has_traces = traces_dir.is_dir() and any(traces_dir.iterdir())

    primary_rating = _coerce_rating(record.get("rating"))
    secondary_rating = _coerce_rating(record.get("secondary_rating"))

    return {
        "schema_version": RECORDS_INDEX_SCHEMA_VERSION,
        "record_id": record_id,
        "category_slug": category_slug,
        "category_title": category_titles.get(category_slug or ""),
        "title": str(display.get("title") or record_id),
        "prompt_preview": str(display.get("prompt_preview") or ""),
        "rating": primary_rating,
        "secondary_rating": secondary_rating,
        "effective_rating": _effective_rating(primary_rating, secondary_rating),
        "author": _string_or_none(record.get("author")),
        "rated_by": _string_or_none(record.get("rated_by")),
        "secondary_rated_by": _string_or_none(record.get("secondary_rated_by")),
        "created_at": _string_or_none(record.get("created_at")),
        "updated_at": _string_or_none(record.get("updated_at")),
        "sdk_package": _string_or_none(record.get("sdk_package")),
        "provider": _string_or_none(record.get("provider")),
        "model_id": _string_or_none(record.get("model_id")),
        "run_kind": _string_or_none(record.get("run_kind")),
        "creator_mode": creator_mode,
        "agent_harness": "pipeline",
        "has_traces": has_traces,
        "thinking_level": _string_or_none(generation.get("thinking_level"))
        if isinstance(generation, dict)
        else None,
        "turn_count": _coerce_int(run_summary.get("turn_count"))
        if isinstance(run_summary, dict)
        else None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_cost_usd": total_cost_usd,
        "run_id": _string_or_none(source.get("run_id")) if isinstance(source, dict) else None,
        "run_status": run_status,
        "active_revision_id": active_revision_id,
        "origin_record_id": _string_or_none(lineage.get("origin_record_id")),
        "parent_record_id": _string_or_none(lineage.get("parent_record_id")),
        "revision_count": revision_count,
        "has_history": revision_count > 1
        or _string_or_none(lineage.get("parent_record_id")) is not None,
        "collections": [str(item) for item in collections],
        "has_provenance": provenance_path.exists(),
        "has_cost": cost_path.exists(),
        "has_compile_report": repo.layout.record_materialization_compile_report_path(
            record_id
        ).exists(),
    }


def _category_titles(repo: StorageRepo) -> dict[str, str]:
    titles: dict[str, str] = {}
    root = repo.layout.categories_root
    if not root.exists():
        return titles
    for path in root.glob("*/category.json"):
        category = repo.read_json(path, default=None)
        if not isinstance(category, dict):
            continue
        slug = _string_or_none(category.get("slug"))
        title = _string_or_none(category.get("title"))
        if slug and title:
            titles[slug] = title
    return titles


def _string_or_none(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _coerce_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _coerce_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _coerce_rating(value: Any) -> int | None:
    return value if isinstance(value, int) and 1 <= value <= 5 else None


def _effective_rating(primary_rating: int | None, secondary_rating: int | None) -> float | None:
    ratings = [float(value) for value in (primary_rating, secondary_rating) if value is not None]
    return sum(ratings) / len(ratings) if ratings else None


def _cost_totals(cost: Any) -> tuple[float | None, int | None, int | None]:
    if not isinstance(cost, dict):
        return None, None, None
    total = cost.get("total")
    if isinstance(total, dict):
        costs_usd = total.get("costs_usd")
        tokens = total.get("tokens")
        total_cost_usd = (
            _coerce_float(costs_usd.get("total")) if isinstance(costs_usd, dict) else None
        )
        input_tokens = (
            _coerce_int(tokens.get("prompt_tokens")) if isinstance(tokens, dict) else None
        )
        output_tokens = (
            _coerce_int(tokens.get("candidates_tokens")) if isinstance(tokens, dict) else None
        )
    else:
        total_cost_usd = input_tokens = output_tokens = None
    if total_cost_usd is None:
        total_cost_usd = _coerce_float(cost.get("total_usd"))
    return total_cost_usd, input_tokens, output_tokens
