from __future__ import annotations

import difflib
import hashlib
import importlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from rich.console import Console

from artisan.values import normalize_model_id
from pipeline.agents.prompts.coding import load_coding_prompt
from pipeline.context_policy import (
    DISABLED_COMPRESSION_POLICY,
    cap_tool_output,
    dedup_read,
    enable_native_compaction,
    estimate_conversation_tokens,
    policy_from_env,
    round_floats,
    sha1_text,
)
from pipeline.cost import build_cost_tracker
from pipeline.harness_base import _ToolLoopHarness
from pipeline.harness_codec import MessageCodec
from pipeline.llm.factory import (
    ProviderConfig,
    create_provider_client,
)
from pipeline.policies import DEFAULT_POLICIES, Policies
from pipeline.prompt_cache import build_openai_prompt_cache_settings
from pipeline.providers import resolve_pipeline_model
from pipeline.timing import _null_measure
from pipeline.traces import TraceWriter
from pipeline.tui.single_run import SingleRunDisplay
from sdk.runtime import runtime as sdk_runtime
from sdk.runtime.accepted_inspection import current_reference, guarded_inspection

logger = logging.getLogger(__name__)


def _driver_reach_block_fraction() -> float:
    raw = os.environ.get("PIPELINE_DRIVER_REACH_BLOCK_FRACTION", "0.7")
    try:
        return float(raw)
    except ValueError:
        return 0.7


MODEL_FILENAME = "model.py"
PROBE_TIMEOUT_S = 120.0
_FEEDBACK_FINDING_LIMIT = 64
_FEEDBACK_IDENTITY_LIMIT = 512
_FEEDBACK_MAX_CHARS = 48_000
_CANDIDATE_WARNING_FAMILIES = (
    "analytic_primitives",
    "coacd",
    "collision_repair",
    "collision_qa",
    "hinge_primitives",
    "cavity_fallback",
    "convex_hull_fallback",
)
PREVIEW_MAX_DEPTH_M = 0.015
_NUMERICAL_FREE_ROOT_CODES = frozenset({"physics_explosion", "non_finite"})
_GROUNDED_SUPPORT_CODES = frozenset(
    {
        "no_ground_contact",
        "ground_penetration",
        "persistent_motion",
        "initial_com_outside_support",
        "final_com_outside_support",
        "tilt_exceeded",
        "xy_drift_exceeded",
        "root_z_displacement",
    }
)
PREVIEW_MAX_VOLUME_M3 = 1e-5


def _candidate_warning_summary(warnings: list[str]) -> str | None:
    if not warnings:
        return None
    counts = {family: 0 for family in (*_CANDIDATE_WARNING_FAMILIES, "other")}
    for warning in warnings:
        prefix, separator, _detail = warning.partition(":")
        family = prefix.strip() if separator else ""
        counts[family if family in _CANDIDATE_WARNING_FAMILIES else "other"] += 1
    nonzero = [
        f"{family}={counts[family]}"
        for family in (*_CANDIDATE_WARNING_FAMILIES, "other")
        if counts[family]
    ]
    return f"CANDIDATE WARNINGS: total={len(warnings)}; families: {', '.join(nonzero)}"


def _finite_feedback_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _eligible_candidate_preview(blocks: list[dict[str, Any]]) -> bool:
    if not 1 <= len(blocks) <= 3:
        return False
    metric_fields = {
        "penetration": ("penetration_depth", "penetration_volume"),
        "connection_ram": ("penetration_span", "penetration_volume"),
    }
    for finding in blocks:
        kind = finding.get("kind")
        pair = finding.get("pair")
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or not all(isinstance(value, str) for value in pair)
        ):
            return False
        if kind == "motion_contact":
            gate = finding.get("gate_evaluation")
            if not isinstance(gate, dict):
                return False
            depth = _finite_feedback_number(gate.get("max_penetration_depth_m"))
            volume = _finite_feedback_number(finding.get("overlap_volume"))
        elif kind in metric_fields:
            depth_field, volume_field = metric_fields[kind]
            depth = _finite_feedback_number(finding.get(depth_field))
            volume = _finite_feedback_number(finding.get(volume_field))
        else:
            return False
        if (
            depth is None
            or volume is None
            or depth < 0.0
            or volume < 0.0
            or depth > PREVIEW_MAX_DEPTH_M
            or volume > PREVIEW_MAX_VOLUME_M3
        ):
            return False
    return True


def _descending_feedback_metric(value: Any) -> tuple[bool, float]:
    number = _finite_feedback_number(value)
    return (number is None, -number if number is not None else 0.0)


def _feedback_string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    if value is None:
        return ()
    return (str(value),)


def _feedback_sort_key(
    kind: str,
    finding: dict[str, Any],
    ordinal: int,
) -> tuple[Any, ...]:
    pair = _feedback_string_tuple(finding.get("pair"))
    elements = _feedback_string_tuple(finding.get("elements"))
    if kind == "penetration":
        return (
            _descending_feedback_metric(finding.get("penetration_depth")),
            _descending_feedback_metric(finding.get("penetration_volume")),
            pair,
            elements,
            ordinal,
        )
    if kind == "motion_contact":
        return (
            _descending_feedback_metric(finding.get("overlap_volume")),
            pair,
            elements,
            ordinal,
        )
    if kind == "connection_ram":
        return (
            _descending_feedback_metric(finding.get("penetration_volume")),
            _descending_feedback_metric(finding.get("penetration_span")),
            pair,
            ordinal,
        )
    if kind == "driver_reach":
        fraction = _finite_feedback_number(finding.get("reachable_fraction"))
        return (
            fraction is None,
            fraction if fraction is not None else 0.0,
            str(finding.get("joint") or ""),
            ordinal,
        )
    if kind == "geometry_island":
        disconnected = finding.get("disconnected")
        count = len(disconnected) if isinstance(disconnected, (list, tuple)) else 0
        return (-count, str(finding.get("part") or ""), ordinal)
    if kind == "candidate_mjcf_penetration":
        return (
            _descending_feedback_metric(finding.get("penetration_depth_m")),
            pair,
            ordinal,
        )
    return (ordinal,)


def _ordered_feedback_findings(
    kind: str,
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    indexed = list(enumerate(findings))
    indexed.sort(key=lambda item: _feedback_sort_key(kind, item[1], item[0]))
    return [finding for _ordinal, finding in indexed]


def _clip_feedback_identity(value: Any, stats: dict[str, int]) -> str:
    text = str(value)
    if len(text) <= _FEEDBACK_IDENTITY_LIMIT:
        return text
    omitted = len(text) - 480
    stats["values_clipped"] += 1
    stats["characters_omitted"] += omitted
    return f"{text[:240]}...<omitted={omitted}>...{text[-240:]}"


_COMPILE_FEEDBACK_LINE_TAG = "compile_feedback_line"
_COMPILE_FEEDBACK_ACCOUNTING_TAG = "compile_feedback_accounting"


@dataclass(frozen=True)
class _CompileFeedbackLine:
    text: str
    category: str | None = None
    bind_records: int = 0
    provenance_steps: int = 0
    preserve: bool = False
    blocking: bool = False


def _empty_compile_feedback_summary() -> dict[str, Any]:
    return {
        "categories": {},
        "nested": {
            "bind_records": {"total": 0, "rendered": 0, "suppressed": 0},
            "provenance_steps": {"total": 0, "rendered": 0, "suppressed": 0},
        },
        "identity_clipping": {"values_clipped": 0, "characters_omitted": 0},
    }


def _merge_compile_feedback_summary(
    target: dict[str, Any],
    incoming: dict[str, Any],
) -> None:
    for section in ("categories", "nested"):
        raw_section = incoming.get(section)
        if not isinstance(raw_section, dict):
            continue
        target_section = target[section]
        for name, raw_counts in raw_section.items():
            if not isinstance(raw_counts, dict):
                continue
            counts = target_section.setdefault(
                str(name), {"total": 0, "rendered": 0, "suppressed": 0}
            )
            for field_name in ("total", "rendered", "suppressed"):
                value = raw_counts.get(field_name)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    counts[field_name] += value
    raw_identity = incoming.get("identity_clipping")
    if isinstance(raw_identity, dict):
        for field_name in ("values_clipped", "characters_omitted"):
            value = raw_identity.get(field_name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                target["identity_clipping"][field_name] += value


def _wrapped_feedback_json(line: str, tag: str) -> dict[str, Any] | None:
    stripped = line.strip()
    opening = f"<{tag}>"
    closing = f"</{tag}>"
    if not stripped.startswith(opening) or not stripped.endswith(closing):
        return None
    try:
        payload = json.loads(stripped[len(opening) : -len(closing)])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _parse_compile_feedback(body: str) -> tuple[list[_CompileFeedbackLine], dict[str, Any]]:
    lines: list[_CompileFeedbackLine] = []
    summary = _empty_compile_feedback_summary()
    opening = f"<{_COMPILE_FEEDBACK_LINE_TAG}>"
    closing = f"</{_COMPILE_FEEDBACK_LINE_TAG}>"
    section: str | None = None
    for source_line in body.splitlines():
        if source_line in {"<failures>", "<warnings>", "<notes>"}:
            section = source_line[1:-1]
            lines.append(_CompileFeedbackLine(source_line, category="compile_report"))
            continue
        if source_line in {"</failures>", "</warnings>", "</notes>"}:
            lines.append(_CompileFeedbackLine(source_line, category="compile_report"))
            section = None
            continue
        accounting = _wrapped_feedback_json(source_line, _COMPILE_FEEDBACK_ACCOUNTING_TAG)
        if accounting is not None:
            _merge_compile_feedback_summary(summary, accounting)
            continue
        marker_index = source_line.rfind(opening)
        if marker_index < 0 or not source_line.endswith(closing):
            lines.append(
                _CompileFeedbackLine(
                    source_line,
                    category="compile_report",
                    blocking=section == "failures",
                )
            )
            continue
        try:
            metadata = json.loads(source_line[marker_index + len(opening) : -len(closing)])
        except json.JSONDecodeError:
            lines.append(
                _CompileFeedbackLine(
                    source_line,
                    category="compile_report",
                    blocking=section == "failures",
                )
            )
            continue
        if not isinstance(metadata, dict):
            lines.append(
                _CompileFeedbackLine(
                    source_line,
                    category="compile_report",
                    blocking=section == "failures",
                )
            )
            continue
        category = metadata.get("category")
        lines.append(
            _CompileFeedbackLine(
                text=source_line[:marker_index].rstrip(),
                category=str(category) if isinstance(category, str) else None,
                bind_records=(
                    int(metadata.get("bind_records", 0))
                    if isinstance(metadata.get("bind_records", 0), int)
                    else 0
                ),
                provenance_steps=(
                    int(metadata.get("provenance_steps", 0))
                    if isinstance(metadata.get("provenance_steps", 0), int)
                    else 0
                ),
                preserve=metadata.get("preserve") is True,
                blocking=section == "failures",
            )
        )

    observed_categories: dict[str, int] = {}
    for line in lines:
        if line.category and line.category != "compile_report":
            observed_categories[line.category] = observed_categories.get(line.category, 0) + 1
    for category, observed in observed_categories.items():
        counts = summary["categories"].setdefault(
            category, {"total": 0, "rendered": 0, "suppressed": 0}
        )
        counts["rendered"] = max(counts["rendered"], observed)
        counts["total"] = max(counts["total"], counts["rendered"] + counts["suppressed"])
    for name, observed in (
        ("bind_records", sum(line.bind_records for line in lines)),
        ("provenance_steps", sum(line.provenance_steps for line in lines)),
    ):
        counts = summary["nested"][name]
        counts["rendered"] = max(counts["rendered"], observed)
        counts["total"] = max(counts["total"], counts["rendered"] + counts["suppressed"])
    return lines, summary


def _compile_feedback_summary_for_selection(
    lines: list[_CompileFeedbackLine],
    semantic_summary: dict[str, Any],
    selected: set[int],
) -> dict[str, Any]:
    summary = _empty_compile_feedback_summary()
    _merge_compile_feedback_summary(summary, semantic_summary)
    generic_total = sum(line.category == "compile_report" for line in lines)
    generic_rendered = sum(
        line.category == "compile_report" and index in selected for index, line in enumerate(lines)
    )
    generic = summary["categories"].setdefault(
        "compile_report", {"total": 0, "rendered": 0, "suppressed": 0}
    )
    generic["total"] += generic_total
    generic["rendered"] += generic_rendered
    generic["suppressed"] += generic_total - generic_rendered
    for index, line in enumerate(lines):
        if index in selected:
            continue
        if line.category and line.category != "compile_report":
            counts = summary["categories"][line.category]
            counts["rendered"] -= 1
            counts["suppressed"] += 1
        for name, amount in (
            ("bind_records", line.bind_records),
            ("provenance_steps", line.provenance_steps),
        ):
            counts = summary["nested"][name]
            counts["rendered"] -= amount
            counts["suppressed"] += amount
    return summary


def _render_bounded_compile_feedback(
    header: str,
    lines: list[_CompileFeedbackLine],
    semantic_summary: dict[str, Any],
    selected: set[int],
) -> str:
    effective_selected = set(selected)
    for index, line in enumerate(lines):
        if line.category != "placement_evidence" or index not in effective_selected:
            continue
        if (
            index == 0
            or lines[index - 1].category != "overlap"
            or index - 1 not in effective_selected
        ):
            effective_selected.remove(index)
    summary = _compile_feedback_summary_for_selection(
        lines,
        semantic_summary,
        effective_selected,
    )
    summary_line = (
        "<feedback_suppression>"
        + json.dumps(summary, sort_keys=True, separators=(",", ":"))
        + "</feedback_suppression>"
    )
    blocking_categories = {
        line.category
        for line in lines
        if line.blocking and line.category not in {None, "compile_report", "placement_evidence"}
    }
    suppressed_blockers = any(
        index not in effective_selected and line.blocking and line.category != "placement_evidence"
        for index, line in enumerate(lines)
    ) or any(
        name in blocking_categories and counts.get("suppressed", 0) > 0
        for name, counts in summary["categories"].items()
    )
    evidence_pointer = (
        "Evidence: suppressed blockers and complete placement records are in "
        'read_file("reports/compile_findings.json"); use them only when its '
        "model_sha256 matches the current model.py."
        if suppressed_blockers
        else None
    )
    return "\n".join(
        [
            header,
            *(line.text for index, line in enumerate(lines) if index in effective_selected),
            *([evidence_pointer] if evidence_pointer else []),
            summary_line,
        ]
    )


def _strip_compile_feedback_metadata(body: str) -> str:
    lines, _summary = _parse_compile_feedback(body)
    return "\n".join(line.text for line in lines)


def _bounded_compile_failure(header: str, body: str) -> str:
    lines, semantic_summary = _parse_compile_feedback(body)
    selected = set(range(len(lines)))
    payload = _render_bounded_compile_feedback(header, lines, semantic_summary, selected)
    if len(payload) <= _FEEDBACK_MAX_CHARS:
        return payload

    selected.clear()
    for preserve in (True, False):
        for index, line in enumerate(lines):
            if line.preserve is not preserve:
                continue
            selected.add(index)
            candidate = _render_bounded_compile_feedback(header, lines, semantic_summary, selected)
            if len(candidate) > _FEEDBACK_MAX_CHARS:
                selected.remove(index)
    return _render_bounded_compile_feedback(header, lines, semantic_summary, selected)


GREEN_SNAPSHOT_DIRNAME = "green"
GREEN_SNAPSHOT_POINTER = "latest.json"


def green_snapshot_root(model_path: Path) -> Path:
    return Path(model_path).parent / "cache" / GREEN_SNAPSHOT_DIRNAME


def record_green_snapshot(model_path: Path, *, compile_summary: str = "") -> dict[str, Any] | None:
    try:
        from sdk.runtime.release_candidate import dependency_sha256_for_model

        model_path = Path(model_path)
        model_sha256 = sdk_runtime.model_source_sha256(model_path)
        root = green_snapshot_root(model_path)
        root.mkdir(parents=True, exist_ok=True)
        source_copy = root / f"{model_sha256}.py"
        if not source_copy.is_file():
            shutil.copy2(model_path, source_copy)
        record: dict[str, Any] = {
            "model_sha256": model_sha256,
            "dependency_sha256": dependency_sha256_for_model(model_path),
            "source_path": str(source_copy),
            "compile_summary": compile_summary,
            "recorded_at": time.time(),
        }
        payload = json.dumps(record, indent=2, sort_keys=True)
        (root / f"{model_sha256}.json").write_text(payload, encoding="utf-8")
        (root / GREEN_SNAPSHOT_POINTER).write_text(payload, encoding="utf-8")
        return record
    except Exception:  # noqa: BLE001 — a snapshot failure must never break a green compile
        logger.warning("green snapshot failed for %s (non-fatal)", model_path, exc_info=True)
        return None


def load_green_snapshot(model_path: Path, model_sha256: str) -> dict[str, Any] | None:
    root = green_snapshot_root(model_path)
    record_path = root / f"{model_sha256}.json"
    source_path = root / f"{model_sha256}.py"
    if not record_path.is_file() or not source_path.is_file():
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict) or record.get("model_sha256") != model_sha256:
        return None
    record["source_path"] = str(source_path)
    return record


SYSTEM_PROMPT = load_coding_prompt("system.md")


@lru_cache(maxsize=None)
def _sdk_dir(sdk_package: str, name: str) -> Path | None:
    try:
        mod = importlib.import_module(sdk_package)
    except Exception:  # noqa: BLE001
        return None
    f = getattr(mod, "__file__", None)
    if not f:
        return None
    package_root = Path(f).resolve().parent
    d = package_root / name
    return d if d.exists() and d.resolve().is_relative_to(package_root) else None


def _sdk_docs_root(sdk_package: str) -> Path | None:
    return _sdk_dir(sdk_package, "_docs")


def _sdk_examples_root(sdk_package: str) -> Path | None:
    return _sdk_dir(sdk_package, "_examples")


@lru_cache(maxsize=4)
def _examples_corpus(
    corpus_root: Path, sdk_package: str
) -> tuple[tuple[Path, str, tuple[str, ...]], ...]:
    from pipeline import retrieval

    seen: set[Path] = set()
    from sdk._resources import curated_example_paths

    sdk_root = _sdk_dir(sdk_package, "_docs")
    use_packaged_corpus = sdk_root is not None and corpus_root == sdk_root.parent
    curated = list(curated_example_paths()) if use_packaged_corpus else []
    files: list[Path] = list(curated)
    roots = (
        () if use_packaged_corpus else (corpus_root / "examples", corpus_root / "docs", corpus_root)
    )
    for root in roots:
        if root.exists():
            files += sorted(root.glob("*.py")) + sorted(root.glob("*.md"))
    for aux in (_sdk_docs_root(sdk_package), _sdk_examples_root(sdk_package)):
        if aux is not None:
            files += sorted(aux.rglob("*.py")) + sorted(aux.rglob("*.md"))

    out: list[tuple[Path, str, tuple[str, ...]]] = []
    for index, f in enumerate(files):
        resolved = f.resolve()
        if not any(
            resolved.is_relative_to(root) for root in _reference_roots(sdk_package, corpus_root)
        ):
            continue
        sdk_internal = (
            sdk_root is not None
            and resolved.is_relative_to(sdk_root.parent)
            and not any(
                resolved.is_relative_to(root.resolve())
                for root in (_sdk_docs_root(sdk_package), _sdk_examples_root(sdk_package))
                if root is not None
            )
        )
        if sdk_internal:
            continue
        f = resolved
        if index >= len(curated) and f in seen:
            continue
        if index >= len(curated):
            seen.add(f)
        if f.stem.lower() == "model_contract":
            continue
        try:
            txt = f.read_text(encoding="utf-8")
        except OSError:
            continue
        head = "\n".join(txt.splitlines()[:40])
        tokens = retrieval.weighted_doc(
            (retrieval.identifier_tokens(f.stem), 3),
            (retrieval.tokenize(head), 2),
            (retrieval.tokenize(txt), 1),
        )
        out.append((f, txt, tuple(tokens)))
    return tuple(out)


def _norm_doc_key(name: str) -> str:
    return Path(name).stem.lower().replace("-", "_")


def _resolve_sdk_doc(sdk_package: str, path: str) -> Path | None:
    requested = Path(path)
    if requested.name != path or requested.suffix.lower() not in {"", ".md"}:
        return None
    want = _norm_doc_key(path)
    candidates: list[Path] = []
    for root in (_sdk_docs_root(sdk_package), _sdk_examples_root(sdk_package)):
        if root is not None:
            candidates += [
                candidate.resolve()
                for candidate in sorted(root.rglob("*.md"))
                if candidate.resolve().is_relative_to(root.resolve())
            ]
    for f in candidates:
        if _norm_doc_key(f.name) == want:
            return f
    for f in candidates:
        stem = _norm_doc_key(f.name)
        nonum = stem.split("_", 1)[-1] if stem[:1].isdigit() else stem
        if nonum == want:
            return f
    return None


def _reference_roots(sdk_package: str, corpus_root: Path | None = None) -> list[Path]:
    roots = [
        root.resolve()
        for root in (_sdk_docs_root(sdk_package), _sdk_examples_root(sdk_package))
        if root is not None
    ]
    if corpus_root is not None and not any(
        corpus_root.resolve().is_relative_to(root.parent) for root in roots
    ):
        roots.append(corpus_root.resolve())
    return roots


def _reference_label(path: Path, sdk_package: str, corpus_root: Path) -> str:
    for root in (_sdk_docs_root(sdk_package), _sdk_examples_root(sdk_package)):
        if root is not None and path.is_relative_to(root.parent.resolve()):
            return (Path(root.parent.name) / path.relative_to(root.parent.resolve())).as_posix()
    if path.is_relative_to(corpus_root.resolve()):
        return path.relative_to(corpus_root.resolve()).as_posix()
    return str(path)


def _example_scope(text: str) -> str:
    if not text.startswith("---"):
        return ""
    header = text.split("---", 2)
    if len(header) < 3:
        return ""
    match = re.search(
        r"^example_kind:\s*[\"']?(complete_model|geometry_pattern)[\"']?\s*$", header[1], re.M
    )
    if match is None:
        return ""
    if match.group(1) == "complete_model":
        return (
            "Example kind: complete_model — complete model structure; adapt it to the blueprint.\n"
        )
    return "Example kind: geometry_pattern — partial geometry pattern, not a complete model.py.\n"


def _sdk_source_error(sdk_package: str, requested: str) -> str:
    topic = Path(requested).stem
    topic = {"types": "core_types", "_testing": "testing", "model": "articulated_object"}.get(
        topic, topic
    )
    doc = _resolve_sdk_doc(sdk_package, topic)
    suggestion = doc.name if doc is not None else "model_contract.md"
    return (
        "ERROR: SDK implementation source is not available through read_file. "
        f'Use read_file("{suggestion}") for the public authoring contract. '
        "No document was read by this request."
    )


def _resolve_read_target(harness: Any, requested: str) -> tuple[Path | None, str | None]:
    workspace = harness.model_path.parent.resolve()
    references = _reference_roots(harness.sdk_package, getattr(harness, "corpus_root", None))
    sdk_references = [
        root.resolve()
        for root in (_sdk_docs_root(harness.sdk_package), _sdk_examples_root(harness.sdk_package))
        if root is not None
    ]
    sdk_roots = {
        root.parent
        for root in (_sdk_docs_root(harness.sdk_package), _sdk_examples_root(harness.sdk_package))
        if root is not None
    }
    artifact_roots = [workspace]
    selected_manifest = getattr(harness, "_candidate_manifest_path", None)
    reference = getattr(harness, "_review_candidate_reference", None)
    if isinstance(reference, dict):
        selected_manifest = reference.get("manifest_path") or selected_manifest
    if selected_manifest:
        artifact_roots.append(Path(selected_manifest).resolve().parent)
    qa_path = getattr(harness, "qa_test_path", None)
    exact_files = {Path(qa_path).absolute()} if qa_path is not None else set()
    path = Path(requested)

    def checked(candidate: Path) -> tuple[Path | None, str | None]:
        resolved = candidate.resolve()
        if any(resolved.is_relative_to(root.resolve()) for root in sdk_roots) and not any(
            resolved.is_relative_to(root) for root in sdk_references
        ):
            return None, _sdk_source_error(harness.sdk_package, requested)
        if any(resolved.is_relative_to(root) for root in artifact_roots):
            return resolved, None
        if candidate.absolute() in exact_files and not candidate.is_symlink():
            return resolved, None
        if any(resolved.is_relative_to(root) for root in references):
            if resolved.is_dir() or resolved.suffix.lower() in {".md", ".py"}:
                return resolved, None
        return (
            None,
            "ERROR: path is outside the workspace, selected candidate, and SDK reference files.",
        )

    try:
        if path.is_absolute():
            return checked(path)
        candidates = [workspace / path]
        if requested == "qa_tests.py" and qa_path is not None:
            candidates.append(Path(qa_path))
        corpus = getattr(harness, "corpus_root", None)
        if corpus is not None:
            candidates.append(Path(corpus) / path)
        for root in sdk_roots:
            candidates.append(root / path)
            if path.parts and path.parts[0] == root.name:
                candidates.append(root.parent / path)
        candidates.extend(root / path for root in references)
        for candidate in candidates:
            if candidate.exists() or candidate.is_symlink():
                return checked(candidate)
        if (
            path.suffix.lower() == ".py"
            and requested not in {"model.py", "qa_tests.py"}
            and not (len(path.parts) == 2 and path.parts[0] == "blender")
        ):
            return None, _sdk_source_error(harness.sdk_package, requested)
        doc = _resolve_sdk_doc(harness.sdk_package, requested)
        if doc is not None:
            return checked(doc)
        return checked(workspace / path)
    except (OSError, RuntimeError, ValueError):
        return None, "ERROR: the requested path cannot be resolved inside an allowed read location."


def _record_read_coverage(
    harness: Any, path: Path, text: str, selected: str, offset: int | None, limit: int | None
) -> tuple[str, bool, str]:
    identity = str(path.resolve())
    digest = sha1_text(text)
    coverage = getattr(harness, "_read_coverage", None)
    if not isinstance(coverage, dict):
        coverage = harness._read_coverage = {}
    record = coverage.get(identity)
    if record is None or record["digest"] != digest:
        record = coverage[identity] = {"digest": digest, "ranges": []}
        getattr(harness, "_docs_read", set()).discard(identity)
    count = len(text.splitlines())
    start = offset or 1
    end = count if limit is None else min(count, start + limit - 1)
    truncated = len(selected) > READ_FILE_LIMIT
    delivered_end = start + selected[:READ_FILE_LIMIT].count("\n") - 1 if truncated else end
    if delivered_end >= start:
        record["ranges"].append((start, delivered_end))
    covered_end = 0
    for first, last in sorted(record["ranges"]):
        if first > covered_end + 1:
            break
        covered_end = max(covered_end, last)
    complete = covered_end >= count
    if complete:
        getattr(harness, "_docs_read", set()).add(identity)
    label = f"lines {start}-{end}"
    if truncated:
        label += f"; truncated after {READ_FILE_LIMIT} characters"
    return digest, complete, label


READ_FILE_LIMIT = 20000

_EXAMPLE_EXCERPT_CHARS = 1500

_MASS_AUDIT_BLOCK_RE = re.compile(r"<mass_audit>.*?</mass_audit>", re.S)


def read_capped(text: str, limit: int = READ_FILE_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return (
        text[:limit] + f"\n\n[NOTE: file truncated — showing the first {limit} of {len(text)} "
        "characters; the rest is NOT shown. Treat your view of this file as "
        "incomplete and do not assert about content beyond this point.]"
    )


def format_with_line_numbers(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(f"L{i}: {line}" for i, line in enumerate(lines, 1))


def read_file_window(
    text: str,
    *,
    offset: int | None = None,
    limit: int | None = None,
    line_numbers: bool = False,
) -> tuple[str | None, str | None]:
    if offset is not None and (
        not isinstance(offset, int) or isinstance(offset, bool) or offset < 1
    ):
        return None, "ERROR: offset must be >= 1"
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        return None, "ERROR: limit must be >= 1"
    if offset is None and limit is None:
        return (format_with_line_numbers(text) if line_numbers else text), None

    lines = text.splitlines()
    first_line = offset or 1
    if not lines:
        if first_line > 1:
            return None, "ERROR: offset exceeds file length"
        return "", None
    if first_line > len(lines):
        return None, "ERROR: offset exceeds file length"
    start = first_line - 1
    end = len(lines) if limit is None else min(len(lines), start + limit)
    if line_numbers:
        selected = [f"L{index}: {lines[index - 1]}" for index in range(start + 1, end + 1)]
    else:
        selected = lines[start:end]
    return "\n".join(selected), None


@dataclass(slots=True)
class HarnessResult:
    success: bool
    reason: str
    conversation: list[dict[str, Any]]
    compile_summary: str
    blocking_signals: list[str] = field(default_factory=list)
    turns: int = 0
    cost_usd: float = 0.0
    developer_note: dict[str, Any] = field(default_factory=dict)
    candidate_id: str | None = None
    candidate_manifest_path: str | None = None
    physics_report_path: str | None = None
    candidate_dirty: str = "clean"


def _tool_schemas() -> list[dict[str, Any]]:
    def fn(name: str, desc: str, props: dict, required: list[str]) -> dict:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        }

    s = {"type": "string"}
    line = {"type": "integer", "minimum": 1}
    arr = {"type": "array", "items": {"type": "string"}}
    return [
        fn(
            "read_file",
            "Read workspace files (model.py, tasks, reports, or authored Blender scripts), "
            "selected candidate artifacts, or public SDK docs/examples. SDK implementation "
            "source is unavailable. Use offset/limit for a 1-based line window.",
            {"path": s, "offset": line, "limit": line},
            ["path"],
        ),
        fn(
            "read_gate_evidence",
            "Read deterministic gate evidence with identity validation. Unlike read_file, "
            "this returns FRESH or STALE for motion evidence and/or the exact candidate "
            "physics report.",
            {
                "scope": {
                    "type": "string",
                    "enum": ["motion", "candidate", "all"],
                }
            },
            [],
        ),
        fn(
            "write_file",
            "Overwrite an entire file. Default target is model.py (the SDK source of "
            "truth). Pass `path` = blender/<name>.py to author a headless-Blender bpy "
            "script (for run_blender) when you need organic/freeform geometry the CAD "
            "SDK cannot express. Pass `path` = tasks.json to author the downstream "
            "manipulation-task spec (see the Manipulation tasks section of your system "
            "prompt); it does not affect the compile gate.",
            {"content": s, "path": s},
            ["content"],
        ),
        fn(
            "replace",
            "Apply one or more exact-string edits to a file in a SINGLE call (default "
            "model.py; pass `path`=blender/<name>.py to edit a Blender script). "
            "`edits` is an ordered list of {old_str, new_str, replace_all?}: by default "
            "each old_str must occur EXACTLY ONCE. Set `replace_all` to true only when "
            "every occurrence should change. Ambiguous edits report every line/column. "
            "ATOMIC — if any edit is invalid or missing, NOTHING is written. Successful "
            "calls return a bounded unified diff. BATCH all of this turn's changes into "
            "one call instead of many single-edit calls.",
            {
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_str": s,
                            "new_str": s,
                            "replace_all": {"type": "boolean"},
                        },
                        "required": ["old_str", "new_str"],
                    },
                },
                "path": s,
            },
            ["edits"],
        ),
        fn(
            "run_blender",
            "Build organic/freeform geometry (soft cushions, upholstery, molded "
            "shells — shapes the CAD SDK cannot express) by running a headless-Blender "
            "bpy script you authored at blender/<name>.py (via write_file). The script "
            "must build a single watertight MESH object and call export_result(obj); "
            "`bpy`, `bmesh`, `mathutils` are in scope. The driver decimates to a "
            "triangle budget and stages inputs/<out_name>.obj — attach it as ONE OF A "
            "PART'S VISUALS via mesh_from_input('<out_name>') (it can coexist with "
            "primitive visuals on the same part). If THIS script owns interfaces later "
            "consumed by placement, pass placement-relevant semantic Anchors through "
            "export_result(obj, anchors=[...]) and resolve them with semantic_anchor; "
            "decorative Meshes need none. Geometry only; set color via the SDK Visual. "
            "Returns the mesh report (tris/watertight/components/aabb/dims/anchor_count) "
            "so you can verify geometry and placement metadata.",
            {"script": s, "out_name": s},
            ["script", "out_name"],
        ),
        fn(
            "find_examples",
            "Search the SDK example corpus. Optionally filter by joint_type "
            "(revolute|prismatic|continuous|fixed).",
            {"query": s, "joint_type": s},
            ["query"],
        ),
        fn(
            "lookup_cadquery_api",
            "Keyword-search the INSTALLED CadQuery API only, so signatures are "
            "version-exact. Use this for CadQuery methods/selectors/classes on the "
            'public API surface, e.g. lookup_cadquery_api("revolve"), '
            'lookup_cadquery_api("slot2D"), lookup_cadquery_api("max Z face"), '
            'lookup_cadquery_api("bevel corner"). It returns the few matches most '
            "likely to fit your intent: qualified name, signature, and one-line "
            "doc. Do NOT use it for Artisan SDK helpers/classes such as "
            "MotionLimits, TestContext, mesh_from_cadquery, Material, Part, or "
            "Articulation; read model_contract.md or the SDK docs for those. "
            "Pass full=true to read ONE entry's complete docstring + signature; "
            "pass klass to scope the search to a CadQuery class available in the "
            "installed package, such as Workplane, Solid, Sketch, or selectors.",
            {"query": s, "full": {"type": "boolean"}, "klass": s},
            ["query"],
        ),
        fn(
            "lookup_blender_api",
            "Keyword-search the INSTALLED Blender bpy API (operators + bmesh.ops + "
            "modifier types) so names/args are version-exact — use it BEFORE writing a "
            'blender/<name>.py script, e.g. lookup_blender_api("solidify thickness"), '
            'lookup_blender_api("decimate"), lookup_blender_api("subsurf levels"), '
            'lookup_blender_api("displace texture"), lookup_blender_api("bmesh bridge '
            'loops"). Returns the few most relevant matches: qualified name, '
            "signature/args, one-line description. Pass full=true to read ONE entry's "
            "complete doc + arg list. For Blender authoring only — use "
            "lookup_cadquery_api for CadQuery and read_file/model_contract.md for the "
            "SDK model API.",
            {"query": s, "full": {"type": "boolean"}},
            ["query"],
        ),
        fn(
            "probe_model",
            "Inspect the current model.py geometry WITHOUT the pass/fail gate. "
            "Call with NO args to get the bare structure (parts, joints, axes, "
            "limits). Pass `code` (a short Python snippet ending in exactly one "
            "`emit(value)`) to MEASURE geometry: helpers injected into scope "
            "include part(name)/visual(...), aabb(obj)/dims(obj)/center(obj), "
            "pair_report(a,b)/overlap_report(a,b)/mount_report(child,parent), "
            "find_clearance_risks(...), geometry_connectivity_report(part). Use "
            "this to ROOT-CAUSE an overlap/size/floating issue — e.g. "
            "`emit(overlap_report(part('a'), part('b')))` or "
            "`emit(dims(part('lamp_head')))` — before repositioning geometry.",
            {"code": s},
            [],
        ),
        fn(
            "request_render",
            "RENDER your current model.py so you can SEE it (you otherwise only have "
            "probe numbers — rendering closes that blind spot for wrong shape/identity, "
            "wrong world orientation (lying vs standing), or a floating/penetrating part). "
            "`views` are named presets (front/side/top/perspective, each optionally with an "
            "`_open` or `_closed` suffix — `_open` drives every moving joint to its limit at "
            "once); `focus=[part_or_joint,...]` aims the camera at and zooms into the named "
            "parts/joints (use the real names from probe_model). The image comes back AS THIS "
            "TOOL'S RESULT — exactly like probe_model/compile output, with no extra delay — so "
            "look at it when you next act (don't say you've seen it in the same turn you call it). "
            "Use it to VERIFY a suspected visual/spatial fault before "
            "AND after an edit; prefer ONE targeted batch over many calls, and still use "
            "probe_model for exact dims/overlap numbers. `segment`=true recolours each PART "
            "a distinct flat colour (a 'semantic map') so a part buried inside another reads "
            "as a foreign colour the dark faithful materials hide — use it to confirm a "
            "suspected penetration / that two parts only touch where they should.",
            {"views": arr, "focus": arr, "segment": {"type": "boolean"}},
            ["views"],
        ),
        fn(
            "compile_model",
            "Compile model.py through Artisan (URDF export + schema validate + "
            "FCL rest-pose overlap + isolated-part / scale QC + run_tests). Call "
            "before finishing. For complete blocker or placement evidence, use "
            "read_file('reports/compile_findings.json'); for complete candidate-warning "
            "strings, use read_file('reports/physics_validation.json'). Treat either report "
            "as current only when its model_sha256 matches the current model.py.",
            {},
            [],
        ),
        fn(
            "record_handoff_note",
            "File a short structured note FOR THE REVIEWAGENT: what you changed this "
            "round, which overlaps/contacts/choices are intentional-by-design (with "
            "the reason), and any incoming issue you believe is a false positive (with "
            "your reasoning). Call this ONLY when the harness asks (a <handoff_ready> "
            "prompt after the motion self-check passes) — filing it hands off to the "
            "ReviewAgent. Do NOT file it during repair iterations.",
            {"changed": s, "intentional": arr, "pushback": arr},
            ["changed"],
        ),
    ]


def _coding_turn_budget(default_budget: int) -> int:
    raw = (os.environ.get("ARTISAN_CODING_TURN_BUDGET") or "").strip()
    if not raw:
        return default_budget
    try:
        budget = int(raw)
    except ValueError as exc:
        raise ValueError(f"ARTISAN_CODING_TURN_BUDGET must be an integer, got {raw!r}") from exc
    if budget <= 0:
        raise ValueError(f"ARTISAN_CODING_TURN_BUDGET must be positive, got {budget}")
    return budget


class SdkCodingHarness(_ToolLoopHarness):
    supports_cost_compaction = True
    """CodingAgent runtime. Owns model.py; SDK-narrow tools only. The multi-turn
    tool loop lives in `_ToolLoopHarness`; this class injects the coding tools,
    the compile-fresh stop condition, and the HarnessResult shape."""

    ROLE = "CodingAgent"
    SYSTEM_PROMPT = SYSTEM_PROMPT
    _CASSETTE_KIND = "coding_turn"

    _policy = DISABLED_COMPRESSION_POLICY
    _turn = 0
    _probe_dirty_epoch = 0
    _consecutive_compile_failures = 0
    _last_compile_failure_sig: str | None = None
    _pending_blender_preview: Path | None = None
    _candidate_id: str | None = None
    _candidate_manifest_path: str | None = None
    _physics_report_path: str | None = None
    _candidate_physics_report: dict[str, Any] = {}
    _candidate_environment_failed = False
    _advisory_nudge_used = False
    _latest_motion_identity: dict[str, Any] = {}
    _latest_motion_evidence: dict[str, Any] = {}
    _pending_nudge_kind: str | None = None
    _pending_nudge_fingerprint: str | None = None
    _pending_nudge_evidence_key: str | None = None
    _candidate_preview_attempts: set[tuple[str, str]] = set()
    _render_round = 0
    _render_seq = 0

    def __init__(
        self,
        model_path: Path,
        *,
        corpus_root: Path,
        sdk_package: str = "sdk",
        provider: str = "openai",
        model_id: str | None = None,
        thinking_level: str = "high",
        max_turns: int | None = None,
        max_cost_usd: float | None = 10.0,
        trace_dir: Path | None = None,
        policies: Policies = DEFAULT_POLICIES,
        cassette: object | None = None,
        backend: object | None = None,
        render_root: Path | None = None,
        recorder: object | None = None,
    ) -> None:
        self.model_path = model_path
        self.corpus_root = corpus_root
        self.sdk_package = sdk_package
        self.policies = policies
        self.cassette = cassette
        self.recorder = recorder.for_agent("coding") if recorder is not None else None
        self.backend = backend
        self.render_root = render_root
        self.max_turns = _coding_turn_budget(
            policies.coding_turn_budget if max_turns is None else max_turns
        )
        self.wrap_up_start_turn = max(1, self.max_turns - 50)
        self.max_cost_usd = max_cost_usd
        self.provider = provider
        _ck, _cret, _cfp = (None, None, None)
        if provider == "openai":
            _ck, _cret, _cfp = build_openai_prompt_cache_settings(
                model_id=model_id or "gpt-5.5",
                sdk_package=sdk_package,
                system_prompt=SYSTEM_PROMPT,
                sdk_docs_context="",
                tools=_tool_schemas(),
            )
        self.llm = create_provider_client(
            ProviderConfig(
                provider=provider,
                model_id=resolve_pipeline_model(provider, model_id),
                thinking_level=thinking_level,
                openai_prompt_cache_key=_ck,
                openai_prompt_cache_retention=_cret,
                openai_canonical_contract_fingerprint=_cfp,
            ),
        )
        self.codec = MessageCodec(provider=provider)
        _mid = self.llm.model_id
        _cost_provider = provider
        self.trace = TraceWriter(trace_dir) if trace_dir else None
        self.cost = build_cost_tracker(_cost_provider, _mid)
        self._dirty = True
        self._last_compile_ok = False
        self._last_compile_summary = ""
        self._last_blocking: list[str] = []
        self._examples_shown: set[str] = set()
        self._examples_truncated: set[str] = set()
        self._docs_read: set[str] = set()
        self._lookup_shown: set[str] = set()
        self._lookup_result_chars: dict[str, int] = {}
        self._handoff_note: dict[str, Any] = {}
        self._policy = policy_from_env()
        enable_native_compaction(
            self.llm,
            self.provider,
            self._policy,
            supports_cost_compaction=self.supports_cost_compaction,
            trace=self.trace,
        )
        self._turn = 0
        self._read_cache: dict[tuple[str, int, int | None], str] = {}
        self._read_seen_turn: dict[tuple[str, int, int | None], int] = {}
        self._probe_cache: dict[str, tuple[int, int, int]] = {}
        self._probe_dirty_epoch = 0
        self._mass_audit_cache: dict[str, str] = {}
        self._candidate_dirty = "clean"
        self._latest_motion_identity: dict[str, Any] = {}
        self._latest_motion_evidence: dict[str, Any] = {}
        self._reported_motion_block_fingerprints: set[str] = set()
        self._candidate_preview_attempts: set[tuple[str, str]] = set()
        self._consecutive_compile_failures = 0
        self._last_compile_failure_sig: str | None = None
        self._pending_blender_preview: Path | None = None
        self._mj_gate_attempts = 0
        self._mj_gate_max = 2
        self._candidate_id = None
        self._candidate_manifest_path = None
        self._physics_report_path = None
        self._candidate_physics_report: dict[str, Any] = {}
        self._candidate_environment_failed = False
        self._candidate_hard_failure = False
        self._advisory_nudge_used = False
        self._pending_motion_nudge: str | None = None
        self._pending_nudge_images: list[Path] = []
        self._pending_nudge_kind = None
        self._pending_nudge_fingerprint = None
        self._pending_nudge_evidence_key = None
        self._debug_candidates_attached: set[str] = set()
        self._note_fresh = False
        self._handoff_armed = False
        self._handoff_arm_attempts = 0
        self._first_compile_render_done = False
        self._handoff_arm_max = 2
        self._pending_handoff_nudge: str | None = None
        self._last_floating: set[str] | None = None
        self.display = SingleRunDisplay(
            console=Console(),
            model_id=_mid,
            thinking_level=thinking_level,
            max_turns=self.max_turns,
            enabled=True,
        )
        unpermitted = [
            t["function"]["name"]
            for t in _tool_schemas()
            if not self.policies.can_use_tool(self.ROLE, t["function"]["name"])
        ]
        if unpermitted:
            raise RuntimeError(
                f"tool/policy mismatch: {self.ROLE} advertises but is denied "
                f"{unpermitted}; fix policies.ROLE_TOOLS"
            )

    def _read_file(
        self,
        path: str,
        offset: int | None = None,
        limit: int | None = None,
    ) -> str:
        p, error = _resolve_read_target(self, path)
        if error is not None or p is None:
            return error or "ERROR: no readable file was resolved"
        if not p.exists():
            model_exists = self.model_path.exists()
            if p == self.model_path.resolve() and not model_exists:
                return (
                    f"NOTE: {MODEL_FILENAME} has not been authored yet — this is a "
                    "fresh build, so there is nothing to read or patch. Start by "
                    "authoring it with write_file. To plan first, read "
                    "model_contract.md, the SDK reference library named in "
                    "model_contract.md, or find_examples()."
                )
            readable = "model.py, model_contract.md, " if model_exists else "model_contract.md, "
            return (
                f"ERROR: no such file: {path}. Readable: {readable}"
                "the SDK reference library named in model_contract.md, and find_examples()."
            )
        if p.is_dir():
            return (
                f"ERROR: {path} is a directory, not a file. read_file reads a single "
                "file — name one inside it, or use probe_model to inspect built "
                "geometry rather than reading asset files directly."
            )
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return "ERROR: the requested file cannot be read as UTF-8 text."
        selected, window_error = read_file_window(
            text,
            offset=offset,
            limit=limit,
            line_numbers=p.suffix == ".py",
        )
        if window_error is not None or selected is None:
            return window_error or "ERROR: unable to read requested window"
        full = read_capped(selected)
        self._record_tier_a_savings("read_file", len(selected) - len(full))
        source_digest, _complete, shown_range = _record_read_coverage(
            self, p, text, selected, offset, limit
        )
        if not self._policy.enabled or not self._policy.read_file_dedup:
            return full
        return dedup_read(
            full,
            path=(
                path
                if offset is None and limit is None
                else f"{path} (offset={offset or 1}, limit={limit or 'EOF'})"
            ),
            key=(str(p), offset or 1, limit),
            read_cache=self._read_cache,
            read_seen_turn=self._read_seen_turn,
            this_turn=self._turn,
            on_saved=lambda saved: self._record_tier_a_savings("read_file", saved),
            source_digest=source_digest,
            shown_range=shown_range,
        )

    def _writable_target(self, path: str | None) -> tuple[Path | None, bool, str | None]:
        ws = self.model_path.parent.resolve()
        if not path or path in (MODEL_FILENAME, "./" + MODEL_FILENAME):
            try:
                if not self.model_path.resolve().is_relative_to(ws):
                    return None, False, "ERROR: model.py must resolve inside the workspace."
            except (OSError, RuntimeError):
                return None, False, "ERROR: model.py cannot be resolved safely."
            return self.model_path, True, None
        cand = (ws / path).resolve()
        try:
            rel = cand.relative_to(ws)
        except ValueError:
            return None, False, f"ERROR: path must be inside the workspace (got {path!r})"
        if str(rel) == MODEL_FILENAME:
            return self.model_path, True, None
        parts = rel.parts
        if len(parts) == 2 and parts[0] == "blender" and rel.suffix == ".py":
            return cand, False, None
        if str(rel) == "tasks.json":
            return cand, False, None
        return (
            None,
            False,
            (
                f"ERROR: writable paths are {MODEL_FILENAME}, tasks.json, or "
                f"blender/<name>.py only (got {path!r})"
            ),
        )

    def _mark_model_edited(self) -> None:
        self._dirty = True
        self._probe_dirty_epoch = getattr(self, "_probe_dirty_epoch", 0) + 1
        self._note_fresh = False
        self._handoff_armed = False
        self._handoff_arm_attempts = 0
        self._candidate_id = None
        self._candidate_manifest_path = None
        self._physics_report_path = None
        self._candidate_environment_failed = False
        self._candidate_hard_failure = False
        self._candidate_dirty = "geometry"

    def _mark_metadata_edited(self) -> None:
        self._note_fresh = False
        self._handoff_armed = False
        self._handoff_arm_attempts = 0
        if getattr(self, "_candidate_dirty", "clean") == "clean":
            self._candidate_dirty = "metadata"
        self._candidate_id = None
        self._candidate_manifest_path = None
        self._physics_report_path = None

    def _write_file(self, content: str, path: str | None = None) -> str:
        target, is_model, err = self._writable_target(path)
        if err:
            return err
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        if is_model:
            self._mark_model_edited()
        elif target.name == "tasks.json":
            self._mark_metadata_edited()
        label = MODEL_FILENAME if is_model else str(Path(path))
        return f"OK: wrote {len(content)} chars to {label}"

    @staticmethod
    def _match_locations(content: str, needle: str) -> list[str]:
        locations: list[str] = []
        start = 0
        while (offset := content.find(needle, start)) >= 0:
            line = content.count("\n", 0, offset) + 1
            line_start = content.rfind("\n", 0, offset) + 1
            locations.append(f"{line}:{offset - line_start + 1}")
            start = offset + len(needle)
        return locations

    @staticmethod
    def _bounded_diff(before: str, after: str, label: str) -> str:
        max_lines = 60
        max_chars = 6_000
        diff = difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"a/{label}",
            tofile=f"b/{label}",
            lineterm="",
            n=3,
        )
        lines: list[str] = []
        chars = 0
        truncated = False
        for line in diff:
            additional = len(line) + 1
            if len(lines) >= max_lines or chars + additional > max_chars:
                truncated = True
                break
            lines.append(line)
            chars += additional
        if truncated:
            lines.append("... diff truncated ...")
        return "\n".join(lines)

    def _replace(self, edits: Any, path: str | None = None) -> str:
        target, is_model, err = self._writable_target(path)
        if err:
            return err
        label = MODEL_FILENAME if is_model else str(Path(path))
        if not target.exists():
            return f"ERROR: {label} does not exist yet; use write_file first"
        if not isinstance(edits, list) or not edits:
            return "ERROR: `edits` must be a non-empty list of {old_str, new_str}"
        original = target.read_text(encoding="utf-8")
        working = original
        for i, e in enumerate(edits, 1):
            if not isinstance(e, dict):
                return f"ERROR: edit #{i} must be an object {{old_str, new_str}} (nothing written)"
            old, new = e.get("old_str"), e.get("new_str")
            if not isinstance(old, str) or not isinstance(new, str):
                return f"ERROR: edit #{i} needs string old_str and new_str (nothing written)"
            if not old:
                return f"ERROR: edit #{i} old_str must not be empty (nothing written)"
            replace_all = e.get("replace_all", False)
            if not isinstance(replace_all, bool):
                return f"ERROR: edit #{i} replace_all must be a boolean (nothing written)"
            n = working.count(old)
            if n == 0:
                return f"ERROR: edit #{i} old_str not found (nothing written): {old[:60]!r}"
            if n > 1 and not replace_all:
                locations = ", ".join(self._match_locations(working, old))
                return (
                    f"ERROR: edit #{i} old_str matches {n} places at {locations}; add "
                    "surrounding context to make it unique or explicitly set replace_all=true "
                    f"(nothing written): {old[:60]!r}"
                )
            working = working.replace(old, new, -1 if replace_all else 1)
        target.write_text(working, encoding="utf-8")
        if is_model:
            self._mark_model_edited()
        elif target.name == "tasks.json":
            self._mark_metadata_edited()
        diff = self._bounded_diff(original, working, label)
        return f"OK: applied {len(edits)} edit(s) to {label}\n{diff}"

    def _run_blender(self, script: str, out_name: str) -> str:
        self._pending_blender_preview = None
        target, is_model, err = self._writable_target(script)
        if err or is_model:
            return f"ERROR: run_blender `script` must be a blender/<name>.py file (got {script!r})"
        if not target.exists():
            return f"ERROR: script not found: {script} — author it with write_file first"
        if not (out_name and str(out_name).isidentifier()):
            return "ERROR: out_name must be a simple identifier (letters/digits/underscore)"
        from sdk.runtime.blender_author import run_script

        try:
            rep = run_script(target, str(out_name), self.model_path.parent)
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: run_blender failed: {type(exc).__name__}: {exc}"
        msg = self._format_blender_report(rep, str(out_name))
        if rep.get("ok"):
            self._mark_model_edited()
            preview = rep.get("preview")
            if preview and Path(preview).exists():
                self._pending_blender_preview = Path(preview)
                msg += (
                    "\nA clay-shaded preview of this mesh is attached below — check the "
                    "silhouette, curvature, and proportions match the intended part; "
                    "edit the bpy and re-run if not."
                )
        return msg

    @staticmethod
    def _format_blender_report(rep: dict, out_name: str) -> str:
        if not rep.get("ok"):
            tail = str(rep.get("stderr", "")).strip()[-1200:]
            return f"ERROR authoring {out_name!r} in Blender:\n{tail}"
        warns = []
        if not rep.get("watertight", True):
            warns.append("NOT watertight (CoACD collider at export may be unreliable)")
        if (rep.get("n_components") or 1) > 1:
            warns.append(f"{rep['n_components']} disconnected components (review may flag islands)")
        msg = (
            f"OK: authored inputs/{out_name}.obj — attach it with "
            f"mesh_from_input('{out_name}') as a part visual. "
            f"visual_tris={rep.get('visual_tris')}, watertight={rep.get('watertight')}, "
            f"components={rep.get('n_components')}, dims={rep.get('dims')}, "
            f"aabb={rep.get('aabb')}, anchor_count={rep.get('anchor_count', 0)}"
        )
        if rep.get("anchor_count"):
            msg += " — resolve the staged names with semantic_anchor(...)"
        if rep.get("cached"):
            msg += " [cached]"
        if warns:
            msg += "\nWARN: " + "; ".join(warns)
        return msg

    def _lookup_dedup_pointer(self, key: str) -> str | None:
        if self._policy.enabled and key in self._lookup_shown:
            compact = (
                "[identical lookup this round — that full result is still above; "
                "re-issuing the same query is a no-op until you change it.]"
            )
            full_chars = getattr(self, "_lookup_result_chars", {}).get(key, 0)
            tool = "lookup_blender_api" if key.startswith("blender_api|") else "lookup_cadquery_api"
            self._record_tier_a_savings(tool, int(full_chars) - len(compact))
            return compact
        return None

    def _lookup_blender_api(self, query: str, full: bool = False) -> str:
        key = f"blender_api|{query}|{bool(full)}"
        hit = self._lookup_dedup_pointer(key)
        if hit is not None:
            return hit
        from pipeline.blender_api_index import lookup_blender_api

        try:
            result = lookup_blender_api(query, full=bool(full))
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: lookup_blender_api failed: {exc}"
        if self._policy.enabled:
            self._lookup_shown.add(key)
            lookup_chars = getattr(self, "_lookup_result_chars", None)
            if not isinstance(lookup_chars, dict):
                lookup_chars = {}
                self._lookup_result_chars = lookup_chars
            lookup_chars[key] = len(result)
        return result

    def _lookup_api(self, query: str, full: bool = False, klass: str | None = None) -> str:
        key = f"cadquery_api|{query}|{bool(full)}|{klass or ''}"
        hit = self._lookup_dedup_pointer(key)
        if hit is not None:
            return hit
        from pipeline.cq_api_index import lookup_api

        try:
            result = lookup_api(query, full=bool(full), klass=klass)
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: lookup_cadquery_api failed: {exc}"
        if self._policy.enabled:
            self._lookup_shown.add(key)
            lookup_chars = getattr(self, "_lookup_result_chars", None)
            if not isinstance(lookup_chars, dict):
                lookup_chars = {}
                self._lookup_result_chars = lookup_chars
            lookup_chars[key] = len(result)
        return result

    def _find_examples(self, query: str, joint_type: str | None = None) -> str:
        from pipeline import retrieval

        corpus = _examples_corpus(self.corpus_root, self.sdk_package)
        if not corpus:
            return "No matching examples."

        jt = joint_type.lower() if joint_type else None
        survivors = [
            (f, txt, tokens)
            for f, txt, tokens in corpus
            if not jt or jt in f.name.lower() or jt in txt.lower()
        ]
        if not survivors:
            return "No matching examples."

        index = retrieval.build_bm25([list(tokens) for _, _, tokens in survivors])
        hits_idx = retrieval.search_bm25(index, retrieval.tokenize(query), k=8)
        ranked = [survivors[i] for i, _ in hits_idx]
        if not ranked and jt:
            ranked = survivors

        fresh: list[str] = []
        pointers: list[str] = []
        handled: set[str] = set()
        for f, txt, _tokens in ranked:
            identity = str(f.resolve())
            if identity in handled:
                continue
            handled.add(identity)
            label = _reference_label(f.resolve(), self.sdk_package, self.corpus_root)
            scope = _example_scope(txt)
            if identity in self._docs_read:
                if len(pointers) < 4:
                    pointer = (
                        f"### {label}\n{scope}(full page already delivered above this round "
                        "via read_file — reuse it.)"
                    )
                    excerpt = txt[:_EXAMPLE_EXCERPT_CHARS]
                    self._record_tier_a_savings(
                        "find_examples", len(f"### {f.name}\n{excerpt}") - len(pointer)
                    )
                    pointers.append(pointer)
                continue
            if identity in self._examples_shown:
                if len(pointers) < 4:
                    if identity in self._examples_truncated:
                        pointer = (
                            f"### {label}\n{scope}(excerpt already shown above this round — "
                            f'read_file("{label}") for the full page.)'
                        )
                    else:
                        pointer = f"### {label}\n{scope}(already shown in full above this round — reuse it.)"
                    excerpt = txt[:_EXAMPLE_EXCERPT_CHARS]
                    self._record_tier_a_savings(
                        "find_examples", len(f"### {f.name}\n{excerpt}") - len(pointer)
                    )
                    pointers.append(pointer)
                continue
            self._examples_shown.add(identity)
            body = txt[:_EXAMPLE_EXCERPT_CHARS]
            if len(txt) > _EXAMPLE_EXCERPT_CHARS:
                self._examples_truncated.add(identity)
                body += f'\n… (excerpt — read_file("{label}") for the full page)'
                self._record_tier_a_savings("find_examples", len(txt) - len(body))
            fresh.append(f"### {label}\n{scope}{body}")
            if len(fresh) >= 4:
                break
        hits = fresh + pointers[: max(0, 4 - len(fresh))]
        if not hits:
            return "No matching examples."
        header = (
            "Reference examples (ADAPT the patterns to this blueprint — do NOT "
            "copy verbatim; names/dims/part counts differ):\n\n"
        )
        return header + "\n\n".join(hits)

    def _record_handoff_note(self, args: dict[str, Any]) -> str:
        self._handoff_note = {
            "changed": str(args.get("changed", "")).strip(),
            "intentional": [str(x) for x in (args.get("intentional") or [])],
            "pushback": [str(x) for x in (args.get("pushback") or [])],
        }
        self._note_fresh = True
        return "OK: handoff note recorded for the ReviewAgent."

    def _accepted_inspection_reference(self) -> dict[str, Any] | None:
        from sdk.runtime.accepted_inspection import candidate_reference

        path = getattr(self, "_candidate_manifest_path", None)
        candidate_id = getattr(self, "_candidate_id", None)
        if not path or not candidate_id:
            return None
        return candidate_reference(
            Path(path),
            candidate_id,
        )

    def _run_structural_probe(self) -> dict[str, Any] | str:
        from sdk.runtime.inspection import automatic_snapshot_reference as snapshot_reference

        reference = snapshot_reference(self.model_path, sdk_package=self.sdk_package)
        request = json.dumps(
            {
                "file_path": str(self.model_path),
                **({"accepted_candidate": current_reference()} if current_reference() else {}),
                "sdk_package": self.sdk_package,
                **({"inspection_snapshot": reference} if reference else {}),
            }
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "sdk.runtime.probe_summary_runner"],
                input=request,
                capture_output=True,
                text=True,
                timeout=PROBE_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return f"ERROR: probe_model timed out ({PROBE_TIMEOUT_S:g}s)."
        except Exception as exc:  # noqa: BLE001
            return f"ERROR launching probe_model runner: {exc}"
        out = (proc.stdout or "").strip()
        if not out:
            return f"ERROR running probe: no output.\nstderr:\n{(proc.stderr or '')[:2000]}"
        try:
            payload = json.loads(out)
        except json.JSONDecodeError:
            return f"ERROR running probe: invalid JSON:\n{out[:2000]}"
        if not payload.get("ok"):
            error = payload.get("error") or {}
            return f"ERROR running probe: {error.get('message') or 'unknown probe failure'}"
        result = payload.get("result")
        if not isinstance(result, dict):
            return "ERROR running probe: runner returned a non-object result."
        return result

    @guarded_inspection
    def _probe_model(self, code: str | None = None) -> tuple[str, str]:
        from sdk.runtime.inspection import inspection_input_key

        current_inputs = inspection_input_key(self.model_path)
        if getattr(self, "_probe_inputs", None) != current_inputs:
            self._probe_cache = {}
            self._probe_inputs = current_inputs
        code = (code or "").strip()
        key = sha1_text(code) if code else "structural"
        duplicate = self._probe_dedup_pointer(key)
        if duplicate is not None:
            return duplicate, duplicate
        if not code:
            probe = self._run_structural_probe()
            if isinstance(probe, str):
                return probe, probe
            return self._budget_probe(key, json.dumps(round_floats(probe), indent=2))
        from sdk.runtime.inspection import automatic_snapshot_reference as snapshot_reference

        reference = snapshot_reference(self.model_path, sdk_package=self.sdk_package)
        request = json.dumps(
            {
                "file_path": str(self.model_path),
                **({"accepted_candidate": current_reference()} if current_reference() else {}),
                "sdk_package": self.sdk_package,
                "code": code,
                **({"inspection_snapshot": reference} if reference is not None else {}),
            }
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pipeline.tools.probe_model.runner"],
                input=request,
                capture_output=True,
                text=True,
                timeout=PROBE_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            err = f"ERROR: probe_model timed out ({PROBE_TIMEOUT_S:g}s)."
            return err, err
        except Exception as exc:  # noqa: BLE001
            err = f"ERROR launching probe_model runner: {exc}"
            return err, err
        out = (proc.stdout or "").strip()
        if not out:
            err = f"ERROR: probe_model returned no output.\nstderr:\n{(proc.stderr or '')[:2000]}"
            return err, err
        try:
            payload = json.loads(out)
        except json.JSONDecodeError:
            err = f"ERROR: probe_model returned invalid JSON:\n{out[:2000]}"
            return err, err
        return self._budget_probe(key, json.dumps(round_floats(payload), indent=2))

    def _probe_dedup_pointer(self, key: str) -> str | None:
        if not self._policy.enabled or not self._policy.probe_dedup:
            return None
        cached = self._probe_cache.get(key)
        if cached is None or cached[0] != self._probe_dirty_epoch:
            return None
        stub = "[identical probe this round; model.py unchanged since — that result is above.]"
        if len(cached) >= 3:
            self._record_tier_a_savings("probe_model", int(cached[2]) - len(stub))
        return stub

    def _budget_probe(self, key: str, full: str) -> tuple[str, str]:
        if not self._policy.enabled:
            return full, full
        from sdk.runtime.inspection import inspection_input_key

        expected_inputs = getattr(self, "_probe_inputs", None)
        if expected_inputs is None or inspection_input_key(self.model_path) == expected_inputs:
            self._probe_cache[key] = (self._probe_dirty_epoch, self._turn, len(full))
        compact = cap_tool_output(
            full, self._policy.tool_output_hard_cap_chars, label="probe_model"
        )
        return compact, full

    def _compile_model(self) -> tuple[str, str]:
        report_path = self.model_path.parent / "_compile_report.json"
        try:
            outcome, text = sdk_runtime.compile_for_agent(
                self.model_path, report_path, sdk_package=self.sdk_package
            )
        except Exception as exc:  # noqa: BLE001
            self._last_compile_ok = False
            self._last_compile_summary = f"compile crashed: {exc}"
            self._last_blocking = [str(exc)]
            self._consecutive_compile_failures += 1
            self._last_compile_failure_sig = sha1_text(self._last_compile_summary)
            header = f"COMPILE FAILURE: {self._last_compile_summary}"
            return _bounded_compile_failure(header, ""), header
        self._last_compile_ok = outcome.ok()
        self._last_compile_summary = outcome.summary
        self._last_blocking = list(outcome.blocking_signals)
        current_floating = (
            set() if outcome.ok() else self._parse_floating_parts(self._last_blocking)
        )
        delta_note = self._floating_delta_note(self._last_floating, current_floating)
        self._last_floating = current_floating
        if outcome.ok():
            self._dirty = False
            self._consecutive_compile_failures = 0
            self._last_compile_failure_sig = None
            try:
                from sdk.runtime.decimate import maybe_decimate_model_meshes

                dec_log = maybe_decimate_model_meshes(self.model_path.parent / "assets")
                if dec_log:
                    rep = self.model_path.parent / "reports" / "decimation.json"
                    rep.parent.mkdir(parents=True, exist_ok=True)
                    rep.write_text(json.dumps(dec_log, indent=2))
            except Exception:  # noqa: BLE001
                pass
            record_green_snapshot(self.model_path, compile_summary=outcome.summary)
            header = f"COMPILE SUCCESS: {outcome.summary}"
        else:
            self._consecutive_compile_failures += 1
            self._last_compile_failure_sig = sha1_text(
                outcome.summary + "".join(outcome.blocking_signals)
            )
            header = f"COMPILE FAILURE: {outcome.summary}"
        if delta_note:
            header = f"{header}\n{delta_note}"
        clean_text = _strip_compile_feedback_metadata(text)
        full = f"{header}\n{clean_text}"
        if not outcome.ok():
            return _bounded_compile_failure(header, text), full
        if not self._policy.enabled:
            return full, full
        display = self._dedup_mass_audit(clean_text)
        capped = cap_tool_output(
            display, self._policy.tool_output_hard_cap_chars, label="compile_model"
        )
        return f"{header}\n{capped}", full

    def _dedup_mass_audit(self, text: str) -> str:
        m = _MASS_AUDIT_BLOCK_RE.search(text)
        if not m:
            return text
        sig = sha1_text(m.group(0))
        if self._mass_audit_cache.get("sig") == sig:
            return text.replace(
                m.group(0),
                "<mass_audit>unchanged since the last green compile — "
                "see the earlier block.</mass_audit>",
            )
        self._mass_audit_cache["sig"] = sig
        return text

    @staticmethod
    def _parse_floating_parts(blocking_signals: list[str]) -> set[str]:
        out: set[str] = set()
        for line in blocking_signals:
            for m in re.finditer(
                r",\s*(part '[^']+'|floating group \[[^\]]*\]) is disconnected", str(line)
            ):
                out.update(re.findall(r"'([^']+)'", m.group(1)))
        return out

    @staticmethod
    def _floating_delta_note(prev: set[str] | None, current: set[str]) -> str:
        if prev is None:
            return ""
        fixed = sorted(prev - current)
        newly = sorted(current - prev)
        persistent = sorted(prev & current)
        if not (fixed or newly or persistent):
            return ""
        segs: list[str] = []
        if fixed:
            segs.append(f"fixed: {fixed}")
        if newly:
            segs.append(f"NEW (your last edit detached these): {newly}")
        if persistent:
            segs.append(f"still floating: {persistent}")
        return "FLOATING DELTA since your last compile — " + " | ".join(segs)

    def _dispatch(self, name: str, args: dict[str, Any]) -> tuple[str, str]:
        result = self._dispatch_impl(name, args)
        return result if isinstance(result, tuple) else (result, result)

    def _dispatch_impl(self, name: str, args: dict[str, Any]):
        if not self.policies.can_use_tool(self.ROLE, name):
            return f"ERROR: tool {name!r} not permitted for {self.ROLE}"
        try:
            if name == "read_file":
                return self._read_file(
                    args["path"],
                    offset=args.get("offset"),
                    limit=args.get("limit"),
                )
            if name == "read_gate_evidence":
                return json.dumps(
                    sdk_runtime.read_gate_evidence(
                        self.model_path, scope=str(args.get("scope") or "all")
                    ),
                    indent=2,
                )
            if name == "write_file":
                return self._write_file(args["content"], args.get("path"))
            if name == "replace":
                return self._replace(args.get("edits"), args.get("path"))
            if name == "run_blender":
                return self._run_blender(args["script"], args["out_name"])
            if name == "find_examples":
                return self._find_examples(args["query"], args.get("joint_type"))
            if name == "lookup_cadquery_api":
                return self._lookup_api(args["query"], args.get("full", False), args.get("klass"))
            if name == "lookup_blender_api":
                return self._lookup_blender_api(args["query"], args.get("full", False))
            if name == "probe_model":
                return self._probe_model(args.get("code"))
            if name == "compile_model":
                return self._compile_model()
            if name == "record_handoff_note":
                return self._record_handoff_note(args)
        except KeyError as exc:
            return f"ERROR: missing arg {exc}"
        return f"ERROR: unknown tool {name!r}"

    async def run(
        self,
        user_message: Any,
        *,
        initial_conversation: list[dict] | None = None,
        entry_reason: str | None = None,
    ) -> HarnessResult:
        self._render_round = self._next_render_round()
        self._render_seq = 0
        self._mj_gate_attempts = 0
        self._advisory_nudge_used = False
        self._pending_motion_nudge = None
        self._pending_nudge_images = []
        self._handoff_note = {}
        self._candidate_dirty = "clean"
        self._note_fresh = False
        self._handoff_armed = False
        self._handoff_arm_attempts = 0
        self._first_compile_render_done = False
        self._pending_handoff_nudge = None
        conversation: list[dict] = list(initial_conversation or [])
        if self.trace and not conversation:
            self.trace.write_message({"role": "system", "content": self.SYSTEM_PROMPT})
        content = user_message
        conversation.append({"role": "user", "content": content})
        self._entry_reason_once = entry_reason
        self._round_task_message = json.loads(json.dumps(conversation[-1]))
        if self.trace:
            self.trace.write_message(conversation[-1])
        return await self._run_loop_with_telemetry(conversation)

    def _build_compaction_refresher(self) -> dict[str, Any] | None:
        summary = str(getattr(self, "_last_compile_summary", "") or "").strip()
        if not summary:
            return None
        text = (
            "[history was compacted - state refresher]\n"
            f"Last compile state: {summary}\n"
            "Re-read only the model.py regions needed for the next change."
        )
        text = text[:8_000]
        return {"role": "user", "content": text}

    def _build_task_rehydration(self) -> dict[str, Any] | None:
        task = getattr(self, "_round_task_message", None)
        if not isinstance(task, dict) or task.get("role") != "user":
            return None
        model_id = normalize_model_id(self._active_model_id)
        restored = json.loads(json.dumps(task))
        if estimate_conversation_tokens([restored], model_id=model_id) <= 8_000:
            return restored

        content = restored.get("content")
        if isinstance(content, str):
            restored["content"] = content[:31_000] + "\n[task rehydration truncated at 8K]"
            return restored
        if not isinstance(content, list):
            return None

        capped_parts: list[dict[str, Any]] = []
        for raw_part in content:
            if not isinstance(raw_part, dict):
                continue
            part = json.loads(json.dumps(raw_part))
            if not capped_parts and part.get("type") == "input_text":
                text = str(part.get("text") or "")
                part["text"] = text[:24_000]
            candidate = {"role": "user", "content": [*capped_parts, part]}
            if estimate_conversation_tokens([candidate], model_id=model_id) > 8_000:
                continue
            capped_parts.append(part)
        if not capped_parts:
            return None
        cap_note = {
            "type": "input_text",
            "text": "[task rehydration capped at 8K; omitted lower-priority evidence]",
        }
        with_note = {"role": "user", "content": [*capped_parts, cap_note]}
        if estimate_conversation_tokens([with_note], model_id=model_id) <= 8_000:
            capped_parts.append(cap_note)
        return {"role": "user", "content": capped_parts}

    def _next_render_round(self) -> int:
        if self.render_root is None:
            return self._render_round + 1
        mx = 0
        if self.render_root.exists():
            for d in self.render_root.glob("P4_build_r*"):
                suffix = d.name[len("P4_build_r") :]
                if suffix.isdigit():
                    mx = max(mx, int(suffix))
        return mx + 1

    def _render_out_dir(self) -> Path:
        self._render_seq += 1
        root = self.render_root or (self.model_path.parent / "renders")
        d = root / f"P4_build_r{self._render_round:02d}" / f"{self._render_seq:02d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @guarded_inspection
    def _request_render(self, args: dict[str, Any]) -> tuple[str, list[Path]]:
        if self.backend is None:
            return "ERROR: rendering is not available in this run.", []
        views = [str(v) for v in (args.get("views") or [])]
        if not views:
            return "ERROR: provide at least one view name.", []
        pose = args.get("pose") or None
        camera = args.get("camera") or None
        focus = [str(f) for f in (args.get("focus") or [])] or None
        segment = bool(args.get("segment"))
        if pose:
            try:
                pose = {str(k): float(v) for k, v in pose.items()}
            except (TypeError, ValueError):
                return "ERROR: pose values must be numbers (joint→value).", []
        try:
            paths = self.backend.render_views(  # type: ignore[attr-defined]
                self.model_path,
                views,
                self._render_out_dir(),
                pose=pose,
                camera=camera,
                focus=focus,
                segment=segment,
            )
        except Exception as exc:  # noqa: BLE001 — surface render errors to the agent
            return f"ERROR rendering: {exc}", []
        if not paths:
            return "ERROR: no renders were produced.", []
        names = ", ".join(Path(p).name for p in paths)
        return f"OK: rendered {len(paths)} image(s): {names}", [Path(p) for p in paths]

    @staticmethod
    def _norm_user(user_message: Any) -> Any:
        if isinstance(user_message, str):
            return user_message
        out = []
        for c in user_message:
            if c.get("type") == "input_image":
                out.append({"type": "image", "path": c["image_path"]})
            else:
                out.append({"type": "text", "text": c.get("text", "")})
        return out

    def _tools(self) -> list[dict[str, Any]]:
        return _tool_schemas()

    def _wrap_up_nudge(self) -> str:
        return (
            "<wrap_up_budget>You have 50 turns left in this CodingAgent invocation. "
            "Stop expanding scope. Resolve the current blocking signals, run "
            "compile_model until the latest model is clean, and record the final "
            "developer handoff. Do not start speculative refinements.</wrap_up_budget>"
        )

    def _no_tool_nudge(self) -> str:
        if self._pending_motion_nudge is not None:
            msg = self._pending_motion_nudge
            self._pending_motion_nudge = None
            self._pending_nudge_kind = None
            self._pending_nudge_fingerprint = None
            self._pending_nudge_evidence_key = None
            return msg
        if self._pending_handoff_nudge is not None:
            msg = self._pending_handoff_nudge
            self._pending_handoff_nudge = None
            return msg
        return (
            "<compile_required>The model has not passed a fresh compile_model. Run "
            "compile_model and fix any blocking signals before concluding."
            "</compile_required>"
        )

    def _no_tool_nudge_metadata(self) -> dict[str, Any]:
        if self._pending_motion_nudge is not None:
            kind = self._pending_nudge_kind or "motion_selfcheck"
            metadata: dict[str, Any] = {
                "source": "harness_nudge",
                "nudge_kind": kind,
            }
            if kind == "motion_selfcheck":
                metadata["diagnostic_fingerprint"] = self._pending_nudge_fingerprint
                metadata["motion_evidence_key"] = self._pending_nudge_evidence_key
            return metadata
        if self._pending_handoff_nudge is not None:
            return {"source": "harness_nudge", "nudge_kind": "handoff_ready"}
        return {"source": "harness_nudge", "nudge_kind": "compile_required"}

    def _set_pending_motion_nudge(
        self,
        text: str,
        *,
        kind: str = "motion_selfcheck",
        diagnostic_fingerprint: str | None = None,
        motion_evidence_key: str | None = None,
    ) -> None:
        self._pending_motion_nudge = text
        self._pending_nudge_kind = kind
        self._pending_nudge_fingerprint = diagnostic_fingerprint
        self._pending_nudge_evidence_key = motion_evidence_key

    def _no_tool_images(self) -> list[Path]:
        images = list(getattr(self, "_pending_nudge_images", []))
        self._pending_nudge_images = []
        return [path for path in images if path.is_file()]

    def _end_turn_after_nudge(self) -> bool:
        return False

    def _on_no_tool_calls(self, conversation: list[dict], turns: int):
        if not (self._last_compile_ok and not self._dirty):
            return None
        self._motion_environment_failed = False
        blocks = (
            self._timed_gate("motion_contract", self._motion_contract_blocks)
            + self._timed_gate("geometry_island", self._geometry_island_blocks)
            + self._timed_gate("penetration", self._penetration_blocks)
            + self._timed_gate("motion_selfcheck", self._motion_selfcheck_definite)
            + self._timed_gate("driver_reach", self._driver_reach_blocks)
            + self._timed_gate("connection_ram", self._connection_ram_blocks)
        )
        if blocks:
            if getattr(self, "_motion_environment_failed", False):
                self._note_unresolved_definite(blocks)
                return self._result(
                    False,
                    "validation_environment_failed",
                    conversation,
                    turns,
                )
            preview_findings = self._timed_gate(
                "candidate_preview",
                lambda: self._candidate_preview_blocks(blocks),
            )
            fingerprint = self._motion_diagnostic_fingerprint(blocks)
            reported = getattr(self, "_reported_motion_block_fingerprints", set())
            if fingerprint in reported:
                self._set_pending_motion_nudge(
                    "<motion_selfcheck>UNCHANGED GATE RESULT "
                    f"fingerprint={fingerprint}. Reuse the full evidence above; no model, "
                    "geometry input, gate declaration, policy, or actual collision mode "
                    "changed. Continue repairing the same blockers.</motion_selfcheck>",
                    diagnostic_fingerprint=fingerprint,
                    motion_evidence_key=str(
                        getattr(self, "_latest_motion_identity", {}).get("motion_evidence_key")
                        or ""
                    ),
                )
            else:
                reported.add(fingerprint)
                self._reported_motion_block_fingerprints = reported
                evidence_key = str(
                    getattr(self, "_latest_motion_identity", {}).get("motion_evidence_key") or ""
                )
                self._set_pending_motion_nudge(
                    self._motion_nudge_text(
                        blocks,
                        motion_evidence_key=evidence_key,
                        preview_findings=preview_findings,
                    ),
                    diagnostic_fingerprint=fingerprint,
                    motion_evidence_key=evidence_key,
                )
                self._pending_nudge_images = self._timed_work(
                    "motion.debug_render", lambda: self._motion_debug_images(blocks)
                )
            self._handoff_armed = False
            return None
        candidate_blocks = self._timed_gate("candidate", self._candidate_gate_blocks)
        if candidate_blocks and self._candidate_environment_failed:
            self._note_unresolved_definite(candidate_blocks)
            return self._result(
                False,
                "validation_environment_failed",
                conversation,
                turns,
            )
        if candidate_blocks and getattr(self, "_candidate_hard_failure", False):
            self._note_unresolved_definite(candidate_blocks)
            return self._result(False, "hard_failure", conversation, turns)
        if candidate_blocks:
            candidate_fingerprint = self._motion_diagnostic_fingerprint(candidate_blocks)
            candidate_evidence_key = str(
                getattr(self, "_latest_motion_identity", {}).get("motion_evidence_key") or ""
            )
            self._set_pending_motion_nudge(
                self._candidate_nudge_text(candidate_blocks),
                diagnostic_fingerprint=candidate_fingerprint,
                motion_evidence_key=candidate_evidence_key,
            )
            debug = self._candidate_physics_report.get("collision_debug")
            if isinstance(debug, dict):
                root_value = debug.get("root")
                images = debug.get("images")
                debug_candidate_id = str(debug.get("candidate_id") or "")
                attached = getattr(self, "_debug_candidates_attached", set())
                if (
                    debug_candidate_id
                    and debug_candidate_id not in attached
                    and isinstance(root_value, str)
                    and isinstance(images, dict)
                ):
                    root = Path(root_value)
                    self._pending_nudge_images = [
                        root / str(images[key])
                        for key in ("collision_overlay", "failing_pose_closeup")
                        if isinstance(images.get(key), str)
                    ]
                    attached.add(debug_candidate_id)
                    self._debug_candidates_attached = attached
            self._handoff_armed = False
            return None
        advisories = [
            item
            for item in self._candidate_physics_report.get("advisory_findings") or []
            if isinstance(item, dict) and item.get("kind") == "free_root_stability"
        ]
        if (
            not self._advisory_nudge_used
            and self._candidate_physics_report.get("status") == "pass"
            and not self._candidate_physics_report.get("blocking_findings")
            and advisories
            and self._candidate_physics_report.get("candidate_id") == self._candidate_id
        ):
            self._advisory_nudge_used = True
            self._set_pending_motion_nudge(
                self._stability_advisory_nudge(advisories),
                kind="stability_advisory",
            )
            self._handoff_armed = False
            return None
        if not self._note_fresh and self._handoff_arm_attempts < self._handoff_arm_max:
            self._handoff_armed = True
            self._handoff_arm_attempts += 1
            self._pending_handoff_nudge = self._handoff_nudge_text()
            return None
        return self._result(True, "code_valid", conversation, turns)

    @staticmethod
    def _stability_advisory_nudge(advisories: list[object]) -> str:
        evidence = "\n".join(
            f"  - {json.dumps(item, sort_keys=True, default=str)}" for item in advisories
        )
        return (
            "<stability_advisory>The exact P4 candidate is physically valid, but its "
            "free-root rollout has semantic stability advisories:\n"
            f"{evidence}\n"
            "Use the remaining turn budget for ONE decision. Make a targeted support/"
            "mass/placement repair only when the behavior conflicts with the blueprint "
            "or object identity. Otherwise leave model.py unchanged and record why the "
            "behavior is intentional in record_handoff_note(intentional=[...]). Hard "
            "physics findings cannot be waived, but none are present in this candidate. "
            "This advisory nudge is issued once for this P4 invocation; do not churn "
            "geometry merely to erase a semantically correct observation."
            "</stability_advisory>"
        )

    def _handoff_nudge_text(self) -> str:
        evidence = getattr(self, "_latest_motion_evidence", {})
        coordination = [
            finding
            for finding in evidence.get("motion_contact_candidates", [])
            if isinstance(finding, dict) and finding.get("classification") == "COORDINATION"
        ]
        coordination_text = ""
        if coordination:
            rows = []
            for finding in coordination[:6]:
                pair = finding.get("pair") or ["?", "?"]
                drivers = finding.get("coordination_joints") or []
                rows.append(
                    f"  - {pair[0]}↔{pair[1]} drivers={drivers} "
                    f"pose={finding.get('worst_pose') or {}}"
                )
            coordination_text = (
                "\nCoordination-only contacts were found only with two or more independent "
                "drivers active; no rest or single-driver witness exists for these pairs:\n"
                + "\n".join(rows)
                + "\nJudge whether each simultaneous configuration is a credible operation. "
                "Repair it if it should work concurrently; otherwise leave geometry unchanged "
                "and record the mutually-exclusive operating assumption under intentional."
            )
        return (
            "<handoff_ready>The motion self-check passed — the build is ready to hand "
            "off. Call record_handoff_note ONCE now to summarize THIS BUILD on the "
            "FINAL model, then you are done (filing it hands off to the ReviewAgent "
            "immediately — do not add more edits after it unless you still need to fix "
            "something, which re-opens this gate):\n"
            "  - changed: what this build does / what you changed this round.\n"
            "  - intentional: non-allowance design choices, each with its reason. Do "
            "NOT re-list allow_overlap pairs (those are already declared with reasons "
            "in run_tests, which Review reads).\n"
            "  - pushback: any incoming issue you believe is a false positive, or a "
            "request you cannot satisfy as stated, with your reasoning.\n"
            f"{coordination_text}\n"
            "Probe the final model if you need to confirm what is intentional before "
            "you write it.</handoff_ready>"
        )

    def _load_gate_blueprint(self) -> dict | None:
        try:
            bp_path = self.model_path.parent / "blueprint.json"
            return json.loads(bp_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — best-effort: no blueprint -> no gates
            return None

    def _candidate_nudge_text(self, blocks: list[dict[str, Any]]) -> str:
        warnings = [str(value) for value in self._candidate_physics_report.get("warnings", [])]
        collision_debug_text: str | None = None
        debug = self._candidate_physics_report.get("collision_debug")
        if isinstance(debug, dict) and isinstance(debug.get("manifest_path"), str):
            collision_debug_text = (
                "<collision_debug>Run-local failed-candidate evidence is attached when "
                f"rendering succeeded. Full geom mapping: {debug['manifest_path']}. "
                "This evidence is diagnostic only and cannot be published.</collision_debug>"
            )
        identity = getattr(self, "_latest_motion_identity", {})
        return self._motion_nudge_text(
            blocks,
            warnings=warnings,
            motion_evidence_key=str(identity.get("motion_evidence_key") or ""),
            collision_debug_text=collision_debug_text,
        )

    def _timed_gate(
        self, name: str, check: Callable[[], list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        return self._timed_work(name, check, kind="gate")

    def _timed_work(self, name: str, work: Callable[[], Any], *, kind: str = "operation") -> Any:
        rec = getattr(self, "recorder", None)
        result = None
        sample = None
        try:
            with rec.measure() if rec is not None else _null_measure() as sample:
                try:
                    result = work()
                    return result
                except BaseException:
                    sample.ok = False
                    raise
        finally:
            if rec is not None and sample is not None:
                extra = (
                    {"findings": len(result)} if kind == "gate" and isinstance(result, list) else {}
                )
                try:
                    rec.record(kind=kind, name=name, sample=sample, **extra)
                except OSError:
                    logger.warning("Could not persist operation timing for %s", name, exc_info=True)

    def _load_motion_evidence(self) -> dict[str, Any]:
        src = self.model_path.read_text(encoding="utf-8")
        evidence = sdk_runtime.build_motion_review_evidence(
            self.model_path,
            src,
            sdk_package=self.sdk_package,
            cache_path=self.model_path.parent / "reports" / "motion_scan.json",
            blueprint=self._load_gate_blueprint(),
        )
        identity = evidence.get("motion_identity")
        self._latest_motion_identity = dict(identity) if isinstance(identity, dict) else {}
        self._latest_motion_evidence = evidence
        execution = evidence.get("motion_execution_status")
        status = (
            str(execution.get("status") or "failed") if isinstance(execution, dict) else "failed"
        )
        if isinstance(execution, dict) and status in {"crashed", "stale_input", "failed"}:
            self._motion_environment_failed = True
        return evidence

    def _motion_execution_blocks(self, evidence: dict[str, Any]) -> list[dict[str, Any]]:
        execution = evidence.get("motion_execution_status")
        if isinstance(execution, dict) and execution.get("status") == "ok":
            return []
        if not isinstance(execution, dict):
            self._motion_environment_failed = True
            return [
                {
                    "kind": "validation_environment",
                    "phase": "motion",
                    "stage": "execution_status",
                    "detail": "motion_execution_status is missing",
                }
            ]
        if execution.get("status") != "budget_exceeded":
            self._motion_environment_failed = True
        return [
            {
                "kind": (
                    "motion_budget"
                    if execution.get("status") == "budget_exceeded"
                    else "validation_environment"
                ),
                "phase": execution.get("phase"),
                "stage": (execution.get("last_progress") or {}).get("stage"),
                "detail": execution,
            }
        ]

    def _motion_contract_blocks(
        self, evidence: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        try:
            from sdk._core.v0.motion_engine.executor import REQUIRED_QUERIES

            resolved_evidence = evidence if evidence is not None else self._load_motion_evidence()
        except Exception as exc:  # noqa: BLE001 - a missing required detector is blocking
            logger.exception("motion acceptance contract could not be loaded")
            self._motion_environment_failed = True
            return [
                {
                    "kind": "validation_environment",
                    "phase": "motion",
                    "stage": "contract_load",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            ]
        execution_blocks = self._motion_execution_blocks(resolved_evidence)
        if execution_blocks:
            return execution_blocks

        required = resolved_evidence.get("required_detector_status")
        missing: list[str] = []
        incomplete: list[str] = []
        if not isinstance(required, dict):
            missing = list(REQUIRED_QUERIES)
        else:
            missing = sorted(set(REQUIRED_QUERIES) - set(required))
            incomplete = sorted(
                name
                for name in REQUIRED_QUERIES
                if name in required
                and (
                    not isinstance(required[name], dict)
                    or required[name].get("status") != "complete"
                )
            )
        if missing or incomplete:
            self._motion_environment_failed = True
            return [
                {
                    "kind": "detector_failure",
                    "detectors": sorted(set(missing) | set(incomplete)),
                    "missing": missing,
                    "incomplete": incomplete,
                    "detail": required,
                }
            ]

        unresolved = [
            dict(value)
            for value in resolved_evidence.get("access_gate_unresolved", []) or []
            if isinstance(value, dict)
        ]
        if unresolved:
            return [{**value, "kind": "access_gate_unresolved"} for value in unresolved]

        blueprint = self._load_gate_blueprint()
        if not isinstance(blueprint, dict):
            self._motion_environment_failed = True
            return [
                {
                    "kind": "validation_environment",
                    "phase": "motion",
                    "stage": "blueprint_load",
                    "detail": "blueprint.json is missing or invalid",
                }
            ]
        try:
            expected_gate_map = sdk_runtime._blueprint_access_gate_map(blueprint)
        except ValueError as exc:
            return [
                {
                    "kind": "declared_gate_witness",
                    "state": "unsupported_gate_topology",
                    "clear": False,
                    "detail": str(exc),
                }
            ]
        expected_targets = set(expected_gate_map)
        witnesses = [
            dict(value)
            for value in resolved_evidence.get("declared_gate_witnesses", []) or []
            if isinstance(value, dict)
        ]
        actual_targets = {str(value.get("joint")) for value in witnesses if value.get("joint")}
        gate_resolution = resolved_evidence.get("gate_resolution")
        witness_counts = {
            target: sum(1 for witness in witnesses if witness.get("joint") == target)
            for target in expected_targets | actual_targets
        }
        if (
            not isinstance(gate_resolution, dict)
            or actual_targets != expected_targets
            or any(count != 1 for count in witness_counts.values())
        ):
            return [
                {
                    "kind": "gate_witness_completeness",
                    "expected_targets": sorted(expected_targets),
                    "actual_targets": sorted(actual_targets),
                    "witness_counts": witness_counts,
                }
            ]
        semantic_blocks = [
            {**witness, "kind": "declared_gate_witness"}
            for witness in witnesses
            if witness.get("state") != "resolved" or not witness.get("clear")
        ]
        if semantic_blocks:
            return semantic_blocks
        accepted_poses = resolved_evidence.get("accepted_fcl_poses")
        if not isinstance(accepted_poses, list) or not accepted_poses:
            self._motion_environment_failed = True
            return [
                {
                    "kind": "detector_failure",
                    "detectors": ["gate_witness"],
                    "detail": "resolved motion evidence published no accepted poses",
                }
            ]
        states = gate_resolution.get("target_states")
        selected = gate_resolution.get("selected_gate_combo_by_target")
        binding_errors: list[dict[str, Any]] = []
        if (
            not isinstance(states, dict)
            or set(states) != expected_targets
            or any(states.get(target) != "resolved" for target in expected_targets)
            or not isinstance(selected, dict)
            or set(selected) != expected_targets
        ):
            binding_errors.append(
                {
                    "reason": "gate resolution target state is inconsistent",
                    "target_states": states,
                    "selected_gate_combo_by_target": selected,
                }
            )
        witness_by_target = {str(witness["joint"]): witness for witness in witnesses}
        for target in sorted(expected_targets):
            gate_pose = witness_by_target[target].get("gate_pose")
            expected_gates = set(expected_gate_map[target])
            target_poses = [
                pose for pose in accepted_poses if isinstance(pose, dict) and target in pose
            ]
            if not isinstance(gate_pose, dict) or set(gate_pose) != expected_gates:
                binding_errors.append(
                    {
                        "target_joint": target,
                        "reason": "witness gate pose does not match the declared gate set",
                        "expected_gate_joints": sorted(expected_gates),
                        "gate_pose": gate_pose,
                    }
                )
                continue
            if not target_poses:
                binding_errors.append(
                    {
                        "target_joint": target,
                        "reason": "accepted replay has no pose for the gated target",
                    }
                )
                continue
            for pose in target_poses:
                mismatched = {
                    gate: {"expected": float(gate_pose[gate]), "actual": pose.get(gate)}
                    for gate in sorted(expected_gates)
                    if pose.get(gate) != float(gate_pose[gate])
                }
                if mismatched:
                    binding_errors.append(
                        {
                            "target_joint": target,
                            "reason": "accepted replay contains an unselected gate combination",
                            "pose": pose,
                            "mismatched_gates": mismatched,
                        }
                    )
                    break
        if binding_errors:
            self._motion_environment_failed = True
            return [
                {
                    "kind": "detector_failure",
                    "detectors": ["gate_witness"],
                    "detail": binding_errors,
                }
            ]
        return []

    def _motion_diagnostic_fingerprint(self, blocks: list[dict[str, Any]]) -> str:
        normalized = round_floats(blocks, 8)
        block_key = sha1_text(
            json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)
        )
        identity = getattr(self, "_latest_motion_identity", {})
        evidence_key = str(identity.get("motion_evidence_key") or "motion-evidence-unknown")
        return sha1_text(f"{evidence_key}:{block_key}")

    def _motion_debug_images(self, blocks: list[dict[str, Any]]) -> list[Path]:
        if getattr(self, "backend", None) is None:
            return []
        finding = next((item for item in blocks if isinstance(item, dict)), None)
        if finding is None:
            return []
        pose = finding.get("worst_pose") or finding.get("blocked_pose") or {}
        if not pose and finding.get("joint") and finding.get("blocked_at") is not None:
            pose = {str(finding["joint"]): finding["blocked_at"]}
        pair = (
            finding.get("pair")
            or finding.get("blocked_by")
            or next(iter(finding.get("blocking_pairs") or []), None)
        )
        _ack, images = self._request_render(
            {
                "views": ["perspective"],
                "pose": pose,
                "focus": pair or [],
                "segment": True,
            }
        )
        return images

    def _motion_selfcheck_definite(self) -> list[dict[str, Any]]:
        try:
            evidence = self._load_motion_evidence()
        except Exception:  # noqa: BLE001
            logger.exception("motion self-check failed (non-fatal); allowing handoff")
            return []
        return [
            {**c, "kind": "motion_contact"}
            for c in evidence.get("motion_contact_candidates", [])
            if c.get("classification") == "DEFINITE"
        ]

    def _candidate_preview_blocks(
        self,
        blocks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not _eligible_candidate_preview(blocks) or not self.model_path.is_file():
            return []
        blueprint_path = self.model_path.parent / "blueprint.json"
        if not blueprint_path.is_file():
            return []
        try:
            from sdk.runtime.render import default_export_formats

            evidence = self._load_motion_evidence()
            identity = evidence.get("motion_identity") or {}
            accepted_fcl_poses = evidence.get("accepted_fcl_poses") or []
            required_identity = (
                "motion_request_key",
                "resolution_request_key",
                "resolved_manifest_digest",
                "simulator_collision_digest",
            )
            if (
                not accepted_fcl_poses
                or not isinstance(identity, dict)
                or any(not identity.get(field) for field in required_identity)
            ):
                return []
            request_key = str(identity["motion_request_key"])
            model_fingerprint = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
            attempt_key = (model_fingerprint, request_key)
            attempts = getattr(self, "_candidate_preview_attempts", set())
            if attempt_key in attempts:
                return []
            attempts.add(attempt_key)
            self._candidate_preview_attempts = attempts
            result = sdk_runtime.preview_validated_release_candidate(
                self.model_path,
                blueprint_path,
                self.model_path.parent / "tmp" / "candidate_preview",
                formats=default_export_formats(),
                sdk_package=self.sdk_package,
                accepted_fcl_poses=accepted_fcl_poses,
                simulator_collision_digest=str(identity["simulator_collision_digest"]),
                collision_resolution_request_key=str(identity["resolution_request_key"]),
                resolved_manifest_digest=str(identity["resolved_manifest_digest"]),
            )
        except Exception:  # noqa: BLE001 - preview infrastructure never blocks P4
            logger.exception("candidate preview failed (non-fatal)")
            return []
        if result.get("status") == "environment-failed":
            return []
        report = result.get("physics_report")
        if not isinstance(report, dict):
            return []
        return [
            dict(finding)
            for finding in report.get("blocking_findings", [])
            if isinstance(finding, dict) and finding.get("kind") == "candidate_mjcf_penetration"
        ]

    def _geometry_island_blocks(self) -> list[dict[str, Any]]:
        if not self.model_path.exists():
            return []
        try:
            evidence = self._load_motion_evidence()
        except Exception:  # noqa: BLE001
            logger.exception("geometry-connectivity self-check failed; allowing outer gate to fail")
            return []
        return [
            {**finding, "kind": "geometry_island"}
            for finding in evidence.get("geometry_island_warnings", [])
        ]

    def _candidate_gate_blocks(self) -> list[dict[str, Any]]:
        self._candidate_environment_failed = False
        self._candidate_hard_failure = False
        self._candidate_physics_report = {}
        blueprint_path = self.model_path.parent / "blueprint.json"
        if not self.model_path.exists() or not blueprint_path.is_file():
            return []
        try:
            from sdk.runtime.render import default_export_formats

            motion_evidence = self._load_motion_evidence()
            contract_blocks = self._motion_contract_blocks(motion_evidence)
            if contract_blocks:
                return contract_blocks
            accepted_fcl_poses = motion_evidence.get("accepted_fcl_poses") or []
            if not accepted_fcl_poses:
                raise RuntimeError("accepted FCL pose grid is unavailable")
            from sdk.runtime.release_candidate import verify_candidate_collision_identity

            motion_identity = motion_evidence.get("motion_identity") or {}
            if not motion_identity.get("simulator_collision_digest") or not motion_identity.get(
                "resolution_request_key"
            ):
                verify_candidate_collision_identity(
                    self.model_path,
                    None,
                    simulator_collision_digest=motion_identity.get("simulator_collision_digest"),
                    collision_resolution_request_key=motion_identity.get("resolution_request_key"),
                    required=True,
                )
            result = sdk_runtime.build_validated_release_candidate(
                self.model_path,
                blueprint_path,
                self.model_path.parent / "candidate",
                self.model_path.parent / "reports" / "physics_validation.json",
                formats=default_export_formats(),
                sdk_package=self.sdk_package,
                accepted_fcl_poses=accepted_fcl_poses,
                simulator_collision_digest=(
                    (motion_evidence.get("motion_identity") or {}).get("simulator_collision_digest")
                ),
                collision_resolution_request_key=(
                    (motion_evidence.get("motion_identity") or {}).get("resolution_request_key")
                ),
                resolved_manifest_digest=motion_identity.get("resolved_manifest_digest"),
            )
        except Exception as exc:  # noqa: BLE001 - required P4 infrastructure is blocking
            logger.exception("release-candidate validation failed")
            self._candidate_environment_failed = True
            self._candidate_manifest_path = None
            blocks = [
                {
                    "kind": "validation_environment",
                    "rig": "candidate",
                    "detail": getattr(exc, "details", f"{type(exc).__name__}: {exc}"),
                }
            ]
            if getattr(exc, "details", None) is not None:
                self._candidate_physics_report = {
                    "status": "environment-failed",
                    "blocking_findings": blocks,
                }
                report_path = self.model_path.parent / "reports" / "candidate_identity_error.json"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(
                    json.dumps(self._candidate_physics_report, indent=2), encoding="utf-8"
                )
            return blocks
        self._candidate_id = str(result.get("candidate_id") or "") or None
        self._physics_report_path = str(result.get("physics_report_path") or "") or None
        self._candidate_environment_failed = result.get("status") == "environment-failed"
        self._candidate_hard_failure = result.get("status") == "hard-failure"
        report = result.get("physics_report") or {}
        self._candidate_physics_report = dict(report) if isinstance(report, dict) else {}
        if result.get("status") == "pass" and result.get("manifest_path"):
            self._candidate_manifest_path = str(result["manifest_path"])
            return []
        self._candidate_manifest_path = None
        findings = [dict(finding) for finding in report.get("blocking_findings", [])]
        if findings:
            return findings
        candidate_status = report.get("candidate_validation_status")
        if self._candidate_hard_failure and isinstance(candidate_status, dict):
            return [
                {
                    "kind": str(candidate_status.get("state") or "candidate_contract"),
                    "rig": "candidate_contract",
                    "detail": candidate_status.get("envelope"),
                }
            ]
        return [
            {
                "kind": "candidate_validation",
                "rig": "candidate",
                "detail": (
                    f"candidate validation returned status={result.get('status')!r} "
                    "without a structured blocking finding"
                ),
            }
        ]

    def _driver_reach_blocks(self) -> list[dict[str, Any]]:
        if not self.model_path.exists():
            return []
        try:
            evidence = self._load_motion_evidence()
        except Exception:  # noqa: BLE001
            logger.exception("driver-reach self-check failed (non-fatal); allowing handoff")
            return []
        threshold = _driver_reach_block_fraction()
        blocks: list[dict[str, Any]] = []
        for f in evidence.get("driver_reach_findings", []):
            frac = f.get("reachable_fraction")
            if isinstance(frac, (int, float)) and frac < threshold:
                blocks.append({**f, "kind": "driver_reach"})
        return blocks

    def _connection_ram_blocks(self) -> list[dict[str, Any]]:
        if not self.model_path.exists():
            return []
        try:
            evidence = self._load_motion_evidence()
        except Exception:  # noqa: BLE001
            logger.exception("connection-ram self-check failed (non-fatal); allowing handoff")
            return []
        return [
            {**f, "kind": "connection_ram"}
            for f in evidence.get("connection_cleanliness_findings", [])
            if f.get("blocking")
        ]

    def _penetration_blocks(self) -> list[dict[str, Any]]:
        if not self.model_path.exists():
            return []
        try:
            evidence = self._load_motion_evidence()
        except Exception:  # noqa: BLE001
            logger.exception("penetration self-check failed (non-fatal); allowing handoff")
            return []
        return [
            {**f, "kind": "penetration"}
            for f in evidence.get("full_collision_penetrations", [])
            if f.get("blocking")
        ]

    @staticmethod
    def _motion_nudge_text(
        blocks: list[dict[str, Any]],
        *,
        warnings: list[str] | None = None,
        motion_evidence_key: str | None = None,
        collision_debug_text: str | None = None,
        preview_findings: list[dict[str, Any]] | None = None,
    ) -> str:
        previews = [
            dict(item)
            for item in (preview_findings or [])
            if item.get("kind") == "candidate_mjcf_penetration"
        ]
        candidate_warnings = [
            str(value).strip() for value in (warnings or []) if str(value).strip()
        ]
        has_open_mesh_warning = any(
            "open mesh collider" in warning for warning in candidate_warnings
        )
        islands = [c for c in blocks if c.get("kind") == "geometry_island"]
        candidate = [
            c
            for c in blocks
            if c.get("kind")
            in {
                "candidate_mjcf_penetration",
                "free_root_stability",
                "validation_environment",
                "candidate_validation",
                "detector_failure",
                "motion_budget",
            }
        ]
        raw_pens = [c for c in blocks if c.get("kind") == "penetration"]
        raw_contacts = [
            c
            for c in blocks
            if c.get("kind")
            not in (
                "geometry_island",
                "candidate_mjcf_penetration",
                "free_root_stability",
                "validation_environment",
                "candidate_validation",
                "detector_failure",
                "motion_budget",
                "driver_reach",
                "declared_gate_witness",
                "connection_ram",
                "penetration",
                "mj_jam",
            )
        ]
        contact_pairs = {tuple(sorted(c.get("pair") or ["?", "?"])) for c in raw_contacts}
        pens = [
            {
                **finding,
                "_new_since_rest": (
                    tuple(sorted(finding.get("pair") or ["?", "?"])) in contact_pairs
                ),
            }
            for finding in raw_pens
        ]
        pen_pairs = {tuple(sorted(c.get("pair") or ["?", "?"])) for c in pens}
        all_rams = [c for c in blocks if c.get("kind") == "connection_ram"]
        rams = [c for c in all_rams if tuple(sorted(c.get("pair") or ["?", "?"])) not in pen_pairs]
        merged_ram_count = len(all_rams) - len(rams)
        jams = [c for c in blocks if c.get("kind") == "driver_reach"]
        gate_witnesses = [c for c in blocks if c.get("kind") == "declared_gate_witness"]
        mjjams = [c for c in blocks if c.get("kind") == "mj_jam"]
        contacts = [
            c for c in raw_contacts if tuple(sorted(c.get("pair") or ["?", "?"])) not in pen_pairs
        ]
        merged_contact_count = len(raw_contacts) - len(contacts)
        category_items: dict[str, list[dict[str, Any]]] = {
            "geometry_island": islands,
            "candidate_mjcf_penetration": [
                item for item in candidate if item.get("kind") == "candidate_mjcf_penetration"
            ],
            "free_root_stability": [
                item for item in candidate if item.get("kind") == "free_root_stability"
            ],
            "validation_environment": [
                item for item in candidate if item.get("kind") == "validation_environment"
            ],
            "candidate_validation": [
                item for item in candidate if item.get("kind") == "candidate_validation"
            ],
            "detector_failure": [
                item for item in candidate if item.get("kind") == "detector_failure"
            ],
            "motion_budget": [item for item in candidate if item.get("kind") == "motion_budget"],
            "penetration": pens,
            "motion_contact": contacts,
            "driver_reach": jams,
            "declared_gate_witness": gate_witnesses,
            "connection_ram": rams,
            "mj_jam": mjjams,
        }
        visible: dict[str, list[dict[str, Any]]] = {}
        category_stats: dict[str, dict[str, int]] = {}
        for kind, items in category_items.items():
            ordered = _ordered_feedback_findings(kind, items)
            kept = ordered[:_FEEDBACK_FINDING_LIMIT]
            visible[kind] = kept
            category_stats[kind] = {
                "total": len(ordered),
                "rendered": len(kept),
                "suppressed": len(ordered) - len(kept),
            }
        category_stats["connection_ram"]["total"] += merged_ram_count
        category_stats["connection_ram"]["suppressed"] += merged_ram_count
        category_stats["motion_contact"]["total"] += merged_contact_count
        category_stats["motion_contact"]["suppressed"] += merged_contact_count
        warning_summary = _candidate_warning_summary(candidate_warnings)
        ordered_previews = _ordered_feedback_findings("candidate_mjcf_penetration", previews)
        visible_previews = ordered_previews[:_FEEDBACK_FINDING_LIMIT]
        category_stats["candidate_preview"] = {
            "total": len(ordered_previews),
            "rendered": len(visible_previews),
            "suppressed": len(ordered_previews) - len(visible_previews),
        }

        unmatched_previews = list(visible_previews)
        for kind in ("penetration", "motion_contact", "connection_ram"):
            for index, finding in enumerate(visible[kind]):
                pair_key = tuple(sorted(_feedback_string_tuple(finding.get("pair"))))
                if len(pair_key) != 2:
                    continue
                exact = [
                    item
                    for item in unmatched_previews
                    if tuple(sorted(_feedback_string_tuple(item.get("pair")))) == pair_key
                ]
                if not exact:
                    continue
                rendered = dict(finding)
                rendered["_exact_preview"] = exact
                visible[kind][index] = rendered
                exact_ids = {id(item) for item in exact}
                unmatched_previews = [
                    item for item in unmatched_previews if id(item) not in exact_ids
                ]

        identity_stats = {"values_clipped": 0, "characters_omitted": 0}

        def identity(value: Any) -> str:
            return _clip_feedback_identity(value, identity_stats)

        def pair_values(finding: dict[str, Any], field: str = "pair") -> tuple[str, str]:
            raw = finding.get(field)
            values = list(raw) if isinstance(raw, (list, tuple)) else []
            values.extend(["?", "?"])
            return identity(values[0]), identity(values[1])

        def source_elements_clause(finding: dict[str, Any]) -> str:
            values = _feedback_string_tuple(finding.get("elements"))
            if len(values) != 2:
                return ""
            if any(
                value.startswith("collision_") or re.search(r"_c[px]\d+(?:_|$)", value)
                for value in values
            ):
                return ""
            return f"source elements {identity(values[0])} | {identity(values[1])}"

        feedback_lines: list[tuple[str, int, str]] = []

        def add_line(kind: str, text: str, *, inline_previews: int = 0) -> None:
            feedback_lines.append((kind, inline_previews, text))

        def exact_suffix(finding: dict[str, Any]) -> tuple[str, int]:
            exact_items = finding.get("_exact_preview")
            if not isinstance(exact_items, list):
                return "", 0
            values: list[str] = []
            for exact in exact_items:
                if not isinstance(exact, dict):
                    continue
                depth = _finite_feedback_number(exact.get("penetration_depth_m"))
                depth_text = f"{depth * 1000:.3g}mm" if depth is not None else "?"
                geoms = pair_values(exact, "geoms")
                values.append(f"exact: depth={depth_text}, geoms={geoms[0]} | {geoms[1]}")
            return (f"; {'; '.join(values)}", len(values)) if values else ("", 0)

        def pose_items(
            finding: dict[str, Any],
            *,
            field: str = "worst_pose",
        ) -> list[tuple[Any, Any]]:
            raw_pose = finding.get(field)
            if not isinstance(raw_pose, dict):
                return []
            if "relevant_joints" not in finding:
                return list(raw_pose.items())
            raw_relevant = finding.get("relevant_joints")
            if not isinstance(raw_relevant, list):
                return []
            relevant = [str(name) for name in raw_relevant if isinstance(name, str)]
            return [(name, raw_pose.get(name, 0.0)) for name in relevant]

        representative_pose_seen: set[tuple[str, tuple[str, str]]] = set()

        def show_pose(kind: str, pair: tuple[str, str]) -> bool:
            key = (kind, tuple(sorted(pair)))
            if key in representative_pose_seen:
                return False
            representative_pose_seen.add(key)
            return True

        for finding in visible["geometry_island"]:
            disconnected = ", ".join(identity(v) for v in finding.get("disconnected", [])) or "?"
            add_line(
                "geometry_island",
                f"  - FLOATING GEOMETRY in Part {finding.get('part', '?')}: "
                f"{finding.get('connected', '?')}/{finding.get('total', '?')} elements connect; "
                f"disconnected: {disconnected} — physically connect or reposition these elements",
            )
        for kind in (
            "candidate_mjcf_penetration",
            "free_root_stability",
            "validation_environment",
            "candidate_validation",
            "detector_failure",
            "motion_budget",
        ):
            for finding in visible[kind]:
                rendered_kind = str(finding.get("kind") or kind)
                if rendered_kind == "candidate_mjcf_penetration":
                    pair = pair_values(finding)
                    geoms = pair_values(finding, "geoms")
                    depth = finding.get("penetration_depth_m")
                    finite_depth = _finite_feedback_number(depth)
                    mm = f"{finite_depth * 1000:.3g}mm" if finite_depth is not None else "?"
                    geom_names = [str(value) for value in geoms]
                    if geom_names and all("_cx" in name for name in geom_names):
                        advice = (
                            "both colliders are CoACD pieces; increase the authored fit clearance "
                            "or use a clearance-by-construction connector"
                        )
                    elif has_open_mesh_warning and any("_cx" not in name for name in geom_names):
                        advice = (
                            "an undecomposed open-mesh collider is present; repair its seam/poles "
                            "or replace it with a watertight/analytic collision shape before changing "
                            "the visible dimensions"
                        )
                    else:
                        advice = "fix positive geometry clearance at the named colliders"
                    pose = f" at {finding.get('pose') or {}}" if show_pose(kind, pair) else ""
                    add_line(
                        kind,
                        f"  - CANDIDATE PENETRATION {pair[0]}↔{pair[1]} depth={mm}{pose} "
                        f"(geoms {geoms[0]} | {geoms[1]}) — {advice}",
                    )
                elif rendered_kind == "free_root_stability":
                    code = str(finding.get("code") or "unknown")
                    support_mode = identity(finding.get("support_mode") or "unknown")
                    pose_name = identity(finding.get("pose") or "unknown")
                    joint_pose = finding.get("joint_pose")
                    pose_values = (
                        ", ".join(
                            f"{identity(name)}={value}"
                            for name, value in sorted(joint_pose.items())
                        )
                        if isinstance(joint_pose, dict) and joint_pose
                        else "closed/rest"
                    )
                    context = f"support_mode={support_mode}, pose={pose_name} [{pose_values}]"
                    if code == "physics_explosion":
                        quantity = identity(finding.get("offending_quantity") or "state")
                        dof = identity(finding.get("offending_dof") or "unknown")
                        body = identity(finding.get("offending_body") or "unknown")
                        value = _finite_feedback_number(finding.get("offending_value"))
                        limit = _finite_feedback_number(finding.get("offending_limit"))
                        first_time = _finite_feedback_number(finding.get("first_explosion_time_s"))
                        value_text = f"{value:.6g}" if value is not None else "?"
                        limit_text = f"{limit:.6g}" if limit is not None else "?"
                        time_text = f" at t={first_time:.6g}s" if first_time is not None else ""
                        add_line(
                            kind,
                            f"  - FREE-ROOT NUMERICAL INSTABILITY ({context}): "
                            f"|{quantity}| at DOF {dof} (body {body}) reached "
                            f"{value_text} > {limit_text}{time_text}. Inspect the named "
                            "joint's spring/damping, the body's mass/inertia, joint limits, "
                            "and contact impulses; do not change support geometry unless a "
                            "separate grounded-support finding names it.",
                        )
                    elif code == "non_finite":
                        add_line(
                            kind,
                            f"  - FREE-ROOT NUMERICAL INSTABILITY ({context}): "
                            f"{finding.get('detail', '?')}. Inspect the named scene's joint/body "
                            "state, mass/inertia, constraints, and contact impulses; do not "
                            "change support geometry unless a separate grounded-support finding "
                            "names it.",
                        )
                    elif support_mode == "grounded" and code in _GROUNDED_SUPPORT_CODES:
                        add_line(
                            kind,
                            f"  - FREE-ROOT GROUND SUPPORT ({context}, code={code}): "
                            f"{finding.get('detail', '?')}",
                        )
                    else:
                        add_line(
                            kind,
                            f"  - FREE-ROOT DYNAMICS ({context}, code={code}): "
                            f"{finding.get('detail', '?')}",
                        )
                elif rendered_kind == "candidate_validation":
                    add_line(kind, f"  - CANDIDATE VALIDATION: {finding.get('detail', '?')}")
                elif rendered_kind == "detector_failure":
                    add_line(
                        kind,
                        "  - REQUIRED DETECTOR FAILURE: "
                        + ", ".join(identity(value) for value in finding.get("detectors", [])),
                    )
                elif rendered_kind == "motion_budget":
                    add_line(
                        kind,
                        f"  - MOTION BUDGET EXCEEDED in {finding.get('phase', '?')}/"
                        f"{finding.get('stage', '?')}: simplify the active collision geometry",
                    )
                else:
                    add_line(
                        kind,
                        f"  - VALIDATION ENVIRONMENT ({finding.get('rig', '?')}): "
                        f"{finding.get('detail', '?')}",
                    )
        for c in visible["penetration"]:
            pair = pair_values(c)
            exact, inline_previews = exact_suffix(c)
            depth = c.get("penetration_depth")
            axis = c.get("axis")
            method = c.get("depth_method") or "unknown"
            finite_depth = _finite_feedback_number(depth)
            mm = (
                f"{round(finite_depth * 1000, 2)}mm"
                if finite_depth is not None
                else f"unavailable ({method})"
            )
            details: list[str] = []
            source_elements = source_elements_clause(c)
            if source_elements:
                details.append(source_elements)
            if show_pose("penetration", pair):
                details.append(
                    "constant_from_rest=true"
                    if c.get("constant")
                    else "pose[" + ", ".join(f"{identity(k)}={v}" for k, v in pose_items(c)) + "]"
                )
            if c.get("_new_since_rest"):
                details.append("new_since_rest=true")
            if exact:
                details.append(exact.removeprefix("; "))
            add_line(
                "penetration",
                f"  - PENETRATION {pair[0]}↔{pair[1]} depth≈{mm}"
                + (f" along {axis}" if axis else "")
                + (f" ({'; '.join(details)})" if details else ""),
                inline_previews=inline_previews,
            )
        for c in visible["motion_contact"]:
            pair = pair_values(c)
            elems = pair_values(c, "elements")
            exact, inline_previews = exact_suffix(c)
            pose = ""
            if show_pose("motion_contact", pair):
                pose = ", ".join(f"{identity(k)}={v}" for k, v in pose_items(c))
            vol = c.get("overlap_volume")
            vol_s = f", overlap_volume≈{vol}" if vol is not None else ""
            gate = c.get("gate_evaluation")
            gate_s = ""
            if isinstance(gate, dict):
                gate_pose = ""
                if pose:
                    gate_pose = ", ".join(
                        f"{identity(k)}={v}"
                        for k, v in pose_items({**c, "worst_pose": gate.get("pose")})
                    )
                gate_depth = gate.get("max_penetration_depth_m")
                depth_s = (
                    f"{float(gate_depth) * 1000:.3g}mm"
                    if isinstance(gate_depth, (int, float))
                    else f"unavailable ({gate.get('depth_method') or 'unknown'})"
                )
                gate_s = (
                    f"; fixed gate pose [{gate_pose}], actual={gate.get('collision_mode')}, "
                    f"depth={depth_s}, direction={gate.get('axis') or 'unknown'}"
                )
            add_line(
                "motion_contact",
                f"  - CONTACT {pair[0]}↔{pair[1]} "
                f"(elements {elems[0]} | {elems[1]}{vol_s}{gate_s}{exact})"
                + (f" — worst pose {pose}" if pose else ""),
                inline_previews=inline_previews,
            )
        for f in visible["driver_reach"]:
            joint = identity(f.get("joint", "?"))
            pair = pair_values(f, "blocked_by")
            elems = pair_values(f, "elements")
            reach = f.get("reachable") or ["?", "?"]
            nominal = f.get("nominal") or ["?", "?"]
            frac = f.get("reachable_fraction")
            pct = f"{round(float(frac) * 100)}%" if isinstance(frac, (int, float)) else "?"
            at = f.get("blocked_at")
            add_line(
                "driver_reach",
                f"  - JAM joint {joint} reaches only {pct} of its travel before "
                f"{pair[0]}↔{pair[1]} collide (elements {elems[0]} | {elems[1]}"
                + (f", jams at q={at}" if at is not None else "")
                + f"); collision-free window [{reach[0]}, {reach[1]}] of declared "
                f"[{nominal[0]}, {nominal[1]}]",
            )
        for witness in visible["declared_gate_witness"]:
            pairs = witness.get("blocking_pairs") or []
            add_line(
                "declared_gate_witness",
                f"  - DECLARED GATE WITNESS FAILED for {witness.get('joint', '?')}: "
                f"no single fixed gate pose clears target range "
                f"{witness.get('target_range')} across {witness.get('sample_count')} samples; "
                f"best gate pose={witness.get('gate_pose')}, blocking_pairs={pairs}. "
                "Repair the geometry so the declared gate can be opened once and held "
                "while the target operates; do not add or change Blueprint preconditions",
            )
        for c in visible["connection_ram"]:
            pair = pair_values(c)
            elems = pair_values(c, "elements")
            exact, inline_previews = exact_suffix(c)
            pose = ""
            if show_pose("connection_ram", pair):
                pose = ", ".join(f"{identity(k)}={v}" for k, v in pose_items(c))
            vol = c.get("penetration_volume")
            span = c.get("penetration_span")
            detail = []
            if vol is not None:
                detail.append(f"penetration_volume≈{vol}m³")
            if span is not None:
                detail.append(f"depth≈{span}m")
            add_line(
                "connection_ram",
                f"  - RAM {pair[0]}↔{pair[1]} (connection elements {elems[0]} | {elems[1]}"
                + (f", {', '.join(detail)}" if detail else "")
                + exact
                + ")"
                + (f" — worst pose {pose}" if pose else ""),
                inline_previews=inline_previews,
            )
        for c in visible["mj_jam"]:
            joint = identity(c.get("joint", "?"))
            pair = pair_values(c, "blocking_pair")
            track = c.get("track")
            pct = f"{round(float(track) * 100)}%" if isinstance(track, (int, float)) else "?"
            force = c.get("peak_force_n")
            add_line(
                "mj_jam",
                f"  - SIM-JAM joint {joint} reaches only {pct} of its contact-free travel "
                f"under FULL collision (filterparent off); worst resisting pair "
                f"{pair[0]}↔{pair[1]}"
                + (f", ~{force}N sustained" if force is not None else "")
                + " — fix the geometry clearance at that pair",
            )
        preview_header = (
            "NON-BLOCKING PREVIEW (next gate layer): exact-layer evidence only; "
            "the sweep findings below remain the blockers."
        )
        for finding in unmatched_previews:
            pair = pair_values(finding)
            geoms = pair_values(finding, "geoms")
            depth = _finite_feedback_number(finding.get("penetration_depth_m"))
            mm = f"{depth * 1000:.3g}mm" if depth is not None else "?"
            add_line(
                "candidate_preview",
                f"  - CANDIDATE PENETRATION {pair[0]}↔{pair[1]} depth={mm} "
                f"(geoms {geoms[0]} | {geoms[1]})",
            )
        free_root_guidance = ""
        if any(
            item.get("code") in _NUMERICAL_FREE_ROOT_CODES
            for item in visible["free_root_stability"]
        ):
            free_root_guidance += (
                "For numerical instability, fix the named joint/body dynamics or contact cause; "
                "do not edit support geometry merely because the validation rig has a free root. "
            )
        if any(
            item.get("support_mode") == "grounded" and item.get("code") in _GROUNDED_SUPPORT_CODES
            for item in visible["free_root_stability"]
        ):
            free_root_guidance += (
                "For grounded-support findings, fix the support footprint and mass distribution; "
                "raising the minimum Z or strengthening actuators does not make a grounded object "
                "stable. "
            )

        header = (
            "<motion_selfcheck>A range-of-motion self-collision sweep found "
            f"{len(blocks)} blocking signal(s) and {len(candidate_warnings)} "
            "exact-candidate warning(s) on the current build:"
        )
        method = (
            "These are SYMPTOMS found together — run ONE diagnosis, not one fix per signal. Kinds: "
            "FLOATING GEOMETRY = disconnected elements inside one Part. PENETRATION = resolved "
            "collision bodies overlap against zero; source-confirm representation-sensitive "
            "interfaces such as bores before changing visible geometry. CONTACT = an undeclared "
            "pair gains a new contact under motion and is folded into PENETRATION as "
            "new_since_rest=true when depth exists. JAM = a "
            "joint cannot traverse its declared travel (its subtree, incl. coupled followers, "
            "penetrates a part partway). RAM = a rest connection drives DEEP into its mate across "
            "travel. DECLARED GATE WITNESS FAILED = no one fixed declared-gate pose clears the "
            "target's whole travel; co-varying per-sample clearance is not an operating witness. "
            "SIM-JAM = MuJoCo under FULL collision tracked only a fraction of the "
            "contact-free travel. CANDIDATE PENETRATION = the exact exported MJCF exceeds the "
            "shared seated-contact tolerance at a required pose. FREE-ROOT NUMERICAL INSTABILITY "
            "= the named DOF/body exceeded a velocity or acceleration cap. FREE-ROOT GROUND "
            "SUPPORT applies only to support_mode=grounded and covers tipping, drift, drop, "
            "support margin, and settling. FREE-ROOT DYNAMICS covers other exact-scene failures "
            "without claiming standing stability.\n"
            "Method: 1) INVENTORY the signals (pair/elements/worst pose). 2) GROUP the ones "
            "sharing a cause. 3) DIAGNOSE each group to its ROOT — the ONE part/dimension in the "
            "way. Each pose[...] lists every joint that changes the pair's relative pose; an "
            "explicit 0 is rest and all other model joints are irrelevant to that pair. Gated "
            "joints were swept with their gate held OPEN, so a surviving signal requires "
            "resolution; the reported pose is where it bites deepest. Probe/render and "
            "source-confirm FIRST when the root or collision representation isn't obvious. "
            "4) PLAN the edits + side effects: a mover RIDES WITH CLEARANCE (bored hinge "
            "knuckle; SlideRailGeometry runner-in-channel; or reshape/relocate the obstruction); "
            "clamp MotionLimits ONLY when the real travel is genuinely shorter than declared, "
            "never to dodge a collision (pushback if travel and geometry truly conflict); "
            "allow_overlap ONLY for a genuine irreducible connection (the reason is AUDITED). "
            f"{free_root_guidance}"
            "5) APPLY the whole plan in one pass, re-check what you touched, then re-run "
            "compile_model. (Full playbook: the motion-collisions section of your system prompt.)"
        )
        evidence_key = identity(motion_evidence_key or "unavailable")
        evidence_footer = (
            f"Evidence: motion_evidence_key={evidence_key}; full unformatted findings: "
            'read_file("reports/motion_findings.json").'
        )
        recovery_footer = (
            "Recovery: use that sidecar only when its motion_evidence_key matches this "
            "message; otherwise run compile_model to generate current evidence."
        )
        fixed_tail = [
            *([warning_summary] if warning_summary else []),
            method,
            *([collision_debug_text] if collision_debug_text else []),
            evidence_footer,
            recovery_footer,
        ]

        def render_payload() -> str:
            summary = {
                "categories": category_stats,
                "nested": {
                    "bind_records": {"total": 0, "rendered": 0, "suppressed": 0},
                    "provenance_steps": {"total": 0, "rendered": 0, "suppressed": 0},
                },
                "identity_clipping": identity_stats,
            }
            summary_line = (
                "<feedback_suppression>"
                + json.dumps(summary, sort_keys=True, separators=(",", ":"))
                + "</feedback_suppression>"
            )
            rendered_feedback: list[str] = []
            label_pending = category_stats["candidate_preview"]["rendered"] > 0
            for kind, inline_previews, text in feedback_lines:
                if label_pending and (inline_previews or kind == "candidate_preview"):
                    rendered_feedback.append(preview_header)
                    label_pending = False
                rendered_feedback.append(text)
            parts = [
                header,
                *rendered_feedback,
                *fixed_tail,
                summary_line,
            ]
            return "\n".join(parts + ["</motion_selfcheck>"])

        payload = render_payload()
        while len(payload) > _FEEDBACK_MAX_CHARS and feedback_lines:
            dropped_kind, inline_previews, _dropped_text = feedback_lines.pop()
            category_stats[dropped_kind]["rendered"] -= 1
            category_stats[dropped_kind]["suppressed"] += 1
            if inline_previews:
                category_stats["candidate_preview"]["rendered"] -= inline_previews
                category_stats["candidate_preview"]["suppressed"] += inline_previews
            payload = render_payload()
        return payload

    def _note_unresolved_definite(self, blocks: list[dict[str, Any]]) -> None:
        parts: list[str] = []
        for c in blocks[:8]:
            if c.get("kind") == "geometry_island":
                parts.append(f"island:{c.get('part', '?')}")
            elif c.get("kind") in {
                "candidate_mjcf_penetration",
                "free_root_stability",
                "validation_environment",
                "contract_mismatch",
                "contract_unsupported",
            }:
                if c.get("kind") == "candidate_mjcf_penetration":
                    pair = c.get("pair") or ["?", "?"]
                    parts.append(f"candidate-penetration:{pair[0]}↔{pair[1]}")
                else:
                    raw_detail = c.get("detail")
                    if (
                        isinstance(raw_detail, dict)
                        and raw_detail.get("kind") == "candidate_identity"
                    ):
                        detail = json.dumps(raw_detail, sort_keys=True)
                    elif isinstance(raw_detail, dict):
                        detail = str(
                            raw_detail.get("first_structural_difference")
                            or raw_detail.get("reason")
                            or raw_detail
                        )[:240]
                    else:
                        detail = str(raw_detail or "")[:240]
                    parts.append(f"candidate:{c.get('kind')}({detail})")
            elif c.get("kind") == "penetration":
                pair = c.get("pair") or ["?", "?"]
                parts.append(f"pen:{pair[0]}↔{pair[1]}")
            elif c.get("kind") == "mj_jam":
                parts.append(f"sim-jam:{c.get('joint', '?')}")
            elif c.get("kind") == "driver_reach":
                pair = c.get("blocked_by") or ["?", "?"]
                parts.append(f"jam:{c.get('joint', '?')}({pair[0]}↔{pair[1]})")
            elif c.get("kind") == "declared_gate_witness":
                parts.append(f"gate-witness:{c.get('joint', '?')}")
            elif c.get("kind") == "connection_ram":
                pair = c.get("pair") or ["?", "?"]
                parts.append(f"ram:{pair[0]}↔{pair[1]}")
            else:
                pair = c.get("pair") or ["?", "?"]
                parts.append(f"{pair[0]}↔{pair[1]}")
        msg = f"unresolved motion self-check after gate: {', '.join(parts)}"
        if msg not in self._last_blocking:
            self._last_blocking = list(self._last_blocking) + [msg]

    def _dispatch_tool(self, name: str, args: dict):
        if name == "request_render":
            text_ack, imgs = self._request_render(args)
            return text_ack, imgs, None, None
        compact, full = self._dispatch(name, args)
        if name == "record_handoff_note" and self._handoff_armed and self._note_fresh:
            terminal = self._result(True, "code_valid", self._conversation, self._turn)
            return compact, [], terminal, full
        images: list[Path] = []
        if name == "run_blender" and self._pending_blender_preview is not None:
            images = [self._pending_blender_preview]
            self._pending_blender_preview = None
        if (
            name == "compile_model"
            and self._last_compile_ok
            and not self._first_compile_render_done
            and os.environ.get("ARTISAN_FIRST_COMPILE_RENDER", "1") != "0"
        ):
            self._first_compile_render_done = True
            try:
                ack, auto_imgs = self._request_render({"views": ["perspective"]})
                compact = f"{compact}\n\n[auto-render after first successful compile] {ack}"
                images = images + auto_imgs
            except Exception:  # noqa: BLE001 — auto-render is best-effort
                pass
        return compact, images, None, full

    def _tool_ok(self, text_ack: str) -> bool:
        return not str(text_ack).startswith(("ERROR", "COMPILE FAILURE"))

    def _zero_cost_terminal_tools(self) -> frozenset[str]:
        return frozenset({"record_handoff_note"})

    def _fallback_result(self, conversation: list[dict], turns: int):
        candidate_required = (self.model_path.parent / "blueprint.json").is_file()
        return self._result(
            self._last_compile_ok
            and not self._dirty
            and (not candidate_required or self._candidate_manifest_path is not None),
            "max_turns",
            conversation,
            turns,
        )

    def _cost_limit_result(self, conversation: list[dict], turns: int):
        return self._result(False, "cost_limit", conversation, turns)

    def _result(self, ok: bool, reason: str, conv: list[dict], turns: int) -> HarnessResult:
        return HarnessResult(
            success=ok,
            reason=reason,
            conversation=conv,
            compile_summary=self._last_compile_summary,
            blocking_signals=self._last_blocking,
            turns=turns,
            cost_usd=self._current_total_cost_usd(),
            developer_note=dict(self._handoff_note),
            candidate_id=self._candidate_id,
            candidate_manifest_path=self._candidate_manifest_path,
            physics_report_path=self._physics_report_path,
            candidate_dirty=getattr(self, "_candidate_dirty", "clean"),
        )
