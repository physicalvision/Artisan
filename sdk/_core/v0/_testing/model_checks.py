from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Set, Tuple

from .common import (
    _JOINT_ORIGIN_TOL_DEFAULT,
    AABB,
    ArticulationType,
    GeometryOverlap,
    _aabb_axis_overlap,
    _aabb_axis_span,
    _axis_index,
    _CoplanarFinding,
    _format_overlap_element,
    _format_unsupported_part_finding,
    _normalize_joint_origin_tol,
    _overlap_rank,
    _pair_key,
    _transform_aabb,
    _unsupported_finding_parts,
    compute_part_world_transforms,
    default_floating_contact_tol_from_env,
    default_jointed_support_tol_from_env,
    default_overlap_tol_from_env,
    default_overlap_volume_tol_from_env,
    find_geometry_overlaps,
    find_geometry_overlaps_in_poses,
    find_joint_origin_distance_findings,
    find_part_geometry_connectivity_findings,
    find_unsupported_parts,
    generate_pose_samples,
    resolve_mesh_path,
    weld_group_of,
)

_OVERLAP_FINDING_LIMIT = 64
_BIND_RECORD_LIMIT = 64
_PROVENANCE_STEP_LIMIT = 64
_IDENTITY_LIMIT = 512
_COMPILE_FEEDBACK_LINE_TAG = "compile_feedback_line"
_COMPILE_FEEDBACK_ACCOUNTING_TAG = "compile_feedback_accounting"


@dataclass
class _CompileFeedbackAccounting:
    overlap_total: int = 0
    overlap_rendered: int = 0
    overlap_suppressed: int = 0
    bind_total: int = 0
    bind_rendered: int = 0
    bind_suppressed: int = 0
    provenance_total: int = 0
    provenance_rendered: int = 0
    provenance_suppressed: int = 0
    placement_evidence_total: int = 0
    placement_evidence_rendered: int = 0
    placement_evidence_suppressed: int = 0

    def to_dict(self, identity_stats: dict[str, int]) -> dict[str, object]:
        return {
            "categories": {
                "overlap": {
                    "total": self.overlap_total,
                    "rendered": self.overlap_rendered,
                    "suppressed": self.overlap_suppressed,
                },
                "placement_evidence": {
                    "total": self.placement_evidence_total,
                    "rendered": self.placement_evidence_rendered,
                    "suppressed": self.placement_evidence_suppressed,
                },
            },
            "nested": {
                "bind_records": {
                    "total": self.bind_total,
                    "rendered": self.bind_rendered,
                    "suppressed": self.bind_suppressed,
                },
                "provenance_steps": {
                    "total": self.provenance_total,
                    "rendered": self.provenance_rendered,
                    "suppressed": self.provenance_suppressed,
                },
            },
            "identity_clipping": dict(identity_stats),
        }


def _compile_feedback_line(
    text: str,
    *,
    category: str | None = None,
    bind_records: int = 0,
    provenance_steps: int = 0,
    preserve: bool = False,
) -> str:
    payload: dict[str, object] = {}
    if category is not None:
        payload["category"] = category
    if bind_records:
        payload["bind_records"] = bind_records
    if provenance_steps:
        payload["provenance_steps"] = provenance_steps
    if preserve:
        payload["preserve"] = True
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"{text} <{_COMPILE_FEEDBACK_LINE_TAG}>{encoded}</{_COMPILE_FEEDBACK_LINE_TAG}>"


def _compile_feedback_accounting_line(
    accounting: _CompileFeedbackAccounting,
    identity_stats: dict[str, int],
) -> str:
    encoded = json.dumps(
        accounting.to_dict(identity_stats),
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"<{_COMPILE_FEEDBACK_ACCOUNTING_TAG}>{encoded}</{_COMPILE_FEEDBACK_ACCOUNTING_TAG}>"


def _current_pose_overlap_sort_key(overlap: GeometryOverlap) -> tuple[object, ...]:
    pair = _pair_key(overlap.link_a, overlap.link_b)
    return (
        -float(overlap.overlap_volume),
        -float(min(overlap.overlap_depth)),
        pair,
        (
            overlap.elem_a_name or "",
            int(overlap.elem_a),
            overlap.elem_b_name or "",
            int(overlap.elem_b),
        ),
    )


def _clip_identity(value: object, stats: dict[str, int]) -> str:
    text = str(value)
    if len(text) <= _IDENTITY_LIMIT:
        return text
    omitted = len(text) - 480
    stats["values_clipped"] += 1
    stats["characters_omitted"] += omitted
    return f"{text[:240]}...<omitted={omitted}>...{text[-240:]}"


def _source_summary(value: object, identity_stats: dict[str, int]) -> str:
    if not isinstance(value, dict):
        return "?"
    return (
        f"{_clip_identity(value.get('source_kind', '?'), identity_stats)}:"
        f"{_clip_identity(value.get('owner_ref', '?'), identity_stats)}"
    )


_BindAttributionRecords = tuple[tuple[str, list[dict[str, object]]], ...]


def _bind_attribution_records(
    model: object,
    overlap: GeometryOverlap,
) -> _BindAttributionRecords:
    governing = getattr(model, "governing_placement_records", None)
    if not callable(governing):
        return ()
    return tuple(
        (side, list(governing(part_name, visual_uid)))
        for side, part_name, visual_uid in (
            ("elem_a", overlap.link_a, overlap.elem_a_uid),
            ("elem_b", overlap.link_b, overlap.elem_b_uid),
        )
    )


def _semantic_overlap_element_label(
    model: object,
    *,
    part_name: str,
    visual_uid: str | None,
    fallback_name: str | None,
    element_index: int,
) -> str:
    for part in getattr(model, "parts", ()):
        if getattr(part, "name", None) != part_name:
            continue
        for visual in getattr(part, "visuals", ()):
            if visual_uid is not None and getattr(visual, "visual_uid", None) == visual_uid:
                name = getattr(visual, "name", None)
                return str(name) if name else f"element#{element_index}"
    if fallback_name:
        return re.sub(r"(?:__component_\d+|_cx\d+)$", "", fallback_name)
    return f"element#{element_index}"


def _overlap_structured_finding(
    model: object,
    overlap: GeometryOverlap,
    *,
    finding_index: int,
) -> dict[str, object]:
    return {
        "kind": "real_overlap",
        "finding_index": finding_index,
        "pose_index": overlap.pose_index,
        "pose": dict(overlap.pose),
        "part_pair": [overlap.link_a, overlap.link_b],
        "element_a": {
            "index": overlap.elem_a,
            "semantic_label": _semantic_overlap_element_label(
                model,
                part_name=overlap.link_a,
                visual_uid=overlap.elem_a_uid,
                fallback_name=overlap.elem_a_name,
                element_index=overlap.elem_a,
            ),
            "raw_name": overlap.elem_a_name,
            "geometry": overlap.elem_a_geometry,
            "visual_uid": overlap.elem_a_uid,
        },
        "element_b": {
            "index": overlap.elem_b,
            "semantic_label": _semantic_overlap_element_label(
                model,
                part_name=overlap.link_b,
                visual_uid=overlap.elem_b_uid,
                fallback_name=overlap.elem_b_name,
                element_index=overlap.elem_b,
            ),
            "raw_name": overlap.elem_b_name,
            "geometry": overlap.elem_b_geometry,
            "visual_uid": overlap.elem_b_uid,
        },
        "overlap_depth": list(overlap.overlap_depth),
        "min_depth": min(overlap.overlap_depth),
        "overlap_volume": overlap.overlap_volume,
        "aabb_a": [list(overlap.aabb_a[0]), list(overlap.aabb_a[1])],
        "aabb_b": [list(overlap.aabb_b[0]), list(overlap.aabb_b[1])],
    }


def _overlap_placement_records(
    overlap: GeometryOverlap,
    records_by_side: _BindAttributionRecords,
    *,
    finding_index: int,
) -> list[dict[str, object]]:
    side_values = {
        "elem_a": (overlap.link_a, overlap.elem_a_uid),
        "elem_b": (overlap.link_b, overlap.elem_b_uid),
    }
    return [
        {
            "finding_index": finding_index,
            "side": side,
            "part": side_values[side][0],
            "visual_uid": side_values[side][1],
            "records": records,
        }
        for side, records in records_by_side
        if records
    ]


def _compact_placement_evidence_line(
    model: object,
    overlap: GeometryOverlap,
    records_by_side: _BindAttributionRecords,
    identity_stats: dict[str, int],
) -> str | None:
    if overlap.elem_a_uid is None or overlap.elem_b_uid is None:
        return None
    sides = {
        "elem_a": (
            overlap.link_a,
            overlap.elem_a_uid,
            overlap.elem_a_name,
            overlap.elem_a,
            overlap.link_b,
            overlap.elem_b_uid,
        ),
        "elem_b": (
            overlap.link_b,
            overlap.elem_b_uid,
            overlap.elem_b_name,
            overlap.elem_b,
            overlap.link_a,
            overlap.elem_a_uid,
        ),
    }
    matches: list[tuple[str, dict[str, object]]] = []
    for side, records in records_by_side:
        part_name, visual_uid, _name, _index, other_part, other_uid = sides[side]
        direct_target = f"visual:{part_name}:{visual_uid}"
        expected_host = f"visual:{other_part}:{other_uid}"
        for record in records:
            if record.get("superseded") or record.get("target_ref") != direct_target:
                continue
            steps = record.get("provenance_steps") or []
            if not isinstance(steps, list) or not steps or not isinstance(steps[-1], dict):
                continue
            step = steps[-1]
            hosts = step.get("host_sources") or []
            if not any(
                isinstance(host, dict)
                and host.get("owner_kind") == "visual"
                and host.get("owner_ref") == expected_host
                for host in hosts
            ):
                continue
            matches.append((side, step))
    if len(matches) != 1:
        return None
    side, step = matches[0]
    part_name, visual_uid, name, index, other_part, other_uid = sides[side]
    del visual_uid, other_uid
    subject_label = _semantic_overlap_element_label(
        model,
        part_name=part_name,
        visual_uid=(overlap.elem_a_uid if side == "elem_a" else overlap.elem_b_uid),
        fallback_name=name,
        element_index=index,
    )
    other_label = _semantic_overlap_element_label(
        model,
        part_name=other_part,
        visual_uid=(overlap.elem_b_uid if side == "elem_a" else overlap.elem_a_uid),
        fallback_name=(overlap.elem_b_name if side == "elem_a" else overlap.elem_a_name),
        element_index=(overlap.elem_b if side == "elem_a" else overlap.elem_a),
    )
    operation = _clip_identity(step.get("operation", "?"), identity_stats)
    commanded_axis_map = json.dumps(
        step.get("commanded_axis_map"), sort_keys=True, separators=(",", ":")
    )
    effective_axis_map = json.dumps(
        step.get("effective_axis_map"), sort_keys=True, separators=(",", ":")
    )
    return (
        "placement_evidence "
        f"elements=({_clip_identity(part_name, identity_stats)}/"
        f"{_clip_identity(subject_label, identity_stats)},"
        f"{_clip_identity(other_part, identity_stats)}/"
        f"{_clip_identity(other_label, identity_stats)}) "
        f"operation={operation} clearance={step.get('commanded_clearance')!r} "
        f"axis_map={commanded_axis_map} effective_axis_map={effective_axis_map}"
    )


def _account_suppressed_bind_attribution(
    records_by_side: _BindAttributionRecords,
    accounting: _CompileFeedbackAccounting,
) -> None:
    for _side, records in records_by_side:
        step_count = sum(len(record.get("provenance_steps") or []) for record in records)
        accounting.bind_total += len(records)
        accounting.bind_suppressed += len(records)
        accounting.provenance_total += step_count
        accounting.provenance_suppressed += step_count


def _bind_attribution_lines(
    model: object,
    overlap: GeometryOverlap,
    identity_stats: dict[str, int],
    *,
    accounting: _CompileFeedbackAccounting | None = None,
    records_by_side: _BindAttributionRecords | None = None,
) -> list[str]:
    if records_by_side is None:
        records_by_side = _bind_attribution_records(model, overlap)
    lines: list[str] = []
    for side, records in records_by_side:
        visible_records = records[:_BIND_RECORD_LIMIT]
        hidden_record_steps = sum(
            len(record.get("provenance_steps") or []) for record in records[_BIND_RECORD_LIMIT:]
        )
        if accounting is not None:
            accounting.bind_total += len(records)
            accounting.bind_rendered += len(visible_records)
            accounting.bind_suppressed += len(records) - len(visible_records)
            accounting.provenance_total += hidden_record_steps
            accounting.provenance_suppressed += hidden_record_steps
        for record in visible_records:
            steps = list(record.get("provenance_steps") or [])
            visible_steps = steps[:_PROVENANCE_STEP_LIMIT]
            rendered_steps: list[str] = []
            for step in visible_steps:
                if not isinstance(step, dict):
                    continue
                hosts = ",".join(
                    _source_summary(source, identity_stats)
                    for source in step.get("host_sources") or []
                )
                rendered_steps.append(
                    f"{_clip_identity(step.get('operation', '?'), identity_stats)}[subject="
                    f"{_source_summary(step.get('subject_source'), identity_stats)};hosts={hosts};"
                    f"clearance={step.get('commanded_clearance')!r};"
                    f"axis_map={step.get('commanded_axis_map')!r};"
                    f"orientation={step.get('orientation_measurement')!r}]"
                )
            rendered_step_count = len(rendered_steps)
            suppressed_steps = len(steps) - rendered_step_count
            if accounting is not None:
                accounting.provenance_total += len(steps)
                accounting.provenance_rendered += rendered_step_count
                accounting.provenance_suppressed += suppressed_steps
            bind_line = (
                f"  attribution {side}: bind#{record.get('ordinal', '?')} "
                f"target={_clip_identity(record.get('target_ref', '?'), identity_stats)} "
                f"steps={rendered_step_count} operations={' -> '.join(rendered_steps) or 'none'}"
            )
            lines.append(
                bind_line
                if accounting is None
                else _compile_feedback_line(
                    bind_line,
                    bind_records=1,
                    provenance_steps=rendered_step_count,
                )
            )
            if suppressed_steps:
                step_note = (
                    f"  attribution {side}: bind#{record.get('ordinal', '?')} "
                    f"[suppressed {suppressed_steps} provenance step(s)]"
                )
                lines.append(
                    step_note
                    if accounting is None
                    else _compile_feedback_line(step_note, preserve=True)
                )
        suppressed_records = len(records) - len(visible_records)
        if suppressed_records:
            record_note = (
                f"  attribution {side}: [suppressed {suppressed_records} governing "
                "bind record(s)]"
                + (
                    f" [suppressed {hidden_record_steps} provenance step(s) under hidden bind record(s)]"
                    if hidden_record_steps
                    else ""
                )
            )
            lines.append(
                record_note
                if accounting is None
                else _compile_feedback_line(record_note, preserve=True)
            )
    return lines


class TestContextModelCheckMixin:
    def check_model_valid(self) -> bool:
        try:
            validate = getattr(self.model, "validate", None)
            if callable(validate):
                validate(strict=True)
                self._model_validated_strict = True
                return self._record("check_model_valid", True)
            return self._record("check_model_valid", False, "model has no .validate()")
        except Exception as exc:
            return self._record("check_model_valid", False, f"{type(exc).__name__}: {exc}")

    def check_mesh_assets_ready(self) -> bool:
        links = getattr(self.model, "parts", None)
        if not isinstance(links, list):
            return self._record(
                "check_mesh_assets_ready", False, "model.parts missing or not a list"
            )
        missing: List[str] = []
        for link in links:
            for field_name in ("visuals",):
                elems = getattr(link, field_name, None)
                if not isinstance(elems, list):
                    continue
                for elem in elems:
                    geom = getattr(elem, "geometry", None)
                    filename = getattr(geom, "filename", None)
                    if filename is None:
                        continue
                    filename_text = str(filename)
                    if not filename_text:
                        continue
                    try:
                        resolved = resolve_mesh_path(
                            filename_text,
                            assets=self.asset_root or getattr(self.model, "assets", None),
                        )
                    except Exception:
                        resolved = None
                    if resolved is None or not resolved.exists():
                        missing.append(str(filename_text if resolved is None else resolved))
        if missing:
            preview = "\n".join(missing[:12])
            more = "" if len(missing) <= 12 else f"\n... ({len(missing) - 12} more)"
            return self._record(
                "check_mesh_assets_ready", False, f"Missing mesh assets:\n{preview}{more}"
            )
        return self._record("check_mesh_assets_ready", True)

    def _check_articulation_origin_far_from_geometry_impl(
        self,
        *,
        tol: float,
        reason: Optional[str],
        check_name: str,
        warn_only: bool = False,
    ) -> bool:
        record = self._record_warning_check if warn_only else self._record
        tol_f = _normalize_joint_origin_tol(tol)
        if not math.isfinite(tol_f) or tol_f < 0.0:
            return record(check_name, False, "tol must be finite and non-negative")
        r = (reason or "").strip()
        if tol_f > _JOINT_ORIGIN_TOL_DEFAULT and r:
            self.warn(f"Relaxed articulation-origin tolerance in use: tol={tol_f:.4g}. Reason: {r}")

        findings = find_joint_origin_distance_findings(
            self.model,
            asset_root=self._asset_root(),
            tol=tol_f,
            validate_model=not self._model_validated_strict,
        )

        failures: List[str] = []
        for finding in findings:
            failures.append(
                f"joint={finding.joint!r} parent={finding.parent!r} child={finding.child!r} "
                f"dist_parent={finding.parent_distance:.4g} "
                f"dist_child={finding.child_distance:.4g} tol={finding.tol:.4g}"
            )

        if failures:
            preview = "\n".join(failures[:10])
            more = "" if len(failures) <= 10 else f"\n... ({len(failures) - 10} more)"
            return record(
                check_name,
                False,
                f"Articulation origin(s) far from geometry:\n{preview}{more}",
            )
        return record(check_name, True)

    def fail_if_articulation_origin_far_from_geometry(
        self,
        *,
        tol: float = _JOINT_ORIGIN_TOL_DEFAULT,
        reason: Optional[str] = None,
        name: Optional[str] = None,
    ) -> bool:
        tol_f = _normalize_joint_origin_tol(tol)
        return self._check_articulation_origin_far_from_geometry_impl(
            tol=tol_f,
            reason=reason,
            check_name=name or f"fail_if_articulation_origin_far_from_geometry(tol={tol_f:.4g})",
        )

    def warn_if_articulation_origin_far_from_geometry(
        self,
        *,
        tol: float = _JOINT_ORIGIN_TOL_DEFAULT,
        reason: Optional[str] = None,
        name: Optional[str] = None,
    ) -> bool:
        self._warn_deprecated_default_helper(
            "warn_if_articulation_origin_far_from_geometry(...)",
            "Use prompt-specific exact `expect_*` checks for mount/placement realism, "
            "and investigate any finding with `probe_model` before relaxing thresholds.",
        )
        tol_f = _normalize_joint_origin_tol(tol)
        return self._check_articulation_origin_far_from_geometry_impl(
            tol=tol_f,
            reason=reason,
            check_name=name or f"warn_if_articulation_origin_far_from_geometry(tol={tol_f:.4g})",
            warn_only=True,
        )

    def fail_if_part_contains_disconnected_geometry_islands(
        self,
        *,
        tol: float = 1e-6,
        name: Optional[str] = None,
    ) -> bool:
        return self._check_part_contains_disconnected_geometry_islands_impl(
            tol=tol,
            check_name=name,
            warn_only=False,
        )

    def warn_if_part_contains_disconnected_geometry_islands(
        self,
        *,
        tol: float = 1e-6,
        name: Optional[str] = None,
    ) -> bool:
        return self._check_part_contains_disconnected_geometry_islands_impl(
            tol=tol,
            check_name=name,
            warn_only=True,
            warn_prefix="warn_if_part_contains_disconnected_geometry_islands",
        )

    def _check_part_contains_disconnected_geometry_islands_impl(
        self,
        *,
        tol: float,
        check_name: Optional[str],
        warn_only: bool,
        warn_prefix: str = "warn_if_part_contains_disconnected_geometry_islands",
    ) -> bool:
        record = self._record_warning_check if warn_only else self._record
        if float(tol) < 0.0:
            prefix = (
                warn_prefix if warn_only else "fail_if_part_contains_disconnected_geometry_islands"
            )
            return record(
                check_name or f"{prefix}(tol={float(tol):.4g})",
                False,
                "tol must be non-negative",
            )

        findings = find_part_geometry_connectivity_findings(
            self.model,
            asset_root=self._asset_root(),
            contact_tol=float(tol),
            validate_model=not self._model_validated_strict,
        )

        allowed_parts = set(self._allow_isolated_parts)
        failures: List[str] = []
        for finding in findings:
            if finding.part in allowed_parts:
                continue
            disconnected = ", ".join(finding.disconnected)
            failures.append(
                f"part={finding.part!r} connected={finding.connected}/{finding.total} "
                f"disconnected=[{disconnected}]"
            )

        prefix = warn_prefix if warn_only else "fail_if_part_contains_disconnected_geometry_islands"
        resolved_name = check_name or f"{prefix}(tol={float(tol):.4g})"
        if failures:
            preview = "\n".join(failures[:10])
            more = "" if len(failures) <= 10 else f"\n... ({len(failures) - 10} more)"
            if not warn_only:
                return self._record(
                    resolved_name,
                    False,
                    f"Disconnected geometry islands detected:\n{preview}{more}",
                    evidence={
                        "blocking_findings": [
                            {"kind": "geometry_island", **asdict(finding)}
                            for finding in findings
                            if finding.part not in allowed_parts
                        ]
                    },
                )
            return record(
                resolved_name, False, f"Disconnected geometry islands detected:\n{preview}{more}"
            )
        return record(resolved_name, True)

    def fail_if_isolated_parts(
        self,
        *,
        max_pose_samples: int = 1,
        contact_tol: Optional[float] = None,
        name: Optional[str] = None,
    ) -> bool:
        sample_count = int(max_pose_samples)
        if sample_count < 1:
            return self._record(
                name or "fail_if_isolated_parts(samples=0)",
                False,
                "max_pose_samples must be >= 1",
            )

        resolved_contact_tol = (
            default_floating_contact_tol_from_env() if contact_tol is None else float(contact_tol)
        )
        resolved_jointed_tol = default_jointed_support_tol_from_env()
        if resolved_contact_tol < 0.0:
            return self._record(
                name
                or (
                    "fail_if_isolated_parts("
                    f"samples={sample_count},contact_tol={resolved_contact_tol:.4g})"
                ),
                False,
                "contact_tol must be non-negative",
            )

        if name is not None:
            resolved_name = name
        elif max_pose_samples == 1 and contact_tol is None:
            resolved_name = "fail_if_isolated_parts()"
        else:
            resolved_name = (
                "fail_if_isolated_parts("
                f"samples={sample_count},contact_tol={resolved_contact_tol:.4g})"
            )

        findings = find_unsupported_parts(
            self.model,
            asset_root=self._asset_root(),
            max_pose_samples=sample_count,
            contact_tol=resolved_contact_tol,
            seed=int(self.seed),
            validate_model=not self._model_validated_strict,
            jointed_tol=resolved_jointed_tol,
        )
        if not findings:
            return self._record(resolved_name, True)

        allowed_parts = set(self._allow_isolated_parts)
        allowed_findings = [
            finding
            for finding in findings
            if (member_parts := _unsupported_finding_parts(finding))
            and all(name in allowed_parts for name in member_parts)
        ]
        remaining_findings = [
            finding
            for finding in findings
            if not (
                (member_parts := _unsupported_finding_parts(finding))
                and all(name in allowed_parts for name in member_parts)
            )
        ]

        if allowed_findings:
            allowed_names = sorted(
                {
                    part_name
                    for finding in allowed_findings
                    for part_name in _unsupported_finding_parts(finding)
                }
            )
            preview = "\n".join(
                _format_unsupported_part_finding(finding) for finding in allowed_findings[:10]
            )
            more = (
                "" if len(allowed_findings) <= 10 else f"\n... ({len(allowed_findings) - 10} more)"
            )
            self.warn(
                "Isolated parts detected but allowed by justification: "
                f"{len(allowed_names)} part(s) [{', '.join(repr(name) for name in allowed_names)}].\n"
                f"{preview}{more}"
            )

        if not remaining_findings:
            return self._record(resolved_name, True)

        preview = "\n".join(
            _format_unsupported_part_finding(finding) for finding in remaining_findings[:10]
        )
        more = (
            "" if len(remaining_findings) <= 10 else f"\n... ({len(remaining_findings) - 10} more)"
        )
        return self._record(
            resolved_name,
            False,
            "Isolated parts detected "
            "(floating support-disconnected component groups from the grounded body; "
            f"samples={sample_count}, contact_tol={resolved_contact_tol:.4g}):\n"
            f"{preview}{more}",
            evidence={
                "blocking_findings": [
                    {"kind": "isolated_part", **asdict(finding)} for finding in remaining_findings
                ]
            },
        )

    def fail_if_parts_overlap_in_current_pose(
        self,
        *,
        overlap_tol: Optional[float] = None,
        overlap_volume_tol: Optional[float] = None,
        fixed_pairs_advisory: bool = False,
        name: Optional[str] = None,
    ) -> bool:
        resolved_overlap_tol = (
            default_overlap_tol_from_env() if overlap_tol is None else float(overlap_tol)
        )
        resolved_overlap_volume_tol = (
            default_overlap_volume_tol_from_env()
            if overlap_volume_tol is None
            else float(overlap_volume_tol)
        )
        if name is not None:
            resolved_name = name
        elif overlap_tol is None and overlap_volume_tol is None:
            resolved_name = "fail_if_parts_overlap_in_current_pose()"
        else:
            resolved_name = (
                "fail_if_parts_overlap_in_current_pose("
                f"overlap_tol={resolved_overlap_tol:.4g},"
                f"overlap_volume_tol={resolved_overlap_volume_tol:.4g})"
            )

        overlaps = find_geometry_overlaps_in_poses(
            self.model,
            asset_root=self._asset_root(),
            poses=[dict(self._pose)],
            overlap_tol=resolved_overlap_tol,
            overlap_volume_tol=resolved_overlap_volume_tol,
            validate_model=not self._model_validated_strict,
        )

        overlaps_by_pair: Dict[Tuple[str, str], List[GeometryOverlap]] = {}
        advisory_by_pair: Dict[Tuple[str, str], GeometryOverlap] = {}
        allowed_pairs: Set[Tuple[str, str]] = set()
        allowed_overlap_count = 0
        welds = weld_group_of(self.model) if fixed_pairs_advisory else {}

        for overlap in overlaps:
            key = _pair_key(overlap.link_a, overlap.link_b)
            if self._overlap_is_allowed(overlap):
                allowed_pairs.add(key)
                allowed_overlap_count += 1
                continue
            wa = welds.get(overlap.link_a)
            if wa is not None and wa == welds.get(overlap.link_b):
                cur = advisory_by_pair.get(key)
                if cur is None or _overlap_rank(overlap) > _overlap_rank(cur):
                    advisory_by_pair[key] = overlap
                continue
            overlaps_by_pair.setdefault(key, []).append(overlap)

        if advisory_by_pair:
            adv = "; ".join(
                f"({k[0]!r},{k[1]!r}) vol={o.overlap_volume:.4g}"
                for k, o in sorted(advisory_by_pair.items())
            )
            self.warn(
                "Overlaps on FIXED-welded pairs (no relative motion) — advisory, not a hard "
                f"gate; Review should judge whether they read badly: {adv}"
            )

        if overlaps_by_pair:
            sorted_groups = [
                (key, sorted(group, key=_current_pose_overlap_sort_key))
                for key, group in overlaps_by_pair.items()
            ]
            sorted_groups.sort(key=lambda item: _current_pose_overlap_sort_key(item[1][0]))
            ordered = [overlap for _key, group in sorted_groups for overlap in group]
            visible = ordered[:_OVERLAP_FINDING_LIMIT]
            details = []
            identity_stats = {"values_clipped": 0, "characters_omitted": 0}
            accounting = _CompileFeedbackAccounting(
                overlap_total=len(ordered),
                overlap_rendered=len(visible),
                overlap_suppressed=len(ordered) - len(visible),
            )
            records_by_finding: list[_BindAttributionRecords] = []
            structured_findings: list[dict[str, object]] = []
            structured_placement_records: list[dict[str, object]] = []
            compact_evidence_by_index: dict[int, str] = {}
            for index, overlap in enumerate(ordered):
                records_by_side = _bind_attribution_records(self.model, overlap)
                records_by_finding.append(records_by_side)
                structured_findings.append(
                    _overlap_structured_finding(self.model, overlap, finding_index=index)
                )
                structured_placement_records.extend(
                    _overlap_placement_records(
                        overlap,
                        records_by_side,
                        finding_index=index,
                    )
                )
                compact_evidence = _compact_placement_evidence_line(
                    self.model,
                    overlap,
                    records_by_side,
                    identity_stats,
                )
                if compact_evidence is not None:
                    compact_evidence_by_index[index] = compact_evidence
            accounting.placement_evidence_total = len(compact_evidence_by_index)
            accounting.placement_evidence_rendered = sum(
                index < len(visible) for index in compact_evidence_by_index
            )
            accounting.placement_evidence_suppressed = (
                accounting.placement_evidence_total - accounting.placement_evidence_rendered
            )
            for index, overlap in enumerate(visible):
                pose = {
                    _clip_identity(name, identity_stats): value
                    for name, value in overlap.pose.items()
                }
                elem_a_label = _semantic_overlap_element_label(
                    self.model,
                    part_name=overlap.link_a,
                    visual_uid=overlap.elem_a_uid,
                    fallback_name=overlap.elem_a_name,
                    element_index=overlap.elem_a,
                )
                elem_b_label = _semantic_overlap_element_label(
                    self.model,
                    part_name=overlap.link_b,
                    visual_uid=overlap.elem_b_uid,
                    fallback_name=overlap.elem_b_name,
                    element_index=overlap.elem_b,
                )
                overlap_line = (
                    f"elements=({_clip_identity(overlap.link_a, identity_stats)}/"
                    f"{_clip_identity(elem_a_label, identity_stats)},"
                    f"{_clip_identity(overlap.link_b, identity_stats)}/"
                    f"{_clip_identity(elem_b_label, identity_stats)}) "
                    f"depth=({overlap.overlap_depth[0]:.4g},{overlap.overlap_depth[1]:.4g},{overlap.overlap_depth[2]:.4g}) "
                    f"min_depth={min(overlap.overlap_depth):.4g} vol={overlap.overlap_volume:.4g} "
                    f"pose={pose}"
                )
                details.append(_compile_feedback_line(overlap_line, category="overlap"))
                compact_evidence = compact_evidence_by_index.get(index)
                if compact_evidence is not None:
                    details.append(
                        _compile_feedback_line(
                            compact_evidence,
                            category="placement_evidence",
                        )
                    )

            suppressed = len(ordered) - len(visible)
            if suppressed:
                details.append(
                    _compile_feedback_line(
                        f"[suppressed {suppressed} overlap finding(s) after deterministic "
                        f"{_OVERLAP_FINDING_LIMIT}-item valve]",
                        preserve=True,
                    )
                )
            if identity_stats["values_clipped"]:
                details.append(
                    _compile_feedback_line(
                        f"[clipped {identity_stats['values_clipped']} dynamic identity value(s); "
                        f"suppressed {identity_stats['characters_omitted']} identity character(s) "
                        f"at the {_IDENTITY_LIMIT}-character valve]",
                        preserve=True,
                    )
                )
            details.append(_compile_feedback_accounting_line(accounting, identity_stats))
            return self._record(
                resolved_name,
                False,
                "Part overlaps detected "
                f"(overlap_tol={resolved_overlap_tol:.4g}, "
                f"overlap_volume_tol={resolved_overlap_volume_tol:.4g}):\n" + "\n".join(details),
                evidence={
                    "blocking_findings": structured_findings,
                    "overlap_placement_records": structured_placement_records,
                },
            )

        if allowed_overlap_count > 0:
            self.warn(
                "Overlaps detected but allowed by justification: "
                f"{allowed_overlap_count} overlaps across {len(allowed_pairs)} part pair(s)."
            )
        return self._record(resolved_name, True)

    def fail_if_parts_overlap_in_sampled_poses(
        self,
        *,
        max_pose_samples: int = 128,
        overlap_tol: Optional[float] = None,
        overlap_volume_tol: Optional[float] = None,
        ignore_adjacent: bool = False,
        ignore_fixed: bool = True,
        name: Optional[str] = None,
    ) -> bool:
        return self._check_overlaps_impl(
            max_pose_samples=max_pose_samples,
            overlap_tol=overlap_tol,
            overlap_volume_tol=overlap_volume_tol,
            ignore_adjacent=ignore_adjacent,
            ignore_fixed=ignore_fixed,
            check_name=name,
            warn_only=False,
        )

    def fail_if_articulation_overlaps(
        self,
        *,
        max_pose_samples: int = 128,
        overlap_tol: Optional[float] = None,
        overlap_volume_tol: Optional[float] = None,
        name: Optional[str] = None,
    ) -> bool:
        return self._check_overlaps_impl(
            max_pose_samples=max_pose_samples,
            overlap_tol=overlap_tol,
            overlap_volume_tol=overlap_volume_tol,
            ignore_adjacent=False,
            ignore_fixed=False,
            check_name=name,
            warn_only=False,
            pair_scope=self._articulation_overlap_pair_scope(),
            default_prefix="fail_if_articulation_overlaps",
        )

    def warn_if_articulation_overlaps(
        self,
        *,
        max_pose_samples: int = 128,
        overlap_tol: Optional[float] = None,
        overlap_volume_tol: Optional[float] = None,
        name: Optional[str] = None,
    ) -> bool:
        return self._check_overlaps_impl(
            max_pose_samples=max_pose_samples,
            overlap_tol=overlap_tol,
            overlap_volume_tol=overlap_volume_tol,
            ignore_adjacent=False,
            ignore_fixed=False,
            check_name=name,
            warn_only=True,
            pair_scope=self._articulation_overlap_pair_scope(),
            default_prefix="warn_if_articulation_overlaps",
        )

    def warn_if_overlaps(
        self,
        *,
        max_pose_samples: int = 128,
        overlap_tol: Optional[float] = None,
        overlap_volume_tol: Optional[float] = None,
        ignore_adjacent: bool = False,
        ignore_fixed: bool = True,
        name: Optional[str] = None,
    ) -> bool:
        self._warn_deprecated_default_helper(
            "warn_if_overlaps(...)",
            "Use prompt-specific exact `expect_*` checks for attachment and clearance first; "
            "add this only when a broad overlap sensor answers a specific uncertainty.",
        )
        return self._check_overlaps_impl(
            max_pose_samples=max_pose_samples,
            overlap_tol=overlap_tol,
            overlap_volume_tol=overlap_volume_tol,
            ignore_adjacent=ignore_adjacent,
            ignore_fixed=ignore_fixed,
            check_name=name,
            warn_only=True,
        )

    def _relation_ignored_pairs(
        self,
        *,
        ignore_adjacent: bool,
        ignore_fixed: bool,
    ) -> List[Tuple[str, str]]:
        ignored_pairs: List[Tuple[str, str]] = []
        if ignore_adjacent or ignore_fixed:
            joints = getattr(self.model, "articulations", None)
            if isinstance(joints, list):
                for joint in joints:
                    parent = getattr(joint, "parent", None)
                    child = getattr(joint, "child", None)
                    if not isinstance(parent, str) or not isinstance(child, str):
                        continue
                    joint_type = getattr(joint, "articulation_type", None)
                    if ignore_adjacent or (ignore_fixed and joint_type == ArticulationType.FIXED):
                        ignored_pairs.append((parent, child))
        return ignored_pairs

    def _articulation_overlap_pair_scope(self) -> Set[Tuple[str, str]]:
        scoped_pairs: Set[Tuple[str, str]] = set()
        joints = getattr(self.model, "articulations", None)
        if not isinstance(joints, list):
            return scoped_pairs
        for joint in joints:
            parent = getattr(joint, "parent", None)
            child = getattr(joint, "child", None)
            if not isinstance(parent, str) or not isinstance(child, str):
                continue
            joint_type = getattr(joint, "articulation_type", None)
            if joint_type not in {
                ArticulationType.REVOLUTE,
                ArticulationType.PRISMATIC,
                ArticulationType.CONTINUOUS,
            }:
                continue
            scoped_pairs.add(_pair_key(parent, child))
        return scoped_pairs

    def _check_overlaps_impl(
        self,
        *,
        max_pose_samples: int,
        overlap_tol: Optional[float],
        overlap_volume_tol: Optional[float],
        ignore_adjacent: bool,
        ignore_fixed: bool,
        check_name: Optional[str],
        warn_only: bool,
        pair_scope: Optional[Set[Tuple[str, str]]] = None,
        default_prefix: Optional[str] = None,
    ) -> bool:
        prefix = default_prefix or (
            "warn_if_overlaps" if warn_only else "fail_if_parts_overlap_in_sampled_poses"
        )
        if check_name is not None:
            resolved_name = check_name
        elif pair_scope is not None:
            resolved_name = f"{prefix}(samples={int(max_pose_samples)})"
        else:
            resolved_name = (
                f"{prefix}("
                f"samples={int(max_pose_samples)},"
                f"ignore_adjacent={bool(ignore_adjacent)},"
                f"ignore_fixed={bool(ignore_fixed)})"
            )
        record = self._record_warning_check if warn_only else self._record
        allowed_pairs = self._relation_ignored_pairs(
            ignore_adjacent=ignore_adjacent,
            ignore_fixed=ignore_fixed,
        )
        resolved_overlap_tol = (
            default_overlap_tol_from_env() if overlap_tol is None else float(overlap_tol)
        )
        resolved_overlap_volume_tol = (
            default_overlap_volume_tol_from_env()
            if overlap_volume_tol is None
            else float(overlap_volume_tol)
        )

        pair_relations: Dict[Tuple[str, str], str] = {}
        joints = getattr(self.model, "articulations", None)
        if isinstance(joints, list):
            for joint in joints:
                parent = getattr(joint, "parent", None)
                child = getattr(joint, "child", None)
                if not isinstance(parent, str) or not isinstance(child, str):
                    continue
                joint_type = getattr(joint, "articulation_type", None)
                key = _pair_key(parent, child)
                if isinstance(joint_type, ArticulationType):
                    pair_relations[key] = f"adjacent-{joint_type.value.lower()}"
                else:
                    pair_relations[key] = "adjacent"

        overlaps = find_geometry_overlaps(
            self.model,
            asset_root=self._asset_root(),
            max_pose_samples=int(max_pose_samples),
            overlap_tol=resolved_overlap_tol,
            overlap_volume_tol=resolved_overlap_volume_tol,
            allowed_pairs=allowed_pairs or None,
            seed=int(self.seed),
            validate_model=not self._model_validated_strict,
        )

        relevant_overlaps = overlaps
        if pair_scope is not None:
            relevant_overlaps = [
                overlap
                for overlap in overlaps
                if _pair_key(overlap.link_a, overlap.link_b) in pair_scope
            ]

        remaining: List[str] = []
        for o in relevant_overlaps:
            key = _pair_key(o.link_a, o.link_b)
            if self._overlap_is_allowed(o):
                continue
            relation = pair_relations.get(key, "unrelated")
            remaining.append(
                f"relation={relation} pair=({o.link_a!r},{o.link_b!r}) pose_index={o.pose_index} "
                f"depth=({o.overlap_depth[0]:.4g},{o.overlap_depth[1]:.4g},{o.overlap_depth[2]:.4g}) "
                f"min_depth={min(o.overlap_depth):.4g} vol={o.overlap_volume:.4g} "
                f"elem_a={_format_overlap_element(o.elem_a, o.elem_a_name, o.elem_a_geometry)} "
                f"elem_b={_format_overlap_element(o.elem_b, o.elem_b_name, o.elem_b_geometry)} "
                f"pose={o.pose}"
            )

        if remaining:
            preview = "\n".join(remaining[:8])
            more = "" if len(remaining) <= 8 else f"\n... ({len(remaining) - 8} more)"
            return record(
                resolved_name,
                False,
                "Overlaps detected "
                f"(overlap_tol={resolved_overlap_tol:.4g}, "
                f"overlap_volume_tol={resolved_overlap_volume_tol:.4g}):\n"
                f"{preview}{more}",
            )

        if relevant_overlaps and (self._allow_pairs or self._allow_elems):
            self.warn(
                f"Overlaps detected but allowed by justification: {len(relevant_overlaps)} overlaps."
            )
        return record(resolved_name, True)

    def warn_if_coplanar_surfaces(
        self,
        *,
        max_pose_samples: int = 32,
        plane_tol: float = 0.001,
        min_overlap: float = 0.02,
        min_overlap_ratio: float = 0.35,
        ignore_adjacent: bool = True,
        ignore_fixed: bool = True,
        name: Optional[str] = None,
    ) -> bool:
        resolved_name = name or (
            f"warn_if_coplanar_surfaces("
            f"samples={int(max_pose_samples)},"
            f"plane_tol={float(plane_tol):.4g},"
            f"min_overlap={float(min_overlap):.4g},"
            f"min_overlap_ratio={float(min_overlap_ratio):.4g},"
            f"ignore_adjacent={bool(ignore_adjacent)},"
            f"ignore_fixed={bool(ignore_fixed)})"
        )
        record = self._record_warning_check

        if int(max_pose_samples) <= 0:
            return record(resolved_name, False, "max_pose_samples must be positive")
        plane_tol_f = float(plane_tol)
        if not math.isfinite(plane_tol_f) or plane_tol_f < 0.0:
            return record(resolved_name, False, "plane_tol must be finite and non-negative")
        min_overlap_f = float(min_overlap)
        if not math.isfinite(min_overlap_f) or min_overlap_f < 0.0:
            return record(resolved_name, False, "min_overlap must be finite and non-negative")
        min_overlap_ratio_f = float(min_overlap_ratio)
        if (
            not math.isfinite(min_overlap_ratio_f)
            or min_overlap_ratio_f < 0.0
            or min_overlap_ratio_f > 1.0
        ):
            return record(
                resolved_name,
                False,
                "min_overlap_ratio must be finite and between 0 and 1",
            )

        links = getattr(self.model, "parts", None)
        joints = getattr(self.model, "articulations", None)
        if not isinstance(links, list) or not isinstance(joints, list):
            return record(resolved_name, False, "model.parts/articulations missing")

        pair_relations: Dict[Tuple[str, str], str] = {}
        ignored_pairs: set[Tuple[str, str]] = set()
        for joint in joints:
            parent = getattr(joint, "parent", None)
            child = getattr(joint, "child", None)
            if not isinstance(parent, str) or not isinstance(child, str):
                continue
            joint_type = getattr(joint, "articulation_type", None)
            key = _pair_key(parent, child)
            if isinstance(joint_type, ArticulationType):
                relation = f"adjacent-{joint_type.value.lower()}"
            else:
                relation = "adjacent"
            pair_relations[key] = relation
            if ignore_adjacent or (ignore_fixed and joint_type == ArticulationType.FIXED):
                ignored_pairs.add(key)

        def is_coplanar_allowed(finding: _CoplanarFinding) -> bool:
            key = _pair_key(finding.link_a, finding.link_b)
            if key not in self._allow_coplanar_pairs:
                return False
            for allowed_key, elem_a, elem_b, _reason in self._allow_coplanar_elems:
                if allowed_key != key:
                    continue
                if elem_a is None and elem_b is None:
                    return True
                names = (finding.elem_a_name, finding.elem_b_name)
                if (elem_a in names) and (elem_b in names):
                    return True
            return False

        risk_rank = {"low": 0, "medium": 1, "high": 2}

        def classify_risk(
            *,
            relation: str,
            plane_delta: float,
            overlap_ratio: float,
            thin_pair: bool,
        ) -> str:
            if relation.startswith("adjacent-"):
                return "low"
            strong_plane = plane_delta <= max(plane_tol_f * 0.25, 1e-5)
            strong_overlap = overlap_ratio >= max(min_overlap_ratio_f + 0.25, 0.75)
            moderate_overlap = overlap_ratio >= max(min_overlap_ratio_f + 0.1, 0.5)
            if strong_plane and strong_overlap and thin_pair:
                return "high"
            if (strong_plane and moderate_overlap) or (strong_overlap and thin_pair):
                return "medium"
            return "low"

        def better_finding(candidate: _CoplanarFinding, current: _CoplanarFinding) -> bool:
            if risk_rank[candidate.risk] != risk_rank[current.risk]:
                return risk_rank[candidate.risk] > risk_rank[current.risk]
            if abs(candidate.plane_delta - current.plane_delta) > 1e-9:
                return candidate.plane_delta < current.plane_delta
            if abs(candidate.overlap_ratio - current.overlap_ratio) > 1e-9:
                return candidate.overlap_ratio > current.overlap_ratio
            return candidate.area > current.area

        try:
            poses = generate_pose_samples(
                self.model,
                max_samples=int(max_pose_samples),
                seed=int(self.seed),
            )
        except Exception as exc:
            return record(resolved_name, False, str(exc))

        findings_by_pair: Dict[Tuple[str, str], _CoplanarFinding] = {}
        for pose_index, pose in enumerate(poses):
            world_tfs = compute_part_world_transforms(self.model, dict(pose))
            part_elem_bounds: Dict[str, List[Tuple[Optional[str], Optional[str], AABB]]] = {}
            for link in links:
                link_name = getattr(link, "name", None)
                if not isinstance(link_name, str):
                    continue
                world_tf = world_tfs.get(link_name)
                if world_tf is None:
                    continue
                elem_bounds = self._part_geometry_items_with_local_aabbs(link_name)
                if not elem_bounds:
                    continue
                part_elem_bounds[link_name] = [
                    (elem_name, elem_geometry, _transform_aabb(local_aabb, world_tf))
                    for elem_name, elem_geometry, local_aabb in elem_bounds
                ]

            names = sorted(part_elem_bounds.keys())
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    link_a = names[i]
                    link_b = names[j]
                    key = _pair_key(link_a, link_b)
                    if key in ignored_pairs:
                        continue
                    relation = pair_relations.get(key, "unrelated")
                    best_pair: Optional[_CoplanarFinding] = None
                    for elem_a_name, elem_a_geometry, aabb_a in part_elem_bounds[link_a]:
                        for elem_b_name, elem_b_geometry, aabb_b in part_elem_bounds[link_b]:
                            for axis_name in ("x", "y", "z"):
                                orth_axes = tuple(
                                    axis for axis in ("x", "y", "z") if axis != axis_name
                                )
                                overlap_0 = _aabb_axis_overlap(aabb_a, aabb_b, axis=orth_axes[0])
                                overlap_1 = _aabb_axis_overlap(aabb_a, aabb_b, axis=orth_axes[1])
                                if overlap_0 < min_overlap_f or overlap_1 < min_overlap_f:
                                    continue

                                axis_idx = _axis_index(axis_name)
                                face_pairs = (
                                    (
                                        "min",
                                        "min",
                                        abs(
                                            float(aabb_a[0][axis_idx]) - float(aabb_b[0][axis_idx])
                                        ),
                                    ),
                                    (
                                        "min",
                                        "max",
                                        abs(
                                            float(aabb_a[0][axis_idx]) - float(aabb_b[1][axis_idx])
                                        ),
                                    ),
                                    (
                                        "max",
                                        "min",
                                        abs(
                                            float(aabb_a[1][axis_idx]) - float(aabb_b[0][axis_idx])
                                        ),
                                    ),
                                    (
                                        "max",
                                        "max",
                                        abs(
                                            float(aabb_a[1][axis_idx]) - float(aabb_b[1][axis_idx])
                                        ),
                                    ),
                                )
                                face_a, face_b, plane_delta = min(
                                    face_pairs, key=lambda item: item[2]
                                )
                                if plane_delta > plane_tol_f:
                                    continue

                                area = overlap_0 * overlap_1
                                face_area_a = _aabb_axis_span(
                                    aabb_a, axis=orth_axes[0]
                                ) * _aabb_axis_span(aabb_a, axis=orth_axes[1])
                                face_area_b = _aabb_axis_span(
                                    aabb_b, axis=orth_axes[0]
                                ) * _aabb_axis_span(aabb_b, axis=orth_axes[1])
                                smaller_face_area = max(min(face_area_a, face_area_b), 1e-9)
                                overlap_ratio = area / smaller_face_area
                                if overlap_ratio < min_overlap_ratio_f:
                                    continue

                                thickness_a = _aabb_axis_span(aabb_a, axis=axis_name)
                                thickness_b = _aabb_axis_span(aabb_b, axis=axis_name)
                                in_plane_min_a = max(
                                    min(
                                        _aabb_axis_span(aabb_a, axis=orth_axes[0]),
                                        _aabb_axis_span(aabb_a, axis=orth_axes[1]),
                                    ),
                                    1e-9,
                                )
                                in_plane_min_b = max(
                                    min(
                                        _aabb_axis_span(aabb_b, axis=orth_axes[0]),
                                        _aabb_axis_span(aabb_b, axis=orth_axes[1]),
                                    ),
                                    1e-9,
                                )
                                thin_pair = (
                                    thickness_a / in_plane_min_a <= 0.12
                                    and thickness_b / in_plane_min_b <= 0.12
                                )
                                candidate = _CoplanarFinding(
                                    link_a=link_a,
                                    link_b=link_b,
                                    relation=relation,
                                    risk=classify_risk(
                                        relation=relation,
                                        plane_delta=plane_delta,
                                        overlap_ratio=overlap_ratio,
                                        thin_pair=thin_pair,
                                    ),
                                    pose_index=pose_index,
                                    pose=dict(pose),
                                    axis_name=axis_name,
                                    overlap_axis_0=orth_axes[0],
                                    overlap_axis_1=orth_axes[1],
                                    face_a=face_a,
                                    face_b=face_b,
                                    plane_delta=plane_delta,
                                    overlap_0=overlap_0,
                                    overlap_1=overlap_1,
                                    area=area,
                                    overlap_ratio=overlap_ratio,
                                    thin_pair=thin_pair,
                                    elem_a_name=elem_a_name,
                                    elem_a_geometry=elem_a_geometry,
                                    elem_b_name=elem_b_name,
                                    elem_b_geometry=elem_b_geometry,
                                )
                                if best_pair is None or better_finding(candidate, best_pair):
                                    best_pair = candidate

                    if best_pair is None:
                        continue
                    if is_coplanar_allowed(best_pair):
                        continue
                    current = findings_by_pair.get(key)
                    if current is None or better_finding(best_pair, current):
                        findings_by_pair[key] = best_pair

        if findings_by_pair:
            max_risk = max(findings_by_pair.values(), key=lambda item: risk_rank[item.risk]).risk
            headline = (
                "Low-confidence coplanar-surface hints detected"
                if max_risk == "low"
                else "Coplanar or nearly coplanar surfaces detected"
            )
            findings = [
                f"risk={candidate.risk} relation={candidate.relation} "
                f"pair=({link_a!r},{link_b!r}) pose_index={candidate.pose_index} "
                f"axis={candidate.axis_name} faces=({candidate.face_a},{candidate.face_b}) "
                f"plane_delta={candidate.plane_delta:.4g} "
                f"{candidate.overlap_axis_0}_overlap={candidate.overlap_0:.4g} "
                f"{candidate.overlap_axis_1}_overlap={candidate.overlap_1:.4g} "
                f"overlap_ratio={candidate.overlap_ratio:.4g} thin_pair={candidate.thin_pair} "
                f"elem_a={candidate.elem_a_name!r}:{candidate.elem_a_geometry or '?'} "
                f"elem_b={candidate.elem_b_name!r}:{candidate.elem_b_geometry or '?'} "
                f"pose={candidate.pose}"
                for (link_a, link_b), candidate in sorted(findings_by_pair.items())
            ]
            preview = "\n".join(findings[:8])
            more = "" if len(findings) <= 8 else f"\n... ({len(findings) - 8} more)"
            return record(
                resolved_name,
                False,
                f"{headline} "
                f"(max_risk={max_risk}; warning-tier heuristic; adjacent flush mounts can be intentional):\n"
                f"{preview}{more}",
            )
        return record(resolved_name, True)
