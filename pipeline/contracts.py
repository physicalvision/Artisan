from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Severity(str, Enum):
    BLOCKING = "blocking"
    MAJOR = "major"
    MINOR = "minor"


class SupportMode(str, Enum):
    GROUNDED = "grounded"
    EXTERNAL_MOUNT = "external_mount"
    MOBILE = "mobile"


def validate_support_mode(brief: object) -> SupportMode:
    if not isinstance(brief, dict):
        raise ValueError("asset_brief must be a JSON object with support_mode")
    value = brief.get("support_mode")
    try:
        return SupportMode(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(mode.value for mode in SupportMode)
        raise ValueError(f"asset_brief.support_mode must be one of: {allowed}") from exc


class Decision(str, Enum):
    PROCEED = "proceed"
    REVISE = "revise"
    PROCEED_WITH_LIMITATION = "proceed_with_documented_limitation"


class StabilityVerdict(str, Enum):
    ACCEPTABLE = "acceptable"
    BLOCKING = "blocking"


class TerminationReason(str, Enum):
    MAX_TURNS = "max_turns"
    BUDGET_RESERVE = "budget_reserve"
    COST_LIMIT = "cost_limit"
    VALIDATION_ENVIRONMENT_FAILED = "validation_environment_failed"
    HARD_FAILURE = "hard_failure"
    SELFCHECK_LIMIT = "selfcheck_limit"


@dataclass(slots=True)
class StabilityAdjudication:
    pose: str
    finding_code: str
    verdict: StabilityVerdict
    rationale: str
    evidence: list[str] = field(default_factory=list)
    issue_id: str | None = None


@dataclass(slots=True)
class DynamicsAdjudication:
    joint: str
    finding_code: str
    verdict: StabilityVerdict
    rationale: str
    evidence: list[str] = field(default_factory=list)
    issue_id: str | None = None


@dataclass(slots=True)
class CodeRef:

    file: str
    symbol: str | None = None
    line: int | None = None


@dataclass(slots=True)
class Issue:
    issue_id: str
    severity: Severity
    phase: str
    target: str
    problem: str
    required_change: str
    evidence: list[str] = field(default_factory=list)
    code_ref: CodeRef | None = None

    def is_blocking(self) -> bool:
        return self.severity is Severity.BLOCKING


@dataclass(slots=True)
class SubmittedReview:
    kind: str = field(default="submitted", init=False)
    report_id: str
    phase: str
    realism_rating: int
    summary: str
    decision: Decision
    issues: list[Issue] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    rubric: dict[str, int] = field(default_factory=dict)
    issue_total: int = 0
    stability_adjudications: list[StabilityAdjudication] = field(default_factory=list)
    dynamics_adjudications: list[DynamicsAdjudication] = field(default_factory=list)

    def blocking_issues(self) -> list[Issue]:
        return [i for i in self.issues if i.is_blocking()]


ReviewReport = SubmittedReview


@dataclass(slots=True)
class ReviewIncomplete:
    report_id: str
    phase: str
    summary: str
    reason: TerminationReason
    kind: str = field(default="incomplete", init=False)

    @property
    def termination_reason(self) -> str:
        return self.reason.value


ReviewResult = SubmittedReview | ReviewIncomplete


@dataclass(slots=True)
class DeveloperHandoff:

    compile_summary: str = ""
    turns: int = 0
    blocking_signals: list[str] = field(default_factory=list)
    parts: list[str] = field(default_factory=list)
    joints: list[dict[str, Any]] = field(default_factory=list)
    declared_allowances: list[str] = field(default_factory=list)
    repaired_issue_ids: list[str] = field(default_factory=list)
    developer_note: dict[str, Any] = field(default_factory=dict)
    candidate_id: str | None = None
    physics_validation: dict[str, Any] = field(default_factory=dict)
    motion_contact_candidates: list[dict[str, Any]] = field(default_factory=list)
    geometry_island_warnings: list[dict[str, Any]] = field(default_factory=list)
    motion_support_findings: list[dict[str, Any]] = field(default_factory=list)
    driver_reach_findings: list[dict[str, Any]] = field(default_factory=list)
    declared_gate_witnesses: list[dict[str, Any]] = field(default_factory=list)
    motion_identity: dict[str, Any] = field(default_factory=dict)
    coupling_rollout: list[dict[str, Any]] = field(default_factory=list)
    articulation_rollout: list[dict[str, Any]] = field(default_factory=list)
    joint_dynamics_evidence: dict[str, Any] = field(default_factory=dict)
    contact_exclusion_evidence: list[dict[str, Any]] = field(default_factory=list)
    connection_cleanliness_findings: list[dict[str, Any]] = field(default_factory=list)
    full_collision_penetrations: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class ExportSmokeReport:
    formats: list[str]
    loaded_ok: bool
    joints_complete: bool
    closed_state_stable: bool
    open_state_reachable: bool
    no_initial_explosion: bool
    issues: list[Issue] = field(default_factory=list)
    notes: str = ""

    def passed(self) -> bool:
        return (
            self.loaded_ok
            and self.joints_complete
            and self.closed_state_stable
            and self.open_state_reachable
            and self.no_initial_explosion
        )


_ENUMS = (Severity, SupportMode, Decision, StabilityVerdict, TerminationReason)


def _encode(obj: Any) -> Any:
    if isinstance(obj, _ENUMS):
        return obj.value
    raise TypeError(f"unserializable: {type(obj)!r}")


def to_json(obj: Any) -> str:
    return json.dumps(asdict(obj), indent=2, default=_encode, sort_keys=False)


def write_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(obj), encoding="utf-8")
    return path


def issue_from_dict(d: dict[str, Any]) -> Issue:
    cr = d.get("code_ref")
    return Issue(
        issue_id=d["issue_id"],
        severity=Severity(d["severity"]),
        phase=d["phase"],
        target=d["target"],
        problem=d["problem"],
        required_change=d["required_change"],
        evidence=list(d.get("evidence", [])),
        code_ref=CodeRef(**cr) if cr else None,
    )


def review_report_from_dict(d: dict[str, Any]) -> ReviewReport:
    if d.get("kind") == "incomplete" or d.get("termination_reason"):
        raise ValueError("incomplete review payload is not a submitted ReviewReport")
    return ReviewReport(
        report_id=d["report_id"],
        phase=d["phase"],
        realism_rating=int(d["realism_rating"]),
        summary=d["summary"],
        decision=Decision(d["decision"]),
        issues=[issue_from_dict(i) for i in d.get("issues", [])],
        evidence=list(d.get("evidence", [])),
        rubric={str(k): int(v) for k, v in (d.get("rubric") or {}).items()},
        issue_total=int(d.get("issue_total", len(d.get("issues", [])))),
        stability_adjudications=[
            StabilityAdjudication(
                pose=str(item["pose"]),
                finding_code=str(item["finding_code"]),
                verdict=StabilityVerdict(item["verdict"]),
                rationale=str(item["rationale"]),
                evidence=[str(value) for value in item.get("evidence", [])],
                issue_id=(str(item["issue_id"]) if item.get("issue_id") else None),
            )
            for item in d.get("stability_adjudications", [])
        ],
        dynamics_adjudications=[
            DynamicsAdjudication(
                joint=str(item["joint"]),
                finding_code=str(item["finding_code"]),
                verdict=StabilityVerdict(item["verdict"]),
                rationale=str(item["rationale"]),
                evidence=[str(value) for value in item.get("evidence", [])],
                issue_id=(str(item["issue_id"]) if item.get("issue_id") else None),
            )
            for item in d.get("dynamics_adjudications", [])
        ],
    )


def review_result_from_dict(d: dict[str, Any]) -> ReviewResult:
    reason = d.get("reason") or d.get("termination_reason")
    if d.get("kind") == "incomplete" or reason:
        return ReviewIncomplete(
            report_id=str(d["report_id"]),
            phase=str(d["phase"]),
            summary=str(d.get("summary") or "review incomplete"),
            reason=TerminationReason(str(reason or TerminationReason.HARD_FAILURE.value)),
        )
    return review_report_from_dict(d)


def developer_handoff_from_dict(d: dict[str, Any]) -> DeveloperHandoff:
    return DeveloperHandoff(
        compile_summary=d.get("compile_summary", ""),
        turns=int(d.get("turns", 0)),
        blocking_signals=list(d.get("blocking_signals", [])),
        parts=list(d.get("parts", [])),
        joints=list(d.get("joints", [])),
        declared_allowances=list(d.get("declared_allowances", [])),
        repaired_issue_ids=list(d.get("repaired_issue_ids", [])),
        developer_note=dict(d.get("developer_note", {})),
        candidate_id=d.get("candidate_id"),
        physics_validation=dict(d.get("physics_validation", {})),
        motion_contact_candidates=list(d.get("motion_contact_candidates", [])),
        geometry_island_warnings=list(d.get("geometry_island_warnings", [])),
        motion_support_findings=list(d.get("motion_support_findings", [])),
        driver_reach_findings=list(d.get("driver_reach_findings", [])),
        declared_gate_witnesses=list(d.get("declared_gate_witnesses", [])),
        motion_identity=dict(d.get("motion_identity", {})),
        coupling_rollout=list(d.get("coupling_rollout", [])),
        articulation_rollout=list(d.get("articulation_rollout", [])),
        joint_dynamics_evidence=dict(d.get("joint_dynamics_evidence", {})),
        contact_exclusion_evidence=list(d.get("contact_exclusion_evidence", [])),
        connection_cleanliness_findings=list(d.get("connection_cleanliness_findings", [])),
        full_collision_penetrations=list(d.get("full_collision_penetrations", [])),
    )


def review_report_from_json(path: Path) -> ReviewReport:
    return review_report_from_dict(json.loads(path.read_text(encoding="utf-8")))


def review_result_from_json(path: Path) -> ReviewResult:
    return review_result_from_dict(json.loads(path.read_text(encoding="utf-8")))
