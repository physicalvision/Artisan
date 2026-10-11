
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Phase(str, Enum):
    P0_INTAKE = "P0_intake"
    P1_BLUEPRINT = "P1_blueprint"
    P2_DESIGN_BOARD = "P2_design_board"
    P3_BLUEPRINT_REVIEW = "P3_blueprint_review"
    P4_BUILD = "P4_build"
    P5_REVIEW = "P5_review"
    P8_EXPORT = "P8_export_smoke"
    P9_FINAL = "P9_final_record"
    DONE = "done"
    FAILED = "failed"


MEDIA_KIND_IMAGE = "image"


class AgentRole(str, Enum):

    DESIGN = "design"
    CODING = "coding"
    REVIEW = "review"
    EXPORT = "export"


PHASE_AGENT: dict[Phase, AgentRole] = {
    Phase.P0_INTAKE: AgentRole.DESIGN,
    Phase.P1_BLUEPRINT: AgentRole.DESIGN,
    Phase.P2_DESIGN_BOARD: AgentRole.DESIGN,
    Phase.P3_BLUEPRINT_REVIEW: AgentRole.DESIGN,
    Phase.P4_BUILD: AgentRole.CODING,
    Phase.P5_REVIEW: AgentRole.REVIEW,
    Phase.P8_EXPORT: AgentRole.EXPORT,
    Phase.P9_FINAL: AgentRole.EXPORT,
}


def agent_for_phase(phase: Phase | str) -> AgentRole:
    try:
        key = phase if isinstance(phase, Phase) else Phase(phase)
    except ValueError:
        return AgentRole.EXPORT
    return PHASE_AGENT.get(key, AgentRole.EXPORT)


TERMINAL: frozenset[Phase] = frozenset({Phase.DONE, Phase.FAILED})

FORWARD_ORDER: tuple[Phase, ...] = (
    Phase.P0_INTAKE,
    Phase.P1_BLUEPRINT,
    Phase.P2_DESIGN_BOARD,
    Phase.P3_BLUEPRINT_REVIEW,
    Phase.P4_BUILD,
    Phase.P5_REVIEW,
    Phase.P8_EXPORT,
    Phase.P9_FINAL,
)


@dataclass(slots=True)
class SessionRef:

    role: str
    path: str | None = None


@dataclass(slots=True)
class PipelineState:
    run_id: str
    prompt: str
    record_id: str
    repo_root: str
    image_path: str | None = None
    references: list[dict] = field(default_factory=list)

    phase: Phase = Phase.P0_INTAKE
    status: str = "running"
    termination_reason: str | None = None
    terminal_details: dict[str, Any] = field(default_factory=dict)

    blueprint_path: str | None = None
    design_board_path: str | None = None
    asset_build_path: str | None = None
    compile_report_path: str | None = None
    candidate_id: str | None = None
    candidate_manifest_path: str | None = None
    physics_report_path: str | None = None
    reviewed_physics_report_sha256: str | None = None
    fallback_candidate: dict[str, Any] = field(default_factory=dict)
    candidate_dirty: str = "clean"
    salvage: dict[str, Any] = field(default_factory=dict)

    phase_iterations: dict[str, int] = field(default_factory=dict)
    last_realism_rating: int | None = None
    last_review_signal: dict[str, Any] = field(default_factory=dict)
    review_no_progress_streak: int = 0
    open_issue_ids: list[str] = field(default_factory=list)
    open_blocking_issue_ids: list[str] = field(default_factory=list)

    last_coding_handoff: dict[str, Any] = field(default_factory=dict)

    phases_passed: list[str] = field(default_factory=list)

    agent_sessions: list[SessionRef] = field(default_factory=list)
    cost_usd: float = 0.0

    elapsed_seconds: float = 0.0
    active_seconds: float = 0.0
    updated_at: float = 0.0

    def bump_iteration(self, phase: Phase) -> int:
        n = self.phase_iterations.get(phase.value, 0) + 1
        self.phase_iterations[phase.value] = n
        return n

    def iterations(self, phase: Phase) -> int:
        return self.phase_iterations.get(phase.value, 0)

    def mark_passed(self, phase: Phase) -> None:
        if phase.value not in self.phases_passed:
            self.phases_passed.append(phase.value)

    def has_passed(self, phase: Phase) -> bool:
        return phase.value in self.phases_passed

    def session_ref(self, role: str) -> SessionRef:
        for s in self.agent_sessions:
            if s.role == role:
                return s
        ref = SessionRef(role=role)
        self.agent_sessions.append(ref)
        return ref

    def image_references(self) -> list[str]:
        return [r["path"] for r in self.references if r.get("media_kind") == MEDIA_KIND_IMAGE]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["phase"] = self.phase.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineState:
        d = dict(d)
        d["phase"] = Phase(d["phase"])
        d["agent_sessions"] = [SessionRef(**s) for s in d.get("agent_sessions", [])]
        if not d.get("references") and d.get("image_path"):
            d["references"] = [{"path": d["image_path"], "media_kind": MEDIA_KIND_IMAGE}]
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


def checkpoint_path(repo_root: Path, run_id: str) -> Path:
    from storage.layout import StorageLayout

    return StorageLayout(root=repo_root).run_dir(run_id) / "pipeline_state.json"


PAUSE_GAP_CAP_S = 7200.0


def events_timing(run_dir: Path) -> tuple[float, float]:
    ev = run_dir / "workspace" / "events.jsonl"
    if not ev.exists():
        return 0.0, 0.0
    ts: list[float] = []
    try:
        for line in ev.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            t = json.loads(line).get("ts")
            if isinstance(t, (int, float)):
                ts.append(float(t))
    except Exception:  # noqa: BLE001 — timing must never break a checkpoint
        return 0.0, 0.0
    if len(ts) < 2:
        return 0.0, 0.0
    ts.sort()
    elapsed = ts[-1] - ts[0]
    active = sum(min(ts[i] - ts[i - 1], PAUSE_GAP_CAP_S) for i in range(1, len(ts)))
    return round(elapsed, 1), round(active, 1)


def save_checkpoint(state: PipelineState) -> Path:
    path = checkpoint_path(Path(state.repo_root), state.run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    state.updated_at = round(time.time(), 1)
    state.elapsed_seconds, state.active_seconds = events_timing(path.parent)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def load_checkpoint(repo_root: Path, run_id: str) -> PipelineState:
    path = checkpoint_path(repo_root, run_id)
    return PipelineState.from_dict(json.loads(path.read_text(encoding="utf-8")))
