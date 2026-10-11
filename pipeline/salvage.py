from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.contracts import TerminationReason

SALVAGEABLE_REASONS = frozenset(
    {
        TerminationReason.BUDGET_RESERVE.value,
        TerminationReason.COST_LIMIT.value,
    }
)

MODEL_SEARCH_PATHS = (
    Path("model.py"),
    Path("fallback_candidate") / "model.py",
    Path("failed_attempt") / "model.py",
)

SALVAGED_EXPORTS_DIRNAME = "salvaged_exports"

GRADE_REVIEWED_BLOCKING = "reviewed_blocking"
GRADE_REVIEWED_ADVISORY = "reviewed_advisory"
GRADE_REVIEWED_CLEAN = "reviewed_clean"
GRADE_UNREVIEWED = "unreviewed"


@dataclass(frozen=True)
class SalvageDecision:

    salvageable: bool
    refusal: str | None = None
    grade: str | None = None
    candidate_id: str | None = None
    manifest_path: Path | None = None
    model_path: Path | None = None
    termination_reason: str | None = None
    candidate_dirty: str | None = None
    realism_rating: int | None = None
    open_blocking_issue_ids: tuple[str, ...] = ()
    issues: tuple[dict[str, Any], ...] = ()

    def as_metadata(self) -> dict[str, Any]:
        return {
            "degraded": True,
            "grade": self.grade,
            "termination_reason": self.termination_reason,
            "candidate_id": self.candidate_id,
            "candidate_dirty": self.candidate_dirty,
            "realism_rating": self.realism_rating,
            "open_blocking_issue_ids": list(self.open_blocking_issue_ids),
            "issues": [dict(issue) for issue in self.issues],
            "model_source": self.model_path.name if self.model_path else None,
        }


def _refuse(reason: str, **known: Any) -> SalvageDecision:
    return SalvageDecision(salvageable=False, refusal=reason, **known)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _grade(state: dict[str, Any], review: dict[str, Any], bound: bool) -> str:
    if not bound or not review:
        return GRADE_UNREVIEWED
    if state.get("open_blocking_issue_ids"):
        return GRADE_REVIEWED_BLOCKING
    return GRADE_REVIEWED_ADVISORY if (review.get("issues") or []) else GRADE_REVIEWED_CLEAN


def evaluate_salvage(run_dir: Path) -> SalvageDecision:
    from sdk.runtime.release_candidate import sha256_file, verify_release_candidate

    run_dir = Path(run_dir)
    workspace = run_dir / "workspace"
    state = _load_json(run_dir / "pipeline_state.json")
    reason = str(state.get("termination_reason") or "")
    known: dict[str, Any] = {
        "termination_reason": reason or None,
        "candidate_dirty": state.get("candidate_dirty"),
    }

    if state.get("status") not in ("incomplete", "failed"):
        return _refuse(
            f"run is not a terminal unsuccessful run (status={state.get('status')!r})", **known
        )
    if (workspace / "exports" / "object.urdf").is_file():
        return _refuse("run already delivered through P8; nothing to salvage", **known)
    if reason not in SALVAGEABLE_REASONS:
        return _refuse(
            f"termination reason {reason!r} is not budget exhaustion "
            f"(salvageable: {sorted(SALVAGEABLE_REASONS)})",
            **known,
        )

    manifest_path = workspace / "candidate" / "manifest.json"
    if not manifest_path.is_file():
        return _refuse("no candidate manifest: the run never had an accepted build", **known)
    try:
        manifest = verify_release_candidate(manifest_path, require_accepted=True)
    except ImportError:
        raise
    except Exception as exc:  # noqa: BLE001 - any integrity fault refuses salvage
        return _refuse(
            f"candidate failed integrity verification: {type(exc).__name__}: {exc}", **known
        )

    known["candidate_id"] = manifest.candidate_id
    known["manifest_path"] = manifest_path

    blueprint_path = workspace / "blueprint.json"
    if not blueprint_path.is_file():
        return _refuse("workspace has no blueprint to bind the candidate to", **known)
    if sha256_file(blueprint_path) != manifest.blueprint_sha256:
        return _refuse(
            "candidate was built against a different blueprint than this workspace holds "
            "(stale attempt in a reused run directory)",
            **known,
        )

    model_path = next(
        (
            workspace / rel
            for rel in MODEL_SEARCH_PATHS
            if (workspace / rel).is_file() and sha256_file(workspace / rel) == manifest.model_sha256
        ),
        None,
    )
    if model_path is None:
        return _refuse(
            "no surviving model.py matches the candidate's model_sha256; the exported "
            "asset is orphaned from its source and would produce a record that cannot "
            "rebuild itself",
            **known,
        )
    known["model_path"] = model_path

    review = _load_json(workspace / "reports" / "P5_review.json")
    bound = bool(
        state.get("reviewed_physics_report_sha256")
        and state.get("reviewed_physics_report_sha256") == manifest.physics_report_sha256
    )
    issues = tuple(review.get("issues") or []) if bound else ()
    return SalvageDecision(
        salvageable=True,
        grade=_grade(state, review, bound),
        realism_rating=state.get("last_realism_rating") if bound else None,
        open_blocking_issue_ids=tuple(state.get("open_blocking_issue_ids") or ()) if bound else (),
        issues=issues,
        **known,
    )


def publish_salvaged_candidate(run_dir: Path, decision: SalvageDecision) -> Path:
    from sdk.runtime.release_candidate import publish_release_candidate

    if not decision.salvageable or decision.manifest_path is None:
        raise ValueError("publish_salvaged_candidate requires a salvageable decision")
    out_dir = Path(run_dir) / "workspace" / SALVAGED_EXPORTS_DIRNAME
    publish_release_candidate(
        decision.manifest_path,
        out_dir,
        expected_candidate_id=decision.candidate_id,
    )
    return out_dir
