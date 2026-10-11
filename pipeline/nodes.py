from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.agents.coding import CodingAgent
from pipeline.agents.design import DesignAgent
from pipeline.agents.design_harness import MOVING_JOINT_TYPES, derive_limits
from pipeline.agents.review import ReviewAgent
from pipeline.backends.base import ImplementationBackend
from pipeline.contracts import (
    CodeRef,
    Issue,
    ReviewIncomplete,
    ReviewReport,
    ReviewResult,
    Severity,
    TerminationReason,
    validate_support_mode,
    write_json,
)
from pipeline.gates import (
    access_gate_check,
    blueprint_presence_check,
    coupling_conformance_check,
    joint_coupling_check,
    joint_reference_frame_check,
    manipulation_binding_check,
    manipulation_tasks_check,
)
from pipeline.graph import StageOutcome, StageStatus
from pipeline.policies import DEFAULT_POLICIES, MAX_NONBLOCKING_PENDING_ISSUES
from pipeline.state import Phase, PipelineState
from sdk.runtime import runtime as sdk_runtime

logger = logging.getLogger(__name__)


def _candidate_dirty_snapshot(model_path: Path) -> tuple[str | None, str | None, str | None]:
    from sdk.runtime.release_candidate import (
        motion_geometry_sha256_for_model,
        sha256_file,
    )

    model_digest = sha256_file(model_path) if model_path.is_file() else None
    geometry_digest = motion_geometry_sha256_for_model(model_path) if model_path.is_file() else None
    tasks_path = model_path.parent / "tasks.json"
    tasks_digest = sha256_file(tasks_path) if tasks_path.is_file() else None
    return model_digest, geometry_digest, tasks_digest


def _candidate_dirty_kind(
    before: tuple[str | None, str | None, str | None],
    after: tuple[str | None, str | None, str | None],
    *,
    reported: str,
) -> str:
    if before[:2] != after[:2] or reported == "geometry":
        return "geometry"
    if before[2] != after[2] or reported == "metadata":
        return "metadata"
    return "clean"


_DISCARDED_EDIT_NAME = "post_green_edit_discarded.py"
_GREEN_ROLLBACK_RECORD_NAME = "post_green_rollback.json"


def _green_compile_report(model_path: Path) -> dict[str, object] | None:
    report_path = Path(model_path).parent / "_compile_report.json"
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not payload.get("ok"):
        return None
    model_sha256 = payload.get("model_sha256")
    if not isinstance(model_sha256, str) or not model_sha256:
        return None
    return payload


def _design_termination_reason(reason: str) -> TerminationReason:
    return {
        "self_review_incomplete": TerminationReason.SELFCHECK_LIMIT,
        "cost_limit": TerminationReason.COST_LIMIT,
        "max_turns": TerminationReason.MAX_TURNS,
    }.get(reason, TerminationReason.HARD_FAILURE)


def _adoptable_green_physics_report(reports_dir: Path, green_sha256: str) -> str | None:
    try:
        payload = json.loads((reports_dir / "physics_validation.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        isinstance(payload, dict)
        and payload.get("model_sha256") == green_sha256
        and payload.get("status") == "pass"
    ):
        return str(reports_dir / "physics_validation.json")
    return None


def _evaluable_green_state(workspace_root: Path, model_path: Path) -> dict[str, object] | None:
    from pipeline.harness_sdk import load_green_snapshot
    from sdk.runtime.release_candidate import dependency_sha256_for_model, sha256_file

    model_path = Path(model_path)
    if not model_path.is_file():
        return None
    report = _green_compile_report(model_path)
    if report is None:
        return None
    green_sha256 = str(report["model_sha256"])
    reports_dir = Path(workspace_root) / "reports"
    workspace_sha256 = sha256_file(model_path)
    if workspace_sha256 == green_sha256:
        return {
            "event": f"workspace already at green {green_sha256}",
            "rolled_back": False,
            "restored_model_sha256": green_sha256,
            "green_physics_report_path": _adoptable_green_physics_report(reports_dir, green_sha256),
        }
    snapshot = load_green_snapshot(model_path, green_sha256)
    if snapshot is None:
        return None
    if snapshot.get("dependency_sha256") != dependency_sha256_for_model(model_path):
        return None

    reports_dir.mkdir(parents=True, exist_ok=True)
    green_physics_report = _adoptable_green_physics_report(reports_dir, green_sha256)
    discarded_path = reports_dir / _DISCARDED_EDIT_NAME
    archived_previous: str | None = None
    if discarded_path.is_file():
        previous_sha256 = sha256_file(discarded_path)
        if previous_sha256 != workspace_sha256:
            archived = discarded_path.with_suffix(f".{previous_sha256[:12]}.py")
            discarded_path.replace(archived)
            archived_previous = str(archived)
    shutil.copy2(model_path, discarded_path)
    shutil.copy2(Path(str(snapshot["source_path"])), model_path)

    record: dict[str, object] = {
        "event": (f"post-green edit discarded at cost fuse; restored green {green_sha256}"),
        "rolled_back": True,
        "restored_model_sha256": green_sha256,
        "discarded_model_sha256": workspace_sha256,
        "discarded_source_path": str(discarded_path),
        "archived_previous_discard": archived_previous,
        "dependency_sha256": snapshot.get("dependency_sha256"),
        "green_compile_summary": snapshot.get("compile_summary", ""),
        "green_physics_report_path": green_physics_report,
        "rolled_back_at": time.time(),
    }
    (reports_dir / _GREEN_ROLLBACK_RECORD_NAME).write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.warning("P4: %s", record["event"])
    return record


def _manipulation_binding_issue(model_path: Path, blueprint: dict[str, object]) -> Issue | None:
    tasks_path = model_path.parent / "tasks.json"
    if not tasks_path.is_file():
        if not blueprint.get("tasks"):
            return None
        failures = ["tasks.json is missing although the blueprint declares manipulation tasks"]
    else:
        failures = []

    if tasks_path.is_file():
        try:
            tasks_json = json.loads(tasks_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures = [f"tasks.json could not be read: {type(exc).__name__}: {exc}"]
        else:
            from sdk.runtime import runtime as sdk_runtime

            try:
                probe = sdk_runtime.probe_summary(model_path, sdk_package="sdk")
            except Exception as exc:  # noqa: BLE001 — fail closed for an authored annotation
                failures = [f"built model could not be probed: {type(exc).__name__}: {exc}"]
            else:
                if not probe.get("build_ok", True):
                    failures = [
                        "built model could not be probed: "
                        f"{probe.get('build_error', 'build_ok=False')}"
                    ]
                else:
                    result = manipulation_binding_check(
                        tasks_json,
                        list(probe.get("parts", [])),
                        list(probe.get("joints", [])),
                        visuals_by_part=dict(probe.get("visuals_by_part", {})),
                        couplings=list(blueprint.get("couplings") or []),
                    )
                    if result.passed:
                        return None
                    failures = list(result.failures)

    return Issue(
        issue_id="manipulation_binding",
        severity=Severity.MAJOR,
        phase=Phase.P4_BUILD.value,
        target="tasks.json",
        problem="The downstream manipulation annotation is structurally invalid: "
        + "; ".join(failures),
        required_change=(
            "Rewrite tasks.json as schema_version 2.0 with one valid task object per operation. "
            "Each task must bind contact_target.link to a real built link and "
            "contact_target.visuals to the minimal non-empty union of uniquely named visuals "
            "on that link, then provide its reachable movable target joint, success threshold, "
            "and documented task_intent. Keep rigid controls as named visuals rather than "
            "creating fixed child links solely for annotation. If a task moves the WHOLE object "
            "instead of one of its joints (push a cart, roll a trolley across the floor), that "
            'is not a joint task: set "target_joint": "__whole_body__" with '
            '"task_intent": "push_whole_body" and a success of '
            '{"joint": "__whole_body__", "to_value": <metres of root travel>}. Never bind such a '
            "task to a caster or wheel joint -- spinning a wheel is not the operation."
        ),
        evidence=failures,
    )


def _load_review_candidate(
    model_path: Path,
    manifest_path: Path,
    report_path: Path,
    *,
    expected_candidate_id: str | None = None,
) -> tuple[object, dict[str, object]]:
    from sdk.runtime.release_candidate import load_manifest, sha256_file

    manifest = load_manifest(manifest_path)
    if manifest.status != "accepted":
        raise ValueError(f"candidate status is {manifest.status!r}, expected 'accepted'")
    if manifest.model_sha256 != sha256_file(model_path):
        raise ValueError("model.py hash does not match accepted candidate")
    if not isinstance(manifest.physics_report_path, str) or not manifest.physics_report_path:
        raise ValueError("candidate physics report path is missing")
    report_relative = Path(manifest.physics_report_path)
    if report_relative.is_absolute() or ".." in report_relative.parts:
        raise ValueError("candidate physics report path escapes candidate root")
    if not isinstance(manifest.physics_report_sha256, str) or not manifest.physics_report_sha256:
        raise ValueError("candidate physics report SHA-256 is missing")
    embedded_report_path = Path(manifest_path).parent / report_relative
    if not embedded_report_path.is_file():
        raise ValueError("embedded physics report is missing")
    if sha256_file(embedded_report_path) != manifest.physics_report_sha256:
        raise ValueError("embedded physics report hash does not match manifest")
    workspace_report_path = Path(report_path)
    if not workspace_report_path.is_file():
        raise ValueError("workspace physics report is missing")
    if sha256_file(workspace_report_path) != manifest.physics_report_sha256:
        raise ValueError("workspace physics report hash does not match manifest")
    physics = json.loads(embedded_report_path.read_text(encoding="utf-8"))
    if not isinstance(physics, dict):
        raise ValueError("physics report must be a JSON object")
    if physics.get("candidate_id") != manifest.candidate_id:
        raise ValueError("physics report candidate_id does not match manifest")
    if expected_candidate_id is not None and expected_candidate_id != manifest.candidate_id:
        raise ValueError("checkpoint candidate_id does not match manifest")
    return manifest, physics


def _replace_open_issues(state: PipelineState, issues: list[Issue]) -> None:
    state.open_issue_ids = [issue.issue_id for issue in issues]
    state.open_blocking_issue_ids = [issue.issue_id for issue in issues if issue.is_blocking()]


def _coupling_conformance_issue(model_path: Path, blueprint: dict) -> Issue | None:
    couplings = blueprint.get("couplings") or []
    if not couplings:
        return None

    def _unverified_issue(reason: str) -> Issue | None:
        return Issue(
            issue_id="coupling_conformance_unverified",
            severity=Severity.BLOCKING,
            phase=Phase.P4_BUILD.value,
            target="model.py couplings",
            problem=f"Declared couplings could not be verified: {reason}",
            required_change=(
                "The structural probe of the built model failed, so the declared "
                "couplings could not be checked against the blueprint. Make model.py "
                "load and probe cleanly (e.g. ensure managed inputs / mesh_from_input "
                "are produced inside the build), then re-wire each declared coupling."
            ),
            evidence=[reason],
            code_ref=CodeRef(file="model.py"),
        )

    try:
        probe = sdk_runtime.probe_summary(model_path, sdk_package="sdk")
    except Exception as exc:  # noqa: BLE001 — structural probe is best-effort
        return _unverified_issue(f"probe raised {type(exc).__name__}: {exc}")
    if not probe.get("build_ok", True):
        return _unverified_issue(str(probe.get("build_error", "build_ok=False")))
    result = coupling_conformance_check(list(probe.get("joints", [])), couplings)
    if result.passed:
        return None
    detail = "; ".join(result.failures)
    declared = "; ".join(
        f"{c.get('follower')}<-{c.get('driver')} ({c.get('type')}, spec={c.get('spec')})"
        for c in couplings
        if isinstance(c, dict)
    )
    return Issue(
        issue_id="coupling_conformance",
        severity=Severity.BLOCKING,
        phase=Phase.P4_BUILD.value,
        target="model.py couplings",
        problem=f"Built couplings do not match the blueprint: {detail}",
        required_change=(
            "Wire each declared coupling with model.couple(driver, follower, sense=..., "
            "mapping=...) copying the blueprint spec verbatim — do not hand-compute the "
            f"multiplier, do not invent undeclared couplings. Declared couplings: {declared}"
        ),
        evidence=list(result.failures),
        code_ref=CodeRef(file="model.py"),
    )


def _latest_review_renders(render_root: Path | None, *, limit: int = 2) -> list[Path]:
    if not render_root or not Path(render_root).exists():
        return []
    rounds = sorted(
        (d for d in Path(render_root).glob("P5_review_r*") if d.is_dir()), key=lambda d: d.name
    )
    if not rounds:
        return []
    pngs = sorted(rounds[-1].rglob("*.png"))
    pngs.sort(
        key=lambda p: ("perspective" in p.name.lower()) + ("open" in p.name.lower()), reverse=True
    )
    return pngs[:limit]


def _select_pending_issues(report: ReviewReport, phase_label: str = "") -> list[Issue]:
    blocking = report.blocking_issues()
    blocking_ids = {id(i) for i in blocking}
    nonblocking = [i for i in report.issues if id(i) not in blocking_ids]
    extra = nonblocking[:MAX_NONBLOCKING_PENDING_ISSUES]
    if len(nonblocking) > MAX_NONBLOCKING_PENDING_ISSUES:
        logger.info(
            "%s: %d non-blocking issues found; transmitting first %d to Coding "
            "alongside %d blocking (all persisted in the report)",
            phase_label,
            len(nonblocking),
            MAX_NONBLOCKING_PENDING_ISSUES,
            len(blocking),
        )
    return blocking + extra


@dataclass
class NodeBundle:
    state: PipelineState
    paths: dict[str, Path]
    design: DesignAgent
    review: ReviewAgent
    coding: CodingAgent
    backend: ImplementationBackend
    recorder: object | None = None
    blueprint: dict = field(default_factory=dict)
    pending_issues: list[Issue] = field(default_factory=list)
    last_blocking: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    coding_cost_usd: float = 0.0
    prior_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        self.prior_cost_usd = self.state.cost_usd
        self.cost_usd = self.state.cost_usd

    def _bp(self) -> dict:
        blueprint_path = self.paths.get("blueprint")
        if not self.blueprint and blueprint_path is not None and blueprint_path.exists():
            self.blueprint = json.loads(blueprint_path.read_text(encoding="utf-8"))
        return self.blueprint

    def refresh_cost(self) -> float:
        self.cost_usd = (
            self.prior_cost_usd + self.design.cost_usd + self.review.cost_usd + self.coding_cost_usd
        )
        self.state.cost_usd = self.cost_usd
        return self.cost_usd


def build_nodes(bundle: NodeBundle) -> dict[Phase, object]:
    s, p = bundle.state, bundle.paths

    async def p0_intake(_s, _c) -> StageOutcome:
        result = await bundle.design.intake(
            s.prompt, p["blueprint"], image_paths=s.image_references()
        )
        if not result.success:
            return StageOutcome(
                StageStatus.INCOMPLETE,
                f"design intake: {result.reason}",
                termination_reason=_design_termination_reason(result.reason),
            )
        brief = result.asset_brief or {}
        s.blueprint_path = str(p["blueprint"])
        return StageOutcome(StageStatus.PASS, f"brief: {brief.get('name', '?')}")

    async def p1_blueprint(_s, _c) -> StageOutcome:
        seed = json.loads(p["blueprint"].read_text(encoding="utf-8"))
        brief = seed.get("asset_brief", {})
        result = await bundle.design.blueprint(
            s.prompt,
            brief,
            p["blueprint"],
            image_paths=s.image_references(),
            reopen=s.iterations(Phase.P3_BLUEPRINT_REVIEW) > 0,
        )
        if not result.success:
            return StageOutcome(
                StageStatus.INCOMPLETE,
                f"blueprint: {result.reason}",
                termination_reason=_design_termination_reason(result.reason),
            )
        bp = result.blueprint or {}
        bundle.blueprint = bp
        s.blueprint_path = str(p["blueprint"])
        g = blueprint_presence_check(bp)
        return StageOutcome(StageStatus.PASS if g.passed else StageStatus.REDO, g.summary)

    async def p2_design_board(_s, _c) -> StageOutcome:
        bp = bundle._bp()
        m = await bundle.design.generate_design_board(
            bp.get("asset_brief", {}),
            bp,
            p["design_board_png"],
            p["design_board_md"],
            p["design_board_meta"],
            image_paths=s.image_references(),
        )
        s.design_board_path = m.get("png") or m.get("md")
        return StageOutcome(StageStatus.PASS, f"design board: {m['status']}")

    async def p3_blueprint_review(_s, _c) -> StageOutcome:
        bp = bundle._bp()
        try:
            support_mode = validate_support_mode(bp.get("asset_brief"))
            support_error = None
        except ValueError as exc:
            support_mode = None
            support_error = str(exc)
        joints = bp.get("joints", [])
        norm = []
        for j in joints:
            lo, hi = derive_limits(j.get("type"), j.get("travel"), j.get("rest"))
            norm.append(
                {
                    "name": j.get("name"),
                    "joint_type": j.get("type"),
                    "joint_axis": j.get("axis"),
                    "joint_limit_lower": lo,
                    "joint_limit_upper": hi,
                }
            )
        g = joint_reference_frame_check(norm)
        cg = joint_coupling_check(joints, bp.get("couplings") or [])
        ag = access_gate_check(bp.get("parts") or [], joints)
        mt = manipulation_tasks_check(bp)
        if support_error is not None:
            status, reason = "redo", support_error
        elif not joints:
            status, reason = "redo", "blueprint has no joints"
        elif not any(str(j.get("type")) in MOVING_JOINT_TYPES for j in joints):
            status, reason = "redo", "blueprint has no moving joint (every joint is fixed)"
        elif not g.passed:
            status, reason = "redo", g.summary
        elif not cg.passed:
            status, reason = "redo", cg.summary
        elif not ag.passed:
            status, reason = "redo", ag.summary
        else:
            status, reason = "pass", "blueprint joints valid"
        report_rows = [
            {
                "name": j.get("name"),
                "type": j.get("type"),
                "axis": j.get("axis"),
                "travel": j.get("travel"),
                "rest": j.get("rest"),
                "limit_lower": n["joint_limit_lower"],
                "limit_upper": n["joint_limit_upper"],
            }
            for j, n in zip(joints, norm)
        ]
        try:
            p["reports"].mkdir(parents=True, exist_ok=True)
            (p["reports"] / "P3_blueprint_review.json").write_text(
                json.dumps(
                    {
                        "phase": "P3_blueprint_review",
                        "status": status,
                        "summary": reason,
                        "support_mode": {
                            "passed": support_error is None,
                            "value": support_mode.value if support_mode is not None else None,
                            "summary": support_error or "asset brief support mode valid",
                        },
                        "joints": report_rows,
                        "reference_frame": {"passed": g.passed, "summary": g.summary},
                        "coupling": {"passed": cg.passed, "summary": cg.summary},
                        "access_gate": {
                            "passed": ag.passed,
                            "summary": ag.summary,
                            "failures": ag.failures,
                        },
                        "manipulation_tasks": {
                            "passed": mt.passed,
                            "summary": mt.summary,
                            "failures": mt.failures,
                        },
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass
        if status == "redo":
            return StageOutcome(StageStatus.REDO, reason)
        bundle.design.reconcile(p["blueprint"], "blueprint_review", ["joints validated"])
        return StageOutcome(StageStatus.PASS, reason)

    async def p4_build(_s, ctx) -> StageOutcome:
        board = None
        if (
            s.design_board_path
            and s.design_board_path.endswith(".png")
            and Path(s.design_board_path).exists()
        ):
            board = Path(s.design_board_path)
        remaining = None
        review_reserve = 0.0
        if ctx.max_cost_usd is not None:
            others = bundle.refresh_cost() - bundle.coding_cost_usd
            policies = getattr(ctx, "policies", DEFAULT_POLICIES)
            review_reserve = policies.review_reserve_usd(ctx.max_cost_usd)
            remaining = max(
                bundle.coding_cost_usd,
                ctx.max_cost_usd - others - review_reserve,
            )
        references = [Path(pth) for pth in s.image_references() if Path(pth).exists()]
        dirty_before = _candidate_dirty_snapshot(p["model"])
        is_review_repair = bool(bundle.pending_issues)
        if bundle.pending_issues:
            repaired_ids = [i.issue_id for i in bundle.pending_issues]
            evidence = _latest_review_renders(p.get("renders"), limit=2)
            res = await bundle.coding.repair(
                bundle.pending_issues,
                evidence_images=evidence,
                review_round=s.iterations(Phase.P5_REVIEW),
                max_cost_usd=remaining,
            )
        elif bundle.coding.has_history():
            repaired_ids = []
            ch = s.last_coding_handoff or {}
            res = await bundle.coding.continue_build(
                compile_summary=ch.get("compile_summary", ""),
                open_items=list(ch.get("blocking_signals", [])),
                max_cost_usd=remaining,
            )
        else:
            repaired_ids = []
            res = await bundle.coding.build(
                bundle._bp(),
                design_board_path=board,
                reference_image_paths=references,
                max_cost_usd=remaining,
            )
        s.last_coding_handoff = {
            "compile_summary": res.compile_summary,
            "turns": res.turns,
            "blocking_signals": list(res.blocking_signals),
            "developer_note": dict(res.developer_note),
            "repaired_issue_ids": repaired_ids,
        }
        dirty_after = _candidate_dirty_snapshot(p["model"])
        candidate_dirty = _candidate_dirty_kind(
            dirty_before,
            dirty_after,
            reported=str(getattr(res, "candidate_dirty", "clean")),
        )
        s.candidate_dirty = candidate_dirty
        s.last_coding_handoff["candidate_dirty"] = candidate_dirty
        s.candidate_id = getattr(res, "candidate_id", None)
        s.candidate_manifest_path = getattr(res, "candidate_manifest_path", None)
        s.physics_report_path = getattr(res, "physics_report_path", None)
        s.reviewed_physics_report_sha256 = None
        s.asset_build_path = str(p["model"])
        bundle.coding_cost_usd = res.cost_usd
        bundle.refresh_cost()

        def _acceptance_outcome(summary: str) -> StageOutcome:
            from sdk.runtime.accepted_inspection import accepted_inspection, candidate_reference

            reference = None
            if s.candidate_manifest_path and s.candidate_id:
                reference = candidate_reference(
                    Path(s.candidate_manifest_path),
                    s.candidate_id,
                )
            try:
                with accepted_inspection(p["model"], reference):
                    return _inspect_acceptance_outcome(summary)
            except Exception as exc:
                return StageOutcome(StageStatus.FAIL, f"P4 candidate inspection failed: {exc}")

        def _inspect_acceptance_outcome(summary: str) -> StageOutcome:
            conformance_issue = _coupling_conformance_issue(p["model"], bundle._bp())
            if conformance_issue is not None:
                bundle.pending_issues = [conformance_issue]
                return StageOutcome(
                    StageStatus.REDO, f"coupling not wired as declared: {conformance_issue.problem}"
                )
            manipulation_issue = _manipulation_binding_issue(p["model"], bundle._bp())
            if manipulation_issue is not None:
                bundle.pending_issues = [manipulation_issue]
                return StageOutcome(
                    StageStatus.REDO,
                    f"manipulation annotation invalid: {manipulation_issue.problem}",
                )
            report_path = str(s.physics_report_path or "").strip()
            if not report_path:
                return StageOutcome(
                    StageStatus.FAIL,
                    "accepted P4 physics report is unavailable: "
                    "no physics validation was recorded for this coding round",
                )
            try:
                physics = json.loads(Path(report_path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                return StageOutcome(
                    StageStatus.FAIL,
                    f"accepted P4 physics report is unavailable: {type(exc).__name__}: {exc}",
                )
            from sdk.runtime.runtime import model_source_sha256

            status = str(physics.get("status") or "")
            if status != "pass":
                return StageOutcome(
                    StageStatus.FAIL,
                    f"accepted P4 physics report did not pass: status={status or 'missing'!r}",
                )
            reported_sha = str(physics.get("model_sha256") or "")
            actual_sha = model_source_sha256(Path(p["model"]))
            if reported_sha != actual_sha:
                return StageOutcome(
                    StageStatus.FAIL,
                    "accepted P4 physics report describes different source: "
                    f"report model_sha256={reported_sha[:12] or 'missing'}, "
                    f"workspace model.py={actual_sha[:12]}",
                )
            s.last_coding_handoff["joint_dynamics_evidence"] = dict(
                physics.get("joint_dynamics") or {}
            )
            bundle.pending_issues = []
            s.fallback_candidate = {}
            s.candidate_dirty = "clean"
            return StageOutcome(StageStatus.PASS, summary)

        if res.success:
            return _acceptance_outcome(res.compile_summary)
        reviewed_candidate_restored = False
        if is_review_repair and res.reason != "max_turns" and s.fallback_candidate:
            from pipeline.candidate_recovery import restore_reviewed_candidate_after_failure

            try:
                restore_reviewed_candidate_after_failure(
                    s,
                    p["root"],
                    failure_reason=res.reason,
                )
            except Exception as exc:  # noqa: BLE001 - failed recovery must be explicit
                return StageOutcome(
                    StageStatus.FAIL,
                    "coding repair failed and reviewed candidate recovery also failed: "
                    f"{type(exc).__name__}: {exc}",
                )
            reviewed_candidate_restored = True
        green_state = None
        if res.reason == "cost_limit" and not reviewed_candidate_restored:
            green_state = _evaluable_green_state(p["root"], p["model"])
        if green_state is not None:
            prior_pending_issues = list(bundle.pending_issues)
            prior_physics_report_path = s.physics_report_path
            green_report = green_state.get("green_physics_report_path")
            if isinstance(green_report, str) and green_report:
                s.physics_report_path = green_report
            restored_outcome = _acceptance_outcome(f"{green_state['event']}; {res.compile_summary}")
            if restored_outcome.status is StageStatus.PASS:
                return restored_outcome
            bundle.pending_issues = prior_pending_issues
            s.physics_report_path = prior_physics_report_path
            logger.info(
                "P4: green source did not clear the acceptance gates (%s); "
                "the cost-fuse outcome stands",
                restored_outcome.summary,
            )
        fuse_note = f" ({green_state['event']})" if green_state is not None else ""
        if res.reason == "cost_limit" and review_reserve > 0.0:
            return StageOutcome(
                StageStatus.INCOMPLETE,
                (
                    "coding stopped at the reserved Review budget boundary; "
                    f"review_reserve_usd={review_reserve:.6g}{fuse_note}"
                ),
                termination_reason=TerminationReason.BUDGET_RESERVE,
            )
        if res.reason == "cost_limit":
            return StageOutcome(
                StageStatus.FAIL,
                f"cost limit in coding{fuse_note}",
                termination_reason=TerminationReason.COST_LIMIT,
            )
        if res.reason == "validation_environment_failed":
            return StageOutcome(
                StageStatus.FAIL,
                "required P4 validation infrastructure failed: " + "; ".join(res.blocking_signals),
                termination_reason=TerminationReason.VALIDATION_ENVIRONMENT_FAILED,
            )
        if res.reason == "hard_failure":
            return StageOutcome(
                StageStatus.FAIL,
                "deterministic P4 integrity failure: " + "; ".join(res.blocking_signals),
                termination_reason=TerminationReason.HARD_FAILURE,
            )
        bundle.last_blocking = res.blocking_signals
        if res.reason == "max_turns":
            return StageOutcome(
                StageStatus.INCOMPLETE,
                f"coding turn budget exhausted: {res.compile_summary}",
                termination_reason=TerminationReason.MAX_TURNS,
            )
        return StageOutcome(
            StageStatus.FAIL,
            f"coding harness stopped unexpectedly ({res.reason}): {res.compile_summary}",
        )

    async def _review(phase: Phase, ctx) -> StageOutcome:
        from pipeline.agents.review import parse_allowances
        from pipeline.contracts import DeveloperHandoff
        from sdk.runtime import runtime as sdk_runtime

        try:
            manifest_value = s.candidate_manifest_path or p.get("candidate_manifest")
            report_value = s.physics_report_path or p.get("physics_report")
            if not manifest_value or not report_value:
                raise ValueError("accepted candidate manifest/report paths are missing")
            manifest_path = Path(manifest_value)
            report_path = Path(report_value)
            manifest, physics = _load_review_candidate(
                p["model"],
                manifest_path,
                report_path,
                expected_candidate_id=s.candidate_id,
            )
        except Exception as exc:  # noqa: BLE001 - integrity failure must stop before LLM review
            return StageOutcome(
                StageStatus.FAIL,
                f"candidate integrity error before review: {type(exc).__name__}: {exc}",
            )

        src = p["model"].read_text(encoding="utf-8")
        bp = bundle._bp()
        allowances = parse_allowances(src)
        from sdk.runtime.accepted_inspection import accepted_inspection, candidate_reference

        try:
            reference = candidate_reference(
                manifest_path,
                manifest.candidate_id,
            )
            with accepted_inspection(p["model"], reference):
                probe = sdk_runtime.probe_summary(p["model"], sdk_package="sdk")
        except Exception as exc:
            return StageOutcome(StageStatus.FAIL, f"P5 candidate inspection failed: {exc}")
        ch = s.last_coding_handoff or {}
        evidence = sdk_runtime.build_motion_review_evidence(
            p["model"],
            src,
            sdk_package="sdk",
            cache_path=p["reports"] / "motion_scan.json",
            blueprint=bp,
        )
        joint_dynamics = dict(physics.get("joint_dynamics") or {})
        handoff = DeveloperHandoff(
            compile_summary=ch.get("compile_summary", ""),
            turns=int(ch.get("turns", 0)),
            blocking_signals=list(ch.get("blocking_signals", [])),
            parts=list(probe.get("parts", [])),
            joints=list(probe.get("joints", [])),
            declared_allowances=allowances,
            repaired_issue_ids=list(ch.get("repaired_issue_ids", [])),
            developer_note=dict(ch.get("developer_note", {})),
            candidate_id=manifest.candidate_id,
            physics_validation=physics,
            motion_contact_candidates=evidence["motion_contact_candidates"],
            geometry_island_warnings=evidence["geometry_island_warnings"],
            motion_support_findings=evidence.get("motion_support_findings", []),
            driver_reach_findings=evidence.get("driver_reach_findings", []),
            declared_gate_witnesses=evidence.get("declared_gate_witnesses", []),
            motion_identity=dict(evidence.get("motion_identity", {})),
            contact_exclusion_evidence=evidence.get("contact_exclusion_evidence", []),
            connection_cleanliness_findings=evidence.get("connection_cleanliness_findings", []),
            full_collision_penetrations=evidence.get("full_collision_penetrations", []),
            joint_dynamics_evidence=joint_dynamics,
        )
        references = [Path(pth) for pth in s.image_references() if Path(pth).exists()]
        board = (
            Path(s.design_board_path)
            if s.design_board_path
            and s.design_board_path.endswith(".png")
            and Path(s.design_board_path).exists()
            else None
        )
        review_ceiling = None
        if ctx.max_cost_usd is not None:
            others = bundle.refresh_cost() - bundle.review.cost_usd
            review_ceiling = max(bundle.review.cost_usd, ctx.max_cost_usd - others)
        result: ReviewResult = await bundle.review.review(
            phase.value,
            blueprint=bp,
            handoff=handoff,
            allowances=allowances,
            reference_images=references,
            design_board=board,
            user_prompt=s.prompt,
            max_cost_usd=review_ceiling,
        )
        if isinstance(result, ReviewIncomplete):
            write_json(result, p["reports"] / f"{phase.value}.json")
            status = (
                StageStatus.INCOMPLETE
                if result.reason in {TerminationReason.MAX_TURNS, TerminationReason.BUDGET_RESERVE}
                else StageStatus.FAIL
            )
            return StageOutcome(
                status,
                result.summary,
                termination_reason=result.reason,
            )
        report: ReviewReport = result
        s.reviewed_physics_report_sha256 = manifest.physics_report_sha256
        all_issues = report.issues
        verdict = ctx.policies.review_verdict(
            report.realism_rating,
            report.rubric,
            has_blocking_issue=bool(report.blocking_issues()),
        )
        report.decision = ctx.policies.review_decision(verdict)
        write_json(report, p["reports"] / f"{phase.value}.json")
        _replace_open_issues(s, all_issues)
        bundle.pending_issues = _select_pending_issues(report, phase.value)
        retry_in_place = verdict == "redo" and not bundle.pending_issues
        if verdict == "redo" and bundle.pending_issues:
            from pipeline.candidate_recovery import snapshot_reviewed_candidate

            try:
                snapshot_reviewed_candidate(s, p["root"])
            except Exception as exc:  # noqa: BLE001 - never repair without a recovery point
                return StageOutcome(
                    StageStatus.FAIL,
                    "cannot snapshot the reviewed candidate before repair: "
                    f"{type(exc).__name__}: {exc}",
                    realism_rating=report.realism_rating,
                    rubric=dict(report.rubric or {}),
                    issues=all_issues,
                )
        return StageOutcome(
            StageStatus.PASS if verdict == "pass" else StageStatus.REDO,
            report.summary,
            realism_rating=report.realism_rating,
            rubric=dict(report.rubric or {}),
            issues=all_issues,
            retry_in_place=retry_in_place,
        )

    async def p5_review(_s, _c) -> StageOutcome:
        return await _review(Phase.P5_REVIEW, _c)

    async def p8_export(_s, _c) -> StageOutcome:
        if s.open_blocking_issue_ids:
            return StageOutcome(
                StageStatus.FAIL,
                "cannot publish candidate with unresolved blocking issues: "
                + ", ".join(s.open_blocking_issue_ids),
            )
        if not s.has_passed(Phase.P5_REVIEW):
            return StageOutcome(
                StageStatus.FAIL,
                "cannot publish before P5 reviewed the accepted candidate",
            )
        expected_candidate_id = s.candidate_id
        if not expected_candidate_id:
            return StageOutcome(StageStatus.FAIL, "cannot publish without a reviewed candidate_id")
        manifest_path = Path(s.candidate_manifest_path or p["candidate_manifest"])
        try:
            from sdk.runtime.release_candidate import load_manifest

            manifest = load_manifest(manifest_path)
        except Exception as exc:  # noqa: BLE001 - malformed identity is terminal
            return StageOutcome(
                StageStatus.FAIL,
                f"candidate manifest unreadable before publication: {type(exc).__name__}: {exc}",
            )
        if manifest.status != "accepted" or manifest.candidate_id != expected_candidate_id:
            return StageOutcome(
                StageStatus.FAIL,
                "candidate manifest does not match the candidate reviewed in P5",
            )
        if (
            not s.reviewed_physics_report_sha256
            or manifest.physics_report_sha256 != s.reviewed_physics_report_sha256
        ):
            return StageOutcome(
                StageStatus.FAIL,
                "candidate physics report differs from the report reviewed in P5",
            )
        try:
            rep = bundle.backend.publish_release_candidate(
                manifest_path,
                p["exports"],
                expected_candidate_id=expected_candidate_id,
            )
        except Exception as exc:  # noqa: BLE001 - integrity/export failure is terminal
            return StageOutcome(
                StageStatus.FAIL,
                f"candidate publication failed: {type(exc).__name__}: {exc}",
            )
        exports = rep.get("exports", {}) if isinstance(rep, dict) else {}
        skipped = rep.get("skipped", []) if isinstance(rep, dict) else []
        if not isinstance(rep, dict) or rep.get("candidate_id") != expected_candidate_id:
            return StageOutcome(
                StageStatus.FAIL,
                "candidate publication returned an ID different from the candidate reviewed in P5",
            )
        glb, urdf = exports.get("glb"), exports.get("urdf")
        ok = bool(glb) and Path(glb).is_file() and bool(urdf) and Path(urdf).is_file()
        detail = f"candidate={rep.get('candidate_id')} exports={list(exports)}"
        if skipped:
            detail += f" skipped={skipped}"
        manipulation: dict[str, object] = {}
        manipulation_path = exports.get("manipulation")
        if manipulation_path:
            try:
                loaded = json.loads(Path(manipulation_path).read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    manipulation = loaded
                    detail += f" manip={len(loaded.get('operations') or [])}"
            except (OSError, json.JSONDecodeError):
                pass
        sizes: dict[str, int | None] = {}
        for fmt, path in exports.items():
            try:
                sizes[fmt] = Path(path).stat().st_size if path else None
            except OSError:
                sizes[fmt] = None
        try:
            p["reports"].mkdir(parents=True, exist_ok=True)
            (p["reports"] / "P8_export_smoke.json").write_text(
                json.dumps(
                    {
                        "phase": "P8_export_smoke",
                        "ok": ok,
                        "exports": exports,
                        "sizes": sizes,
                        "skipped": skipped,
                        "manipulation": manipulation,
                        "candidate_id": rep.get("candidate_id"),
                        "publish_only": True,
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass
        return StageOutcome(StageStatus.PASS if ok else StageStatus.FAIL, detail)

    async def p9_final(_s, _c) -> StageOutcome:
        from pipeline.ingest import ingest_run

        workspace_record = {
            "record_id": s.record_id,
            "run_id": s.run_id,
            "prompt": s.prompt,
            "kind": "articulated_object",
            "artifacts": {
                "asset_build": s.asset_build_path,
                "blueprint": s.blueprint_path,
                "design_board": s.design_board_path,
                "candidate_manifest": s.candidate_manifest_path,
                "physics_validation": s.physics_report_path,
                "export_manifest": str(p["exports"] / "export_manifest.json"),
            },
            "stages_completed": s.phases_passed,
            "cost_usd": s.cost_usd,
        }
        (p["root"] / "record.json").write_text(
            json.dumps(workspace_record, indent=2), encoding="utf-8"
        )
        try:
            result = ingest_run(Path(s.repo_root), s.run_id)
        except Exception as exc:  # noqa: BLE001 — ingest failure shouldn't undo a green run
            logger.exception("pipeline ingest failed for %s: %s", s.run_id, exc)
            return StageOutcome(StageStatus.PASS, f"record written (ingest skipped: {exc})")
        logger.info(
            "pipeline %s ingested as %s (category=%s)",
            s.run_id,
            result.get("record_id"),
            result.get("category_slug"),
        )
        return StageOutcome(
            StageStatus.PASS,
            f"record written; ingested as {result.get('record_id')}",
        )

    raw = {
        Phase.P0_INTAKE: p0_intake,
        Phase.P1_BLUEPRINT: p1_blueprint,
        Phase.P2_DESIGN_BOARD: p2_design_board,
        Phase.P3_BLUEPRINT_REVIEW: p3_blueprint_review,
        Phase.P4_BUILD: p4_build,
        Phase.P5_REVIEW: p5_review,
        Phase.P8_EXPORT: p8_export,
        Phase.P9_FINAL: p9_final,
    }

    def _wrap(ph, fn):
        async def wrapped(st, ctx):
            from sdk.runtime.release_candidate import CandidateError

            if bundle.recorder is not None:
                bundle.recorder.set_phase(ph.value)
            try:
                return await fn(st, ctx)
            except CandidateError as exc:
                if ph is Phase.P4_BUILD:
                    harness = getattr(bundle.coding, "harness", None)
                    if harness is not None:
                        bundle.coding_cost_usd = float(harness.cost_usd)
                return StageOutcome(StageStatus.FAIL, f"candidate integrity failure: {exc}")
            finally:
                bundle.refresh_cost()

        return wrapped

    return {ph: _wrap(ph, fn) for ph, fn in raw.items()}
