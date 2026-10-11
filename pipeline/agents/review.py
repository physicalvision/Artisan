from __future__ import annotations

import json
import logging
from pathlib import Path

from pipeline.agents.prompts.review import load_review_prompt
from pipeline.context_policy import boundary_compact
from pipeline.contracts import DeveloperHandoff, ReviewResult
from pipeline.harness_review import SdkReviewHarness

logger = logging.getLogger(__name__)


def parse_allowances(src: str) -> list[str]:
    from sdk.runtime.runtime import extract_declared_allowances

    out: list[str] = []
    for al in extract_declared_allowances(src):
        tail = f" — {al['reason']}" if al.get("reason") else ""
        if al["kind"] == "overlap":
            out.append(f"{al['a']} ↔ {al['b']} (intended overlap){tail}")
        else:
            out.append(f"{al['part']} (intended freestanding part){tail}")
    return out


def _acceptance_block(blueprint: dict) -> str:
    acceptance = blueprint.get("acceptance") or []
    if not acceptance:
        return ""
    body = "\n".join(f"- {a}" for a in acceptance)
    return (
        "ACCEPTANCE criteria the blueprint commits to (the STABLE goalposts — "
        "grade against EACH, do not invent unrelated bounds):\n" + body + "\n\n"
    )


def _allowances_block(allowances: list[str]) -> str:
    if not allowances:
        return ""
    body = "\n".join(f"- {a}" for a in allowances)
    return "DECLARED allowances (intentional contacts at the stated connection):\n" + body + "\n\n"


def _mimic_suffix(mimic: object) -> str:
    if not mimic:
        return ""
    if isinstance(mimic, dict):
        driver = mimic.get("joint")
        mult = mimic.get("multiplier")
        suffix = f" driven-by:{driver}"
        if isinstance(mult, (int, float)):
            suffix += f" x{mult:.4g}"
        off = mimic.get("offset")
        if isinstance(off, (int, float)) and abs(off) > 1e-12:
            suffix += f"+{off:.4g}"
        return suffix
    return f" driven-by:{mimic}"


def _stability_advisory_keys(
    handoff: DeveloperHandoff | None,
) -> set[tuple[str, str]]:
    if handoff is None:
        return set()
    keys: set[tuple[str, str]] = set()
    for finding in handoff.physics_validation.get("advisory_findings", []) or []:
        if not isinstance(finding, dict):
            continue
        if finding.get("kind") != "free_root_stability":
            continue
        pose = str(finding.get("pose") or "").strip()
        code = str(finding.get("code") or "").strip()
        if pose and code:
            keys.add((pose, code))
    return keys


def _dynamics_inconclusive_keys(
    handoff: DeveloperHandoff | None,
) -> set[tuple[str, str]]:
    if handoff is None:
        return set()
    findings = handoff.physics_validation.get("joint_dynamics", {}).get("inconclusive_findings", [])
    return {
        (str(item.get("joint")), str(item.get("finding_code")))
        for item in findings
        if isinstance(item, dict) and item.get("joint") and item.get("finding_code")
    }


def _handoff_block(h: DeveloperHandoff | None) -> str:
    if h is None:
        return ""
    lines = ["DEVELOPER HANDOFF (verify against the evidence — they can be wrong):"]
    if h.compile_summary:
        lines.append(f"- compile: {h.compile_summary}")
    if h.candidate_id:
        lines.append(f"- accepted P4 candidate: {h.candidate_id}")
    if h.physics_validation:
        physics = h.physics_validation
        fixed = physics.get("fixed_root") or {}
        free = physics.get("free_root") or {}
        closed = free.get("closed") or {}
        lines.append(
            "- P4 physics validation: "
            f"status={physics.get('status')} fixed_root={fixed.get('status')} "
            f"max_penetration={fixed.get('max_penetration_depth_m')}m "
            f"free_root={free.get('status')} support_mode={free.get('support_mode')} "
            f"closed_tilt={closed.get('tilt_degrees')}deg "
            f"support_margin={closed.get('support_margin_m')}m"
        )
        pose_results = [("closed", closed)] + [
            (f"representative_open:{name}", result)
            for name, result in (free.get("representative_open") or {}).items()
            if isinstance(result, dict)
        ]
        for pose_name, result in pose_results:
            if not isinstance(result, dict):
                continue
            lines.append(
                f"  pose={pose_name} stability_status={result.get('stability_status')} "
                f"tilt_degrees={result.get('tilt_degrees')} "
                f"xy_drift_m={result.get('xy_drift_m')} "
                f"z_displacement_m={result.get('z_displacement_m')} "
                f"final_linear_speed_m_s={result.get('final_linear_speed_m_s')} "
                f"final_angular_speed_rad_s={result.get('final_angular_speed_rad_s')} "
                f"support_margin_m={result.get('support_margin_m')} "
                "max_recent_ground_penetration_m="
                f"{result.get('max_recent_ground_penetration_m')} "
                f"ground_penetration_threshold_m={result.get('ground_penetration_threshold_m')} "
                f"max_abs_qvel={result.get('max_abs_qvel')} "
                f"max_abs_qacc={result.get('max_abs_qacc')}"
            )
        advisories = physics.get("advisory_findings") or []
        stability_advisories = [
            item
            for item in advisories
            if isinstance(item, dict) and item.get("kind") == "free_root_stability"
        ]
        dynamics_advisories = [
            item
            for item in advisories
            if isinstance(item, dict) and item.get("kind") == "joint_dynamics"
        ]
        if stability_advisories:
            lines.append(
                "  stability advisories (every pose/code pair requires one "
                "stability_adjudications entry):"
            )
            lines.extend(
                f"    - {json.dumps(item, sort_keys=True, default=str)}"
                for item in stability_advisories
            )
        if dynamics_advisories:
            lines.append(
                "  joint dynamics advisories (review against intended use; accept reasonable "
                "defaults or authored MotionFeel, and raise a normal Coding issue only when "
                "the observed behavior is a real product defect):"
            )
            lines.extend(
                f"    - {json.dumps(item, sort_keys=True, default=str)}"
                for item in dynamics_advisories
            )
        dynamics_inconclusive = physics.get("joint_dynamics", {}).get("inconclusive_findings", [])
        if dynamics_inconclusive:
            lines.append(
                "  dynamics inconclusive findings (every joint/finding_code pair requires "
                "one dynamics_adjudications entry):"
            )
            lines.extend(
                f"    - {json.dumps(item, sort_keys=True, default=str)}"
                for item in dynamics_inconclusive
            )
        lines.append(
            "  artifact hashes: "
            f"object_mjcf_sha256={physics.get('object_mjcf_sha256')} "
            f"scene_mjcf_sha256={physics.get('scene_mjcf_sha256')}"
        )
    if h.parts:
        lines.append(f"- parts ({len(h.parts)}): {', '.join(h.parts)}")
    if h.joints:
        jdesc = ", ".join(
            f"{j.get('name')}({j.get('type')})" + _mimic_suffix(j.get("mimic")) for j in h.joints
        )
        lines.append(f"- joints ({len(h.joints)}): {jdesc}")
        coupled = [j for j in h.joints if j.get("mimic")]
        if coupled:
            lines.append(
                "  (coupled joints follow their driver — verify each declared coupling with "
                "ctx.expect_coupled_motion(follower, driver, sense=...))"
            )
    if h.repaired_issue_ids:
        lines.append(f"- issues addressed this round: {', '.join(h.repaired_issue_ids)}")
    note = h.developer_note or {}
    if note.get("changed"):
        lines.append(f"- developer says changed: {note['changed']}")
    intentional = note.get("intentional", []) or []
    if intentional:
        lines.append("- developer says intentional-by-design (verify; they can be wrong):")
        lines.extend(f"  - {it}" for it in intentional)
    pushback = note.get("pushback", []) or []
    if pushback:
        lines.append(
            "- developer pushback (a suspected false issue, OR an unsatisfiable / "
            "contradictory blueprint requirement — verify each; a CONFIRMED, "
            "specifically-named conflict is a documented limitation, not a redo):"
        )
        lines.extend(f"  - {pb}" for pb in pushback)
    if h.motion_contact_candidates:
        lines.append("")
        lines.append(
            "CODE-SCREENED MOTION CONTACTS (evidence, NOT a verdict). Apply the system "
            "motion-contact and access-gate policy to each named pair and pose; the "
            "gated-precondition rows include the P4 v2 fixed-witness evidence."
        )
        for sc in h.motion_contact_candidates[:6]:
            pair = sc.get("pair", ["?", "?"])
            elems = sc.get("elements", ["?", "?"])
            cls = sc.get("classification")
            tag = {
                "DEFINITE": "non-declared",
                "SUSPICIOUS": "declared-pair",
                "CONDITIONAL": "gated-precondition",
                "COORDINATION": "coordination-only",
            }.get(cls, "declared-pair")
            vol = sc.get("overlap_volume")
            vol_s = f", overlap_volume≈{vol}" if vol is not None else ""
            clear_pose = sc.get("clear_pose")
            if cls == "CONDITIONAL" and clear_pose:
                evaluation = sc.get("gate_evaluation") or {}
                required = {
                    "pose",
                    "policy_id",
                    "penetration_depth_tol_m",
                    "collision_mode",
                    "target_joint",
                    "selected_combo_index",
                }
                if required.issubset(evaluation):
                    pose_s = (
                        "P4 motion engine v2 already verified the selected fixed witness "
                        f"(target={evaluation['target_joint']!r}, "
                        f"combo={evaluation['selected_combo_index']!r}, "
                        f"policy={evaluation['policy_id']!r}, "
                        f"mode={evaluation['collision_mode']!r}); render the gate-OPEN "
                        f"WITNESS pose={clear_pose} only to judge whether it is a sensible "
                        "operating configuration"
                    )
                else:
                    pose_s = (
                        "STALE gate witness has no complete v2 fixed-witness provenance; "
                        "do not accept the CONDITIONAL label. "
                        "Require regenerated motion evidence before judging this contact"
                    )
            elif cls == "COORDINATION":
                drivers = sc.get("coordination_joints") or []
                pose_s = (
                    f"render pose={sc.get('worst_pose', {})}; this contact appeared only "
                    f"with independent drivers {drivers} active and had no single-driver "
                    "witness, so judge whether that simultaneous operation is credible"
                )
            else:
                pose_s = f"render pose={sc.get('worst_pose', {})}"
            lines.append(
                f"  - [{tag}] {pair[0]}↔{pair[1]} (elements {elems[0]} | {elems[1]}{vol_s}) "
                f"— {pose_s} (use segment=true)"
            )
    if h.geometry_island_warnings:
        lines.append("")
        lines.append(
            "P4-BLOCKING DISCONNECTED SUB-MESHES. A normal accepted candidate should have none; "
            "if stale handoff evidence names one, inspect it as a semantic/visual backstop."
        )
        for gi in h.geometry_island_warnings[:6]:
            part = gi.get("part", "?")
            disc = ", ".join(gi.get("disconnected", [])[:6])
            lines.append(
                f"  - {part}: {gi.get('connected')}/{gi.get('total')} sub-meshes connected; "
                f"disconnected=[{disc}]"
            )
    if h.motion_support_findings:
        lines.append("")
        lines.append(
            "CODE-SCREENED OPEN-POSE FLOAT (evidence, NOT a verdict). Apply the system "
            "support/connector policy to each named endpoint pose."
        )
        for sf in h.motion_support_findings[:6]:
            parts = ", ".join(sf.get("parts", []) or [])
            pose = sf.get("pose", {})
            gap = sf.get("approx_gap")
            gap_s = f", approx_gap={gap}m" if isinstance(gap, (int, float)) else ""
            lines.append(
                f"  - [{parts}] floats at pose={pose} "
                f"(nearest grounded={sf.get('nearest_grounded_part')}{gap_s})"
            )
    if h.driver_reach_findings:
        lines.append("")
        lines.append(
            "DRIVER-REACH FINDINGS (evidence, NOT a verdict). Apply the system "
            "driver-reach policy to each named joint, blocked pose, pair, and reachable fraction."
        )
        for rf in h.driver_reach_findings[:6]:
            nom = rf.get("nominal", ["?", "?"])
            rch = rf.get("reachable", ["?", "?"])
            frac = rf.get("reachable_fraction")
            by = rf.get("blocked_by") or ["?", "?"]
            elems = rf.get("elements") or ["?", "?"]
            pct = f"{frac:.0%}" if isinstance(frac, (int, float)) else "?"
            lines.append(
                f"  - {rf.get('joint')}: nominal={nom} reachable={rch} ({pct}) "
                f"blocked_at={rf.get('blocked_at')} by {by[0]}↔{by[1]} "
                f"(elements {elems[0]} | {elems[1]})"
            )
    if h.declared_gate_witnesses:
        lines.append("")
        lines.append(
            "DECLARED GATE WITNESSES (deterministic P4 evidence). Each entry uses one "
            "fixed gate pose over the target joint's complete sampled travel."
        )
        for witness in h.declared_gate_witnesses[:6]:
            lines.append(
                f"  - {witness.get('joint')}: clear={witness.get('clear')} "
                f"gate_pose={witness.get('gate_pose')} target_range="
                f"{witness.get('target_range')} samples={witness.get('sample_count')}"
            )
    if h.motion_identity:
        lines.append(
            "- motion evidence identity: "
            f"{h.motion_identity.get('motion_evidence_key')} "
            f"requested={h.motion_identity.get('requested_collision_mode')} "
            f"actual={h.motion_identity.get('actual_collision_mode')}"
        )
    if h.contact_exclusion_evidence:
        lines.append("")
        lines.append(
            "CONTACT-EXCLUSION EVIDENCE (evidence, NOT a verdict). Apply the system "
            "contact-exclusion policy to each rest-clear-to-motion-contact pair."
        )
        for ce in h.contact_exclusion_evidence[:6]:
            pr = ce.get("pair") or ["?", "?"]
            lines.append(
                f"  - {pr[0]}↔{pr[1]}: rest_clearance={ce.get('rest_distance')} "
                f"motion_min={ce.get('motion_min_distance')} at pose={ce.get('pose')}"
            )
    if h.full_collision_penetrations:
        lines.append("")
        lines.append(
            "FULL-COLLISION PENETRATIONS (evidence, NOT a verdict). Apply the system "
            "penetration policy to each named pair, depth, axis, pose, and blocking status."
        )
        for pf in h.full_collision_penetrations[:6]:
            pr = pf.get("pair") or ["?", "?"]
            els = pf.get("elements") or ["?", "?"]
            depth = pf.get("penetration_depth")
            mm = f"{round(float(depth) * 1000, 2)}mm" if isinstance(depth, (int, float)) else "?"
            lines.append(
                f"  - {pr[0]}↔{pr[1]} (elements {els[0]} | {els[1]}): depth≈{mm} "
                f"along {pf.get('axis')} joint={pf.get('joint')} constant={pf.get('constant')} "
                f"declared={pf.get('declared')} blocking={pf.get('blocking')} "
                f"at pose={pf.get('worst_pose')}"
            )
    if h.connection_cleanliness_findings:
        lines.append("")
        lines.append(
            "CONNECTION-CLEANLINESS FINDINGS (evidence, NOT a verdict). Apply the system "
            "connection-cleanliness policy to each named pair and worst pose."
        )
        for cf in h.connection_cleanliness_findings[:6]:
            pr = cf.get("pair") or ["?", "?"]
            els = cf.get("elements") or ["?", "?"]
            lines.append(
                f"  - {pr[0]}↔{pr[1]} (elements {els[0]} | {els[1]}): "
                f"penetration_volume={cf.get('penetration_volume')}m³ "
                f"depth={cf.get('penetration_span')}m blocking={cf.get('blocking')} "
                f"at pose={cf.get('worst_pose')}"
            )
    if h.coupling_rollout:
        lines.append("")
        lines.append(
            "COUPLING ROLLOUT SMOKE (evidence, NOT a verdict). Apply the system coupling "
            "rollout policy to each status, check, track, blocking pair, and force value."
        )
        for cr in h.coupling_rollout[:6]:
            cp = cr.get("coupling", {})
            extra = ""
            if cr.get("follower_track") is not None:
                extra += f" follower_track={cr['follower_track']}"
            if cr.get("blocking_pair"):
                extra += f" blocking_pair={cr['blocking_pair']}"
            if cr.get("peak_force_n") is not None:
                extra += f" peak_force={cr['peak_force_n']}N"
            lines.append(
                f"  - [{cr.get('status')}] {cp.get('follower')}<-{cp.get('driver')} "
                f"({cp.get('kind')}, {cr.get('check')}){extra}"
            )
    if h.articulation_rollout:
        lines.append("")
        lines.append(
            "LEGACY HIGH-KP ARTICULATION DIAGNOSTIC (not acceptance authority). Never use a pass "
            "here to override P4 candidate penetration or free-root failure."
        )
        for ar in h.articulation_rollout[:6]:
            extra = ""
            if ar.get("track") is not None:
                extra += f" track={ar['track']}"
            if ar.get("blocking_pair"):
                extra += f" blocking_pair={ar['blocking_pair']}"
            if ar.get("peak_force_n") is not None:
                extra += f" peak_force={ar['peak_force_n']}N"
            lines.append(f"  - [{ar.get('status')}] joint {ar.get('joint')}{extra}")
    if h.joint_dynamics_evidence:
        dynamics = h.joint_dynamics_evidence
        lines.append("")
        lines.append(
            "EXACT CONTACTS-OFF PASSIVE JOINT-DYNAMICS DIAGNOSTIC from the servo-free "
            "candidate MJCF. Breakaway, hold, and decay isolate intrinsic joint response: "
            "contact friction, guide forces, and mechanical retainers are absent, so every "
            "numeric pass/fail and every optional Blueprint dynamics_intent mismatch is Review "
            "evidence, not a P4 verdict. Judge it against the requested use and raise a normal "
            "Coding issue only for a real product defect. NEVER request hand-written damping/"
            "friction or a geometry change merely to tune feel."
        )
        lines.append(
            f"  - overall={dynamics.get('status')} algorithm={dynamics.get('algorithm')} "
            f"enforcement={dynamics.get('enforcement')} raw_status={dynamics.get('raw_status')} "
            f"mjcf_sha256={dynamics.get('mjcf_sha256')}"
        )
        for joint in dynamics.get("joints", [])[:8]:
            lines.append(
                f"  - [{joint.get('status')}] joint {joint.get('joint')}: "
                f"breakaway={joint.get('breakaway', {}).get('status')} "
                f"hold={joint.get('hold', {}).get('status')} "
                f"decay={joint.get('decay', {}).get('status')}"
            )
    return "\n".join(lines) + "\n\n"


def _reference_note(n_references: int, has_board: bool) -> str:
    if n_references:
        refs = "\n".join(
            f"- Reference {i + 1} (AUTHORITATIVE for the view/state it shows)"
            for i in range(n_references)
        )
        head = (
            f"Attached: {n_references} USER REFERENCE photo(s) (AUTHORITATIVE for "
            "appearance/identity). They may be different VIEWS and/or open/closed "
            "STATES of the SAME object — judge each render against the reference that "
            f"shows the same view/state:\n{refs}\n"
        )
        head += (
            "Plus a design board (a generated aid).\n\n"
            if has_board
            else "No design board this run.\n\n"
        )
        return head
    if has_board:
        return (
            "Attached: a design board (a generated visual aid, NOT authoritative); no "
            "user reference photo. Judge primarily against the blueprint spec.\n\n"
        )
    return "No reference imagery is attached; judge against the blueprint SPEC alone.\n\n"


def _user_request_block(user_prompt: str) -> str:
    text = (user_prompt or "").strip()
    if not text:
        return ""
    return (
        "ORIGINAL USER REQUEST (the ultimate intent — what the user actually asked "
        "for; the BLUEPRINT below is Design's engineering TRANSLATION of it and may "
        "have mis-read it). Judge primarily against the blueprint + ACCEPTANCE (the "
        "stable, measurable goalposts). Use this original request as the HIGHER "
        "authority ONLY to catch a STRUCTURAL mismatch the blueprint missed or "
        "mis-translated — an explicitly requested part / mechanism / behaviour that "
        "is ABSENT, fundamentally WRONG, or whose topology does not match what the "
        "words describe (raise it `blocking` even if the blueprint itself 'passes'). "
        "Do NOT mint new aesthetic or dimensional bounds from it, and do NOT "
        "re-litigate choices the blueprint legitimately made.\n"
        f'"""\n{text}\n"""\n\n'
    )


def _build_task_text(
    phase: str,
    blueprint: dict,
    handoff: DeveloperHandoff | None,
    allowances: list[str],
    n_references: int,
    has_board: bool,
    user_prompt: str = "",
) -> str:
    reference_note = _reference_note(n_references, has_board)
    return (
        load_review_prompt("review.user.md")
        .replace("{user_request}", _user_request_block(user_prompt))
        .replace("{reference_note}", reference_note)
        .replace("{handoff}", _handoff_block(handoff))
        .replace("{acceptance}", _acceptance_block(blueprint))
        .replace("{allowances}", _allowances_block(allowances))
        .replace("{blueprint}", json.dumps(blueprint, indent=2))
    )


def _build_repair_task_text(
    handoff: DeveloperHandoff | None,
    allowances: list[str],
) -> str:
    blocks = [
        "REPAIR ROUND — the developer reports the prior round's issues addressed "
        "(handoff below). The ORIGINAL USER REQUEST, blueprint, ACCEPTANCE criteria, "
        "rubric, checklist, and reference imagery from the first review message above "
        "STILL BIND (they are not re-pasted; prior-round renders were dropped to save "
        "context).\n"
        "This round is VERIFY-FIRST, not a from-scratch review. Deterministic evidence "
        "gathered on the CURRENT build is attached below when available: your "
        "accumulated qa_tests re-run and a fresh standard render set. Work from it:\n"
        "1. Check each issue you raised last round against the evidence — did the fix "
        "actually land?\n"
        "2. Check the handoff's changed/intentional claims against what you can see; "
        "changes BEYOND the issues' scope deserve the same scrutiny as the fixes "
        "(a repair can introduce a new fault elsewhere).\n"
        "3. Gather MORE evidence (tests/probes/renders/crops) only where the attached "
        "set cannot answer — a failing qa test, a suspicious region, a part the "
        "standard views don't show (interior → render open + focus). You judge how "
        "deep to dig: a clean fix needs no extra evidence; anything surprising does.\n"
        "4. Re-grade EVERY rubric dimension against the SAME acceptance, raise only "
        "the issues that REMAIN or were NEWLY introduced, then submit_review as "
        "before.\n\n",
    ]
    handoff_block = _handoff_block(handoff)
    if handoff_block:
        blocks.append(handoff_block)
    allowances_block = _allowances_block(allowances)
    if allowances_block:
        blocks.append(allowances_block)
    return "".join(blocks)


def _pair_dangling_tool_calls(conv: list[dict]) -> list[dict]:
    have_output = {
        str(m.get("tool_call_id"))
        for m in conv
        if isinstance(m, dict) and m.get("role") == "tool" and m.get("tool_call_id")
    }
    missing: list[dict] = []
    for m in conv:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            cid = tc.get("id") or tc.get("tool_call_id")
            if not cid or str(cid) in have_output:
                continue
            have_output.add(str(cid))
            name = (tc.get("function") or {}).get("name") or tc.get("name") or "tool"
            missing.append(
                {
                    "role": "tool",
                    "tool_call_id": cid,
                    "name": name,
                    "content": f"OK: {name} acknowledged.",
                }
            )
    if missing:
        logging.getLogger(__name__).warning(
            "review session had %d dangling tool call(s); paired before persist "
            "(a terminal tool likely skipped appending its output)",
            len(missing),
        )
        return list(conv) + missing
    return conv


class ReviewAgent:
    def __init__(
        self,
        model_path: Path,
        *,
        backend: object,
        sdk_package: str = "sdk",
        session_path: Path | None = None,
        max_cost_usd: float | None = None,
        trace_dir: Path | None = None,
        render_root: Path | None = None,
        qa_test_path: Path | None = None,
        cassette: object | None = None,
        recorder: object | None = None,
    ) -> None:
        from pipeline.providers import pipeline_model, pipeline_provider, pipeline_thinking

        self.session_path = session_path
        self.harness = SdkReviewHarness(
            model_path,
            backend=backend,
            sdk_package=sdk_package,
            provider=pipeline_provider("review"),
            model_id=pipeline_model("review"),
            thinking_level=pipeline_thinking("review"),
            max_cost_usd=max_cost_usd,
            trace_dir=trace_dir,
            cassette=cassette,
            render_root=render_root,
            qa_test_path=qa_test_path,
            recorder=recorder,
        )

    @property
    def cost_usd(self) -> float:
        return self.harness.cost_usd

    def _load_conv(self) -> list[dict]:
        if not (self.session_path and self.session_path.exists()):
            return []
        conv = json.loads(self.session_path.read_text(encoding="utf-8"))
        return boundary_compact(self.harness, conv)

    def _save_conv(self, conv: list[dict]) -> None:
        if not self.session_path:
            return
        conv = _pair_dangling_tool_calls(conv)
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_path.write_text(json.dumps(conv, indent=2), encoding="utf-8")

    async def review(
        self,
        phase: str,
        *,
        blueprint: dict,
        handoff: DeveloperHandoff | None = None,
        allowances: list[str] | None = None,
        reference_images: list[Path] | None = None,
        design_board: Path | None = None,
        user_prompt: str = "",
        max_cost_usd: float | None = None,
    ) -> ReviewResult:
        if max_cost_usd is not None:
            self.harness.max_cost_usd = max_cost_usd
        refs = [Path(p) for p in (reference_images or []) if Path(p).exists()]
        prior_conv = self._load_conv()
        first_round = not prior_conv
        if first_round:
            task_text = _build_task_text(
                phase,
                blueprint,
                handoff,
                allowances or [],
                n_references=len(refs),
                has_board=bool(design_board and Path(design_board).exists()),
                user_prompt=user_prompt,
            )
            images: list[Path] = list(refs)
            if design_board and Path(design_board).exists():
                images.append(Path(design_board))
        else:
            task_text = _build_repair_task_text(handoff, allowances or [])
            images = []
        report, conv = await self.harness.run(
            phase,
            task_text,
            images=images,
            initial_conversation=prior_conv,
            attach_evidence=not first_round,
            expected_stability_advisories=_stability_advisory_keys(handoff),
            expected_dynamics_inconclusive=_dynamics_inconclusive_keys(handoff),
        )
        self._save_conv(conv)
        return report
