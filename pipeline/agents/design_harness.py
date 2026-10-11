from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline.agents.prompts.design import load_design_prompt
from pipeline.cost import build_cost_tracker
from pipeline.gates import (
    access_gate_check,
    blueprint_decision_provenance_check,
    blueprint_presence_check,
    joint_coupling_check,
    joint_reference_frame_check,
    manipulation_tasks_check,
)
from pipeline.harness_codec import MessageCodec
from pipeline.llm.factory import (
    ProviderConfig,
    create_provider_client,
)
from pipeline.policies import (
    DEFAULT_POLICIES,
    JOINT_PROP_AXIS,
    JOINT_PROP_LIMIT_LOWER,
    JOINT_PROP_LIMIT_UPPER,
    JOINT_PROP_TYPE,
    Policies,
)
from pipeline.prompt_cache import build_openai_prompt_cache_settings
from pipeline.providers import resolve_pipeline_model
from pipeline.timing import _null_measure

logger = logging.getLogger(__name__)

ALLOWED_JOINT_TYPES = frozenset({"revolute", "continuous", "prismatic", "fixed"})

MOVING_JOINT_TYPES = ALLOWED_JOINT_TYPES - frozenset({"fixed"})
_SESSION_STAGES = frozenset(
    {"intake", "intake_done", "blueprint_authoring", "self_review", "done", "incomplete"}
)


@dataclass(slots=True)
class DesignResult:
    success: bool
    reason: str
    blueprint: dict[str, Any] | None
    conversation: list[dict[str, Any]] = field(default_factory=list)
    turns: int = 0
    cost_usd: float = 0.0
    asset_brief: dict[str, Any] | None = None


def _design_tool_schemas() -> list[dict[str, Any]]:
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
    return [
        fn(
            "submit_intake",
            "Submit the completed intake JSON. It is validated and stored as "
            "the blueprint asset_brief before engineering authoring begins.",
            {"content": s},
            ["content"],
        ),
        fn(
            "write_blueprint",
            "Submit the complete engineering blueprint JSON (the filled template). It is "
            "validated immediately; on failure you get the errors and must fix "
            "and resubmit. A valid submission proceeds to self-review.",
            {"content": s},
            ["content"],
        ),
        fn(
            "submit_blueprint_selfcheck",
            "Submit the agent-led review of the current blueprint. PASS releases "
            "P1; REDO requires a complete revised blueprint in this conversation.",
            {
                "decision": {"type": "string", "enum": ["PASS", "REDO"]},
                "summary": s,
                "checks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "area": s,
                            "verdict": {
                                "type": "string",
                                "enum": ["PASS", "REDO"],
                            },
                            "rationale": s,
                        },
                        "required": ["area", "verdict", "rationale"],
                    },
                },
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "problem": s,
                            "evidence": s,
                            "required_revision": s,
                        },
                        "required": ["problem", "evidence", "required_revision"],
                        "additionalProperties": False,
                    },
                },
                "blueprint_sha256": s,
            },
            ["decision", "summary", "checks", "findings", "blueprint_sha256"],
        ),
    ]


def _design_prompt_cache_settings(
    provider: str, model_id: str | None, system_prompt: str
) -> tuple[str | None, str | None, str | None]:
    if provider != "openai":
        return None, None, None
    from pipeline.llm.openai import DEFAULT_OPENAI_MODEL

    return build_openai_prompt_cache_settings(
        model_id=model_id or DEFAULT_OPENAI_MODEL,
        sdk_package="sdk",
        system_prompt=system_prompt,
        sdk_docs_context="",
        tools=_design_tool_schemas(),
    )


def derive_limits(joint_type: Any, travel: Any, rest: Any) -> tuple[float | None, float | None]:
    jt = str(joint_type or "").lower()
    if jt in {"continuous", "fixed", "detachable"}:
        return (None, None)
    try:
        t = float(travel)
    except (TypeError, ValueError):
        return (None, None)
    if not (t > 0):
        return (None, None)
    r = str(rest or "at_min").lower()
    if r == "centered":
        return (-t / 2.0, t / 2.0)
    if r == "at_max":
        return (-t, 0.0)
    return (0.0, t)


def _normalize_joints(joints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for j in joints:
        lo, hi = derive_limits(j.get("type"), j.get("travel"), j.get("rest"))
        out.append(
            {
                "name": j.get("name"),
                JOINT_PROP_TYPE: j.get("type"),
                JOINT_PROP_AXIS: j.get("axis"),
                JOINT_PROP_LIMIT_LOWER: lo,
                JOINT_PROP_LIMIT_UPPER: hi,
            }
        )
    return out


class DesignBlueprintHarness:

    ROLE = "DesignAgent"

    @property
    def cost_usd(self) -> float:
        return self.cost.all_in_total_cost_usd if self.cost else 0.0

    def __init__(
        self,
        blueprint_path: Path,
        *,
        session_path: Path | None = None,
        selfcheck_report_path: Path | None = None,
        provider: str = "openai",
        model_id: str | None = None,
        thinking_level: str = "high",
        max_turns: int = 12,
        max_cost_usd: float | None = 10.0,
        trace_path: Path | None = None,
        policies: Policies = DEFAULT_POLICIES,
        cassette: object | None = None,
        recorder: object | None = None,
    ) -> None:
        self.blueprint_path = blueprint_path
        self.session_path = session_path
        self.selfcheck_report_path = selfcheck_report_path
        self.policies = policies
        self.cassette = cassette
        self.recorder = recorder.for_agent("design") if recorder is not None else None
        self.max_turns = max_turns
        self.max_cost_usd = max_cost_usd
        self.provider = provider
        self.system_prompt = load_design_prompt("system.md")
        _ck, _cret, _cfp = (None, None, None)
        if provider == "openai":
            _ck, _cret, _cfp = _design_prompt_cache_settings(
                provider,
                model_id,
                self.system_prompt,
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
        self.cost = build_cost_tracker(_cost_provider, _mid)
        self.model_id = _mid
        self._trace_path = trace_path
        self._trace_fp = None
        self._trace_system_once = True
        self._blueprint: dict[str, Any] | None = None
        self._asset_brief: dict[str, Any] | None = None
        self._last_ok = False
        self._stage = "intake"
        self._conversation: list[dict[str, Any]] = []
        self._review_rounds: list[dict[str, Any]] = []
        self._review_verdicts_this_run = 0
        self._terminal_reason: str | None = None
        self._load_session()
        unpermitted = [
            t["function"]["name"]
            for t in _design_tool_schemas()
            if not self.policies.can_use_tool(self.ROLE, t["function"]["name"])
        ]
        if unpermitted:
            raise RuntimeError(
                f"tool/policy mismatch: {self.ROLE} advertises but is denied "
                f"{unpermitted}; fix policies.ROLE_TOOLS"
            )

    def _load_session(self) -> None:
        if self.session_path is None or not self.session_path.is_file():
            return
        try:
            raw = json.loads(self.session_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid DesignAgent session: {exc}") from exc
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise RuntimeError("invalid DesignAgent session: expected schema_version 1 object")
        conversation = raw.get("conversation")
        if not isinstance(conversation, list):
            raise RuntimeError("invalid DesignAgent session: conversation must be a list")
        self._conversation = conversation
        self._stage = str(raw.get("stage") or "intake")
        if self._stage not in _SESSION_STAGES:
            raise RuntimeError(f"invalid DesignAgent session stage: {self._stage!r}")
        brief = raw.get("asset_brief")
        self._asset_brief = brief if isinstance(brief, dict) else None
        rounds = raw.get("review_rounds")
        self._review_rounds = rounds if isinstance(rounds, list) else []
        if self.blueprint_path.is_file():
            try:
                bp = json.loads(self.blueprint_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                bp = None
            self._blueprint = bp if isinstance(bp, dict) else None
        if self._stage != "intake" and self._asset_brief is None:
            raise RuntimeError("invalid DesignAgent session: completed intake has no asset_brief")
        if self._stage in {"self_review", "done", "incomplete"} and self._blueprint is None:
            raise RuntimeError(
                f"invalid DesignAgent session: stage {self._stage!r} has no valid blueprint"
            )
        if (
            self._blueprint is not None
            and self._asset_brief is not None
            and self._blueprint.get("asset_brief") != self._asset_brief
        ):
            raise RuntimeError(
                "invalid DesignAgent session: blueprint asset_brief differs from intake"
            )
        self._last_ok = self._stage in {"self_review", "done"} and self._blueprint is not None

    def _save_session(self) -> None:
        if self.session_path is None:
            return
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "stage": self._stage,
            "asset_brief": self._asset_brief,
            "review_rounds": self._review_rounds,
            "conversation": self._conversation,
        }
        temporary = self.session_path.with_name(f".{self.session_path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        temporary.replace(self.session_path)

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _trace(self, event: dict[str, Any]) -> None:
        if self._trace_path is None:
            return
        if self._trace_fp is None:
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._trace_fp = self._trace_path.open("a", encoding="utf-8")
        if event.get("type") == "design_blueprint_turn" and self._trace_system_once:
            event["system"] = self.system_prompt
            self._trace_system_once = False
        self._trace_fp.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        self._trace_fp.flush()

    def _submit_intake(self, content: str) -> str:
        if getattr(self, "_stage", "intake") != "intake":
            return "INTAKE REJECTED: intake has already been completed"
        try:
            brief = json.loads(content)
        except json.JSONDecodeError as exc:
            return f"INTAKE REJECTED: not valid JSON ({exc})"
        if not isinstance(brief, dict):
            return "INTAKE REJECTED: top-level value must be a JSON object"
        try:
            from pipeline.contracts import validate_support_mode

            validate_support_mode(brief)
        except ValueError as exc:
            return f"INTAKE REJECTED: {exc}"

        self._asset_brief = brief
        self._blueprint = {"asset_brief": brief}
        self._stage = "intake_done"
        self._terminal_reason = "intake_valid"
        self.blueprint_path.parent.mkdir(parents=True, exist_ok=True)
        self.blueprint_path.write_text(
            json.dumps(self._blueprint, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return "INTAKE OK: asset_brief validated and saved."

    def _write_blueprint(self, content: str) -> str:
        stage = getattr(self, "_stage", "blueprint_authoring")
        if stage != "blueprint_authoring":
            return f"BLUEPRINT REJECTED: write_blueprint is not available during {stage}"
        try:
            bp = json.loads(content)
        except json.JSONDecodeError as exc:
            return f"BLUEPRINT REJECTED: not valid JSON ({exc})"
        if not isinstance(bp, dict):
            return "BLUEPRINT REJECTED: top-level value must be a JSON object"

        asset_brief = getattr(self, "_asset_brief", None)
        if asset_brief is not None:
            bp["asset_brief"] = asset_brief

        presence = blueprint_presence_check(bp)
        if not presence.passed:
            return "BLUEPRINT REJECTED (structure): " + "; ".join(presence.failures)

        acceptance = bp.get("acceptance")
        if not isinstance(acceptance, list) or not acceptance:
            return (
                "BLUEPRINT REJECTED: 'acceptance' must be a non-empty list of "
                "plain-language verification points (the ReviewAgent turns each "
                "into a test)."
            )

        joints = bp.get("joints") or []
        if not joints:
            return (
                "BLUEPRINT REJECTED: an articulated object needs at least one "
                "moving joint (revolute/continuous/prismatic)."
            )
        bad_types = sorted(
            {str(j.get("type")) for j in joints if j.get("type") not in ALLOWED_JOINT_TYPES}
        )
        if bad_types:
            return (
                f"BLUEPRINT REJECTED: joint type(s) {bad_types} are not SDK-supported. "
                f"Use only {sorted(ALLOWED_JOINT_TYPES)} (decompose multi-DOF into a "
                "chain of these 1-DOF joints)."
            )
        if not any(str(j.get("type")) in MOVING_JOINT_TYPES for j in joints):
            return (
                "BLUEPRINT REJECTED: every joint is `fixed`; an articulated object "
                f"needs at least one MOVING joint ({sorted(MOVING_JOINT_TYPES)})."
            )
        jcheck = joint_reference_frame_check(_normalize_joints(joints))
        if not jcheck.passed:
            return "BLUEPRINT REJECTED (joints): " + "; ".join(jcheck.failures)

        couplings = bp.get("couplings") or []
        ccheck = joint_coupling_check(joints, couplings)
        if not ccheck.passed:
            return "BLUEPRINT REJECTED (couplings): " + "; ".join(ccheck.failures)

        agcheck = access_gate_check(bp.get("parts") or [], joints)
        if not agcheck.passed:
            return "BLUEPRINT REJECTED (access gates): " + "; ".join(agcheck.failures)

        mtcheck = manipulation_tasks_check(bp)
        provenance_check = blueprint_decision_provenance_check(bp)

        bp.setdefault("version_history", [])
        self.blueprint_path.parent.mkdir(parents=True, exist_ok=True)
        self.blueprint_path.write_text(json.dumps(bp, indent=2), encoding="utf-8")
        self._blueprint = bp
        self._last_ok = True
        self._stage = "self_review"
        self._terminal_reason = None
        coupling_note = f", {len(couplings)} coupling(s)" if couplings else ""
        n_gates = sum(
            1
            for p in (bp.get("parts") or [])
            if isinstance(p, dict)
            and isinstance(p.get("motion_gate"), dict)
            and (p["motion_gate"].get("gates") or [])
        )
        gate_note = f", {n_gates} access gate(s)" if n_gates else ""
        n_tasks = len(bp.get("tasks") or [])
        tasks_note = f", {n_tasks} task(s)" if n_tasks else ""
        manip_note = (
            ""
            if mtcheck.passed
            else " [WARNING manipulation.tasks: " + "; ".join(mtcheck.failures) + "]"
        )
        provenance_note = (
            ""
            if provenance_check.passed
            else " [WARNING decision_provenance: " + "; ".join(provenance_check.failures) + "]"
        )
        return (
            f"BLUEPRINT OK: {len(bp.get('parts', []))} part(s), {len(joints)} joint(s)"
            f"{coupling_note}{gate_note}{tasks_note} validated and saved.{manip_note}"
            f"{provenance_note} Now review this exact saved blueprint before deciding "
            "PASS or REDO."
        )

    def _submit_blueprint_selfcheck(
        self,
        *,
        decision: str,
        summary: str,
        checks: list[dict[str, Any]],
        findings: list[dict[str, Any]],
        blueprint_sha256: str,
    ) -> str:
        if self._stage != "self_review" or not self.blueprint_path.is_file():
            return "SELFCHECK REJECTED: write a valid blueprint before reviewing it"
        actual_sha256 = self._sha256(self.blueprint_path)
        if blueprint_sha256 != actual_sha256:
            return (
                "SELFCHECK REJECTED: blueprint_sha256 does not match the current saved "
                f"blueprint ({actual_sha256})"
            )
        normalized = str(decision).upper()
        if normalized not in {"PASS", "REDO"}:
            return "SELFCHECK REJECTED: decision must be PASS or REDO"
        if not isinstance(summary, str) or not summary.strip():
            return "SELFCHECK REJECTED: summary must be non-empty"
        if not isinstance(checks, list) or not checks:
            return "SELFCHECK REJECTED: checks must be a non-empty list"
        for index, check in enumerate(checks):
            if not isinstance(check, dict):
                return f"SELFCHECK REJECTED: checks[{index}] must be an object"
            if not all(str(check.get(key) or "").strip() for key in ("area", "rationale")):
                return f"SELFCHECK REJECTED: checks[{index}] requires non-empty area and rationale"
            if str(check.get("verdict") or "").upper() not in {"PASS", "REDO"}:
                return f"SELFCHECK REJECTED: checks[{index}].verdict must be PASS or REDO"
        if not isinstance(findings, list) or not all(isinstance(item, dict) for item in findings):
            return "SELFCHECK REJECTED: findings must be a list of objects"
        for index, finding in enumerate(findings):
            if not all(
                str(finding.get(key) or "").strip()
                for key in ("problem", "evidence", "required_revision")
            ):
                return (
                    f"SELFCHECK REJECTED: findings[{index}] requires non-empty "
                    "problem, evidence, and required_revision"
                )
        has_redo_check = any(str(check.get("verdict") or "").upper() == "REDO" for check in checks)
        if normalized == "PASS" and (has_redo_check or findings):
            return (
                "SELFCHECK REJECTED: PASS is inconsistent with REDO checks or unresolved findings"
            )
        if normalized == "REDO" and not findings:
            return "SELFCHECK REJECTED: REDO requires at least one concrete finding"

        review_round = len(self._review_rounds) + 1
        self._review_verdicts_this_run += 1
        record = {
            "review_round": review_round,
            "decision": normalized,
            "summary": summary.strip(),
            "checks": checks,
            "findings": findings,
            "blueprint_sha256": actual_sha256,
        }
        self._review_rounds.append(record)
        report = {
            "schema_version": 1,
            "artifact": "blueprint.json",
            "latest": record,
            "rounds": self._review_rounds,
        }
        if self.selfcheck_report_path is not None:
            try:
                self.selfcheck_report_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.selfcheck_report_path.with_name(
                    f".{self.selfcheck_report_path.name}.tmp"
                )
                temporary.write_text(
                    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                temporary.replace(self.selfcheck_report_path)
            except OSError:
                logger.warning(
                    "could not persist audit-only Blueprint selfcheck report %s",
                    self.selfcheck_report_path,
                    exc_info=True,
                )

        if normalized == "PASS":
            self._stage = "done"
            self._terminal_reason = "blueprint_valid"
            return "SELFCHECK PASS: the current blueprint is released to P2."
        self._last_ok = False
        if self._review_verdicts_this_run >= 2:
            self._stage = "incomplete"
            self._terminal_reason = "self_review_incomplete"
            return (
                "SELFCHECK REDO LIMIT: two review verdicts were used; P1 is incomplete "
                "and must resume later."
            )

        self._stage = "blueprint_authoring"
        self._last_ok = False
        self._terminal_reason = None
        return (
            "SELFCHECK REDO: revise the blueprint to resolve the findings, then submit "
            "the complete replacement with write_blueprint."
        )

    def _dispatch(self, name: str, args: dict[str, Any]) -> str:
        if not self.policies.can_use_tool(self.ROLE, name):
            return f"ERROR: tool {name!r} not permitted for {self.ROLE}"
        try:
            if name == "submit_intake":
                return self._submit_intake(args["content"])
            if name == "write_blueprint":
                return self._write_blueprint(args["content"])
            if name == "submit_blueprint_selfcheck":
                return self._submit_blueprint_selfcheck(
                    decision=args["decision"],
                    summary=args["summary"],
                    checks=args["checks"],
                    findings=args["findings"],
                    blueprint_sha256=args["blueprint_sha256"],
                )
        except KeyError as exc:
            return f"ERROR: missing arg {exc}"
        return f"ERROR: unknown tool {name!r}"

    def _append_user(self, content: Any, *, subtask: str) -> None:
        normalized = content
        self._conversation.append({"role": "user", "content": normalized})
        self._trace(
            {
                "ts": time.time(),
                "type": "design_prompt",
                "agent": "design",
                "subtask": subtask,
                "content": content,
            }
        )
        self._save_session()

    def _append_self_review_prompt(self) -> None:
        blueprint = self.blueprint_path.read_text(encoding="utf-8")
        prompt = (
            load_design_prompt("self_review.user.md")
            .replace("{blueprint}", blueprint)
            .replace("{blueprint_sha256}", self._sha256(self.blueprint_path))
            .replace("{review_round}", str(len(self._review_rounds) + 1))
        )
        self._append_user(prompt, subtask="blueprint_self_review")

    async def run_intake(self, user_message: Any) -> DesignResult:
        if self._asset_brief is not None and self._stage != "intake":
            return self._result(True, "intake_valid", 0)
        self._terminal_reason = None
        if not self._conversation:
            self._append_user(user_message, subtask="intake")
        return await self._run_loop()

    async def run_blueprint(
        self,
        user_message: Any,
        *,
        asset_brief: dict[str, Any] | None = None,
        reopen: bool = False,
    ) -> DesignResult:
        if (
            asset_brief is not None
            and self._asset_brief is not None
            and asset_brief != self._asset_brief
        ):
            raise ValueError("blueprint asset_brief differs from the persisted P0 intake")
        if asset_brief is not None:
            self._asset_brief = asset_brief
        if self._asset_brief is None:
            return self._result(False, "missing_intake", 0)
        if self._stage == "done":
            if not reopen:
                return self._result(True, "blueprint_valid", 0)
            self._stage = "blueprint_authoring"
            self._terminal_reason = None
            self._review_verdicts_this_run = 0
            self._append_user(user_message, subtask="blueprint_revision")
        if self._stage == "intake":
            self._stage = "intake_done"
        if self._stage == "intake_done":
            self._stage = "blueprint_authoring"
            self._terminal_reason = None
            self._append_user(user_message, subtask="blueprint")
        elif self._stage == "incomplete":
            self._stage = "blueprint_authoring"
            self._terminal_reason = None
            self._review_verdicts_this_run = 0
            self._append_user(
                "<blueprint_resume>Resume the existing Blueprint REDO. Resolve the latest "
                "self-review findings, then submit a complete replacement with "
                "write_blueprint.</blueprint_resume>",
                subtask="blueprint_resume",
            )
        elif self._stage == "self_review":
            last = self._conversation[-1] if self._conversation else {}
            content = str(last.get("content") or "") if isinstance(last, dict) else ""
            if "BLUEPRINT SELF-REVIEW" not in content:
                self._append_self_review_prompt()
        return await self._run_loop()

    async def run(self, user_message: Any) -> DesignResult:
        return await self.run_blueprint(user_message)

    async def _next_turn(
        self,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], Any]:
        sample = None

        key = "design_turn"
        if self.cassette is not None and getattr(self.cassette, "replaying", False):
            rec = self.cassette.replay(self.ROLE, key)
            assistant = rec.get("assistant") or {}
            tool_calls = rec.get("tool_calls") or []
            usage = rec.get("usage") or {}
        else:
            with (
                self.recorder.measure() if self.recorder is not None else _null_measure()
            ) as sample:
                response = await self.llm.generate_with_tools(
                    self.system_prompt, self._conversation, _design_tool_schemas()
                )
            tool_calls = self.codec.extract_tool_calls(response)
            usage = self.codec.extract_usage(response) or {}
            assistant = self.codec.build_assistant_message(response)
            if self.cassette is not None and getattr(self.cassette, "recording", False):
                self.cassette.record(
                    self.ROLE,
                    key,
                    {"assistant": assistant, "tool_calls": tool_calls, "usage": usage},
                )
        return assistant, tool_calls, usage, sample

    def _decode_tool_call(self, tool_call: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
        name = self.codec.tool_call_name(tool_call)
        raw = (
            tool_call.get("function", {}).get("arguments")
            or tool_call.get("custom", {}).get("input")
            or tool_call.get("input")
            or "{}"
        )
        try:
            args = raw if isinstance(raw, dict) else json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            args = {}
        if not isinstance(args, dict):
            args = {}
        tool_id = str(tool_call.get("id") or tool_call.get("tool_call_id") or name)
        return name, args, tool_id

    def _required_nudge(self) -> str:
        if self._stage == "intake":
            return (
                "<intake_required>Submit the complete intake object with submit_intake; "
                "fix any reported validation error and resubmit.</intake_required>"
            )
        if self._stage == "self_review":
            return (
                "<selfcheck_required>Review the current saved blueprint and call "
                "submit_blueprint_selfcheck with PASS or REDO.</selfcheck_required>"
            )
        return (
            "<blueprint_required>Submit a complete valid blueprint with write_blueprint; "
            "fix any reported errors and resubmit.</blueprint_required>"
        )

    async def _run_loop(self) -> DesignResult:
        turns = 0
        while turns < self.max_turns:
            turns += 1
            assistant, tool_calls, usage, sample = await self._next_turn()
            if any(
                key in assistant
                for key in ("content", "tool_calls", "thought_summary", "extra_content")
            ):
                self._conversation.append(assistant)
            self._trace(
                {
                    "ts": time.time(),
                    "type": "design_blueprint_turn",
                    "agent": "design",
                    "provider": self.provider,
                    "model_id": self.model_id,
                    "stage": self._stage,
                    "turn": turns,
                    "assistant": assistant,
                    "tool_calls": tool_calls,
                    "usage": usage,
                }
            )
            turn_cost = 0.0
            if usage and self.cost:
                turn_cost = self.cost.add_turn(usage).total_cost
            if sample is not None and self.recorder is not None:
                self.recorder.record(
                    kind="llm",
                    name=self.ROLE,
                    sample=sample,
                    tokens=usage or None,
                    cost_usd=turn_cost if usage else None,
                    model_id=self.model_id,
                    expected_cold=False,
                )
            if usage and self.cost:
                total = self.cost.all_in_total_cost_usd
                if self.max_cost_usd is not None and total > self.max_cost_usd:
                    self._save_session()
                    return self._result(False, "cost_limit", turns)

            if not tool_calls:
                self._conversation.append({"role": "user", "content": self._required_nudge()})
                self._save_session()
                continue

            entered_self_review = False
            for tool_call in tool_calls:
                name, args, tool_id = self._decode_tool_call(tool_call)
                previous_stage = self._stage
                out = self._dispatch(name, args)
                self._conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "name": name,
                        "content": out,
                    }
                )
                entered_self_review = (
                    entered_self_review
                    or previous_stage == "blueprint_authoring"
                    and self._stage == "self_review"
                )

            if self._terminal_reason == "intake_valid":
                self._save_session()
                return self._result(True, "intake_valid", turns)
            if self._stage == "done":
                self._save_session()
                return self._result(True, "blueprint_valid", turns)
            if self._stage == "incomplete":
                self._save_session()
                return self._result(False, "self_review_incomplete", turns)
            if entered_self_review:
                self._append_self_review_prompt()
            else:
                self._save_session()

        self._save_session()
        return self._result(False, "max_turns", turns)

    def _result(
        self,
        ok: bool,
        reason: str,
        conversation_or_turns: list[dict[str, Any]] | int,
        turns: int | None = None,
    ) -> DesignResult:
        if turns is None:
            turn_count = int(conversation_or_turns)
            conversation = getattr(self, "_conversation", [])
        else:
            turn_count = turns
            conversation = conversation_or_turns
        if self._trace_fp is not None:
            self._trace_fp.close()
            self._trace_fp = None
        return DesignResult(
            success=ok,
            reason=reason,
            blueprint=self._blueprint,
            conversation=conversation,
            turns=turn_count,
            cost_usd=(self.cost.all_in_total_cost_usd if self.cost else 0.0),
            asset_brief=getattr(self, "_asset_brief", None),
        )
