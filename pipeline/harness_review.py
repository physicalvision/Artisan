from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from rich.console import Console

from pipeline.agents.prompts.review import load_review_prompt
from pipeline.context_policy import (
    DISABLED_COMPRESSION_POLICY,
    cap_tool_output,
    dedup_read,
    enable_native_compaction,
    policy_from_env,
    round_floats,
    sha1_text,
)
from pipeline.contracts import (
    CodeRef,
    Decision,
    DynamicsAdjudication,
    Issue,
    ReviewIncomplete,
    ReviewReport,
    ReviewResult,
    Severity,
    StabilityAdjudication,
    StabilityVerdict,
    TerminationReason,
)
from pipeline.cost import build_cost_tracker
from pipeline.harness_base import _ToolLoopHarness
from pipeline.harness_codec import MessageCodec
from pipeline.harness_sdk import (
    _record_read_coverage,
    _resolve_read_target,
    read_capped,
    read_file_window,
)
from pipeline.llm.factory import (
    ProviderConfig,
    create_provider_client,
)
from pipeline.llm.openai import DEFAULT_OPENAI_MODEL
from pipeline.policies import (
    CONDITIONAL_RUBRIC_DIMS,
    DEFAULT_POLICIES,
    RUBRIC_DIMS,
    Policies,
)
from pipeline.prompt_cache import build_openai_prompt_cache_settings
from pipeline.providers import resolve_pipeline_model
from pipeline.traces import TraceWriter
from pipeline.tui.single_run import SingleRunDisplay
from sdk.runtime import render as sdk_render
from sdk.runtime import runtime as sdk_runtime
from sdk.runtime.accepted_inspection import current_reference, guarded_inspection

logger = logging.getLogger(__name__)

QA_TEST_FILENAME = "qa_tests.py"


def _required_rubric(phase: str, workspace: Path | None = None) -> tuple[str, ...]:
    del phase
    return tuple(
        dim
        for dim in RUBRIC_DIMS
        if dim not in CONDITIONAL_RUBRIC_DIMS
        or (workspace is not None and (Path(workspace) / CONDITIONAL_RUBRIC_DIMS[dim]).is_file())
    )


def _as_score(value: Any) -> int | None:
    try:
        return max(0, min(10, int(value)))
    except (TypeError, ValueError):
        return None


def _window_covers_whole_file(text: str, offset: int | None, limit: int | None) -> bool:
    if offset is not None and offset != 1:
        return False
    if limit is None:
        return True
    return limit >= len(text.splitlines())


def _stability_submit_missing(
    payload: dict[str, Any],
    expected: set[tuple[str, str]],
) -> list[str]:
    missing: list[str] = []
    raw_adjudications = payload.get("stability_adjudications") or []
    if not isinstance(raw_adjudications, list):
        return ["stability_adjudications must be a list"]
    issues = {
        str(item.get("issue_id")): str(item.get("severity"))
        for item in (payload.get("issues") or [])
        if isinstance(item, dict) and item.get("issue_id")
    }
    seen: set[tuple[str, str]] = set()
    covered: set[tuple[str, str]] = set()
    for index, raw in enumerate(raw_adjudications):
        if not isinstance(raw, dict):
            missing.append(f"stability adjudication #{index + 1} must be an object")
            continue
        pose = str(raw.get("pose") or "").strip()
        code = str(raw.get("finding_code") or "").strip()
        key = (pose, code)
        if not pose or not code:
            missing.append(f"stability adjudication #{index + 1} needs pose and finding_code")
            continue
        if key in seen:
            missing.append(f"duplicate stability adjudication for {pose}/{code}")
            continue
        seen.add(key)
        if key not in expected:
            missing.append(f"unknown stability advisory {pose}/{code}")
            continue
        covered.add(key)
        rationale = str(raw.get("rationale") or "").strip()
        raw_evidence = raw.get("evidence")
        if not isinstance(raw_evidence, list):
            missing.append(f"stability adjudication {pose}/{code} evidence must be a list")
            evidence: list[str] = []
        else:
            evidence = [str(value).strip() for value in raw_evidence if str(value).strip()]
        if not rationale:
            missing.append(f"stability adjudication {pose}/{code} needs a rationale")
        if not evidence:
            missing.append(f"stability adjudication {pose}/{code} needs evidence")
        verdict = str(raw.get("verdict") or "")
        if verdict == StabilityVerdict.BLOCKING.value:
            issue_id = str(raw.get("issue_id") or "").strip()
            if not issue_id or issues.get(issue_id) != Severity.BLOCKING.value:
                missing.append(
                    f"blocking stability adjudication {pose}/{code} must reference a blocking issue"
                )
        elif verdict != StabilityVerdict.ACCEPTABLE.value:
            missing.append(
                f"stability adjudication {pose}/{code} verdict must be acceptable or blocking"
            )
    for pose, code in sorted(expected - covered):
        missing.append(f"missing stability adjudication for {pose}/{code}")
    return missing


def _dynamics_submit_missing(
    payload: dict[str, Any],
    expected: set[tuple[str, str]],
) -> list[str]:
    missing: list[str] = []
    raw_adjudications = payload.get("dynamics_adjudications") or []
    if not isinstance(raw_adjudications, list):
        return ["dynamics_adjudications must be a list"]
    issues = {
        str(item.get("issue_id")): str(item.get("severity"))
        for item in (payload.get("issues") or [])
        if isinstance(item, dict) and item.get("issue_id")
    }
    covered: set[tuple[str, str]] = set()
    for index, raw in enumerate(raw_adjudications):
        if not isinstance(raw, dict):
            missing.append(f"dynamics adjudication #{index + 1} must be an object")
            continue
        joint = str(raw.get("joint") or "").strip()
        code = str(raw.get("finding_code") or "").strip()
        key = (joint, code)
        if not joint or not code:
            missing.append(f"dynamics adjudication #{index + 1} needs joint and finding_code")
            continue
        if key in covered:
            missing.append(f"duplicate dynamics adjudication for {joint}/{code}")
            continue
        if key not in expected:
            missing.append(f"unknown dynamics finding {joint}/{code}")
            continue
        covered.add(key)
        rationale = str(raw.get("rationale") or "").strip()
        evidence = raw.get("evidence")
        if not rationale:
            missing.append(f"dynamics adjudication {joint}/{code} needs a rationale")
        if not isinstance(evidence, list) or not any(str(value).strip() for value in evidence):
            missing.append(f"dynamics adjudication {joint}/{code} needs evidence")
        verdict = str(raw.get("verdict") or "")
        if verdict == StabilityVerdict.BLOCKING.value:
            issue_id = str(raw.get("issue_id") or "").strip()
            if not issue_id or issues.get(issue_id) != Severity.BLOCKING.value:
                missing.append(
                    f"blocking dynamics adjudication {joint}/{code} must reference a blocking issue"
                )
        elif verdict != StabilityVerdict.ACCEPTABLE.value:
            missing.append(
                f"dynamics adjudication {joint}/{code} verdict must be acceptable or blocking"
            )
    for joint, code in sorted(expected - covered):
        missing.append(f"missing dynamics adjudication for {joint}/{code}")
    return missing


SYSTEM_PROMPT = load_review_prompt("system.md")


def _review_tool_schemas() -> list[dict[str, Any]]:
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
    num = {"type": "number"}
    arr_s = {"type": "array", "items": {"type": "string"}}
    return [
        fn(
            "read_file",
            "Read model.py (ground every claim in the real source), qa_tests.py, "
            "the blueprint, the CodingAgent's tasks.json (the downstream manipulation "
            "spec — verify its bindings), selected candidate artifacts, or public SDK "
            "docs/examples. Reads are restricted to the workspace, selected artifacts, "
            "and public references; SDK implementation source is unavailable. "
            "Use offset/limit for a 1-based line window.",
            {"path": s, "offset": line, "limit": line},
            ["path"],
        ),
        fn(
            "read_gate_evidence",
            "Read deterministic gate evidence with identity validation. Returns FRESH "
            "or STALE independently for motion evidence and exact candidate physics.",
            {
                "scope": {
                    "type": "string",
                    "enum": ["motion", "candidate", "all"],
                }
            },
            [],
        ),
        fn(
            "probe_model",
            "Inspect the model WITHOUT a pass/fail gate. NO args → the STRUCTURE "
            "(real part/joint names + axes/limits, so `target`/`code_ref` are exact). "
            "Pass `code` (a short Python snippet ending in exactly one `emit(value)`) "
            "to MEASURE geometry and verify a render suspicion — helpers in scope "
            "include part(name)/visual(...), aabb/dims/center, "
            "pair_report(a,b)/overlap_report(a,b)/mount_report(child,parent)/"
            'geometry_connectivity_report(...) (signatures: read_file("probe_tooling")). '
            "Prefer this over authoring a throwaway qa_test just to read one number.",
            {"code": s},
            [],
        ),
        fn(
            "write_test",
            "Write the WHOLE qa_tests.py (one `def qa_tests(ctx):`). Use this for "
            "the FIRST version or a full rewrite; for small round-to-round changes "
            "prefer edit_test. The file persists across rounds (goalpost stability).",
            {"content": s},
            ["content"],
        ),
        fn(
            "edit_test",
            "Incrementally edit qa_tests.py: replace the first exact occurrence of "
            "old_str with new_str (like a surgical patch). Prefer this over "
            "rewriting the whole file when adding/adjusting one assertion.",
            {"old_str": s, "new_str": s},
            ["old_str", "new_str"],
        ),
        fn(
            "run_tests",
            "Run the current qa_tests.py against the model. Returns "
            "{ok, failures:[{name,details}], error}.",
            {},
            [],
        ),
        fn(
            "request_render",
            "Render the build; the image comes back as this tool's result, visible when you "
            "next act (same as any tool's output, no extra delay). `views`: named "
            "presets (front/side/top/perspective, optional _open/_closed). `pose`: "
            "joint->value to drive a specific configuration (motion sweep). "
            "`camera`: a free angle {azimuth_deg,elevation_deg,distance_mult}. "
            "`focus`: part/joint name(s) to AIM the camera at and ZOOM into (a "
            "joint = a tight close-up on its pivot); combine with views/camera for a "
            "clean line of sight — a PAID render, so use crop_region for a free zoom "
            "of an existing image first. "
            "`segment`=true recolours each PART a distinct flat colour (a 'semantic "
            "map') so a part buried inside another reads as a foreign colour the dark "
            "realism materials hide — use it to verify any overlap/declared contact.",
            {
                "views": arr_s,
                "pose": {
                    "type": "object",
                    "additionalProperties": num,
                    "description": (
                        "Joint name -> position map (free-form), e.g. {'input_shaft_spin': 0.785}, "
                        "to drive a specific configuration for a motion-sweep render. Radians for "
                        "revolute/continuous, metres for prismatic. Omit (or {}) for the rest pose. "
                        "Pose the DRIVER of a coupled mechanism; the follower tracks it automatically."
                    ),
                },
                "camera": {
                    "type": "object",
                    "properties": {
                        "azimuth_deg": num,
                        "elevation_deg": num,
                        "distance_mult": num,
                    },
                    "description": (
                        "Free camera angle: azimuth_deg + elevation_deg orbit the object; "
                        "distance_mult scales framing distance (>1 pulls back, <1 zooms in). "
                        "Omit for the default view."
                    ),
                },
                "focus": arr_s,
                "segment": {"type": "boolean"},
            },
            ["views"],
        ),
        fn(
            "crop_region",
            "Zoom a sub-region of a rendered image to inspect a small/contact "
            "feature (hinge seam, latch, rim). Coordinates are NORMALIZED 0..1 with "
            "the ORIGIN at the TOP-LEFT, where 0.5 is the center; coarse is fine.",
            {
                "image": {
                    "type": "string",
                    "description": "Name of the rendered image to crop (e.g. 'perspective_open.png').",
                },
                "x1": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Left edge (0=left, 0.5=center).",
                },
                "y1": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Top edge (0=top, 0.5=center).",
                },
                "x2": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Right edge (>x1).",
                },
                "y2": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Bottom edge (>y1).",
                },
            },
            ["image", "x1", "y1", "x2", "y2"],
        ),
        fn(
            "submit_review",
            "Finish the review with the structured verdict. Provide overall "
            "realism_rating (0-10), rubric scores for THIS phase's dimensions "
            "(0-10 each; you'll be told which — score others only if you gathered "
            "evidence, never a guess), a summary, and the issues you found. You do "
            "NOT state a proceed/revise decision: it is DERIVED from your scores + "
            "issue severities (a blocking issue, or any gating dim < 5, routes a "
            "redo). Complete every required evidence step before submitting.",
            {
                "realism_rating": {"type": "integer"},
                "rubric": {
                    "type": "object",
                    "properties": {d: {"type": "integer"} for d in RUBRIC_DIMS},
                },
                "summary": s,
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "issue_id": s,
                            "severity": {
                                "type": "string",
                                "enum": ["blocking", "major", "minor"],
                            },
                            "target": s,
                            "problem": s,
                            "required_change": s,
                            "evidence": arr_s,
                            "code_ref": {
                                "type": "object",
                                "properties": {
                                    "file": s,
                                    "symbol": s,
                                    "line": {"type": "integer"},
                                },
                            },
                        },
                    },
                },
                "stability_adjudications": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pose": s,
                            "finding_code": s,
                            "verdict": {
                                "type": "string",
                                "enum": ["acceptable", "blocking"],
                            },
                            "rationale": s,
                            "evidence": arr_s,
                            "issue_id": s,
                        },
                        "required": [
                            "pose",
                            "finding_code",
                            "verdict",
                            "rationale",
                            "evidence",
                        ],
                    },
                },
                "dynamics_adjudications": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "joint": s,
                            "finding_code": s,
                            "verdict": {
                                "type": "string",
                                "enum": ["acceptable", "blocking"],
                            },
                            "rationale": s,
                            "evidence": arr_s,
                            "issue_id": s,
                        },
                        "required": [
                            "joint",
                            "finding_code",
                            "verdict",
                            "rationale",
                            "evidence",
                        ],
                    },
                },
            },
            ["realism_rating", "rubric", "summary"],
        ),
    ]


_PHASE_REQUIRED: dict[str, tuple[str, ...]] = {
    "P5_review": ("ran_qa", "rendered", "rendered_open"),
}
_REQUIRE_LABELS = {
    "ran_qa": "run a QA test (write_test + run_tests)",
    "rendered": "request at least one render",
    "rendered_open": "render an open / swept pose (pose= or a *_open view)",
}


class SdkReviewHarness(_ToolLoopHarness):

    ROLE = "ReviewAgent"
    SYSTEM_PROMPT = SYSTEM_PROMPT
    _CASSETTE_KIND = "review_turn"

    _policy = DISABLED_COMPRESSION_POLICY
    _turn = 0
    _consecutive_compile_failures = 0
    _last_compile_failure_sig: str | None = None
    _probe_done = False
    _probe_turn = 0
    _expected_stability_advisories: frozenset[tuple[str, str]] = frozenset()
    _expected_dynamics_inconclusive: frozenset[tuple[str, str]] = frozenset()

    def __init__(
        self,
        model_path: Path,
        *,
        backend: Any,
        sdk_package: str = "sdk",
        provider: str = "openai",
        model_id: str | None = None,
        thinking_level: str = "high",
        max_cost_usd: float | None = None,
        trace_dir: Path | None = None,
        policies: Policies = DEFAULT_POLICIES,
        cassette: object | None = None,
        render_root: Path | None = None,
        qa_test_path: Path | None = None,
        recorder: object | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.backend = backend
        self.sdk_package = sdk_package
        self.policies = policies
        self.cassette = cassette
        self.recorder = recorder.for_agent("review") if recorder is not None else None
        self.max_turns = policies.review_turn_budget
        self.wrap_up_start_turn = policies.agent_wrap_up_start_turn
        self.max_cost_usd = max_cost_usd
        self.render_root = Path(render_root) if render_root else self.model_path.parent / "renders"
        self.qa_test_path = (
            Path(qa_test_path) if qa_test_path else self.model_path.parent / QA_TEST_FILENAME
        )

        self.provider = provider
        _ck, _cret, _cfp = (None, None, None)
        if provider == "openai":
            _ck, _cret, _cfp = build_openai_prompt_cache_settings(
                model_id=model_id or DEFAULT_OPENAI_MODEL,
                sdk_package=sdk_package,
                system_prompt=SYSTEM_PROMPT,
                sdk_docs_context="",
                tools=_review_tool_schemas(),
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
        self.cost = build_cost_tracker(provider, _mid)
        self.trace = TraceWriter(trace_dir) if trace_dir else None
        self.display = SingleRunDisplay(
            console=Console(),
            model_id=_mid,
            thinking_level=thinking_level,
            max_turns=self.max_turns,
            enabled=True,
        )
        self._ran_qa = False
        self._rendered = False
        self._rendered_open = False
        self._read_tasks = False
        self._read_tasks_partial = False
        self._render_seq = 0
        self._phase = ""
        self._expected_stability_advisories: set[tuple[str, str]] = set()
        self._expected_dynamics_inconclusive: set[tuple[str, str]] = set()
        self._round = 0
        self._round_by_phase: dict[str, int] = {}
        self._policy = policy_from_env()
        enable_native_compaction(self.llm, self.provider, self._policy, trace=self.trace)
        self._turn = 0
        self._read_cache: dict[str, str] = {}
        self._read_seen_turn: dict[str, int] = {}
        self._probe_done = False
        self._probe_turn = 0
        self._probe_full_chars = 0
        self._probe_code_seen: set[str] = set()
        self._probe_code_full_chars: dict[str, int] = {}
        self._qa_run_cache: dict[str, int] = {}
        self._consecutive_compile_failures = 0
        self._last_compile_failure_sig: str | None = None

        unpermitted = [
            t["function"]["name"]
            for t in _review_tool_schemas()
            if not self.policies.can_use_tool(self.ROLE, t["function"]["name"])
        ]
        if unpermitted:
            raise RuntimeError(
                f"tool/policy mismatch: {self.ROLE} advertises but is denied "
                f"{unpermitted}; fix policies.ROLE_TOOLS"
            )

    @property
    def cost_usd(self) -> float:
        return self._current_total_cost_usd()

    def _read_file(
        self,
        path: str,
        offset: int | None = None,
        limit: int | None = None,
    ) -> tuple[str, str]:
        p, error = _resolve_read_target(self, path)
        if error is not None or p is None:
            error = error or "ERROR: no readable file was resolved"
            return error, error
        if not p.exists():
            if p == self.qa_test_path.resolve() and not self.qa_test_path.exists():
                note = (
                    "NOTE: qa_tests.py is not created yet. This is expected early — "
                    "first read_file('testing'), probe the model, and read model.py, "
                    "then author write_test(...)."
                )
                return note, note
            present = [
                n
                for n in ("model.py", "qa_tests.py", "blueprint.json", "tasks.json")
                if (self.model_path.parent / n).exists()
            ]
            files_hint = ", ".join(f"`{n}`" for n in present) if present else "(none written yet)"
            err = (
                f"ERROR: no such file: {path}. Readable workspace files: {files_hint}; "
                "or an SDK reference doc by topic name — `testing` (the ctx QA API), "
                "`probe_tooling`, `placement`, `core_types`, `articulated_object`. Use one "
                "of these exact names; do not guess other paths."
            )
            return err, err
        if p.is_dir():
            error = "ERROR: read_file reads a single text file, not a directory."
            return error, error
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            error = "ERROR: the requested file cannot be read as UTF-8 text."
            return error, error
        selected, window_error = read_file_window(
            text,
            offset=offset,
            limit=limit,
            line_numbers=p.suffix == ".py",
        )
        if window_error is not None or selected is None:
            error = window_error or "ERROR: unable to read requested window"
            return error, error
        full = read_capped(selected)
        self._record_tier_a_savings("read_file", len(selected) - len(full))
        source_digest, complete, shown_range = _record_read_coverage(
            self, p, text, selected, offset, limit
        )
        if p == (self.model_path.parent / "tasks.json").resolve():
            self._read_tasks = complete
            self._read_tasks_partial = not complete
        if not self._policy.enabled or not self._policy.read_file_dedup:
            return full, full
        compact = dedup_read(
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
        return compact, compact

    def _accepted_inspection_reference(self) -> dict[str, Any] | None:
        from sdk.runtime.accepted_inspection import workspace_reference

        if not hasattr(self, "_review_candidate_reference"):
            self._review_candidate_reference = workspace_reference(
                self.model_path,
            )
        return self._review_candidate_reference

    def _refresh_inspection_inputs(self) -> None:
        from sdk.runtime.inspection import inspection_input_key

        identity = inspection_input_key(self.model_path)
        if getattr(self, "_probe_inputs", None) != identity:
            self._probe_inputs = identity
            self._probe_done = False
            self._probe_code_seen = set()
            self._probe_code_full_chars = {}

    @guarded_inspection
    def _probe_model(self, code: str | None = None) -> tuple[str, str]:
        self._refresh_inspection_inputs()
        code = (code or "").strip()
        if code:
            return self._probe_measure(code)
        from sdk.runtime.inspection import _load_reference
        from sdk.runtime.inspection import automatic_snapshot_reference as snapshot_reference

        try:
            reference = snapshot_reference(
                self.model_path,
                sdk_package=self.sdk_package,
                require_current=(self.model_path.parent / "candidate/manifest.json").is_file(),
            )
            manifest = (
                _load_reference(reference, self.model_path, self.sdk_package) if reference else None
            )
        except Exception as exc:
            err = f"ERROR: inspection snapshot is not current: {exc}"
            return err, err
        if self._policy.enabled and self._policy.probe_dedup and self._probe_done:
            stub = (
                "[identical structural probe this round; model.py is unchanged "
                "during review — that result is above.]"
            )
            self._record_tier_a_savings(
                "probe_model", int(getattr(self, "_probe_full_chars", 0)) - len(stub)
            )
            return stub, stub
        try:
            probe = sdk_runtime.probe_summary(
                self.model_path,
                sdk_package=self.sdk_package,
                preserve_existing_meshes=True,
                **({"collision_manifest": manifest} if manifest is not None else {}),
            )
        except Exception as exc:  # noqa: BLE001
            err = f"ERROR running probe: {exc}"
            return err, err
        full = json.dumps(round_floats(probe), indent=2)
        if not self._policy.enabled:
            return full, full
        self._probe_done = True
        self._probe_turn = self._turn
        self._probe_full_chars = len(full)
        compact = cap_tool_output(
            full, self._policy.tool_output_hard_cap_chars, label="probe_model"
        )
        return compact, full

    @guarded_inspection
    def _probe_measure(self, code: str) -> tuple[str, str]:
        self._refresh_inspection_inputs()
        from sdk.runtime.inspection import automatic_snapshot_reference as snapshot_reference

        try:
            reference = snapshot_reference(
                self.model_path,
                sdk_package=self.sdk_package,
                require_current=(self.model_path.parent / "candidate/manifest.json").is_file(),
            )
        except Exception as exc:
            err = f"ERROR: inspection snapshot is not current: {exc}"
            return err, err
        key = sha1_text(code)
        if self._policy.enabled and self._policy.probe_dedup and key in self._probe_code_seen:
            stub = (
                "[identical measurement probe this round; model.py is unchanged "
                "during review — that result is above.]"
            )
            full_chars = getattr(self, "_probe_code_full_chars", {}).get(key, 0)
            self._record_tier_a_savings("probe_model", int(full_chars) - len(stub))
            return stub, stub
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
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            err = "ERROR: probe_model timed out (120s)."
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
        self._probe_code_seen.add(key)
        full = json.dumps(round_floats(payload), indent=2)
        probe_chars = getattr(self, "_probe_code_full_chars", None)
        if not isinstance(probe_chars, dict):
            probe_chars = {}
            self._probe_code_full_chars = probe_chars
        probe_chars[key] = len(full)
        if not self._policy.enabled:
            return full, full
        compact = cap_tool_output(
            full, self._policy.tool_output_hard_cap_chars, label="probe_model"
        )
        return compact, full

    def _write_test(self, content: str) -> str:
        if "def qa_tests" not in content:
            return "ERROR: content must define `def qa_tests(ctx):`"
        overwrote = bool(
            self.qa_test_path.exists() and self.qa_test_path.read_text(encoding="utf-8").strip()
        )
        self.qa_test_path.parent.mkdir(parents=True, exist_ok=True)
        self.qa_test_path.write_text(content, encoding="utf-8")
        msg = f"OK: wrote {len(content)} chars to {self.qa_test_path.name}"
        if overwrote:
            msg += (
                "\nNOTE: you OVERWROTE an existing qa_tests.py — KEEP every assertion the "
                "model already PASSED (do not loosen a threshold it satisfies) and do not "
                "silently drop prior checks. Prefer edit_test for small changes."
            )
        return msg

    def _edit_test(self, old_str: str, new_str: str) -> str:
        if not self.qa_test_path.exists():
            return "ERROR: no qa_tests.py yet — write_test the first version first."
        src = self.qa_test_path.read_text(encoding="utf-8")
        if old_str not in src:
            return "ERROR: old_str not found exactly once"
        self.qa_test_path.write_text(src.replace(old_str, new_str, 1), encoding="utf-8")
        return "OK: edited qa_tests.py"

    @guarded_inspection
    def _run_tests(self) -> tuple[str, str]:
        if not self.qa_test_path.exists():
            err = "ERROR: no qa_tests.py yet — write_test first."
            return err, err
        code = self.qa_test_path.read_text(encoding="utf-8")
        self._ran_qa = True
        if self._policy.enabled and self._policy.probe_dedup:
            try:
                model_sha = sha1_text(self.model_path.read_text(encoding="utf-8"))
            except OSError:
                model_sha = ""
            key = f"{sha1_text(code)}:{model_sha}"
            cached_chars = self._qa_run_cache.get(key)
            if cached_chars:
                stub = (
                    "[identical run_tests this round; qa_tests.py and model.py are both "
                    "unchanged — that full result is above. Re-running is a no-op until you "
                    "edit the test (or the source changes).]"
                )
                if isinstance(cached_chars, int):
                    self._record_tier_a_savings("run_tests", int(cached_chars) - len(stub))
                return stub, stub
        result = sdk_runtime.run_qa_tests(
            self.model_path,
            code,
            sdk_package=self.sdk_package,
            preserve_existing_meshes=True,
        )
        body = json.dumps(result, indent=2)
        prefix = ""
        if not result.get("ok") and result.get("error") and not result.get("failures"):
            prefix = (
                "TESTER ERROR — your qa_tests(ctx) code itself failed (it raised, used a "
                "missing helper, or the model could not load). This is a bug in YOUR test, "
                "NOT a model fault: fix or drop the offending assertion. Do NOT raise an "
                "issue or route this to Coding.\n\n"
            )
        full = prefix + body
        if self._policy.enabled and self._policy.probe_dedup:
            self._qa_run_cache[key] = len(full)
        if not self._policy.enabled:
            return full, full
        return prefix + self._compact_qa(result), full

    @staticmethod
    def _compact_qa(result: dict[str, Any]) -> str:
        if result.get("error") and not result.get("failures"):
            return f"error: {result['error']}"
        checks = result.get("checks") or []
        failures = result.get("failures") or []
        warnings = result.get("warnings") or []
        total = len(checks)
        failed = len(failures)
        passed = max(0, total - failed)
        head = f"qa_tests: {passed}/{total} passed · {failed} failed · {len(warnings)} warning(s)"
        if not failures and not warnings:
            return head + " ✓"
        lines = [head]
        if failures:
            lines.append("FAILURES:")
            for f in failures:
                name = f.get("name", "?") if isinstance(f, dict) else str(f)
                details = f.get("details", "") if isinstance(f, dict) else ""
                lines.append(f"  - {name}" + (f" — {details}" if details else ""))
        if warnings:
            lines.append("WARNINGS:")
            lines.extend(f"  - {w}" for w in warnings)
        return "\n".join(lines)

    def _render_out_dir(self) -> Path:
        self._render_seq += 1
        d = self.render_root / f"{self._phase}_r{self._round:02d}" / f"{self._render_seq:02d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _call_tag(self) -> str:
        return f"r{self._round:02d}c{self._render_seq:02d}"

    @staticmethod
    def _tag_outputs(paths: list[Path], tag: str) -> list[Path]:
        out: list[Path] = []
        for p in (Path(x) for x in paths):
            dest = p.with_name(f"{p.stem}__{tag}{p.suffix}")
            try:
                p.rename(dest)
                out.append(dest)
            except OSError:
                out.append(p)
        return out

    @staticmethod
    def _pose_note(pose: dict | None, focus: list[str] | None) -> str:
        bits = []
        if pose:
            moved = {k: v for k, v in pose.items() if v}
            bits.append("pose: " + (", ".join(f"{k}={v}" for k, v in moved.items()) or "rest"))
        else:
            bits.append("pose: rest")
        if focus:
            bits.append("focus: " + ",".join(focus))
        return " · ".join(bits)

    @guarded_inspection
    def _request_render(self, args: dict[str, Any]) -> tuple[str, list[Path]]:
        views = [str(v) for v in (args.get("views") or [])]
        if not views:
            return "ERROR: provide at least one view name.", []
        pose = args.get("pose") or None
        camera = args.get("camera") or None
        focus = [str(f) for f in (args.get("focus") or [])] or None
        segment = bool(args.get("segment"))
        if pose:
            pose = {str(k): float(v) for k, v in pose.items()}
        out_dir = self._render_out_dir()
        tag = self._call_tag()
        try:
            from sdk.runtime.inspection import automatic_snapshot_reference as snapshot_reference

            reference = snapshot_reference(
                self.model_path,
                sdk_package=self.sdk_package,
                require_current=(self.model_path.parent / "candidate/manifest.json").is_file(),
            )
            paths = self.backend.render_views(
                self.model_path,
                views,
                out_dir,
                pose=pose,
                camera=camera,
                focus=focus,
                segment=segment,
                **({"inspection_snapshot": reference} if reference else {}),
            )
        except Exception as exc:  # noqa: BLE001
            return f"ERROR rendering: {exc}", []
        if not paths:
            return "ERROR: no renders were produced.", []
        self._rendered = True
        if pose or any(v.endswith("_open") for v in views):
            self._rendered_open = True
        tagged = self._tag_outputs([Path(p) for p in paths], tag)
        names = ", ".join(p.name for p in tagged)
        return (
            f"OK: rendered {len(tagged)} image(s): {names}  ({self._pose_note(pose, focus)})",
            tagged,
        )

    def _resolve_render(self, image: str) -> Path | None:
        src = Path(image)
        if src.is_absolute() and src.exists():
            return src
        matches = sorted(self.render_root.rglob(Path(image).name), key=lambda p: p.stat().st_mtime)
        return matches[-1] if matches else (src if src.exists() else None)

    def _crop_region(self, args: dict[str, Any]) -> tuple[str, list[Path]]:
        image = args.get("image")
        try:
            x1, y1, x2, y2 = (float(args[k]) for k in ("x1", "y1", "x2", "y2"))
        except (KeyError, TypeError, ValueError):
            return "ERROR: need image + normalized x1,y1,x2,y2 in 0..1 (top-left origin).", []
        if not image:
            return "ERROR: need the name of the rendered image to crop.", []
        if not all(0.0 <= c <= 1.0 for c in (x1, y1, x2, y2)):
            return "ERROR: coordinates must be normalized 0..1 (top-left origin).", []
        if x1 >= x2 or y1 >= y2:
            return "ERROR: invalid box — need x1<x2 and y1<y2.", []
        src = self._resolve_render(image)
        if src is None:
            return f"ERROR: no such rendered image: {image}", []
        out_dir = self._render_out_dir()
        tag = self._call_tag()
        try:
            out = sdk_render.crop_image(src, (x1, y1, x2, y2), out_dir)
        except Exception as exc:  # noqa: BLE001
            return f"ERROR cropping: {exc}", []
        out = self._tag_outputs([out], tag)[0]
        return f"OK: cropped {src.name} -> {out.name} ({x1:.2f},{y1:.2f})-({x2:.2f},{y2:.2f})", [
            out
        ]

    def _submit_missing(self, payload: dict[str, Any]) -> list[str]:
        missing: list[str] = []
        rubric = payload.get("rubric") or {}
        for d in _required_rubric(self._phase, self.model_path.parent):
            if _as_score(rubric.get(d)) is None:
                missing.append(f"score the rubric dimension `{d}` (a number 0-10)")
        tasks_path = self.model_path.parent / CONDITIONAL_RUBRIC_DIMS["manip_annotation"]
        if tasks_path.is_file() and not self._read_tasks:
            if getattr(self, "_read_tasks_partial", False):
                missing.append(
                    "read ALL of tasks.json (your last read was a partial window) "
                    "and judge the manipulation annotation"
                )
            else:
                missing.append("read tasks.json and judge the manipulation annotation")
        flags = {
            "ran_qa": self._ran_qa,
            "rendered": self._rendered,
            "rendered_open": self._rendered_open,
        }
        for req in _PHASE_REQUIRED.get(self._phase, ("rendered",)):
            if flags.get(req, False):
                continue
            missing.append(_REQUIRE_LABELS[req])
        missing.extend(
            _stability_submit_missing(
                payload,
                set(getattr(self, "_expected_stability_advisories", set())),
            )
        )
        missing.extend(
            _dynamics_submit_missing(
                payload,
                set(getattr(self, "_expected_dynamics_inconclusive", set())),
            )
        )
        return missing

    def _existing_max_round(self, phase: str) -> int:
        mx = 0
        if self.render_root.exists():
            pat = re.compile(rf"^{re.escape(phase)}_r(\d+)$")
            for d in self.render_root.iterdir():
                m = pat.match(d.name) if d.is_dir() else None
                if m:
                    mx = max(mx, int(m.group(1)))
        return mx

    async def run(
        self,
        phase: str,
        task_text: str,
        *,
        images: list[Path] | None = None,
        initial_conversation: list[dict] | None = None,
        attach_evidence: bool = False,
        expected_stability_advisories: set[tuple[str, str]] | None = None,
        expected_dynamics_inconclusive: set[tuple[str, str]] | None = None,
    ) -> tuple[ReviewResult, list[dict]]:
        from sdk.runtime.accepted_inspection import workspace_reference

        self._review_candidate_reference = workspace_reference(
            self.model_path,
        )
        self._start_tier_a_telemetry()
        self._phase = phase
        self._expected_stability_advisories = set(expected_stability_advisories or set())
        self._expected_dynamics_inconclusive = set(expected_dynamics_inconclusive or set())
        if phase not in self._round_by_phase:
            self._round_by_phase[phase] = self._existing_max_round(phase)
        self._round_by_phase[phase] += 1
        self._round = self._round_by_phase[phase]
        self._ran_qa = self._rendered = self._rendered_open = False
        self._read_tasks = False
        self._read_tasks_partial = False
        self._render_seq = 0
        self._qa_run_cache = {}
        self._probe_code_seen = set()
        self._probe_code_full_chars = {}

        conversation: list[dict] = list(initial_conversation or [])
        if self.trace and not conversation:
            self.trace.write_message({"role": "system", "content": self.SYSTEM_PROMPT})
        first: list[dict] = [{"type": "input_text", "text": task_text}]
        for img in images or []:
            if img and Path(img).exists():
                first.append({"type": "input_image", "image_path": str(img), "detail": "high"})
        if attach_evidence:
            first.extend(self._gather_repair_evidence())
        conversation.append({"role": "user", "content": first if len(first) > 1 else task_text})
        if self.trace:
            self.trace.write_message(conversation[-1])

        report = await self._run_loop_with_telemetry(conversation)
        return report, conversation

    _EVIDENCE_VIEWS = ("front", "side", "top", "perspective", "front_open", "perspective_open")

    def _gather_repair_evidence(self) -> list[dict]:
        evidence_meta = {"review_evidence_round": self._round}
        parts: list[dict] = []
        lines: list[str] = []
        if self.qa_test_path.exists():
            try:
                compact, _full = self._run_tests()
                lines.append(f"qa_tests re-run on the current build:\n{compact}")
            except Exception:  # noqa: BLE001 — evidence is best-effort
                logger.exception("repair-evidence qa_tests re-run failed (non-fatal)")
        try:
            ack, imgs = self._request_render({"views": list(self._EVIDENCE_VIEWS)})
            if imgs:
                lines.append(f"Standard render set: {ack}")
                parts.extend(
                    {
                        "type": "input_image",
                        "image_path": str(p),
                        "detail": "high",
                        "_meta": dict(evidence_meta),
                    }
                    for p in imgs
                )
        except Exception:  # noqa: BLE001 — evidence is best-effort
            logger.exception("repair-evidence render failed (non-fatal)")
        if lines:
            parts.insert(
                0,
                {
                    "type": "input_text",
                    "text": "<attached_evidence>\n" + "\n\n".join(lines) + "\n</attached_evidence>",
                    "_meta": dict(evidence_meta),
                },
            )
        return parts

    def _tools(self) -> list[dict[str, Any]]:
        return _review_tool_schemas()

    def _wrap_up_nudge(self) -> str:
        return (
            "<wrap_up_budget>You have 50 turns left in this ReviewAgent invocation. "
            "Stop exploratory work. Gather only acceptance evidence still required "
            "by the submit gate, then call submit_review with the final scores, "
            "summary, and actionable issues.</wrap_up_budget>"
        )

    def _end_turn_after_nudge(self) -> bool:
        return True

    def _no_tool_nudge(self) -> str:
        return (
            "<submit_required>You have not called submit_review yet. Finish the "
            "review by calling submit_review with the overall realism_rating, this "
            "phase's rubric dimensions, a summary, and any issues (the proceed/revise "
            "decision is derived from your scores + issue severities — you do not "
            "state it).</submit_required>"
        )

    def _dispatch_tool(self, name: str, args: dict):
        if not self.policies.can_use_tool(self.ROLE, name):
            return f"ERROR: tool {name!r} not permitted for {self.ROLE}", [], None, None
        if name == "submit_review":
            missing = self._submit_missing(args)
            if missing:
                gate = load_review_prompt("submit_gate.md").replace(
                    "{missing}", "\n".join(f"- {m}" for m in missing)
                )
                return gate, [], None, None
            applicable = _required_rubric(self._phase, self.model_path.parent)
            return (
                "OK: review submitted.",
                [],
                _to_report(self._phase, args, rubric_dims=applicable),
                None,
            )
        if name == "request_render":
            text_ack, imgs = self._request_render(args)
            return text_ack, imgs, None, None
        if name == "crop_region":
            text_ack, imgs = self._crop_region(args)
            return text_ack, imgs, None, None
        compact, full = self._dispatch_simple(name, args)
        return compact, [], None, full

    def _zero_cost_terminal_tools(self) -> frozenset[str]:
        return frozenset({"submit_review"})

    def _fallback_result(self, conversation: list[dict], turns: int) -> ReviewIncomplete:
        return self._incomplete_result(
            self._phase,
            "review did not submit within turn budget",
            reason=TerminationReason.MAX_TURNS,
        )

    def _cost_limit_result(self, conversation: list[dict], turns: int) -> ReviewIncomplete:
        return self._incomplete_result(
            self._phase,
            "review hit cost limit",
            reason=TerminationReason.COST_LIMIT,
        )

    def _dispatch_simple(self, name: str, args: dict[str, Any]) -> tuple[str, str]:
        try:
            if name == "read_file":
                return self._read_file(
                    args["path"],
                    offset=args.get("offset"),
                    limit=args.get("limit"),
                )
            if name == "read_gate_evidence":
                body = json.dumps(
                    sdk_runtime.read_gate_evidence(
                        self.model_path, scope=str(args.get("scope") or "all")
                    ),
                    indent=2,
                )
                return body, body
            if name == "probe_model":
                return self._probe_model(args.get("code"))
            if name == "write_test":
                ack = self._write_test(args["content"])
                return ack, ack
            if name == "edit_test":
                ack = self._edit_test(args["old_str"], args["new_str"])
                return ack, ack
            if name == "run_tests":
                return self._run_tests()
        except KeyError as exc:
            err = f"ERROR: missing arg {exc}"
            return err, err
        err = f"ERROR: unknown tool {name!r}"
        return err, err

    def _incomplete_result(
        self,
        phase: str,
        why: str,
        *,
        reason: TerminationReason,
    ) -> ReviewIncomplete:
        logger.error("review fallback (%s): %s", phase, why)
        return ReviewIncomplete(
            report_id=f"{phase}-incomplete",
            phase=phase,
            summary=why,
            reason=reason,
        )


def _to_report(
    phase: str,
    payload: dict[str, Any],
    *,
    rubric_dims: tuple[str, ...] = RUBRIC_DIMS,
) -> ReviewReport:
    issues: list[Issue] = []
    for i, raw in enumerate(payload.get("issues", []) or []):
        cr = raw.get("code_ref")
        try:
            issues.append(
                Issue(
                    issue_id=raw.get("issue_id") or f"{phase}-{i + 1}",
                    severity=Severity(raw.get("severity", "major")),
                    phase=phase,
                    target=raw.get("target", "?"),
                    problem=raw.get("problem", ""),
                    required_change=raw.get("required_change", ""),
                    evidence=list(raw.get("evidence", []) or []),
                    code_ref=CodeRef(**cr) if isinstance(cr, dict) and cr.get("file") else None,
                )
            )
        except (ValueError, TypeError):
            continue
    decision = Decision.REVISE
    rating = _as_score(payload.get("realism_rating"))
    rating = rating if rating is not None else 0
    _rb = payload.get("rubric")
    rubric = {}
    if isinstance(_rb, dict):
        for d in rubric_dims:
            s = _as_score(_rb.get(d))
            if s is not None:
                rubric[d] = s
    stability_adjudications: list[StabilityAdjudication] = []
    for raw in payload.get("stability_adjudications", []) or []:
        if not isinstance(raw, dict):
            continue
        try:
            stability_adjudications.append(
                StabilityAdjudication(
                    pose=str(raw["pose"]),
                    finding_code=str(raw["finding_code"]),
                    verdict=StabilityVerdict(raw["verdict"]),
                    rationale=str(raw["rationale"]),
                    evidence=[str(value) for value in raw.get("evidence", [])],
                    issue_id=(str(raw["issue_id"]) if raw.get("issue_id") else None),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    dynamics_adjudications: list[DynamicsAdjudication] = []
    for raw in payload.get("dynamics_adjudications", []) or []:
        if not isinstance(raw, dict):
            continue
        try:
            dynamics_adjudications.append(
                DynamicsAdjudication(
                    joint=str(raw["joint"]),
                    finding_code=str(raw["finding_code"]),
                    verdict=StabilityVerdict(raw["verdict"]),
                    rationale=str(raw["rationale"]),
                    evidence=[str(value) for value in raw.get("evidence", [])],
                    issue_id=(str(raw["issue_id"]) if raw.get("issue_id") else None),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return ReviewReport(
        report_id=f"{phase}-r",
        phase=phase,
        realism_rating=rating,
        summary=payload.get("summary", ""),
        decision=decision,
        issues=issues,
        rubric=rubric,
        issue_total=len(issues),
        stability_adjudications=stability_adjudications,
        dynamics_adjudications=dynamics_adjudications,
    )
