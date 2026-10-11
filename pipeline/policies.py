from __future__ import annotations

from dataclasses import dataclass, field

from pipeline.contracts import Decision

RUBRIC_DIMS: tuple[str, ...] = (
    "proportion_fidelity",
    "part_decomposition",
    "contacts_clearances",
    "kinematics",
    "reference_parity",
    "material_finish",
    "manip_annotation",
)

CONDITIONAL_RUBRIC_DIMS: dict[str, str] = {"manip_annotation": "tasks.json"}

JOINT_FRAME_SUFFIX = "_joint_frame"
JOINT_PROP_TYPE = "joint_type"
JOINT_PROP_AXIS = "joint_axis"
JOINT_PROP_PARENT = "joint_parent"
JOINT_PROP_CHILD = "joint_child"
JOINT_PROP_LIMIT_LOWER = "joint_limit_lower"
JOINT_PROP_LIMIT_UPPER = "joint_limit_upper"

DIMENSION_PRIORITY: tuple[str, ...] = (
    "user_specified",
    "image_estimate",
    "category_prior",
    "derived_from_geometry",
    "placeholder",
)

MAX_NONBLOCKING_PENDING_ISSUES = 6

ROLE_TOOLS: dict[str, frozenset[str]] = {
    "DesignAgent": frozenset({"submit_intake", "write_blueprint", "submit_blueprint_selfcheck"}),
    "CodingAgent": frozenset(
        {
            "read_file",
            "read_gate_evidence",
            "write_file",
            "replace",
            "run_blender",
            "compile_model",
            "probe_model",
            "request_render",
            "find_examples",
            "lookup_cadquery_api",
            "lookup_blender_api",
            "record_handoff_note",
        }
    ),
    "ReviewAgent": frozenset(
        {
            "read_file",
            "read_gate_evidence",
            "probe_model",
            "write_test",
            "edit_test",
            "run_tests",
            "request_render",
            "crop_region",
            "submit_review",
        }
    ),
    "JudgeAgent": frozenset(
        {
            "read_file",
            "probe_structure",
            "probe_measure",
            "declare_gated",
            "request_render",
            "crop_region",
            "submit_review",
        }
    ),
}


@dataclass(slots=True, frozen=True)
class Policies:
    realism_pass: int = 8
    realism_acceptable: int = 5
    gating_rubric_dims: tuple[str, ...] = (
        "proportion_fidelity",
        "part_decomposition",
        "contacts_clearances",
        "kinematics",
    )
    repair_iteration_budget: int = 5
    review_no_progress_budget: int = 2
    coding_turn_budget: int = 150
    review_turn_budget: int = 150
    agent_wrap_up_start_turn: int = 100
    review_reserve_ratio: float = 0.20
    review_reserve_max_usd: float = 2.5

    axis_angle_tol_rad: float = 0.25
    origin_pos_tol_m: float = 0.05
    revolute_limit_range_rad: tuple[float, float] = (0.0, 6.2832)
    prismatic_limit_range_m: tuple[float, float] = (0.0, 3.0)

    confidence_proceed_threshold: float = 0.6

    tool_failure_escalation_threshold: int = 3

    render_resolution: tuple[int, int] = (1024, 1024)
    required_views: tuple[str, ...] = field(
        default=(
            "front_closed",
            "side_closed",
            "top_closed",
            "perspective_closed",
            "front_open",
            "perspective_open",
        )
    )

    review_image_max_dim: int = 1024

    def can_use_tool(self, role: str, tool: str) -> bool:
        return tool in ROLE_TOOLS.get(role, frozenset())

    def review_reserve_usd(self, max_cost_usd: float | None) -> float:
        if max_cost_usd is None:
            return 0.0
        return max(
            0.0,
            min(
                float(self.review_reserve_max_usd),
                float(self.review_reserve_ratio) * float(max_cost_usd),
            ),
        )

    def realism_verdict(self, rating: int) -> str:
        if rating >= self.realism_pass:
            return "pass"
        if rating >= self.realism_acceptable:
            return "acceptable"
        return "redo"

    def review_verdict(
        self,
        rating: int,
        rubric: dict[str, int] | None = None,
        has_blocking_issue: bool = False,
    ) -> str:
        worst = min(
            (
                v
                for k, v in (rubric or {}).items()
                if k in self.gating_rubric_dims and isinstance(v, int)
            ),
            default=None,
        )
        if (
            has_blocking_issue
            or rating < self.realism_acceptable
            or (worst is not None and worst < self.realism_acceptable)
        ):
            return "redo"
        if rating >= self.realism_pass and (worst is None or worst >= self.realism_pass):
            return "pass"
        return "acceptable"

    def review_decision(self, verdict: str) -> Decision:
        if verdict == "redo":
            return Decision.REVISE
        if verdict == "pass":
            return Decision.PROCEED
        return Decision.PROCEED_WITH_LIMITATION

    def budget_exhausted(self, iterations: int) -> bool:
        return iterations >= self.repair_iteration_budget

    def review_made_progress(self, prev: dict | None, cur: dict) -> bool:
        if not prev:
            return True

        def _i(d: dict, k: str) -> int:
            v = d.get(k)
            return v if isinstance(v, int) else -1

        if _i(cur, "rating") > _i(prev, "rating"):
            return True
        if _i(cur, "worst_gating") > _i(prev, "worst_gating"):
            return True
        if _i(cur, "blocking") >= 0 and _i(prev, "blocking") >= 0:
            if _i(cur, "blocking") < _i(prev, "blocking"):
                return True
        return False


DEFAULT_POLICIES = Policies()
