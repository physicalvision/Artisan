from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class RecordAgentMeta(BaseModel):

    agent: str
    provider: str | None = None
    model: str | None = None
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float | None = None
    active_seconds: float | None = None


class RecordSummaryResponse(BaseModel):
    record_id: str
    title: str
    prompt_preview: str
    rating: int | None = None
    secondary_rating: int | None = None
    effective_rating: float | None = None
    author: str | None = None
    rated_by: str | None = None
    secondary_rated_by: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    viewer_asset_updated_at: str | None = None
    sdk_package: str | None = None
    provider: str | None = None
    model_id: str | None = None
    has_traces: bool = False
    thinking_level: str | None = None
    turn_count: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_cost_usd: float | None = None
    run_kind: str | None = None
    agents: list[RecordAgentMeta] = Field(default_factory=list)
    total_active_seconds: float | None = None
    elapsed_seconds: float | None = None
    category_slug: str | None = None
    run_id: str | None = None
    run_status: str | None = None
    run_message: str | None = None
    active_revision_id: str | None = None
    origin_record_id: str | None = None
    parent_record_id: str | None = None
    revision_count: int = 0
    has_history: bool = False
    collections: list[str] = Field(default_factory=list)
    materialization_status: str | None = None
    has_compile_report: bool = False
    has_provenance: bool = False
    has_cost: bool = False


class RecordTextFileResponse(BaseModel):
    record_id: str
    file_path: str
    content: str
    truncated: bool = False
    byte_count: int
    preview_byte_limit: int | None = None



ConvAgent = Literal["design", "coding", "review"]


class ConvTextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str = ""


class ConvImagePart(BaseModel):
    type: Literal["image"] = "image"
    src: str
    label: str | None = None
    detail: str | None = None


ConvPart = Annotated[Union[ConvTextPart, ConvImagePart], Field(discriminator="type")]


class ConvToolResult(BaseModel):
    parts: list[ConvPart] = Field(default_factory=list)


class ConvToolCall(BaseModel):
    id: str | None = None
    name: str = ""
    arguments: str = ""
    result: ConvToolResult | None = None


class ConvAgentMessage(BaseModel):
    kind: Literal["agent_message"] = "agent_message"
    ts: float
    agent: ConvAgent
    text: str = ""
    thinking: str | None = None
    tool_calls: list[ConvToolCall] = Field(default_factory=list)
    usage: dict[str, Any] | None = None
    cost_usd: float | None = None


class ConvCompaction(BaseModel):
    kind: Literal["compaction"] = "compaction"
    ts: float
    agent: ConvAgent | None = None
    event_type: str = "compaction"
    count: int = 1
    reason: str | None = None
    prompt_tokens: int | None = None
    pressure_ratio: float | None = None
    saved_tokens: int | None = None


class ConvArtifactFile(BaseModel):
    name: str
    label: str | None = None
    src: str
    size: int | None = None
    category: str = "file"


class ConvArtifacts(BaseModel):
    kind: Literal["artifacts"] = "artifacts"
    ts: float
    title: str = "Final artifacts"
    status: str | None = None
    files: list[ConvArtifactFile] = Field(default_factory=list)


class ConvGateReport(BaseModel):
    kind: Literal["gate_report"] = "gate_report"
    ts: float
    phase: str
    title: str
    status: str | None = None
    summary: str | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)


class ConvHarnessPrompt(BaseModel):
    kind: Literal["harness_prompt"] = "harness_prompt"
    ts: float
    agent: ConvAgent | None = None
    label: str | None = None
    parts: list[ConvPart] = Field(default_factory=list)


class ConvPhaseMarker(BaseModel):
    kind: Literal["phase_marker"] = "phase_marker"
    ts: float
    phase: str
    iter: int = 0
    status: str | None = None
    rating: float | None = None
    next: str | None = None
    note: str | None = None
    cost_usd: float | None = None


class ConvReviewReport(BaseModel):
    kind: Literal["review_report"] = "review_report"
    ts: float
    report_id: str | None = None
    phase: str | None = None
    realism_rating: float | None = None
    summary: str | None = None
    decision: str | None = None
    rubric: dict[str, Any] = Field(default_factory=dict)
    issues: list[dict[str, Any]] = Field(default_factory=list)
    issue_total: int | None = None


ConvItem = Annotated[
    Union[
        ConvAgentMessage,
        ConvHarnessPrompt,
        ConvPhaseMarker,
        ConvReviewReport,
        ConvCompaction,
        ConvArtifacts,
        ConvGateReport,
    ],
    Field(discriminator="kind"),
]


class ConvAgentCost(BaseModel):
    agent: ConvAgent
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    active_seconds: float | None = None


class ConversationHeader(BaseModel):
    run_id: str
    status: str | None = None
    phase: str | None = None
    termination_reason: str | None = None
    terminal_details: dict[str, Any] = Field(default_factory=dict)
    cost_usd: float | None = None
    elapsed_seconds: float | None = None
    active_seconds: float | None = None
    started_at: float | None = None
    last_activity_at: float | None = (
        None
    )
    last_realism_rating: float | None = None
    prompt: str | None = None
    provider: str | None = None
    model: str | None = None
    is_active: bool = False


class ConversationResponse(BaseModel):
    header: ConversationHeader
    items: list[ConvItem] = Field(default_factory=list)
    cost_breakdown: list[ConvAgentCost] = Field(default_factory=list)
    generated_at: str


class PipelineRunSummary(BaseModel):
    run_id: str
    status: str | None = None
    phase: str | None = None
    termination_reason: str | None = None
    terminal_details: dict[str, Any] = Field(default_factory=dict)
    prompt: str | None = None
    cost_usd: float | None = None
    last_realism_rating: float | None = None
    updated_at: float | None = None
    is_active: bool = False
