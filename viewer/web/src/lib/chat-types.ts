// TypeScript mirror of the backend conversation schema (viewer/api/schemas.py).
// A pipeline run is rendered as one globally-ts-ordered conversation.

export type ConvAgent = "design" | "coding" | "review";

export interface ConvTextPart {
  type: "text";
  text: string;
}

export interface ConvImagePart {
  type: "image";
  src: string; // opaque token -> /api/pipeline/runs/{run}/conversation/image?src=...
  label: string | null;
  detail: string | null;
}

export type ConvPart = ConvTextPart | ConvImagePart;

export interface ConvToolResult {
  parts: ConvPart[];
}

export interface ConvToolCall {
  id: string | null;
  name: string;
  arguments: string; // raw JSON-encoded string
  result: ConvToolResult | null; // null == no result row recorded
}

export interface ConvAgentMessage {
  kind: "agent_message";
  ts: number;
  agent: ConvAgent;
  text: string;
  thinking: string | null;
  tool_calls: ConvToolCall[];
  usage: Record<string, number> | null;
  cost_usd: number | null;
}

export interface ConvCompaction {
  kind: "compaction";
  ts: number;
  agent: ConvAgent | null;
  event_type: string;
  count: number;
  reason: string | null;
  prompt_tokens: number | null;
  pressure_ratio: number | null;
  saved_tokens: number | null;
}

export interface ConvArtifactFile {
  name: string;
  label: string | null;
  src: string;
  size: number | null;
  category: string;
}

export interface ConvArtifacts {
  kind: "artifacts";
  ts: number;
  title: string;
  status: string | null;
  files: ConvArtifactFile[];
}

export interface ConvGateReport {
  kind: "gate_report";
  ts: number;
  phase: string;
  title: string;
  status: string | null;
  summary: string | null;
  columns: string[];
  rows: Array<Record<string, unknown>>;
}

export interface ConvHarnessPrompt {
  kind: "harness_prompt";
  ts: number;
  agent: ConvAgent | null;
  label: string | null;
  parts: ConvPart[];
}

export interface ConvPhaseMarker {
  kind: "phase_marker";
  ts: number;
  phase: string;
  iter: number;
  status: string | null;
  rating: number | null;
  next: string | null;
  note: string | null;
  cost_usd: number | null;
}

export interface ConvReviewReport {
  kind: "review_report";
  ts: number;
  report_id: string | null;
  phase: string | null;
  realism_rating: number | null;
  summary: string | null;
  decision: string | null;
  rubric: Record<string, number>;
  issues: Array<Record<string, unknown>>;
  issue_total: number | null;
}

export type ConvItem =
  | ConvAgentMessage
  | ConvHarnessPrompt
  | ConvPhaseMarker
  | ConvReviewReport
  | ConvCompaction
  | ConvArtifacts
  | ConvGateReport;

export interface ConvAgentCost {
  agent: ConvAgent;
  turns: number;
  input_tokens: number;
  output_tokens: number;
  cached_tokens: number;
  cost_usd: number;
  active_seconds: number | null;
}

export interface ConversationHeader {
  run_id: string;
  status: string | null;
  phase: string | null;
  termination_reason: string | null;
  terminal_details: Record<string, unknown>;
  cost_usd: number | null;
  elapsed_seconds: number | null;
  active_seconds: number | null;
  started_at: number | null;
  last_activity_at: number | null;
  last_realism_rating: number | null;
  prompt: string | null;
  provider: string | null;
  model: string | null;
  is_active: boolean;
}

export interface ConversationResponse {
  header: ConversationHeader;
  items: ConvItem[];
  cost_breakdown: ConvAgentCost[];
  generated_at: string;
}

export interface PipelineRunSummary {
  run_id: string;
  status: string | null;
  phase: string | null;
  termination_reason: string | null;
  terminal_details: Record<string, unknown>;
  prompt: string | null;
  cost_usd: number | null;
  last_realism_rating: number | null;
  updated_at: number | null;
  is_active: boolean;
}
