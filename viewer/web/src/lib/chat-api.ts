import type { ConversationResponse, PipelineRunSummary } from "@/lib/chat-types";

async function getJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

export async function fetchPipelineRuns(init?: RequestInit): Promise<PipelineRunSummary[]> {
  return getJson<PipelineRunSummary[]>("/api/pipeline/runs", init);
}

export async function fetchConversation(
  runId: string,
  init?: RequestInit,
): Promise<ConversationResponse> {
  return getJson<ConversationResponse>(
    `/api/pipeline/runs/${encodeURIComponent(runId)}/conversation`,
    init,
  );
}

export function conversationImageUrl(runId: string, token: string): string {
  return `/api/pipeline/runs/${encodeURIComponent(runId)}/conversation/image?src=${encodeURIComponent(token)}`;
}
