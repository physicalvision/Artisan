import { queryOptions } from "@tanstack/react-query";

import { fetchConversation, fetchPipelineRuns } from "@/lib/chat-api";

const RUNS_POLL_MS = 5_000;
const CONVERSATION_POLL_MS = 2_500;

export const chatQueryKeys = {
  root: () => ["chat"] as const,
  runs: () => [...chatQueryKeys.root(), "runs"] as const,
  conversation: (runId: string) => [...chatQueryKeys.root(), "conversation", runId] as const,
};

export function pipelineRunsQueryOptions() {
  return queryOptions({
    queryKey: chatQueryKeys.runs(),
    queryFn: ({ signal }) => fetchPipelineRuns({ signal }),
    // Keep the run list fresh while any run is still executing.
    refetchInterval: (query) =>
      query.state.data?.some((run) => run.is_active) ? RUNS_POLL_MS : false,
    staleTime: 2_000,
  });
}

export function conversationQueryOptions(runId: string | null) {
  return queryOptions({
    queryKey: chatQueryKeys.conversation(runId ?? "__none__"),
    queryFn: ({ signal }) => fetchConversation(runId as string, { signal }),
    enabled: Boolean(runId),
    // Append-only traces are re-read every poll, so new bubbles surface live;
    // polling stops automatically once the run reaches a terminal status.
    refetchInterval: (query) =>
      query.state.data?.header.is_active ? CONVERSATION_POLL_MS : false,
    refetchIntervalInBackground: true,
    placeholderData: (previous) => previous,
    staleTime: 1_000,
  });
}
