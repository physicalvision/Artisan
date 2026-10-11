import { useEffect, useRef, useState, type JSX } from "react";
import { useQuery } from "@tanstack/react-query";
import { Loader2, MessagesSquare } from "lucide-react";

import { conversationQueryOptions, pipelineRunsQueryOptions } from "@/lib/chat-queries";
import { ChatHeader } from "@/components/chat/ChatHeader";
import { ConversationTimeline } from "@/components/chat/ConversationTimeline";

function readRunFromUrl(): string | null {
  return new URLSearchParams(window.location.search).get("run");
}

function writeRunToUrl(runId: string): void {
  const url = new URL(window.location.href);
  url.searchParams.set("run", runId);
  window.history.replaceState(null, "", url.toString());
}

function CenteredNotice({ children }: { children: React.ReactNode }): JSX.Element {
  return (
    <div className="flex min-h-[40vh] flex-col items-center justify-center gap-2 text-[12px] text-[var(--text-quaternary)]">
      {children}
    </div>
  );
}

export default function ChatShell(): JSX.Element {
  const [selectedRun, setSelectedRun] = useState<string | null>(() => readRunFromUrl());

  const runsQuery = useQuery(pipelineRunsQueryOptions());
  const runs = runsQuery.data;

  // Default to the most recent run once the list arrives.
  useEffect(() => {
    if (!selectedRun && runs && runs.length > 0) {
      setSelectedRun(runs[0].run_id);
    }
  }, [selectedRun, runs]);

  const handleSelect = (runId: string): void => {
    setSelectedRun(runId);
    writeRunToUrl(runId);
  };

  const conversationQuery = useQuery(conversationQueryOptions(selectedRun));
  const conversation = conversationQuery.data;

  // Auto-scroll to the newest message while the user is near the bottom.
  const scrollRef = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);
  const itemCount = conversation?.items.length ?? 0;

  const handleScroll = (): void => {
    const el = scrollRef.current;
    if (!el) return;
    stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 140;
  };

  useEffect(() => {
    const el = scrollRef.current;
    if (el && stickToBottom.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [itemCount, selectedRun]);

  const isActive = conversation?.header.is_active ?? false;

  return (
    <div className="flex h-full min-h-0 flex-col">
      <ChatHeader
        runId={selectedRun}
        header={conversation?.header ?? null}
        costBreakdown={conversation?.cost_breakdown ?? []}
        onSelectRun={handleSelect}
        isFetching={conversationQuery.isFetching && !conversationQuery.isLoading}
      />
      <div ref={scrollRef} onScroll={handleScroll} className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-3xl px-4 py-6">
          {!selectedRun ? (
            <CenteredNotice>
              <MessagesSquare className="size-5" />
              {runsQuery.isLoading ? "Loading runs…" : "Select a pipeline run to view its conversation."}
            </CenteredNotice>
          ) : conversationQuery.isLoading ? (
            <CenteredNotice>
              <Loader2 className="size-5 animate-spin" />
              Loading conversation…
            </CenteredNotice>
          ) : conversationQuery.isError ? (
            <CenteredNotice>
              <span className="text-[var(--destructive)]">Failed to load run “{selectedRun}”.</span>
            </CenteredNotice>
          ) : conversation && conversation.items.length > 0 ? (
            <>
              <ConversationTimeline runId={selectedRun} items={conversation.items} />
              {isActive ? (
                <div className="mt-4 flex items-center justify-center gap-2 text-[11px] text-[var(--text-quaternary)]">
                  <Loader2 className="size-3 animate-spin" />
                  Run in progress — updating live…
                </div>
              ) : null}
            </>
          ) : (
            <CenteredNotice>No conversation data recorded for this run yet.</CenteredNotice>
          )}
        </div>
      </div>
    </div>
  );
}
