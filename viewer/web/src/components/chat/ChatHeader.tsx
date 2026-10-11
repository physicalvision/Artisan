import { type JSX } from "react";
import { ChevronLeft, RefreshCw } from "lucide-react";

import { navigateTo } from "@/lib/router";
import { formatDuration } from "@/lib/utils";
import type { ConvAgentCost, ConversationHeader } from "@/lib/chat-types";
import { Badge, type BadgeProps } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { RunSelector } from "@/components/chat/RunSelector";
import { AGENT_META, formatTokens, formatUsd } from "@/components/chat/agent-meta";

const STATUS_VARIANT: Record<string, BadgeProps["variant"]> = {
  running: "default",
  done: "success",
  failed: "destructive",
};

function CostBreakdownPopover({
  costUsd,
  breakdown,
}: {
  costUsd: number;
  breakdown: ConvAgentCost[];
}): JSX.Element {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          className="rounded px-1 text-[11px] text-[var(--text-tertiary)] underline-offset-2 hover:underline"
          title="Cost by agent"
        >
          {formatUsd(costUsd)}
        </button>
      </PopoverTrigger>
      <PopoverContent className="w-80 p-2 text-[11px]">
        <div className="mb-1 font-semibold text-[var(--text-secondary)]">Cost &amp; time by agent</div>
        <table className="w-full border-collapse">
          <thead>
            <tr className="text-[10px] text-[var(--text-quaternary)]">
              <th className="text-left font-medium">agent</th>
              <th className="text-right font-medium">turns</th>
              <th className="text-right font-medium">in/out</th>
              <th className="text-right font-medium">time</th>
              <th className="text-right font-medium">cost</th>
            </tr>
          </thead>
          <tbody>
            {breakdown.map((row) => (
              <tr key={row.agent}>
                <td className={`py-0.5 ${AGENT_META[row.agent].nameClass}`}>{AGENT_META[row.agent].label}</td>
                <td className="text-right text-[var(--text-tertiary)]">{row.turns}</td>
                <td className="text-right text-[var(--text-quaternary)]">
                  {formatTokens(row.input_tokens)}/{formatTokens(row.output_tokens)}
                </td>
                <td className="text-right text-[var(--text-tertiary)]">{formatDuration(row.active_seconds)}</td>
                <td className="text-right font-medium text-[var(--text-secondary)]">{formatUsd(row.cost_usd)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </PopoverContent>
    </Popover>
  );
}

export function ChatHeader({
  runId,
  header,
  costBreakdown,
  onSelectRun,
  isFetching,
}: {
  runId: string | null;
  header: ConversationHeader | null;
  costBreakdown: ConvAgentCost[];
  onSelectRun: (runId: string) => void;
  isFetching: boolean;
}): JSX.Element {
  const status = header?.status ?? null;
  const isActive = header?.is_active ?? false;
  const interrupted = status === "running" && !isActive; // server still says running, but no recent activity
  // Use recorded activity so elapsed time stops advancing for interrupted runs.
  const startedAt = header?.started_at ?? null;
  const lastActivity = header?.last_activity_at ?? null;
  const finalElapsed = header?.elapsed_seconds ?? header?.active_seconds ?? null;
  const activityElapsed =
    startedAt != null && lastActivity != null ? Math.max(0, lastActivity - startedAt) : null;
  const elapsedSeconds =
    status === "running" ? (activityElapsed ?? finalElapsed) : (finalElapsed ?? activityElapsed);
  return (
    <header className="flex h-11 shrink-0 items-center gap-3 border-b border-[var(--border-default)] bg-[var(--surface-0)] px-4">
      <div className="flex items-center gap-2 text-[12px]">
        <span className="font-semibold tracking-[-0.02em] text-[var(--text-primary)]">Artisan</span>
        <span className="text-[var(--border-strong)]">/</span>
        <span className="text-[var(--text-tertiary)]">Chat</span>
      </div>

      <RunSelector runId={runId} onSelect={onSelectRun} />

      <div className="flex min-w-0 flex-1 items-center gap-2">
        {status ? (
          <Badge
            variant={interrupted ? "secondary" : (STATUS_VARIANT[status] ?? "secondary")}
            className="text-[10px]"
          >
            {status === "running" && isActive ? (
              <span className="mr-1 inline-block size-1.5 animate-pulse rounded-full bg-current align-middle" />
            ) : null}
            {interrupted ? "interrupted" : status}
          </Badge>
        ) : null}
        {header?.phase ? (
          <span className="text-[11px] text-[var(--text-tertiary)]">{header.phase.replace(/_/g, " ")}</span>
        ) : null}
        {header?.model ? (
          <span className="rounded bg-[var(--surface-2)] px-1.5 text-[10px] font-medium text-[var(--text-tertiary)]">
            {header.model}
          </span>
        ) : null}
        {header?.last_realism_rating != null ? (
          <span className="text-[11px] text-[var(--text-tertiary)]">★ {header.last_realism_rating}</span>
        ) : null}
        {header?.cost_usd != null ? (
          costBreakdown.length > 0 ? (
            <CostBreakdownPopover costUsd={header.cost_usd} breakdown={costBreakdown} />
          ) : (
            <span className="text-[11px] text-[var(--text-tertiary)]">{formatUsd(header.cost_usd)}</span>
          )
        ) : null}
        {elapsedSeconds != null ? (
          <span
            className="text-[11px] text-[var(--text-tertiary)]"
            title={
              interrupted
                ? "Run time before it stopped"
                : status === "running"
                  ? "Elapsed (to last activity)"
                  : "Total run time"
            }
          >
            {formatDuration(elapsedSeconds)}
          </span>
        ) : null}
        {isFetching ? (
          <RefreshCw className="size-3 animate-spin text-[var(--text-quaternary)]" />
        ) : null}
      </div>

      <div className="flex items-center gap-1">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => navigateTo({ page: "viewer" })}
          className="h-7 gap-1 rounded-md px-2 text-[11px] text-[var(--text-tertiary)] hover:text-[var(--text-secondary)]"
        >
          <ChevronLeft className="size-3" />
          Results
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => navigateTo({ page: "viewer" })}
          className="h-7 gap-1 rounded-md px-2 text-[11px] text-[var(--text-tertiary)] hover:text-[var(--text-secondary)]"
        >
          Viewer
        </Button>
      </div>
    </header>
  );
}
