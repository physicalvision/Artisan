import { type JSX } from "react";
import { useQuery } from "@tanstack/react-query";

import { pipelineRunsQueryOptions } from "@/lib/chat-queries";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const STATUS_DOT: Record<string, string> = {
  running: "bg-sky-500 animate-pulse",
  done: "bg-emerald-500",
  failed: "bg-red-500",
};

export function RunSelector({
  runId,
  onSelect,
}: {
  runId: string | null;
  onSelect: (runId: string) => void;
}): JSX.Element {
  const { data: runs } = useQuery(pipelineRunsQueryOptions());

  return (
    <Select value={runId ?? undefined} onValueChange={onSelect}>
      <SelectTrigger className="h-7 w-[280px] text-[12px]">
        <SelectValue placeholder="Select a run…" />
      </SelectTrigger>
      <SelectContent className="max-h-[60vh]">
        {(runs ?? []).map((run) => (
          <SelectItem key={run.run_id} value={run.run_id} className="text-[12px]">
            <span className="flex items-center gap-2">
              <span
                className={`size-1.5 shrink-0 rounded-full ${
                  run.status === "running" && !run.is_active
                    ? "bg-slate-400" // interrupted/stale run: gray, no pulse
                    : (STATUS_DOT[run.status ?? ""] ?? "bg-slate-400")
                }`}
              />
              <span className="font-mono">{run.run_id}</span>
              {run.last_realism_rating != null ? (
                <span className="text-[10px] text-[var(--text-quaternary)]">★{run.last_realism_rating}</span>
              ) : null}
            </span>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
