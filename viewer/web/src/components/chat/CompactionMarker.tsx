import { type JSX } from "react";
import { Archive } from "lucide-react";

import type { ConvCompaction } from "@/lib/chat-types";
import { AGENT_META, formatTokens } from "@/components/chat/agent-meta";

export function CompactionMarker({ item }: { item: ConvCompaction }): JSX.Element {
  const applied = item.event_type === "compaction";
  const who = item.agent ? AGENT_META[item.agent].label : "context";
  return (
    <div className="my-1 flex items-center justify-center">
      <div className="flex items-center gap-1.5 rounded-full border border-dashed border-[var(--border-default)] bg-[var(--surface-1)] px-2.5 py-0.5 text-[10px] text-[var(--text-quaternary)]">
        <Archive className="size-3" />
        <span className="font-medium text-[var(--text-tertiary)]">{who} context</span>
        <span>·</span>
        <span>{item.reason}</span>
        {item.count > 1 ? <span>· {item.count} checks</span> : null}
        {item.pressure_ratio != null ? (
          <span>· peak pressure {(item.pressure_ratio * 100).toFixed(0)}%</span>
        ) : null}
        {applied && item.saved_tokens ? (
          <span className="text-emerald-600">· saved ≈{formatTokens(item.saved_tokens)}</span>
        ) : null}
      </div>
    </div>
  );
}
