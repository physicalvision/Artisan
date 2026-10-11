import { type JSX } from "react";

import type { ConvItem } from "@/lib/chat-types";
import { AgentBubble } from "@/components/chat/AgentBubble";
import { HarnessBubble } from "@/components/chat/HarnessBubble";
import { PhaseDivider } from "@/components/chat/PhaseDivider";
import { ReviewReportCard } from "@/components/chat/ReviewReportCard";
import { CompactionMarker } from "@/components/chat/CompactionMarker";
import { ArtifactsCard } from "@/components/chat/ArtifactsCard";
import { GateReportCard } from "@/components/chat/GateReportCard";

export function ConversationTimeline({
  runId,
  items,
}: {
  runId: string;
  items: ConvItem[];
}): JSX.Element {
  return (
    <div className="flex flex-col gap-3">
      {items.map((item, idx) => {
        const key = `${item.kind}-${idx}-${item.ts}`;
        switch (item.kind) {
          case "phase_marker":
            return <PhaseDivider key={key} marker={item} />;
          case "review_report":
            return <ReviewReportCard key={key} report={item} />;
          case "harness_prompt":
            return <HarnessBubble key={key} runId={runId} prompt={item} />;
          case "agent_message":
            return <AgentBubble key={key} runId={runId} message={item} />;
          case "compaction":
            return <CompactionMarker key={key} item={item} />;
          case "artifacts":
            return <ArtifactsCard key={key} runId={runId} item={item} />;
          case "gate_report":
            return <GateReportCard key={key} item={item} />;
          default:
            return null;
        }
      })}
    </div>
  );
}
