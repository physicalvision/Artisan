import { type JSX } from "react";

import type { ConvPhaseMarker } from "@/lib/chat-types";

const STATUS_TONE: Record<string, string> = {
  pass: "border-emerald-200 bg-emerald-50 text-emerald-700",
  redo: "border-amber-200 bg-amber-50 text-amber-700",
  fail: "border-red-200 bg-red-50 text-red-700",
};

function prettyPhase(phase: string): string {
  const match = phase.match(/^(P\d+)_(.*)$/);
  if (match) {
    return `${match[1]} · ${match[2].replace(/_/g, " ")}`;
  }
  return phase.replace(/_/g, " ");
}

export function PhaseDivider({ marker }: { marker: ConvPhaseMarker }): JSX.Element {
  const tone =
    STATUS_TONE[marker.status ?? ""] ??
    "border-[var(--border-default)] bg-[var(--surface-1)] text-[var(--text-tertiary)]";

  return (
    <div className="my-3 flex items-center gap-3">
      <div className="h-px flex-1 bg-[var(--border-subtle)]" />
      <div className={`flex items-center gap-2 rounded-full border px-3 py-1 text-[11px] font-medium ${tone}`}>
        <span className="font-semibold">{prettyPhase(marker.phase)}</span>
        {marker.iter > 0 ? <span className="opacity-70">iter {marker.iter}</span> : null}
        {marker.status ? <span className="uppercase tracking-wide opacity-80">{marker.status}</span> : null}
        {marker.rating != null ? <span>★ {marker.rating}</span> : null}
        {marker.cost_usd != null ? (
          <span className="opacity-70">${marker.cost_usd.toFixed(2)}</span>
        ) : null}
      </div>
      <div className="h-px flex-1 bg-[var(--border-subtle)]" />
    </div>
  );
}
