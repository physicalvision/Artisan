import { useState, type JSX } from "react";

import type { ConvHarnessPrompt, ConvImagePart } from "@/lib/chat-types";
import { formatClock } from "@/components/chat/agent-meta";
import { ChatMarkdown } from "@/components/chat/Markdown";
import { ConvImageGrid } from "@/components/chat/ConvImage";

const COLLAPSE_CHARS = 700;

export function HarnessBubble({ runId, prompt }: { runId: string; prompt: ConvHarnessPrompt }): JSX.Element {
  const [expanded, setExpanded] = useState(false);
  const text = prompt.parts
    .filter((p) => p.type === "text")
    .map((p) => p.text)
    .join("\n\n")
    .trim();
  const images = prompt.parts.filter((p): p is ConvImagePart => p.type === "image");

  const isLong = text.length > COLLAPSE_CHARS;
  const shown = isLong && !expanded ? `${text.slice(0, COLLAPSE_CHARS)}…` : text;

  return (
    <div className="flex justify-end">
      <div className="flex max-w-[82%] flex-col items-end gap-1">
        <div className="flex items-center gap-2 pr-1">
          <span className="text-[11px] font-semibold text-[var(--text-tertiary)]">
            {prompt.label ?? "User"}
          </span>
          {prompt.agent ? (
            <span className="text-[10px] text-[var(--text-quaternary)]">→ {prompt.agent}</span>
          ) : null}
          <span className="text-[10px] text-[var(--text-quaternary)]">{formatClock(prompt.ts)}</span>
        </div>
        <div className="rounded-2xl rounded-tr-sm border border-slate-200 bg-slate-100 px-3 py-2">
          {text ? (
            <>
              <ChatMarkdown>{shown}</ChatMarkdown>
              {isLong ? (
                <button
                  type="button"
                  onClick={() => setExpanded((v) => !v)}
                  className="mt-1 text-[11px] font-medium text-[var(--accent)] hover:underline"
                >
                  {expanded ? "show less" : "show more"}
                </button>
              ) : null}
            </>
          ) : null}
          {images.length > 0 ? (
            <div className={text ? "mt-2" : ""}>
              <ConvImageGrid runId={runId} parts={images} size="lg" />
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
