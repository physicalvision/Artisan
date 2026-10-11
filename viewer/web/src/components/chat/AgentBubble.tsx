import { useState, type JSX } from "react";
import { Brain, ChevronDown, ChevronRight } from "lucide-react";

import type { ConvAgentMessage } from "@/lib/chat-types";
import { AGENT_META, formatClock, formatTokens, formatUsd } from "@/components/chat/agent-meta";
import { ChatMarkdown } from "@/components/chat/Markdown";
import { CodeBlock, prettyMaybeJson } from "@/components/chat/CodeBlock";
import { ToolCallCard } from "@/components/chat/ToolCallCard";

/** An agent turn whose whole body is a JSON document (e.g. the DesignAgent
 * intake brief) renders as a pretty-printed, highlighted block; anything else
 * goes through the markdown renderer. */
function MessageBody({ text }: { text: string }): JSX.Element {
  const pretty = prettyMaybeJson(text);
  if (pretty.language === "json") {
    return <CodeBlock code={pretty.text} language="json" className="max-h-96 overflow-auto" />;
  }
  return <ChatMarkdown>{text}</ChatMarkdown>;
}

function TurnMeta({ message }: { message: ConvAgentMessage }): JSX.Element | null {
  const usage = message.usage ?? {};
  const inTok = Number(usage.prompt_tokens ?? 0);
  const outTok = Number(usage.candidates_tokens ?? usage.completion_tokens ?? 0);
  const cached = Number(usage.cached_tokens ?? 0);
  if (!inTok && !outTok && message.cost_usd == null) return null;
  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 pt-0.5 text-[10px] text-[var(--text-quaternary)]">
      {inTok ? <span>↑ {formatTokens(inTok)}</span> : null}
      {cached ? <span className="text-emerald-600">⚡{formatTokens(cached)} cached</span> : null}
      {outTok ? <span>↓ {formatTokens(outTok)}</span> : null}
      {message.cost_usd != null ? (
        <span className="font-medium text-[var(--text-tertiary)]">{formatUsd(message.cost_usd)}</span>
      ) : null}
    </div>
  );
}

function ThoughtBlock({ text }: { text: string }): JSX.Element {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-md border border-dashed border-[var(--border-subtle)] bg-[var(--surface-1)]">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-1.5 px-2 py-1 text-left text-[11px] font-medium text-[var(--text-quaternary)]"
      >
        <Brain className="size-3" />
        <span>thinking</span>
        {open ? <ChevronDown className="ml-auto size-3" /> : <ChevronRight className="ml-auto size-3" />}
      </button>
      {open ? (
        <div className="whitespace-pre-wrap px-2 pb-2 text-[12px] italic leading-relaxed text-[var(--text-tertiary)]">
          {text}
        </div>
      ) : null}
    </div>
  );
}

export function AgentBubble({ runId, message }: { runId: string; message: ConvAgentMessage }): JSX.Element {
  const meta = AGENT_META[message.agent];
  return (
    <div className="flex justify-start">
      <div className="flex max-w-[82%] gap-2.5">
        <div
          className={`mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full text-[12px] font-bold ${meta.avatarClass}`}
        >
          {meta.avatar}
        </div>
        <div className={`min-w-0 flex-1 rounded-2xl rounded-tl-sm border px-3 py-2 ${meta.cardClass}`}>
          <div className="mb-1.5 flex items-center gap-2">
            <span className={`text-[12px] font-semibold ${meta.nameClass}`}>{meta.label}</span>
            <span className="text-[10px] text-[var(--text-quaternary)]">{formatClock(message.ts)}</span>
          </div>
          <div className="flex flex-col gap-2">
            {message.thinking ? <ThoughtBlock text={message.thinking} /> : null}
            {message.text ? <MessageBody text={message.text} /> : null}
            {message.tool_calls.map((call, idx) => (
              <ToolCallCard key={call.id ?? `${message.ts}-${idx}`} runId={runId} call={call} />
            ))}
          </div>
          <TurnMeta message={message} />
        </div>
      </div>
    </div>
  );
}
