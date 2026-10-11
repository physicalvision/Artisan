import { type JSX } from "react";

import type { RecordAgentMeta } from "@/lib/types";
import { formatDuration } from "@/lib/utils";
import { AGENT_META, formatUsd } from "@/components/chat/agent-meta";
import type { ConvAgent } from "@/lib/chat-types";

const KNOWN_AGENTS: readonly string[] = ["design", "coding", "review"];

function agentLabel(agent: string): string {
  if (KNOWN_AGENTS.includes(agent)) return AGENT_META[agent as ConvAgent].label;
  return agent;
}

function agentNameClass(agent: string): string {
  return KNOWN_AGENTS.includes(agent)
    ? AGENT_META[agent as ConvAgent].nameClass
    : "text-[var(--text-secondary)]";
}

export function AgentBreakdownTable({
  agents,
  totalCostUsd,
  totalActiveSeconds,
}: {
  agents: RecordAgentMeta[];
  totalCostUsd?: number | null;
  totalActiveSeconds?: number | null;
}): JSX.Element | null {
  if (agents.length === 0) return null;
  const costSum = agents.reduce((acc, a) => acc + (a.cost_usd ?? 0), 0);
  const timeSum = agents.reduce((acc, a) => acc + (a.active_seconds ?? 0), 0);
  return (
    <table className="w-full border-collapse text-[10px]">
      <thead>
        <tr className="text-[9px] uppercase tracking-[0.04em] text-[var(--text-quaternary)]">
          <th className="text-left font-medium">agent</th>
          <th className="text-left font-medium">model</th>
          <th className="text-right font-medium">turns</th>
          <th className="text-right font-medium">time</th>
          <th className="text-right font-medium">cost</th>
        </tr>
      </thead>
      <tbody>
        {agents.map((agent) => (
          <tr key={agent.agent} className="align-top">
            <td className={`py-0.5 pr-2 font-medium ${agentNameClass(agent.agent)}`}>
              {agentLabel(agent.agent)}
            </td>
            <td className="py-0.5 pr-2 font-mono text-[var(--text-tertiary)]">
              {agent.model ?? agent.provider ?? "--"}
            </td>
            <td className="py-0.5 text-right text-[var(--text-tertiary)]">{agent.turns}</td>
            <td className="py-0.5 text-right text-[var(--text-tertiary)]">
              {formatDuration(agent.active_seconds)}
            </td>
            <td className="py-0.5 text-right font-medium text-[var(--text-secondary)]">
              {formatUsd(agent.cost_usd)}
            </td>
          </tr>
        ))}
        <tr className="border-t border-[var(--border-subtle)] font-semibold">
          <td className="py-0.5 pr-2 text-[var(--text-secondary)]">Total</td>
          <td />
          <td />
          <td className="py-0.5 text-right text-[var(--text-secondary)]">
            {formatDuration(totalActiveSeconds ?? timeSum)}
          </td>
          <td className="py-0.5 text-right text-[var(--text-secondary)]">
            {formatUsd(totalCostUsd ?? costSum)}
          </td>
        </tr>
      </tbody>
    </table>
  );
}
