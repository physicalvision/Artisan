import type { ConvAgent } from "@/lib/chat-types";

export interface AgentMeta {
  label: string;
  avatar: string;
  // Static Tailwind class strings (kept whole so the JIT can see them).
  avatarClass: string;
  nameClass: string;
  cardClass: string;
  railClass: string;
}

export const AGENT_META: Record<ConvAgent, AgentMeta> = {
  design: {
    label: "DesignAgent",
    avatar: "D",
    avatarClass: "bg-violet-500 text-white",
    nameClass: "text-violet-700",
    cardClass: "border-violet-200 bg-violet-50/50",
    railClass: "bg-violet-400",
  },
  coding: {
    label: "CodingAgent",
    avatar: "C",
    avatarClass: "bg-emerald-500 text-white",
    nameClass: "text-emerald-700",
    cardClass: "border-emerald-200 bg-emerald-50/50",
    railClass: "bg-emerald-400",
  },
  review: {
    label: "ReviewAgent",
    avatar: "R",
    avatarClass: "bg-amber-500 text-white",
    nameClass: "text-amber-700",
    cardClass: "border-amber-200 bg-amber-50/50",
    railClass: "bg-amber-400",
  },
};

export function formatUsd(value: number | null | undefined): string {
  if (value == null) return "—";
  if (value === 0) return "$0";
  if (value < 1) return `$${value.toFixed(4)}`;
  return `$${value.toFixed(2)}`;
}

export function formatTokens(value: number | null | undefined): string {
  if (value == null) return "—";
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return String(value);
}

export function formatBytes(value: number | null | undefined): string {
  if (value == null) return "—";
  if (value >= 1_048_576) return `${(value / 1_048_576).toFixed(1)} MB`;
  if (value >= 1024) return `${(value / 1024).toFixed(0)} KB`;
  return `${value} B`;
}

export function formatClock(ts: number): string {
  if (!ts || ts <= 0) return "";
  try {
    return new Date(ts * 1000).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return "";
  }
}
