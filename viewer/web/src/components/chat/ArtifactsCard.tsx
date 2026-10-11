import { type JSX } from "react";
import { Box, Download, FileCode, FileJson, Boxes } from "lucide-react";

import { conversationImageUrl } from "@/lib/chat-api";
import type { ConvArtifactFile, ConvArtifacts } from "@/lib/chat-types";
import { formatBytes } from "@/components/chat/agent-meta";

const CATEGORY_LABEL: Record<string, string> = {
  code: "model.py",
  blueprint: "blueprint",
  qa: "qa tests",
  urdf: "URDF",
  mjcf: "MJCF",
  usd: "USD",
  mesh: "mesh / GLB",
  other: "file",
};

function iconFor(category: string): JSX.Element {
  if (category === "code" || category === "qa") return <FileCode className="size-3.5" />;
  if (category === "blueprint") return <FileJson className="size-3.5" />;
  if (category === "mesh") return <Boxes className="size-3.5" />;
  return <Box className="size-3.5" />;
}

function ArtifactRow({ runId, file }: { runId: string; file: ConvArtifactFile }): JSX.Element {
  return (
    <a
      href={conversationImageUrl(runId, file.src)}
      download={file.name}
      target="_blank"
      rel="noreferrer"
      className="group flex items-center gap-2 rounded-md border border-[var(--border-subtle)] bg-[var(--surface-0)] px-2 py-1.5 transition hover:border-[var(--accent)]"
    >
      <span className="text-[var(--text-tertiary)]">{iconFor(file.category)}</span>
      <span className="font-mono text-[12px] text-[var(--text-primary)]">
        {file.label ?? file.name}
      </span>
      <span className="rounded bg-[var(--surface-2)] px-1 text-[10px] text-[var(--text-tertiary)]">
        {CATEGORY_LABEL[file.category] ?? file.category}
      </span>
      <span className="ml-auto text-[10px] text-[var(--text-quaternary)]">{formatBytes(file.size)}</span>
      <Download className="size-3 text-[var(--text-quaternary)] opacity-0 transition group-hover:opacity-100" />
    </a>
  );
}

export function ArtifactsCard({ runId, item }: { runId: string; item: ConvArtifacts }): JSX.Element {
  return (
    <div className="mx-auto my-1 w-full max-w-2xl rounded-xl border border-[var(--border-default)] bg-[var(--surface-1)] p-3.5">
      <div className="mb-2 flex items-center gap-2">
        <Box className="size-4 text-[var(--text-tertiary)]" />
        <span className="text-[12px] font-semibold text-[var(--text-primary)]">{item.title}</span>
        {item.status ? (
          <span className="rounded bg-[var(--surface-2)] px-1 text-[10px] uppercase text-[var(--text-tertiary)]">
            {item.status}
          </span>
        ) : null}
        <span className="text-[10px] text-[var(--text-quaternary)]">{item.files.length} files</span>
      </div>
      <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
        {item.files.map((file) => (
          <ArtifactRow key={file.src} runId={runId} file={file} />
        ))}
      </div>
    </div>
  );
}
