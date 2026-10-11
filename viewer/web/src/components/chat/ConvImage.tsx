import { useState, type JSX } from "react";

import { conversationImageUrl } from "@/lib/chat-api";
import type { ConvImagePart } from "@/lib/chat-types";
import {
  Dialog,
  DialogContent,
  DialogTitle,
} from "@/components/ui/dialog";

export function ConvImage({
  runId,
  part,
  size = "md",
}: {
  runId: string;
  part: ConvImagePart;
  size?: "md" | "lg";
}): JSX.Element {
  const [open, setOpen] = useState(false);
  const url = conversationImageUrl(runId, part.src);
  const dim = size === "lg" ? "h-40 w-40" : "h-28 w-28";

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="group relative overflow-hidden rounded-md border border-[var(--border-default)] bg-[var(--surface-1)] transition hover:border-[var(--accent)]"
        title={part.label ?? "image"}
      >
        <img
          src={url}
          loading="lazy"
          alt={part.label ?? "render"}
          className={`${dim} object-cover transition group-hover:opacity-90`}
        />
        {part.label ? (
          <span className="absolute inset-x-0 bottom-0 truncate bg-black/55 px-1.5 py-0.5 text-[10px] font-medium text-white">
            {part.label}
          </span>
        ) : null}
      </button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-w-5xl border-[var(--border-default)] bg-[var(--surface-0)] p-3">
          <DialogTitle className="sr-only">{part.label ?? "Image"}</DialogTitle>
          <img
            src={url}
            alt={part.label ?? "render"}
            className="mx-auto max-h-[82vh] w-auto rounded-md object-contain"
          />
          {part.label ? (
            <p className="mt-2 text-center text-[12px] text-[var(--text-tertiary)]">{part.label}</p>
          ) : null}
        </DialogContent>
      </Dialog>
    </>
  );
}

export function ConvImageGrid({
  runId,
  parts,
  size = "md",
}: {
  runId: string;
  parts: ConvImagePart[];
  size?: "md" | "lg";
}): JSX.Element | null {
  if (parts.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-2">
      {parts.map((part, idx) => (
        <ConvImage key={`${part.src}-${idx}`} runId={runId} part={part} size={size} />
      ))}
    </div>
  );
}
