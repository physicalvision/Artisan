import { type JSX } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

const PROSE_CLASSES = [
  "text-[13px] leading-relaxed text-[var(--text-secondary)]",
  "[&_p]:my-1 [&_p:first-child]:mt-0 [&_p:last-child]:mb-0",
  "[&_ul]:my-1 [&_ul]:list-disc [&_ul]:pl-4",
  "[&_ol]:my-1 [&_ol]:list-decimal [&_ol]:pl-4",
  "[&_li]:my-0.5",
  "[&_code]:rounded [&_code]:bg-[var(--surface-2)] [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-[12px] [&_code]:font-mono",
  "[&_pre]:overflow-x-auto [&_pre]:rounded-md [&_pre]:bg-[var(--surface-2)] [&_pre]:p-2 [&_pre]:text-[12px]",
  "[&_strong]:font-semibold [&_strong]:text-[var(--text-primary)]",
  "[&_h1]:mt-2 [&_h1]:text-[14px] [&_h1]:font-semibold [&_h1]:text-[var(--text-primary)]",
  "[&_h2]:mt-2 [&_h2]:text-[13px] [&_h2]:font-semibold [&_h2]:text-[var(--text-primary)]",
  "[&_h3]:mt-2 [&_h3]:text-[13px] [&_h3]:font-semibold [&_h3]:text-[var(--text-primary)]",
  "[&_a]:text-[var(--accent)] [&_a]:underline",
  "[&_table]:my-1 [&_table]:border-collapse [&_th]:border [&_th]:border-[var(--border-default)] [&_th]:px-1.5 [&_td]:border [&_td]:border-[var(--border-default)] [&_td]:px-1.5",
].join(" ");

export function ChatMarkdown({ children }: { children: string }): JSX.Element {
  return (
    <div className={PROSE_CLASSES}>
      <Markdown remarkPlugins={[remarkGfm]}>{children}</Markdown>
    </div>
  );
}
