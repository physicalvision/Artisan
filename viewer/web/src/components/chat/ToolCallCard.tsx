import { useState, type JSX } from "react";
import { ChevronDown, ChevronRight, Wrench } from "lucide-react";

import type { ConvImagePart, ConvToolCall } from "@/lib/chat-types";
import { Badge, type BadgeProps } from "@/components/ui/badge";
import { CodeBlock, prettyMaybeJson } from "@/components/chat/CodeBlock";
import { ConvImageGrid } from "@/components/chat/ConvImage";

const STATUS_TOOLS = new Set(["compile_model", "run_tests", "write_test", "edit_test"]);

function summarizeArgs(args: string): string {
  if (!args) return "";
  try {
    const obj = JSON.parse(args);
    if (obj && typeof obj === "object" && !Array.isArray(obj)) {
      const chunks: string[] = [];
      for (const [key, value] of Object.entries(obj)) {
        if (typeof value === "string") {
          const tv = value.trim();
          if (tv.startsWith("{") || tv.startsWith("[")) {
            // Nested JSON payload (e.g. write_blueprint content) — don't dump it.
            chunks.push(`${key}=${tv.startsWith("[") ? "[…]" : "{…}"}`);
          } else {
            const short = value.length > 36 ? `${value.slice(0, 36)}…` : value;
            chunks.push(`${key}=${JSON.stringify(short)}`);
          }
        } else if (typeof value === "number" || typeof value === "boolean") {
          chunks.push(`${key}=${value}`);
        } else {
          chunks.push(`${key}=…`);
        }
        if (chunks.length >= 3) break;
      }
      return chunks.join(", ");
    }
  } catch {
    /* not JSON */
  }
  return args.length > 56 ? `${args.slice(0, 56)}…` : args;
}

// Match explicit failures; a zero failure count must remain a success.
const FAILURE_PATTERNS: RegExp[] = [
  /\bfailures?\s*[=:]\s*[1-9]/, // failures=3, failure: 1 (non-zero only)
  /\bok\s*[=:]\s*false\b/i, //    compact verdict "ok=false" / "ok: false" (non-JSON)
  /status\s*[=:]\s*fail/i, //     status=fail / status: failed
  /compile\s+failure/i, //        "COMPILE FAILURE: ..." header
  /\berror\b/i, //                "ERROR: ..." / leading error ack
  /traceback/i, //               Python traceback dump
  /exception/i, //               raised exception text
  /❌/,
];

// A tool result that is a JSON object with a top-level boolean `ok` carries its
// own pass/fail verdict — trust it directly. Returns null when the text is not
// such an object (e.g. compile_model's plain "COMPILE SUCCESS" header).
function jsonOk(text: string): boolean | null {
  const t = text.trim();
  if (!t.startsWith("{")) return null;
  try {
    const obj = JSON.parse(t);
    if (obj && typeof obj === "object" && typeof obj.ok === "boolean") return obj.ok;
  } catch {
    /* not JSON */
  }
  return null;
}

function resultStatus(name: string, text: string): { label: string; variant: BadgeProps["variant"] } | null {
  if (!STATUS_TOOLS.has(name)) return null;
  // Structured verdict wins: run_tests / write_test / edit_test return JSON with a
  // top-level boolean `ok`. The text heuristics below can't be trusted on these —
  // the `\bok\b` success pattern matches the JSON KEY "ok" regardless of its value,
  // so a failed suite ({"ok": false, "failures": [...]}) would mis-read as a pass.
  const verdict = jsonOk(text);
  if (verdict !== null) {
    return verdict
      ? { label: "ok", variant: "success" }
      : { label: "failed", variant: "destructive" };
  }
  const lower = text.toLowerCase();
  if (FAILURE_PATTERNS.some((re) => re.test(text))) {
    return { label: "failed", variant: "destructive" };
  }
  if (/\bok\b|pass|compiled|success|green|✓/.test(lower)) {
    return { label: "ok", variant: "success" };
  }
  return null;
}

export function ToolCallCard({ runId, call }: { runId: string; call: ConvToolCall }): JSX.Element {
  const [argsOpen, setArgsOpen] = useState(false);
  const [resultOpen, setResultOpen] = useState(false);

  const parts = call.result?.parts ?? [];
  const images = parts.filter((p): p is ConvImagePart => p.type === "image");
  const resultText = parts
    .filter((p) => p.type === "text")
    .map((p) => p.text)
    .join("\n")
    .trim();

  const status = resultText ? resultStatus(call.name, resultText) : null;
  const pretty = resultText ? prettyMaybeJson(resultText) : null;
  const firstLine = resultText.split("\n").find((line) => line.trim()) ?? "";
  const argsPretty = call.arguments ? prettyMaybeJson(call.arguments) : null;
  const hasArgs = Boolean(call.arguments && call.arguments.trim() && call.arguments.trim() !== "{}");

  // JSON results render formatted + highlighted by default; short text inlines;
  // only long plain-text dumps (file reads, compiler logs) stay collapsed.
  const isJson = pretty?.language === "json";
  const isShortPlain =
    Boolean(resultText) && !isJson && resultText.length <= 200 && resultText.split("\n").length <= 2;
  const isLongPlain = Boolean(resultText) && !isJson && !isShortPlain;

  return (
    <div className="rounded-md border border-[var(--border-subtle)] bg-[var(--surface-0)]">
      <button
        type="button"
        onClick={() => setArgsOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-2 py-1.5 text-left"
        disabled={!hasArgs}
      >
        <Wrench className="size-3 shrink-0 text-[var(--text-quaternary)]" />
        <code className="shrink-0 font-mono text-[12px] font-semibold text-[var(--text-primary)]">
          {call.name || "tool"}
        </code>
        {summarizeArgs(call.arguments) ? (
          <span className="truncate font-mono text-[11px] text-[var(--text-tertiary)]">
            {summarizeArgs(call.arguments)}
          </span>
        ) : null}
        {hasArgs ? (
          <span className="ml-auto shrink-0 text-[var(--text-quaternary)]">
            {argsOpen ? <ChevronDown className="size-3" /> : <ChevronRight className="size-3" />}
          </span>
        ) : null}
      </button>

      {argsOpen && argsPretty ? (
        <div className="px-2 pb-2">
          <CodeBlock code={argsPretty.text} language={argsPretty.language} />
        </div>
      ) : null}

      {(images.length > 0 || resultText || call.result === null) && (
        <div className="border-t border-[var(--border-subtle)] px-2 py-1.5">
          {call.result === null ? (
            <span className="text-[11px] italic text-[var(--text-quaternary)]">no result recorded</span>
          ) : (
            <div className="flex flex-col gap-1.5">
              {(status || resultText) && (
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-semibold uppercase tracking-wide text-[var(--text-quaternary)]">
                    result
                  </span>
                  {isJson ? (
                    <span className="rounded bg-[var(--surface-2)] px-1 text-[9px] font-medium text-[var(--text-quaternary)]">
                      json
                    </span>
                  ) : null}
                  {status ? (
                    <Badge variant={status.variant} className="px-1.5 py-0 text-[10px]">
                      {status.label}
                    </Badge>
                  ) : null}
                  {isLongPlain ? (
                    <button
                      type="button"
                      onClick={() => setResultOpen((v) => !v)}
                      className="ml-auto flex items-center gap-1 text-[11px] text-[var(--text-tertiary)] hover:text-[var(--text-secondary)]"
                    >
                      {resultOpen ? "hide" : "show"}
                      {resultOpen ? <ChevronDown className="size-3" /> : <ChevronRight className="size-3" />}
                    </button>
                  ) : null}
                </div>
              )}
              {resultText && isJson && pretty ? (
                <CodeBlock code={pretty.text} language="json" className="max-h-80 overflow-auto" />
              ) : null}
              {resultText && isShortPlain ? (
                <p className="whitespace-pre-wrap break-words font-mono text-[11px] text-[var(--text-tertiary)]">
                  {resultText}
                </p>
              ) : null}
              {resultText && isLongPlain && !resultOpen ? (
                <p className="truncate font-mono text-[11px] text-[var(--text-tertiary)]">{firstLine}</p>
              ) : null}
              {resultText && isLongPlain && resultOpen && pretty ? (
                <CodeBlock code={pretty.text} language={pretty.language} className="max-h-80 overflow-auto" />
              ) : null}
              <ConvImageGrid runId={runId} parts={images} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
