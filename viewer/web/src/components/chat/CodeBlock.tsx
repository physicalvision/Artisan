import { type JSX } from "react";

import {
  SyntaxHighlighter,
  codeTheme,
  normalizeHighlightLanguage,
} from "@/components/inspector/syntax-highlighting";

export interface PrettyText {
  text: string;
  language: "json" | "python" | "xml" | undefined;
}

/** Recursively parse string values that are themselves JSON (e.g. tools that
 * pass a payload as a JSON-encoded string like write_blueprint's `content`), so
 * the rendered tree shows real structure instead of an escaped `\n \"…` string. */
function deepUnwrapJson(value: unknown): unknown {
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (trimmed.length > 2 && (trimmed.startsWith("{") || trimmed.startsWith("["))) {
      try {
        return deepUnwrapJson(JSON.parse(trimmed));
      } catch {
        return value;
      }
    }
    return value;
  }
  if (Array.isArray(value)) {
    return value.map(deepUnwrapJson);
  }
  if (value && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [key, val] of Object.entries(value)) {
      out[key] = deepUnwrapJson(val);
    }
    return out;
  }
  return value;
}

/** Pretty-print a string as JSON when possible; otherwise return it verbatim. */
export function prettyMaybeJson(raw: string): PrettyText {
  const trimmed = raw.trim();
  if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
    try {
      return { text: JSON.stringify(deepUnwrapJson(JSON.parse(trimmed)), null, 2), language: "json" };
    } catch {
      // fall through
    }
  }
  return { text: raw, language: undefined };
}

export function CodeBlock({
  code,
  language,
  className,
}: {
  code: string;
  language?: string;
  className?: string;
}): JSX.Element {
  const lang = normalizeHighlightLanguage(language);
  const base =
    "overflow-x-auto rounded-md border border-[var(--border-subtle)] bg-[var(--surface-1)] p-2 text-[12px] leading-relaxed";
  if (!lang) {
    return (
      <pre className={`${base} whitespace-pre-wrap break-words font-mono text-[var(--text-secondary)] ${className ?? ""}`}>
        {code}
      </pre>
    );
  }
  return (
    <div className={`${base} ${className ?? ""}`}>
      <SyntaxHighlighter
        language={lang}
        style={codeTheme}
        customStyle={{ background: "transparent", margin: 0, padding: 0, fontSize: "12px" }}
        wrapLongLines
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
}
