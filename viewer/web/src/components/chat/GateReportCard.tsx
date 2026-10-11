import { type JSX } from "react";
import { ShieldCheck } from "lucide-react";

import type { ConvGateReport } from "@/lib/chat-types";
import { Badge, type BadgeProps } from "@/components/ui/badge";

const STATUS_VARIANT: Record<string, BadgeProps["variant"]> = {
  pass: "success",
  redo: "warning",
  fail: "destructive",
};

function cell(value: unknown): string {
  if (value == null || value === "") return "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

export function GateReportCard({ item }: { item: ConvGateReport }): JSX.Element {
  return (
    <div className="mx-auto my-1 w-full max-w-2xl rounded-xl border border-[var(--border-default)] bg-[var(--surface-1)] p-3.5">
      <div className="mb-2 flex items-center gap-2">
        <ShieldCheck className="size-4 text-[var(--text-tertiary)]" />
        <span className="text-[12px] font-semibold text-[var(--text-primary)]">{item.title}</span>
        {item.status ? (
          <Badge variant={STATUS_VARIANT[item.status] ?? "secondary"} className="text-[10px]">
            {item.status}
          </Badge>
        ) : null}
        <span className="ml-auto text-[10px] text-[var(--text-quaternary)]">{item.rows.length} rows</span>
      </div>
      {item.summary ? (
        <p className="mb-2 text-[11px] text-[var(--text-tertiary)]">{item.summary}</p>
      ) : null}
      {item.rows.length > 0 ? (
        <div className="max-h-72 overflow-auto rounded-md border border-[var(--border-subtle)]">
          <table className="w-full border-collapse text-[11px]">
            <thead className="sticky top-0 bg-[var(--surface-2)]">
              <tr>
                {item.columns.map((col) => (
                  <th
                    key={col}
                    className="border-b border-[var(--border-subtle)] px-2 py-1 text-left font-semibold text-[var(--text-tertiary)]"
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {item.rows.map((row, idx) => (
                <tr key={idx} className="odd:bg-[var(--surface-0)]">
                  {item.columns.map((col) => (
                    <td
                      key={col}
                      className="border-b border-[var(--border-subtle)] px-2 py-1 font-mono text-[var(--text-secondary)]"
                    >
                      {cell(row[col])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );
}
