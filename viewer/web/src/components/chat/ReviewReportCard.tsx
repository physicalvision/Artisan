import { type JSX } from "react";
import { ClipboardCheck } from "lucide-react";

import type { ConvReviewReport } from "@/lib/chat-types";
import { Badge, type BadgeProps } from "@/components/ui/badge";

const DECISION_VARIANT: Record<string, BadgeProps["variant"]> = {
  proceed: "success",
  revise: "warning",
  proceed_with_documented_limitation: "secondary",
};

const SEVERITY_VARIANT: Record<string, BadgeProps["variant"]> = {
  blocking: "destructive",
  major: "warning",
  minor: "secondary",
};

function str(value: unknown): string {
  return typeof value === "string" ? value : value == null ? "" : String(value);
}

function RubricBar({ name, value }: { name: string; value: number }): JSX.Element {
  const pct = Math.max(0, Math.min(100, value * 10));
  return (
    <div className="flex items-center gap-2">
      <span className="w-36 shrink-0 truncate text-[11px] text-[var(--text-tertiary)]">
        {name.replace(/_/g, " ")}
      </span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-[var(--surface-2)]">
        <div className="h-full rounded-full bg-amber-400" style={{ width: `${pct}%` }} />
      </div>
      <span className="w-8 shrink-0 text-right text-[11px] font-medium text-[var(--text-secondary)]">
        {value}
      </span>
    </div>
  );
}

export function ReviewReportCard({ report }: { report: ConvReviewReport }): JSX.Element {
  const decisionVariant = DECISION_VARIANT[report.decision ?? ""] ?? "secondary";
  const rubric = Object.entries(report.rubric ?? {});

  return (
    <div className="mx-auto my-1 w-full max-w-2xl rounded-xl border border-amber-200 bg-amber-50/60 p-3.5">
      <div className="flex items-center gap-2">
        <ClipboardCheck className="size-4 text-amber-600" />
        <span className="text-[12px] font-semibold text-amber-800">ReviewAgent verdict</span>
        <div className="ml-auto flex items-center gap-2">
          {report.decision ? (
            <Badge variant={decisionVariant} className="text-[10px]">
              {report.decision.replace(/_/g, " ")}
            </Badge>
          ) : null}
          {report.realism_rating != null ? (
            <span className="text-[13px] font-bold text-amber-800">{report.realism_rating}/10</span>
          ) : null}
        </div>
      </div>

      {report.summary ? (
        <p className="mt-2 text-[12px] leading-relaxed text-[var(--text-secondary)]">{report.summary}</p>
      ) : null}

      {rubric.length > 0 ? (
        <div className="mt-3 flex flex-col gap-1">
          {rubric.map(([name, value]) => (
            <RubricBar key={name} name={name} value={Number(value)} />
          ))}
        </div>
      ) : null}

      {report.issues.length > 0 ? (
        <div className="mt-3 flex flex-col gap-2">
          <span className="text-[10px] font-semibold uppercase tracking-wide text-[var(--text-quaternary)]">
            issues ({report.issue_total ?? report.issues.length})
          </span>
          {report.issues.map((issue, idx) => {
            const severity = str(issue.severity);
            return (
              <div
                key={str(issue.issue_id) || idx}
                className="rounded-md border border-[var(--border-subtle)] bg-[var(--surface-0)] p-2"
              >
                <div className="flex items-center gap-2">
                  {severity ? (
                    <Badge variant={SEVERITY_VARIANT[severity] ?? "secondary"} className="text-[10px]">
                      {severity}
                    </Badge>
                  ) : null}
                  <span className="truncate text-[11px] font-medium text-[var(--text-primary)]">
                    {str(issue.target)}
                  </span>
                  {str(issue.route_to) ? (
                    <span className="ml-auto text-[10px] text-[var(--text-quaternary)]">
                      → {str(issue.route_to)}
                    </span>
                  ) : null}
                </div>
                {str(issue.problem) ? (
                  <p className="mt-1 text-[11px] leading-relaxed text-[var(--text-tertiary)]">
                    {str(issue.problem)}
                  </p>
                ) : null}
                {str(issue.required_change) ? (
                  <p className="mt-1 text-[11px] leading-relaxed text-[var(--text-secondary)]">
                    <span className="font-medium">Fix: </span>
                    {str(issue.required_change)}
                  </p>
                ) : null}
              </div>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}
