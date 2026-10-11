import {
  startTransition,
  useCallback,
  useEffect,
  useRef,
  useState,
  type JSX,
} from "react";
import { useQuery } from "@tanstack/react-query";
import * as THREE from "three";

import { updateUrlSearchParams } from "@/lib/url";
import { navigateTo } from "@/lib/router";
import { fetchRecords, fetchRecordSummary, fetchRecordText } from "@/lib/api";
import {
  SceneCanvas,
  type SnapshotExporter,
} from "@/components/viewer3d/SceneCanvas";
import { useJointController } from "@/components/viewer3d/useJointController";
import { useRenderOptions } from "@/components/viewer3d/useRenderOptions";
import type { UrdfJoint, UrdfSpec } from "@/components/viewer3d/urdf-parser";
import { JointPanel } from "@/components/inspector/JointPanel";
import { RenderOptionsPanel } from "@/components/inspector/RenderOptionsPanel";
import { AgentBreakdownTable } from "@/components/inspector/AgentBreakdownTable";
import { CodeBlock } from "@/components/chat/CodeBlock";
import { Button } from "@/components/ui/button";

const JOINT_POSE_QUERY_PARAM = "pose";
const JOINT_POSE_PRECISION = 1e5;
const PREVIEW_UI_SYNC_MS = 100;
const PREVIEW_MIN_CYCLE_SECONDS = 2.6;
const PREVIEW_LINEAR_SPEED_MPS = 0.08;
const PREVIEW_ANGULAR_SPEED_RAD_PER_SECOND = Math.PI / 4;
const PREVIEW_CONTINUOUS_ANGULAR_SPEED_RAD_PER_SECOND = Math.PI / 5;
const PREVIEW_FALLBACK_PRISMATIC_TRAVEL_METERS = 0.24;
const PREVIEW_FALLBACK_REVOLUTE_TRAVEL_RADIANS = (Math.PI * 2) / 3;

type JointPoseSnapshot = {
  recordId: string;
  values: Record<string, number>;
};

type PreviewJointMotion = {
  joint: UrdfJoint;
  cycleSeconds: number;
  phaseOffset: number;
};

function roundJointValue(value: number): number {
  return Math.round(value * JOINT_POSE_PRECISION) / JOINT_POSE_PRECISION;
}

function readJointPoseFromUrl(
  recordId: string | null,
  validJointNames: Set<string>,
): Map<string, number> {
  if (typeof window === "undefined" || !recordId) {
    return new Map();
  }

  try {
    const raw = new URLSearchParams(window.location.search).get(
      JOINT_POSE_QUERY_PARAM,
    );
    if (!raw) {
      return new Map();
    }

    const parsed = JSON.parse(raw) as Partial<JointPoseSnapshot>;
    if (
      parsed.recordId !== recordId ||
      !parsed.values ||
      typeof parsed.values !== "object"
    ) {
      return new Map();
    }

    const nextValues = new Map<string, number>();
    for (const [jointName, value] of Object.entries(parsed.values)) {
      if (
        !validJointNames.has(jointName) ||
        typeof value !== "number" ||
        !Number.isFinite(value)
      ) {
        continue;
      }
      if (Math.abs(value) <= 1 / JOINT_POSE_PRECISION) {
        continue;
      }
      nextValues.set(jointName, roundJointValue(value));
    }

    return nextValues;
  } catch {
    return new Map();
  }
}

function syncJointPoseToUrl(
  recordId: string | null,
  jointValues: Map<string, number>,
): void {
  updateUrlSearchParams((params) => {
    if (!recordId) {
      params.delete(JOINT_POSE_QUERY_PARAM);
      return;
    }

    const values = Object.fromEntries(
      Array.from(jointValues.entries())
        .map(
          ([jointName, value]) => [jointName, roundJointValue(value)] as const,
        )
        .filter(
          ([, value]) =>
            Number.isFinite(value) &&
            Math.abs(value) > 1 / JOINT_POSE_PRECISION,
        ),
    );

    if (Object.keys(values).length === 0) {
      params.delete(JOINT_POSE_QUERY_PARAM);
      return;
    }

    params.set(
      JOINT_POSE_QUERY_PARAM,
      JSON.stringify({
        recordId,
        values,
      } satisfies JointPoseSnapshot),
    );
  });
}

function isPreviewJoint(joint: UrdfJoint): boolean {
  const mimic = (joint as UrdfJoint & { mimic?: unknown }).mimic;
  return (
    !mimic &&
    (joint.type === "revolute" ||
      joint.type === "continuous" ||
      joint.type === "prismatic")
  );
}

function buildRootLinks(joints: UrdfJoint[]): string[] {
  const parents = new Set<string>();
  const children = new Set<string>();

  for (const joint of joints) {
    parents.add(joint.parent);
    children.add(joint.child);
  }

  const roots = Array.from(parents).filter((name) => !children.has(name));
  return roots.length > 0
    ? roots
    : Array.from(new Set(joints.map((joint) => joint.parent)));
}

function buildPreviewJointSequence(urdfSpec: UrdfSpec): UrdfJoint[] {
  const jointsByParent = new Map<string, UrdfJoint[]>();

  for (const joint of urdfSpec.joints) {
    const current = jointsByParent.get(joint.parent) ?? [];
    current.push(joint);
    jointsByParent.set(joint.parent, current);
  }

  const orderedJoints: UrdfJoint[] = [];
  const seenLinks = new Set<string>();

  const visitLink = (linkName: string): void => {
    if (seenLinks.has(linkName)) {
      return;
    }
    seenLinks.add(linkName);

    for (const joint of jointsByParent.get(linkName) ?? []) {
      if (isPreviewJoint(joint)) {
        orderedJoints.push(joint);
      }
      visitLink(joint.child);
    }
  };

  for (const rootLink of buildRootLinks(urdfSpec.joints)) {
    visitLink(rootLink);
  }

  const seenJointNames = new Set(orderedJoints.map((joint) => joint.name));
  for (const joint of urdfSpec.joints) {
    if (isPreviewJoint(joint) && !seenJointNames.has(joint.name)) {
      orderedJoints.push(joint);
    }
  }

  return orderedJoints;
}

function previewJointTravelSpan(joint: UrdfJoint): number {
  if (joint.type === "continuous") {
    return Math.PI * 2;
  }

  const lower = joint.limit?.lower;
  const upper = joint.limit?.upper;
  const hasRange =
    typeof lower === "number" &&
    Number.isFinite(lower) &&
    typeof upper === "number" &&
    Number.isFinite(upper) &&
    upper > lower;

  if (hasRange) {
    return upper - lower;
  }

  if (joint.type === "prismatic") {
    return PREVIEW_FALLBACK_PRISMATIC_TRAVEL_METERS;
  }

  return PREVIEW_FALLBACK_REVOLUTE_TRAVEL_RADIANS;
}

function previewJointCycleSeconds(joint: UrdfJoint): number {
  const span = previewJointTravelSpan(joint);

  if (joint.type === "continuous") {
    return span / PREVIEW_CONTINUOUS_ANGULAR_SPEED_RAD_PER_SECOND;
  }

  const speed =
    joint.type === "prismatic"
      ? PREVIEW_LINEAR_SPEED_MPS
      : PREVIEW_ANGULAR_SPEED_RAD_PER_SECOND;
  return Math.max(PREVIEW_MIN_CYCLE_SECONDS, (span * 2) / speed);
}

function previewJointPhaseOffset(jointName: string, index: number): number {
  let hash = 0;
  for (const character of jointName) {
    hash = (hash * 33 + character.charCodeAt(0)) % 4096;
  }

  return THREE.MathUtils.euclideanModulo(
    hash / 4096 + index * 0.61803398875,
    1,
  );
}

function buildPreviewJointMotions(urdfSpec: UrdfSpec): PreviewJointMotion[] {
  return buildPreviewJointSequence(urdfSpec).map((joint, index) => ({
    joint,
    cycleSeconds: previewJointCycleSeconds(joint),
    phaseOffset: previewJointPhaseOffset(joint.name, index),
  }));
}

function previewJointValue(joint: UrdfJoint, phase: number): number {
  if (joint.type === "continuous") {
    return (
      THREE.MathUtils.euclideanModulo(
        phase * Math.PI * 2 + Math.PI,
        Math.PI * 2,
      ) - Math.PI
    );
  }

  const lower = joint.limit?.lower;
  const upper = joint.limit?.upper;
  const hasRange =
    typeof lower === "number" &&
    Number.isFinite(lower) &&
    typeof upper === "number" &&
    Number.isFinite(upper) &&
    upper > lower;

  if (hasRange) {
    const normalized = 0.5 - 0.5 * Math.cos(phase * Math.PI * 2);
    return THREE.MathUtils.lerp(lower, upper, normalized);
  }

  const wave = Math.sin(phase * Math.PI * 2);

  if (joint.type === "prismatic") {
    return wave * 0.12;
  }

  return wave * (Math.PI / 3);
}

export default function ViewerShell(): JSX.Element {
  const [selectedId, setSelectedId] = useState<string | null>(() =>
    new URLSearchParams(window.location.search).get("record"),
  );
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(0);
  const [tab, setTab] = useState<"inspect" | "render" | "code" | "metadata">(
    "inspect",
  );
  const records = useQuery({
    queryKey: ["records", query, page],
    queryFn: ({ signal }) => fetchRecords(query, page * 100, signal),
    refetchInterval: 5000,
  });
  const summary = useQuery({
    queryKey: ["record", selectedId],
    queryFn: ({ signal }) => fetchRecordSummary(selectedId!, signal),
    enabled: selectedId !== null,
    refetchInterval: 5000,
  });
  const code = useQuery({
    queryKey: ["code", selectedId],
    queryFn: ({ signal }) => fetchRecordText(selectedId!, "model.py", signal),
    enabled: selectedId !== null && tab === "code",
  });
  const metadata = useQuery({
    queryKey: ["metadata", selectedId],
    queryFn: ({ signal }) =>
      fetchRecordText(selectedId!, "record.json", signal),
    enabled: selectedId !== null && tab === "metadata",
  });
  const record = summary.data;
  const { options, setOption } = useRenderOptions();
  const [spec, setSpec] = useState<UrdfSpec | null>(null);
  const [nodes, setNodes] = useState<Map<string, THREE.Object3D> | null>(null);
  const [snapshot, setSnapshot] = useState<SnapshotExporter | null>(null);
  const [previewValues, setPreviewValues] = useState<Map<string, number>>(
    new Map(),
  );
  const { jointValues, setJointValue, applyJointValues, resetAll } =
    useJointController(nodes, spec);
  const manualValues = useRef(jointValues);
  const invalidate = useRef<(() => void) | null>(null);
  const posedSelection = useRef<string | null>(null);

  const selectRecord = useCallback((id: string): void => {
    setSelectedId(id);
    updateUrlSearchParams((params) => {
      params.set("record", id);
      params.delete("pose");
    });
  }, []);

  useEffect(() => {
    if (!selectedId && records.data?.length)
      selectRecord(records.data[0].record_id);
  }, [records.data, selectedId, selectRecord]);

  useEffect(() => {
    manualValues.current = jointValues;
  }, [jointValues]);

  const onSpecChange = useCallback(
    (
      next: UrdfSpec | null,
      nextNodes: Map<string, THREE.Object3D> | null,
    ): void => {
      setSpec(next);
      setNodes(nextNodes);
      posedSelection.current = null;
    },
    [],
  );
  const onSnapshot = useCallback(
    (value: SnapshotExporter | null): void => setSnapshot(() => value),
    [],
  );
  const onInvalidate = useCallback((value: (() => void) | null): void => {
    invalidate.current = value;
  }, []);

  useEffect(() => {
    if (!spec || !selectedId) return;
    const restored = readJointPoseFromUrl(
      selectedId,
      new Set(spec.joints.map((joint) => joint.name)),
    );
    applyJointValues(restored, { commit: true });
    posedSelection.current = selectedId;
  }, [spec, selectedId, applyJointValues]);

  useEffect(() => {
    if (posedSelection.current === selectedId)
      syncJointPoseToUrl(selectedId, jointValues);
  }, [jointValues, selectedId]);

  useEffect(() => {
    if (!spec || !options.autoAnimate) {
      setPreviewValues(new Map());
      applyJointValues(manualValues.current);
      return;
    }
    const motions = buildPreviewJointMotions(spec);
    let frame = 0;
    let lastSync = 0;
    const tick = (now: number): void => {
      const values = new Map<string, number>();
      for (const motion of motions) {
        const phase = THREE.MathUtils.euclideanModulo(
          now / 1000 / motion.cycleSeconds + motion.phaseOffset,
          1,
        );
        values.set(motion.joint.name, previewJointValue(motion.joint, phase));
      }
      applyJointValues(values);
      invalidate.current?.();
      if (now - lastSync >= PREVIEW_UI_SYNC_MS) {
        lastSync = now;
        startTransition(() => setPreviewValues(values));
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(frame);
      applyJointValues(manualValues.current);
      invalidate.current?.();
    };
  }, [spec, options.autoAnimate, applyJointValues]);

  const displayedValues = options.autoAnimate ? previewValues : jointValues;
  return (
    <div className="flex h-screen flex-col bg-[var(--surface-2)] text-[var(--text-primary)]">
      <header className="flex h-12 shrink-0 items-center gap-4 border-b border-[var(--border-default)] px-4">
        <strong>Artisan</strong>
        <span className="text-xs">Results</span>
        <span className="min-w-0 flex-1 truncate text-xs">{record?.title}</span>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => navigateTo({ page: "chat" })}
        >
          Pipeline runs
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => {
            void records.refetch();
            void summary.refetch();
          }}
        >
          Refresh
        </Button>
      </header>
      <main className="flex min-h-0 flex-1">
        <aside className="flex w-64 shrink-0 flex-col border-r border-[var(--border-default)] bg-[var(--surface-0)]">
          <input
            className="m-3 rounded border border-[var(--border-default)] p-2 text-xs"
            placeholder="Search results or run ID"
            aria-label="Search results"
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              setPage(0);
            }}
          />
          <div className="min-h-0 flex-1 overflow-y-auto">
            {records.isError ? (
              <p className="p-3 text-xs text-red-600">
                Unable to load results.
              </p>
            ) : null}
            {records.isLoading ? <p className="p-3 text-xs">Loading…</p> : null}
            {records.data?.length === 0 ? (
              <p className="p-3 text-xs">No generated results yet.</p>
            ) : null}
            {records.data?.map((item) => (
              <button
                type="button"
                key={item.record_id}
                onClick={() => selectRecord(item.record_id)}
                aria-current={selectedId === item.record_id}
                className={`block w-full border-b border-[var(--border-subtle)] p-3 text-left text-xs ${selectedId === item.record_id ? "bg-[var(--surface-2)]" : ""}`}
              >
                <span className="block font-medium">{item.title}</span>
                <span className="mt-1 block truncate text-[10px] text-[var(--text-tertiary)]">
                  {item.run_id ?? item.record_id}
                </span>
                <span className="mt-1 block text-[10px]">
                  {item.rating == null ? "Unrated" : `${item.rating} / 5`} ·{" "}
                  {item.total_cost_usd == null
                    ? "Cost unavailable"
                    : `$${item.total_cost_usd.toFixed(3)}`}
                </span>
              </button>
            ))}
          </div>
          <div className="flex justify-between p-2">
            <Button
              variant="ghost"
              size="sm"
              disabled={page === 0}
              onClick={() => setPage((value) => value - 1)}
            >
              Previous
            </Button>
            <Button
              variant="ghost"
              size="sm"
              disabled={!records.data || records.data.length < 100}
              onClick={() => setPage((value) => value + 1)}
            >
              Next
            </Button>
          </div>
        </aside>
        <section className="relative min-w-0 flex-1">
          {selectedId ? (
            <SceneCanvas
              baseFileUrl={`/api/records/${encodeURIComponent(selectedId)}/files`}
              assetRevisionKey={record?.viewer_asset_updated_at ?? null}
              selectionKey={selectedId}
              jointPoseSignal={displayedValues}
              renderOptions={options}
              onUrdfSpecChange={onSpecChange}
              onSnapshotReady={onSnapshot}
              onInvalidateReady={onInvalidate}
            />
          ) : (
            <div className="flex h-full items-center justify-center text-xs text-[var(--text-tertiary)]">
              Select a generated result
            </div>
          )}
        </section>
        <aside className="flex w-96 shrink-0 flex-col border-l border-[var(--border-default)] bg-[var(--surface-0)]">
          <nav className="flex border-b border-[var(--border-default)]">
            {(["inspect", "render", "code", "metadata"] as const).map(
              (value) => (
                <button
                  type="button"
                  key={value}
                  onClick={() => setTab(value)}
                  className={`flex-1 py-3 text-xs capitalize ${tab === value ? "font-semibold" : "text-[var(--text-tertiary)]"}`}
                >
                  {value}
                </button>
              ),
            )}
          </nav>
          <div className="min-h-0 flex-1 overflow-auto p-3">
            {summary.isError ? (
              <p className="text-xs text-red-600">
                Unable to load this result.
              </p>
            ) : null}
            {record ? (
              <p className="mb-3 text-xs" role="status">
                Run status: {record.run_status ?? "unavailable"}
              </p>
            ) : null}
            {tab === "render" ? (
              <RenderOptionsPanel
                options={options}
                onOptionChange={setOption}
                onSnapshot={snapshot}
              />
            ) : null}
            {tab === "inspect" ? (
              <div className="space-y-4">
                <p className="whitespace-pre-wrap text-xs">
                  {record?.prompt_preview ?? "Select a result to inspect."}
                </p>
                {record?.run_id ? (
                  <a
                    className="text-xs underline"
                    href={`/chat?run=${encodeURIComponent(record.run_id)}`}
                  >
                    View pipeline conversation, reports and artifacts
                  </a>
                ) : null}
                {record ? (
                  <AgentBreakdownTable
                    agents={record.agents}
                    totalCostUsd={record.total_cost_usd}
                    totalActiveSeconds={record.total_active_seconds}
                  />
                ) : null}
                <JointPanel
                  urdfSpec={spec}
                  jointValues={displayedValues}
                  onJointChange={setJointValue}
                  onResetAll={resetAll}
                />
              </div>
            ) : null}
            {tab === "code" ? (
              code.isError ? (
                <p className="text-xs">Model source is unavailable.</p>
              ) : (
                <CodeBlock code={code.data ?? ""} language="python" />
              )
            ) : null}
            {tab === "metadata" ? (
              metadata.isError ? (
                <p className="text-xs">Metadata is unavailable.</p>
              ) : (
                <CodeBlock code={metadata.data ?? ""} language="json" />
              )
            ) : null}
          </div>
        </aside>
      </main>
    </div>
  );
}
