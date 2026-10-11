import type { JSX } from "react";

import { Slider } from "@/components/ui/slider";

export type UrdfJoint = {
  name: string;
  type: "fixed" | "revolute" | "continuous" | "prismatic" | "floating" | "planar";
  parent: string;
  child: string;
  origin?: { xyz?: [number, number, number]; rpy?: [number, number, number] };
  axis?: [number, number, number];
  limit?: { lower?: number; upper?: number; effort?: number; velocity?: number };
  mimic?: { joint: string; multiplier: number; offset: number };
};

type JointSliderProps = {
  joint: UrdfJoint;
  value: number;
  onChange: (name: string, value: number) => void;
};

function radToDeg(rad: number): string {
  return (rad * (180 / Math.PI)).toFixed(1);
}

function jointRange(joint: UrdfJoint): [number, number] {
  if (joint.type === "continuous") {
    return [-Math.PI, Math.PI];
  }

  const lower = joint.limit?.lower ?? 0;
  const upper = joint.limit?.upper ?? 0;
  return [lower, upper];
}

function formatValue(joint: UrdfJoint, value: number): string {
  if (joint.type === "revolute" || joint.type === "continuous") {
    return `${radToDeg(value)}°`;
  }
  return `${value.toFixed(3)}m`;
}

export function JointSlider({ joint, value, onChange }: JointSliderProps): JSX.Element {
  const [min, max] = jointRange(joint);

  return (
    <div className="space-y-2 rounded-sm border border-[#e8e8e8] bg-white px-2.5 py-2.5">
      <div className="flex items-baseline justify-between gap-2">
        <span className="min-w-0 truncate font-mono text-[11px] text-[#1e1e1e]" title={joint.name}>
          {joint.name}
        </span>
        <span className="shrink-0 font-mono text-[10px] text-[#aaa]">
          {joint.type}
        </span>
      </div>

      <Slider
        min={min}
        max={max}
        step={0.01}
        value={[value]}
        onValueChange={([v]) => onChange(joint.name, v)}
      />

      <div className="flex justify-between font-mono text-[10px] text-[#bbb]">
        <span>{formatValue(joint, min)}</span>
        <span className="text-[#666]">{formatValue(joint, value)}</span>
        <span>{formatValue(joint, max)}</span>
      </div>
    </div>
  );
}
