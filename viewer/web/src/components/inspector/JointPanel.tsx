import type { JSX } from "react";
import { RotateCcw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { JointSlider, type UrdfJoint } from "@/components/inspector/JointSlider";

type JointPanelProps = {
  urdfSpec: { joints: UrdfJoint[] } | null;
  jointValues: Map<string, number>;
  onJointChange: (name: string, value: number) => void;
  onResetAll: () => void;
};

export function JointPanel({ urdfSpec, jointValues, onJointChange, onResetAll }: JointPanelProps): JSX.Element {
  const movableJoints = (urdfSpec?.joints ?? []).filter((j) => j.type !== "fixed" && !j.mimic);

  if (!urdfSpec || movableJoints.length === 0) {
    return (
      <div className="flex h-32 items-center justify-center">
        <p className="text-[11px] text-[#bbb]">No movable joints</p>
      </div>
    );
  }

  return (
    <ScrollArea className="h-full">
      <div className="space-y-2 pb-1">
        <div className="flex items-center justify-between">
          <span className="font-mono text-[10px] tabular-nums text-[#999]">
            {movableJoints.length} joint{movableJoints.length !== 1 ? "s" : ""}
          </span>
          <Button
            variant="ghost"
            size="sm"
            onClick={onResetAll}
            className="h-6 gap-1 px-2 text-[11px] text-[#999] hover:text-[#666]"
          >
            <RotateCcw className="size-3" />
            Reset
          </Button>
        </div>

        {movableJoints.map((joint) => (
          <JointSlider
            key={joint.name}
            joint={joint}
            value={jointValues.get(joint.name) ?? 0}
            onChange={onJointChange}
          />
        ))}
      </div>
    </ScrollArea>
  );
}
