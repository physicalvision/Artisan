from __future__ import annotations

from dataclasses import dataclass

from protocol_loader import RobotSpec

MOVABLE_TYPES = frozenset({"revolute", "continuous", "prismatic"})
ROOT_PART = "part_root"


@dataclass(frozen=True)
class PartPartition:
    link_to_part: dict[str, str]


def part_partition(robot: RobotSpec) -> PartPartition:
    parent_joint = {joint.child: joint for joint in robot.joints}

    def owner(link: str) -> str:
        seen: set[str] = set()
        while link in parent_joint and link not in seen:
            seen.add(link)
            joint = parent_joint[link]
            if joint.joint_type in MOVABLE_TYPES:
                return f"part_{joint.child}"
            link = joint.parent
        return ROOT_PART

    return PartPartition({name: owner(name) for name, link in robot.links.items() if link.visuals})
