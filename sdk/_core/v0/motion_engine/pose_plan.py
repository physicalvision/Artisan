from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from ..geometry_qc import generate_pose_samples
from ..self_collision import (
    _gate_value_grid,
    _moving_joints,
    _screen_range,
    generate_screening_poses,
    moving_subtree,
)


@dataclass(frozen=True, slots=True)
class PosePlanPolicy:
    selfcol_per_joint: int = 7
    selfcol_random_fill: int = 200
    selfcol_max_poses: int = 1500
    refine_proximity_tol: float = 0.02
    refine_budget: int = 400
    refine_max_depth: int = 8
    reach_samples_per_joint: int = 25
    reach_pose_budget: int = 2000
    support_gate_per: int = 5
    support_contact_tol: float = 1e-3
    exclusion_samples: int = 64
    coupling_samples: int = 12
    seed: int = 0

    @property
    def broad_phase_margin(self) -> float:
        return max(self.refine_proximity_tol, self.support_contact_tol)

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PoseRow:
    pose_id: int
    qpos: dict[str, float]
    tags: tuple[str, ...]
    provenance: tuple[str, ...]


@dataclass(slots=True)
class PoseTable:
    rows: list[PoseRow] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    _index: dict[tuple[tuple[str, bytes], ...], int] = field(default_factory=dict, repr=False)

    def add(
        self,
        qpos: dict[str, float],
        *,
        tags: Iterable[str],
        provenance: Iterable[str],
    ) -> int:
        normalized = {str(name): float(value) for name, value in qpos.items()}
        key = tuple((name, struct.pack(">d", value)) for name, value in sorted(normalized.items()))
        existing = self._index.get(key)
        new_tags = tuple(sorted(set(str(value) for value in tags)))
        new_provenance = tuple(sorted(set(str(value) for value in provenance)))
        if existing is not None:
            row = self.rows[existing]
            self.rows[existing] = PoseRow(
                pose_id=row.pose_id,
                qpos=row.qpos,
                tags=tuple(sorted(set(row.tags) | set(new_tags))),
                provenance=tuple(sorted(set(row.provenance) | set(new_provenance))),
            )
            return existing
        pose_id = len(self.rows)
        self._index[key] = pose_id
        self.rows.append(
            PoseRow(
                pose_id=pose_id,
                qpos=normalized,
                tags=new_tags,
                provenance=new_provenance,
            )
        )
        return pose_id

    def rows_for(self, tag: str) -> list[PoseRow]:
        return [row for row in self.rows if tag in row.tags]


def _add_self_collision_poses(table: PoseTable, model: object, policy: PosePlanPolicy) -> None:
    poses = generate_screening_poses(
        model,
        per_joint=policy.selfcol_per_joint,
        random_fill=policy.selfcol_random_fill,
        max_poses=policy.selfcol_max_poses,
        seed=policy.seed,
    )
    table.metadata["selfcol_pose_ids"] = [
        table.add(pose, tags=("selfcol",), provenance=(f"selfcol:{index}",))
        for index, pose in enumerate(poses)
    ]


def _add_reach_poses(
    table: PoseTable,
    model: object,
    gate_map: dict[str, list[str]],
    policy: PosePlanPolicy,
) -> None:
    from ..reachability import _build_outward_pose_units, _serial_prismatic_pose_grid

    joints = _moving_joints(model)
    ranges = {
        joint.name: tuple(float(value) for value in _screen_range(model, joint))
        for joint in joints
        if _screen_range(model, joint)[1] > _screen_range(model, joint)[0]
    }
    swept = [joint for joint in joints if joint.name in ranges]
    by_name = {joint.name: joint for joint in swept}
    gate_combos: dict[str, list[dict[str, float]]] = {}
    gate_links: set[str] = set()
    for target, names in gate_map.items():
        if target in names:
            raise ValueError(f"target joint {target!r} cannot appear in its own gate combo")
        gates = [by_name[name] for name in names if name in by_name]
        if target in by_name and gates:
            gate_combos[target] = _gate_value_grid(model, gates)
            for gate in gates:
                gate_links.update(moving_subtree(model, gate))
    per_joint = min(
        policy.reach_samples_per_joint,
        policy.reach_pose_budget // max(1, len(swept)),
    )
    poses, meta, sides = _build_outward_pose_units(swept, ranges, gate_combos, per_joint)
    local_to_global: list[int] = []
    for index, (pose, (joint_name, _value, side)) in enumerate(zip(poses, meta, strict=True)):
        tag = f"reach:{joint_name}" if joint_name else "rest"
        local_to_global.append(
            table.add(pose, tags=("reach", tag), provenance=(f"reach:{index}:{side}",))
        )
    mapped_sides: dict[str, dict[str, list[list[int]]]] = {}
    sample_units: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for joint_name, values in sides.items():
        mapped_sides[joint_name] = {
            side: [[local_to_global[index] for index in unit] for unit in units]
            for side, units in values.items()
        }
        sample_units[joint_name] = {
            side: [
                {
                    "owner_joint": joint_name,
                    "side": side,
                    "target_value": table.rows[global_unit[0]].qpos.get(joint_name),
                    "combo_pose_ids": list(global_unit),
                }
                for unit in units
                if (global_unit := [local_to_global[index] for index in unit])
            ]
            for side, units in values.items()
        }
    witness_rest: dict[str, list[int]] = {}
    for joint_name, combos in gate_combos.items():
        lo, hi = ranges[joint_name]
        rest = min(max(0.0, lo), hi)
        witness_rest[joint_name] = [
            table.add(
                {joint_name: rest, **combo},
                tags=("reach", f"reach:{joint_name}", "gate_grid"),
                provenance=(f"reach:{joint_name}:witness-rest:{index}",),
            )
            for index, combo in enumerate(combos)
        ]
    combined_ids = [
        table.add(
            pose,
            tags=("reach", "reach:serial_prismatic"),
            provenance=(f"reach:serial-prismatic:{index}",),
        )
        for index, pose in enumerate(_serial_prismatic_pose_grid(swept, ranges, gate_combos))
    ]
    search_pose_ids = list(local_to_global)
    for pose_ids in witness_rest.values():
        search_pose_ids.extend(pose_ids)
    search_pose_ids.extend(combined_ids)
    table.metadata["reach"] = {
        "ranges": ranges,
        "joint_order": [joint.name for joint in swept],
        "movers": {joint.name: sorted(moving_subtree(model, joint)) for joint in swept},
        "gate_map": {name: list(gate_map.get(name, ())) for name in sorted(gate_combos)},
        "gate_combos": {
            name: [dict(combo) for combo in combos] for name, combos in sorted(gate_combos.items())
        },
        "gate_links": sorted(gate_links),
        "sides": mapped_sides,
        "sample_units": sample_units,
        "witness_rest_pose_ids": witness_rest,
        "combined_pose_ids": combined_ids,
        "outward_pose_ids": list(local_to_global),
        "search_pose_ids": list(dict.fromkeys(search_pose_ids)),
    }


def _add_support_poses(
    table: PoseTable,
    model: object,
    gate_map: dict[str, list[str]],
    policy: PosePlanPolicy,
) -> None:
    moving = _moving_joints(model)
    by_name = {joint.name: joint for joint in moving}
    pose_ids: list[int] = []
    for joint in moving:
        lo, hi = _screen_range(model, joint)
        ends = sorted({round(float(value), 9) for value in (lo, hi) if abs(value) > 1e-9})
        gates = [by_name[name] for name in gate_map.get(joint.name, ()) if name in by_name]
        combos = _gate_value_grid(model, gates, per_gate=policy.support_gate_per) if gates else [{}]
        for endpoint in ends:
            for combo_index, combo in enumerate(combos):
                pose_ids.append(
                    table.add(
                        {joint.name: endpoint, **combo},
                        tags=("support",),
                        provenance=(f"support:{joint.name}:{endpoint!r}:{combo_index}",),
                    )
                )
    table.metadata["support_pose_ids"] = pose_ids


def build_pose_table(
    model: object,
    *,
    gate_map: dict[str, list[str]] | None = None,
    policy: PosePlanPolicy | None = None,
) -> PoseTable:
    resolved_policy = policy or PosePlanPolicy()
    resolved_gate_map = gate_map or {}
    table = PoseTable()
    table.add({}, tags=("rest",), provenance=("rest",))
    _add_self_collision_poses(table, model, resolved_policy)
    _add_reach_poses(table, model, resolved_gate_map, resolved_policy)
    _add_support_poses(table, model, resolved_gate_map, resolved_policy)
    exclusion_ids: list[int] = []
    for index, pose in enumerate(
        generate_pose_samples(
            model,
            max_samples=resolved_policy.exclusion_samples,
            seed=resolved_policy.seed,
        )
    ):
        if all(isinstance(value, (int, float)) for value in pose.values()):
            exclusion_ids.append(
                table.add(
                    pose,  # type: ignore[arg-type]
                    tags=("exclusion",),
                    provenance=(f"exclusion:{index}",),
                )
            )
    table.metadata["exclusion_pose_ids"] = exclusion_ids
    table.metadata["policy"] = asdict(resolved_policy)
    table.metadata["policy_digest"] = resolved_policy.digest
    return table
