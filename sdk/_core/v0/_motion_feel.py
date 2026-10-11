from __future__ import annotations

import math

import numpy as np

from .errors import ValidationError
from .geometry_qc import (
    _mat4_mul,
    _origin_to_mat4,
    compute_part_world_transforms,
)
from .types import ArticulationType, MotionProperties

MAPPING_VERSION = "motion-feel-v5"
GRAVITY = 9.81
HOLD_MARGIN = 1.25
HOLD_SOLREF = (0.005, 1.0)
HOLD_SOLIMP = (0.998, 0.9999, 0.001, 0.5, 2.0)
_EFFORT_RATIO_MIN = 0.02
_EFFORT_RATIO_MAX = 0.60
_DECAY_TIME_MAX_S = 3.0
_DECAY_TIME_MIN_S = 0.15
_POSE_SAMPLES = 13
_EPS = 1e-12
SYNTHESIZED_FRICTION_GRAVITY_RATIO = 0.2
SYNTHESIZED_DECAY_TIME_S = 2.0
_SPRING_BEHAVIOR_KINDS = frozenset({"latch_release", "self_closer"})
_BEHAVIOR_SETTLE_TIME_S = {"soft": 1.0, "firm": 0.65, "strong": 0.4}
_BEHAVIOR_GRAVITY_MARGIN = {"soft": 1.1, "firm": 1.5, "strong": 2.0}
_BEHAVIOR_DAMPING_RATIO = 1.0


def _log_lerp(low: float, high: float, value: float) -> float:
    return math.exp(math.log(low) + float(value) * (math.log(high) - math.log(low)))


def _subtree_part_names(model: object, root: str) -> list[str]:
    by_parent: dict[str, list[str]] = {}
    for articulation in getattr(model, "articulations", []) or []:
        by_parent.setdefault(str(articulation.parent), []).append(str(articulation.child))
    names: list[str] = []
    queue = [root]
    while queue:
        name = queue.pop(0)
        if name in names:
            continue
        names.append(name)
        queue.extend(by_parent.get(name, ()))
    return names


def _rotation(tf: object) -> np.ndarray:
    return np.asarray([[float(tf[row][col]) for col in range(3)] for row in range(3)])


def _translation(tf: object) -> np.ndarray:
    return np.asarray([float(tf[row][3]) for row in range(3)])


def _normalized(value: object) -> np.ndarray:
    vector = np.asarray(tuple(float(item) for item in value), dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= _EPS:
        raise ValidationError("motion_feel joint axis must be non-zero")
    return vector / norm


def _joint_frame(model: object, articulation: object) -> tuple[np.ndarray, np.ndarray]:
    transforms = compute_part_world_transforms(model, {})
    parent_tf = transforms[str(articulation.parent)]
    joint_tf = _mat4_mul(parent_tf, _origin_to_mat4(articulation.origin))
    origin = _translation(joint_tf)
    axis = _rotation(joint_tf) @ _normalized(articulation.axis)
    return origin, axis / np.linalg.norm(axis)


def _inertia_matrix(inertia: object) -> np.ndarray:
    return np.asarray(
        [
            [inertia.ixx, inertia.ixy, inertia.ixz],
            [inertia.ixy, inertia.iyy, inertia.iyz],
            [inertia.ixz, inertia.iyz, inertia.izz],
        ],
        dtype=float,
    )


def _part_mass_data(
    model: object,
    part_names: list[str],
    transforms: dict[str, object],
) -> list[tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
    parts = {str(part.name): part for part in getattr(model, "parts", []) or []}
    result: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    for name in part_names:
        part = parts.get(name)
        inertial = getattr(part, "inertial", None)
        if inertial is None:
            raise ValidationError(
                f"motion_feel resolution requires enriched inertial for part {name!r}"
            )
        mass = float(inertial.mass)
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValidationError(f"motion_feel part {name!r} has invalid mass {mass!r}")
        inertial_tf = _mat4_mul(transforms[name], _origin_to_mat4(inertial.origin))
        result.append(
            (
                mass,
                _translation(inertial_tf),
                _rotation(inertial_tf),
                _inertia_matrix(inertial.inertia),
            )
        )
    return result


def _equivalent_inertia_and_radius(
    mass_data: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]],
    joint_origin: np.ndarray,
    joint_axis: np.ndarray,
) -> tuple[float, float, float]:
    total_mass = sum(item[0] for item in mass_data)
    equivalent = 0.0
    weighted_radius = 0.0
    for mass, center, rotation, inertia in mass_data:
        offset = center - joint_origin
        perpendicular = offset - joint_axis * float(np.dot(offset, joint_axis))
        radius = float(np.linalg.norm(perpendicular))
        inertia_world = rotation @ inertia @ rotation.T
        equivalent += float(joint_axis @ inertia_world @ joint_axis) + mass * radius * radius
        weighted_radius += mass * radius
    equivalent = max(equivalent, _EPS)
    radius_from_inertia = math.sqrt(equivalent / max(total_mass, _EPS))
    reference_radius = max(weighted_radius / total_mass, radius_from_inertia)
    return equivalent, total_mass, reference_radius


def _sample_window(articulation: object) -> list[float]:
    if articulation.articulation_type == ArticulationType.CONTINUOUS:
        lower, upper = -math.pi, math.pi
    else:
        limits = articulation.motion_limits
        if limits is None or limits.lower is None or limits.upper is None:
            raise ValidationError(
                f"motion_feel joint {articulation.name!r} requires finite motion limits"
            )
        lower, upper = float(limits.lower), float(limits.upper)
    return [lower + (upper - lower) * idx / (_POSE_SAMPLES - 1) for idx in range(_POSE_SAMPLES)]


def _gravity_load(
    model: object,
    articulation: object,
    part_names: list[str],
    joint_origin: np.ndarray,
    joint_axis: np.ndarray,
    total_mass: float,
) -> tuple[float, float]:
    if articulation.articulation_type == ArticulationType.PRISMATIC:
        force = np.asarray((0.0, 0.0, -total_mass * GRAVITY))
        return abs(float(np.dot(force, joint_axis))), 0.0

    worst_load = 0.0
    worst_pose = 0.0
    for pose in _sample_window(articulation):
        transforms = compute_part_world_transforms(model, {str(articulation.name): pose})
        load = 0.0
        for mass, center, _rotation_value, _inertia in _part_mass_data(
            model, part_names, transforms
        ):
            force = np.asarray((0.0, 0.0, -mass * GRAVITY))
            load += float(np.dot(joint_axis, np.cross(center - joint_origin, force)))
        if abs(load) > worst_load:
            worst_load = abs(load)
            worst_pose = pose
    return worst_load, worst_pose


def resolve_motion_feel(enriched_model: object) -> tuple[object, list[dict[str, object]]]:
    evidence: list[dict[str, object]] = []
    for articulation in getattr(enriched_model, "articulations", []) or []:
        feel = getattr(articulation, "motion_feel", None)
        behavior = getattr(articulation, "behavior", None)
        behavior_kind = str(getattr(behavior, "kind", "") or "")
        explicit_properties = getattr(articulation, "motion_properties", None)
        behavior_spring = (
            feel is None
            and behavior_kind in _SPRING_BEHAVIOR_KINDS
            and getattr(explicit_properties, "spring_stiffness", None) is not None
        )
        synthesized = feel is None and not behavior_spring
        if (
            feel is None
            and not behavior_spring
            and (
                explicit_properties is not None
                or articulation.mimic is not None
                or articulation.behavior is not None
                or articulation.articulation_type
                not in {ArticulationType.REVOLUTE, ArticulationType.PRISMATIC}
            )
        ):
            continue
        if synthesized:
            limits = articulation.motion_limits
            if (
                limits is None
                or limits.lower is None
                or limits.upper is None
                or not math.isfinite(float(limits.lower) + float(limits.upper))
                or float(limits.upper) <= float(limits.lower)
            ):
                continue
        meta = getattr(articulation, "meta", None)
        if not isinstance(meta, dict):
            meta = {}
            articulation.meta = meta
        if feel is not None and (
            articulation.mimic is not None or articulation.behavior is not None
        ):
            raise ValidationError(
                f"motion_feel requires a plain joint, got coupled joint {articulation.name!r}"
            )

        part_names = _subtree_part_names(enriched_model, str(articulation.child))
        rest_transforms = compute_part_world_transforms(enriched_model, {})
        mass_data = _part_mass_data(enriched_model, part_names, rest_transforms)
        joint_origin, joint_axis = _joint_frame(enriched_model, articulation)
        if articulation.articulation_type == ArticulationType.PRISMATIC:
            total_mass = sum(item[0] for item in mass_data)
            equivalent = total_mass
            reference_radius = 0.0
        else:
            equivalent, total_mass, reference_radius = _equivalent_inertia_and_radius(
                mass_data, joint_origin, joint_axis
            )

        max_gravity, worst_pose = _gravity_load(
            enriched_model,
            articulation,
            part_names,
            joint_origin,
            joint_axis,
            total_mass,
        )
        effort_reference = (
            total_mass * GRAVITY
            if articulation.articulation_type == ArticulationType.PRISMATIC
            else max(max_gravity, total_mass * GRAVITY * reference_radius)
        )
        if behavior_spring:
            strength = str(getattr(behavior, "strength", "") or "")
            if strength not in _BEHAVIOR_SETTLE_TIME_S:
                raise ValidationError(
                    f"behavior spring {articulation.name!r} has unknown strength {strength!r}"
                )
            limits = articulation.motion_limits
            lower = getattr(limits, "lower", None) if limits is not None else None
            upper = getattr(limits, "upper", None) if limits is not None else None
            if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
                raise ValidationError(
                    f"behavior spring {articulation.name!r} requires finite motion limits"
                )
            span = abs(float(upper) - float(lower))
            if not math.isfinite(span) or span <= _EPS:
                raise ValidationError(
                    f"behavior spring {articulation.name!r} has invalid motion span {span!r}"
                )
            settle_time = _BEHAVIOR_SETTLE_TIME_S[strength]
            natural_frequency = 4.0 / settle_time
            dynamic_stiffness = equivalent * natural_frequency * natural_frequency
            gravity_stiffness = _BEHAVIOR_GRAVITY_MARGIN[strength] * max_gravity / span
            stiffness = max(dynamic_stiffness, gravity_stiffness, _EPS)
            resolved_damping = (
                2.0 * _BEHAVIOR_DAMPING_RATIO * math.sqrt(stiffness * max(equivalent, _EPS))
            )
            prior_resolution = meta.get("_motion_feel_resolution")
            prior_is_resolved = (
                isinstance(prior_resolution, dict)
                and prior_resolution.get("source") == "behavior_spring"
            )
            prior_has_explicit_damping = (
                prior_is_resolved and prior_resolution.get("damping_source") == "explicit"
            )
            explicit_damping = (
                getattr(explicit_properties, "damping", None)
                if explicit_properties is not None
                and (not prior_is_resolved or prior_has_explicit_damping)
                else None
            )
            damping = (
                float(explicit_damping)
                if isinstance(explicit_damping, (int, float))
                else resolved_damping
            )
            articulation.motion_properties = MotionProperties(
                damping=damping,
                friction=getattr(explicit_properties, "friction", None),
                spring_stiffness=stiffness,
                spring_ref=getattr(explicit_properties, "spring_ref", None),
            )
            item = {
                "joint": str(articulation.name),
                "version": MAPPING_VERSION,
                "source": "behavior_spring",
                "behavior_kind": behavior_kind,
                "strength": strength,
                "equivalent_inertia_or_mass": equivalent,
                "subtree_mass": total_mass,
                "reference_radius": reference_radius,
                "max_gravity_load": max_gravity,
                "worst_gravity_pose": worst_pose,
                "motion_span": span,
                "target_settle_time": settle_time,
                "gravity_margin": _BEHAVIOR_GRAVITY_MARGIN[strength],
                "dynamic_stiffness": dynamic_stiffness,
                "gravity_stiffness": gravity_stiffness,
                "spring_stiffness": stiffness,
                "damping": damping,
                "damping_ratio": _BEHAVIOR_DAMPING_RATIO,
                "damping_source": "explicit" if explicit_damping is not None else "resolved",
            }
            meta["_motion_feel_resolution"] = dict(item)
            evidence.append(item)
            continue
        if synthesized:
            if max_gravity <= _EPS:
                continue
            effort_resistance = SYNTHESIZED_FRICTION_GRAVITY_RATIO * max_gravity
            hold_floor = 0.0
            frictionloss = effort_resistance
            decay_time = SYNTHESIZED_DECAY_TIME_S
            source = "synthesized_default"
        else:
            effort_resistance = (
                _log_lerp(_EFFORT_RATIO_MIN, _EFFORT_RATIO_MAX, feel.effort) * effort_reference
            )
            hold_floor = HOLD_MARGIN * max_gravity if feel.hold == "holds_anywhere" else 0.0
            frictionloss = max(effort_resistance, hold_floor)
            decay_time = _log_lerp(_DECAY_TIME_MAX_S, _DECAY_TIME_MIN_S, feel.motion)
            source = "motion_feel"
        damping = equivalent / decay_time

        prior = articulation.motion_properties
        articulation.motion_properties = MotionProperties(
            damping=damping,
            friction=frictionloss,
            spring_stiffness=getattr(prior, "spring_stiffness", None),
            spring_ref=getattr(prior, "spring_ref", None),
        )
        item: dict[str, object] = {
            "joint": str(articulation.name),
            "version": MAPPING_VERSION,
            "source": source,
            "equivalent_inertia_or_mass": equivalent,
            "subtree_mass": total_mass,
            "reference_radius": reference_radius,
            "effort_reference": effort_reference,
            "effort_resistance": effort_resistance,
            "max_gravity_load": max_gravity,
            "worst_gravity_pose": worst_pose,
            "frictionloss": frictionloss,
            "damping": damping,
            "decay_time": decay_time,
            "friction_gravity_ratio": (SYNTHESIZED_FRICTION_GRAVITY_RATIO if synthesized else None),
            "solreffriction": (
                list(HOLD_SOLREF) if not synthesized and feel.hold == "holds_anywhere" else None
            ),
            "solimpfriction": (
                list(HOLD_SOLIMP) if not synthesized and feel.hold == "holds_anywhere" else None
            ),
        }
        meta["_motion_feel_resolution"] = dict(item)
        evidence.append(item)
    return enriched_model, evidence
