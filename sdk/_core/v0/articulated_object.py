from __future__ import annotations

import copy
import math
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

from ._coupling import (
    NONLINEAR_TEMPLATES,
    full_multiplier_magnitude,
    nonlinear_template_poly,
    secant_multiplier,
    template_fit_error,
    template_monotone_window,
    turns_to_multiplier,
)
from .assets import AssetContext, AssetSession, coerce_asset_context, resolve_asset_context
from .errors import ValidationError
from .types import (
    BEHAVIOR_KINDS,
    Articulation,
    ArticulationBehavior,
    ArticulationType,
    Box,
    Cylinder,
    Geometry,
    Inertial,
    Material,
    Mesh,
    Mimic,
    MotionFeel,
    MotionLimits,
    MotionProperties,
    Origin,
    Part,
    Sphere,
    Vec3,
    Visual,
    _normalize_material_rgba,
)

_BEHAVIOR_STIFFNESS = {"soft": 10.0, "firm": 50.0, "strong": 200.0}
_BEHAVIOR_SPEED = {"slow": 0.5, "normal": 1.0, "fast": 2.0}

_ALLOW_EXPLICIT_COLLISIONS_ATTR = "_sdk_allow_explicit_collisions"
_SCALAR_ARTICULATION_TYPES = {
    ArticulationType.REVOLUTE,
    ArticulationType.PRISMATIC,
    ArticulationType.CONTINUOUS,
}
_ANGULAR_ARTICULATION_TYPES = {
    ArticulationType.REVOLUTE,
    ArticulationType.CONTINUOUS,
}


def _part_name_ref(value: Union[str, Part], *, field_name: str) -> str:
    if isinstance(value, str):
        name = value
    else:
        name = getattr(value, "name", None)
        if not isinstance(name, str):
            raise ValidationError(f"{field_name} must be a part name or Part instance")
    name = name.strip()
    if not name:
        raise ValidationError(f"{field_name} must be non-empty")
    return name


def _articulation_name_ref(value: Union[str, Articulation], *, field_name: str) -> str:
    if isinstance(value, str):
        name = value
    else:
        name = getattr(value, "name", None)
        if not isinstance(name, str):
            raise ValidationError(
                f"{field_name} must be an articulation name or Articulation instance"
            )
    name = name.strip()
    if not name:
        raise ValidationError(f"{field_name} must be non-empty")
    return name


@dataclass
class ArticulatedObject:

    name: str
    parts: List[Part] = field(default_factory=list)
    articulations: List[Articulation] = field(default_factory=list)
    materials: List[Material] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)
    assets: Optional[AssetContext] = None
    _part_index: Dict[str, Part] = field(default_factory=dict, init=False, repr=False)
    _articulation_index: Dict[str, Articulation] = field(
        default_factory=dict, init=False, repr=False
    )
    _visual_uid_counter: int = field(default=0, init=False, repr=False)
    _placement_bind_records: List[Dict[str, object]] = field(
        default_factory=list, init=False, repr=False
    )
    _placement_bind_ordinal: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.assets = resolve_asset_context(self.assets)
        for part in self.parts:
            part.assets = coerce_asset_context(getattr(part, "assets", None)) or self.assets
        self._normalize_visual_uids()

    def _normalize_visual_uids(self) -> None:
        seen: set[str] = set()
        highest = self._visual_uid_counter
        missing: list[Visual] = []
        for part in self.parts:
            for visual in part.visuals:
                uid = visual.visual_uid
                committed = getattr(visual, "_visual_uid_committed", uid)
                if uid != committed:
                    raise ValidationError(f"attempted Visual UID rewrite: {committed!r} -> {uid!r}")
                if uid is None:
                    missing.append(visual)
                    continue
                match = re.fullmatch(r"visual-([0-9]{6,})", uid)
                if match is None:
                    raise ValidationError(f"invalid Visual UID: {uid!r}")
                if uid in seen:
                    raise ValidationError(f"duplicate Visual UID: {uid!r}")
                seen.add(uid)
                highest = max(highest, int(match.group(1)))
        self._visual_uid_counter = highest
        for visual in missing:
            self._visual_uid_counter += 1
            uid = f"visual-{self._visual_uid_counter:06d}"
            while uid in seen:
                self._visual_uid_counter += 1
                uid = f"visual-{self._visual_uid_counter:06d}"
            visual._assign_visual_uid(uid)
            seen.add(uid)

    @property
    def placement_bind_records(self) -> tuple[Dict[str, object], ...]:
        return tuple(copy.deepcopy(record) for record in self._placement_bind_records)

    def governing_placement_records(
        self,
        part_name: str,
        visual_uid: Optional[str],
    ) -> tuple[Dict[str, object], ...]:
        active = {
            str(record.get("target_ref")): record
            for record in self._placement_bind_records
            if not record.get("superseded")
        }
        governing: list[Dict[str, object]] = []
        if visual_uid is not None:
            element = active.get(f"visual:{part_name}:{visual_uid}")
            if element is not None:
                governing.append(element)
        by_child = {joint.child: joint for joint in self.articulations}
        current = part_name
        visited: set[str] = set()
        while current in by_child and current not in visited:
            visited.add(current)
            joint = by_child[current]
            record = active.get(f"joint:{joint.name}")
            if record is not None:
                governing.append(record)
            current = joint.parent
        return tuple(copy.deepcopy(record) for record in governing)

    @property
    def links(self) -> List[Part]:
        return self.parts

    @links.setter
    def links(self, value: List[Part]) -> None:
        self.parts = value

    @property
    def joints(self) -> List[Articulation]:
        return self.articulations

    @joints.setter
    def joints(self, value: List[Articulation]) -> None:
        self.articulations = value

    def part(
        self,
        name: str,
        *,
        visuals: Optional[Iterable[Visual]] = None,
        inertial: Optional[Inertial] = None,
        est_mass_kg: Optional[float] = None,
        meta: Optional[Dict[str, object]] = None,
    ) -> Part:
        part = Part(
            name=name,
            visuals=list(visuals or []),
            collisions=[],
            inertial=inertial,
            meta=dict(meta or {}),
            assets=self.assets,
            est_mass_kg=est_mass_kg,
        )
        self.parts.append(part)
        self._part_index[name] = part
        self._normalize_visual_uids()
        return part

    def link(
        self,
        name: str,
        *,
        visuals: Optional[Iterable[Visual]] = None,
        inertial: Optional[Inertial] = None,
        est_mass_kg: Optional[float] = None,
        meta: Optional[Dict[str, object]] = None,
    ) -> Part:
        return self.part(
            name,
            visuals=visuals,
            inertial=inertial,
            est_mass_kg=est_mass_kg,
            meta=meta,
        )

    def articulation(
        self,
        name: str,
        articulation_type: Union[ArticulationType, str],
        parent: Union[str, Part],
        child: Union[str, Part],
        *,
        origin: Optional[Origin] = None,
        axis: Optional[Vec3] = None,
        motion_limits: Optional[MotionLimits] = None,
        motion_properties: Optional[MotionProperties] = None,
        motion_feel: Optional[Union[MotionFeel, Dict[str, object]]] = None,
        mimic: Optional[Mimic] = None,
        meta: Optional[Dict[str, object]] = None,
    ) -> Articulation:
        try:
            resolved_type = (
                articulation_type
                if isinstance(articulation_type, ArticulationType)
                else ArticulationType(str(articulation_type))
            )
        except ValueError as exc:
            raise ValidationError(f"Unknown articulation type: {articulation_type}") from exc

        articulation = Articulation(
            name=name,
            articulation_type=resolved_type,
            parent=_part_name_ref(parent, field_name="parent"),
            child=_part_name_ref(child, field_name="child"),
            origin=origin or Origin(),
            axis=axis or (0.0, 0.0, 1.0),
            motion_limits=motion_limits,
            motion_properties=motion_properties,
            motion_feel=motion_feel,
            mimic=mimic,
            meta=dict(meta or {}),
        )
        self.articulations.append(articulation)
        self._articulation_index[name] = articulation
        return articulation

    def joint(
        self,
        name: str,
        joint_type: Union[ArticulationType, str],
        parent: Union[str, Part],
        child: Union[str, Part],
        *,
        origin: Optional[Origin] = None,
        axis: Optional[Vec3] = None,
        limit: Optional[MotionLimits] = None,
        dynamics: Optional[MotionProperties] = None,
        motion_feel: Optional[Union[MotionFeel, Dict[str, object]]] = None,
        mimic: Optional[Mimic] = None,
        meta: Optional[Dict[str, object]] = None,
    ) -> Articulation:
        return self.articulation(
            name,
            articulation_type=joint_type,
            parent=parent,
            child=child,
            origin=origin,
            axis=axis,
            motion_limits=limit,
            motion_properties=dynamics,
            motion_feel=motion_feel,
            mimic=mimic,
            meta=meta,
        )

    def couple(
        self,
        driver: Union[str, Articulation],
        follower: Union[str, Articulation],
        *,
        kind: str = "proportional",
        sense: str = "same",
        mapping: Union[str, float] = "full",
        turns: Union[float, None] = None,
        **params: float,
    ) -> Articulation:
        driver_art = self.get_articulation(driver)
        follower_art = self.get_articulation(follower)

        if sense not in ("same", "opposite"):
            raise ValidationError(f"couple sense must be 'same' or 'opposite', got {sense!r}")

        if driver_art.name == follower_art.name and kind != "self_closer":
            raise ValidationError(
                f"couple: a joint cannot be coupled to itself ({follower_art.name!r}); "
                "only kind='self_closer' is self-driven"
            )
        if (kind == "proportional" or kind in NONLINEAR_TEMPLATES) and driver_art.mimic is not None:
            raise ValidationError(
                f"couple: chained couplings are not supported — driver {driver_art.name!r} is "
                f"itself a follower; couple {follower_art.name!r} directly to the root driver "
                "(compose the ratios)"
            )
        if follower_art.motion_feel is not None:
            owner = "behavior-owned joint" if kind in BEHAVIOR_KINDS else "coupled follower"
            raise ValidationError(f"motion_feel is not supported on {owner} {follower_art.name!r}")

        if kind in NONLINEAR_TEMPLATES:
            if driver_art.articulation_type not in _ANGULAR_ARTICULATION_TYPES:
                raise ValidationError(
                    f"couple: {kind} driver {driver_art.name!r} must be rotary "
                    f"(revolute/continuous), got {driver_art.articulation_type.value}"
                )
            if follower_art.articulation_type != ArticulationType.PRISMATIC:
                raise ValidationError(
                    f"couple: {kind} follower {follower_art.name!r} must be prismatic (linear), "
                    f"got {follower_art.articulation_type.value}"
                )
            window = self._driver_angular_window(driver_art, kind, params)
            poly = nonlinear_template_poly(kind, driver_window=window, sense=sense, **params)
            secant = secant_multiplier(poly, window)
            fit_err = template_fit_error(kind, window, poly, **params)
            follower_art.mimic = Mimic(joint=driver_art.name, poly=poly, linear_multiplier=secant)
            follower_art.meta["coupling"] = {
                "kind": kind,
                "params": dict(params),
                "sense": sense,
                "window": [float(window[0]), float(window[1])],
                "fit_error": float(fit_err),
            }
            return follower_art
        if kind in BEHAVIOR_KINDS:
            return self._couple_behavior(driver_art, follower_art, kind, dict(params))
        if kind != "proportional":
            raise ValidationError(
                f"couple kind must be 'proportional', a nonlinear template "
                f"{NONLINEAR_TEMPLATES}, or a behavior {BEHAVIOR_KINDS}, got {kind!r}"
            )

        sign = 1.0 if sense == "same" else -1.0
        if turns is not None:
            if mapping != "full":
                raise ValidationError(
                    f"couple {follower_art.name!r}: pass either turns= OR mapping=, not both"
                )
            if driver_art.articulation_type not in _ANGULAR_ARTICULATION_TYPES:
                raise ValidationError(
                    f"couple {follower_art.name!r}: turns= needs a rotary driver "
                    f"(revolute/continuous), got {driver_art.articulation_type.value}"
                )
            if not (math.isfinite(float(turns)) and float(turns) > 0.0):
                raise ValidationError(
                    f"couple {follower_art.name!r}: turns must be a positive number of revolutions"
                )
            magnitude = turns_to_multiplier(follower_art, float(turns))
            if magnitude is None or magnitude <= 0.0:
                raise ValidationError(
                    f"couple {follower_art.name!r}: turns= needs a BOUNDED follower (its travel is "
                    "what the N revolutions sweep)"
                )
        elif isinstance(mapping, str):
            if mapping != "full":
                raise ValidationError(f"couple mapping must be 'full' or a number, got {mapping!r}")
            magnitude = full_multiplier_magnitude(follower_art, driver_art)
            if magnitude is None or magnitude <= 0.0:
                raise ValidationError(
                    f"couple {follower_art.name!r}: mapping='full' needs a bounded follower and "
                    "a driver with a usable reference span (continuous=2*pi, or a finite window)"
                )
        else:
            if _motion_domain(driver_art.articulation_type) != _motion_domain(
                follower_art.articulation_type
            ):
                raise ValidationError(
                    f"couple {follower_art.name!r}: a numeric mapping is a dimensionless ratio for "
                    "SAME-DOMAIN drives only; this coupling is cross-domain (rotary<->linear) where "
                    "the number would be an unguessable m/rad. Use turns=N to sweep the follower's "
                    "full travel over N driver revolutions, or mapping='full'."
                )
            magnitude = float(mapping)
            if not math.isfinite(magnitude) or magnitude <= 0.0:
                raise ValidationError(
                    f"couple {follower_art.name!r}: numeric mapping must be a positive finite ratio"
                )

        follower_art.mimic = Mimic(joint=driver_art.name, multiplier=sign * magnitude, offset=0.0)
        return follower_art

    @staticmethod
    def _driver_angular_window(
        driver_art: Articulation, template: str, params: Dict[str, object]
    ) -> tuple[float, float]:
        fparams = {k: float(v) for k, v in params.items() if isinstance(v, (int, float))}
        mono_lo, mono_hi = template_monotone_window(template, **fparams)
        ml = driver_art.motion_limits
        lo = getattr(ml, "lower", None) if ml is not None else None
        hi = getattr(ml, "upper", None) if ml is not None else None
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and hi > lo:
            lo, hi = float(lo), float(hi)
            tol = 1e-9
            if lo < mono_lo - tol or hi > mono_hi + tol:
                raise ValidationError(
                    f"nonlinear coupling driver {driver_art.name!r}: limits "
                    f"[{lo:.4f}, {hi:.4f}] exceed the {template} monotone window "
                    f"[{mono_lo:.4f}, {mono_hi:.4f}] — the follower would reverse / the "
                    "curve is not single-valued there. Tighten the driver limits "
                    "(a one-sided template's window must not cross rest q=0)."
                )
            return lo, hi
        if driver_art.articulation_type == ArticulationType.CONTINUOUS:
            return mono_lo, mono_hi
        raise ValidationError(
            f"nonlinear coupling driver {driver_art.name!r} needs finite limits or be continuous"
        )

    @staticmethod
    def _limit_by(art: Articulation, which: str) -> float:
        ml = art.motion_limits
        lo = getattr(ml, "lower", None) if ml is not None else None
        hi = getattr(ml, "upper", None) if ml is not None else None
        if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
            raise ValidationError(
                f"behavior coupling needs finite limits on {art.name!r} to resolve '{which}'"
            )
        if which not in ("min", "max"):
            raise ValidationError(f"behavior position must be 'min' or 'max', got {which!r}")
        return float(hi if which == "max" else lo)

    def _couple_behavior(
        self,
        driver_art: Articulation,
        follower_art: Articulation,
        kind: str,
        params: Dict[str, object],
    ) -> Articulation:
        if follower_art.motion_feel is not None:
            raise ValidationError(
                f"motion_feel is not supported on behavior-owned joint {follower_art.name!r}"
            )
        strength = str(params.get("strength", "firm"))
        if strength not in _BEHAVIOR_STIFFNESS:
            raise ValidationError(
                f"behavior strength must be one of {sorted(_BEHAVIOR_STIFFNESS)}, got {strength!r}"
            )
        if kind == "latch_release":
            target = self._limit_by(follower_art, str(params.get("follower_goes", "max")))
            threshold = self._limit_by(driver_art, str(params.get("trigger_at", "max")))
            follower_art.motion_properties = self._with_spring(
                follower_art, _BEHAVIOR_STIFFNESS[strength], target
            )
            behavior = ArticulationBehavior(
                kind=kind,
                driver=driver_art.name,
                target=target,
                threshold=threshold,
                strength=strength,
            )
        elif kind == "self_closer":
            target = self._limit_by(follower_art, str(params.get("closes_to", "min")))
            threshold = self._limit_by(follower_art, str(params.get("threshold", "max")))
            follower_art.motion_properties = self._with_spring(
                follower_art, _BEHAVIOR_STIFFNESS[strength], target
            )
            behavior = ArticulationBehavior(
                kind=kind,
                driver=follower_art.name,
                target=target,
                threshold=threshold,
                strength=strength,
            )
        elif kind == "momentary":
            which = str(params.get("follower_direction", "max"))
            target = self._limit_by(follower_art, which)
            speed_key = str(params.get("speed", "normal"))
            if speed_key not in _BEHAVIOR_SPEED:
                raise ValidationError(
                    f"behavior speed must be one of {sorted(_BEHAVIOR_SPEED)}, got {speed_key!r}"
                )
            speed = _BEHAVIOR_SPEED[speed_key]
            behavior = ArticulationBehavior(
                kind=kind,
                driver=driver_art.name,
                target=target,
                direction=1.0 if which == "max" else -1.0,
                speed=speed,
            )
        else:  # pragma: no cover - guarded by the caller
            raise ValidationError(f"unknown behavior kind {kind!r}")

        follower_art.behavior = behavior
        follower_art.meta["coupling"] = {"kind": kind, "params": dict(params)}
        return follower_art

    @staticmethod
    def _with_spring(art: Articulation, stiffness: float, ref: float) -> MotionProperties:
        mp = art.motion_properties
        return MotionProperties(
            damping=getattr(mp, "damping", None) if mp else None,
            friction=getattr(mp, "friction", None) if mp else None,
            spring_stiffness=stiffness,
            spring_ref=ref,
        )

    def material(
        self,
        name: str,
        *,
        rgba: Optional[Sequence[float]] = None,
        color: Optional[Sequence[float]] = None,
        texture: Optional[str] = None,
        material_class: Optional[str] = None,
        metallic: Optional[float] = None,
        roughness: Optional[float] = None,
        emissive: Optional[float] = None,
        texture_set: Optional[str] = None,
        transmission: Optional[float] = None,
        ior: Optional[float] = None,
        attenuation_color: Optional[Sequence[float]] = None,
        attenuation_distance: Optional[float] = None,
    ) -> Material:
        if rgba is not None and color is not None:
            raise ValidationError("Material cannot set both rgba and color")
        material = Material(
            name=name,
            rgba=_normalize_material_rgba(
                rgba if rgba is not None else color,
                field_name="Material rgba",
            ),
            texture=texture,
            material_class=material_class,
            metallic=metallic,
            roughness=roughness,
            emissive=emissive,
            texture_set=texture_set,
            transmission=transmission,
            ior=ior,
            attenuation_color=attenuation_color,
            attenuation_distance=attenuation_distance,
        )
        self.materials.append(material)
        return material

    def set_assets(
        self,
        assets: Optional[Union[AssetContext, AssetSession, str, Path]],
    ) -> Optional[AssetContext]:
        self.assets = resolve_asset_context(assets)
        for part in self.parts:
            part.assets = self.assets
        return self.assets

    def _rebuild_indices_if_needed(self) -> None:
        self._part_index = {part.name: part for part in self.parts}
        self._articulation_index = {
            articulation.name: articulation for articulation in self.articulations
        }

    def get_part(self, name: Union[str, Part]) -> Part:
        self._rebuild_indices_if_needed()
        key = _part_name_ref(name, field_name="name")
        part = self._part_index.get(key)
        if part is None:
            self._part_index = {p.name: p for p in self.parts}
            part = self._part_index.get(key)
        if part is None:
            raise ValidationError(f"Unknown part: {name!r}")
        return part

    def get_link(self, name: Union[str, Part]) -> Part:
        return self.get_part(name)

    def get_articulation(self, name: Union[str, Articulation]) -> Articulation:
        self._rebuild_indices_if_needed()
        key = _articulation_name_ref(name, field_name="name")
        articulation = self._articulation_index.get(key)
        if articulation is None:
            self._articulation_index = {a.name: a for a in self.articulations}
            articulation = self._articulation_index.get(key)
        if articulation is None:
            raise ValidationError(f"Unknown articulation: {name!r}")
        return articulation

    def get_joint(self, name: Union[str, Articulation]) -> Articulation:
        return self.get_articulation(name)

    def bind_placement(self, target: object, result: object) -> None:

        from .placement import (
            PlacementResult,
            PlacementTarget,
            _bind_placement_target,
            _placement_bind_record,
        )

        if not isinstance(target, PlacementTarget):
            raise ValidationError("target must be a PlacementTarget")
        if not isinstance(result, PlacementResult):
            raise ValidationError("result must be a PlacementResult")
        self._normalize_visual_uids()
        next_ordinal = self._placement_bind_ordinal + 1
        record = _placement_bind_record(
            self,
            target,
            result,
            ordinal=next_ordinal,
        )
        _bind_placement_target(self, target, result)
        self._placement_bind_ordinal = next_ordinal
        target_ref = record["target_ref"]
        if any(
            not existing.get("superseded") and existing.get("target_ref") == target_ref
            for existing in self._placement_bind_records
        ):
            warnings.warn(
                f"bind_placement: {target_ref} was already bound; the earlier placement is "
                "superseded and this one wins. Bind each target once unless the override "
                "is deliberate.",
                UserWarning,
                stacklevel=2,
            )
        self._placement_bind_records = [
            {**existing, "superseded": True}
            if not existing.get("superseded") and existing.get("target_ref") == target_ref
            else existing
            for existing in self._placement_bind_records
        ]
        self._placement_bind_records.append(record)

    def root_parts(self) -> List[Part]:
        child_names = {articulation.child for articulation in self.articulations}
        return [part for part in self.parts if part.name not in child_names]

    def root_links(self) -> List[Part]:
        return self.root_parts()

    def validate(
        self,
        *,
        strict: bool = True,
        strict_mesh_paths: bool = False,
    ) -> None:
        if strict:
            self._normalize_visual_uids()
        if not self.parts:
            raise ValidationError("Articulated object must contain at least one part")

        part_names = [part.name for part in self.parts]
        if len(set(part_names)) != len(part_names):
            raise ValidationError("Part names must be unique")

        articulation_names = [articulation.name for articulation in self.articulations]
        if len(set(articulation_names)) != len(articulation_names):
            raise ValidationError("Articulation names must be unique")
        articulation_lookup = {
            articulation.name: articulation for articulation in self.articulations
        }

        material_names = [material.name for material in self.materials]
        if len(set(material_names)) != len(material_names):
            raise ValidationError("Material names must be unique")
        material_name_set = set(material_names)
        for material in self.materials:
            _validate_material(material, f"material {material.name!r}")

        part_name_set = set(part_names)
        child_to_articulation: Dict[str, Articulation] = {}
        for articulation in self.articulations:
            if articulation.parent not in part_name_set:
                raise ValidationError(
                    f"Articulation {articulation.name!r} references missing parent part {articulation.parent!r}"
                )
            if articulation.child not in part_name_set:
                raise ValidationError(
                    f"Articulation {articulation.name!r} references missing child part {articulation.child!r}"
                )
            if articulation.parent == articulation.child:
                raise ValidationError(
                    f"Articulation {articulation.name!r} parent and child cannot be the same"
                )
            if articulation.child in child_to_articulation:
                raise ValidationError(
                    f"Part {articulation.child!r} has multiple parent articulations: "
                    f"{child_to_articulation[articulation.child].name!r} and {articulation.name!r}"
                )
            child_to_articulation[articulation.child] = articulation

            if strict and articulation.articulation_type in _SCALAR_ARTICULATION_TYPES:
                if len(articulation.axis) != 3:
                    raise ValidationError(
                        f"Articulation {articulation.name!r} axis must have 3 values"
                    )
                if _norm(articulation.axis) == 0.0:
                    raise ValidationError(
                        f"Articulation {articulation.name!r} axis must be non-zero"
                    )

            if strict and articulation.articulation_type in {
                ArticulationType.REVOLUTE,
                ArticulationType.PRISMATIC,
            }:
                if articulation.motion_limits is None:
                    raise ValidationError(
                        f"Articulation {articulation.name!r} must include motion limits"
                    )
                if (
                    articulation.motion_limits.lower is None
                    or articulation.motion_limits.upper is None
                ):
                    raise ValidationError(
                        f"Articulation {articulation.name!r} requires lower and upper limits"
                    )

            if strict and articulation.articulation_type == ArticulationType.CONTINUOUS:
                if articulation.motion_limits is None:
                    raise ValidationError(
                        f"Continuous articulation {articulation.name!r} must set "
                        "motion_limits=MotionLimits(effort=..., velocity=...) and "
                        "cannot set lower/upper limits"
                    )
                if (
                    articulation.motion_limits.lower is not None
                    or articulation.motion_limits.upper is not None
                ):
                    raise ValidationError(
                        f"Articulation {articulation.name!r} cannot include lower/upper limits"
                    )

            if (
                articulation.motion_limits
                and articulation.articulation_type not in _SCALAR_ARTICULATION_TYPES
            ):
                raise ValidationError(
                    f"Articulation {articulation.name!r} does not support motion limits"
                )

            if articulation.motion_limits:
                if (
                    articulation.motion_limits.effort <= 0
                    or articulation.motion_limits.velocity <= 0
                ):
                    raise ValidationError(
                        f"Articulation {articulation.name!r} limits must be positive"
                    )
                if (
                    articulation.motion_limits.lower is not None
                    and articulation.motion_limits.upper is not None
                    and articulation.motion_limits.lower >= articulation.motion_limits.upper
                ):
                    raise ValidationError(
                        f"Articulation {articulation.name!r} requires lower < upper "
                        "(equal limits make it immovable — use ArticulationType.FIXED instead)"
                    )

        _validate_mimic_cycles(articulation_lookup)

        for articulation in self.articulations:
            mimic = articulation.mimic
            if mimic is None:
                continue
            if articulation.motion_feel is not None:
                raise ValidationError(
                    f"Articulation {articulation.name!r} cannot combine motion_feel with a mimic"
                )
            if _motion_domain(articulation.articulation_type) is None:
                raise ValidationError(
                    f"Articulation {articulation.name!r} only supports mimic on "
                    "revolute, continuous, or prismatic articulations"
                )
            source = articulation_lookup.get(mimic.joint)
            if source is None:
                raise ValidationError(
                    f"Articulation {articulation.name!r} mimic references missing articulation "
                    f"{mimic.joint!r}"
                )
            if source.name == articulation.name:
                raise ValidationError(f"Articulation {articulation.name!r} cannot mimic itself")
            if _motion_domain(source.articulation_type) is None:
                raise ValidationError(
                    f"Articulation {articulation.name!r} mimic source {source.name!r} must be "
                    "revolute, continuous, or prismatic"
                )
            if source.mimic is not None:
                raise ValidationError(
                    f"Articulation {articulation.name!r} mimics {source.name!r}, which is itself "
                    "mimic-driven; chained couplings are not supported — couple this follower "
                    "directly to the root driver (compose the multipliers)."
                )
            _validate_coupling_containment(articulation, source, mimic)

        for articulation in self.articulations:
            behavior = articulation.behavior
            if behavior is None:
                continue
            if articulation.motion_feel is not None:
                raise ValidationError(
                    f"Articulation {articulation.name!r} cannot combine motion_feel with a behavior"
                )
            if articulation.mimic is not None:
                raise ValidationError(
                    f"Articulation {articulation.name!r} cannot have both a mimic and a dynamic "
                    "behavior — a joint is either kinematically coupled or behavior-driven."
                )
            if _motion_domain(articulation.articulation_type) is None:
                raise ValidationError(
                    f"Articulation {articulation.name!r} behavior requires a revolute, continuous, "
                    "or prismatic joint"
                )
            if behavior.driver != articulation.name and behavior.driver not in articulation_lookup:
                raise ValidationError(
                    f"Articulation {articulation.name!r} behavior references missing driver "
                    f"{behavior.driver!r}"
                )

        for part in self.parts:
            if part.inertial and part.inertial.mass <= 0:
                raise ValidationError(f"Part {part.name!r} inertial mass must be positive")
            for visual in part.visuals:
                if visual.material is not None:
                    if isinstance(visual.material, str):
                        if str(visual.material) not in material_name_set:
                            raise ValidationError(
                                f"Part {part.name!r} visual references unknown material {visual.material!r}"
                            )
                    elif not isinstance(visual.material, Material):
                        raise ValidationError(
                            f"Part {part.name!r} visual material must be a Material or str name"
                        )
                    else:
                        _validate_material(
                            visual.material,
                            f"part {part.name!r} visual material {visual.material.name!r}",
                        )
                _validate_geometry(
                    visual.geometry,
                    f"part {part.name} visual",
                    strict_mesh_paths=strict_mesh_paths,
                )
            if part.collisions and not bool(getattr(self, _ALLOW_EXPLICIT_COLLISIONS_ATTR, False)):
                raise ValidationError(
                    f"Part {part.name!r} defines explicit collisions. "
                    "Source models must express geometry only through visuals and "
                    "primitive provenance instead of "
                    "authoring Part.collisions directly."
                )
            for collision in part.collisions:
                _validate_geometry(
                    collision.geometry,
                    f"part {part.name} collision",
                    strict_mesh_paths=strict_mesh_paths,
                )
        if strict:
            self._validate_connectivity(part_name_set)

    def to_urdf(
        self,
        *,
        pretty: bool = True,
        asset_root: Optional[Union[str, Path]] = None,
    ) -> str:

        from ._urdf_export import compile_object_to_urdf_xml

        return compile_object_to_urdf_xml(self, pretty=pretty, asset_root=asset_root)

    def _validate_connectivity(self, part_name_set: set[str]) -> None:
        if len(part_name_set) <= 1:
            return

        parent_to_children: Dict[str, List[str]] = {name: [] for name in part_name_set}
        child_set: set[str] = set()
        for articulation in self.articulations:
            parent_to_children.setdefault(articulation.parent, []).append(articulation.child)
            child_set.add(articulation.child)

        roots = sorted(part_name_set - child_set)
        if not roots:
            raise ValidationError(
                "Articulated object has no root part (cycle or every part is a child)"
            )
        if len(roots) > 1:
            raise ValidationError(
                "Articulated object must have exactly one root part; "
                f"found multiple root parts: {roots}"
            )

        visited: set[str] = set()
        queue: List[str] = list(roots)
        while queue:
            current = queue.pop()
            if current in visited:
                continue
            visited.add(current)
            queue.extend(parent_to_children.get(current, []))

        if visited != part_name_set:
            missing = sorted(part_name_set - visited)
            raise ValidationError(
                "Articulated object contains unreachable parts (cycle or broken articulations): "
                f"{missing}"
            )


def _norm(values: Sequence[float]) -> float:
    return sum(float(v) ** 2 for v in values) ** 0.5


def _motion_domain(articulation_type: ArticulationType) -> str | None:
    if articulation_type in _ANGULAR_ARTICULATION_TYPES:
        return "angular"
    if articulation_type == ArticulationType.PRISMATIC:
        return "linear"
    return None


def _validate_coupling_containment(
    follower: Articulation, source: Articulation, mimic: Mimic
) -> None:
    src_ml = source.motion_limits
    src_lo = getattr(src_ml, "lower", None) if src_ml is not None else None
    src_hi = getattr(src_ml, "upper", None) if src_ml is not None else None
    fol_ml = follower.motion_limits
    fol_lo = getattr(fol_ml, "lower", None) if fol_ml is not None else None
    fol_hi = getattr(fol_ml, "upper", None) if fol_ml is not None else None
    if not all(isinstance(v, (int, float)) for v in (src_lo, src_hi, fol_lo, fol_hi)):
        return
    if mimic.poly:
        from ._coupling import evaluate_poly

        n = 17
        vals = [
            evaluate_poly(mimic.poly, float(src_lo) + (float(src_hi) - float(src_lo)) * k / (n - 1))
            for k in range(n)
        ]
        reach_lo, reach_hi = min(vals), max(vals)
        tol = 1e-4
    else:
        a = mimic.multiplier * float(src_lo) + mimic.offset
        b = mimic.multiplier * float(src_hi) + mimic.offset
        reach_lo, reach_hi = (a, b) if a <= b else (b, a)
        tol = 1e-6
    if reach_lo < float(fol_lo) - tol or reach_hi > float(fol_hi) + tol:
        raise ValidationError(
            f"Articulation {follower.name!r} coupling drives the follower to "
            f"[{reach_lo:.4f}, {reach_hi:.4f}], outside its limits "
            f"[{float(fol_lo):.4f}, {float(fol_hi):.4f}]. The coupling has no offset (both "
            "parts rest at q=0), so model the follower at the pose corresponding to the "
            "driver's rest, tighten the driver limits, or use a numeric ratio sized to the "
            "follower travel."
        )


def _validate_mimic_cycles(articulation_lookup: Dict[str, Articulation]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()
    path: list[str] = []

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            cycle_start = path.index(name)
            cycle = path[cycle_start:] + [name]
            raise ValidationError(f"Mimic cycle detected: {' -> '.join(cycle)}")

        visiting.add(name)
        path.append(name)
        mimic = articulation_lookup[name].mimic
        if mimic is not None and mimic.joint in articulation_lookup:
            visit(mimic.joint)
        path.pop()
        visiting.remove(name)
        visited.add(name)

    for name in articulation_lookup:
        visit(name)


def _validate_geometry(
    geometry: Geometry,
    context: str,
    *,
    strict_mesh_paths: bool = False,
) -> None:
    if isinstance(geometry, Box):
        if len(geometry.size) != 3:
            raise ValidationError(f"{context} box size must have 3 values")
        if any(float(v) <= 0 for v in geometry.size):
            raise ValidationError(f"{context} box size must be positive")
    elif isinstance(geometry, Cylinder):
        if geometry.radius <= 0 or geometry.length <= 0:
            raise ValidationError(f"{context} cylinder radius/length must be positive")
    elif isinstance(geometry, Sphere):
        if geometry.radius <= 0:
            raise ValidationError(f"{context} sphere radius must be positive")
    elif isinstance(geometry, Mesh):
        if not geometry.filename and not geometry.name:
            raise ValidationError(f"{context} mesh filename or name is required")
        if strict_mesh_paths:
            filename = str(geometry.filename or "")
            if os.path.isabs(filename):
                raise ValidationError(
                    f"{context} mesh filename must be relative (got absolute path): {filename}"
                )
            if ".." in Path(filename).parts:
                raise ValidationError(
                    f"{context} mesh filename must not contain '..' segments: {filename}"
                )
        if geometry.scale:
            if len(geometry.scale) != 3:
                raise ValidationError(f"{context} mesh scale must have 3 values")
            if any(float(v) <= 0 for v in geometry.scale):
                raise ValidationError(f"{context} mesh scale must be positive")
    else:
        raise ValidationError(f"{context} has unsupported geometry type")


def _validate_material(material: Material, context: str) -> None:
    if not isinstance(material.name, str) or not material.name:
        raise ValidationError(f"{context} name is required")
    if material.rgba is not None and len(material.rgba) not in (3, 4):
        raise ValidationError(f"{context} rgba must have 3 or 4 values")
