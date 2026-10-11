from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence, Union

from ..errors import ValidationError
from .common import (
    ArticulationType,
    _aabb_axis_gap,
    _aabb_axis_overlap,
    _aabb_axis_separation,
    _aabb_center,
    _axes_label,
    _axis_index,
    _named_ref,
    _normalize_axes_spec,
    _normalize_axis_name,
    _normalize_direction_name,
    _point_distance_on_axes,
)


class TestContextExpectationMixin:
    def expect_gate_witness_clear(
        self,
        link_a: object,
        link_b: object,
        *,
        witness_pose: Mapping[object, object],
        expected_policy_id: str,
        expected_penetration_depth_tol_m: float,
        expected_collision_mode: str,
        name: Optional[str] = None,
    ) -> bool:
        from ..geometry_qc import (  # noqa: PLC0415 - keep testing imports acyclic
            FULL_COLLISION_EPA_POLICY_ID,
            default_penetration_depth_tol_from_env,
            evaluate_full_collision_pair_poses,
        )
        from ..self_collision import (  # noqa: PLC0415 - lazy compile-policy dependency
            compile_for_motion_screen,
            motion_screen_collision_mode,
        )

        link_a_name = _named_ref(link_a, kind="link_a")
        link_b_name = _named_ref(link_b, kind="link_b")
        check_name = name or f"expect_gate_witness_clear({link_a_name},{link_b_name})"
        actual_threshold = default_penetration_depth_tol_from_env()
        try:
            expected_threshold = float(expected_penetration_depth_tol_m)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "gate witness penetration-depth provenance must be a finite number"
            ) from exc
        provenance_ok = expected_policy_id == FULL_COLLISION_EPA_POLICY_ID and math.isclose(
            expected_threshold,
            actual_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        if not provenance_ok:
            raise ValidationError(
                "gate witness provenance mismatch: "
                f"expected=({expected_policy_id},{expected_threshold:.9g},"
                f"{expected_collision_mode}) active=({FULL_COLLISION_EPA_POLICY_ID},"
                f"{actual_threshold:.9g},unknown-before-compile); expected values validate provenance "
                "and cannot override the active policy"
            )
        missing = [name for name in (link_a_name, link_b_name) if name not in self._parts_by_name()]
        if missing:
            return self._record(check_name, False, f"missing part(s): {missing}")
        try:
            with self.pose(dict(witness_pose)):
                requested_mode = motion_screen_collision_mode()
                compiled = self._motion_screen_compiled_model_cache.get(requested_mode)
                if compiled is None:
                    compiled = compile_for_motion_screen(
                        self.model, asset_root=self._asset_root(), validate=False
                    )
                    self._motion_screen_compiled_model_cache[requested_mode] = compiled
                actual_mode = motion_screen_collision_mode(compiled)
                if expected_collision_mode != actual_mode:
                    raise ValidationError(
                        "gate witness provenance mismatch: "
                        f"expected collision_mode={expected_collision_mode!r}, "
                        f"actual collision_mode={actual_mode!r}; the collision compiler "
                        "may have fallen back and the witness must be regenerated"
                    )
                evaluation = evaluate_full_collision_pair_poses(
                    compiled,
                    asset_root=self._asset_root(),
                    source_model=self.model,
                    pair_poses=[((link_a_name, link_b_name), dict(self._pose))],
                )[0]
        except ValidationError:
            raise
        except Exception as exc:  # noqa: BLE001 - replay failures are tester errors
            raise ValidationError(f"could not replay gate witness: {exc}") from exc
        return self._record(
            check_name,
            evaluation.clear,
            f"policy_id={evaluation.policy_id} collision_mode={actual_mode} "
            f"max_penetration_depth_m="
            f"{evaluation.max_penetration_depth_m if evaluation.max_penetration_depth_m is not None else 'unavailable'} "
            f"penetration_depth_tol_m={evaluation.penetration_depth_tol_m:.9g} "
            f"depth_method={evaluation.depth_method} depth_signed={evaluation.depth_signed} "
            f"depth_degenerate={evaluation.depth_degenerate} "
            f"elements={evaluation.elements} pose={evaluation.pose}",
        )

    def expect_origin_distance(
        self,
        link_a: object,
        link_b: object,
        *,
        axes: Union[str, Sequence[str]] = "xy",
        min_dist: float = 0.0,
        max_dist: Optional[float] = None,
        name: Optional[str] = None,
    ) -> bool:
        link_a_name = _named_ref(link_a, kind="link_a")
        link_b_name = _named_ref(link_b, kind="link_b")
        axes_key, axes_err = _normalize_axes_spec(axes, field_name="axes")
        check_name = (
            name
            or f"expect_origin_distance({link_a_name},{link_b_name},axes={_axes_label(axes_key) or axes})"
        )
        if axes_err is not None:
            return self._record(check_name, False, axes_err)
        min_dist_f = float(min_dist)
        if min_dist_f < 0.0:
            return self._record(check_name, False, "min_dist must be >= 0")
        max_dist_f = None if max_dist is None else float(max_dist)
        if max_dist_f is not None and max_dist_f < min_dist_f:
            return self._record(check_name, False, "max_dist must be >= min_dist")

        pa = self.link_world_position(link_a)
        pb = self.link_world_position(link_b)
        if pa is None or pb is None:
            return self._record(check_name, False, "missing link world position(s)")

        dist = _point_distance_on_axes(pa, pb, axes=axes_key)
        ok = dist >= min_dist_f
        if max_dist_f is not None:
            ok = ok and dist <= max_dist_f
        upper_txt = "inf" if max_dist_f is None else f"{max_dist_f:.4g}"
        return self._record(
            check_name,
            ok,
            f"origin_dist[{_axes_label(axes_key)}]({link_a_name!r},{link_b_name!r})={dist:.4g} "
            f"min_dist={min_dist_f:.4g} max_dist={upper_txt} pose={self._pose}",
        )

    def expect_origin_gap(
        self,
        positive_link: object,
        negative_link: object,
        *,
        axis: str,
        min_gap: float = 0.0,
        max_gap: Optional[float] = None,
        name: Optional[str] = None,
    ) -> bool:
        positive_name = _named_ref(positive_link, kind="positive_link")
        negative_name = _named_ref(negative_link, kind="negative_link")
        axis_key, axis_sign, axis_err = _normalize_axis_name(axis)
        check_name = (
            name or f"expect_origin_gap({positive_name},{negative_name},axis={axis_key or axis})"
        )
        if axis_err is not None or axis_key is None or axis_sign < 0.0:
            return self._record(check_name, False, axis_err or "axis must be one of: x, y, z")
        min_gap_f = float(min_gap)
        max_gap_f = None if max_gap is None else float(max_gap)
        if max_gap_f is not None and max_gap_f < min_gap_f:
            return self._record(check_name, False, "max_gap must be >= min_gap")

        positive_pos = self.link_world_position(positive_link)
        negative_pos = self.link_world_position(negative_link)
        if positive_pos is None or negative_pos is None:
            return self._record(check_name, False, "missing link world position(s)")

        idx = _axis_index(axis_key)
        gap = float(positive_pos[idx]) - float(negative_pos[idx])
        ok = gap >= min_gap_f
        if max_gap_f is not None:
            ok = ok and gap <= max_gap_f
        upper_txt = "inf" if max_gap_f is None else f"{max_gap_f:.4g}"
        return self._record(
            check_name,
            ok,
            f"origin_gap_{axis_key}={gap:.4g} min_gap={min_gap_f:.4g} max_gap={upper_txt} pose={self._pose}",
        )

    def expect_contact(
        self,
        link_a: object,
        link_b: object,
        *,
        contact_tol: float = 1e-6,
        elem_a: Optional[object] = None,
        elem_b: Optional[object] = None,
        name: Optional[str] = None,
    ) -> bool:
        link_a_name = _named_ref(link_a, kind="link_a")
        link_b_name = _named_ref(link_b, kind="link_b")
        check_name = name or f"expect_contact({link_a_name},{link_b_name})"
        contact_tol_f = float(contact_tol)
        if contact_tol_f < 0.0:
            return self._record(check_name, False, "contact_tol must be >= 0")

        elements_a, _resolved_a, elem_a_name, error_a = self._resolve_exact_elements(
            link_a,
            elem=elem_a,
            kind_prefix="elem_a",
        )
        elements_b, _resolved_b, elem_b_name, error_b = self._resolve_exact_elements(
            link_b,
            elem=elem_b,
            kind_prefix="elem_b",
        )
        if error_a or error_b or elements_a is None or elements_b is None:
            errors = [item for item in (error_a, error_b) if item]
            return self._record(check_name, False, "; ".join(errors))

        min_distance, collided = self._exact_pair_distance(elements_a, elements_b)
        ok = collided or min_distance <= contact_tol_f
        return self._record(
            check_name,
            ok,
            f"min_distance={min_distance:.4g} contact_tol={contact_tol_f:.4g} "
            f"elem_a={elem_a_name!r} elem_b={elem_b_name!r} pose={self._pose}",
        )

    def expect_gap(
        self,
        positive_link: object,
        negative_link: object,
        *,
        axis: str,
        min_gap: Optional[float] = None,
        max_gap: Optional[float] = None,
        max_penetration: Optional[float] = None,
        positive_elem: Optional[object] = None,
        negative_elem: Optional[object] = None,
        elem_a: Optional[object] = None,
        elem_b: Optional[object] = None,
        name: Optional[str] = None,
    ) -> bool:
        positive_name = _named_ref(positive_link, kind="positive_link")
        negative_name = _named_ref(negative_link, kind="negative_link")
        axis_key, axis_sign, axis_err = _normalize_axis_name(axis)
        check_name = name or f"expect_gap({positive_name},{negative_name},axis={axis_key or axis})"
        if axis_err is not None or axis_key is None or axis_sign < 0.0:
            return self._record(check_name, False, axis_err or "axis must be one of: x, y, z")
        if positive_elem is not None and elem_a is not None:
            raise TypeError("expect_gap() accepts only one of 'positive_elem' or alias 'elem_a'")
        if negative_elem is not None and elem_b is not None:
            raise TypeError("expect_gap() accepts only one of 'negative_elem' or alias 'elem_b'")
        if elem_a is not None:
            positive_elem = elem_a
        if elem_b is not None:
            negative_elem = elem_b

        positive_elements, _resolved_positive, positive_elem_name, positive_error = (
            self._resolve_exact_elements(
                positive_link,
                elem=positive_elem,
                kind_prefix="positive",
            )
        )
        negative_elements, _resolved_negative, negative_elem_name, negative_error = (
            self._resolve_exact_elements(
                negative_link,
                elem=negative_elem,
                kind_prefix="negative",
            )
        )
        if (
            positive_error
            or negative_error
            or positive_elements is None
            or negative_elements is None
        ):
            errors = [item for item in (positive_error, negative_error) if item]
            return self._record(check_name, False, "; ".join(errors))

        positive_min, _positive_max = self._elements_projection_interval(
            positive_elements,
            axis=axis_key,
        )
        _negative_min, negative_max = self._elements_projection_interval(
            negative_elements,
            axis=axis_key,
        )

        if min_gap is None:
            max_penetration_f = 0.0 if max_penetration is None else float(max_penetration)
            min_gap_f = -max_penetration_f
        else:
            min_gap_f = float(min_gap)
            if max_penetration is not None:
                max_penetration_f = float(max_penetration)
                expected_min_gap = -max_penetration_f
                if abs(expected_min_gap - min_gap_f) > 1e-9:
                    self.warn(
                        "expect_gap received both min_gap and max_penetration with different bounds; "
                        "using min_gap as the lower bound."
                    )
            else:
                max_penetration_f = max(0.0, -min_gap_f)
        max_gap_f = None if max_gap is None else float(max_gap)
        if max_gap_f is not None and max_gap_f < min_gap_f:
            return self._record(check_name, False, "max_gap must be >= min_gap")

        gap = positive_min - negative_max
        ok = gap >= min_gap_f
        if max_gap_f is not None:
            ok = ok and gap <= max_gap_f
        upper_txt = "inf" if max_gap_f is None else f"{max_gap_f:.4g}"
        return self._record(
            check_name,
            ok,
            f"gap_{axis_key}={gap:.4g} min_gap={min_gap_f:.4g} max_gap={upper_txt} "
            f"max_penetration={max_penetration_f:.4g} "
            f"positive_elem={positive_elem_name!r} negative_elem={negative_elem_name!r} "
            f"pose={self._pose}",
        )

    def expect_overlap(
        self,
        link_a: object,
        link_b: object,
        *,
        axes: Union[str, Sequence[str]] = "xy",
        min_overlap: float = 0.0,
        elem_a: Optional[object] = None,
        elem_b: Optional[object] = None,
        name: Optional[str] = None,
    ) -> bool:
        link_a_name = _named_ref(link_a, kind="link_a")
        link_b_name = _named_ref(link_b, kind="link_b")
        axes_key, axes_err = _normalize_axes_spec(axes, field_name="axes")
        check_name = (
            name
            or f"expect_overlap({link_a_name},{link_b_name},axes={_axes_label(axes_key) or axes})"
        )
        if axes_err is not None:
            return self._record(check_name, False, axes_err)

        elements_a, _resolved_a, elem_a_name, error_a = self._resolve_exact_elements(
            link_a,
            elem=elem_a,
            kind_prefix="elem_a",
        )
        elements_b, _resolved_b, elem_b_name, error_b = self._resolve_exact_elements(
            link_b,
            elem=elem_b,
            kind_prefix="elem_b",
        )
        if error_a or error_b or elements_a is None or elements_b is None:
            errors = [item for item in (error_a, error_b) if item]
            return self._record(check_name, False, "; ".join(errors))

        min_overlap_f = float(min_overlap)
        ok = True
        axis_details: list[str] = []
        for axis_name in axes_key:
            min_a, max_a = self._elements_projection_interval(elements_a, axis=axis_name)
            min_b, max_b = self._elements_projection_interval(elements_b, axis=axis_name)
            overlap = min(max_a, max_b) - max(min_a, min_b)
            axis_ok = overlap >= min_overlap_f
            ok = ok and axis_ok
            axis_details.append(f"overlap_{axis_name}={overlap:.4g}")

        return self._record(
            check_name,
            ok,
            " ".join(axis_details)
            + f" min_overlap={min_overlap_f:.4g} elem_a={elem_a_name!r} elem_b={elem_b_name!r} pose={self._pose}",
        )

    def expect_within(
        self,
        inner_link: object,
        outer_link: object,
        *,
        axes: Union[str, Sequence[str]] = "xy",
        margin: float = 0.0,
        inner_elem: Optional[object] = None,
        outer_elem: Optional[object] = None,
        elem_a: Optional[object] = None,
        elem_b: Optional[object] = None,
        name: Optional[str] = None,
    ) -> bool:
        inner_name = _named_ref(inner_link, kind="inner_link")
        outer_name = _named_ref(outer_link, kind="outer_link")
        axes_key, axes_err = _normalize_axes_spec(axes, field_name="axes")
        check_name = (
            name or f"expect_within({inner_name},{outer_name},axes={_axes_label(axes_key) or axes})"
        )
        if axes_err is not None:
            return self._record(check_name, False, axes_err)
        if inner_elem is not None and elem_a is not None:
            raise TypeError("expect_within() accepts only one of 'inner_elem' or alias 'elem_a'")
        if outer_elem is not None and elem_b is not None:
            raise TypeError("expect_within() accepts only one of 'outer_elem' or alias 'elem_b'")
        if elem_a is not None:
            inner_elem = elem_a
        if elem_b is not None:
            outer_elem = elem_b

        inner_elements, _resolved_inner, inner_elem_name, inner_error = (
            self._resolve_exact_elements(
                inner_link,
                elem=inner_elem,
                kind_prefix="inner",
            )
        )
        outer_elements, _resolved_outer, outer_elem_name, outer_error = (
            self._resolve_exact_elements(
                outer_link,
                elem=outer_elem,
                kind_prefix="outer",
            )
        )
        if inner_error or outer_error or inner_elements is None or outer_elements is None:
            errors = [item for item in (inner_error, outer_error) if item]
            return self._record(check_name, False, "; ".join(errors))

        margin_f = float(margin)
        ok = True
        axis_details: list[str] = []
        for axis_name in axes_key:
            inner_min, inner_max = self._elements_projection_interval(
                inner_elements, axis=axis_name
            )
            outer_min, outer_max = self._elements_projection_interval(
                outer_elements, axis=axis_name
            )
            axis_ok = inner_min >= outer_min - margin_f and inner_max <= outer_max + margin_f
            ok = ok and axis_ok
            axis_details.append(
                f"{axis_name}=({inner_min:.4g},{inner_max:.4g}) in ({outer_min:.4g},{outer_max:.4g})"
            )

        return self._record(
            check_name,
            ok,
            f"inner={inner_name!r} outer={outer_name!r} axes={_axes_label(axes_key)} "
            f"margin={margin_f:.4g} inner_elem={inner_elem_name!r} outer_elem={outer_elem_name!r} "
            + " ".join(axis_details)
            + f" pose={self._pose}",
        )

    def expect_coupled_motion(
        self,
        follower: object,
        driver: object,
        *,
        sense: str = "same",
        min_ratio: float = 0.0,
        q0: Optional[float] = None,
        q1: Optional[float] = None,
        name: Optional[str] = None,
    ) -> bool:
        f_name = _named_ref(follower, kind="follower")
        d_name = _named_ref(driver, kind="driver")
        check_name = name or f"expect_coupled_motion({f_name}<-{d_name},{sense})"

        sense_key = str(sense).strip().lower()
        if sense_key not in {"same", "opposite"}:
            return self._record(check_name, False, "sense must be 'same' or 'opposite'")

        def _find(jname: str) -> object:
            getter = getattr(self.model, "get_articulation", None) or getattr(
                self.model, "get_joint", None
            )
            if callable(getter):
                try:
                    obj = getter(jname)
                except Exception:  # noqa: BLE001 - fall back to a linear scan
                    obj = None
                if obj is not None:
                    return obj
            arts = getattr(self.model, "articulations", None)
            if not isinstance(arts, list):
                arts = getattr(self.model, "joints", None)
            if isinstance(arts, list):
                for j in arts:
                    if getattr(j, "name", None) == jname:
                        return j
            return None

        driver_obj = _find(d_name)
        follower_obj = _find(f_name)
        if driver_obj is None:
            return self._record(check_name, False, f"driver joint {d_name!r} not found")
        if follower_obj is None:
            return self._record(check_name, False, f"follower joint {f_name!r} not found")

        mimic = getattr(follower_obj, "mimic", None)
        src = getattr(mimic, "joint", None) if mimic is not None else None
        if src != d_name:
            return self._record(
                check_name,
                False,
                f"follower {f_name!r} is not coupled to driver {d_name!r} "
                f"(its mimic source is {src!r}); wire it with model.couple({d_name!r}, {f_name!r}, ...)",
            )
        if getattr(driver_obj, "mimic", None) is not None:
            return self._record(
                check_name,
                False,
                f"driver {d_name!r} is itself mimic-driven (chained couplings are not "
                "supported); couple the follower directly to the root driver",
            )

        if q0 is None or q1 is None:
            jt = getattr(driver_obj, "joint_type", None)
            lim = getattr(driver_obj, "limit", None)
            if jt == ArticulationType.CONTINUOUS:
                dq0, dq1 = 0.0, 1.5707963267948966
            else:
                lower = None if lim is None else getattr(lim, "lower", None)
                upper = None if lim is None else getattr(lim, "upper", None)
                if lower is None or upper is None:
                    return self._record(
                        check_name,
                        False,
                        f"driver {d_name!r} has no finite limits; provide q0 and q1 explicitly",
                    )
                dq0, dq1 = float(lower), float(upper)
        else:
            dq0, dq1 = float(q0), float(q1)
        if abs(dq1 - dq0) <= 1e-12:
            return self._record(check_name, False, "q0 and q1 must differ")

        from ..geometry_qc import _resolve_joint_positions  # noqa: PLC0415 - avoid import cycle

        base = {k: v for k, v in dict(self._pose).items() if k != f_name}
        try:
            r0 = _resolve_joint_positions(self.model, {**base, d_name: dq0})
            r1 = _resolve_joint_positions(self.model, {**base, d_name: dq1})
        except Exception as exc:  # noqa: BLE001 - surfaced as a check failure
            return self._record(check_name, False, f"could not resolve poses: {exc}")

        fv0, fv1 = r0.get(f_name), r1.get(f_name)
        if not isinstance(fv0, (int, float)) or not isinstance(fv1, (int, float)):
            return self._record(check_name, False, f"follower {f_name!r} value is non-scalar")
        delta_f = float(fv1) - float(fv0)
        delta_d = dq1 - dq0
        ratio = abs(delta_f / delta_d) if delta_d != 0 else 0.0

        if abs(delta_f) <= 1e-9:
            return self._record(
                check_name,
                False,
                f"follower {f_name!r} does not move when driver {d_name!r} moves "
                f"(delta_follower={delta_f:.4g}); coupling missing or not wired",
            )
        same_sign = (delta_f >= 0) == (delta_d >= 0)
        sign_ok = same_sign if sense_key == "same" else (not same_sign)
        ratio_ok = ratio >= float(min_ratio)
        details = (
            f"sense={sense_key} ratio={ratio:.4g} (min {float(min_ratio):.4g}); "
            f"delta_follower={delta_f:.4g} delta_driver={delta_d:.4g} q0={dq0:.4g} q1={dq1:.4g}"
        )
        if not sign_ok:
            moved = "with" if same_sign else "against"
            details += f" | wrong direction: follower moved {moved} the driver"
        return self._record(check_name, sign_ok and ratio_ok, details)
