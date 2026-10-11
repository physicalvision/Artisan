
from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from .geometry_qc import _find_collision_overlaps_fcl_for_poses
from .types import ArticulationType

if TYPE_CHECKING:
    from .articulated_object import ArticulatedObject

_REST_CONTACT_TOL = 1e-3

_COUPLING_CONTACT_TOL = 5e-3


def _coerce(value: Any) -> ArticulationType:
    return value if isinstance(value, ArticulationType) else ArticulationType(str(value))


def _add_pair(pairs: set[tuple[str, str]], a: Any, b: Any) -> None:
    sa, sb = str(a), str(b)
    if sa and sb and sa != sb:
        pairs.add((sa, sb) if sa <= sb else (sb, sa))


def _coupling_clusters(compiled_model: ArticulatedObject) -> dict[str, set[str]]:
    child_of_joint = {art.name: art.child for art in compiled_model.articulations}
    clusters: dict[str, set[str]] = {}
    for art in compiled_model.articulations:
        m = getattr(art, "mimic", None)
        if m is None:
            continue
        driver_joint = str(getattr(m, "joint", ""))
        cluster = clusters.setdefault(driver_joint, set())
        driver_part = child_of_joint.get(getattr(m, "joint", None))
        if driver_part:
            cluster.add(driver_part)
        cluster.add(art.child)
    return clusters


def _cluster_pairs(members: set[str]) -> list[tuple[str, str]]:
    ordered = sorted(members)
    return [
        (ordered[i], ordered[j]) for i in range(len(ordered)) for j in range(i + 1, len(ordered))
    ]


def _principled_pairs(
    compiled_model: ArticulatedObject,
    declared_overlap_pairs: Iterable[tuple[str, str]] | None,
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()

    for art in compiled_model.articulations:
        _add_pair(pairs, art.parent, art.child)

    for pair in declared_overlap_pairs or ():
        tup = tuple(pair)
        if len(tup) == 2:
            _add_pair(pairs, tup[0], tup[1])

    for cluster in _coupling_clusters(compiled_model).values():
        for a, b in _cluster_pairs(cluster):
            _add_pair(pairs, a, b)

    return pairs


def _rest_overlap_pairs(
    compiled_model: ArticulatedObject,
    asset_root: str | os.PathLike[str] | Path | None,
    rest_contact_tol: float,
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    try:
        rest = {
            art.name: 0.0
            for art in compiled_model.articulations
            if _coerce(art.articulation_type)
            not in (ArticulationType.FIXED, ArticulationType.FLOATING)
            and getattr(art, "mimic", None) is None
        }
        overlaps = _find_collision_overlaps_fcl_for_poses(
            compiled_model,
            asset_root=Path(asset_root) if asset_root is not None else None,
            source_model=compiled_model,
            poses=[rest],
            overlap_tol=0.0,
            overlap_volume_tol=0.0,
            allowed_pairs=None,
            contact_distance_tol=rest_contact_tol,
        )
        for ov in overlaps:
            _add_pair(pairs, ov.link_a, ov.link_b)
    except Exception as exc:  # noqa: BLE001 — best-effort: declared + adjacency still apply
        import warnings

        warnings.warn(
            f"source-4 rest-overlap exemptions skipped ({type(exc).__name__}: {exc}); "
            "MJCF/USD contact exclusions may miss rest-seated or CoACD-piece contacts.",
            stacklevel=2,
        )
    return pairs


@dataclass(frozen=True)
class ContactExemptions:

    pairs: list[tuple[str, str]]
    principled: frozenset[tuple[str, str]]
    source4_only: frozenset[tuple[str, str]]


def contact_exempt_pairs(
    compiled_model: ArticulatedObject,
    *,
    asset_root: str | os.PathLike[str] | Path | None,
    declared_overlap_pairs: Iterable[tuple[str, str]] | None = None,
    rest_contact_tol: float = _REST_CONTACT_TOL,
) -> list[tuple[str, str]]:
    principled = _principled_pairs(compiled_model, declared_overlap_pairs)
    rest = _rest_overlap_pairs(compiled_model, asset_root, rest_contact_tol)
    return sorted(principled | rest)


def contact_exemptions_detailed(
    compiled_model: ArticulatedObject,
    *,
    asset_root: str | os.PathLike[str] | Path | None,
    declared_overlap_pairs: Iterable[tuple[str, str]] | None = None,
    rest_contact_tol: float = _REST_CONTACT_TOL,
) -> ContactExemptions:
    principled = _principled_pairs(compiled_model, declared_overlap_pairs)
    rest = _rest_overlap_pairs(compiled_model, asset_root, rest_contact_tol)
    return ContactExemptions(
        pairs=sorted(principled | rest),
        principled=frozenset(principled),
        source4_only=frozenset(rest - principled),
    )


def _driver_sweep_range(
    art: Any, ranges: dict[str, tuple[float, float]], name: str
) -> tuple[float, float] | None:
    rng = ranges.get(name)
    if rng is not None and rng[1] > rng[0]:
        return rng
    if art is not None:
        ml = getattr(art, "motion_limits", None)
        lo = getattr(ml, "lower", None) if ml is not None else None
        hi = getattr(ml, "upper", None) if ml is not None else None
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and float(hi) > float(lo):
            return float(lo), float(hi)
        if (
            _coerce(getattr(art, "articulation_type", ArticulationType.FIXED))
            == ArticulationType.CONTINUOUS
        ):
            return 0.0, 2.0 * math.pi
    return None


def coupling_contact_pairs(
    compiled_model: ArticulatedObject,
    *,
    asset_root: str | os.PathLike[str] | Path | None,
    contact_tol: float = _COUPLING_CONTACT_TOL,
    samples: int = 12,
) -> set[tuple[str, str]]:
    clusters = _coupling_clusters(compiled_model)
    if not clusters:
        return set()
    all_pairs: set[tuple[str, str]] = set()
    for cluster in clusters.values():
        for a, b in _cluster_pairs(cluster):
            _add_pair(all_pairs, a, b)
    if not all_pairs:
        return set()
    try:
        from ._coupling import driver_effective_ranges
        from .geometry_qc import pair_min_distances_for_poses

        ranges = driver_effective_ranges(compiled_model)
        arts_by_name = {art.name: art for art in compiled_model.articulations}
        base = {
            art.name: 0.0
            for art in compiled_model.articulations
            if _coerce(art.articulation_type)
            not in (ArticulationType.FIXED, ArticulationType.FLOATING)
            and getattr(art, "mimic", None) is None
        }
        root = Path(asset_root) if asset_root is not None else None
        keep: set[tuple[str, str]] = set()
        for driver_joint, members in clusters.items():
            pairs = _cluster_pairs(members)
            if not pairs:
                continue
            rng = _driver_sweep_range(arts_by_name.get(driver_joint), ranges, driver_joint)
            if rng is None or samples < 2:
                vals = [base.get(driver_joint, 0.0)]
            else:
                lo, hi = rng
                vals = [lo + (hi - lo) * k / (samples - 1) for k in range(samples)]
            poses = []
            for v in vals:
                p = dict(base)
                p[driver_joint] = v
                poses.append(p)
            dmins = pair_min_distances_for_poses(
                compiled_model,
                asset_root=root,
                source_model=compiled_model,
                poses=poses,
                pairs=pairs,
            )
            for pair, (dmin, _idx) in dmins.items():
                if dmin <= contact_tol:
                    _add_pair(keep, pair[0], pair[1])
        return keep
    except Exception as exc:  # noqa: BLE001 — conservative: keep every cluster pair
        warnings.warn(
            f"coupling_contact_pairs fell back to excluding ALL cluster pairs "
            f"({type(exc).__name__}: {exc}); a coupled mechanism keeps its full excludes.",
            stacklevel=2,
        )
        return all_pairs
