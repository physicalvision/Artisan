from __future__ import annotations

from pathlib import Path

SDK_ROOT = Path(__file__).resolve().parent

_CURATED_EXAMPLES = (
    "_examples/cadquery/a_parametric_enclosure.md",
    "_examples/cadquery/chained_pinned_arm_with_cadquery_shells.md",
    "_examples/cadquery/shelling_to_create_thin_features.md",
    "_examples/cadquery/single_continuous_rotary_shaft.md",
    "_examples/cadquery/the_classic_occ_bottle.md",
    "_examples/base/through_axis_swivel_cover_fork.md",
    "_examples/cadquery/prismaticrevolute_chain.md",
    "_examples/cadquery/orthogonal_xy_stage.md",
)


def curated_example_paths() -> tuple[Path, ...]:
    return tuple(SDK_ROOT / relative for relative in _CURATED_EXAMPLES)
