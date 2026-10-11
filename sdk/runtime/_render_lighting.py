from __future__ import annotations

from dataclasses import dataclass

Vec3 = tuple[float, float, float]

REVIEW_VIEW_TRANSFORM = "Khronos PBR Neutral"
REVIEW_WORLD_COLOR = (0.55, 0.55, 0.58, 1.0)
REVIEW_WORLD_STRENGTH = 0.8
REVIEW_LIGHT_PREFIX = "artisan_review_"


@dataclass(frozen=True, slots=True)
class AreaLightSpec:

    name: str
    location: Vec3
    energy: float
    size: float


_AREA_LIGHT_POLICY: tuple[tuple[str, Vec3, float, float], ...] = (
    ("key", (2.5, 2.0, 3.5), 110.0, 3.0),
    ("fill", (-3.0, 1.0, 2.0), 55.0, 4.0),
    ("rim", (0.0, -3.0, 2.5), 70.0, 3.5),
)


def review_area_light_specs(center: Vec3, radius: float) -> tuple[AreaLightSpec, ...]:

    safe_radius = max(float(radius), 1e-3)
    radius_sq = safe_radius * safe_radius
    return tuple(
        AreaLightSpec(
            name=f"{REVIEW_LIGHT_PREFIX}{name}",
            location=tuple(
                float(center[index]) + safe_radius * float(offset[index]) for index in range(3)
            ),
            energy=float(energy_per_radius_sq) * radius_sq,
            size=float(size_scale) * safe_radius,
        )
        for name, offset, energy_per_radius_sq, size_scale in _AREA_LIGHT_POLICY
    )
