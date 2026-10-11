from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from .errors import ValidationError
from .mesh import BoxGeometry, MeshGeometry, boolean_difference, boolean_intersection

ShellSide = Literal["full", "left", "right", "center"]

__all__ = [
    "ShellPartitionRegion",
    "ShellPartitionSpec",
    "partition_shell",
]


@dataclass(frozen=True)
class ShellPartitionRegion:
    name: str
    side: ShellSide = "full"
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    z_min: Optional[float] = None
    z_max: Optional[float] = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValidationError("ShellPartitionRegion.name must be non-empty")
        object.__setattr__(self, "name", name)

        if self.side not in {"full", "left", "right", "center"}:
            raise ValidationError(
                "ShellPartitionRegion.side must be one of: full, left, right, center"
            )

        for lo_name, hi_name in (("x_min", "x_max"), ("y_min", "y_max"), ("z_min", "z_max")):
            lo = getattr(self, lo_name)
            hi = getattr(self, hi_name)
            if lo is not None:
                object.__setattr__(self, lo_name, float(lo))
            if hi is not None:
                object.__setattr__(self, hi_name, float(hi))
            lo = getattr(self, lo_name)
            hi = getattr(self, hi_name)
            if lo is not None and hi is not None and float(lo) >= float(hi):
                raise ValidationError(f"{lo_name} must be < {hi_name}")


@dataclass(frozen=True)
class ShellPartitionSpec:
    shell: MeshGeometry
    regions: tuple[ShellPartitionRegion, ...]
    splitters: tuple[MeshGeometry, ...] = ()
    remainder_name: Optional[str] = None
    center_gap: float = 0.0
    padding: float = 0.01

    def __post_init__(self) -> None:
        if not isinstance(self.shell, MeshGeometry):
            raise ValidationError("ShellPartitionSpec.shell must be MeshGeometry")
        if not self.shell.vertices or not self.shell.faces:
            raise ValidationError("ShellPartitionSpec.shell must be a non-empty manifold solid")

        regions = tuple(
            region if isinstance(region, ShellPartitionRegion) else ShellPartitionRegion(**region)  # type: ignore[arg-type]
            for region in self.regions
        )
        if not regions:
            raise ValidationError("ShellPartitionSpec.regions must contain at least one region")
        names = [region.name for region in regions]
        if len(set(names)) != len(names):
            raise ValidationError("ShellPartitionSpec.regions must have unique names")
        object.__setattr__(self, "regions", regions)

        splitters = tuple(self.splitters)
        for idx, splitter in enumerate(splitters):
            if not isinstance(splitter, MeshGeometry):
                raise ValidationError(f"splitters[{idx}] must be MeshGeometry")
            if not splitter.vertices or not splitter.faces:
                raise ValidationError(f"splitters[{idx}] must be non-empty")
        object.__setattr__(self, "splitters", splitters)

        center_gap = float(self.center_gap)
        if center_gap < 0.0:
            raise ValidationError("center_gap must be >= 0")
        object.__setattr__(self, "center_gap", center_gap)

        padding = float(self.padding)
        if padding <= 0.0:
            raise ValidationError("padding must be > 0")
        object.__setattr__(self, "padding", padding)

        if self.remainder_name is not None:
            remainder_name = str(self.remainder_name).strip()
            if not remainder_name:
                raise ValidationError("remainder_name must be non-empty when provided")
            if remainder_name in set(names):
                raise ValidationError("remainder_name must not duplicate any region name")
            object.__setattr__(self, "remainder_name", remainder_name)


def _geometry_bounds(
    geometry: MeshGeometry,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if not geometry.vertices:
        raise ValidationError("geometry must contain at least one vertex")
    xs = [float(v[0]) for v in geometry.vertices]
    ys = [float(v[1]) for v in geometry.vertices]
    zs = [float(v[2]) for v in geometry.vertices]
    return (
        (min(xs), min(ys), min(zs)),
        (max(xs), max(ys), max(zs)),
    )


def _region_box(
    region: ShellPartitionRegion,
    *,
    shell_bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    center_gap: float,
    padding: float,
) -> MeshGeometry:
    (mn, mx) = shell_bounds
    x_min = mn[0] - padding if region.x_min is None else float(region.x_min)
    x_max = mx[0] + padding if region.x_max is None else float(region.x_max)
    y_min = mn[1] - padding if region.y_min is None else float(region.y_min)
    y_max = mx[1] + padding if region.y_max is None else float(region.y_max)
    z_min = mn[2] - padding if region.z_min is None else float(region.z_min)
    z_max = mx[2] + padding if region.z_max is None else float(region.z_max)

    if region.side == "left":
        x_max = min(x_max, -0.5 * center_gap)
    elif region.side == "right":
        x_min = max(x_min, 0.5 * center_gap)
    elif region.side == "center":
        if center_gap <= 0.0:
            raise ValidationError("center_gap must be > 0 when using side='center'")
        x_min = max(x_min, -0.5 * center_gap)
        x_max = min(x_max, 0.5 * center_gap)

    if x_min >= x_max:
        raise ValidationError(f"Partition region {region.name!r} collapses on X")
    if y_min >= y_max:
        raise ValidationError(f"Partition region {region.name!r} collapses on Y")
    if z_min >= z_max:
        raise ValidationError(f"Partition region {region.name!r} collapses on Z")

    box = BoxGeometry((x_max - x_min, y_max - y_min, z_max - z_min))
    box.translate(
        0.5 * (x_min + x_max),
        0.5 * (y_min + y_max),
        0.5 * (z_min + z_max),
    )
    return box


def _coerce_partition_spec(
    spec: ShellPartitionSpec | MeshGeometry,
    /,
    **overrides,
) -> ShellPartitionSpec:
    if isinstance(spec, ShellPartitionSpec):
        if overrides:
            return ShellPartitionSpec(**({**spec.__dict__, **overrides}))
        return spec
    if "regions" not in overrides:
        raise ValidationError("partition_shell(shell, ...) requires regions=...")
    return ShellPartitionSpec(shell=spec, **overrides)


def partition_shell(
    spec: ShellPartitionSpec | MeshGeometry,
    /,
    **overrides,
) -> dict[str, MeshGeometry]:

    spec = _coerce_partition_spec(spec, **overrides)
    working = spec.shell.copy()
    for splitter in spec.splitters:
        working = boolean_difference(working, splitter)

    bounds = _geometry_bounds(working)
    parts: dict[str, MeshGeometry] = {}
    remaining = working

    for region in spec.regions:
        cutter = _region_box(
            region,
            shell_bounds=bounds,
            center_gap=spec.center_gap,
            padding=spec.padding,
        )
        piece = boolean_intersection(remaining, cutter)
        if not piece.vertices or not piece.faces:
            raise ValidationError(f"Partition region {region.name!r} did not capture any geometry")
        parts[region.name] = piece
        remaining = boolean_difference(remaining, piece)

    if spec.remainder_name is not None:
        parts[spec.remainder_name] = remaining

    return parts
