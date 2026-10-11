from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean
from typing import Any

APPEARANCE = (
    ("IoU↑", "mask_iou"),
    ("SSIM↑", "ssim"),
    ("PSNR↑", "psnr_full"),
    ("LPIPS↓", "lpips"),
    ("CLIP↑", "clip"),
)
MATERIAL = (("Base Color↑", "base_color"), ("Metallic↑", "metallic"), ("Roughness↑", "roughness"))


def metric_kinds(mode: str, dataset: str | None = None) -> tuple[str, ...]:
    if dataset == "LightWheel":
        if mode.startswith("material_"):
            raise ValueError(f"--mode {mode} is ArtVIP-only; LightWheel supports 2d, 3d and all")
        if mode == "all":
            return ("2d", "3d")
    choices = {
        "2d": ("2d",),
        "3d": ("3d",),
        "material_direct": ("material_direct",),
        "material_transfer": ("material_transfer",),
        "all": ("2d", "3d", "material_direct", "material_transfer"),
    }
    return choices[mode]


GEOMETRY = (("oCD↓", "object_cd"), ("pCD↓", "part_cd"), ("pIoU↑", "part_miou"))


def metric_mean(
    results: list[dict[str, Any]], kind: str, states: tuple[str, ...], key: str
) -> float | None:
    if not results:
        return None
    objects = []
    for result in results:
        values = []
        for state in states:
            metrics = result.get(kind, {}).get(state, {})
            if kind == "2d":
                metrics = metrics.get("mean", {})
            value = metrics.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                values = []
                break
            values.append(value)
        if values:
            objects.append(mean(values))
    return float(mean(objects)) if objects else None


def summary_table(
    results: list[dict[str, Any]], method: str, mode: str = "all", dataset: str | None = None
) -> dict[str, Any]:
    kinds = metric_kinds(mode, dataset)
    table: dict[str, Any] = {"Method": method}
    if "2d" in kinds:
        table["Appearance"] = {
            label: metric_mean(results, "2d", ("close", "open"), key) for label, key in APPEARANCE
        }
    if "3d" in kinds:
        table["Rest-Pose Geometry"] = {
            label: metric_mean(results, "3d", ("close",), key) for label, key in GEOMETRY
        }
        table["Articulated Geometry"] = {
            label: metric_mean(results, "3d", ("open",), key) for label, key in GEOMETRY
        }
    for kind, title in (
        ("material_direct", "Material_Direct"),
        ("material_transfer", "Material_Transfer"),
    ):
        if kind in kinds:
            table[title] = {}
            for label, key in MATERIAL:
                values = [row.get(kind, {}).get("aggregate", {}).get(key) for row in results]
                valid = [
                    float(v)
                    for v in values
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                ]
                table[title][label] = mean(valid) if valid else None
    return table


def format_metric(label: str, value: float | None) -> str:
    if value is None:
        return "—"
    return format(
        value, ".2f" if label in {"PSNR↑", "Base Color↑", "Metallic↑", "Roughness↑"} else ".4f"
    )


def print_table(summary: dict[str, Any]) -> None:
    table = summary["table"]
    groups = tuple(group for group in table if group != "Method")
    method_width = max(len("Method"), len(table["Method"]))
    column_width = max(8, max(len(label) for group in groups for label in table[group]))
    separator = " | "
    header = separator.join(
        ["Method".ljust(method_width)]
        + [group.center(len(table[group]) * (column_width + 2) - 2) for group in groups]
    )
    labels = separator.join(
        [" " * method_width]
        + ["  ".join(label.center(column_width) for label in table[group]) for group in groups]
    )
    values = separator.join(
        [table["Method"].ljust(method_width)]
        + [
            "  ".join(
                format_metric(label, value).rjust(column_width)
                for label, value in table[group].items()
            )
            for group in groups
        ]
    )
    rule = "─" * len(header)
    print(f"\n{rule}\n{header}\n{labels}\n{rule}\n{values}\n{rule}", flush=True)


def write_tables(output: Path, summary: dict[str, Any], name: str) -> None:
    table = summary["table"]
    groups = tuple(group for group in table if group != "Method")
    labels = [label for group in groups for label in table[group]]
    values = [value for group in groups for value in table[group].values()]
    with (output / f"{name}.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        group_row = ["Method"]
        for group in groups:
            group_row.extend([group, *[""] * (len(table[group]) - 1)])
        writer.writerow(group_row)
        writer.writerow(["", *labels])
        writer.writerow(
            [
                table["Method"],
                *[
                    "" if value is None else format_metric(label, value)
                    for label, value in zip(labels, values, strict=True)
                ],
            ]
        )
