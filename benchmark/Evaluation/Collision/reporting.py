from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean
from typing import Any

METRICS = ("Single", "Double", "Multi")


def read_metrics(value: Any) -> dict[str, float | None]:
    metrics = value
    if not isinstance(metrics, dict) or set(metrics) != set(METRICS):
        raise ValueError("Collision result must contain Single, Double and Multi. percentages")
    for number in metrics.values():
        if number is not None and (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or not 0 <= number <= 100
        ):
            raise ValueError("Collision percentages must be finite and between 0 and 100")
    return metrics


def summary_table(rows: list[dict[str, float | None]], method: str) -> dict[str, Any]:
    values = {}
    for key in METRICS:
        present = [row[key] for row in rows if row[key] is not None]
        label = "Multi." if key == "Multi" else key
        values[f"{label}↓"] = mean(present) if present else None
    return {"Method": method, "Collision (%)": values}


def format_metric(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}"


def print_table(table: dict[str, Any], dataset: str | None) -> None:
    width = max(len(table["Method"]), len("Method"))
    heading = "Collision (%)" + (f" · {dataset}" if dataset else "")
    metrics = table["Collision (%)"]
    header = f"{'Method':<{width}} | {heading:^31}"
    labels = f"{'':<{width}} | " + "  ".join(f"{key:^9}" for key in metrics)
    values = f"{table['Method']:<{width}} | " + "  ".join(
        f"{format_metric(value):>9}" for value in metrics.values()
    )
    rule = "─" * len(header)
    print(f"\n{rule}\n{header}\n{labels}\n{rule}\n{values}\n{rule}", flush=True)


def write_tables(output: Path, table: dict[str, Any]) -> None:
    metrics = table["Collision (%)"]
    with (output / "summary_collision.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Method", "Collision (%)", "", ""])
        writer.writerow(["", *metrics])
        writer.writerow(
            [
                table["Method"],
                *["" if value is None else format_metric(value) for value in metrics.values()],
            ]
        )
