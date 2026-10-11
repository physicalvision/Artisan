from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from artisan.benchmark import dataset_cases

METRICS = ("Rest", "Open")


def read_metrics(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not set(METRICS).issubset(value):
        raise ValueError("Stability results must contain Rest and Open")
    if any(value[key] is not None and not isinstance(value[key], bool) for key in METRICS):
        raise ValueError("Stability results must be boolean or null")
    return value


def summary_table(
    rows: list[dict[str, Any]], method: str, roster_size: int | None = None
) -> dict[str, Any]:
    denominator = (
        roster_size
        if roster_size is not None
        else len(dataset_cases(Path(__file__).absolute().parents[2] / "data", "Stability"))
    )
    if denominator <= 0 or len(rows) > denominator:
        raise ValueError("Invalid benchmark roster size")
    return {
        "Method": method,
        "Stability": {
            f"{key}↑": 100.0 * sum(row.get(key) is True for row in rows) / denominator
            for key in METRICS
        },
    }


def print_table(table: dict[str, Any]) -> None:
    width = max(len(table["Method"]), len("Method"))
    metrics = table["Stability"]
    heading = f"{'Method':<{width}} | {'Stability':^20}"
    labels = f"{'':<{width}} | " + "  ".join(f"{key:^9}" for key in metrics)
    values = f"{table['Method']:<{width}} | " + "  ".join(
        ("—" if value is None else f"{value:>8.0f}%") for value in metrics.values()
    )
    line = "─" * len(heading)
    print(f"\n{line}\n{heading}\n{labels}\n{line}\n{values}\n{line}", flush=True)


def write_tables(output: Path, table: dict[str, Any]) -> None:
    metrics = table["Stability"]
    with (output / "summary_stability.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Method", "Stability", ""])
        writer.writerow(["", *metrics])
        writer.writerow(
            [
                table["Method"],
                *[("—" if value is None else f"{value:.0f}%") for value in metrics.values()],
            ]
        )
