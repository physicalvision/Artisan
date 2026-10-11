from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean
from typing import Any

COLUMNS = (("MnRE", "MnRE↑", 3), ("ADE", "ADE (kg)↓", 2), ("APE", "APE↓", 3))


def read_metrics(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or any(key not in value for key, _, _ in COLUMNS):
        raise ValueError("Mass results must contain MnRE, ADE and APE")
    scores = [value[key] for key, _, _ in COLUMNS]
    if all(score is None for score in scores):
        return value
    if any(
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or score < 0
        for score in scores
    ):
        raise ValueError("Mass metrics must be finite nonnegative numbers, or all null")
    if value["MnRE"] > 1:
        raise ValueError("MnRE must be between zero and one")
    return value


def summary_table(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    valid = [row for row in rows if all(row[key] is not None for key, _, _ in COLUMNS)]
    return {
        "Method": method,
        "Mass": {
            label: mean(row[key] for row in valid) if valid else None for key, label, _ in COLUMNS
        },
    }


def formatted(table: dict[str, Any]) -> list[str]:
    return [
        "—" if table["Mass"][label] is None else f"{table['Mass'][label]:.{digits}f}"
        for _, label, digits in COLUMNS
    ]


def print_table(table: dict[str, Any]) -> None:
    width = max(len(table["Method"]), len("Method"))
    heading = f"{'Method':<{width}} | {'Mass':^31}"
    labels = f"{'':<{width}} | " + "  ".join(f"{label:^9}" for _, label, _ in COLUMNS)
    values = f"{table['Method']:<{width}} | " + "  ".join(
        f"{value:>9}" for value in formatted(table)
    )
    line = "─" * len(heading)
    print(f"\n{line}\n{heading}\n{labels}\n{line}\n{values}\n{line}", flush=True)


def write_tables(output: Path, table: dict[str, Any]) -> None:
    with (output / "summary_mass.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Method", "Mass", "", ""])
        writer.writerow(["", *[label for _, label, _ in COLUMNS]])
        writer.writerow([table["Method"], *formatted(table)])
