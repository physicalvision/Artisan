from __future__ import annotations

import argparse
import csv
import fcntl
import json
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from metrics import evaluate_mass, positive_mass
from reporting import print_table, read_metrics, summary_table, write_tables

from artisan.benchmark import dataset_cases, results_directory

PROJECT = Path(__file__).absolute().parents[3]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prediction_urdf(path: Path) -> Path:
    if path.is_file() and path.name == "object.urdf":
        return path
    for prefix in ("", "native", "exports", "workspace/exports"):
        candidate = path / prefix / "object.urdf"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No Ours object.urdf found at {path}")


def has_prediction(path: Path) -> bool:
    try:
        prediction_urdf(path)
    except FileNotFoundError:
        return False
    return True


def score_case(row: dict[str, Any]) -> dict[str, Any]:
    try:
        root = row["root"]
        if not root.is_dir() or root.is_symlink():
            raise ValueError(f"Missing object directory or unsupported symlink: {root}")
        if (root / "evaluation").is_symlink():
            raise ValueError("Evaluation output must not be a symlink")
        prediction_urdf(row["prediction"])
        with (root / ".mass.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            target = root / "evaluation/metrics_mass.json"
            gt = float(row["gt"].read_text().strip()) if isinstance(row["gt"], Path) else row["gt"]
            metrics = evaluate_mass(prediction_urdf(row["prediction"]), positive_mass(gt))
            read_metrics(metrics)
            atomic_json(target, metrics)
        result = {"id": row["id"], "metrics": metrics}
        if metrics["MnRE"] is None:
            result["error"] = metrics.get("reason") or "No valid mass prediction"
        return result
    except Exception as exc:
        return {
            "id": row["id"],
            "metrics": {"MnRE": None, "ADE": None, "APE": None},
            "error": f"{type(exc).__name__}: {exc}",
        }


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(allow_abbrev=False)
    inputs = command.add_mutually_exclusive_group()
    inputs.add_argument("--results", type=Path)
    inputs.add_argument("--prediction", type=Path)
    command.add_argument("--asset-dir", type=Path)
    command.add_argument("--only", nargs="+")
    command.add_argument("--gt-csv", type=Path)
    command.add_argument("--method")
    command.add_argument("--outputs-root", type=Path)
    command.add_argument("--data-root", type=Path, default=PROJECT / "benchmark/data")
    command.add_argument("--gt-mass", type=float)
    command.add_argument("--output", type=Path)
    command.set_defaults(dataset="ABO-Mass")
    return command


def main() -> int:
    command = parser()
    args = command.parse_args()
    if args.outputs_root is not None and (
        args.results is not None or args.prediction is not None or args.asset_dir is not None
    ):
        command.error("--outputs-root cannot accompany an explicit input path")
    gt_rows = {}
    if args.gt_csv is not None:
        with args.gt_csv.open() as stream:
            reader = csv.DictReader(stream, skipinitialspace=True)
            reader.fieldnames = [name.strip() for name in reader.fieldnames or []]
            gt_rows = {row["asin"].strip(): positive_mass(float(row["mass_kg"])) for row in reader}
    if args.asset_dir is not None and (args.prediction is not None or args.results is not None):
        command.error("--asset-dir cannot accompany --prediction or --results")
    if args.asset_dir is not None:
        if args.output is None or not gt_rows:
            command.error("--asset-dir requires --output and --gt-csv")
        source = args.asset_dir.expanduser().absolute()
        output = args.output.expanduser().absolute()
        names = args.only or [name for name in gt_rows if (source / name).is_dir()]
        if len(names) != len(set(names)) or set(names) - set(gt_rows):
            command.error("Select distinct IDs present in the GT CSV")
        rows = [
            {
                "id": name,
                "root": output / name,
                "prediction": source / name,
                "gt": gt_rows[name],
            }
            for name in names
        ]
        for row in rows:
            row["root"].mkdir(parents=True, exist_ok=True)
        method = args.method or source.name
    elif args.prediction is not None:
        if args.only is not None:
            command.error("--only requires --results or --asset-dir")
        if args.output is None or args.gt_mass is None:
            command.error("--prediction requires --output and --gt-mass")
        positive_mass(args.gt_mass)
        output = args.output.expanduser().absolute()
        source = args.prediction.expanduser().absolute()
        try:
            prediction_urdf(source)
        except FileNotFoundError as exc:
            command.error(str(exc))
        output.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "id": source.stem,
                "root": output,
                "prediction": source,
                "gt": args.gt_mass,
            }
        ]
        method = args.method or "Artisan"
    else:
        if args.gt_mass is not None:
            command.error("--gt-mass is only used with --prediction")
        root = (
            args.results.expanduser().absolute()
            if args.results is not None
            else results_directory(PROJECT, args.dataset, args.method, args.outputs_root)
        )
        if not root.is_dir() or root.is_symlink():
            command.error(
                f"Results directory does not exist or is a symlink: {root}; use --results to select existing exports"
            )
        data_root = args.data_root.expanduser().absolute()
        names = list(gt_rows) if gt_rows else dataset_cases(data_root, args.dataset)
        roster = set(names)
        if args.only:
            if len(args.only) != len(set(args.only)) or set(args.only) - set(names):
                command.error("Select distinct IDs present in the ground truth")
            names = args.only
        if not names:
            command.error("ABO-Mass contains no benchmark cases")
        unknown = {
            p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
        } - roster
        if unknown:
            command.error(
                f"Object directories have no corresponding ground truth: {', '.join(sorted(unknown))}"
            )
        rows = [
            {
                "id": name,
                "root": root / name,
                "prediction": root / name,
                "gt": gt_rows[name] if gt_rows else data_root / args.dataset / name / "mass_kg.txt",
            }
            for name in names
        ]
        output = args.output.expanduser().absolute() if args.output is not None else root
        method = args.method or root.name
    if args.prediction is None and args.only is None:
        rows = [row for row in rows if has_prediction(row["prediction"])]
    output.mkdir(parents=True, exist_ok=True)
    results = [score_case(row) for row in rows]
    failures = [row for row in results if "error" in row]
    for row in failures:
        print(f"{row['id']}: {row['error']}", file=sys.stderr)
    table = summary_table([row["metrics"] for row in results], method)
    with (output / ".mass_summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        staging = output / ".cache"
        if staging.is_symlink():
            raise ValueError("Summary cache must not be a symlink")
        staging.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="mass-summary-", dir=staging) as directory:
            temporary = Path(directory)
            atomic_json(temporary / "summary_mass.json", {"table": table})
            write_tables(temporary, table)
            for suffix in ("json", "csv"):
                name = f"summary_mass.{suffix}"
                (temporary / name).replace(output / name)
    print_table(table)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
