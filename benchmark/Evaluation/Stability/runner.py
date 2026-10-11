from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import math
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

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


def score_case(row: dict[str, Any], refresh: bool, timeout: float) -> dict[str, Any]:
    try:
        root = row["root"]
        if not root.is_dir() or root.is_symlink():
            raise ValueError(f"Missing object directory or unsupported symlink: {root}")
        output = root / "evaluation"
        staging = root / ".cache"
        if output.is_symlink() or staging.is_symlink():
            raise ValueError("Evaluation output and cache must not be symlinks")
        prediction_urdf(row["prediction"])
        with (root / ".stability.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            target = output / "metrics_stability.json"
            if (
                target.is_file()
                and not refresh
                and set(json.loads(target.read_text())) == {"Rest", "Open"}
            ):
                metrics = read_metrics(json.loads(target.read_text()))
            else:
                source = prediction_urdf(row["prediction"])
                staging.mkdir(exist_ok=True)
                with tempfile.TemporaryDirectory(prefix="stability-", dir=staging) as directory:
                    report = Path(directory) / "result.json"
                    process = subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).with_name("worker.py")),
                            str(source),
                            str(report),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        cwd=directory,
                        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                    )
                    if process.returncode in (130, -2):
                        raise KeyboardInterrupt
                    if process.returncode:
                        raise RuntimeError(
                            process.stderr.strip()
                            or f"Simulation exited with code {process.returncode}"
                        )
                    metrics = read_metrics(json.loads(report.read_text()))
                atomic_json(target, metrics)
        return {"id": row["id"], "metrics": metrics}
    except Exception as exc:
        return {
            "id": row["id"],
            "metrics": {"Rest": None, "Open": None},
            "error": f"{type(exc).__name__}: {exc}",
        }


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(allow_abbrev=False)
    inputs = command.add_mutually_exclusive_group()
    inputs.add_argument("--results", type=Path)
    inputs.add_argument("--prediction", type=Path)
    command.add_argument("--method")
    command.add_argument("--outputs-root", type=Path)
    command.add_argument("--data-root", type=Path, default=PROJECT / "benchmark/data")
    command.add_argument("--only", nargs="+")
    command.add_argument("--output", type=Path)
    command.add_argument("--refresh", action="store_true")
    command.add_argument(
        "--asset-dir",
        type=Path,
    )
    command.add_argument("--workers", type=int, default=1)
    command.add_argument("--timeout", type=float, default=600)
    command.set_defaults(dataset="Stability")
    return command


def main() -> int:
    command = parser()
    args = command.parse_args()
    roster_size = len(dataset_cases(args.data_root.expanduser().absolute(), "Stability"))
    if args.workers < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        command.error("--workers and --timeout must be positive and finite")
    if args.outputs_root is not None and (
        args.results is not None or args.prediction is not None or args.asset_dir is not None
    ):
        command.error("--outputs-root cannot accompany an explicit input path")
    if args.asset_dir is not None and (args.prediction is not None or args.results is not None):
        command.error("--asset-dir cannot accompany --prediction or --results")
    if args.asset_dir is not None:
        if args.output is None:
            command.error("--asset-dir requires --output to preserve input assets")
        root = args.asset_dir.expanduser().absolute()
        roster = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
        names = args.only if args.only is not None else roster
        if not names or len(set(names)) != len(names) or set(names) - set(roster):
            command.error("Select distinct object IDs present in --asset-dir")
        output = args.output.expanduser().absolute()
        rows = [{"id": name, "root": output / name, "prediction": root / name} for name in names]
        for row in rows:
            row["root"].mkdir(parents=True, exist_ok=True)
        method = args.method or root.name
    elif args.prediction is not None:
        if args.output is None or args.only is not None:
            command.error("--prediction requires --output and cannot use --only")
        output = args.output.expanduser().absolute()
        output.mkdir(parents=True, exist_ok=True)
        source = args.prediction.expanduser().absolute()
        rows = [{"id": source.stem, "root": output, "prediction": source}]
        method = args.method or "Artisan"
    else:
        root = (
            args.results.expanduser().absolute()
            if args.results is not None
            else results_directory(PROJECT, args.dataset, args.method, args.outputs_root)
        )
        if not root.is_dir() or root.is_symlink():
            command.error("Expected an existing method directory")
        roster = dataset_cases(args.data_root.expanduser().absolute(), args.dataset)
        names = args.only if args.only is not None else roster
        if not names or len(names) != len(set(names)) or set(names) - set(roster):
            command.error("Select distinct prompt IDs from the Stability dataset")
        unknown = {
            p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
        } - set(roster)
        if unknown:
            command.error(
                f"Object directories have no corresponding prompt: {', '.join(sorted(unknown))}"
            )
        rows = [
            {"id": name, "root": root / name, "prediction": root / name / "native"}
            for name in names
        ]
        output = args.output.expanduser().absolute() if args.output is not None else root
        method = args.method or root.name
    if args.prediction is None and args.only is None:
        rows = [row for row in rows if has_prediction(row["prediction"])]
    output.mkdir(parents=True, exist_ok=True)
    if args.workers == 1:
        results = [score_case(row, args.refresh, args.timeout) for row in rows]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(
                pool.map(
                    score_case,
                    rows,
                    [args.refresh] * len(rows),
                    [args.timeout] * len(rows),
                )
            )
    failures = [row for row in results if "error" in row]
    for row in failures:
        print(f"{row['id']}: {row['error']}", file=sys.stderr)
    table = summary_table([row["metrics"] for row in results], method, roster_size)
    with (output / ".stability_summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        staging = output / ".cache"
        if staging.is_symlink():
            raise ValueError("Summary cache must not be a symlink")
        staging.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="stability-summary-", dir=staging) as directory:
            temporary = Path(directory)
            atomic_json(temporary / "summary_stability.json", {"table": table})
            write_tables(temporary, table)
            for suffix in ("json", "csv"):
                name = f"summary_stability.{suffix}"
                (temporary / name).replace(output / name)
    print_table(table)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
