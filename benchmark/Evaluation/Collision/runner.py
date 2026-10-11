from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from reporting import METRICS, print_table, read_metrics, summary_table, write_tables

PROJECT = Path(__file__).absolute().parents[3]
LEVELS = ("L1", "L2", "Lmulti")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def gt_urdf(path: Path) -> Path:
    if path.is_file() and path.suffix.lower() == ".urdf":
        return path
    if path.is_dir():
        found = (
            list((path / "urdf").glob("*.urdf"))
            if (path / "urdf").is_dir()
            else list(path.glob("*.urdf"))
        )
        if len(found) == 1:
            return found[0]
    raise ValueError(f"Expected one GT URDF with mesh resources: {path}")


def has_prediction(path: Path) -> bool:
    from collision.model import resolve_prediction

    try:
        resolve_prediction(path)
    except FileNotFoundError:
        return False
    return True


def score_case(row: dict[str, Any], refresh: bool) -> dict[str, Any]:
    try:
        root = row["root"]
        if not root.is_dir() or root.is_symlink():
            raise ValueError(f"Missing object directory or unsupported symlink: {root}")
        output = root / "evaluation"
        if output.is_symlink():
            raise ValueError("Evaluation output must not be a symlink")
        with (root / ".collision.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            target = output / "metrics_collision.json"
            if target.is_file() and not refresh:
                metrics = read_metrics(json.loads(target.read_text()))
            else:
                from collision.conditioned import evaluate_pair
                from collision.model import resolve_prediction

                report = evaluate_pair(
                    gt_urdf(row["gt"]),
                    resolve_prediction(row["prediction"]),
                    levels=LEVELS,
                    gt_cache_dir=row.get("gt_cache"),
                    policy_name=row.get("policy_name", "rev10"),
                    refresh=refresh,
                )
                metrics = {}
                for name, level in zip(METRICS, LEVELS, strict=True):
                    counts = report["levels"][level]
                    denominator = counts["denominator"]
                    metrics[name] = (
                        100.0 * counts["violating"] / denominator if denominator else None
                    )
                read_metrics(metrics)
                atomic_json(target, metrics)
        return {"id": row["id"], "metrics": metrics}
    except Exception as exc:
        return {"id": row["id"], "error": f"{type(exc).__name__}: {exc}"}


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        allow_abbrev=False,
    )
    inputs = command.add_mutually_exclusive_group()
    inputs.add_argument("--results", type=Path)
    inputs.add_argument(
        "--prediction",
        type=Path,
    )
    command.add_argument("--gt", type=Path)
    command.add_argument(
        "--output",
        type=Path,
    )
    command.add_argument("--dataset", choices=("ArtVIP", "LightWheel"))
    command.add_argument("--outputs-root", type=Path)
    command.add_argument("--data-root", type=Path, default=PROJECT / "benchmark/data")
    command.add_argument("--only", nargs="+")
    command.add_argument("--method")
    command.add_argument(
        "--refresh",
        action="store_true",
    )
    command.add_argument(
        "--gt-cache",
        type=Path,
    )
    command.add_argument(
        "--policy-name",
        default="rev10",
    )
    command.add_argument("--workers", type=int, default=1)
    return command


def main() -> int:
    command = parser()
    args = command.parse_args()
    if args.prediction is None and args.results is None:
        if args.dataset is None:
            command.error("Specify --dataset, --results or --prediction")
        from artisan.benchmark import results_directory

        args.results = results_directory(PROJECT, args.dataset, args.method, args.outputs_root)
    elif args.outputs_root is not None:
        command.error("--outputs-root requires --dataset without an explicit input path")
    if args.workers < 1:
        command.error("--workers must be positive")
    rows = []
    if args.results is not None:
        root = args.results.expanduser().absolute()
        if not root.is_dir() or root.is_symlink():
            command.error("--results must be an existing method directory")
        if args.gt is not None:
            command.error("--gt is only used with --prediction")
        dataset = args.dataset or root.parent.name
        if dataset not in {"ArtVIP", "LightWheel"}:
            command.error("Specify --dataset ArtVIP or LightWheel")
        names = (
            args.only
            if args.only is not None
            else sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
        )
        if (
            not names
            or len(set(names)) != len(names)
            or any(
                not name or name.startswith(".") or "/" in name or "\\" in name for name in names
            )
        ):
            command.error("Select distinct object directory names")
        data = args.data_root.expanduser().absolute() / dataset
        rows = [
            {
                "id": name,
                "root": root / name,
                "prediction": root / name / "native"
                if (root / name / "native").is_dir()
                else root / name,
                "gt": data / name,
            }
            for name in names
        ]
        output = args.output.expanduser().absolute() if args.output is not None else root
        method = args.method or root.name
    else:
        if args.gt is None or args.output is None:
            command.error("--prediction requires --gt and --output")
        if args.only is not None:
            command.error("--only requires --results")
        output = args.output.expanduser().absolute()
        gt = args.gt.expanduser().absolute()
        name = gt.stem if gt.is_file() else gt.name
        case = output / name
        case.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "id": name,
                "root": case,
                "prediction": args.prediction.expanduser().absolute(),
                "gt": gt,
            }
        ]
        dataset = args.dataset
        method = args.method or "Artisan"
    if args.prediction is None and args.only is None:
        rows = [row for row in rows if has_prediction(row["prediction"])]
    output.mkdir(parents=True, exist_ok=True)
    for row in rows:
        row["gt_cache"] = args.gt_cache or output / ".cache/collision-gt"
        row["policy_name"] = args.policy_name
    if args.workers == 1:
        results = [score_case(row, args.refresh) for row in rows]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(score_case, rows, [args.refresh] * len(rows)))
    failures = [row for row in results if "error" in row]
    for failure in failures:
        print(f"{failure['id']}: {failure['error']}", file=sys.stderr)
    table = summary_table([row["metrics"] for row in results if "metrics" in row], method)
    with (output / ".collision_summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        staging = output / ".cache"
        staging.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="collision-summary-", dir=staging) as temporary:
            target = Path(temporary)
            atomic_json(target / "summary_collision.json", {"table": table})
            write_tables(target, table)
            for name in ("summary_collision.json", "summary_collision.csv"):
                (target / name).replace(output / name)
    print_table(table, dataset)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
