from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from metrics_2d import DEFAULT_METRICS, ImageScorer
from reporting import metric_kinds, print_table, summary_table, write_tables

from artisan.benchmark import results_directory
from artisan.config import apply_runtime_config, cache_dir, load_config


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        allow_abbrev=False,
    )
    p.add_argument(
        "--mode",
        choices=("2d", "3d", "material_direct", "material_transfer", "all"),
        default="all",
    )
    p.add_argument("--method")
    source = p.add_mutually_exclusive_group()
    source.add_argument(
        "--prediction",
        type=Path,
    )
    source.add_argument(
        "--manifest",
        type=Path,
    )
    source.add_argument("--results", type=Path)
    p.add_argument("--dataset", choices=("ArtVIP", "LightWheel"))
    p.add_argument("--outputs-root", type=Path)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--only", nargs="+")
    recompute = p.add_mutually_exclusive_group()
    recompute.add_argument("--refresh", action="store_true")
    recompute.add_argument(
        "--recompute-metrics",
        action="store_true",
    )
    p.add_argument(
        "--gt",
        type=Path,
    )
    p.add_argument(
        "--output",
        type=Path,
    )
    p.add_argument(
        "--blender",
        type=Path,
    )
    p.add_argument("--threads", type=int)
    p.add_argument("--device", default="auto")
    p.add_argument("--model-cache", type=Path)
    p.add_argument("--clip-checkpoint", type=Path)
    p.add_argument("--offline", action="store_true")
    p.set_defaults(
        points=100000,
        seed=0,
        resolution=1024,
        render_samples=128,
        views=["front", "iso", "top"],
        metrics=list(DEFAULT_METRICS),
    )
    return p


def json_write(path: Path, value: Any) -> None:
    def convert(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)

    with path.open("x") as stream:
        json.dump(value, stream, indent=2, default=convert, allow_nan=False)


def cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.prediction is not None:
        if args.gt is None:
            raise ValueError("--gt is required with --prediction")
        rows = [
            {"id": args.gt.name, "prediction": args.prediction.absolute(), "gt": args.gt.absolute()}
        ]
    else:
        if args.gt is not None:
            raise ValueError("--gt cannot be combined with --manifest")
        rows = json.loads(args.manifest.read_text())
        if not isinstance(rows, list) or not rows:
            raise ValueError("Manifest must be a nonempty list")
        base = args.manifest.absolute().parent
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"id", "prediction", "gt"}:
                raise ValueError("Each manifest entry requires exactly id, prediction and gt")
            for key in ("prediction", "gt"):
                row[key] = (base / row[key]).absolute()
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs")
    for sample in ids:
        if (
            not isinstance(sample, str)
            or sample in {"", ".", ".."}
            or "/" in sample
            or "\\" in sample
        ):
            raise ValueError("Sample IDs must be directory names")
    for row in rows:
        row["dataset"] = args.dataset or (
            "LightWheel" if row["gt"].parent.name == "LightWheel" else "ArtVIP"
        )
        metric_kinds(args.mode, row["dataset"])
    return rows


def main() -> int:
    command = parser()
    args = command.parse_args()
    metric_kinds(args.mode, args.dataset)
    project = Path(__file__).absolute().parents[3]
    if args.prediction is None and args.manifest is None and args.results is None:
        if args.dataset is None:
            command.error("Specify --dataset, --results, --prediction or --manifest")
        args.results = results_directory(project, args.dataset, args.method, args.outputs_root)
    elif args.outputs_root is not None:
        command.error("--outputs-root requires --dataset without an explicit input path")
    configured = os.environ.get("ARTISAN_CONFIG", "").strip()
    config_path = Path(configured).expanduser() if configured else project / ".artisan/config.toml"
    config = load_config(config_path) if configured or config_path.is_file() else {}
    apply_runtime_config(
        {"cache_dir": str(project / ".artisan/cache"), **config, "render_device": "gpu"},
        {"blender": args.blender},
    )
    args.model_cache = (args.model_cache or cache_dir() / "fidelity_models").expanduser().absolute()
    args.threads = args.threads if args.threads is not None else config.get("cpu_threads", 4)
    for key in ("points", "resolution", "render_samples", "threads"):
        if getattr(args, key) <= 0:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    if len(set(args.views)) != len(args.views):
        raise ValueError("Views must be distinct")
    if args.seed < 0 or args.resolution < 8:
        raise ValueError("Seed must be nonnegative and resolution at least 8")
    if args.results is not None:
        from results import run_results

        return run_results(args, project)
    if args.output is None:
        raise ValueError("--output is required with --prediction or --manifest")
    if args.only or args.data_root or args.refresh or args.recompute_metrics:
        raise ValueError("--only, --data-root and recompute options require --results")
    args.method = args.method or "Artisan"
    rows = cases(args)
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    results, failures = [], []
    models: dict[str, ImageScorer] = {}
    for row in rows:
        folder = output / row["id"]
        folder.mkdir()
        try:
            from results import evaluate_case

            result = evaluate_case({**row, "root": folder}, args, models)
            results.append(result)
            failures.extend(
                {"id": row["id"], "stage": kind, "error": error}
                for kind, error in result.get("failures", {}).items()
            )
        except Exception as exc:
            failure = {"id": row["id"], "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            print(f"{row['id']}: {failure['error']}", file=sys.stderr, flush=True)
    dataset = "LightWheel" if all(row["dataset"] == "LightWheel" for row in rows) else "ArtVIP"
    summary = {"table": summary_table(results, args.method, args.mode, dataset)}
    json_write(output / f"summary_{args.mode}.json", summary)
    write_tables(output, summary, f"summary_{args.mode}")
    print_table(summary)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
