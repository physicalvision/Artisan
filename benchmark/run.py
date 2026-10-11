from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).absolute().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def directory_name(value: str) -> str:
    if value in {"", ".", ".."} or value.startswith(".") or any(c in value for c in "/\\\n\r\0"):
        raise argparse.ArgumentTypeError("Expected a non-hidden directory name without separators")
    return value


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
    )
    parser.add_argument(
        "--dataset", choices=("ArtVIP", "LightWheel", "Stability", "ABO-Mass"), required=True
    )
    selected = parser.add_mutually_exclusive_group(required=True)
    selected.add_argument("--case", nargs="+", type=directory_name)
    selected.add_argument("--all", action="store_true")
    parser.add_argument("--method", type=directory_name)
    parser.add_argument("--outputs-root", type=Path, default=PROJECT / "outputs")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "benchmark/data")
    parser.add_argument("--images", choices=("close", "open", "both", "reference", "none"))
    parser.add_argument(
        "--provider", choices=("openai", "anthropic", "openrouter", "gemini", "google")
    )
    parser.add_argument("--model")
    parser.add_argument("--thinking", choices=("low", "med", "medium", "high", "xhigh", "max"))
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument(
        "--from-run",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    return parser.parse_args(argv)


def settings(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path]:
    sys.path.insert(0, str(PROJECT))
    from artisan.config import apply_runtime_config, load_config
    from pipeline.llm.model_registry import DEFAULT_TEXT_MODEL

    configured = os.environ.get("ARTISAN_CONFIG", "").strip()
    path = (
        Path(configured).expanduser().absolute()
        if configured
        else PROJECT / ".artisan/config.toml"
    )
    config = load_config(path) if configured or path.is_file() else {}
    overrides = {name: getattr(args, name) for name in ("provider", "model", "thinking")}
    apply_runtime_config(config, overrides)
    effective = {
        "provider": os.environ.get("ARTISAN_PIPELINE_PROVIDER", "openai"),
        "model": os.environ.get("ARTISAN_PIPELINE_MODEL", DEFAULT_TEXT_MODEL),
        "thinking": os.environ.get("ARTISAN_PIPELINE_THINKING", "high"),
    }
    effective["roles"] = {
        role.lower(): {
            key: os.environ.get(f"ARTISAN_PIPELINE_{key.upper()}_{role}", value)
            for key, value in effective.items()
            if key != "roles"
        }
        for role in ("DESIGN", "CODING", "REVIEW")
    }
    workspace = (
        Path(
            os.environ.get("ARTISAN_WORKSPACE")
            or config.get("workspace")
            or path.parent / "workspace"
        )
        .expanduser()
        .absolute()
    )
    return effective, workspace, path


def inputs(folder: Path, image_mode: str) -> dict[str, Any]:
    prompt = folder if folder.suffix == ".txt" else folder / "prompt.txt"
    if not prompt.is_file() or not prompt.read_text().strip():
        raise ValueError(f"Missing or empty prompt: {prompt}")
    filenames = {
        "both": ("close.png", "open.png"),
        "close": ("close.png",),
        "open": ("open.png",),
        "reference": ("image.png",),
        "none": (),
    }
    images = [folder / name for name in filenames[image_mode]]
    for path in images:
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "prompt": str(prompt),
        "images": [str(p) for p in images],
    }


def case_input(args: argparse.Namespace, case: str) -> Path:
    name = f"{case}.txt" if args.dataset == "Stability" else case
    return args.data_root / args.dataset / name


def locate_run(value: str, workspace: Path) -> Path:
    if re.fullmatch(r"pl_[A-Za-z0-9_-]{1,100}", value):
        return workspace / "data/cache/runs" / value
    path = Path(value).expanduser().absolute()
    if not (path / "pipeline_state.json").is_file():
        raise ValueError("--from-run must identify a run directory containing pipeline_state.json")
    return path


def has_delivery(state: dict[str, Any]) -> bool:
    if state.get("phase") == "done" and state.get("status") == "done":
        return True
    return (
        state.get("status") in {"incomplete", "failed"}
        and state.get("salvage", {}).get("delivered") is True
    )


def delivered_exports(run: Path, workspace: Path) -> tuple[dict[str, Any], Path, str]:
    from artisan.cli import _canonical_record_id

    state = read_json(run / "pipeline_state.json")
    if not has_delivery(state):
        raise ValueError(
            f"Run {state.get('run_id')} has no successful delivery: {state.get('phase')} / {state.get('status')}"
        )
    if Path(state["repo_root"]).resolve() != workspace.resolve():
        raise ValueError("Run belongs to a different configured workspace")
    record_id = _canonical_record_id(workspace, state["run_id"])
    if record_id is None:
        raise ValueError("Run has no canonical registered record")
    normal = state.get("phase") == "done" and state.get("status") == "done"
    if not normal and state["salvage"].get("record_id") != record_id:
        raise ValueError("Salvaged delivery does not match the canonical registered record")
    exports = run / "workspace" / ("exports" if normal else "salvaged_exports")
    for name in ("object.urdf", "object.articulated.glb"):
        if not (exports / name).is_file():
            raise FileNotFoundError(exports / name)
    return state, exports, record_id


def collect(source: Path, destination: Path) -> None:
    if destination.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(
        destination.resolve()
    ):
        raise ValueError("Native destination must be separate from the source exports")
    if destination.exists():
        raise ValueError(
            "Incomplete native directory exists; choose another output directory or remove it before retrying"
        )
    stage = destination.with_name(f".native.{uuid.uuid4().hex}")
    try:
        shutil.copytree(source, stage)
        stage.rename(destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def process_case(
    args: argparse.Namespace,
    case: str,
    method_dir: Path,
    workspace: Path,
    config_path: Path,
) -> None:
    folder = method_dir / case
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".case.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if all(
            (folder / "native" / name).is_file()
            for name in ("object.urdf", "object.articulated.glb")
        ):
            print(f"{args.dataset}/{case}: using existing native exports", flush=True)
            return
        source_inputs = inputs(case_input(args, case), args.images)
        binding_path = folder / ".run.json"
        binding = read_json(binding_path) if binding_path.is_file() else None
        if binding is not None:
            run = Path(binding["run_directory"])
            if args.from_run and locate_run(args.from_run, workspace) != run:
                raise ValueError(
                    "Case is already bound to another run; choose a different method directory"
                )
        else:
            if args.resume:
                raise ValueError("No saved run for this case; omit --resume to start generation")
            run = (
                locate_run(args.from_run, workspace)
                if args.from_run
                else workspace / "data/cache/runs" / f"pl_bench_{uuid.uuid4().hex}"
            )
            binding = {"run_id": run.name, "run_directory": str(run), "status": "pending"}
            if args.from_run:
                delivered_exports(run, workspace)
            write_json(binding_path, binding)
        checkpoint = run / "pipeline_state.json"
        state = read_json(checkpoint) if checkpoint.is_file() else None
        if not args.from_run and (
            state is None
            or not has_delivery(state)
            or (args.resume and state.get("phase") != "done")
        ):
            if state is None and args.resume:
                raise ValueError("The saved run has no checkpoint to resume")
            if state is not None and not args.resume:
                raise ValueError(
                    f"Run {binding['run_id']} is incomplete; use --resume to continue it"
                )
            if state is None and run.exists():
                raise ValueError(
                    "Run directory exists without a checkpoint; inspect it before retrying"
                )
            command = [str(PROJECT / "run.sh")]
            if state is None:
                command += [
                    "run",
                    "--run-id",
                    binding["run_id"],
                    "--prompt-file",
                    source_inputs["prompt"],
                ]
                for image in source_inputs["images"]:
                    command += ["--image", image]
            else:
                command += ["resume", binding["run_id"]]
            command += ["--workspace", str(workspace)]
            if config_path.is_file():
                command += ["--config", str(config_path)]
            for name in ("provider", "model", "thinking", "max_cost_usd"):
                value = getattr(args, name)
                if value is not None:
                    command += ["--" + name.replace("_", "-"), str(value)]
            binding["status"] = "running"
            write_json(binding_path, binding)
            print(
                f"{args.dataset}/{case}: running {binding['run_id']} (log: {folder / '.run.log'})",
                flush=True,
            )
            with (folder / ".run.log").open("ab") as log:
                process = subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
            binding["last_exit_code"] = process.returncode
            if process.returncode != 0:
                binding["status"] = "incomplete"
                write_json(binding_path, binding)
                if process.returncode in (130, -2):
                    print(f"Inference interrupted; see {folder / '.run.log'}", file=sys.stderr)
                    raise KeyboardInterrupt
                state = read_json(checkpoint) if checkpoint.is_file() else {}
                if process.returncode != 1 or not has_delivery(state):
                    raise RuntimeError(
                        f"Inference stopped with exit code {process.returncode}; see {folder / '.run.log'}"
                    )
        state, exports, record_id = delivered_exports(run, workspace)
        binding["record_id"] = record_id
        binding["delivery"] = "normal" if exports.name == "exports" else "degraded"
        collect(exports, folder / "native")
        binding["status"] = "collected"
        write_json(binding_path, binding)
        print(f"{args.dataset}/{case}: native exports collected", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.data_root = args.data_root.expanduser().absolute()
    args.outputs_root = args.outputs_root.expanduser().absolute()
    if args.outputs_root.resolve().is_relative_to(args.data_root.resolve()):
        raise ValueError("Outputs must be outside the benchmark data directory")
    if args.max_cost_usd is not None and (
        not math.isfinite(args.max_cost_usd) or args.max_cost_usd <= 0
    ):
        raise ValueError("--max-cost-usd must be finite and positive")
    effective, workspace, config = settings(args)
    from artisan.benchmark import dataset_cases, default_method, results_directory

    method = args.method or default_method(effective["roles"]["coding"]["model"])
    dataset_root = args.data_root / args.dataset
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    cases = dataset_cases(args.data_root, args.dataset) if args.all else args.case
    args.images = args.images or {"Stability": "none", "ABO-Mass": "reference"}.get(
        args.dataset, "open"
    )
    if args.dataset == "Stability" and args.images != "none":
        raise ValueError("Stability inputs are text-only; --images must be none")
    if args.dataset == "ABO-Mass" and args.images not in {"reference", "none"}:
        raise ValueError("ABO-Mass has one reference image; use --images reference or none")
    if not cases or len(cases) != len(set(cases)):
        raise ValueError("Select a nonempty set of distinct benchmark cases")
    if args.from_run and len(cases) != 1:
        raise ValueError("--from-run requires exactly one selected case")
    if args.from_run and args.resume:
        raise ValueError("--from-run cannot be combined with --resume")
    for case in cases:
        inputs(case_input(args, case), args.images)
    method_dir = results_directory(PROJECT, args.dataset, method, args.outputs_root)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "cases": cases,
                    "method": method,
                    "outputs": str(method_dir),
                    "images": args.images,
                    "settings": effective,
                    "from_run": args.from_run,
                },
                indent=2,
            )
        )
        return 0
    failures = []
    for case in cases:
        try:
            process_case(args, case, method_dir, workspace, config)
        except Exception as exc:
            failures.append({"case": case, "error": f"{type(exc).__name__}: {exc}"})
            print(f"{case}: {exc}", file=sys.stderr, flush=True)
    if failures:
        print("Generation or collection failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
