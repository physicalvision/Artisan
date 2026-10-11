from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from artifacts import load_points, read_metrics, saved_matching
from camera import pose_normalization, read_camera
from geometry import evaluate_3d, sample_pair
from metrics_2d import ImageScorer
from protocol_config import blender_bin
from reporting import metric_kinds, print_table, summary_table, write_tables

from assets import prediction_exports, transform_glb


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, default=json_default, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextlib.contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"Lock must not be a symlink: {path}")
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another evaluation is using {path.parent}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def create_poses(
    row: dict[str, Any], args: argparse.Namespace, work: Path, paths: dict[str, Path]
) -> dict[str, Path]:
    from protocol import create_protocol_poses

    return create_protocol_poses(row, args, work, paths)


def publish_files(root: Path, products: dict[str, Path]) -> None:
    for relative, source in products.items():
        destination = root / relative
        directory = source.is_dir()
        for path in sorted(source.rglob("*")) if directory else [source]:
            if path.is_file() and not path.name.startswith(".nfs"):
                target = destination / path.relative_to(source) if directory else destination
                if target.parent.is_symlink() or target.is_symlink():
                    raise ValueError(f"Managed output must not be a symlink: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                path.replace(target)


def ensure_gt_poses(root: Path, gt_path: Path, args: argparse.Namespace) -> None:
    directory = root / "eval_poses"
    missing = [
        state for state in ("close", "open") if not (directory / f"gt_{state}.glb").is_file()
    ]
    if not missing:
        return
    normalization = pose_normalization(root)
    if normalization is None:
        gt = sample_pair(gt_path / "close.glb", gt_path / "open.glb", args.points, args.seed)
        normalization = {
            "center": (gt.close.min(axis=0) + gt.close.max(axis=0)) / 2,
            "longest_side": float(np.ptp(gt.close, axis=0).max()),
        }
    scale = (
        1.0 / float(normalization["longest_side"])
        if "longest_side" in normalization
        else float(normalization["scale"])
    )
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Degenerate GT closed bounds")
    transform = np.eye(4)
    transform[:3, :3] *= scale
    transform[:3, 3] = -np.asarray(normalization["center"], dtype=float) * scale
    with tempfile.TemporaryDirectory(prefix="gt-", dir=root / ".cache") as temporary:
        products = {}
        for state in missing:
            target = Path(temporary) / f"gt_{state}.glb"
            transform_glb(gt_path / f"{state}.glb", target, transform)
            products[f"eval_poses/gt_{state}.glb"] = target
        publish_files(root, products)


def existing_renders(root: Path) -> dict[str, list[Path]] | None:
    directory = root / "evaluation/renders"
    summary = directory / "render_results.json"
    if not summary.is_file():
        output = {}
        for state in ("close", "open"):
            folder = directory / state
            views = (
                sorted(path for path in folder.iterdir() if path.is_dir())
                if folder.is_dir()
                else []
            )
            if not views:
                return None
            for role in ("pred", "gt"):
                paths = [view / f"{role}.png" for view in views]
                if not all(path.is_file() for path in paths):
                    raise ValueError(f"Incomplete cached render pairs: {folder}")
                output[f"{role}_{state}"] = paths
        return output
    images = json.loads(summary.read_text())["images"]
    output = {}
    for name in ("pred_close", "pred_open", "gt_close", "gt_open"):
        paths = [directory / path for path in images[name]]
        if not paths or not all(path.is_file() for path in paths):
            return None
        output[name] = paths
    return output


def render_inputs(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    root = row["root"]
    models = {
        f"{role}_{state}": str(
            root / "eval_poses" / (f"gt_{state}.glb" if role == "gt" else f"{state}.glb")
        )
        for role in ("gt", "pred")
        for state in ("close", "open")
    }
    return {
        "models": models,
        "raw_gt_models": {
            f"gt_{state}": str(row["gt"] / f"{state}.glb") for state in ("close", "open")
        },
        "resolution": args.resolution,
        "samples": args.render_samples,
        "threads": args.threads,
        "views": args.views,
        "seed": args.seed,
        "camera": read_camera(row["gt"], pose_normalization(root)),
    }


def render_case(row: dict[str, Any], args: argparse.Namespace) -> dict[str, list[Path]]:
    root = row["root"]
    with tempfile.TemporaryDirectory(prefix="render-", dir=root / ".cache") as temporary:
        work = Path(temporary)
        config = {
            **render_inputs(row, args),
            "output": str(work / "renders"),
        }
        atomic_json(work / "render_config.json", config)
        with (work / "render.log").open("x") as log:
            rendered_process = subprocess.run(
                [
                    blender_bin(),
                    "--background",
                    "--factory-startup",
                    "--python-exit-code",
                    "1",
                    "--python",
                    str(Path(__file__).with_name("render_blender.py")),
                    "--",
                    "--manifest",
                    str(work / "render_config.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if rendered_process.returncode:
            detail = next(
                (
                    line.removeprefix("RuntimeError: ")
                    for line in (work / "render.log").read_text().splitlines()
                    if line.startswith("RuntimeError: Fidelity ")
                ),
                f"Blender failed ({rendered_process.returncode})",
            )
            publish_files(root, {"evaluation/render_error.log": work / "render.log"})
            raise RuntimeError(f"{detail}; see evaluation/render_error.log")
        rendered = json.loads((work / "renders/render_results.json").read_text())
        for kind in ("images", "masks"):
            if kind in rendered:
                rendered[kind] = {
                    name: [Path(path).name for path in paths]
                    for name, paths in rendered[kind].items()
                }
        atomic_json(work / "renders/render_results.json", rendered)
        config["output"] = str(root / "evaluation/renders")
        atomic_json(work / "render_config.json", config)
        publish_files(
            root,
            {
                "evaluation/renders": work / "renders",
                "evaluation/render.log": work / "render.log",
                "evaluation/render_config.json": work / "render_config.json",
            },
        )
    output = existing_renders(root)
    if output is None:
        raise ValueError("Incomplete rendered views")
    return output


def evaluate_stage(
    row: dict[str, Any], args: argparse.Namespace, models: dict[str, ImageScorer]
) -> dict[str, Any]:
    kinds = metric_kinds(args.mode, args.dataset)
    root = row["root"]
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Missing object directory or unsupported symlink: {root}")
    for folder in ("eval_poses", "evaluation", ".cache"):
        if (root / folder).is_symlink():
            raise ValueError(f"Managed directory must not be a symlink: {root / folder}")
    with exclusive_lock(root / ".evaluation.lock"):
        result: dict[str, Any] = {}
        missing = []
        for kind in kinds:
            path = root / "evaluation" / f"metrics_{kind}.json"
            if path.is_file() and not args.refresh and not args.recompute_metrics:
                result[kind] = read_metrics(path, kind)
            else:
                missing.append(kind)
        if not missing:
            return result
        prediction_exports(row["prediction"])
        cache_root = root / ".cache"
        cache_root.mkdir(exist_ok=True)
        points = root / "evaluation/points_3d.npz"
        reuse_matching = points.is_file() and not args.refresh
        images = existing_renders(root) if "2d" in missing and not args.refresh else None
        from material import buffers_exist, saved_transfer_parts

        transfer_needs_parts = "material_transfer" in missing and (
            args.refresh
            or not buffers_exist(root / "evaluation/material_transfer", args.views, "transfer")
        )
        need_points = (
            "3d" in missing
            or (transfer_needs_parts and (args.refresh or saved_transfer_parts(root) is None))
        ) and (args.refresh or not points.is_file())
        render_materials = [
            kind
            for kind in missing
            if kind.startswith("material_")
            and (
                args.refresh
                or not buffers_exist(
                    root / "evaluation" / kind, args.views, kind.removeprefix("material_")
                )
            )
        ]
        need_poses = need_points or ("2d" in missing and images is None) or bool(render_materials)
        if need_poses:
            poses_exist = all(
                (root / "eval_poses" / f"{state}.glb").is_file() for state in ("close", "open")
            )
            if not poses_exist or args.refresh or need_points:
                urdf, glb = prediction_exports(row["prediction"])
                paths = {
                    "urdf": urdf,
                    "articulated_glb": glb,
                    "gt_close": row["gt"] / "close.glb",
                    "gt_open": row["gt"] / "open.glb",
                }
                with tempfile.TemporaryDirectory(prefix="pose-", dir=cache_root) as temporary:
                    publish_files(root, create_poses(row, args, Path(temporary), paths))
            else:
                ensure_gt_poses(root, row["gt"], args)
        if "3d" in missing:
            pred, gt = load_points(points)
            mapping = saved_matching(root / "evaluation", pred, gt) if reuse_matching else None
            result["3d"] = evaluate_3d(pred, gt, mapping)
            atomic_json(root / "evaluation/matching.json", result["3d"]["matching"])
            result["3d"] = {state: result["3d"][state] for state in ("close", "open")}
            atomic_json(root / "evaluation/metrics_3d.json", result["3d"])
        if "2d" in missing:
            images = images if images is not None else render_case(row, args)
            if "scorer" not in models:
                models["scorer"] = ImageScorer(
                    args.metrics,
                    args.device,
                    args.model_cache,
                    clip_checkpoint=args.clip_checkpoint,
                    offline=args.offline,
                )
            result["2d"] = {
                state: models["scorer"].score_pairs(
                    list(zip(images[f"pred_{state}"], images[f"gt_{state}"], strict=True))
                )
                for state in ("close", "open")
            }
            atomic_json(root / "evaluation/metrics_2d.json", result["2d"])
        for kind in missing:
            if kind.startswith("material_"):
                from material import evaluate_material

                result[kind] = evaluate_material(row, args, kind.removeprefix("material_"))
                atomic_json(root / "evaluation" / f"metrics_{kind}.json", result[kind])
        return result


def evaluate_case(
    row: dict[str, Any], args: argparse.Namespace, models: dict[str, ImageScorer]
) -> dict[str, Any]:
    dataset = args.dataset or ("LightWheel" if row["gt"].parent.name == "LightWheel" else "ArtVIP")
    kinds = metric_kinds(args.mode, dataset)
    result: dict[str, Any] = {}
    failures = {}
    for kind in ("3d", "2d", "material_direct", "material_transfer"):
        if kind not in kinds:
            continue
        stage_args = argparse.Namespace(**(vars(args) | {"mode": kind, "dataset": dataset}))
        try:
            result.update(evaluate_stage(row, stage_args, models))
        except Exception as exc:
            failures[kind] = f"{type(exc).__name__}: {exc}"
            print(f"{row['id']}/{kind}: {failures[kind]}", file=sys.stderr, flush=True)
    if failures:
        result["failures"] = failures
    return result


def run_results(args: argparse.Namespace, project: Path) -> int:
    root = args.results.expanduser().absolute()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("--results must be an existing method directory")
    if args.output is not None and args.output.absolute() != root:
        raise ValueError("With --results, --output must be omitted or equal the method directory")
    if args.gt is not None:
        raise ValueError("--gt cannot be combined with --results")
    dataset = args.dataset or root.parent.name
    if dataset not in {"ArtVIP", "LightWheel"}:
        raise ValueError("Fidelity supports the ArtVIP and LightWheel datasets")
    args.dataset = dataset
    metric_kinds(args.mode, dataset)
    data = (args.data_root or project / "benchmark/data").expanduser().absolute() / dataset
    names = (
        args.only
        if args.only is not None
        else sorted(
            path.name for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
        )
    )
    if (
        not names
        or len(set(names)) != len(names)
        or any(not name or name in {".", ".."} or "/" in name or "\\" in name for name in names)
    ):
        raise ValueError("Select distinct object directory names")
    if args.only is None:
        names = [name for name in names if (root / name / "native").is_dir()]
    models: dict[str, ImageScorer] = {}
    results, failures = [], []
    for name in names:
        row = {
            "id": name,
            "root": root / name,
            "prediction": root / name / "native",
            "gt": data / name,
        }
        try:
            result = evaluate_case(row, args, models)
            results.append(result)
            failures.extend(
                {"id": name, "stage": kind, "error": error}
                for kind, error in result.get("failures", {}).items()
            )
        except Exception as exc:
            failures.append({"id": name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"{name}: {failures[-1]['error']}", file=sys.stderr, flush=True)
    summary = {"table": summary_table(results, args.method or root.name, args.mode, dataset)}
    with exclusive_lock(root / ".summary.lock"):
        staging = root / ".cache"
        if staging.is_symlink():
            raise ValueError("Summary staging directory must not be a symlink")
        staging.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="summary-", dir=staging) as temporary:
            output = Path(temporary)
            stem = f"summary_{args.mode}"
            atomic_json(output / f"{stem}.json", summary)
            write_tables(output, summary, stem)
            for suffix in ("json", "csv"):
                (output / f"{stem}.{suffix}").replace(root / f"{stem}.{suffix}")
    print_table(summary)
    return 1 if failures else 0
