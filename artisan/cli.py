from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from artisan.config import (
    THREAD_ENVIRONMENT_VARIABLES,
    apply_runtime_config,
    default_config_path,
    default_cost_limit,
    initialize_config,
    load_config,
    load_credentials,
)

_REQUIRED_MODULES = {
    "cadquery": "cadquery",
    "numpy": "numpy",
    "scipy": "scipy",
    "trimesh": "trimesh",
    "fcl": "python-fcl",
    "mujoco": "mujoco",
    "coacd": "coacd",
    "PIL": "Pillow",
    "openai": "openai",
    "bm25s": "bm25s",
}


def _run_id(value: str) -> str:
    if not re.fullmatch(r"pl_[A-Za-z0-9_-]{1,100}", value):
        raise argparse.ArgumentTypeError(
            "Run IDs must start with pl_ and contain letters, numbers, '_' or '-'"
        )
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="artisan", description="Generate articulated assets with Artisan."
    )
    parser.add_argument("--version", action="version", version="Artisan 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "resume", "inspect", "viewer", "doctor", "init"):
        sub = commands.add_parser(command)
        sub.add_argument("--workspace", type=Path, default=None)
        sub.add_argument("--config", type=Path, default=None)
        if command in {"run", "resume", "doctor", "init"}:
            sub.add_argument("--credentials", type=Path, default=None)
            sub.add_argument("--blender", default=None)
        if command in {"run", "resume", "doctor"}:
            sub.add_argument(
                "--provider",
                choices=("openai", "anthropic", "openrouter", "gemini", "google"),
                default=None,
            )
            sub.add_argument("--model", default=None)
            sub.add_argument(
                "--thinking", choices=("low", "med", "medium", "high", "xhigh", "max"), default=None
            )
        if command in {"run", "resume"}:
            sub.add_argument("--max-cost-usd", type=float, default=None)
        if command == "run":
            sub.add_argument("prompt", nargs="?")
            sub.add_argument("--prompt-file", type=Path)
            sub.add_argument("--image", action="append", type=Path, default=[])
            sub.add_argument("--run-id", type=_run_id)
        if command in {"resume", "inspect"}:
            sub.add_argument("run_id", type=_run_id)
        if command == "viewer":
            sub.add_argument("--host", default="127.0.0.1")
            sub.add_argument("--port", type=int, default=8765)
    return parser


def _preflight() -> dict[str, Any]:
    from artisan.physics_policy import penetration_depth_tolerance_from_env
    from pipeline.blender.runtime import blender_bin
    from pipeline.llm.model_registry import validate_runtime_configuration

    configuration = validate_runtime_configuration(require_credentials=False)
    tolerance = penetration_depth_tolerance_from_env()

    missing = [
        package
        for module, package in _REQUIRED_MODULES.items()
        if importlib.util.find_spec(module) is None
    ]
    try:
        blender = blender_bin()
    except (FileNotFoundError, RuntimeError) as exc:
        blender = None
        missing.append(str(exc))
    return {
        "ok": not missing,
        **configuration,
        "api_key_configured": configuration["credentials"]["text"],
        "penetration_depth_tolerance_m": tolerance,
        "compaction_trigger_tokens": os.environ.get(
            "OPENAI_COMPACTION_COST_TRIGGER_TOKENS", "110000"
        ),
        "cpu_threads": {name: os.environ.get(name) for name in THREAD_ENVIRONMENT_VARIABLES},
        "render_device": os.environ.get("ARTISAN_RENDER_DEVICE", "cpu"),
        "blender": blender,
        "missing": missing,
    }


def _canonical_record_id(workspace: Path, run_id: str) -> str | None:
    from storage.identifiers import validate_record_id
    from storage.records_index import load_records_index
    from storage.repo import StorageRepo

    repo = StorageRepo(workspace)
    matches: list[str] = []
    for row in load_records_index(repo):
        if row.get("run_id") != run_id:
            continue
        try:
            record_id = validate_record_id(row.get("record_id"))
            record_path = repo.layout.record_metadata_path(record_id)
            record_path.resolve().relative_to(repo.layout.records_root.resolve())
            if not record_path.is_file():
                continue
            record = repo.read_json(record_path)
        except (ValueError, OSError):
            continue
        if not isinstance(record, dict) or record.get("record_id") != record_id:
            continue
        source = record.get("source")
        if isinstance(source, dict) and source.get("run_id") == run_id:
            matches.append(record_id)
    if len(matches) > 1:
        raise ValueError("Multiple canonical records are registered for this run")
    return matches[0] if matches else None


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            print(
                json.dumps(
                    initialize_config(
                        args.config,
                        workspace=args.workspace,
                        blender=args.blender,
                        credentials=args.credentials,
                    ),
                    indent=2,
                )
            )
            return 0
        config = load_config(args.config)
        apply_runtime_config(config, vars(args))
        if args.command in {"run", "resume", "doctor"}:
            credentials = (
                args.credentials
                or os.environ.get("ARTISAN_CREDENTIALS")
                or config.get("credentials_file")
            )
            load_credentials(Path(credentials) if credentials else None)
        workspace = (
            Path(
                args.workspace
                or os.environ.get("ARTISAN_WORKSPACE")
                or config.get("workspace")
                or (args.config or default_config_path()).parent / "workspace"
            )
            .expanduser()
            .absolute()
        )
        if args.command == "doctor":
            report = _preflight()
            selected_config = args.config or default_config_path()
            report["config_file"] = (
                str(selected_config.expanduser().absolute()) if selected_config.is_file() else None
            )
            print(json.dumps(report, indent=2))
            return 0 if report["ok"] else 1
        if args.command == "viewer":
            import uvicorn

            from viewer.api.app import create_app

            uvicorn.run(create_app(repo_root=workspace), host=args.host, port=args.port)
            return 0
        if args.command == "inspect":
            from pipeline.state import load_checkpoint

            state = load_checkpoint(workspace, args.run_id)
            print(json.dumps(state.to_dict(), indent=2))
            return 0
        report = _preflight()
        if not report["ok"]:
            raise RuntimeError("Missing runtime dependencies: " + ", ".join(report["missing"]))
        if not all(report["credentials"].values()):
            raise ValueError(
                "Configure API credentials for the selected text provider and OpenAI images, or pass --credentials"
            )
        cost = float(
            args.max_cost_usd
            if args.max_cost_usd is not None
            else os.environ.get("ARTISAN_MAX_COST_USD")
            or config.get("max_cost_usd", default_cost_limit(report["text_models"]["coding"]))
        )
        if not 0 < cost < float("inf"):
            raise ValueError("--max-cost-usd must be a positive finite number")
        from pipeline import run as pipeline_run
        from pipeline.state import Phase, checkpoint_path, load_checkpoint

        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        if args.command == "run":
            if bool(args.prompt) == bool(args.prompt_file):
                raise ValueError("Provide either a prompt or --prompt-file")
            prompt = (
                args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else args.prompt
            )
            if not prompt.strip():
                raise ValueError("Prompt must not be empty")
            for image in args.image:
                if not image.is_file():
                    raise FileNotFoundError(f"Reference image does not exist: {image}")
            if args.run_id and (
                checkpoint_path(workspace, args.run_id).parent.exists()
                or _canonical_record_id(workspace, args.run_id) is not None
            ):
                raise ValueError("Run already exists; use artisan resume")
            state = asyncio.run(
                pipeline_run.run(
                    prompt,
                    repo_root=workspace,
                    max_cost_usd=cost,
                    run_id=args.run_id,
                    image_paths=[str(p.absolute()) for p in args.image],
                )
            )
        else:
            checkpoint = load_checkpoint(workspace, args.run_id)
            if Path(checkpoint.repo_root).expanduser().resolve() != workspace.resolve():
                raise ValueError(
                    "Checkpoint belongs to a different workspace. Restore its original "
                    "workspace location before resuming; relocated workspaces are not supported."
                )
            state = asyncio.run(
                pipeline_run.resume(args.run_id, repo_root=workspace, max_cost_usd=cost)
            )
        canonical_record_id = _canonical_record_id(workspace, state.run_id)
        registered = canonical_record_id is not None
        print(
            json.dumps(
                {
                    "run_id": state.run_id,
                    "phase": state.phase.value,
                    "status": state.status,
                    "cost_usd": state.cost_usd,
                    "registered": registered,
                    "canonical_record_id": canonical_record_id,
                },
                indent=2,
            )
        )
        return 0 if state.phase == Phase.DONE and registered else 1
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"artisan: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
