from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROMPT_MANIFEST_SCHEMA_VERSION = 1

_DYNAMIC_SOURCE_RELATIVE_PATHS = (
    "pipeline/agents/_llm.py",
    "pipeline/agents/coding.py",
    "pipeline/agents/design.py",
    "pipeline/agents/design_harness.py",
    "pipeline/agents/review.py",
    "pipeline/context_policy.py",
    "pipeline/harness_base.py",
    "pipeline/harness_review.py",
    "pipeline/harness_sdk.py",
    "pipeline/policies.py",
    "pipeline/providers.py",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _relative_key(path: Path, source_root: Path) -> str:
    try:
        return path.resolve().relative_to(source_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _file_hashes(paths: Iterable[Path], source_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted({Path(item) for item in paths}, key=lambda item: item.as_posix()):
        if not path.is_file():
            raise FileNotFoundError(f"prompt provenance source missing: {path}")
        hashes[_relative_key(path, source_root)] = _sha256_bytes(path.read_bytes())
    return hashes


def _files_under(roots: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"prompt provenance root missing: {root}")
        files.extend(
            path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts
        )
    return files


def git_source_info(source_root: Path) -> dict[str, str | bool | None]:

    def _run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=source_root,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    commit = _run("rev-parse", "HEAD")
    status = _run("status", "--porcelain", "--untracked-files=all")
    return {"commit": commit or None, "dirty": bool(status) if status is not None else None}


def build_prompt_snapshot(
    *,
    source_root: Path,
    phase: str,
    agent_settings: Mapping[str, Mapping[str, Any]],
    prompt_paths: Sequence[Path],
    tool_schemas: Mapping[str, Sequence[Mapping[str, Any]]],
    sdk_roots: Sequence[Path],
    dynamic_source_paths: Sequence[Path],
    git_info: Mapping[str, Any] | None = None,
    captured_at: str | None = None,
) -> dict[str, Any]:
    root = source_root.resolve()
    prompt_hashes = _file_hashes(prompt_paths, root)
    system_prompts = {
        path: digest for path, digest in prompt_hashes.items() if Path(path).name == "system.md"
    }
    templates = {
        path: digest for path, digest in prompt_hashes.items() if Path(path).name != "system.md"
    }
    system_prompts_sha256 = _sha256_json(system_prompts)
    tool_hashes = {
        role: _sha256_json(list(schemas)) for role, schemas in sorted(tool_schemas.items())
    }
    sdk_files = _file_hashes(_files_under(sdk_roots), root)
    sdk_docs = {"files": sdk_files, "sha256": _sha256_json(sdk_files)}
    dynamic_sources = _file_hashes(dynamic_source_paths, root)
    git = dict(git_info) if git_info is not None else git_source_info(root)
    agents = {role: dict(settings) for role, settings in sorted(agent_settings.items())}

    stable_payload = {
        "agents": agents,
        "system_prompts": system_prompts,
        "templates": templates,
        "tool_schemas": tool_hashes,
        "sdk_docs": sdk_docs,
        "dynamic_sources": dynamic_sources,
        "git": git,
    }
    return {
        "captured_at": captured_at or _utc_now(),
        "phase": phase,
        **stable_payload,
        "system_prompts_sha256": system_prompts_sha256,
        "combined_digest": _sha256_json(stable_payload),
    }


def write_prompt_manifest(path: Path, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    manifest: dict[str, Any]
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict) or loaded.get("schema_version") != 1:
            raise ValueError(f"unsupported prompt manifest: {path}")
        snapshots = loaded.get("snapshots")
        if not isinstance(snapshots, list):
            raise ValueError(f"prompt manifest snapshots must be a list: {path}")
        manifest = loaded
    else:
        manifest = {"schema_version": PROMPT_MANIFEST_SCHEMA_VERSION, "snapshots": []}

    snapshots = manifest["snapshots"]
    if snapshots and snapshots[-1].get("combined_digest") == snapshot.get("combined_digest"):
        return manifest
    snapshots.append(dict(snapshot))

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return manifest


def capture_pipeline_prompt_manifest(
    path: Path,
    *,
    source_root: Path,
    phase: str,
    agent_settings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    from pipeline.agents.design_harness import _design_tool_schemas
    from pipeline.harness_review import _review_tool_schemas
    from pipeline.harness_sdk import _tool_schemas

    prompt_root = source_root / "pipeline" / "agents" / "prompts"
    prompt_paths = sorted(
        path
        for path in prompt_root.rglob("*")
        if path.is_file() and path.suffix in {".md", ".json"}
    )
    sdk_roots = (
        source_root / "sdk" / "_docs",
        source_root / "sdk" / "_examples",
    )
    snapshot = build_prompt_snapshot(
        source_root=source_root,
        phase=phase,
        agent_settings=agent_settings,
        prompt_paths=prompt_paths,
        tool_schemas={
            "design": _design_tool_schemas(),
            "coding": _tool_schemas(),
            "review": _review_tool_schemas(),
        },
        sdk_roots=sdk_roots,
        dynamic_source_paths=[source_root / path for path in _DYNAMIC_SOURCE_RELATIVE_PATHS],
    )
    return write_prompt_manifest(path, snapshot)
