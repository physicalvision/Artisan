from __future__ import annotations

import json
import shutil
from pathlib import Path

from storage.layout import StorageLayout

INPUTS_SUBDIR = "inputs"
INPUTS_MANIFEST = "manifest.json"


def workspace_root(repo_root: Path, run_id: str) -> Path:
    return StorageLayout(root=repo_root).run_dir(run_id) / "workspace"


def references_inputs_dir(repo_root: Path, run_id: str) -> Path:
    return workspace_root(repo_root, run_id) / "references" / INPUTS_SUBDIR


def paths(repo_root: Path, run_id: str) -> dict[str, Path]:
    w = workspace_root(repo_root, run_id)
    return {
        "root": w,
        "blueprint": w / "blueprint.json",
        "p1_selfcheck": w / "reports" / "P1_selfcheck.json",
        "asset_build": w / "asset_build.py",
        "model": w / "model.py",
        "blender": w / "blender",
        "inputs": w / "inputs",
        "design_board_png": w / "references" / "design_board.png",
        "design_board_md": w / "references" / "design_board.md",
        "design_board_meta": w / "references" / "design_board_meta.json",
        "renders": w / "renders",
        "reports": w / "reports",
        "candidate": w / "candidate",
        "candidate_manifest": w / "candidate" / "manifest.json",
        "physics_report": w / "reports" / "physics_validation.json",
        "exports": w / "exports",
        "sessions": w / "sessions",
    }


def ensure(repo_root: Path, run_id: str) -> dict[str, Path]:
    p = paths(repo_root, run_id)
    for k in ("root", "renders", "reports", "sessions", "blender", "inputs"):
        p[k].mkdir(parents=True, exist_ok=True)
    (p["root"] / "references").mkdir(parents=True, exist_ok=True)
    return p


def _unique_dest(inputs_dir: Path, name: str, used: set[str]) -> Path:
    base = name or "input"
    if base not in used:
        used.add(base)
        return inputs_dir / base
    stem, dot, ext = base.partition(".")
    i = 1
    while True:
        cand = f"{stem}_{i}{dot}{ext}" if dot else f"{stem}_{i}"
        if cand not in used:
            used.add(cand)
            return inputs_dir / cand
        i += 1


def stage_references(
    repo_root: Path, run_id: str, references: list[dict], *, prompt: str = ""
) -> list[dict]:
    w = workspace_root(repo_root, run_id)
    inputs_dir = w / "references" / INPUTS_SUBDIR
    try:
        w_resolved = w.resolve()
    except OSError:
        w_resolved = w
    used: set[str] = set()
    staged: list[dict] = []
    manifest: list[dict] = []
    for ref in references:
        new_ref = dict(ref)
        src = ref.get("path")
        kind = ref.get("media_kind")
        if src:
            src_path = Path(src)
            already_inside = False
            try:
                already_inside = src_path.resolve().is_relative_to(w_resolved)
            except (ValueError, OSError):
                already_inside = False
            if already_inside:
                pass
            elif src_path.is_file():
                inputs_dir.mkdir(parents=True, exist_ok=True)
                dest = _unique_dest(inputs_dir, src_path.name, used)
                shutil.copy2(src_path, dest)
                new_ref.setdefault("source", src)
                new_ref["path"] = str(dest)
            else:
                new_ref.setdefault("source", src)
        staged.append(new_ref)
        manifest.append(
            {
                "media_kind": kind,
                "source": new_ref.get("source", src),
                "path": new_ref.get("path"),
            }
        )
    if references:
        inputs_dir.mkdir(parents=True, exist_ok=True)
        (inputs_dir / INPUTS_MANIFEST).write_text(
            json.dumps({"run_id": run_id, "prompt": prompt, "references": manifest}, indent=2),
            encoding="utf-8",
        )
    return staged
