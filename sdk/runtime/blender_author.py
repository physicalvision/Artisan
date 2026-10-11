from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from pipeline import blender as blender_package

_AUTHOR_DRIVER = Path(blender_package.__file__).with_name("_author_driver.py")

DEFAULT_VISUAL_BUDGET = 12000


def _validated_anchor_sidecar(obj_path: Path, report: dict[str, Any]) -> Path | None:
    anchor_count = int(report.get("anchor_count") or 0)
    if anchor_count == 0:
        return None
    sidecar = obj_path.with_suffix(f"{obj_path.suffix}.anchors.json")
    if not sidecar.is_file():
        raise ValueError(f"Blender report declared {anchor_count} Anchors but sidecar is missing")
    from sdk import validate_mesh_anchor_sidecar

    parsed = validate_mesh_anchor_sidecar(obj_path, sidecar)
    if len(parsed) != anchor_count:
        raise ValueError(
            f"Blender Anchor sidecar count mismatch: report={anchor_count} parsed={len(parsed)}"
        )
    return sidecar


def _stage_cached_outputs(
    cached_obj: Path,
    out_obj: Path,
    report: dict[str, Any],
) -> None:
    sidecar = _validated_anchor_sidecar(cached_obj, report)
    shutil.copy(cached_obj, out_obj)
    staged_sidecar = out_obj.with_suffix(f"{out_obj.suffix}.anchors.json")
    if sidecar is not None:
        shutil.copy(sidecar, staged_sidecar)
        report["anchors"] = str(staged_sidecar)
    else:
        staged_sidecar.unlink(missing_ok=True)


def _cache_key(script_text: str, out_name: str, visual_budget: int, blender: str) -> str:
    h = hashlib.sha256()
    h.update(script_text.encode("utf-8"))
    h.update(f"|{out_name}|{visual_budget}|{blender}".encode("utf-8"))
    return h.hexdigest()[:16]


def run_script(
    script_path: str | Path,
    out_name: str,
    workspace: str | Path,
    *,
    visual_budget: int = DEFAULT_VISUAL_BUDGET,
    timeout: int = 300,
) -> dict[str, Any]:
    from pipeline.blender.runtime import blender_bin

    script_path = Path(script_path)
    workspace = Path(workspace)
    inputs = workspace / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    out_obj = inputs / f"{out_name}.obj"

    blender = blender_bin()
    script_text = script_path.read_text(encoding="utf-8")
    key = _cache_key(script_text, out_name, visual_budget, blender)
    cache_dir = workspace / "blender" / "_cache" / key
    cached_obj = cache_dir / "out.obj"
    cached_report = cache_dir / "report.json"

    if cached_obj.exists() and cached_report.exists():
        try:
            rep = json.loads(cached_report.read_text(encoding="utf-8"))
            if rep.get("ok"):
                _stage_cached_outputs(cached_obj, out_obj, rep)
                rep["out"] = str(out_obj)
                rep["cached"] = True
                return rep
        except Exception:  # noqa: BLE001 — corrupt cache entry; recompute
            pass

    cache_dir.mkdir(parents=True, exist_ok=True)
    report_path = cache_dir / "report.json"
    preview_path = cache_dir / "preview.png"

    cmd = [
        blender,
        "--background",
        "--factory-startup",
        "--python",
        str(_AUTHOR_DRIVER),
        "--",
        "--script",
        str(script_path),
        "--out",
        str(cached_obj),
        "--report",
        str(report_path),
        "--preview",
        str(preview_path),
        "--visual-budget",
        str(visual_budget),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "stderr": f"blender authoring timed out after {timeout}s"}

    if not report_path.exists():
        return {
            "ok": False,
            "stderr": (
                f"blender produced no report (rc={proc.returncode}).\n"
                f"STDOUT:\n{proc.stdout[-1500:]}\nSTDERR:\n{proc.stderr[-1500:]}"
            ),
        }

    rep = json.loads(report_path.read_text(encoding="utf-8"))
    if rep.get("ok") and cached_obj.exists():
        try:
            _stage_cached_outputs(cached_obj, out_obj, rep)
        except Exception as exc:  # noqa: BLE001 - return validation error to repair loop
            rep["ok"] = False
            rep["stderr"] = f"invalid Blender semantic Anchor sidecar: {type(exc).__name__}: {exc}"
        else:
            rep["out"] = str(out_obj)
            rep["cached"] = False
    return rep
