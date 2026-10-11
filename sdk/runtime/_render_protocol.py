from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
import zlib
from pathlib import Path
from typing import Any


class RenderIntegrityError(RuntimeError):
    pass


_TEXTURE_MAPS = ("color.png", "roughness.png", "metalness.png", "normal.png")


def input_identity(glb: Path) -> dict[str, Any]:
    sidecar = glb.with_suffix(".materials.json")
    texture_files: dict[str, str | None] = {}
    if sidecar.exists():
        specs = json.loads(sidecar.read_text())
        directories = {
            str(spec["texture_dir"]) for spec in specs.values() if spec.get("texture_dir")
        }
        for directory in sorted(directories):
            for name in _TEXTURE_MAPS:
                path = Path(directory).absolute() / name
                texture_files[str(path)] = (
                    hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
                )
    return {
        "glb_sha256": hashlib.sha256(glb.read_bytes()).hexdigest(),
        "materials_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest()
        if sidecar.exists()
        else None,
        "texture_files": texture_files,
    }


def verify_input(glb: Path, expected: dict[str, Any]) -> None:
    if input_identity(glb) != expected:
        raise RenderIntegrityError("Render input bytes changed")


def stage_input(source: Path, destination: Path, expected: dict[str, Any]) -> dict[str, Any]:
    verify_input(source, expected)
    shutil.copy2(source, destination)
    sidecar = source.with_suffix(".materials.json")
    if sidecar.exists():
        specs = json.loads(sidecar.read_text())
        copies: dict[str, str] = {}
        for spec in specs.values():
            if not spec.get("texture_dir"):
                continue
            directory = str(spec["texture_dir"])
            if directory not in copies:
                target = destination.parent / "textures" / str(len(copies))
                target.mkdir(parents=True)
                for name in _TEXTURE_MAPS:
                    src = Path(directory).absolute() / name
                    digest = expected["texture_files"][str(src)]
                    if digest is not None:
                        payload = src.read_bytes()
                        if hashlib.sha256(payload).hexdigest() != digest:
                            raise RenderIntegrityError("Texture changed while staging")
                        (target / name).write_bytes(payload)
                copies[directory] = str(target.absolute())
            spec["texture_dir"] = copies[directory]
        write_report(destination.with_suffix(".materials.json"), specs)
    verify_input(source, expected)
    staged = input_identity(destination)
    if staged["glb_sha256"] != expected["glb_sha256"]:
        raise RenderIntegrityError("GLB changed while staging")
    return staged


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_response(
    report: dict[str, Any],
    *,
    request_id: str,
    expected_input: dict[str, Any],
    views: list[str],
    attempt_root: Path,
) -> dict[str, Path]:
    if report.get("request_id") != request_id or report.get("input_identity") != expected_input:
        raise RenderIntegrityError("Render response identity mismatch")
    if report.get("errors"):
        if report.get("error_kind") == "RenderIntegrityError":
            raise RenderIntegrityError("Render input changed in the worker")
        raise RuntimeError("Render worker failed: " + "\n".join(str(e) for e in report["errors"]))
    renders = report.get("renders")
    if not isinstance(renders, dict) or set(renders) != set(views):
        raise RenderIntegrityError("Render response does not contain every requested view")
    root = attempt_root.resolve()
    validated: dict[str, Path] = {}
    for view, name in renders.items():
        path = Path(name).resolve()
        if not path.is_relative_to(root):
            raise RenderIntegrityError("Render output escaped its request directory")
        _validate_png(path)
        validated[view] = path
    return validated


def _validate_png(path: Path) -> None:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise RenderIntegrityError("Render output is not a PNG")
    offset = 8
    chunks = []
    while offset + 12 <= len(data):
        length = int.from_bytes(data[offset : offset + 4], "big")
        end = offset + 12 + length
        if end > len(data):
            break
        kind = data[offset + 4 : offset + 8]
        checksum = int.from_bytes(data[end - 4 : end], "big")
        if zlib.crc32(data[offset + 4 : end - 4]) != checksum:
            raise RenderIntegrityError("Render output has a corrupt PNG chunk")
        chunks.append(kind)
        offset = end
        if kind == b"IEND":
            if length != 0 or offset != len(data):
                raise RenderIntegrityError("Render output has an invalid PNG end")
            break
    if not chunks or chunks[0] != b"IHDR" or b"IDAT" not in chunks or chunks[-1] != b"IEND":
        raise RenderIntegrityError("Render output is an incomplete PNG")
