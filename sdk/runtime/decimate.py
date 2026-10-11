from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_MARKER = "# artisan-decimated"


def _face_count(obj_path: Path) -> int:
    n = 0
    try:
        with open(obj_path) as fh:
            for line in fh:
                if line.startswith("f "):
                    n += 1
    except OSError:
        return 0
    return n


def _is_decimated(obj_path: Path) -> bool:
    try:
        with open(obj_path) as fh:
            return fh.readline().startswith(_MARKER)
    except OSError:
        return False


def _cache_obj_for(named_obj: Path, cache_dir: Path) -> Path | None:
    destination_key = hashlib.sha256(named_obj.resolve().as_posix().encode("utf-8")).hexdigest()
    sidecar = cache_dir / "destinations" / f"{destination_key}.json"
    if not sidecar.exists():
        return None
    try:
        key = json.loads(sidecar.read_text()).get("cache_key")
    except (OSError, ValueError):
        return None
    if not key:
        return None
    cache_obj = cache_dir / f"{key}.obj"
    return cache_obj if cache_obj.exists() else None


def _write_marked(path: Path, mesh: Any) -> None:
    obj_text = mesh.export(file_type="obj")
    if isinstance(obj_text, bytes):
        obj_text = obj_text.decode()
    path.write_text(f"{_MARKER}\n{obj_text}")


def maybe_decimate_model_meshes(
    asset_root: Path | str,
    *,
    total_threshold: int = 250_000,
    ratio: float = 0.16,
    min_faces: int = 5_000,
) -> list[dict[str, Any]]:
    asset_root = Path(asset_root)
    mesh_dir = asset_root / "meshes"
    cache_dir = asset_root.parent / "cache" / "cadquery"
    if not mesh_dir.is_dir():
        return []

    objs = sorted(p for p in mesh_dir.glob("*.obj"))
    counts = {p: _face_count(p) for p in objs}
    total = sum(counts.values())
    if total <= total_threshold:
        return []

    import trimesh

    log: list[dict[str, Any]] = []
    for p in objs:
        nf = counts[p]
        if nf <= min_faces or _is_decimated(p):
            continue
        try:
            mesh = trimesh.load(p, force="mesh")
            target = max(4, int(round(len(mesh.faces) * ratio)))
            dec = mesh.simplify_quadric_decimation(face_count=target)
        except Exception as exc:  # noqa: BLE001 — never break a good compile
            log.append({"mesh": p.name, "faces": nf, "decimated": None, "error": str(exc)})
            continue
        _write_marked(p, dec)
        cache_obj = _cache_obj_for(p, cache_dir)
        if cache_obj is not None:
            _write_marked(cache_obj, dec)
        log.append(
            {
                "mesh": p.name,
                "faces": nf,
                "decimated": int(len(dec.faces)),
                "persisted": cache_obj is not None,
            }
        )
    return log
