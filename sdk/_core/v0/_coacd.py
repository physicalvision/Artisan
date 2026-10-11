from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator, Optional

_COACD_PARAMS: dict[str, object] = {
    "threshold": 0.05,
    "max_convex_hull": 32,
    "preprocess_mode": "auto",
    "preprocess_resolution": 50,
    "resolution": 2000,
    "mcts_nodes": 20,
    "mcts_iterations": 150,
    "mcts_max_depth": 3,
    "pca": False,
    "merge": True,
}
_COACD_SEED = 0
_COACD_CACHE_VERSION = 1


@contextmanager
def _cache_key_lock(cache_dir: Path, key: str) -> Iterator[None]:
    lock_dir = cache_dir / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f"{key}.lock").open("a+b") as handle:
        _lock_file(handle)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _lock_file(handle: IO[bytes]) -> None:
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _cached_pieces(out_dir: Path) -> list[Path] | None:
    manifest = out_dir / "pieces.json"
    if not manifest.is_file():
        return None
    try:
        names = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(names, list) or not names:
        return None
    pieces = [out_dir / str(name) for name in names]
    if any(piece.parent != out_dir or not piece.is_file() for piece in pieces):
        return None
    return pieces


def _record_attempt_event(action: str, key: str) -> None:
    path = os.environ.get("ARTISAN_COACD_ATTEMPT_LOG")
    if not path:
        return
    try:
        with Path(path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"action": action, "key": key}) + "\n")
    except OSError:
        pass


def coacd_version() -> str:
    try:
        return str(importlib.metadata.version("coacd"))
    except importlib.metadata.PackageNotFoundError:
        return "absent"


def coacd_policy_version() -> str:
    params = hashlib.sha256(
        json.dumps(_COACD_PARAMS, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return f"{coacd_version()}-cache{_COACD_CACHE_VERSION}-seed{_COACD_SEED}-params{params}"


def cache_key(mesh_bytes: bytes, *, params: Optional[dict[str, object]] = None) -> str:
    merged = dict(_COACD_PARAMS)
    if params:
        merged.update(params)
    h = hashlib.sha256()
    h.update(mesh_bytes)
    h.update(json.dumps(merged, sort_keys=True, default=str).encode("utf-8"))
    h.update(f"seed={_COACD_SEED};v{_COACD_CACHE_VERSION};coacd={coacd_version()}".encode())
    return h.hexdigest()


_COACD_SUBPROCESS_TIMEOUT_S = float(os.environ.get("ARTISAN_COACD_TIMEOUT_S", "900"))


def _coacd_child(payload_path: str) -> None:
    import json as _json

    import numpy as _np

    from sdk._dependencies import require_coacd as _require

    _coacd = _require(feature="Convex collision decomposition (CoACD)")
    payload = Path(payload_path)
    data = _np.load(payload, allow_pickle=False)
    spec = _json.loads(payload.with_suffix(".params.json").read_text(encoding="utf-8"))
    seed = spec.pop("__seed__")
    try:
        _coacd.set_log_level("error")
    except Exception:  # noqa: BLE001 — older coacd without log control
        pass
    mesh = _coacd.Mesh(data["vertices"], data["faces"])
    parts = _coacd.run_coacd(mesh, seed=seed, **spec)
    out: dict[str, object] = {"count": _np.asarray([len(parts)], dtype=_np.int64)}
    for index, part in enumerate(parts):
        out[f"v{index}"] = _np.asarray(part[0], dtype=_np.float64)
        out[f"f{index}"] = _np.asarray(part[1], dtype=_np.int64)
    _np.savez(payload.with_suffix(".out.npz"), **out)


def _run_coacd_isolated(vertices, faces, call_params: dict, seed: int) -> list:
    import json as _json
    import multiprocessing as _mp

    import numpy as _np

    scratch = Path(tempfile.mkdtemp(prefix=".coacd-solve."))
    try:
        payload = scratch / "in.npz"
        _np.savez(payload, vertices=vertices, faces=faces)
        spec = dict(call_params)
        spec["__seed__"] = seed
        payload.with_suffix(".params.json").write_text(_json.dumps(spec), encoding="utf-8")
        try:
            ctx = _mp.get_context("forkserver")
        except ValueError:  # pragma: no cover — platform without forkserver
            ctx = _mp.get_context("spawn")
        proc = ctx.Process(target=_coacd_child, args=(str(payload),))
        proc.start()
        proc.join(_COACD_SUBPROCESS_TIMEOUT_S)
        if proc.is_alive():
            proc.terminate()
            proc.join(10)
            raise RuntimeError(
                f"CoACD solve exceeded {_COACD_SUBPROCESS_TIMEOUT_S:g}s and was terminated"
            )
        if proc.exitcode != 0:
            raise RuntimeError(f"CoACD solver crashed (exitcode={proc.exitcode})")
        result_path = payload.with_suffix(".out.npz")
        if not result_path.is_file():
            raise RuntimeError("CoACD solver exited cleanly but wrote no pieces")
        with _np.load(result_path, allow_pickle=False) as result:
            count = int(result["count"][0])
            return [(result[f"v{i}"], result[f"f{i}"]) for i in range(count)]
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def decompose_mesh(
    mesh_path: str | Path,
    *,
    cache_dir: str | Path,
    params: Optional[dict[str, object]] = None,
) -> list[Path]:
    import numpy as np
    import trimesh

    from sdk._dependencies import require_coacd

    require_coacd(feature="Convex collision decomposition (CoACD)")

    mesh_path = Path(mesh_path)
    mesh_bytes = mesh_path.read_bytes()
    key = cache_key(mesh_bytes, params=params)
    _record_attempt_event("attempted", key)
    cache_root = Path(cache_dir)
    out_dir = cache_root / key
    cached = _cached_pieces(out_dir)
    if cached is not None:
        return cached

    with _cache_key_lock(cache_root, key):
        cached = _cached_pieces(out_dir)
        if cached is not None:
            return cached
        if out_dir.exists():
            shutil.rmtree(out_dir)

        tm = trimesh.load(mesh_path, force="mesh")
        vertices = np.asarray(tm.vertices, dtype=np.float64)
        faces = np.asarray(tm.faces, dtype=np.int32)
        call_params = dict(_COACD_PARAMS)
        if params:
            call_params.update(params)
        parts = _run_coacd_isolated(vertices, faces, call_params, _COACD_SEED)
        cache_root.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=cache_root))
        try:
            names: list[str] = []
            for i, part in enumerate(parts):
                verts, faces = part[0], part[1]
                piece = trimesh.Trimesh(
                    vertices=np.asarray(verts, dtype=np.float64),
                    faces=np.asarray(faces, dtype=np.int64),
                    process=False,
                )
                name = f"piece_{i:03d}.obj"
                piece.export(temp_dir / name)
                names.append(name)

            manifest = temp_dir / "pieces.json"
            manifest.write_text(json.dumps(names), encoding="utf-8")
            with manifest.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temp_dir, out_dir)
            _record_attempt_event("published", key)
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

        published = _cached_pieces(out_dir)
        if published is None:
            raise RuntimeError(f"CoACD cache publish produced no pieces for key {key}")
        return published
