from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator

_REFERENCE: ContextVar[dict[str, Any] | None] = ContextVar("accepted_inspection", default=None)


def current_reference() -> dict[str, Any] | None:
    reference = _REFERENCE.get()
    return dict(reference) if reference is not None else None


def candidate_reference(manifest_path: Path, candidate_id: str) -> dict[str, Any]:
    return {
        "manifest_path": str(manifest_path),
        "candidate_id": candidate_id,
    }


def workspace_reference(model_path: Path) -> dict[str, Any] | None:
    from sdk.runtime.release_candidate import load_manifest

    path = model_path.parent / "candidate/manifest.json"
    if not path.is_file():
        return None
    return candidate_reference(path, load_manifest(path).candidate_id)


def _verify(model_path: Path, reference: dict[str, Any]) -> Any:
    from sdk.runtime.release_candidate import (
        CandidateError,
        _hash_payload,
        dependency_sha256_for_model,
        sha256_file,
        verify_release_candidate,
    )

    if not isinstance(reference, dict) or set(reference) != {
        "manifest_path",
        "candidate_id",
    }:
        raise CandidateError("invalid accepted inspection reference")
    if not all(
        isinstance(reference[k], str) and reference[k] for k in ("manifest_path", "candidate_id")
    ):
        raise CandidateError("accepted inspection requires manifest path and candidate ID")
    path = Path(reference["manifest_path"])
    manifest = verify_release_candidate(path)
    if manifest.candidate_id != reference["candidate_id"]:
        raise CandidateError("accepted inspection candidate ID changed")
    if sha256_file(model_path) != manifest.model_sha256:
        raise CandidateError("accepted inspection source does not match candidate")
    if dependency_sha256_for_model(model_path) != manifest.dependency_sha256:
        raise CandidateError("accepted inspection dependencies do not match candidate")
    blueprint = model_path.parent / "blueprint.json"
    bp_sha = sha256_file(blueprint) if blueprint.is_file() else _hash_payload({})
    if bp_sha != manifest.blueprint_sha256:
        raise CandidateError("accepted inspection blueprint does not match candidate")
    return manifest


@contextmanager
def accepted_inspection(model_path: Path, reference: dict[str, Any] | None) -> Iterator[None]:
    if reference is None:
        yield
        return
    from sdk._core.v0.assets import preserve_mesh_inventory
    from sdk.runtime.release_candidate import _dependency_manifest, dependency_sha256_for_model

    manifest = _verify(model_path, reference)
    dependencies = _dependency_manifest(model_path)
    inventory = {
        (model_path.parent / str(entry["path"])).resolve(): (
            model_path.parent / str(entry["path"])
        ).read_bytes()
        for entry in dependencies
        if str(entry["path"]).endswith(".obj")
    }
    if dependency_sha256_for_model(model_path) != manifest.dependency_sha256:
        raise ValueError("accepted inspection dependencies changed during inventory capture")
    token = _REFERENCE.set(dict(reference))
    try:
        with preserve_mesh_inventory(inventory):
            try:
                yield
            finally:
                _verify(model_path, reference)
    finally:
        _REFERENCE.reset(token)


def guarded_inspection(method: Callable[..., Any]) -> Callable[..., Any]:

    @wraps(method)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        with accepted_inspection(self.model_path, self._accepted_inspection_reference()):
            return method(self, *args, **kwargs)

    return wrapped
