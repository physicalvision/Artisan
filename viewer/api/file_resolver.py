from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Protocol

from fastapi import HTTPException

from storage.identifiers import validate_record_id
from storage.repo import StorageRepo
from storage.revisions import (
    active_cost_path,
    active_inputs_dir,
    active_model_path,
    active_prompt_path,
    active_provenance_path,
    active_revision_id,
)
from viewer.api.store_types import MaterializeRecordAssetsResult

logger = logging.getLogger(__name__)


class MaterializingViewerStore(Protocol):
    repo: StorageRepo

    def materialize_record_assets(
        self,
        record_id: str,
        *,
        force: bool = False,
        target: str = "full",
    ) -> MaterializeRecordAssetsResult: ...


class ViewerFileResolver:
    def __init__(self, store: MaterializingViewerStore) -> None:
        self.store = store
        self.repo = store.repo

    def resolve_record_target(self, record_id: str, file_path: str) -> tuple[Path, Path]:
        self._validate_record_id(record_id)
        record_dir = self.repo.layout.record_dir(record_id)
        self._ensure_within_root(record_dir.resolve(), self.repo.root)

        if not record_dir.exists():
            raise HTTPException(status_code=404, detail=f"Record not found: {record_id}")

        requested_path = self._validated_relative_path(file_path)

        if requested_path.parts == ("model.urdf",):
            root = self.repo.layout.record_materialization_dir(record_id).resolve()
            target = self.repo.layout.record_materialization_urdf_path(record_id).resolve()
        elif requested_path.parts == ("compile_report.json",):
            root = self.repo.layout.record_materialization_dir(record_id).resolve()
            target = self.repo.layout.record_materialization_compile_report_path(
                record_id
            ).resolve()
        elif len(requested_path.parts) >= 2 and requested_path.parts[0] == "assets":
            root = self.repo.layout.record_materialization_dir(record_id).resolve()
            target = (root / requested_path).resolve()
        elif requested_path.parts in {
            ("prompt.txt",),
            ("model.py",),
            ("provenance.json",),
            ("cost.json",),
        }:
            record = self.repo.read_json(self.repo.layout.record_metadata_path(record_id))
            if not isinstance(record, dict):
                raise HTTPException(status_code=404, detail=f"Record not found: {record_id}")
            root = self.repo.layout.record_revision_dir(
                record_id,
                active_revision_id(self.repo, record_id, record=record),
            ).resolve()
            active_paths = {
                ("prompt.txt",): active_prompt_path(self.repo, record_id, record=record),
                ("model.py",): active_model_path(self.repo, record_id, record=record),
                ("provenance.json",): active_provenance_path(self.repo, record_id, record=record),
                ("cost.json",): active_cost_path(self.repo, record_id, record=record),
            }
            target = active_paths[requested_path.parts].resolve()
            root = target.parent.resolve()
        elif len(requested_path.parts) >= 2 and requested_path.parts[0] == "inputs":
            record = self.repo.read_json(self.repo.layout.record_metadata_path(record_id))
            if not isinstance(record, dict):
                raise HTTPException(status_code=404, detail=f"Record not found: {record_id}")
            root = active_inputs_dir(self.repo, record_id, record=record).resolve()
            target = (root / Path(*requested_path.parts[1:])).resolve()
        else:
            root = record_dir.resolve()
            target = (record_dir / requested_path).resolve()

        self._ensure_within_root(root, self.repo.root)
        self._ensure_within_root(target, root)

        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        return root, target

    async def resolve_record_target_with_materialization(
        self, record_id: str, file_path: str
    ) -> tuple[Path, Path]:
        requested_path = Path(file_path)
        try:
            return self.resolve_record_target(record_id, file_path)
        except HTTPException as exc:
            if exc.status_code != 404 or not should_attempt_materialize_for_record_path(file_path):
                raise
        try:
            await asyncio.to_thread(
                self.store.materialize_record_assets,
                record_id,
                target="visual",
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        try:
            return self.resolve_record_target(record_id, file_path)
        except HTTPException as exc:
            needs_forced_rebuild = exc.status_code == 404 and (
                len(requested_path.parts) >= 2
                and requested_path.parts[0] == "assets"
                and requested_path.parts[1] in {"meshes", "glb", "viewer"}
            )
            if not needs_forced_rebuild:
                raise
        try:
            await asyncio.to_thread(
                self.store.materialize_record_assets,
                record_id,
                force=True,
                target="visual",
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return self.resolve_record_target(record_id, file_path)

    def _validate_record_id(self, record_id: str) -> None:
        try:
            validate_record_id(record_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid record ID format")

    @staticmethod
    def _validated_relative_path(file_path: str) -> Path:
        requested_path = Path(file_path)
        if requested_path.is_absolute() or ".." in requested_path.parts:
            raise HTTPException(status_code=400, detail="Invalid file path")
        return requested_path

    @staticmethod
    def _ensure_within_root(target: Path, root: Path) -> None:
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="Access denied") from exc


def should_attempt_materialize_for_record_path(file_path: str) -> bool:
    requested_path = Path(file_path)
    if requested_path.parts == ("model.urdf",):
        return True
    if len(requested_path.parts) >= 2 and requested_path.parts[0] == "assets":
        return requested_path.parts[1] in {"meshes", "glb", "viewer"}
    return False
