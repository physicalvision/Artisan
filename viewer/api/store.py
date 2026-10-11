from __future__ import annotations

import threading
from pathlib import Path

from storage.materialize import MaterializationStore
from storage.records import RecordStore
from storage.repo import StorageRepo
from viewer.api.store_materialization import ViewerMaterializationStore
from viewer.api.store_records import ViewerRecordsStore


class ViewerStore:
    def __init__(self, repo_root: Path, *, ensure_search_index: bool = True) -> None:
        self.repo_root = repo_root.resolve()
        self.repo = StorageRepo(self.repo_root)
        self.repo.ensure_layout()
        self.record_store = RecordStore(self.repo)
        self.materialization_store = MaterializationStore(self.repo)
        self._compile_locks_guard = threading.Lock()
        self._compile_locks: dict[str, threading.Lock] = {}
        self.materialization = ViewerMaterializationStore(self)
        self.records = ViewerRecordsStore(self)
