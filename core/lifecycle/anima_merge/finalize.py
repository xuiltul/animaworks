from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit, resumable removal of an Anima merge tombstone."""

import asyncio
import json
import os
import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from core.anima_factory import validate_anima_name
from core.memory._io import atomic_write_text
from core.platform.locks import acquire_file_lock, release_file_lock
from core.time_utils import ensure_aware, now_local

from .journal import FinalizePhase, MergeJournal, MergePhase
from .service import AnimaMergeError, _read_json


@dataclass(frozen=True)
class FinalizeResult:
    source: str
    target: str
    dry_run: bool
    archive_path: Path
    merge_journal_path: Path
    journal_path: Path | None = None
    plan: dict[str, Any] | None = None


class AnimaMergeFinalizeService:
    """Finalize a completed merge after its rollback window."""

    def __init__(self, data_dir: Path, source: str, target: str) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.source = source
        self.target = target
        self.animas_dir = self.data_dir / "animas"
        self.source_dir = self.animas_dir / source
        self.state_dir = self.data_dir / "state"
        self.merge_journal_path = self.state_dir / f"merge_journal_{source}_{target}.json"
        self.journal_path = self.state_dir / f"merge_finalize_journal_{source}_{target}.json"
        self.lock_path = self.state_dir / "anima_merge.lock"

    def run(self, *, execute: bool = False, resume: bool = False) -> FinalizeResult:
        if resume and not execute:
            raise AnimaMergeError("--resume requires --execute")
        self._validate_names()
        with self._runtime_data_dir(), self._global_lock():
            if not execute:
                merge = self._preflight(None, resume=False, enforce_rollback_window=False)
                archive = self.data_dir / "archive" / "merged" / f"{self.source}_<timestamp>"
                return FinalizeResult(
                    source=self.source,
                    target=self.target,
                    dry_run=True,
                    archive_path=archive,
                    merge_journal_path=self.merge_journal_path,
                    plan={
                        "merge_status": merge["merge_status"],
                        "source_tombstoned": True,
                        "rollback_deadline": merge["rollback_deadline"],
                        "rollback_ready": merge["rollback_ready"],
                        "archive_path": str(archive),
                        "steps": [phase.value for phase in FinalizePhase if phase is not FinalizePhase.DONE],
                    },
                )

            journal = self._open_journal(resume=resume)
            self._run_phase(
                journal,
                FinalizePhase.PREFLIGHT,
                lambda: self._preflight(
                    journal,
                    resume=resume,
                    enforce_rollback_window=True,
                ),
            )
            archive_path = self._archive_path(journal)
            self._run_phase(
                journal,
                FinalizePhase.ARCHIVE_SOURCE,
                lambda: self._archive_source(archive_path),
            )
            self._run_phase(journal, FinalizePhase.REMOVE_CONFIG, self._remove_source_config)
            self._run_phase(
                journal,
                FinalizePhase.PURGE_NEO4J,
                lambda: self._purge_neo4j(archive_path),
            )
            self._run_phase(
                journal,
                FinalizePhase.PURGE_RESIDUALS,
                lambda: self._purge_residuals(archive_path),
            )
            self._run_phase(
                journal,
                FinalizePhase.VERIFY_REMOVAL,
                lambda: self._verify_removal(archive_path),
            )
            if not journal.is_completed(FinalizePhase.DONE):
                journal.start(FinalizePhase.DONE)
                journal.complete(
                    FinalizePhase.DONE,
                    {
                        "archive_path": str(archive_path),
                        "source_directory_absent": True,
                        "source_config_absent": True,
                        "org_sync_verified": True,
                    },
                )
            journal.finish()
            return FinalizeResult(
                source=self.source,
                target=self.target,
                dry_run=False,
                archive_path=archive_path,
                merge_journal_path=self.merge_journal_path,
                journal_path=self.journal_path,
            )

    def _validate_names(self) -> None:
        if self.source == self.target:
            raise AnimaMergeError("SOURCE and TARGET must be different")
        for label, name in (("SOURCE", self.source), ("TARGET", self.target)):
            error = validate_anima_name(name)
            if error:
                raise AnimaMergeError(f"Invalid {label} name '{name}': {error}")

    @contextmanager
    def _global_lock(self) -> Iterator[None]:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_file:
            try:
                acquire_file_lock(lock_file, exclusive=True, blocking=False)
            except OSError as exc:
                raise AnimaMergeError(f"Another Anima merge is in progress ({self.lock_path})") from exc
            try:
                yield
            finally:
                release_file_lock(lock_file)

    @contextmanager
    def _runtime_data_dir(self) -> Iterator[None]:
        previous = os.environ.get("ANIMAWORKS_DATA_DIR")
        os.environ["ANIMAWORKS_DATA_DIR"] = str(self.data_dir)
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("ANIMAWORKS_DATA_DIR", None)
            else:
                os.environ["ANIMAWORKS_DATA_DIR"] = previous

    def _open_journal(self, *, resume: bool) -> MergeJournal:
        try:
            return MergeJournal(self.journal_path, self.source, self.target, resume=resume)
        except ValueError as exc:
            raise AnimaMergeError(str(exc)) from exc

    @staticmethod
    def _run_phase(journal: MergeJournal, phase: FinalizePhase, action: Any) -> None:
        if journal.is_completed(phase):
            return
        journal.start(phase)
        try:
            artifacts = action() or {}
        except Exception as exc:
            journal.fail(phase, f"{type(exc).__name__}: {exc}")
            raise
        journal.complete(phase, artifacts)

    def _load_merge_journal(self) -> dict[str, Any]:
        if not self.merge_journal_path.is_file():
            raise AnimaMergeError(f"Completed merge journal not found: {self.merge_journal_path}")
        try:
            value = json.loads(self.merge_journal_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AnimaMergeError(f"Invalid merge journal {self.merge_journal_path}: {exc}") from exc
        if not isinstance(value, dict):
            raise AnimaMergeError(f"Invalid merge journal {self.merge_journal_path}")
        if value.get("source") != self.source or value.get("target") != self.target:
            raise AnimaMergeError("Merge journal belongs to a different source/target pair")
        done = value.get("phases", {}).get(MergePhase.DONE.value, {}).get("status")
        tombstone = value.get("phases", {}).get(MergePhase.TOMBSTONE.value, {}).get("status")
        if value.get("status") != "done" or done != "completed" or tombstone != "completed":
            raise AnimaMergeError("merge-finalize requires a DONE merge journal with a completed TOMBSTONE")
        return value

    def _preflight(
        self,
        journal: MergeJournal | None,
        *,
        resume: bool,
        enforce_rollback_window: bool,
    ) -> dict[str, Any]:
        merge = self._load_merge_journal()
        tombstone_artifacts = (
            merge.get("phases", {})
            .get(MergePhase.TOMBSTONE.value, {})
            .get("artifacts", {})
        )
        deadline_text = (
            tombstone_artifacts.get("rollback_deadline")
            if isinstance(tombstone_artifacts, dict)
            else None
        )
        if not isinstance(deadline_text, str) or not deadline_text:
            raise AnimaMergeError("Merge TOMBSTONE is missing rollback_deadline")
        try:
            rollback_deadline = ensure_aware(datetime.fromisoformat(deadline_text))
        except ValueError as exc:
            raise AnimaMergeError(f"Invalid TOMBSTONE rollback_deadline: {deadline_text}") from exc
        rollback_ready = now_local() >= rollback_deadline
        if enforce_rollback_window and not rollback_ready:
            raise AnimaMergeError(
                f"Rollback window is still active until {rollback_deadline.isoformat()}"
            )
        source_exists = self.source_dir.is_dir()
        if source_exists:
            status = _read_json(self.source_dir / "status.json")
            if status.get("enabled") is not False:
                raise AnimaMergeError("merge-finalize requires source status.json.enabled=false")
        elif not resume and not (journal and journal.is_completed(FinalizePhase.ARCHIVE_SOURCE)):
            raise AnimaMergeError(f"Source tombstone directory does not exist: {self.source_dir}")
        if not (self.animas_dir / self.target).is_dir():
            raise AnimaMergeError(f"Target Anima directory does not exist: animas/{self.target}")
        return {
            "merge_journal": str(self.merge_journal_path),
            "merge_status": str(merge["status"]),
            "source_tombstoned": True,
            "source_directory_present": source_exists,
            "rollback_deadline": rollback_deadline.isoformat(),
            "rollback_ready": rollback_ready,
        }

    def _archive_path(self, journal: MergeJournal) -> Path:
        created = str(journal.data.get("created_at", ""))
        timestamp = re.sub(r"[^0-9]", "", created)[:20] or "unknown"
        return self.data_dir / "archive" / "merged" / f"{self.source}_{timestamp}"

    def _archive_source(self, archive_path: Path) -> dict[str, Any]:
        source_exists = self.source_dir.is_dir()
        archive_exists = archive_path.is_dir()
        if source_exists and archive_exists:
            raise AnimaMergeError(f"Source and archive both exist: {self.source_dir}, {archive_path}")
        if source_exists:
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(self.source_dir), str(archive_path))
        elif not archive_exists:
            raise AnimaMergeError(f"Source and archive are both missing: {self.source_dir}, {archive_path}")
        return {"archive_path": str(archive_path), "moved": source_exists}

    def _remove_source_config(self) -> dict[str, Any]:
        config_path = self.data_dir / "config.json"
        if not config_path.is_file():
            return {"config_path": str(config_path), "entry_removed": False}
        config = _read_json(config_path)
        animas = config.get("animas")
        removed = isinstance(animas, dict) and self.source in animas
        if removed:
            del animas[self.source]
            atomic_write_text(
                config_path,
                json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
        return {"config_path": str(config_path), "entry_removed": removed}

    def _source_backend(self) -> str:
        merge = self._load_merge_journal()
        backend = (
            merge.get("phases", {})
            .get(MergePhase.PREFLIGHT.value, {})
            .get("artifacts", {})
            .get("memory_backend", {})
        )
        return str(backend.get("source", "legacy")) if isinstance(backend, dict) else "legacy"

    def _purge_neo4j(self, archive_path: Path) -> dict[str, Any]:
        if self._source_backend() != "neo4j":
            return {"status": "skipped_not_configured", "group_id": self.source}

        async def reset() -> None:
            from core.memory.backend.registry import get_backend

            backend = get_backend("neo4j", archive_path, group_id=self.source)
            try:
                await backend.reset()
            finally:
                await backend.close()

        asyncio.run(reset())
        return {"status": "purged", "group_id": self.source}

    def _purge_residuals(self, archive_path: Path) -> dict[str, Any]:
        return {
            "chroma_collections": self._purge_chroma(archive_path),
            "run_paths": self._purge_run_paths(),
            "credentials": self._purge_credentials(),
            "routing_entries_removed": self._purge_routing_maps(),
        }

    def _purge_chroma(self, archive_path: Path) -> list[str]:
        vectordb = archive_path / "vectordb"
        if not vectordb.is_dir():
            return []
        from core.memory.rag.store import create_chroma_vector_store

        store = create_chroma_vector_store(persist_dir=vectordb, anima_name=self.source)
        removed: list[str] = []
        try:
            for name in store.list_collections():
                if name.startswith(f"{self.source}_") and store.delete_collection(name):
                    removed.append(name)
        finally:
            close = getattr(store, "close", None)
            if callable(close):
                close()
        return sorted(removed)

    def _purge_run_paths(self) -> list[str]:
        paths = [
            self.data_dir / "run" / "inbox_wake" / self.source,
            self.data_dir / "run" / "events" / self.source,
            self.data_dir / "run" / "sockets" / f"{self.source}.sock",
            self.data_dir / "run" / "animas" / f"{self.source}.pid",
            self.data_dir / "run" / "animas" / f"{self.source}.lock",
        ]
        removed: list[str] = []
        for path in paths:
            if not path.exists() and not path.is_symlink():
                continue
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
            removed.append(path.relative_to(self.data_dir).as_posix())
        return removed

    def _credential_candidates(self) -> list[dict[str, str]]:
        merge = self._load_merge_journal()
        candidates = (
            merge.get("phases", {})
            .get(MergePhase.REWRITE_REFS.value, {})
            .get("substeps", {})
            .get("messaging", {})
            .get("artifacts", {})
            .get("credential_disable_candidates", [])
        )
        if not isinstance(candidates, list):
            return []
        return [item for item in candidates if isinstance(item, dict)]

    def _purge_credentials(self) -> list[dict[str, str]]:
        removed: list[dict[str, str]] = []
        for item in self._credential_candidates():
            storage = str(item.get("storage", ""))
            key = str(item.get("key", ""))
            if not key:
                continue
            changed = False
            if storage == "shared/credentials.json":
                changed = self._remove_json_key(self.data_dir / storage, key)
            elif storage == "vault.json":
                changed = self._remove_json_key(self.data_dir / storage, key, nested="shared")
            elif storage == ".env":
                changed = self._remove_env_key(self.data_dir / ".env", key)
            elif storage == "environment":
                changed = key in os.environ
                os.environ.pop(key, None)
            if changed:
                removed.append({"storage": storage, "key": key})
        return removed

    @staticmethod
    def _remove_json_key(path: Path, key: str, *, nested: str | None = None) -> bool:
        if not path.is_file():
            return False
        value = _read_json(path)
        container = value.get(nested) if nested else value
        if not isinstance(container, dict) or key not in container:
            return False
        del container[key]
        atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        return True

    @staticmethod
    def _remove_env_key(path: Path, key: str) -> bool:
        if not path.is_file():
            return False
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        pattern = re.compile(rf"^\s*(?:export\s+)?{re.escape(key)}\s*=")
        retained = [line for line in lines if not pattern.match(line)]
        if len(retained) == len(lines):
            return False
        atomic_write_text(path, "".join(retained))
        return True

    def _purge_routing_maps(self) -> int:
        removed = 0
        for path in (
            self.data_dir / "run" / "notification_map.json",
            self.data_dir / "run" / "discord_thread_map.json",
        ):
            if not path.is_file():
                continue
            value = _read_json(path)
            rewritten, count = self._purge_routing_value(value)
            if count:
                atomic_write_text(path, json.dumps(rewritten, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
                removed += count
        return removed

    def _purge_routing_value(self, value: Any) -> tuple[Any, int]:
        if isinstance(value, list):
            result: list[Any] = []
            changed = 0
            for item in value:
                if item == self.source:
                    changed += 1
                    continue
                rewritten, count = self._purge_routing_value(item)
                result.append(rewritten)
                changed += count
            return result, changed
        if not isinstance(value, dict):
            return value, 0
        result: dict[str, Any] = {}
        changed = 0
        for key, item in value.items():
            if item == self.source or (isinstance(item, dict) and item.get("anima") == self.source):
                changed += 1
                continue
            rewritten, count = self._purge_routing_value(item)
            result[key] = rewritten
            changed += count
        return result, changed

    def _verify_removal(self, archive_path: Path) -> dict[str, Any]:
        from core.org_sync import sync_org_structure

        config_path = self.data_dir / "config.json"
        discovered = sync_org_structure(self.animas_dir, config_path=config_path)
        if self.source_dir.exists():
            raise AnimaMergeError(f"Source directory reappeared after org sync: {self.source_dir}")
        config = _read_json(config_path) if config_path.is_file() else {}
        animas = config.get("animas", {})
        if isinstance(animas, dict) and self.source in animas:
            raise AnimaMergeError("Source config entry reappeared after org sync")
        if not archive_path.is_dir():
            raise AnimaMergeError(f"Source archive is missing: {archive_path}")
        return {
            "source_directory_absent": True,
            "source_config_absent": True,
            "archive_present": True,
            "org_sync_discovered": sorted(discovered),
        }


__all__ = ["AnimaMergeFinalizeService", "FinalizeResult"]
