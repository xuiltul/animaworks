from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Durable operation journal for a resumable Anima merge."""

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from core.memory._io import atomic_write_text
from core.time_utils import now_iso


class MergePhase(StrEnum):
    PREFLIGHT = "PREFLIGHT"
    QUIESCE = "QUIESCE"
    SNAPSHOT = "SNAPSHOT"
    MERGE_MEMORY = "MERGE_MEMORY"
    REBUILD_INDEXES = "REBUILD_INDEXES"
    REWRITE_REFS = "REWRITE_REFS"
    VERIFY = "VERIFY"
    TOMBSTONE = "TOMBSTONE"
    DONE = "DONE"


class FinalizePhase(StrEnum):
    PREFLIGHT = "PREFLIGHT"
    ARCHIVE_SOURCE = "ARCHIVE_SOURCE"
    REMOVE_CONFIG = "REMOVE_CONFIG"
    PURGE_NEO4J = "PURGE_NEO4J"
    PURGE_RESIDUALS = "PURGE_RESIDUALS"
    VERIFY_REMOVAL = "VERIFY_REMOVAL"
    DONE = "DONE"


class MergeJournal:
    """JSON journal whose phase records and artifacts survive interruption."""

    VERSION = 1

    def __init__(self, path: Path, source: str, target: str, *, resume: bool = False) -> None:
        self.path = path
        self.source = source
        self.target = target
        if path.exists():
            self.data = self._load()
            if self.data.get("source") != source or self.data.get("target") != target:
                raise ValueError("Merge journal belongs to a different source/target pair")
            if not resume and self.data.get("status") != "done":
                raise ValueError(f"Incomplete merge journal already exists; rerun with --resume: {path}")
        else:
            if resume:
                raise ValueError(f"No merge journal exists to resume: {path}")
            created = now_iso()
            self.data: dict[str, Any] = {
                "version": self.VERSION,
                "source": source,
                "target": target,
                "status": "running",
                "created_at": created,
                "updated_at": created,
                "phases": {},
                "artifacts": {},
            }
            self._save()

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid merge journal: {self.path}: {exc}") from exc
        if not isinstance(data, dict) or data.get("version") != self.VERSION:
            raise ValueError(f"Unsupported merge journal format: {self.path}")
        return data

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_at"] = now_iso()
        atomic_write_text(
            self.path,
            json.dumps(self.data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

    def is_completed(self, phase: StrEnum) -> bool:
        return self.data.get("phases", {}).get(phase.value, {}).get("status") in {"completed", "skipped"}

    def start(self, phase: StrEnum) -> None:
        record = self.data.setdefault("phases", {}).setdefault(phase.value, {})
        record.update({"status": "started", "started_at": now_iso()})
        record.pop("completed_at", None)
        record.pop("error", None)
        self.data["status"] = "running"
        self._save()

    def complete(self, phase: StrEnum, artifacts: dict[str, Any] | None = None) -> None:
        record = self.data.setdefault("phases", {}).setdefault(phase.value, {})
        record.update({"status": "completed", "completed_at": now_iso()})
        if artifacts:
            record["artifacts"] = artifacts
            self.data.setdefault("artifacts", {}).update(artifacts)
        self._save()

    def skip(self, phase: StrEnum, reason: str) -> None:
        self.data.setdefault("phases", {})[phase.value] = {
            "status": "skipped",
            "completed_at": now_iso(),
            "reason": reason,
        }
        self._save()

    def fail(self, phase: StrEnum, error: str) -> None:
        record = self.data.setdefault("phases", {}).setdefault(phase.value, {})
        record.update({"status": "failed", "failed_at": now_iso(), "error": error})
        self.data["status"] = "failed"
        self._save()

    def finish(self) -> None:
        self.data["status"] = "done"
        self.data["completed_at"] = now_iso()
        self._save()

    def phase_artifacts(self, phase: StrEnum) -> dict[str, Any]:
        value = self.data.get("phases", {}).get(phase.value, {}).get("artifacts", {})
        return value if isinstance(value, dict) else {}

    def is_substep_completed(self, phase: MergePhase, substep: str) -> bool:
        """Return whether a resumable phase substep is complete or skipped."""
        status = self.data.get("phases", {}).get(phase.value, {}).get("substeps", {}).get(substep, {}).get("status")
        return status in {"completed", "skipped"}

    def start_substep(self, phase: MergePhase, substep: str) -> None:
        record = (
            self.data.setdefault("phases", {})
            .setdefault(phase.value, {})
            .setdefault("substeps", {})
            .setdefault(substep, {})
        )
        record.update({"status": "started", "started_at": now_iso()})
        record.pop("completed_at", None)
        record.pop("failed_at", None)
        record.pop("error", None)
        self._save()

    def complete_substep(
        self,
        phase: MergePhase,
        substep: str,
        artifacts: dict[str, Any] | None = None,
    ) -> None:
        record = (
            self.data.setdefault("phases", {})
            .setdefault(phase.value, {})
            .setdefault("substeps", {})
            .setdefault(substep, {})
        )
        record.update({"status": "completed", "completed_at": now_iso()})
        if artifacts is not None:
            record["artifacts"] = artifacts
        self._save()

    def skip_substep(self, phase: MergePhase, substep: str, reason: str) -> None:
        self.data.setdefault("phases", {}).setdefault(phase.value, {}).setdefault("substeps", {})[substep] = {
            "status": "skipped",
            "completed_at": now_iso(),
            "reason": reason,
        }
        self._save()

    def fail_substep(self, phase: MergePhase, substep: str, error: str) -> None:
        record = (
            self.data.setdefault("phases", {})
            .setdefault(phase.value, {})
            .setdefault("substeps", {})
            .setdefault(substep, {})
        )
        record.update({"status": "failed", "failed_at": now_iso(), "error": error})
        self._save()

    def substep_artifacts(self, phase: MergePhase, substep: str) -> dict[str, Any]:
        value = (
            self.data.get("phases", {}).get(phase.value, {}).get("substeps", {}).get(substep, {}).get("artifacts", {})
        )
        return value if isinstance(value, dict) else {}
