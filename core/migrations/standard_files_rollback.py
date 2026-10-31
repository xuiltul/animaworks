# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Rollback support for the standard workspace-file migration (A2)."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.migrations.steps import STANDARD_WORKSPACE_FILES_STEP_ID


@dataclass
class RollbackResult:
    """Files touched by a standard workspace-file rollback."""

    dry_run: bool
    animas: list[str] = field(default_factory=list)
    moved_new_files: list[str] = field(default_factory=list)
    restored_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tracker_entry_removed: bool = False
    details: list[str] = field(default_factory=list)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_base(anima_dir: Path) -> Path:
    return anima_dir / "archive" / "standard_files_migration"


def _latest_archive(anima_dir: Path) -> Path | None:
    base = _archive_base(anima_dir)
    if not base.is_dir():
        return None
    candidates = [
        child
        for child in sorted(base.iterdir(), reverse=True)
        if child.is_dir() and (child / "manifest.json").is_file()
    ]
    return candidates[0] if candidates else None


def _load_manifest(archive: Path) -> dict[str, Any]:
    return json.loads((archive / "manifest.json").read_text(encoding="utf-8"))


def _rollback_one(anima_dir: Path, archive: Path, result: RollbackResult) -> None:
    manifest = _load_manifest(archive)
    new_files = [str(name) for name in manifest.get("new_files", [])]
    new_sha256 = manifest.get("new_sha256", {})
    rolled_back_dir = archive / "rolled_back_new"

    # Move the files created by the migration away first. This matters on a
    # case-insensitive filesystem: IDENTITY.md must disappear before
    # identity.md can be restored as a distinct name.
    for name in new_files:
        source = anima_dir / name
        if not source.is_file():
            result.warnings.append(f"{anima_dir.name}: missing migrated file {name}; skipped")
            continue
        recorded = new_sha256.get(name) if isinstance(new_sha256, dict) else None
        if recorded and _sha256(source) != recorded:
            result.warnings.append(
                f"{anima_dir.name}: {name} was edited after migration; moved to {rolled_back_dir / name}"
            )
        destination = rolled_back_dir / name
        if not result.dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
        result.moved_new_files.append(str(destination))
        result.details.append(f"{anima_dir.name}: moved {name} to rolled_back_new/{name}")

    for old_name in manifest.get("old_files", []):
        old_name = str(old_name)
        source = archive / old_name
        if not source.is_file():
            result.warnings.append(f"{anima_dir.name}: archived original {old_name} is missing; skipped")
            continue
        if not result.dry_run:
            shutil.copy2(source, anima_dir / old_name)
        result.restored_files.append(old_name)
        result.details.append(f"{anima_dir.name}: restored {old_name}")

    for entry in manifest.get("rewritten", []):
        rel = str(entry.get("path", ""))
        if not rel or rel in new_files:
            # Newly created standard files are removed/moved above; do not
            # recreate them during the rollback.
            continue
        source = archive / "rewritten" / rel
        if not source.is_file():
            result.warnings.append(f"{anima_dir.name}: archived rewrite source {rel} is missing; skipped")
            continue
        if not result.dry_run:
            destination = anima_dir / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        result.restored_files.append(rel)
        result.details.append(f"{anima_dir.name}: restored rewritten {rel}")


def rollback_standard_files(
    data_dir: Path,
    *,
    anima_name: str | None = None,
    dry_run: bool = False,
) -> RollbackResult:
    """Restore the latest standard workspace-file archive for each anima.

    The archive manifest is the source of truth. New standard files are moved
    to ``archive/standard_files_migration/<timestamp>/rolled_back_new/`` and
    legacy files plus pre-migration prose are copied back byte-for-byte.
    """
    result = RollbackResult(dry_run=dry_run)
    animas_dir = data_dir / "animas"
    if anima_name:
        candidates = [animas_dir / anima_name]
    elif animas_dir.is_dir():
        candidates = [child for child in sorted(animas_dir.iterdir()) if child.is_dir()]
    else:
        candidates = []

    for anima_dir in candidates:
        if not anima_dir.is_dir():
            result.warnings.append(f"Anima directory not found: {anima_dir}")
            continue
        archive = _latest_archive(anima_dir)
        if archive is None:
            continue
        try:
            result.animas.append(anima_dir.name)
            _rollback_one(anima_dir, archive, result)
        except Exception as exc:
            result.warnings.append(f"{anima_dir.name}: rollback failed: {exc}")

    if not dry_run and result.animas:
        from core.migrations.tracker import MigrationTracker

        tracker = MigrationTracker(data_dir)
        result.tracker_entry_removed = tracker.unmark_applied(STANDARD_WORKSPACE_FILES_STEP_ID)

    return result


__all__ = ["RollbackResult", "rollback_standard_files"]
