from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Filesystem transaction helpers for Skill Hub imports."""

import shutil
from pathlib import Path
from typing import Protocol

from core.time_utils import now_iso


class HubPaths(Protocol):
    backup_dir: Path
    rel_base: Path

    def rel(self, path: Path) -> str: ...


def pending_destination(paths: HubPaths, skill_name: str) -> Path:
    return paths.backup_dir / "pending" / f"{skill_name}.install-{_stamp()}"


def removed_destination(paths: HubPaths, skill_name: str) -> Path:
    return paths.backup_dir / "removed" / f"{skill_name}.remove-{_stamp()}"


def activate_destination(prepared: Path, destination: Path, *, replace: bool, paths: HubPaths) -> str | None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.move(str(prepared), str(destination))
        return None
    if not replace:
        shutil.rmtree(prepared, ignore_errors=True)
        raise FileExistsError(f"Skill already exists: {destination}")

    backup = paths.backup_dir / f"{destination.name}.backup-{_stamp()}"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(destination), str(backup))
    try:
        shutil.move(str(prepared), str(destination))
    except Exception:
        if not destination.exists() and backup.exists():
            shutil.move(str(backup), str(destination))
        raise
    return paths.rel(backup / "SKILL.md")


def rollback_activation(destination: Path, backup_path: str | None, paths: HubPaths) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    if backup_path:
        backup = paths.rel_base / backup_path
        restore = backup.parent if backup.name == "SKILL.md" else backup
        if restore.exists():
            shutil.move(str(restore), str(destination))


def _stamp() -> str:
    return now_iso().replace(":", "").replace("+", "_").replace("-", "").replace(".", "")
