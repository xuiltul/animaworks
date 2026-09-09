"""Read-only boundary between legacy task files and canonical execution."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from core.i18n import t
from core.taskboard.tasks import task_database_path


def require_task_store_ready(anima_dir: Path) -> None:
    """Refuse implicit migration of populated legacy state on a runtime read.

    This preflight never creates a database or import marker. Explicit offline
    migration owns the transition; an empty/new runtime can initialize safely.
    """
    database = task_database_path(anima_dir)
    if database.is_file():
        with sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True) as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='task_imports'").fetchone():
                if db.execute("SELECT 1 FROM task_imports WHERE anima=?", (anima_dir.name,)).fetchone():
                    return
    legacy_files = [anima_dir / "state" / name for name in ("task_queue.jsonl", "task_queue_archive.jsonl")]
    has_legacy = any(path.is_file() and bool(path.read_text(encoding="utf-8").strip()) for path in legacy_files)
    if not has_legacy:
        has_legacy = any(
            path.is_file() and path.stat().st_size for path in (anima_dir / "state" / "pending").rglob("*.json")
        )
    if has_legacy:
        raise RuntimeError(t("task_store.migration_required", name=anima_dir.name))
