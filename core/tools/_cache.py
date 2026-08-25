# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Shared SQLite message cache base class for communication tools."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


# ── BaseMessageCache ───────────────────────────────────────


class BaseMessageCache:
    """SQLite-backed message cache base class.

    Provides common database lifecycle management, WAL mode, and
    ``row_factory`` setup.  Subclasses supply their own schema SQL
    and domain-specific query methods.

    Args:
        db_path: Path to the SQLite database file.  Parent directories
            are created automatically.
        schema_sql: SQL script executed once at initialisation to
            create tables and indexes.
    """

    def __init__(self, db_path: Path, schema_sql: str) -> None:
        self.db_path = db_path
        self.readonly = False
        conn: sqlite3.Connection | None = None
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_path))
            # DELETE, not WAL: sandboxed animas mount the cache dir read-only
            # (write-access charter) and a WAL database cannot even be *read*
            # there, because SQLite must create a -shm file alongside it.
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.executescript(schema_sql)
            conn.commit()
        except (OSError, sqlite3.OperationalError) as exc:
            if conn is not None:
                conn.close()
            if not db_path.exists():
                raise
            # The cron collectors keep this cache fresh outside the sandbox,
            # so read-only access still answers search/unreplied/mentions.
            # Writes fail loudly on this connection -- never silently.
            logger.warning("Cache DB %s is not writable, opening read-only: %s", db_path, exc)
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            self.readonly = True
        self.conn = conn
        self.conn.row_factory = sqlite3.Row

    # ── Lifecycle ──────────────────────────────────────────

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    # ── Shared helpers ─────────────────────────────────────

    def _fetchall_dicts(self, query: str, params: list | tuple = ()) -> list[dict]:
        """Execute *query* and return rows as plain dicts."""
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        """Return basic cache statistics.

        Subclasses may override to add domain-specific stats.
        """
        tables = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        stats: dict[str, int] = {}
        for (tbl_name,) in tables:
            if tbl_name.startswith("sqlite_"):
                continue
            count = self.conn.execute(
                f"SELECT COUNT(*) FROM [{tbl_name}]"  # noqa: S608
            ).fetchone()[0]
            stats[tbl_name] = count
        return stats
