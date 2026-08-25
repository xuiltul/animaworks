"""A read-only cache dir (sandboxed anima, write-access charter) must still
serve reads: the cron collectors keep the DB fresh outside the sandbox, so
failing to open it turns fresh data into a fake 'nothing found'."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from core.tools._cache import BaseMessageCache

SCHEMA = "CREATE TABLE IF NOT EXISTS msgs (id TEXT PRIMARY KEY, body TEXT);"


def _seed(db_path: Path) -> None:
    cache = BaseMessageCache(db_path, SCHEMA)
    cache.conn.execute("INSERT INTO msgs VALUES ('1', 'hello')")
    cache.conn.commit()
    cache.close()


def test_writable_dir_uses_delete_journal(tmp_path: Path) -> None:
    # WAL would need a -shm file, which a read-only mount forbids.
    db = tmp_path / "messages.db"
    cache = BaseMessageCache(db, SCHEMA)
    assert cache.conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    assert cache.readonly is False
    cache.close()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_readonly_dir_falls_back_to_reads(tmp_path: Path) -> None:
    db = tmp_path / "store" / "messages.db"
    _seed(db)
    db.chmod(0o444)  # a sandbox mounts the file read-only, not just its dir
    db.parent.chmod(0o555)
    try:
        cache = BaseMessageCache(db, SCHEMA)
        assert cache.conn.execute("SELECT body FROM msgs").fetchone()[0] == "hello"
        with pytest.raises(sqlite3.OperationalError):  # writes fail loudly
            cache.conn.execute("INSERT INTO msgs VALUES ('2', 'nope')")
        cache.close()
    finally:
        db.parent.chmod(0o755)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_readonly_dir_without_db_still_raises(tmp_path: Path) -> None:
    # No cache to fall back on: stay loud rather than pretend to be empty.
    (tmp_path / "store").mkdir()
    (tmp_path / "store").chmod(0o555)
    try:
        with pytest.raises((OSError, sqlite3.OperationalError)):
            BaseMessageCache(tmp_path / "store" / "messages.db", SCHEMA)
    finally:
        (tmp_path / "store").chmod(0o755)
