from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from core.paths import get_taskboard_db_path
from core.taskboard.models import AttentionVisibility, BoardColumn
from core.taskboard.store import TaskBoardStore


def _open_fd_count() -> int:
    fd_dir = Path("/proc/self/fd")
    if not fd_dir.exists():
        pytest.skip("/proc/self/fd is not available on this platform")
    return len(os.listdir(fd_dir))


def _count_open_taskboard_fds(db_path: Path) -> int:
    fd_dir = Path("/proc/self/fd")
    if not fd_dir.exists():
        pytest.skip("/proc/self/fd is not available on this platform")

    targets = {
        str(db_path.resolve()),
        str(db_path.with_name(f"{db_path.name}-wal").resolve()),
        str(db_path.with_name(f"{db_path.name}-shm").resolve()),
    }
    count = 0
    for fd in fd_dir.iterdir():
        try:
            if os.readlink(fd) in targets:
                count += 1
        except OSError:
            continue
    return count


def test_store_closes_sqlite_connections(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    real_connect = sqlite3.connect
    closed_connections = 0

    class TrackingConnection(sqlite3.Connection):
        def close(self) -> None:
            nonlocal closed_connections
            closed_connections += 1
            super().close()

    def tracking_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        kwargs["factory"] = TrackingConnection
        return real_connect(*args, **kwargs)

    monkeypatch.setattr("core.taskboard.store.sqlite3.connect", tracking_connect)

    store = TaskBoardStore(tmp_path / "taskboard.sqlite3")
    store.upsert_metadata(anima_name="sakura", task_id="task-1", actor="alice")
    store.get_metadata("sakura", "task-1")
    store.list_metadata()
    store.list_events(anima_name="sakura", task_id="task-1")

    assert closed_connections == 5


def test_store_releases_wal_file_descriptors_after_operations(tmp_path: Path) -> None:
    db_path = tmp_path / "taskboard.sqlite3"
    store = TaskBoardStore(db_path)

    assert _count_open_taskboard_fds(db_path) == 0

    for index in range(50):
        task_id = f"task-{index}"
        store.upsert_metadata(anima_name="kotoha", task_id=task_id, actor="fd-test")
        store.get_metadata("kotoha", task_id)
        store.record_surface(anima_name="kotoha", task_id=task_id, actor="fd-test")
        store.list_metadata("kotoha")
        store.list_events(anima_name="kotoha", task_id=task_id)

    assert _count_open_taskboard_fds(db_path) == 0


def test_get_taskboard_db_path_uses_shared_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))

    assert get_taskboard_db_path() == tmp_path.resolve() / "shared" / "taskboard.sqlite3"


def test_store_uses_default_taskboard_db_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))

    store = TaskBoardStore()

    assert store.db_path == tmp_path.resolve() / "shared" / "taskboard.sqlite3"
    assert store.db_path.exists()


def test_store_creates_wal_database_and_schema_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "shared" / "taskboard.sqlite3"

    TaskBoardStore(db_path)
    TaskBoardStore(db_path)

    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        tables = {
            row[0]
            for row in conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """
            )
        }

    assert journal_mode == "wal"
    assert "taskboard_metadata" in tables
    assert "taskboard_events" in tables


def test_store_operations_do_not_leak_sqlite_connections(tmp_path: Path) -> None:
    store = TaskBoardStore(tmp_path / "taskboard.sqlite3")
    store.upsert_metadata(anima_name="sakura", task_id="task-1", actor="test")

    before = _open_fd_count()
    for idx in range(50):
        store.get_metadata("sakura", "task-1")
        store.list_metadata(anima_name="sakura")
        store.record_surface(anima_name="sakura", task_id=f"task-{idx}", actor="test")
        store.list_events(anima_name="sakura")
    after = _open_fd_count()

    assert after <= before + 2


def test_store_migrates_event_type_constraint_for_stale_processing_event(tmp_path: Path) -> None:
    db_path = tmp_path / "shared" / "taskboard.sqlite3"
    db_path.parent.mkdir(parents=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE taskboard_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                actor TEXT NOT NULL,
                event_type TEXT NOT NULL
                    CHECK(event_type IN ('metadata_upserted', 'surface_recorded')),
                anima_name TEXT NOT NULL,
                task_id TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}'
            );
            INSERT INTO taskboard_events (ts, actor, event_type, anima_name, task_id, payload_json)
            VALUES ('2026-05-14T00:00:00+09:00', 'test', 'surface_recorded', 'sakura', 'task-1', '{}');
            """
        )

    store = TaskBoardStore(db_path)
    store.append_event(
        event_type="stale_processing_recovered",
        anima_name="sakura",
        task_id="task-1",
        actor="housekeeping",
        payload={"queue_synced": True},
    )

    events = store.list_events(anima_name="sakura", task_id="task-1")
    assert [event["event_type"] for event in events] == [
        "surface_recorded",
        "stale_processing_recovered",
    ]


def test_upsert_and_read_metadata_appends_events(tmp_path: Path) -> None:
    store = TaskBoardStore(tmp_path / "taskboard.sqlite3")

    created = store.upsert_metadata(
        anima_name="sakura",
        task_id="task-1",
        actor="alice",
        visibility="active",
        column=BoardColumn.WAITING,
        position=20.0,
        source_ref="task_queue:sakura:task-1",
    )
    updated = store.upsert_metadata(
        anima_name="sakura",
        task_id="task-1",
        actor="bob",
        column="blocked",
    )

    assert created.visibility == AttentionVisibility.ACTIVE
    assert updated.column == BoardColumn.BLOCKED
    assert updated.position == 20.0
    assert updated.updated_by == "bob"

    read_back = store.get_metadata("sakura", "task-1")
    assert read_back == updated

    events = store.list_events(anima_name="sakura", task_id="task-1")
    assert [event["event_type"] for event in events] == ["metadata_upserted", "column_changed"]
    assert events[1]["payload"]["updates"] == {"column": "blocked"}


def test_visibility_change_uses_specific_event_type(tmp_path: Path) -> None:
    store = TaskBoardStore(tmp_path / "taskboard.sqlite3")

    store.upsert_metadata(anima_name="sakura", task_id="task-1", actor="alice")
    archived = store.upsert_metadata(
        anima_name="sakura",
        task_id="task-1",
        actor="alice",
        visibility=AttentionVisibility.ARCHIVED,
    )

    assert archived.visibility == AttentionVisibility.ARCHIVED
    assert store.list_events(anima_name="sakura", task_id="task-1")[-1]["event_type"] == "archived"


def test_invalid_visibility_and_column_are_rejected(tmp_path: Path) -> None:
    store = TaskBoardStore(tmp_path / "taskboard.sqlite3")

    with pytest.raises(ValueError):
        store.upsert_metadata(anima_name="sakura", task_id="task-1", visibility="forgotten")

    with pytest.raises(ValueError):
        store.upsert_metadata(anima_name="sakura", task_id="task-2", column="triage")


def test_record_surface_increments_count_and_records_event(tmp_path: Path) -> None:
    store = TaskBoardStore(tmp_path / "taskboard.sqlite3")

    first = store.record_surface(anima_name="sakura", task_id="task-1", actor="runtime", notification_key="n1")
    second = store.record_surface(anima_name="sakura", task_id="task-1", actor="runtime")

    assert first.surface_count == 1
    assert first.notification_key == "n1"
    assert second.surface_count == 2
    assert second.notification_key == "n1"

    events = store.list_events(anima_name="sakura", task_id="task-1")
    assert [event["event_type"] for event in events] == ["surface_recorded", "surface_recorded"]
