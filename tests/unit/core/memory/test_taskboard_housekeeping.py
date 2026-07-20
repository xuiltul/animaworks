from __future__ import annotations

import json
import os
import time
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from core.memory.task_queue import TaskQueueManager
from core.memory.taskboard_housekeeping import (
    _archive_current_state_for_housekeeping,
    _cleanup_current_state,
    cleanup_taskboard_stale_artifacts,
)
from core.platform.processing_lease import processing_lease_path, write_processing_lease
from core.taskboard.store import TaskBoardStore
from core.time_utils import now_local


def _anima_dir(data_dir: Path, name: str = "sakura") -> Path:
    anima_dir = data_dir / "animas" / name
    (anima_dir / "state").mkdir(parents=True, exist_ok=True)
    (anima_dir / "episodes").mkdir(parents=True, exist_ok=True)
    return anima_dir


def _write_json(path: Path, payload: dict[str, object], *, age_hours: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    if age_hours:
        old_ts = time.time() - (age_hours * 3600)
        os.utime(path, (old_ts, old_ts))
    return path


def test_stale_processing_moves_to_failed_syncs_queue_and_appends_event(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    anima_dir = _anima_dir(data_dir)
    queue = TaskQueueManager(anima_dir)
    queue.add_task(
        source="human",
        original_instruction="recover this task",
        assignee="sakura",
        summary="recover this task",
        status="in_progress",
        task_id="task-processing",
    )
    processing = _write_json(
        anima_dir / "state" / "pending" / "processing" / "task-processing.json",
        {"task_id": "task-processing"},
        age_hours=25,
    )

    result = cleanup_taskboard_stale_artifacts(data_dir, 24, 48, 24, 30)

    assert result["processing_recovered"] == 1
    assert result["processing_queue_synced"] == 1
    assert not processing.exists()
    assert (anima_dir / "state" / "pending" / "failed" / "task-processing.json").exists()

    task = queue.get_task_by_id("task-processing")
    assert task is not None
    assert task.status == "failed"
    assert task.summary == "FAILED: stale processing task recovered by housekeeping"

    events = TaskBoardStore(data_dir / "shared" / "taskboard.sqlite3").list_events(
        anima_name="sakura",
        task_id="task-processing",
    )
    assert events[-1]["event_type"] == "stale_processing_recovered"
    assert events[-1]["payload"]["queue_synced"] is True


def test_stale_processing_with_live_lease_is_skipped(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    anima_dir = _anima_dir(data_dir)
    queue = TaskQueueManager(anima_dir)
    queue.add_task(
        source="human",
        original_instruction="keep running",
        assignee="sakura",
        summary="live task",
        status="in_progress",
        task_id="live-processing",
    )
    processing = _write_json(
        anima_dir / "state" / "pending" / "processing" / "live-processing.json",
        {"task_id": "live-processing"},
        age_hours=25,
    )
    lease = write_processing_lease(processing, anima="sakura", task_id="live-processing")

    with patch("core.memory.taskboard_housekeeping.is_processing_lease_live", return_value=True):
        result = cleanup_taskboard_stale_artifacts(data_dir, 24, 48, 24, 30)

    assert result["processing_recovered"] == 0
    assert result["processing_live_leases_skipped"] == 1
    assert processing.exists()
    assert lease.exists()
    assert queue.get_task_by_id("live-processing").status == "in_progress"


def test_stale_processing_with_dead_lease_is_recovered(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    anima_dir = _anima_dir(data_dir)
    queue = TaskQueueManager(anima_dir)
    queue.add_task(
        source="human",
        original_instruction="recover dead lease",
        assignee="sakura",
        summary="dead task",
        status="in_progress",
        task_id="dead-processing",
    )
    processing = _write_json(
        anima_dir / "state" / "pending" / "processing" / "dead-processing.json",
        {"task_id": "dead-processing"},
        age_hours=25,
    )
    write_processing_lease(processing, anima="sakura", task_id="dead-processing", pid=999_999_999)

    with patch("core.memory.taskboard_housekeeping.is_processing_lease_live", return_value=False):
        result = cleanup_taskboard_stale_artifacts(data_dir, 24, 48, 24, 30)

    failed = anima_dir / "state" / "pending" / "failed" / "dead-processing.json"
    assert result["processing_recovered"] == 1
    assert not processing.exists()
    assert failed.exists()
    assert not processing_lease_path(processing).exists()
    assert processing_lease_path(failed).exists()
    assert queue.get_task_by_id("dead-processing").status == "failed"


def test_unreadable_processing_file_is_moved_without_queue_sync(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    anima_dir = _anima_dir(data_dir)
    path = anima_dir / "state" / "pending" / "processing" / "bad.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{bad json", encoding="utf-8")
    old_ts = time.time() - (25 * 3600)
    os.utime(path, (old_ts, old_ts))

    result = cleanup_taskboard_stale_artifacts(data_dir, 24, 48, 24, 30)

    assert result["processing_recovered"] == 1
    assert result["processing_unreadable"] == 1
    assert result["processing_queue_synced"] == 0
    assert not path.exists()
    assert (anima_dir / "state" / "pending" / "failed" / "bad.json").exists()


def test_deferred_wakes_elapsed_snooze_and_fails_stale_unsnoozed_file(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    anima_dir = _anima_dir(data_dir)
    elapsed = (now_local() - timedelta(minutes=1)).isoformat()
    _write_json(
        anima_dir / "state" / "pending" / "deferred" / "elapsed.json",
        {"task_id": "elapsed", "snoozed_until": elapsed},
    )
    _write_json(
        anima_dir / "state" / "pending" / "deferred" / "stale.json",
        {"task_id": "stale"},
        age_hours=25,
    )

    result = cleanup_taskboard_stale_artifacts(data_dir, 24, 48, 24, 30)

    assert result["deferred_woken"] == 1
    assert result["deferred_failed"] == 1
    assert (anima_dir / "state" / "pending" / "elapsed.json").exists()
    assert (anima_dir / "state" / "pending" / "failed" / "stale.json").exists()


def test_suppressed_retention_and_background_running_cleanup(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    anima_dir = _anima_dir(data_dir)
    old_suppressed = _write_json(
        anima_dir / "state" / "pending" / "suppressed" / "old.json",
        {"task_id": "old"},
        age_hours=31 * 24,
    )
    recent_suppressed = _write_json(
        anima_dir / "state" / "pending" / "suppressed" / "recent.json",
        {"task_id": "recent"},
    )
    old_running = _write_json(
        anima_dir / "state" / "background_tasks" / "running.json",
        {"task_id": "running", "status": "running", "created_at": time.time() - (49 * 3600)},
    )
    missing_created_running = _write_json(
        anima_dir / "state" / "background_tasks" / "missing-created.json",
        {"task_id": "missing-created", "status": "running"},
        age_hours=100,
    )
    completed = _write_json(
        anima_dir / "state" / "background_tasks" / "completed.json",
        {"task_id": "completed", "status": "completed", "created_at": 1, "completed_at": 1},
        age_hours=100,
    )

    result = cleanup_taskboard_stale_artifacts(data_dir, 24, 48, 24, 30)

    assert result["suppressed_deleted"] == 1
    assert result["background_running_deleted"] == 1
    assert not old_suppressed.exists()
    assert recent_suppressed.exists()
    assert not old_running.exists()
    assert missing_created_running.exists()
    assert completed.exists()


def test_stale_current_state_archives_and_resets_when_no_active_visible_task(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    anima_dir = _anima_dir(data_dir)
    state_path = anima_dir / "state" / "current_state.md"
    state_path.write_text("status: working\nold notes", encoding="utf-8")
    old_ts = time.time() - (25 * 3600)
    os.utime(state_path, (old_ts, old_ts))

    result = _cleanup_current_state(data_dir / "animas", 24, TaskBoardStore(data_dir / "shared" / "taskboard.sqlite3"))

    assert result["archived"] == 1
    assert state_path.read_text(encoding="utf-8") == "status: idle\n"
    episode = next((anima_dir / "episodes").glob("*.md"))
    assert "## Working notes archived by TaskBoard housekeeping" in episode.read_text(encoding="utf-8")
    assert "old notes" in episode.read_text(encoding="utf-8")


def test_current_state_is_kept_when_active_visible_task_exists(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    anima_dir = _anima_dir(data_dir)
    TaskQueueManager(anima_dir).add_task(
        source="human",
        original_instruction="active task",
        assignee="sakura",
        summary="active task",
        task_id="active-task",
    )
    state_path = anima_dir / "state" / "current_state.md"
    state_path.write_text("status: working\nactive notes", encoding="utf-8")
    old_ts = time.time() - (25 * 3600)
    os.utime(state_path, (old_ts, old_ts))

    result = _cleanup_current_state(data_dir / "animas", 24, TaskBoardStore(data_dir / "shared" / "taskboard.sqlite3"))

    assert result["archived"] == 0
    assert result["active_visible"] == 1
    assert state_path.read_text(encoding="utf-8") == "status: working\nactive notes"


def test_current_state_archive_failure_leaves_original_unchanged(
    monkeypatch,
    tmp_path: Path,
) -> None:
    anima_dir = _anima_dir(tmp_path / "data")
    state_path = anima_dir / "state" / "current_state.md"
    state_path.write_text("status: working\nkeep me", encoding="utf-8")

    def fail_atomic_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("core.memory._io.atomic_write_text", fail_atomic_write)

    assert _archive_current_state_for_housekeeping(anima_dir, state_path, "status: working\nkeep me") == "error"
    assert state_path.read_text(encoding="utf-8") == "status: working\nkeep me"


def test_current_state_archive_skips_if_file_changed_after_read(tmp_path: Path) -> None:
    anima_dir = _anima_dir(tmp_path / "data")
    state_path = anima_dir / "state" / "current_state.md"
    state_path.write_text("status: working\nold", encoding="utf-8")
    old_ts = time.time() - 10
    os.utime(state_path, (old_ts, old_ts))
    expected_mtime = state_path.stat().st_mtime
    state_path.write_text("status: working\nfresh", encoding="utf-8")

    outcome = _archive_current_state_for_housekeeping(
        anima_dir,
        state_path,
        "status: working\nold",
        expected_mtime=expected_mtime,
    )

    assert outcome == "changed"
    assert state_path.read_text(encoding="utf-8") == "status: working\nfresh"
    assert not list((anima_dir / "episodes").glob("*.md"))


def _stale_iso(hours: float) -> str:
    return (now_local() - timedelta(hours=hours)).isoformat()


def test_orphan_metadata_without_queue_entry_is_archived(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _anima_dir(data_dir)
    store = TaskBoardStore(data_dir / "shared" / "taskboard.sqlite3")
    store.upsert_metadata(
        anima_name="sakura",
        task_id="ghost-task",
        actor="test",
        visibility="active",
        column="todo",
        updated_at=_stale_iso(25),
    )

    result = cleanup_taskboard_stale_artifacts(data_dir, 24, 48, 24, 30, 24)

    assert result["orphan_archived"] == 1
    meta = store.get_metadata("sakura", "ghost-task")
    assert meta is not None
    assert meta.visibility.value == "archived"
    assert meta.column.value == "done"
    assert meta.tombstone_reason == "queue_missing_reconciled"
    assert meta.updated_by == "housekeeping"
    events = store.list_events(anima_name="sakura", task_id="ghost-task")
    assert any(event["event_type"] == "archived" for event in events)


def test_orphan_metadata_with_queue_entry_is_left_alone(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    anima_dir = _anima_dir(data_dir)
    TaskQueueManager(anima_dir).add_task(
        source="human",
        original_instruction="still live",
        assignee="sakura",
        summary="still live",
        task_id="live-task",
    )
    store = TaskBoardStore(data_dir / "shared" / "taskboard.sqlite3")
    store.upsert_metadata(
        anima_name="sakura",
        task_id="live-task",
        actor="test",
        visibility="active",
        column="todo",
        updated_at=_stale_iso(25),
    )

    result = cleanup_taskboard_stale_artifacts(data_dir, 24, 48, 24, 30, 24)

    assert result["orphan_archived"] == 0
    meta = store.get_metadata("sakura", "live-task")
    assert meta is not None
    assert meta.visibility.value == "active"
    assert meta.column.value == "todo"


def test_recent_orphan_metadata_is_not_archived_within_grace(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _anima_dir(data_dir)
    store = TaskBoardStore(data_dir / "shared" / "taskboard.sqlite3")
    store.upsert_metadata(
        anima_name="sakura",
        task_id="fresh-ghost",
        actor="test",
        visibility="active",
        column="todo",
        updated_at=_stale_iso(1),
    )

    result = cleanup_taskboard_stale_artifacts(data_dir, 24, 48, 24, 30, 24)

    assert result["orphan_archived"] == 0
    meta = store.get_metadata("sakura", "fresh-ghost")
    assert meta is not None
    assert meta.visibility.value == "active"


def test_orphan_metadata_for_missing_anima_dir_is_archived(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    # Ensure animas/ exists so cleanup runs, but the named anima dir is absent.
    (data_dir / "animas").mkdir(parents=True, exist_ok=True)
    store = TaskBoardStore(data_dir / "shared" / "taskboard.sqlite3")
    store.upsert_metadata(
        anima_name="vanished",
        task_id="orphan-after-merge",
        actor="test",
        visibility="snoozed",
        column="waiting",
        updated_at=_stale_iso(48),
    )

    result = cleanup_taskboard_stale_artifacts(data_dir, 24, 48, 24, 30, 24)

    assert result["orphan_archived"] == 1
    meta = store.get_metadata("vanished", "orphan-after-merge")
    assert meta is not None
    assert meta.visibility.value == "archived"
    assert meta.tombstone_reason == "queue_missing_reconciled"


def test_stale_archived_metadata_is_purged_after_retention(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _anima_dir(data_dir)
    store = TaskBoardStore(data_dir / "shared" / "taskboard.sqlite3")
    store.upsert_metadata(
        anima_name="sakura",
        task_id="old-archived",
        actor="test",
        visibility="archived",
        column="done",
        updated_at=_stale_iso(31 * 24),
    )
    store.upsert_metadata(
        anima_name="sakura",
        task_id="recent-archived",
        actor="test",
        visibility="archived",
        column="done",
        updated_at=_stale_iso(5),
    )

    result = cleanup_taskboard_stale_artifacts(data_dir, 24, 48, 24, 30, 24)

    assert result["purged_deleted"] == 1
    assert store.get_metadata("sakura", "old-archived") is None
    assert store.get_metadata("sakura", "recent-archived") is not None
