from __future__ import annotations

from pathlib import Path

from core.memory.task_queue import TaskQueueManager
from core.schemas import TaskEntry
from core.taskboard.models import AttentionVisibility, BoardColumn, TaskBoardMetadata
from core.taskboard.projector import (
    QUEUE_STATUS_TO_COLUMN,
    _project_queue_task,
    project_all,
    project_anima,
)
from core.taskboard.store import TaskBoardStore


def _queue(tmp_path: Path, anima_name: str) -> TaskQueueManager:
    anima_dir = tmp_path / "animas" / anima_name
    (anima_dir / "state").mkdir(parents=True)
    return TaskQueueManager(anima_dir)


def test_projection_replays_latest_status_and_skips_corrupt_lines(tmp_path: Path) -> None:
    manager = _queue(tmp_path, "sakura")
    task = manager.add_task(
        source="human",
        original_instruction="implement projection",
        assignee="sakura",
        summary="projection",
        task_id="task-1",
    )
    manager.update_status(task.task_id, "in_progress", summary="projection running")
    manager.queue_path.write_text(manager.queue_path.read_text(encoding="utf-8") + "{corrupt-json\n", encoding="utf-8")

    projected = project_anima(manager.anima_dir, TaskBoardStore(tmp_path / "taskboard.sqlite3"))

    assert len(projected) == 1
    assert projected[0].task_id == "task-1"
    assert projected[0].queue_status == "in_progress"
    assert projected[0].summary == "projection running"
    assert projected[0].column == BoardColumn.RUNNING


def test_projection_status_to_column_mapping(tmp_path: Path) -> None:
    manager = _queue(tmp_path, "sakura")
    statuses = ["pending", "in_progress", "delegated", "done", "cancelled"]

    for status in statuses:
        task = manager.add_task(
            source="human",
            original_instruction=status,
            assignee="sakura",
            summary=status,
            task_id=f"task-{status}",
            status="in_progress" if status == "in_progress" else "pending",
        )
        if status not in {"pending", "in_progress"}:
            manager.update_status(task.task_id, status)

    projected = project_anima(
        manager.anima_dir,
        TaskBoardStore(tmp_path / "taskboard.sqlite3"),
        include_archived=True,
    )
    by_status = {task.queue_status: task for task in projected}

    assert {status: by_status[status].column for status in statuses} == QUEUE_STATUS_TO_COLUMN
    for terminal_status in {"done", "cancelled"}:
        assert by_status[terminal_status].visibility == AttentionVisibility.ARCHIVED


def test_metadata_column_overrides_board_only_without_mutating_queue_status(tmp_path: Path) -> None:
    manager = _queue(tmp_path, "sakura")
    task = manager.add_task(
        source="human",
        original_instruction="keep queue status",
        assignee="sakura",
        summary="keep queue status",
        task_id="task-1",
    )
    store = TaskBoardStore(tmp_path / "taskboard.sqlite3")
    store.upsert_metadata(
        anima_name="sakura",
        task_id=task.task_id,
        column="blocked",
        position=5.0,
    )

    projected = project_anima(manager.anima_dir, store)

    assert projected[0].queue_status == "pending"
    assert projected[0].column == BoardColumn.BLOCKED
    assert projected[0].source_ref == "task_queue:sakura:task-1"
    assert manager.get_task_by_id(task.task_id).status == "pending"


def test_terminal_tasks_are_archived_by_default_view(tmp_path: Path) -> None:
    manager = _queue(tmp_path, "sakura")
    task = manager.add_task(
        source="human",
        original_instruction="finish",
        assignee="sakura",
        summary="finish",
        task_id="task-1",
    )
    manager.update_status(task.task_id, "done")
    store = TaskBoardStore(tmp_path / "taskboard.sqlite3")

    assert project_anima(manager.anima_dir, store) == []

    projected = project_anima(manager.anima_dir, store, include_archived=True)
    assert projected[0].visibility == AttentionVisibility.ARCHIVED
    assert projected[0].column == BoardColumn.DONE


def test_missing_queue_metadata_is_hidden_unless_requested(tmp_path: Path) -> None:
    store = TaskBoardStore(tmp_path / "taskboard.sqlite3")
    store.upsert_metadata(
        anima_name="sakura",
        task_id="missing",
        visibility="active",
        column="review",
        source_ref="task_queue:sakura:missing",
    )
    anima_dir = tmp_path / "animas" / "sakura"
    (anima_dir / "state").mkdir(parents=True)

    assert project_anima(anima_dir, store) == []

    projected = project_anima(anima_dir, store, include_missing=True)
    assert len(projected) == 1
    assert projected[0].queue_missing is True
    assert projected[0].column == BoardColumn.REVIEW


def test_same_task_id_is_scoped_per_anima(tmp_path: Path) -> None:
    sakura = _queue(tmp_path, "sakura")
    hinata = _queue(tmp_path, "hinata")
    sakura.add_task(
        source="human",
        original_instruction="shared id",
        assignee="sakura",
        summary="sakura task",
        task_id="same",
    )
    hinata.add_task(
        source="human",
        original_instruction="shared id",
        assignee="hinata",
        summary="hinata task",
        task_id="same",
        status="in_progress",
    )

    projected = project_all(tmp_path / "animas", TaskBoardStore(tmp_path / "taskboard.sqlite3"))

    assert {(task.anima_name, task.task_id, task.column) for task in projected} == {
        ("sakura", "same", BoardColumn.TODO),
        ("hinata", "same", BoardColumn.RUNNING),
    }


def _queue_entry(
    *,
    status: str,
    updated_at: str,
    task_id: str = "task-1",
) -> TaskEntry:
    return TaskEntry(
        task_id=task_id,
        ts="2026-01-01T00:00:00+09:00",
        source="human",
        original_instruction="test",
        assignee="sakura",
        status=status,
        summary="test",
        updated_at=updated_at,
    )


def test_stale_active_metadata_ignored_when_queue_is_done() -> None:
    """Queue done/cancelled with older active metadata → default archived/done."""
    task = _queue_entry(status="done", updated_at="2026-06-01T12:00:00+09:00")
    metadata = TaskBoardMetadata(
        anima_name="sakura",
        task_id=task.task_id,
        visibility=AttentionVisibility.ACTIVE,
        column=BoardColumn.WAITING,
        updated_at="2026-05-01T12:00:00+09:00",
        updated_by="delegator",
    )

    projected = _project_queue_task(task=task, anima_name="sakura", metadata=metadata)

    assert projected.visibility == AttentionVisibility.ARCHIVED
    assert projected.column == BoardColumn.DONE
    assert projected.queue_status == "done"
    # Other metadata fields still surface.
    assert projected.board_updated_at == metadata.updated_at
    assert projected.board_updated_by == "delegator"


def test_newer_metadata_respected_when_queue_is_done() -> None:
    """Post-completion UI override (newer metadata) still wins over queue defaults."""
    task = _queue_entry(status="done", updated_at="2026-06-01T12:00:00+09:00")
    metadata = TaskBoardMetadata(
        anima_name="sakura",
        task_id=task.task_id,
        visibility=AttentionVisibility.ACTIVE,
        column=BoardColumn.REVIEW,
        updated_at="2026-06-01T13:00:00+09:00",
        updated_by="human",
    )

    projected = _project_queue_task(task=task, anima_name="sakura", metadata=metadata)

    assert projected.visibility == AttentionVisibility.ACTIVE
    assert projected.column == BoardColumn.REVIEW


def test_stale_suppressed_metadata_respected_when_queue_is_done() -> None:
    """A terminal queue update must preserve the explicit suppression reason."""
    task = _queue_entry(status="cancelled", updated_at="2026-06-01T12:00:00+09:00")
    metadata = TaskBoardMetadata(
        anima_name="sakura",
        task_id=task.task_id,
        visibility=AttentionVisibility.EXPIRED,
        column=BoardColumn.SUPPRESSED,
        updated_at="2026-05-01T12:00:00+09:00",
        updated_by="planner",
    )

    projected = _project_queue_task(task=task, anima_name="sakura", metadata=metadata)

    assert projected.visibility == AttentionVisibility.EXPIRED
    assert projected.column == BoardColumn.SUPPRESSED


def test_pending_queue_keeps_metadata_precedence() -> None:
    """Non-archived queue statuses keep traditional metadata-first visibility/column."""
    task = _queue_entry(status="pending", updated_at="2026-06-01T12:00:00+09:00")
    metadata = TaskBoardMetadata(
        anima_name="sakura",
        task_id=task.task_id,
        visibility=AttentionVisibility.SNOOZED,
        column=BoardColumn.WAITING,
        updated_at="2026-05-01T12:00:00+09:00",
        updated_by="planner",
    )

    projected = _project_queue_task(task=task, anima_name="sakura", metadata=metadata)

    assert projected.visibility == AttentionVisibility.SNOOZED
    assert projected.column == BoardColumn.WAITING


def test_unparseable_timestamps_keep_metadata_precedence() -> None:
    """Parse failures / missing timestamps must not drop metadata overrides."""
    task = _queue_entry(status="done", updated_at="not-a-timestamp")
    metadata = TaskBoardMetadata(
        anima_name="sakura",
        task_id=task.task_id,
        visibility=AttentionVisibility.ACTIVE,
        column=BoardColumn.WAITING,
        updated_at="also-bad",
        updated_by="delegator",
    )

    projected = _project_queue_task(task=task, anima_name="sakura", metadata=metadata)

    assert projected.visibility == AttentionVisibility.ACTIVE
    assert projected.column == BoardColumn.WAITING


def test_integration_stale_metadata_hidden_from_default_board_view(tmp_path: Path) -> None:
    """End-to-end-ish: done queue + old active metadata is not shown by default."""
    manager = _queue(tmp_path, "sakura")
    task = manager.add_task(
        source="human",
        original_instruction="finish",
        assignee="sakura",
        summary="finish",
        task_id="task-stale",
    )
    # Mark done without pre-existing metadata, then inject a stale active row.
    manager.update_status(task.task_id, "done")
    store = TaskBoardStore(tmp_path / "taskboard.sqlite3")
    store.upsert_metadata(
        anima_name="sakura",
        task_id=task.task_id,
        actor="ghost",
        visibility="active",
        column="waiting",
        updated_at="2000-01-01T00:00:00+09:00",
    )

    assert project_anima(manager.anima_dir, store) == []

    projected = project_anima(manager.anima_dir, store, include_archived=True)
    assert len(projected) == 1
    assert projected[0].visibility == AttentionVisibility.ARCHIVED
    assert projected[0].column == BoardColumn.DONE
