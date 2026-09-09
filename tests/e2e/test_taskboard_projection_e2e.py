from __future__ import annotations

from pathlib import Path

import pytest

from core.memory.task_queue import TaskQueueManager
from core.taskboard.models import AttentionVisibility, BoardColumn
from core.taskboard.projector import project_all
from core.taskboard.store import TaskBoardStore

pytestmark = pytest.mark.e2e


def test_taskboard_projection_uses_jsonl_queue_and_sqlite_metadata(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    animas_dir = data_dir / "animas"
    sakura_dir = animas_dir / "sakura"
    hinata_dir = animas_dir / "hinata"
    (sakura_dir / "state").mkdir(parents=True)
    (hinata_dir / "state").mkdir(parents=True)

    store = TaskBoardStore(data_dir / "shared" / "taskboard.sqlite3")
    sakura_queue = TaskQueueManager(sakura_dir)
    hinata_queue = TaskQueueManager(hinata_dir)

    sakura_task = sakura_queue.add_task(
        source="human",
        original_instruction="prepare rollout",
        assignee="sakura",
        summary="prepare rollout",
        task_id="task-rollout",
    )
    hinata_task = hinata_queue.add_task(
        source="human",
        original_instruction="review rollout",
        assignee="hinata",
        summary="review rollout",
        task_id="task-review",
    )
    store.upsert_metadata(
        anima_name="sakura",
        task_id=sakura_task.task_id,
        actor="planner",
        column="waiting",
        position=1.0,
        source_ref="task_queue:sakura:task-rollout",
    )
    store.record_surface(anima_name="sakura", task_id=sakura_task.task_id, actor="runtime")
    store.upsert_metadata(
        anima_name="hinata",
        task_id=hinata_task.task_id,
        actor="planner",
        visibility="snoozed",
        snoozed_until="2026-05-15T00:00:00+09:00",
    )
    store.upsert_metadata(
        anima_name="sakura",
        task_id="missing-task",
        actor="planner",
        visibility="tombstoned",
        column="suppressed",
        tombstone_reason="queue compaction removed it",
    )

    projected = project_all(
        animas_dir,
        store,
        include_missing=True,
        include_archived=True,
    )
    by_key = {(task.anima_name, task.task_id): task for task in projected}

    assert by_key[("sakura", "task-rollout")].column == BoardColumn.WAITING
    assert by_key[("sakura", "task-rollout")].surface_count == 1
    assert by_key[("hinata", "task-review")].queue_status == "pending"
    assert by_key[("hinata", "task-review")].visibility == AttentionVisibility.SNOOZED
    assert by_key[("sakura", "missing-task")].queue_missing is True
    assert by_key[("sakura", "missing-task")].tombstone_reason == "queue compaction removed it"
    assert [event["event_type"] for event in store.list_events(anima_name="sakura", task_id="task-rollout")] == [
        "metadata_upserted",
        "surface_recorded",
    ]


def test_terminal_status_syncs_metadata_and_hides_from_active_board(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Completing a delegated task archives board metadata and drops it from the active view."""
    data_dir = tmp_path / "data"
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(data_dir))
    animas_dir = data_dir / "animas"
    sakura_dir = animas_dir / "sakura"
    (sakura_dir / "state").mkdir(parents=True)

    store = TaskBoardStore(data_dir / "shared" / "taskboard.sqlite3")
    queue = TaskQueueManager(sakura_dir)
    task = queue.add_task(
        source="human",
        original_instruction="delegated work",
        assignee="sakura",
        summary="delegated work",
        task_id="task-complete",
    )
    store.upsert_metadata(
        anima_name="sakura",
        task_id=task.task_id,
        actor="delegator",
        visibility="active",
        column="waiting",
    )

    updated = queue.update_status(task.task_id, "done", summary="delegated work done")
    assert updated is not None
    assert updated.status == "done"

    meta = store.get_metadata("sakura", task.task_id)
    assert meta is not None
    assert meta.visibility == AttentionVisibility.ARCHIVED
    assert meta.column == BoardColumn.DONE

    active = project_all(animas_dir, store)
    assert all(t.task_id != task.task_id for t in active)

    archived = project_all(animas_dir, store, include_archived=True)
    by_id = {t.task_id: t for t in archived}
    assert by_id[task.task_id].visibility == AttentionVisibility.ARCHIVED
    assert by_id[task.task_id].column == BoardColumn.DONE
    assert by_id[task.task_id].queue_status == "done"
    assert any(
        event["event_type"] == "archived"
        for event in store.list_events(anima_name="sakura", task_id=task.task_id)
    )
