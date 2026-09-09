from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from core.memory.task_queue import TaskQueueManager
from core.taskboard.models import AttentionVisibility
from core.taskboard.store import TaskBoardStore
from core.taskboard.tasks import process_identity
from core.tasks_dispatch import publish_tasks

pytestmark = pytest.mark.e2e


def test_archived_cancelled_task_cannot_be_republished_as_new_work(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    anima_dir = data_dir / "animas" / "sakura"
    (anima_dir / "state").mkdir(parents=True, exist_ok=True)

    queue = TaskQueueManager(anima_dir)
    payload = {"task_id": "archived1234", "title": "do not resurrect", "description": "do not resurrect this task"}
    entry = publish_tasks(anima_dir, [payload])[0]
    queue.update_status(entry.task_id, "cancelled")

    store = TaskBoardStore(data_dir / "shared" / "taskboard.sqlite3")
    store.upsert_metadata(
        anima_name="sakura",
        task_id=entry.task_id,
        visibility=AttentionVisibility.ARCHIVED,
    )

    # Duplicate delivery is idempotent, not a resume/reconstruction request.
    publish_tasks(anima_dir, [payload])

    assert not (anima_dir / "state" / "pending" / f"{entry.task_id}.json").exists()
    assert not (anima_dir / "state" / "pending" / "deferred" / f"{entry.task_id}.json").exists()
    assert TaskQueueManager(anima_dir).get_task_by_id(entry.task_id).status == "cancelled"
    assert queue.store.pending("sakura") == []
    assert queue.store.claim("sakura", entry.task_id, process_identity()) is None


def test_snoozed_task_needs_explicit_resume_and_retains_complete_input(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    anima_dir = data_dir / "animas" / "sakura"
    (anima_dir / "state").mkdir(parents=True, exist_ok=True)

    queue = TaskQueueManager(anima_dir)
    payload = {"task_id": "snoozed1234", "title": "wake later", "description": "full original task context"}
    entry = publish_tasks(anima_dir, [payload])[0]
    attempt = queue.store.claim("sakura", entry.task_id, process_identity())
    queue.store.finish(attempt["_attempt_token"], status="pending", stop_kind="interrupted")
    TaskBoardStore(data_dir / "shared" / "taskboard.sqlite3").upsert_metadata(
        anima_name="sakura",
        task_id=entry.task_id,
        visibility=AttentionVisibility.SNOOZED,
        snoozed_until=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
    )

    assert queue.store.pending("sakura") == []
    publish_tasks(anima_dir, [{"task_id": entry.task_id, "resume": True}])

    assert [item["task_id"] for item in queue.store.pending("sakura")] == [entry.task_id]
    assert queue.store.get_input("sakura", entry.task_id)["description"] == payload["description"]
    assert not (anima_dir / "state" / "pending" / f"{entry.task_id}.json").exists()
    assert not (anima_dir / "state" / "pending" / "deferred" / f"{entry.task_id}.json").exists()
