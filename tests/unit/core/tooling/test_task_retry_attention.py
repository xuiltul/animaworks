from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from core.memory.task_queue import TaskQueueManager
from core.taskboard.models import AttentionVisibility
from core.taskboard.store import TaskBoardStore


def _make_handler(tmp_path: Path) -> Any:
    from core.tooling.handler_skills import SkillsToolsMixin

    handler = object.__new__(SkillsToolsMixin)
    handler._anima_dir = tmp_path / "data" / "animas" / "sakura"
    handler._anima_name = "sakura"
    handler._activity = MagicMock()
    handler._pending_executor_wake = None
    (handler._anima_dir / "state").mkdir(parents=True, exist_ok=True)
    return handler


def _store_for(handler: Any) -> TaskBoardStore:
    return TaskBoardStore(handler._anima_dir.parent.parent / "shared" / "taskboard.sqlite3")


def test_update_task_pending_returns_task_suppressed_for_archived_task(tmp_path: Path) -> None:
    handler = _make_handler(tmp_path)
    queue = TaskQueueManager(handler._anima_dir)
    entry = queue.add_task(
        source="human",
        original_instruction="old work",
        assignee="sakura",
        summary="old work",
        task_id="archived1234",
        meta={"task_desc": {"title": "old work"}},
    )
    # "failed" was retired (A1 task-model teardown); "cancelled" is the
    # closest still-valid terminal status for this archived-task scenario.
    queue.update_status(entry.task_id, "cancelled", summary="cancelled: old failure")
    _store_for(handler).upsert_metadata(
        anima_name="sakura",
        task_id=entry.task_id,
        visibility=AttentionVisibility.ARCHIVED,
    )

    result = json.loads(handler._handle_update_task({"task_id": entry.task_id, "status": "pending"}))

    assert result["status"] == "error"
    assert result["error_type"] == "TaskSuppressed"
    assert not (handler._anima_dir / "state" / "pending" / f"{entry.task_id}.json").exists()
    assert TaskQueueManager(handler._anima_dir).get_task_by_id(entry.task_id).status == "cancelled"


def test_update_task_pending_persists_retry_count(tmp_path: Path) -> None:
    handler = _make_handler(tmp_path)
    queue = TaskQueueManager(handler._anima_dir)
    entry = queue.add_task(
        source="human",
        original_instruction="retry work",
        assignee="sakura",
        summary="retry work",
        task_id="retry1234",
        meta={"task_desc": {"title": "retry work"}},
    )

    result = json.loads(handler._handle_update_task({"task_id": entry.task_id, "status": "pending"}))

    assert result["status"] == "in_progress"
    assert (handler._anima_dir / "state" / "pending" / f"{entry.task_id}.json").exists()
    reloaded = TaskQueueManager(handler._anima_dir).get_task_by_id(entry.task_id)
    assert reloaded is not None
    assert reloaded.meta["retry_count"] == 1


def test_update_task_pending_existing_processing_keeps_in_progress_status(tmp_path: Path) -> None:
    handler = _make_handler(tmp_path)
    queue = TaskQueueManager(handler._anima_dir)
    entry = queue.add_task(
        source="human",
        original_instruction="already processing",
        assignee="sakura",
        summary="already processing",
        task_id="processing1234",
        meta={"task_desc": {"title": "already processing"}},
    )
    processing_dir = handler._anima_dir / "state" / "pending" / "processing"
    processing_dir.mkdir(parents=True, exist_ok=True)
    (processing_dir / f"{entry.task_id}.json").write_text("{}", encoding="utf-8")

    result = json.loads(handler._handle_update_task({"task_id": entry.task_id, "status": "pending"}))

    assert result["status"] == "in_progress"
    reloaded = TaskQueueManager(handler._anima_dir).get_task_by_id(entry.task_id)
    assert reloaded is not None
    assert reloaded.status == "in_progress"
