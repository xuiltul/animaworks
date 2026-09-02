from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from core.memory.task_queue import TaskQueueManager


def _make_handler(tmp_path: Path) -> Any:
    from core.tooling.handler_skills import SkillsToolsMixin

    handler = object.__new__(SkillsToolsMixin)
    handler._anima_dir = tmp_path / "data" / "animas" / "sakura"
    handler._anima_name = "sakura"
    handler._activity = MagicMock()
    handler._pending_executor_wake = None
    (handler._anima_dir / "state").mkdir(parents=True, exist_ok=True)
    return handler


def _add_task(handler: Any, task_id: str) -> Any:
    queue = TaskQueueManager(handler._anima_dir)
    return queue.add_task(
        source="human",
        original_instruction="some work",
        assignee="sakura",
        summary="some work",
        task_id=task_id,
        meta={"task_desc": {"title": "some work"}},
    )


def test_update_task_pending_just_marks_pending(tmp_path: Path) -> None:
    """update_task(pending) records pending and nothing else.

    The harness no longer regenerates a descriptor or re-dispatches the task:
    the anima sees it in its own pending list and decides what to do.
    """
    handler = _make_handler(tmp_path)
    entry = _add_task(handler, "retry1234")

    result = json.loads(handler._handle_update_task({"task_id": entry.task_id, "status": "pending"}))

    assert result["status"] == "pending"
    assert not (handler._anima_dir / "state" / "pending" / f"{entry.task_id}.json").exists()
    reloaded = TaskQueueManager(handler._anima_dir).get_task_by_id(entry.task_id)
    assert reloaded is not None
    assert reloaded.status == "pending"
    assert "retry_count" not in reloaded.meta


def test_update_task_pending_requeues_a_cancelled_task(tmp_path: Path) -> None:
    """An explicit re-queue is the only way out of cancelled, and it is allowed."""
    handler = _make_handler(tmp_path)
    entry = _add_task(handler, "cancelled1234")
    TaskQueueManager(handler._anima_dir).update_status(entry.task_id, "cancelled", summary="gave up")

    result = json.loads(handler._handle_update_task({"task_id": entry.task_id, "status": "pending"}))

    assert result["status"] == "pending"
    assert not (handler._anima_dir / "state" / "pending" / f"{entry.task_id}.json").exists()


def test_update_task_pending_on_unknown_task_is_an_error(tmp_path: Path) -> None:
    handler = _make_handler(tmp_path)

    result = json.loads(handler._handle_update_task({"task_id": "missing1234", "status": "pending"}))

    assert result["status"] == "error"
    assert result["error_type"] == "TaskNotFound"
