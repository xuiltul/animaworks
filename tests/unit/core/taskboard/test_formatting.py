from __future__ import annotations

from core.taskboard.formatting import format_tasks_for_priming
from core.taskboard.models import AttentionVisibility, BoardColumn, BoardTask


def _task(task_id: str, status: str, *, meta: dict | None = None) -> BoardTask:
    return BoardTask(
        anima_name="fixture",
        task_id=task_id,
        source="human",
        summary=f"Summary {task_id}",
        queue_status=status,
        meta=meta or {},
        visibility=AttentionVisibility.ACTIVE,
        column=BoardColumn.TODO,
    )


def test_terminal_tasks_are_excluded_but_active_statuses_remain() -> None:
    output = format_tasks_for_priming(
        [
            _task("pending1", "pending"),
            _task("progress", "in_progress"),
            _task("done0001", "done"),
            _task("cancel001", "cancelled"),
        ]
    )

    assert "pending1" in output
    assert "progress" in output
    assert "done0001" not in output
    assert "cancel001" not in output


def test_delegated_tasks_keep_result_status_and_icon() -> None:
    output = format_tasks_for_priming(
        [
            _task("delegate", "delegated", meta={"delegated_to": "worker", "delegated_status": "done"}),
            _task("delegate2", "delegated", meta={"delegated_to": "worker", "delegated_status": "cancelled"}),
        ]
    )

    assert "delegate" in output
    assert "done ✅" in output
    assert "cancelled 🚫" in output
