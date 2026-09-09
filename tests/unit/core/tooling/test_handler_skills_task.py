from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for submit_tasks handler — reply_to and task descriptor validation."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from core.tooling.handler import ToolHandler

# ── Test 5: submit_tasks sets reply_to ────────────────────────────────────


def test_submit_tasks_sets_reply_to(tmp_path: Path) -> None:
    """Test that submit_tasks handler sets reply_to = self._anima_name in task descriptors.

    This requires mocking the file system for pending task files.
    """
    anima_dir = tmp_path / "animas" / "sakura"
    anima_dir.mkdir(parents=True)
    (anima_dir / "permissions.md").write_text("", encoding="utf-8")
    (anima_dir / "state").mkdir(exist_ok=True)

    memory = MagicMock()
    memory.read_permissions.return_value = ""

    handler = ToolHandler(
        anima_dir=anima_dir,
        memory=memory,
        tool_registry=[],
    )

    result = handler.handle(
        "submit_tasks",
        {
            "batch_id": "test-batch-1",
            "tasks": [
                {
                    "task_id": "task-a",
                    "title": "Task A",
                    "description": "Do thing A",
                },
                {
                    "task_id": "task-b",
                    "title": "Task B",
                    "description": "Do thing B",
                    "depends_on": ["task-a"],
                },
            ],
        },
    )

    parsed = json.loads(result)
    assert parsed.get("status") == "submitted"
    assert parsed.get("batch_id") == "test-batch-1"
    assert set(parsed.get("task_ids", [])) == {"task-a", "task-b"}

    from core.memory.task_queue import TaskQueueManager

    manager = TaskQueueManager(anima_dir)
    assert not list((anima_dir / "state" / "pending").glob("*.json"))

    for task_id in ("task-a", "task-b"):
        data = manager.store.get_input("sakura", task_id)
        assert data is not None
        assert data.get("reply_to") == "sakura", (
            f"Task {task_id} should have reply_to='sakura' (anima_dir.name), got {data.get('reply_to')!r}"
        )
        assert data.get("submitted_by") == "sakura"
        assert data.get("task_type") == "llm"
        assert data.get("batch_id") == "test-batch-1"
