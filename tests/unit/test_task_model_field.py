"""SSoT compliance tests for the per-task ``model`` override.

The model override's single source of truth is the pending task_desc and the
task queue entry meta (``meta.model``), **not** a column on the TaskBoard /
external-task projection schemas.  These models must still load legacy JSON
without a ``model`` key, and the SSoT propagation paths (queue meta ->
pending task_desc) must carry the model through.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from core.external_tasks.models import ExternalTask
from core.schemas import TaskEntry
from core.taskboard.models import AttentionVisibility, BoardColumn, BoardTask


def _entry(**overrides) -> TaskEntry:
    fields = {
        "task_id": "abc123",
        "ts": "2026-01-01T00:00:00+00:00",
        "source": "anima",
        "original_instruction": "do the thing",
        "assignee": "anima",
        "status": "blocked",
        "summary": "thing",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    fields.update(overrides)
    return TaskEntry(**fields)


def _board_task(**overrides):
    fields = {
        "anima_name": "a",
        "task_id": "t1",
        "visibility": AttentionVisibility.ACTIVE,
        "column": BoardColumn.TODO,
    }
    fields.update(overrides)
    return BoardTask(**fields)


class TestExternalTaskNoModelField:
    def test_loads_without_model_field(self):
        # Legacy JSON (without any model key) must still parse cleanly.
        data = {
            "id": "github-pr-42",
            "title": "Fix bug",
            "status": "open",
            "source_type": "github",
            "source_icon": "gh",
            "created_at": "2026-01-01T00:00:00+00:00",
            "last_updated_at": "2026-01-01T00:00:00+00:00",
            "priority": 1,
        }
        task = ExternalTask.model_validate(data)
        assert task.id == "github-pr-42"

    def test_model_is_not_a_schema_field(self):
        # The ``model`` key is no longer part of the ExternalTask schema
        # (the SSoT is the task queue meta + pending task_desc).
        assert "model" not in ExternalTask.model_fields


class TestBoardTaskNoModelField:
    def test_loads_without_model_field(self):
        task = _board_task()
        assert task.task_id == "t1"

    def test_model_is_not_a_schema_field(self):
        # SSoT is the pending task_desc + queue meta, not the TaskBoard row.
        assert "model" not in BoardTask.model_fields
