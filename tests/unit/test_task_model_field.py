"""Unit tests for the per-task optional ``model`` schema field.

Verifies backward compatibility: JSON snapshots without the ``model`` field
still load, and the field round-trips when present.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from core.external_tasks.models import ExternalTask
from core.taskboard.models import AttentionVisibility, BoardColumn, BoardTask


def _board_task(**overrides):
    fields = {
        "anima_name": "a",
        "task_id": "t1",
        "visibility": AttentionVisibility.ACTIVE,
        "column": BoardColumn.TODO,
    }
    fields.update(overrides)
    return BoardTask(**fields)


class TestExternalTaskModelField:
    def test_loads_without_model_field(self):
        # JSON produced before the field existed must still parse (default None).
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
        assert task.model is None

    def test_round_trip_preserves_model(self):
        task = ExternalTask(
            id="github-pr-42",
            title="Fix bug",
            status="open",
            source_type="github",
            source_icon="gh",
            created_at="2026-01-01T00:00:00+00:00",
            last_updated_at="2026-01-01T00:00:00+00:00",
            priority=1,
            model="c:codex/gpt-5.6-sol",
        )
        restored = ExternalTask.model_validate(task.model_dump())
        assert restored.model == "c:codex/gpt-5.6-sol"


class TestBoardTaskModelField:
    def test_loads_without_model_field(self):
        task = _board_task()
        assert task.model is None

    def test_round_trip_preserves_model(self):
        task = _board_task(model="claude-sonnet-4-6")
        restored = BoardTask.model_validate(task.model_dump())
        assert restored.model == "claude-sonnet-4-6"
