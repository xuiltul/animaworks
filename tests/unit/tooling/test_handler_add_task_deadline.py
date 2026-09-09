"""Tests for ToolHandler._handle_backlog_task() required-field validation.

deadline was retired from backlog_task (A1 task-model teardown, 2026-09) —
this file used to cover deadline validation exclusively; it now covers what
remains: backlog_task no longer accepts/requires a deadline, and the other
required fields are still enforced.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.tooling.handler import ToolHandler

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "test-anima"
    d.mkdir(parents=True)
    (d / "state").mkdir(parents=True)
    (d / "permissions.md").write_text("", encoding="utf-8")
    return d


@pytest.fixture
def memory(anima_dir: Path) -> MagicMock:
    m = MagicMock()
    m.read_permissions.return_value = ""
    m.search_memory_text.return_value = []
    return m


@pytest.fixture
def handler(anima_dir: Path, memory: MagicMock) -> ToolHandler:
    return ToolHandler(
        anima_dir=anima_dir,
        memory=memory,
        messenger=None,
        tool_registry=[],
    )


# ── Tests ─────────────────────────────────────────────────────


class TestHandlerBacklogTaskValidation:
    """Tests for _handle_backlog_task's required-field validation."""

    def test_no_deadline_needed_succeeds(self, handler: ToolHandler):
        """backlog_task must succeed without a deadline (the param was removed)."""
        result = handler.handle(
            "backlog_task",
            {
                "source": "human",
                "original_instruction": "Do something",
                "assignee": "rin",
                "summary": "Test task",
                # deadline intentionally omitted — no longer a parameter at all
            },
        )
        parsed = json.loads(result)
        assert parsed.get("status") != "error"
        assert parsed["assignee"] == "rin"
        assert "deadline" not in parsed

    def test_extra_deadline_argument_is_ignored(self, handler: ToolHandler):
        """A caller still passing a stray 'deadline' arg must not break or persist it."""
        result = handler.handle(
            "backlog_task",
            {
                "source": "human",
                "original_instruction": "Do something",
                "assignee": "rin",
                "summary": "Test task",
                "deadline": "1h",
            },
        )
        parsed = json.loads(result)
        assert parsed.get("status") != "error"
        assert "deadline" not in parsed

    def test_missing_instruction_returns_error(self, handler: ToolHandler):
        result = handler.handle(
            "backlog_task",
            {
                "source": "human",
                "original_instruction": "",
                "assignee": "rin",
                "summary": "Test task",
            },
        )
        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert "original_instruction" in parsed["message"]

    def test_missing_assignee_returns_error(self, handler: ToolHandler):
        result = handler.handle(
            "backlog_task",
            {
                "source": "human",
                "original_instruction": "Do something",
                "assignee": "",
                "summary": "Test task",
            },
        )
        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert "assignee" in parsed["message"]
