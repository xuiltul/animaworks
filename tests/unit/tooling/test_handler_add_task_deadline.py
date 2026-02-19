"""Tests for deadline validation in ToolHandler._handle_add_task()."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import datetime
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


class TestHandlerDeadlineValidation:
    """Tests that _handle_add_task returns errors for missing/invalid deadline."""

    def test_missing_deadline_returns_error(self, handler: ToolHandler):
        """Handler should return a structured error when deadline is not provided."""
        result = handler.handle(
            "add_task",
            {
                "source": "human",
                "original_instruction": "Do something",
                "assignee": "rin",
                "summary": "Test task",
                # deadline intentionally omitted
            },
        )
        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert parsed["error_type"] == "InvalidArguments"
        assert "deadline" in parsed["message"].lower()

    def test_empty_deadline_returns_error(self, handler: ToolHandler):
        """Handler should return error for empty string deadline."""
        result = handler.handle(
            "add_task",
            {
                "source": "human",
                "original_instruction": "Do something",
                "assignee": "rin",
                "summary": "Test task",
                "deadline": "",
            },
        )
        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert parsed["error_type"] == "InvalidArguments"
        assert "deadline" in parsed["message"].lower()

    def test_none_deadline_returns_error(self, handler: ToolHandler):
        """Handler should return error when deadline is None."""
        result = handler.handle(
            "add_task",
            {
                "source": "human",
                "original_instruction": "Do something",
                "assignee": "rin",
                "summary": "Test task",
                "deadline": None,
            },
        )
        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert parsed["error_type"] == "InvalidArguments"
        assert "deadline" in parsed["message"].lower()

    def test_invalid_deadline_format_returns_error(self, handler: ToolHandler):
        """Handler should catch ValueError from _parse_deadline for bad formats."""
        result = handler.handle(
            "add_task",
            {
                "source": "human",
                "original_instruction": "Do something",
                "assignee": "rin",
                "summary": "Test task",
                "deadline": "not-a-valid-format",
            },
        )
        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert parsed["error_type"] == "InvalidArguments"
        assert "invalid deadline format" in parsed["message"].lower()

    def test_valid_relative_deadline_succeeds(self, handler: ToolHandler):
        """Handler should successfully create a task with a relative deadline."""
        result = handler.handle(
            "add_task",
            {
                "source": "human",
                "original_instruction": "Do something",
                "assignee": "rin",
                "summary": "Test task",
                "deadline": "1h",
            },
        )
        parsed = json.loads(result)
        # On success, the result is the TaskEntry JSON (no "status": "error")
        assert "error" not in parsed.get("status", "")
        assert parsed["assignee"] == "rin"
        assert parsed["deadline"] is not None
        # Verify the deadline was converted from relative to ISO8601
        dt = datetime.fromisoformat(parsed["deadline"])
        assert dt > datetime.now()

    def test_valid_iso8601_deadline_succeeds(self, handler: ToolHandler):
        """Handler should successfully create a task with an ISO8601 deadline."""
        result = handler.handle(
            "add_task",
            {
                "source": "human",
                "original_instruction": "Do something",
                "assignee": "rin",
                "summary": "Test task",
                "deadline": "2026-12-31T23:59:59",
            },
        )
        parsed = json.loads(result)
        assert "error" not in parsed.get("status", "")
        assert parsed["deadline"] == "2026-12-31T23:59:59"

    def test_missing_instruction_still_returns_error(self, handler: ToolHandler):
        """Missing instruction should be caught before deadline validation."""
        result = handler.handle(
            "add_task",
            {
                "source": "human",
                "original_instruction": "",
                "assignee": "rin",
                "summary": "Test task",
                "deadline": "1h",
            },
        )
        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert "original_instruction" in parsed["message"]

    def test_missing_assignee_returns_error(self, handler: ToolHandler):
        """Missing assignee should be caught before deadline validation."""
        result = handler.handle(
            "add_task",
            {
                "source": "human",
                "original_instruction": "Do something",
                "assignee": "",
                "summary": "Test task",
                "deadline": "1h",
            },
        )
        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert "assignee" in parsed["message"]
