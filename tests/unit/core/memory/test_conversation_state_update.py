# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for conversation_state_update fuzzy match fix (GH #145 Bug E)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory.conversation_state_update import _update_state_from_summary


def _make_parsed(resolved_items: list[str]):
    parsed = MagicMock()
    parsed.resolved_items = resolved_items
    parsed.new_tasks = []
    return parsed


def _make_task(task_id: str, summary: str):
    task = MagicMock()
    task.task_id = task_id
    task.summary = summary
    return task


class TestFuzzyMatchMinLength:
    """Bug E: fuzzy match should reject short substring matches."""

    def test_short_summary_not_matched(self, tmp_path):
        """A 3-char summary should NOT match via substring."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)

        task = _make_task("t1", "fix")
        mock_tqm = MagicMock()
        mock_tqm.load_active_tasks.return_value = {"t1": task}

        parsed = _make_parsed(["fix the big authentication bug"])
        memory_mgr = MagicMock()
        memory_mgr.anima_dir = anima_dir

        with patch(
            "core.memory.task_queue.TaskQueueManager",
            return_value=mock_tqm,
        ):
            _update_state_from_summary(anima_dir, memory_mgr, parsed)

        mock_tqm.update_status.assert_not_called()

    def test_long_summary_matched(self, tmp_path):
        """An 8+ char summary that IS a substring should match."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)

        task = _make_task("t1", "implement authentication flow")
        mock_tqm = MagicMock()
        mock_tqm.load_active_tasks.return_value = {"t1": task}

        parsed = _make_parsed(["implement authentication flow for users"])
        memory_mgr = MagicMock()
        memory_mgr.anima_dir = anima_dir

        with patch(
            "core.memory.task_queue.TaskQueueManager",
            return_value=mock_tqm,
        ):
            _update_state_from_summary(anima_dir, memory_mgr, parsed)

        mock_tqm.update_status.assert_called_once_with(
            "t1", "done", summary="implement authentication flow for users"
        )

    def test_exact_match_short_rejected(self, tmp_path):
        """Even exact match of short string (<8 chars) should be rejected."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)

        task = _make_task("t1", "テスト追加")
        mock_tqm = MagicMock()
        mock_tqm.load_active_tasks.return_value = {"t1": task}

        parsed = _make_parsed(["テスト追加"])
        memory_mgr = MagicMock()
        memory_mgr.anima_dir = anima_dir

        with patch(
            "core.memory.task_queue.TaskQueueManager",
            return_value=mock_tqm,
        ):
            _update_state_from_summary(anima_dir, memory_mgr, parsed)

        mock_tqm.update_status.assert_not_called()

    def test_boundary_8_chars_matched(self, tmp_path):
        """Exactly 8 chars should be matched."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)

        task = _make_task("t1", "12345678")
        mock_tqm = MagicMock()
        mock_tqm.load_active_tasks.return_value = {"t1": task}

        parsed = _make_parsed(["12345678"])
        memory_mgr = MagicMock()
        memory_mgr.anima_dir = anima_dir

        with patch(
            "core.memory.task_queue.TaskQueueManager",
            return_value=mock_tqm,
        ):
            _update_state_from_summary(anima_dir, memory_mgr, parsed)

        mock_tqm.update_status.assert_called_once()

    def test_boundary_7_chars_not_matched(self, tmp_path):
        """7 chars should NOT be matched (below threshold)."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)

        task = _make_task("t1", "1234567")
        mock_tqm = MagicMock()
        mock_tqm.load_active_tasks.return_value = {"t1": task}

        parsed = _make_parsed(["1234567"])
        memory_mgr = MagicMock()
        memory_mgr.anima_dir = anima_dir

        with patch(
            "core.memory.task_queue.TaskQueueManager",
            return_value=mock_tqm,
        ):
            _update_state_from_summary(anima_dir, memory_mgr, parsed)

        mock_tqm.update_status.assert_not_called()

    def test_no_resolved_items(self, tmp_path):
        """Empty resolved_items should not call update_status."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)

        mock_tqm = MagicMock()
        mock_tqm.load_active_tasks.return_value = {}

        parsed = _make_parsed([])
        memory_mgr = MagicMock()
        memory_mgr.anima_dir = anima_dir

        with patch(
            "core.memory.task_queue.TaskQueueManager",
            return_value=mock_tqm,
        ):
            _update_state_from_summary(anima_dir, memory_mgr, parsed)

        mock_tqm.update_status.assert_not_called()
