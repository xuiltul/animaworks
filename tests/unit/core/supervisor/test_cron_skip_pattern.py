"""Unit tests for skip_pattern filtering in command-type cron tasks."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.schemas import CronTask
from core.supervisor.scheduler_manager import SchedulerManager


# ── Helpers ──────────────────────────────────────────────────


def _make_scheduler_mgr() -> SchedulerManager:
    """Create a SchedulerManager with minimal config for unit testing."""
    mock_anima = MagicMock()
    mock_anima.memory = MagicMock()
    mock_anima.run_cron_command = AsyncMock()
    mock_anima.run_cron_task = AsyncMock()
    mock_anima.run_heartbeat = AsyncMock()

    mgr = SchedulerManager(
        anima=mock_anima,
        anima_name="test",
        anima_dir=Path("/tmp/animas/test"),
        emit_event=MagicMock(),
    )
    return mgr


def _make_command_task(
    name: str = "test_task",
    skip_pattern: str | None = None,
    trigger_heartbeat: bool = True,
) -> CronTask:
    return CronTask(
        name=name,
        schedule="*/5 * * * *",
        type="command",
        tool="test_tool",
        skip_pattern=skip_pattern,
        trigger_heartbeat=trigger_heartbeat,
    )


# ── TestSkipPatternFiltering ─────────────────────────────────


class TestSkipPatternFiltering:
    """Tests for skip_pattern in _run_cron_task."""

    @pytest.mark.asyncio
    async def test_empty_array_matches_skip_pattern(self):
        """stdout of '[]' matches skip_pattern '^\\[\\s*\\]$' and suppresses heartbeat."""
        mgr = _make_scheduler_mgr()
        mgr._anima.run_cron_command.return_value = {
            "task": "test_task",
            "exit_code": 0,
            "stdout": "[]",
            "stderr": "",
            "duration_ms": 100,
        }

        task = _make_command_task(skip_pattern=r"^\[\s*\]$")

        await mgr._run_cron_task(task)

        # heartbeat/cron LLM should NOT be triggered
        mgr._anima.run_cron_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_empty_array_does_not_match_skip_pattern(self):
        """stdout with actual data does not match skip_pattern, heartbeat triggers."""
        mgr = _make_scheduler_mgr()
        mgr._anima.run_cron_command.return_value = {
            "task": "test_task",
            "exit_code": 0,
            "stdout": '[{"message_id": "123"}]',
            "stderr": "",
            "duration_ms": 100,
        }

        task = _make_command_task(skip_pattern=r"^\[\s*\]$")

        await mgr._run_cron_task(task)

        # cron LLM follow-up SHOULD be triggered
        mgr._anima.run_cron_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_skip_pattern_triggers_heartbeat(self):
        """Without skip_pattern, any non-empty stdout triggers heartbeat."""
        mgr = _make_scheduler_mgr()
        mgr._anima.run_cron_command.return_value = {
            "task": "test_task",
            "exit_code": 0,
            "stdout": "[]",
            "stderr": "",
            "duration_ms": 100,
        }

        task = _make_command_task(skip_pattern=None)

        await mgr._run_cron_task(task)

        # Without skip_pattern, even '[]' triggers cron LLM follow-up
        mgr._anima.run_cron_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_regex_continues_to_heartbeat(self, caplog):
        """Invalid skip_pattern regex logs warning and continues to trigger heartbeat."""
        import logging

        mgr = _make_scheduler_mgr()
        mgr._anima.run_cron_command.return_value = {
            "task": "test_task",
            "exit_code": 0,
            "stdout": "some output",
            "stderr": "",
            "duration_ms": 100,
        }

        task = _make_command_task(skip_pattern="[unterminated")

        with caplog.at_level(logging.WARNING):
            await mgr._run_cron_task(task)

        # Should still trigger cron LLM despite invalid pattern
        mgr._anima.run_cron_task.assert_called_once()
        assert "Invalid skip_pattern" in caplog.text

    @pytest.mark.asyncio
    async def test_empty_stdout_no_heartbeat(self):
        """Empty stdout (after strip) does not trigger heartbeat."""
        mgr = _make_scheduler_mgr()
        mgr._anima.run_cron_command.return_value = {
            "task": "test_task",
            "exit_code": 0,
            "stdout": "   \n  ",
            "stderr": "",
            "duration_ms": 100,
        }

        task = _make_command_task(skip_pattern=r"^\[\s*\]$")

        await mgr._run_cron_task(task)

        mgr._anima.run_cron_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_trigger_heartbeat_false_suppresses_heartbeat(self):
        """trigger_heartbeat=False suppresses heartbeat even with non-empty stdout."""
        mgr = _make_scheduler_mgr()
        mgr._anima.run_cron_command.return_value = {
            "task": "test_task",
            "exit_code": 0,
            "stdout": '[{"message_id": "123"}]',
            "stderr": "",
            "duration_ms": 100,
        }

        task = _make_command_task(trigger_heartbeat=False)

        await mgr._run_cron_task(task)

        # cron LLM follow-up should NOT be triggered
        mgr._anima.run_cron_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_trigger_heartbeat_false_skips_pending_write(self):
        """trigger_heartbeat=False also skips writing to pending.md."""
        mgr = _make_scheduler_mgr()
        mgr._anima.run_cron_command.return_value = {
            "task": "test_task",
            "exit_code": 0,
            "stdout": "important output",
            "stderr": "",
            "duration_ms": 100,
        }

        task = _make_command_task(trigger_heartbeat=False)

        await mgr._run_cron_task(task)

        mgr._anima.run_cron_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_trigger_heartbeat_true_allows_heartbeat(self):
        """trigger_heartbeat=True (default) allows heartbeat as normal."""
        mgr = _make_scheduler_mgr()
        mgr._anima.run_cron_command.return_value = {
            "task": "test_task",
            "exit_code": 0,
            "stdout": "some output",
            "stderr": "",
            "duration_ms": 100,
        }

        task = _make_command_task(trigger_heartbeat=True)

        await mgr._run_cron_task(task)

        mgr._anima.run_cron_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_heartbeat_false_takes_precedence_over_skip_pattern(self):
        """trigger_heartbeat=False returns before skip_pattern is even checked."""
        mgr = _make_scheduler_mgr()
        mgr._anima.run_cron_command.return_value = {
            "task": "test_task",
            "exit_code": 0,
            "stdout": '[{"data": "real"}]',  # would NOT match skip_pattern
            "stderr": "",
            "duration_ms": 100,
        }

        task = _make_command_task(
            skip_pattern=r"^\[\s*\]$",
            trigger_heartbeat=False,
        )

        await mgr._run_cron_task(task)

        # Despite stdout not matching skip_pattern,
        # trigger_heartbeat=False suppresses cron LLM follow-up
        mgr._anima.run_cron_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_nonzero_exit_code_no_heartbeat(self):
        """Non-zero exit code does not trigger heartbeat regardless of stdout."""
        mgr = _make_scheduler_mgr()
        mgr._anima.run_cron_command.return_value = {
            "task": "test_task",
            "exit_code": 1,
            "stdout": "error occurred",
            "stderr": "some error",
            "duration_ms": 100,
        }

        task = _make_command_task(skip_pattern=None)

        await mgr._run_cron_task(task)

        mgr._anima.run_cron_task.assert_not_called()
