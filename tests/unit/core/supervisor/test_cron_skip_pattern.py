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
from core.supervisor.runner import AnimaRunner


# ── Helpers ──────────────────────────────────────────────────


def _make_runner() -> AnimaRunner:
    """Create a AnimaRunner with minimal config for unit testing."""
    runner = AnimaRunner(
        anima_name="test",
        socket_path=Path("/tmp/test.sock"),
        animas_dir=Path("/tmp/animas"),
        shared_dir=Path("/tmp/shared"),
    )
    runner.anima = MagicMock()
    runner.anima.memory = MagicMock()
    runner.anima.run_cron_command = AsyncMock()
    runner.anima.run_heartbeat = AsyncMock()
    return runner


def _make_command_task(
    name: str = "test_task",
    skip_pattern: str | None = None,
) -> CronTask:
    return CronTask(
        name=name,
        schedule="*/5 * * * *",
        type="command",
        tool="test_tool",
        skip_pattern=skip_pattern,
    )


# ── TestSkipPatternFiltering ─────────────────────────────────


class TestSkipPatternFiltering:
    """Tests for skip_pattern in _run_cron_task."""

    @pytest.mark.asyncio
    async def test_empty_array_matches_skip_pattern(self):
        """stdout of '[]' matches skip_pattern '^\\[\\s*\\]$' and suppresses heartbeat."""
        runner = _make_runner()
        runner.anima.run_cron_command.return_value = {
            "task": "test_task",
            "exit_code": 0,
            "stdout": "[]",
            "stderr": "",
            "duration_ms": 100,
        }

        task = _make_command_task(skip_pattern=r"^\[\s*\]$")

        with patch.object(runner, "_emit_event"):
            await runner._run_cron_task(task)

        # heartbeat should NOT be triggered
        runner.anima.memory.update_pending.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_empty_array_does_not_match_skip_pattern(self):
        """stdout with actual data does not match skip_pattern, heartbeat triggers."""
        runner = _make_runner()
        runner.anima.run_cron_command.return_value = {
            "task": "test_task",
            "exit_code": 0,
            "stdout": '[{"message_id": "123"}]',
            "stderr": "",
            "duration_ms": 100,
        }

        task = _make_command_task(skip_pattern=r"^\[\s*\]$")

        with patch.object(runner, "_emit_event"), \
             patch.object(runner, "_heartbeat_tick", new_callable=AsyncMock):
            await runner._run_cron_task(task)

        # heartbeat SHOULD be triggered
        runner.anima.memory.update_pending.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_skip_pattern_triggers_heartbeat(self):
        """Without skip_pattern, any non-empty stdout triggers heartbeat."""
        runner = _make_runner()
        runner.anima.run_cron_command.return_value = {
            "task": "test_task",
            "exit_code": 0,
            "stdout": "[]",
            "stderr": "",
            "duration_ms": 100,
        }

        task = _make_command_task(skip_pattern=None)

        with patch.object(runner, "_emit_event"), \
             patch.object(runner, "_heartbeat_tick", new_callable=AsyncMock):
            await runner._run_cron_task(task)

        # Without skip_pattern, even '[]' triggers heartbeat
        runner.anima.memory.update_pending.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_regex_continues_to_heartbeat(self, caplog):
        """Invalid skip_pattern regex logs warning and continues to trigger heartbeat."""
        import logging

        runner = _make_runner()
        runner.anima.run_cron_command.return_value = {
            "task": "test_task",
            "exit_code": 0,
            "stdout": "some output",
            "stderr": "",
            "duration_ms": 100,
        }

        task = _make_command_task(skip_pattern="[unterminated")

        with patch.object(runner, "_emit_event"), \
             patch.object(runner, "_heartbeat_tick", new_callable=AsyncMock), \
             caplog.at_level(logging.WARNING):
            await runner._run_cron_task(task)

        # Should still trigger heartbeat despite invalid pattern
        runner.anima.memory.update_pending.assert_called_once()
        assert "Invalid skip_pattern" in caplog.text

    @pytest.mark.asyncio
    async def test_empty_stdout_no_heartbeat(self):
        """Empty stdout (after strip) does not trigger heartbeat."""
        runner = _make_runner()
        runner.anima.run_cron_command.return_value = {
            "task": "test_task",
            "exit_code": 0,
            "stdout": "   \n  ",
            "stderr": "",
            "duration_ms": 100,
        }

        task = _make_command_task(skip_pattern=r"^\[\s*\]$")

        with patch.object(runner, "_emit_event"):
            await runner._run_cron_task(task)

        runner.anima.memory.update_pending.assert_not_called()

    @pytest.mark.asyncio
    async def test_nonzero_exit_code_no_heartbeat(self):
        """Non-zero exit code does not trigger heartbeat regardless of stdout."""
        runner = _make_runner()
        runner.anima.run_cron_command.return_value = {
            "task": "test_task",
            "exit_code": 1,
            "stdout": "error occurred",
            "stderr": "some error",
            "duration_ms": 100,
        }

        task = _make_command_task(skip_pattern=None)

        with patch.object(runner, "_emit_event"):
            await runner._run_cron_task(task)

        runner.anima.memory.update_pending.assert_not_called()
