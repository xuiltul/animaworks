"""Unit tests for skip_pattern filtering in command-type cron tasks."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.runtime.cron_followup import command_followup_output
from core.runtime.scheduler_manager import SchedulerManager
from core.schemas import CronTask

# ── Helpers ──────────────────────────────────────────────────


def _make_scheduler_mgr(tmp_path: Path) -> SchedulerManager:
    """Create a SchedulerManager with minimal config for unit testing."""
    mock_anima = MagicMock()
    mock_anima.memory = MagicMock()
    mock_anima.shared_dir = tmp_path / "shared"
    mock_anima.shared_dir.mkdir()
    mock_anima.run_cron_command = AsyncMock(
        return_value={"task": "test_task", "exit_code": 0, "stdout": "", "stderr": "", "duration_ms": 100}
    )

    anima_dir = tmp_path / "animas" / "test"
    anima_dir.mkdir(parents=True)
    mgr = SchedulerManager(
        anima=mock_anima,
        anima_name="test",
        anima_dir=anima_dir,
        emit_event=MagicMock(),
    )
    mgr._task_runner_supervisor.run_cron = AsyncMock(
        return_value={"result": {"action": "completed", "summary": "ok"}, "success": True}
    )
    mgr._task_runner_supervisor.run_cron_followup = AsyncMock(
        return_value={"result": {"action": "completed", "summary": "reviewed"}, "success": True}
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
        command="echo hi",
        skip_pattern=skip_pattern,
        trigger_heartbeat=trigger_heartbeat,
    )


def _command_result(
    *,
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> dict:
    return {
        "task": "test_task",
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "duration_ms": 100,
    }


# ── TestDispatchDelegation ─────────────────────────────────


class TestDispatchDelegation:
    """LLM cron tasks use runners; shell command execution stays in the scheduler root."""

    @pytest.mark.asyncio
    async def test_llm_task_delegates_to_supervisor(self, tmp_path: Path):
        mgr = _make_scheduler_mgr(tmp_path)
        task = CronTask(
            name="review",
            schedule="0 9 * * *",
            type="llm",
            description="Review PRs",
            skills=["github-pr-review"],
        )

        await mgr._run_cron_task(task)

        mgr._task_runner_supervisor.run_cron.assert_awaited_once_with(task)
        mgr._anima.run_cron_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_command_task_runs_in_root_without_runner(self, tmp_path: Path):
        mgr = _make_scheduler_mgr(tmp_path)
        task = _make_command_task(trigger_heartbeat=False)

        await mgr._run_cron_task(task)

        mgr._anima.run_cron_command.assert_awaited_once()
        mgr._task_runner_supervisor.run_cron.assert_not_awaited()
        mgr._task_runner_supervisor.run_cron_followup.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tool_cron_remains_on_existing_runner_path(self, tmp_path: Path):
        mgr = _make_scheduler_mgr(tmp_path)
        task = CronTask(name="tool", schedule="*/5 * * * *", type="command", tool="internal_tool")

        await mgr._run_cron_task(task)

        mgr._task_runner_supervisor.run_cron.assert_awaited_once_with(task)
        mgr._anima.run_cron_command.assert_not_awaited()


# ── TestSkipPatternFiltering ─────────────────────────────────


class TestSkipPatternFiltering:
    """Skip-pattern semantics shared by root execution and isolated follow-ups."""

    def test_empty_stdout_is_suppressed(self):
        assert (
            command_followup_output(_make_command_task(skip_pattern=r"^\[\s*\]$"), _command_result(stdout="")) is None
        )

    def test_empty_array_matches_skip_pattern(self):
        out = command_followup_output(_make_command_task(skip_pattern=r"^\[\s*\]$"), _command_result(stdout="[]"))
        assert out is None

    def test_non_empty_array_does_not_match_skip_pattern(self):
        out = command_followup_output(
            _make_command_task(skip_pattern=r"^\[\s*\]$"),
            _command_result(stdout='[{"message_id": "123"}]'),
        )
        assert out == '[{"message_id": "123"}]'

    def test_no_skip_pattern_passes_stdout_through(self):
        out = command_followup_output(_make_command_task(skip_pattern=None), _command_result(stdout="[]"))
        assert out == "[]"

    def test_invalid_regex_continues_to_followup(self, caplog):
        with caplog.at_level(logging.WARNING):
            out = command_followup_output(
                _make_command_task(skip_pattern="[unterminated"),
                _command_result(stdout="some output"),
            )
        assert out == "some output"
        assert "Invalid skip_pattern" in caplog.text

    def test_empty_stdout_no_heartbeat(self):
        out = command_followup_output(
            _make_command_task(skip_pattern=r"^\[\s*\]$"),
            _command_result(stdout="   \n  "),
        )
        assert out is None

    def test_trigger_heartbeat_false_suppresses_heartbeat(self):
        out = command_followup_output(
            _make_command_task(trigger_heartbeat=False),
            _command_result(stdout='[{"message_id": "123"}]'),
        )
        assert out is None

    def test_trigger_heartbeat_true_allows_heartbeat(self):
        out = command_followup_output(
            _make_command_task(trigger_heartbeat=True),
            _command_result(stdout="some output"),
        )
        assert out == "some output"

    def test_trigger_heartbeat_false_takes_precedence_over_skip_pattern(self):
        out = command_followup_output(
            _make_command_task(skip_pattern=r"^\[\s*\]$", trigger_heartbeat=False),
            _command_result(stdout='[{"data": "real"}]'),
        )
        assert out is None

    def test_nonzero_exit_code_triggers_failure_review(self):
        out = command_followup_output(
            _make_command_task(skip_pattern=None),
            _command_result(stdout="error occurred", stderr="some error", exit_code=1),
        )
        assert "some error" in out


class TestRootCommandFollowup:
    @pytest.mark.asyncio
    async def test_skip_pattern_prevents_followup_runner(self, tmp_path: Path):
        mgr = _make_scheduler_mgr(tmp_path)
        task = _make_command_task(skip_pattern=r"^\[\s*\]$")
        mgr._anima.run_cron_command.return_value = _command_result(stdout="[]")

        await mgr._run_cron_task(task)

        mgr._task_runner_supervisor.run_cron_followup.assert_not_awaited()
        mgr._task_runner_supervisor.run_cron.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_trigger_heartbeat_false_suppresses_followup(self, tmp_path: Path):
        mgr = _make_scheduler_mgr(tmp_path)
        task = _make_command_task(trigger_heartbeat=False)
        mgr._anima.run_cron_command.return_value = _command_result(stdout="new data")

        await mgr._run_cron_task(task)

        mgr._task_runner_supervisor.run_cron_followup.assert_not_awaited()
        mgr._task_runner_supervisor.run_cron.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_command_failure_runs_followup_notification(self, tmp_path: Path):
        mgr = _make_scheduler_mgr(tmp_path)
        task = _make_command_task()
        mgr._anima.run_cron_command.return_value = _command_result(exit_code=2, stderr="sensor failed")

        await mgr._run_cron_task(task)

        mgr._task_runner_supervisor.run_cron_followup.assert_awaited_once()
        command_output = mgr._task_runner_supervisor.run_cron_followup.await_args.args[1]
        assert '"exit_code": 2' in command_output
        assert '"stderr": "sensor failed"' in command_output
        mgr._task_runner_supervisor.run_cron.assert_not_awaited()
        mgr._anima.memory.append_cron_event.assert_called_once_with(
            task.name,
            "failed",
            reason="execution failed",
            schedule=task.schedule,
        )
