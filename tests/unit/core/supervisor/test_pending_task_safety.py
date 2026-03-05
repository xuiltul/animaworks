# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for pending task failure safety — file move lifecycle and failure notifications.

Validates:
- File lifecycle: pending/ → processing/ → success: delete | fail: failed/
- _execute_llm_task writes FAILED result and sends reply_to notification on error
- Orphaned processing/ files are recovered to failed/ on startup
- i18n template for task_fail_notify
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.supervisor.pending_executor import PendingTaskExecutor


# ── Helpers ──────────────────────────────────────────────────


def _make_executor(tmp_path: Path) -> PendingTaskExecutor:
    """Create a PendingTaskExecutor with mocked anima."""
    anima_dir = tmp_path / "animas" / "test-anima"
    anima_dir.mkdir(parents=True, exist_ok=True)

    mock_anima = MagicMock()
    mock_anima.agent.background_manager = MagicMock()
    mock_anima._background_lock = asyncio.Lock()
    mock_anima._status_slots = {"background": "idle"}
    mock_anima._task_slots = {"background": ""}
    mock_anima.messenger = MagicMock()
    mock_anima._task_semaphore = None

    return PendingTaskExecutor(
        anima=mock_anima,
        anima_name="test-anima",
        anima_dir=anima_dir,
        shutdown_event=asyncio.Event(),
    )


def _stop_after_first(executor: PendingTaskExecutor):
    """Return a mock for asyncio.wait_for that stops the loop after one iteration."""
    async def _mock(coro, *, timeout):
        executor._shutdown_event.set()
        raise asyncio.TimeoutError
    return _mock


# ── TestCommandPendingFileLifecycle ──────────────────────────


class TestCommandPendingFileLifecycle:
    """Command-type pending tasks use processing/ → failed/ lifecycle."""

    @pytest.mark.asyncio
    async def test_success_removes_processing_file(self, tmp_path: Path) -> None:
        """On success, processing/ file is deleted."""
        executor = _make_executor(tmp_path)
        pending_dir = executor._anima_dir / "state" / "background_tasks" / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)

        task = {"task_id": "cmd-1", "tool_name": "test_tool", "subcommand": "", "raw_args": []}
        (pending_dir / "cmd-1.json").write_text(json.dumps(task))

        with patch("core.supervisor.pending_executor.asyncio.wait_for",
                    side_effect=_stop_after_first(executor)):
            await executor.watcher_loop()

        assert not (pending_dir / "cmd-1.json").exists()
        processing_dir = pending_dir / "processing"
        assert not (processing_dir / "cmd-1.json").exists()

    @pytest.mark.asyncio
    async def test_failure_moves_to_failed(self, tmp_path: Path) -> None:
        """On execution failure, file is moved from processing/ to failed/."""
        executor = _make_executor(tmp_path)
        pending_dir = executor._anima_dir / "state" / "background_tasks" / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)

        task = {"task_id": "cmd-fail", "tool_name": "bad_tool", "subcommand": "", "raw_args": []}
        (pending_dir / "cmd-fail.json").write_text(json.dumps(task))

        original_execute = executor.execute_pending_task

        async def failing_execute(task_desc):
            raise RuntimeError("Simulated failure")

        executor.execute_pending_task = failing_execute  # type: ignore[assignment]

        with patch("core.supervisor.pending_executor.asyncio.wait_for",
                    side_effect=_stop_after_first(executor)):
            await executor.watcher_loop()

        failed_dir = pending_dir / "failed"
        assert (failed_dir / "cmd-fail.json").exists()
        assert not (pending_dir / "cmd-fail.json").exists()

    @pytest.mark.asyncio
    async def test_invalid_json_still_deleted(self, tmp_path: Path) -> None:
        """Invalid JSON files are deleted (not moved)."""
        executor = _make_executor(tmp_path)
        pending_dir = executor._anima_dir / "state" / "background_tasks" / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)

        (pending_dir / "bad.json").write_text("{invalid")

        with patch("core.supervisor.pending_executor.asyncio.wait_for",
                    side_effect=_stop_after_first(executor)):
            await executor.watcher_loop()

        assert not (pending_dir / "bad.json").exists()


# ── TestLLMPendingFileLifecycle ──────────────────────────────


class TestLLMPendingFileLifecycle:
    """LLM-type pending tasks use processing/ → failed/ lifecycle."""

    @pytest.mark.asyncio
    async def test_success_removes_processing_file(self, tmp_path: Path) -> None:
        """On success, LLM pending processing/ file is deleted."""
        executor = _make_executor(tmp_path)
        llm_dir = executor._anima_dir / "state" / "pending"
        llm_dir.mkdir(parents=True, exist_ok=True)

        task = {"task_type": "llm", "task_id": "llm-1", "description": "test"}
        (llm_dir / "llm-1.json").write_text(json.dumps(task))

        with patch.object(executor, "execute_pending_task", new_callable=AsyncMock):
            with patch("core.supervisor.pending_executor.asyncio.wait_for",
                        side_effect=_stop_after_first(executor)):
                await executor.watcher_loop()

        assert not (llm_dir / "llm-1.json").exists()
        assert not (llm_dir / "processing" / "llm-1.json").exists()

    @pytest.mark.asyncio
    async def test_failure_moves_to_failed(self, tmp_path: Path) -> None:
        """On LLM execution failure, file moves to failed/."""
        executor = _make_executor(tmp_path)
        llm_dir = executor._anima_dir / "state" / "pending"
        llm_dir.mkdir(parents=True, exist_ok=True)

        task = {"task_type": "llm", "task_id": "llm-fail", "description": "failing task"}
        (llm_dir / "llm-fail.json").write_text(json.dumps(task))

        async def failing_execute(task_desc):
            raise RuntimeError("LLM failure")

        executor.execute_pending_task = failing_execute  # type: ignore[assignment]

        with patch("core.supervisor.pending_executor.asyncio.wait_for",
                    side_effect=_stop_after_first(executor)):
            await executor.watcher_loop()

        failed_dir = llm_dir / "failed"
        assert (failed_dir / "llm-fail.json").exists()
        assert not (llm_dir / "llm-fail.json").exists()


# ── TestRecoverProcessing ────────────────────────────────────


class TestRecoverProcessing:
    """Orphaned files in processing/ are moved to failed/ on startup."""

    def test_recovers_orphaned_files(self, tmp_path: Path) -> None:
        processing_dir = tmp_path / "processing"
        processing_dir.mkdir()
        failed_dir = tmp_path / "failed"
        failed_dir.mkdir()

        (processing_dir / "orphan1.json").write_text('{"task_id":"o1"}')
        (processing_dir / "orphan2.json").write_text('{"task_id":"o2"}')

        PendingTaskExecutor._recover_processing(processing_dir, failed_dir)

        assert not list(processing_dir.glob("*.json"))
        assert (failed_dir / "orphan1.json").exists()
        assert (failed_dir / "orphan2.json").exists()

    def test_no_op_when_processing_dir_missing(self, tmp_path: Path) -> None:
        processing_dir = tmp_path / "processing"
        failed_dir = tmp_path / "failed"
        failed_dir.mkdir()
        PendingTaskExecutor._recover_processing(processing_dir, failed_dir)

    @pytest.mark.asyncio
    async def test_watcher_loop_recovers_on_startup(self, tmp_path: Path) -> None:
        """watcher_loop recovers processing/ orphans before entering main loop."""
        executor = _make_executor(tmp_path)

        cmd_processing = executor._anima_dir / "state" / "background_tasks" / "pending" / "processing"
        cmd_processing.mkdir(parents=True)
        cmd_failed = executor._anima_dir / "state" / "background_tasks" / "pending" / "failed"
        cmd_failed.mkdir(parents=True)
        (cmd_processing / "orphan-cmd.json").write_text('{"task_id":"oc"}')

        llm_processing = executor._anima_dir / "state" / "pending" / "processing"
        llm_processing.mkdir(parents=True)
        llm_failed = executor._anima_dir / "state" / "pending" / "failed"
        llm_failed.mkdir(parents=True)
        (llm_processing / "orphan-llm.json").write_text('{"task_id":"ol"}')

        with patch("core.supervisor.pending_executor.asyncio.wait_for",
                    side_effect=_stop_after_first(executor)):
            await executor.watcher_loop()

        assert (cmd_failed / "orphan-cmd.json").exists()
        assert (llm_failed / "orphan-llm.json").exists()
        assert not list(cmd_processing.glob("*.json"))
        assert not list(llm_processing.glob("*.json"))


# ── TestExecuteLLMTaskFailureHandling ────────────────────────


class TestExecuteLLMTaskFailureHandling:
    """_execute_llm_task writes FAILED result and notifies reply_to on error."""

    @pytest.mark.asyncio
    async def test_writes_failed_result(self, tmp_path: Path) -> None:
        """_execute_llm_task calls _write_failed_result on exception."""
        executor = _make_executor(tmp_path)

        with patch.object(executor, "_run_llm_task", side_effect=RuntimeError("boom")):
            task_desc = {"task_id": "fail-1", "description": "test task"}
            await executor._execute_llm_task(task_desc)

        result_path = executor._anima_dir / "state" / "task_results" / "fail-1.md"
        assert result_path.exists()
        content = result_path.read_text()
        assert content.startswith("FAILED:")
        assert "RuntimeError" in content

    @pytest.mark.asyncio
    async def test_sends_reply_to_notification_dict(self, tmp_path: Path) -> None:
        """When reply_to is a dict with 'name', sends failure notification."""
        executor = _make_executor(tmp_path)

        with patch.object(executor, "_run_llm_task", side_effect=RuntimeError("oops")):
            with patch("core.i18n.t", return_value="failure msg"):
                task_desc = {
                    "task_id": "fail-notify",
                    "description": "failing task",
                    "reply_to": {"name": "manager-anima", "content": "please notify"},
                }
                await executor._execute_llm_task(task_desc)

        executor._anima.messenger.send.assert_called_once()
        call_kwargs = executor._anima.messenger.send.call_args
        assert call_kwargs[1]["to"] == "manager-anima"

    @pytest.mark.asyncio
    async def test_sends_reply_to_notification_string(self, tmp_path: Path) -> None:
        """When reply_to is a string, uses it as the recipient."""
        executor = _make_executor(tmp_path)

        with patch.object(executor, "_run_llm_task", side_effect=RuntimeError("oops")):
            with patch("core.i18n.t", return_value="failure msg"):
                task_desc = {
                    "task_id": "fail-str",
                    "description": "task",
                    "reply_to": "some-anima",
                }
                await executor._execute_llm_task(task_desc)

        executor._anima.messenger.send.assert_called_once()
        call_kwargs = executor._anima.messenger.send.call_args
        assert call_kwargs[1]["to"] == "some-anima"

    @pytest.mark.asyncio
    async def test_no_notification_without_reply_to(self, tmp_path: Path) -> None:
        """When no reply_to, only writes failed result, no notification."""
        executor = _make_executor(tmp_path)

        with patch.object(executor, "_run_llm_task", side_effect=RuntimeError("fail")):
            task_desc = {"task_id": "fail-noreply", "description": "test"}
            await executor._execute_llm_task(task_desc)

        executor._anima.messenger.send.assert_not_called()
        result_path = executor._anima_dir / "state" / "task_results" / "fail-noreply.md"
        assert result_path.exists()

    @pytest.mark.asyncio
    async def test_notification_failure_does_not_propagate(self, tmp_path: Path) -> None:
        """If messenger.send fails, the exception is swallowed."""
        executor = _make_executor(tmp_path)
        executor._anima.messenger.send.side_effect = ConnectionError("network error")

        with patch.object(executor, "_run_llm_task", side_effect=RuntimeError("boom")):
            with patch("core.i18n.t", return_value="msg"):
                task_desc = {
                    "task_id": "fail-notif-err",
                    "description": "task",
                    "reply_to": {"name": "boss"},
                }
                await executor._execute_llm_task(task_desc)

        result_path = executor._anima_dir / "state" / "task_results" / "fail-notif-err.md"
        assert result_path.exists()

    @pytest.mark.asyncio
    async def test_status_slots_reset_on_failure(self, tmp_path: Path) -> None:
        """Status slots are reset to idle even on failure."""
        executor = _make_executor(tmp_path)

        with patch.object(executor, "_run_llm_task", side_effect=RuntimeError("err")):
            task_desc = {"task_id": "slot-test", "description": "test"}
            await executor._execute_llm_task(task_desc)

        assert executor._anima._status_slots["background"] == "idle"
        assert executor._anima._task_slots["background"] == ""


# ── TestI18nTemplate ─────────────────────────────────────────


class TestI18nTemplate:
    """Verify task_fail_notify i18n template exists and formats correctly."""

    def test_ja_template(self) -> None:
        from core.i18n import t
        result = t(
            "pending_executor.task_fail_notify",
            locale="ja",
            task_id="test-123",
            title="テストタスク",
            error="RuntimeError: boom",
        )
        assert "test-123" in result
        assert "テストタスク" in result
        assert "RuntimeError: boom" in result

    def test_en_template(self) -> None:
        from core.i18n import t
        result = t(
            "pending_executor.task_fail_notify",
            locale="en",
            task_id="test-456",
            title="Test Task",
            error="ValueError: bad",
        )
        assert "test-456" in result
        assert "Test Task" in result
        assert "ValueError: bad" in result
