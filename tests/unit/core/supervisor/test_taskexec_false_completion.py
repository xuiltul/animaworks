# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for TaskExec false-completion fixes (GH #145).

Covers:
- Bug A: cancelled/expired sentinel → correct queue status
- Bug B: error chunk detection → TaskExecError
- Bug B+: retry_start resets had_error
- Bug C: serial batch failed_dependency → queue sync
- _classify_task_result helper
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.supervisor.pending_executor import (
    PendingTaskExecutor,
    TaskExecError,
    _SENTINEL_CANCELLED,
    _SENTINEL_EXPIRED,
    _classify_task_result,
)


# ── Helpers ──────────────────────────────────────────────


def _make_executor(tmp_path: Path) -> PendingTaskExecutor:
    anima_dir = tmp_path / "animas" / "test-anima"
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "state").mkdir(parents=True, exist_ok=True)
    mock_anima = MagicMock()
    mock_anima.agent.background_manager = MagicMock()
    mock_anima._background_lock = asyncio.Lock()
    mock_anima._status_slots = {"background": "idle"}
    mock_anima._task_slots = {"background": ""}
    mock_anima._active_parallel_tasks = {}
    return PendingTaskExecutor(
        anima=mock_anima,
        anima_name="test-anima",
        anima_dir=anima_dir,
        shutdown_event=asyncio.Event(),
    )


def _make_task_desc(**overrides) -> dict:
    base = {
        "task_id": "test-task-1",
        "title": "Test Task",
        "description": "Do something",
        "context": "",
        "acceptance_criteria": [],
        "constraints": [],
        "file_paths": [],
        "reply_to": None,
        "submitted_by": "unit-test",
        "submitted_at": "",
    }
    base.update(overrides)
    return base


async def _fake_streaming(*chunks):
    """Create an async generator that yields the given chunks."""

    async def _stream(prompt, trigger, **kw):
        for c in chunks:
            yield c

    return _stream


# ── _classify_task_result ──────────────────────────────────


class TestClassifyTaskResult:
    def test_cancelled(self):
        status, summary = _classify_task_result(_SENTINEL_CANCELLED)
        assert status == "cancelled"
        assert "cancelled" in summary.lower()

    def test_expired(self):
        status, summary = _classify_task_result(_SENTINEL_EXPIRED)
        assert status == "cancelled"
        assert "expired" in summary.lower()

    def test_normal_result(self):
        status, summary = _classify_task_result("Task completed successfully")
        assert status == "done"
        assert summary == "Task completed successfully"

    def test_empty_result(self):
        status, summary = _classify_task_result("")
        assert status == "done"
        assert summary == ""

    def test_long_result_truncated(self):
        long_text = "x" * 500
        status, summary = _classify_task_result(long_text)
        assert status == "done"
        assert len(summary) == 200


# ── Bug B: error chunk detection ──────────────────────────


class TestRunLlmTaskErrorDetection:
    @pytest.mark.asyncio
    async def test_error_chunk_raises_taskexec_error(self, tmp_path):
        """error chunk in stream → TaskExecError raised."""
        executor = _make_executor(tmp_path)
        task = _make_task_desc()

        chunks = [
            {"type": "text_delta", "text": "partial output"},
            {"type": "error", "message": "Agent SDK timeout"},
            {"type": "cycle_done", "cycle_result": {"summary": "partial output"}},
        ]

        async def fake_stream(prompt, trigger, **kw):
            for c in chunks:
                yield c

        executor._anima.agent.run_cycle_streaming = fake_stream
        executor._anima.agent.reset_reply_tracking = MagicMock()
        executor._anima.agent.reset_read_paths = MagicMock()
        executor._anima.agent.set_task_cwd = MagicMock()
        executor._anima.agent.set_interrupt_event = MagicMock()

        with patch("core.paths.load_prompt", return_value="prompt"), \
             patch("core.memory.activity.ActivityLogger"), \
             patch("core.memory.streaming_journal.StreamingJournal"):
            with pytest.raises(TaskExecError, match="Agent SDK timeout"):
                await executor._run_llm_task(task)

    @pytest.mark.asyncio
    async def test_retry_start_resets_error(self, tmp_path):
        """error → retry_start → cycle_done should NOT raise."""
        executor = _make_executor(tmp_path)
        task = _make_task_desc()

        chunks = [
            {"type": "text_delta", "text": "start"},
            {"type": "error", "message": "transient error"},
            {"type": "retry_start", "retry": 1, "max_retries": 3},
            {"type": "text_delta", "text": "recovered output"},
            {"type": "cycle_done", "cycle_result": {"summary": "recovered output"}},
        ]

        async def fake_stream(prompt, trigger, **kw):
            for c in chunks:
                yield c

        executor._anima.agent.run_cycle_streaming = fake_stream
        executor._anima.agent.reset_reply_tracking = MagicMock()
        executor._anima.agent.reset_read_paths = MagicMock()
        executor._anima.agent.set_task_cwd = MagicMock()
        executor._anima.agent.set_interrupt_event = MagicMock()

        with patch("core.paths.load_prompt", return_value="prompt"), \
             patch("core.memory.activity.ActivityLogger"), \
             patch("core.memory.streaming_journal.StreamingJournal"):
            result = await executor._run_llm_task(task)
            assert "recovered" in result

    @pytest.mark.asyncio
    async def test_normal_cycle_no_error(self, tmp_path):
        """Normal stream without error chunks → returns summary."""
        executor = _make_executor(tmp_path)
        task = _make_task_desc()

        chunks = [
            {"type": "text_delta", "text": "all good"},
            {"type": "cycle_done", "cycle_result": {"summary": "all good"}},
        ]

        async def fake_stream(prompt, trigger, **kw):
            for c in chunks:
                yield c

        executor._anima.agent.run_cycle_streaming = fake_stream
        executor._anima.agent.reset_reply_tracking = MagicMock()
        executor._anima.agent.reset_read_paths = MagicMock()
        executor._anima.agent.set_task_cwd = MagicMock()
        executor._anima.agent.set_interrupt_event = MagicMock()

        with patch("core.paths.load_prompt", return_value="prompt"), \
             patch("core.memory.activity.ActivityLogger"), \
             patch("core.memory.streaming_journal.StreamingJournal"):
            result = await executor._run_llm_task(task)
            assert result == "all good"


# ── Bug A: cancelled/expired in _execute_llm_task ──────────


class TestExecuteLlmTaskStatusMapping:
    @pytest.mark.asyncio
    async def test_cancelled_maps_to_cancelled_status(self, tmp_path):
        executor = _make_executor(tmp_path)
        task = _make_task_desc()

        with patch.object(executor, "_run_llm_task", return_value=_SENTINEL_CANCELLED), \
             patch.object(executor, "_sync_task_queue") as mock_sync:
            await executor._execute_llm_task(task)
            mock_sync.assert_called_once_with(
                "test-task-1", "cancelled", summary="cancelled before execution"
            )

    @pytest.mark.asyncio
    async def test_expired_maps_to_cancelled_status(self, tmp_path):
        executor = _make_executor(tmp_path)
        task = _make_task_desc()

        with patch.object(executor, "_run_llm_task", return_value=_SENTINEL_EXPIRED), \
             patch.object(executor, "_sync_task_queue") as mock_sync:
            await executor._execute_llm_task(task)
            mock_sync.assert_called_once_with(
                "test-task-1", "cancelled", summary="expired (TTL exceeded)"
            )

    @pytest.mark.asyncio
    async def test_normal_result_maps_to_done(self, tmp_path):
        executor = _make_executor(tmp_path)
        task = _make_task_desc()

        with patch.object(executor, "_run_llm_task", return_value="success"), \
             patch.object(executor, "_sync_task_queue") as mock_sync:
            await executor._execute_llm_task(task)
            mock_sync.assert_called_once_with(
                "test-task-1", "done", summary="success"
            )

    @pytest.mark.asyncio
    async def test_error_exception_maps_to_failed(self, tmp_path):
        executor = _make_executor(tmp_path)
        task = _make_task_desc()

        with patch.object(
            executor, "_run_llm_task", side_effect=TaskExecError("boom")
        ), patch.object(executor, "_sync_task_queue") as mock_sync, \
             patch.object(executor, "_write_failed_result"):
            await executor._execute_llm_task(task)
            mock_sync.assert_called_once()
            assert mock_sync.call_args[0][1] == "failed"


# ── Bug C: serial batch failed_dependency queue sync ────────


class TestSerialBatchFailedDependency:
    @pytest.mark.asyncio
    async def test_failed_dependency_syncs_to_queue(self, tmp_path):
        """Serial batch failed_dependency should call _sync_task_queue with 'failed'."""
        executor = _make_executor(tmp_path)

        tasks = [
            {"task_id": "dep1", "description": "dep", "depends_on": [], "parallel": False},
            {"task_id": "child1", "description": "child", "depends_on": ["dep1"], "parallel": False},
        ]

        with patch.object(executor, "_run_llm_task", side_effect=RuntimeError("dep failed")), \
             patch.object(executor, "_sync_task_queue") as mock_sync, \
             patch.object(executor, "_write_failed_result"), \
             patch.object(executor, "_get_semaphore", return_value=asyncio.Lock()):
            await executor._dispatch_batch("test-batch", tasks)

            sync_calls = {call[0][0]: call[0][1] for call in mock_sync.call_args_list}
            assert "dep1" in sync_calls
            assert sync_calls["dep1"] == "failed"
            assert "child1" in sync_calls
            assert sync_calls["child1"] == "failed"
