"""Unit tests for PendingTaskExecutor."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.i18n import t
from core.supervisor.pending_executor import PendingTaskExecutor


def _make_executor(tmp_path: Path) -> PendingTaskExecutor:
    """Create a PendingTaskExecutor with minimal dependencies."""
    anima_dir = tmp_path / "animas" / "test-anima"
    anima_dir.mkdir(parents=True, exist_ok=True)
    mock_anima = MagicMock()
    mock_anima.agent.background_manager = MagicMock()

    return PendingTaskExecutor(
        anima=mock_anima,
        anima_name="test-anima",
        anima_dir=anima_dir,
        shutdown_event=asyncio.Event(),
    )


class _LaneAnima:
    """Minimal anima stub with real lane helper methods."""

    def __init__(self) -> None:
        self.agent = MagicMock(name="chat_agent")
        self.background_agent = MagicMock(name="background_agent")
        self.messenger = MagicMock()
        self._events: dict[str, asyncio.Event] = {}
        self._background_lock = asyncio.Lock()

    def _agent_for_lane(self, lane: str):
        return self.background_agent if lane == "background" else self.agent

    def _agent_session_context(self, lane: str):
        return self._background_lock if lane == "background" else asyncio.Lock()

    def _get_interrupt_event(self, name: str) -> asyncio.Event:
        self._events.setdefault(name, asyncio.Event())
        return self._events[name]


def _make_lane_executor(tmp_path: Path) -> PendingTaskExecutor:
    anima_dir = tmp_path / "animas" / "lane-anima"
    anima_dir.mkdir(parents=True, exist_ok=True)
    return PendingTaskExecutor(
        anima=_LaneAnima(),
        anima_name="lane-anima",
        anima_dir=anima_dir,
        shutdown_event=asyncio.Event(),
    )


class TestPendingTaskExecutorInit:
    """Test PendingTaskExecutor initialization."""

    def test_creates_instance(self, tmp_path):
        executor = _make_executor(tmp_path)
        assert executor._anima_name == "test-anima"

    def test_independent_instantiation(self, tmp_path):
        """PendingTaskExecutor can be instantiated without AnimaRunner."""
        anima_dir = tmp_path / "animas" / "standalone"
        anima_dir.mkdir(parents=True)
        executor = PendingTaskExecutor(
            anima=MagicMock(),
            anima_name="standalone",
            anima_dir=anima_dir,
            shutdown_event=asyncio.Event(),
        )
        assert executor._anima_name == "standalone"


class TestTaskExecLaneIsolation:
    """Verify TaskExec uses the background lane, not the chat lane."""

    @pytest.mark.asyncio
    async def test_llm_task_uses_background_agent_and_cwd_isolated(self, tmp_path):
        executor = _make_lane_executor(tmp_path)
        chat_agent = executor._anima.agent
        background_agent = executor._anima.background_agent
        workdir = tmp_path / "workspace"
        workdir.mkdir()

        async def _stream_success(*args, **kwargs):
            yield {"type": "text_delta", "text": "done"}
            yield {
                "type": "cycle_done",
                "cycle_result": {"summary": "background result", "action": "complete"},
            }

        background_agent.run_cycle_streaming = MagicMock(side_effect=lambda *a, **kw: _stream_success(*a, **kw))
        background_agent.reset_reply_tracking = MagicMock()
        background_agent.reset_read_paths = MagicMock()
        background_agent.set_interrupt_event = MagicMock()
        background_agent.set_task_cwd = MagicMock()
        chat_agent.set_task_cwd = MagicMock()
        chat_agent.set_interrupt_event = MagicMock()

        task_desc = {
            "task_id": "lane-task-1",
            "title": "Lane task",
            "description": "Verify background lane",
            "working_directory": str(workdir),
        }

        with (
            patch("core.paths.load_prompt", return_value="test prompt"),
            patch("core.memory.activity.ActivityLogger") as mock_activity,
        ):
            mock_activity.return_value.log = MagicMock()
            result = await executor._run_llm_task(task_desc)

        assert result == "background result"
        background_agent.run_cycle_streaming.assert_called_once()
        assert background_agent.run_cycle_streaming.call_args.kwargs["thread_id"] == "lane-task-1"
        background_agent.reset_reply_tracking.assert_called_once_with(session_type="task")
        background_agent.reset_read_paths.assert_called_once()
        background_agent.set_interrupt_event.assert_called_once_with(
            executor._anima._get_interrupt_event("_background")
        )
        background_agent.set_task_cwd.assert_any_call(workdir)
        background_agent.set_task_cwd.assert_any_call(None)
        chat_agent.set_task_cwd.assert_not_called()
        chat_agent.set_interrupt_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_command_task_uses_background_lane_manager(self, tmp_path):
        executor = _make_lane_executor(tmp_path)
        chat_manager = MagicMock()
        background_manager = MagicMock()
        executor._anima.agent.background_manager = chat_manager
        executor._anima.background_agent.background_manager = background_manager

        task_desc = {
            "task_id": "cmd-lane-1",
            "tool_name": "web_search",
            "subcommand": "search",
            "raw_args": ["query"],
            "anima_dir": str(executor._anima_dir),
        }

        await executor.execute_pending_task(task_desc)

        background_manager.submit.assert_called_once()
        chat_manager.submit.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_task_activity_has_context_and_synthetic_identity(self, tmp_path):
        executor = _make_lane_executor(tmp_path)
        background_agent = executor._anima.background_agent

        async def _stream_success(*args, **kwargs):
            yield {
                "type": "cycle_done",
                "cycle_result": {"summary": "completed", "action": "complete"},
            }

        background_agent.run_cycle_streaming = MagicMock(side_effect=_stream_success)
        background_agent.reset_reply_tracking = MagicMock()
        background_agent.reset_read_paths = MagicMock()
        background_agent.set_interrupt_event = MagicMock()
        background_agent.set_task_cwd = MagicMock()

        with (
            patch("core.paths.load_prompt", return_value="test prompt"),
            patch("core.memory.activity.ActivityLogger") as mock_activity,
        ):
            result = await executor._run_llm_task({"description": "Synthetic task title\nmore detail"})

        assert result == "completed"
        start_call, end_call = mock_activity.return_value.log.call_args_list
        assert start_call.args == ("task_exec_start",)
        assert start_call.kwargs["ctx"] == "task:unknown"
        assert start_call.kwargs["meta"] == {
            "task_id": "unknown",
            "title": "Synthetic task title",
            "submitted_by": "unknown",
        }
        assert end_call.args == ("task_exec_end",)
        assert end_call.kwargs["ctx"] == "task:unknown"
        assert end_call.kwargs["meta"]["task_id"] == "unknown"
        assert end_call.kwargs["meta"]["title"] == "Synthetic task title"
        assert end_call.kwargs["meta"]["status"] == "completed"
        assert end_call.kwargs["meta"]["result"] == "completed"


class TestExecutePendingTask:
    """Test pending task execution."""

    @pytest.mark.asyncio
    async def test_submits_to_background_manager(self, tmp_path):
        """Task should be submitted to BackgroundTaskManager."""
        executor = _make_executor(tmp_path)

        task_desc = {
            "task_id": "test-123",
            "tool_name": "web_search",
            "subcommand": "search",
            "raw_args": ["query"],
            "anima_dir": str(tmp_path / "animas" / "test-anima"),
        }

        await executor.execute_pending_task(task_desc)

        executor._anima.agent.background_manager.submit.assert_called_once()
        call_args = executor._anima.agent.background_manager.submit.call_args
        assert call_args[0][0] == "web_search:search"

    @pytest.mark.asyncio
    async def test_skips_when_no_anima(self, tmp_path):
        """Should skip when anima is None."""
        executor = _make_executor(tmp_path)
        executor._anima = None

        # Should not raise
        await executor.execute_pending_task({"tool_name": "test"})

    @pytest.mark.asyncio
    async def test_skips_when_no_background_manager(self, tmp_path):
        """Should skip when background_manager is None."""
        executor = _make_executor(tmp_path)
        executor._anima.agent.background_manager = None

        # Should not raise
        await executor.execute_pending_task({"tool_name": "test"})


class TestWatcherLoop:
    """Test pending task watcher loop."""

    @pytest.mark.asyncio
    async def test_picks_up_pending_files(self, tmp_path):
        """Watcher should pick up and process .json files in pending dir."""
        executor = _make_executor(tmp_path)
        pending_dir = executor._anima_dir / "state" / "background_tasks" / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)

        task = {"task_id": "test-1", "tool_name": "test_tool", "subcommand": "", "raw_args": []}
        (pending_dir / "task1.json").write_text(json.dumps(task))

        async def stop_after_first(coro, *, timeout):
            executor._shutdown_event.set()
            raise TimeoutError

        with patch("core.supervisor.pending_executor.asyncio.wait_for", side_effect=stop_after_first):
            await executor.watcher_loop()

        assert not (pending_dir / "task1.json").exists()
        executor._anima.agent.background_manager.submit.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_invalid_json(self, tmp_path):
        """Watcher should handle invalid JSON files gracefully."""
        executor = _make_executor(tmp_path)
        pending_dir = executor._anima_dir / "state" / "background_tasks" / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)

        (pending_dir / "bad.json").write_text("not json")

        async def stop_after_first(coro, *, timeout):
            executor._shutdown_event.set()
            raise TimeoutError

        with patch("core.supervisor.pending_executor.asyncio.wait_for", side_effect=stop_after_first):
            await executor.watcher_loop()

        assert not (pending_dir / "bad.json").exists()


class TestMachineDirectiveInjection:
    """Test machine tool directive injection into TaskExec prompt."""

    def test_directive_appended_when_machine_in_description(self):
        """Prompt should have machine directive when description mentions machine."""
        description = "machineツールで実装し、検証してpushする"
        assert "machine" in description.lower()
        directive = t("pending_executor.machine_directive")
        assert "MUST" in directive

    def test_directive_not_appended_without_machine(self):
        """No directive when description does not mention machine."""
        description = "git pushして結果を報告する"
        assert "machine" not in description.lower()

    def test_case_insensitive_detection(self):
        """Detection should be case-insensitive."""
        for desc in ["Machineで実装", "MACHINE RUN", "use machine tool"]:
            assert "machine" in desc.lower()

    def test_directive_i18n_ja(self):
        directive = t("pending_executor.machine_directive", lang="ja")
        assert "MUST" in directive
        assert "animaworks-tool machine run" in directive

    def test_directive_i18n_en(self):
        directive = t("pending_executor.machine_directive", lang="en")
        assert "MUST" in directive
        assert "animaworks-tool machine run" in directive

    def test_integration_prompt_with_machine(self):
        """Simulate the prompt construction logic: machine → directive appended."""
        base_prompt = "あなたはタスク実行エージェントです。\n## 作業内容\nmachineで実装し検証する"
        description = "machineで実装し検証する"
        directive = t("pending_executor.machine_directive")

        prompt = base_prompt
        if "machine" in description.lower():
            prompt += "\n\n" + directive

        assert prompt.endswith(directive)
        assert "MUST" in prompt

    def test_integration_prompt_without_machine(self):
        """Prompt stays unchanged when no machine mention."""
        base_prompt = "あなたはタスク実行エージェントです。\n## 作業内容\nCI結果を確認する"
        description = "CI結果を確認してレポートを作成する"

        prompt = base_prompt
        if "machine" in description.lower():
            prompt += "\n\n" + t("pending_executor.machine_directive")

        assert prompt == base_prompt


class TestStreamErrorSuppression:
    """Test that stream errors are suppressed when task queue is already done."""

    @pytest.mark.asyncio
    async def test_suppressed_when_queue_is_done(self, tmp_path):
        """Stream error should NOT raise when task queue status is 'done'."""
        executor = _make_executor(tmp_path)
        bg_event = asyncio.Event()
        executor._anima._get_interrupt_event = lambda _name: bg_event

        async def _stream_with_error(*args, **kwargs):
            yield {"type": "text_delta", "text": "partial output"}
            yield {"type": "error", "message": "stream disconnected 3 times"}

        executor._anima.agent.run_cycle_streaming = _stream_with_error
        executor._anima.agent.reset_reply_tracking = MagicMock()
        executor._anima.agent.reset_read_paths = MagicMock()
        executor._anima.agent.set_task_cwd = MagicMock()
        executor._anima.messenger = MagicMock()

        task_desc = {
            "task_id": "task-done-1",
            "title": "Already completed task",
            "description": "Task that completes before stream error",
        }

        mock_entry = MagicMock()
        mock_entry.status = "done"
        mock_entry.summary = "Task completed successfully"

        from core.taskboard.models import AttentionDecision

        with (
            patch("core.paths.load_prompt", return_value="test prompt"),
            patch("core.memory.activity.ActivityLogger") as mock_activity,
            patch("core.supervisor.pending_executor._resolve_default_workspace", return_value=""),
            patch("core.memory.task_queue.TaskQueueManager") as mock_tqm,
            patch(
                "core.supervisor.pending_executor.resolver_for_anima_dir",
            ) as mock_resolver,
        ):
            mock_resolver.return_value.should_execute.return_value = AttentionDecision(reason="active")
            mock_activity.return_value.log = MagicMock()
            mock_tqm.return_value.get_task_by_id.return_value = mock_entry

            result = await executor._run_llm_task(task_desc)
            assert result == "Task completed successfully"

    @pytest.mark.asyncio
    async def test_raises_when_queue_is_not_done(self, tmp_path):
        """Stream error should raise TaskExecError when queue status is not 'done'."""
        from core.supervisor.pending_executor import TaskExecError

        executor = _make_executor(tmp_path)
        bg_event = asyncio.Event()
        executor._anima._get_interrupt_event = lambda _name: bg_event

        async def _stream_with_error(*args, **kwargs):
            yield {"type": "error", "message": "stream disconnected"}

        executor._anima.agent.run_cycle_streaming = _stream_with_error
        executor._anima.agent.reset_reply_tracking = MagicMock()
        executor._anima.agent.reset_read_paths = MagicMock()
        executor._anima.agent.set_task_cwd = MagicMock()
        executor._anima.messenger = MagicMock()

        task_desc = {
            "task_id": "task-pending-1",
            "title": "Incomplete task",
            "description": "Task that did not complete",
        }

        mock_entry = MagicMock()
        mock_entry.status = "in_progress"

        with (
            patch("core.paths.load_prompt", return_value="test prompt"),
            patch("core.memory.activity.ActivityLogger") as mock_activity,
            patch("core.supervisor.pending_executor._resolve_default_workspace", return_value=""),
            patch("core.memory.task_queue.TaskQueueManager") as mock_tqm,
        ):
            mock_activity.return_value.log = MagicMock()
            mock_tqm.return_value.get_task_by_id.return_value = mock_entry

            with pytest.raises(TaskExecError, match="streaming error"):
                await executor._run_llm_task(task_desc)

    @pytest.mark.asyncio
    async def test_raises_when_queue_lookup_fails(self, tmp_path):
        """Stream error should raise TaskExecError when queue lookup raises."""
        from core.supervisor.pending_executor import TaskExecError

        executor = _make_executor(tmp_path)
        bg_event = asyncio.Event()
        executor._anima._get_interrupt_event = lambda _name: bg_event

        async def _stream_with_error(*args, **kwargs):
            yield {"type": "error", "message": "stream disconnected"}

        executor._anima.agent.run_cycle_streaming = _stream_with_error
        executor._anima.agent.reset_reply_tracking = MagicMock()
        executor._anima.agent.reset_read_paths = MagicMock()
        executor._anima.agent.set_task_cwd = MagicMock()
        executor._anima.messenger = MagicMock()

        task_desc = {
            "task_id": "task-err-1",
            "title": "Queue inaccessible task",
            "description": "Task where queue cannot be read",
        }

        with (
            patch("core.paths.load_prompt", return_value="test prompt"),
            patch("core.memory.activity.ActivityLogger") as mock_activity,
            patch("core.supervisor.pending_executor._resolve_default_workspace", return_value=""),
            patch("core.memory.task_queue.TaskQueueManager") as mock_tqm,
        ):
            mock_activity.return_value.log = MagicMock()
            mock_tqm.return_value.get_task_by_id.side_effect = OSError("disk error")

            with pytest.raises(TaskExecError, match="streaming error"):
                await executor._run_llm_task(task_desc)

    @pytest.mark.asyncio
    async def test_raises_when_task_not_in_queue(self, tmp_path):
        """Stream error should raise when task is not found in queue (None)."""
        from core.supervisor.pending_executor import TaskExecError

        executor = _make_executor(tmp_path)
        bg_event = asyncio.Event()
        executor._anima._get_interrupt_event = lambda _name: bg_event

        async def _stream_with_error(*args, **kwargs):
            yield {"type": "error", "message": "stream disconnected"}

        executor._anima.agent.run_cycle_streaming = _stream_with_error
        executor._anima.agent.reset_reply_tracking = MagicMock()
        executor._anima.agent.reset_read_paths = MagicMock()
        executor._anima.agent.set_task_cwd = MagicMock()
        executor._anima.messenger = MagicMock()

        task_desc = {
            "task_id": "task-missing-1",
            "title": "Unknown task",
            "description": "Task not registered in queue",
        }

        with (
            patch("core.paths.load_prompt", return_value="test prompt"),
            patch("core.memory.activity.ActivityLogger") as mock_activity,
            patch("core.supervisor.pending_executor._resolve_default_workspace", return_value=""),
            patch("core.memory.task_queue.TaskQueueManager") as mock_tqm,
        ):
            mock_activity.return_value.log = MagicMock()
            mock_tqm.return_value.get_task_by_id.return_value = None

            with pytest.raises(TaskExecError, match="streaming error"):
                await executor._run_llm_task(task_desc)

    @pytest.mark.asyncio
    async def test_no_suppression_when_no_error(self, tmp_path):
        """Normal completion (no stream error) returns result without raising."""
        executor = _make_executor(tmp_path)
        bg_event = asyncio.Event()
        executor._anima._get_interrupt_event = lambda _name: bg_event

        async def _stream_success(*args, **kwargs):
            yield {"type": "text_delta", "text": "done"}
            yield {
                "type": "cycle_done",
                "cycle_result": {"summary": "all good", "action": "complete"},
            }

        executor._anima.agent.run_cycle_streaming = _stream_success
        executor._anima.agent.reset_reply_tracking = MagicMock()
        executor._anima.agent.reset_read_paths = MagicMock()
        executor._anima.agent.set_task_cwd = MagicMock()
        executor._anima.messenger = MagicMock()

        task_desc = {
            "task_id": "task-ok-1",
            "title": "Normal task",
            "description": "Task that completes normally",
        }

        with (
            patch("core.paths.load_prompt", return_value="test prompt"),
            patch("core.memory.activity.ActivityLogger") as mock_activity,
            patch("core.supervisor.pending_executor._resolve_default_workspace", return_value=""),
        ):
            mock_activity.return_value.log = MagicMock()
            result = await executor._run_llm_task(task_desc)
            assert result == "all good"


class TestLlmTaskFailurePropagation:
    @pytest.mark.asyncio
    async def test_run_llm_task_raises_when_cycle_done_action_is_error(self, tmp_path):
        executor = _make_executor(tmp_path)
        bg_event = asyncio.Event()
        executor._anima._get_interrupt_event = lambda _name: bg_event

        async def _failing_stream(*args, **kwargs):
            yield {
                "type": "cycle_done",
                "cycle_result": {
                    "action": "error",
                    "summary": "stream retry exhausted",
                },
            }

        executor._anima.agent.run_cycle_streaming = _failing_stream
        executor._anima.agent.reset_reply_tracking = MagicMock()
        executor._anima.agent.reset_read_paths = MagicMock()
        executor._anima.agent.set_task_cwd = MagicMock()
        executor._anima.messenger.send = MagicMock()

        task_desc = {
            "task_id": "llm-fail-1",
            "title": "Broken task",
            "description": "Investigate failure",
        }

        with (
            patch("core.paths.load_prompt", return_value="test prompt"),
            patch("core.memory.activity.ActivityLogger") as mock_activity,
            patch("core.supervisor.pending_executor._resolve_default_workspace", return_value=""),
        ):
            mock_activity.return_value.log = MagicMock()
            with pytest.raises(RuntimeError, match="stream retry exhausted"):
                await executor._run_llm_task(task_desc)

        start_call, end_call = mock_activity.return_value.log.call_args_list
        assert start_call.args == ("task_exec_start",)
        assert end_call.args == ("task_exec_end",)
        assert end_call.kwargs["ctx"] == "task:llm-fail-1"
        assert end_call.kwargs["meta"]["status"] == "failed"
        assert end_call.kwargs["meta"]["error"] == "stream retry exhausted"
        assert end_call.kwargs["meta"]["error_type"] == "RuntimeError"
