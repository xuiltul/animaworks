# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for 3-path execution separation (Heartbeat/Inbox/TaskExec).

Covers:
- 3-lock structure (_conversation_locks, _inbox_lock, _background_lock, _state_file_lock)
- Trigger-based prompt section filtering (chat/inbox/heartbeat/cron/task)
- New prompt template loading (inbox_message, task_exec, task_delegation_rules)
- Heartbeat plan-focus (no inbox processing, Observe/Plan/Reflect)
- Cron LLM sessions with heartbeat-equivalent context
- InboxRateLimiter wiring (process_inbox_message instead of run_heartbeat)
- PendingTaskExecutor LLM task dispatch
- Runner IPC handler registration (process_inbox)
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.tooling.handler import active_session_type


# ── Lock Structure ──────────────────────────────────────────


class TestThreeLockStructure:
    """Verify the 3-lock structure on DigitalAnima."""

    def test_inbox_lock_exists(self, data_dir, make_anima):
        anima_dir = make_anima("lock_test")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.ConversationMemory"):
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            assert hasattr(dp, "_inbox_lock")
            assert isinstance(dp._inbox_lock, asyncio.Lock)

    def test_state_file_lock_exists(self, data_dir, make_anima):
        import threading
        anima_dir = make_anima("lock_test2")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.ConversationMemory"):
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            assert hasattr(dp, "_state_file_lock")
            assert isinstance(dp._state_file_lock, type(threading.Lock()))

    def test_three_status_slots(self, data_dir, make_anima):
        anima_dir = make_anima("lock_test3")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.ConversationMemory"):
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            assert "inbox" in dp._status_slots
            assert "conversation" in dp._status_slots
            assert "background" in dp._status_slots

    async def test_inbox_and_conversation_concurrent(self, data_dir, make_anima):
        """_inbox_lock and _conversation_locks can both be held simultaneously."""
        anima_dir = make_anima("concurrent_test")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.ConversationMemory"):
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)

            async with dp._get_thread_lock("default"):
                acquired = dp._inbox_lock.locked() is False
                assert acquired, "_inbox_lock should be available while conversation lock is held"
                async with dp._inbox_lock:
                    assert dp._get_thread_lock("default").locked()
                    assert dp._inbox_lock.locked()

    async def test_inbox_and_background_concurrent(self, data_dir, make_anima):
        """_inbox_lock and _background_lock can both be held simultaneously."""
        anima_dir = make_anima("concurrent_test2")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.ConversationMemory"):
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)

            async with dp._background_lock:
                async with dp._inbox_lock:
                    assert dp._background_lock.locked()
                    assert dp._inbox_lock.locked()


# ── Prompt Template Loading ─────────────────────────────────


class TestPromptTemplates:
    """Verify new prompt templates load correctly."""

    def test_inbox_message_template_loads(self):
        from core.paths import load_prompt
        result = load_prompt("inbox_message", messages="test msg", task_delegation_rules="rules")
        assert "test msg" in result
        assert "rules" in result

    def test_task_exec_template_loads(self):
        from core.paths import load_prompt
        result = load_prompt(
            "task_exec",
            task_id="test-001",
            title="Test Task",
            submitted_by="heartbeat",
            description="Do something",
            context="Background info",
            acceptance_criteria="criteria",
            constraints="none",
            file_paths="core/anima.py",
        )
        assert "test-001" in result
        assert "Test Task" in result

    def test_task_delegation_rules_loads(self):
        from core.paths import load_prompt
        result = load_prompt("task_delegation_rules")
        assert "タスク委任ルール" in result
        assert "禁止パターン" in result

    def test_task_complete_notify_loads(self):
        from core.paths import load_prompt
        result = load_prompt("task_complete_notify", task_id="t1", title="Done", result_summary="OK")
        assert "t1" in result
        assert "Done" in result

    def test_heartbeat_template_has_plan_section(self):
        from core.paths import load_prompt
        result = load_prompt(
            "heartbeat",
            checklist="- check item",
            task_delegation_rules="rules here",
        )
        assert "Plan" in result
        assert "Act" not in result or "実際の作業" in result

    def test_heartbeat_checklist_no_inbox_check(self):
        from core.paths import load_prompt
        result = load_prompt("heartbeat_default_checklist")
        assert "Inboxに未読メッセージがあるか" not in result
        assert "pending/" in result


# ── Trigger-Based Prompt Filtering ──────────────────────────


class TestTriggerBasedPromptFiltering:
    """Verify build_system_prompt filters sections based on trigger."""

    def _build(self, trigger: str, anima_dir: Path) -> str:
        from core.memory.manager import MemoryManager
        from core.prompt.builder import build_system_prompt
        mm = MemoryManager(anima_dir)
        result = build_system_prompt(mm, trigger=trigger)
        return result.system_prompt

    def test_task_trigger_excludes_behavior_rules(self, data_dir, make_anima):
        anima_dir = make_anima("filter_test")
        prompt = self._build("task:test-id", anima_dir)
        assert "メタ設定" not in prompt or len(prompt) < 5000

    def test_task_trigger_shorter_than_chat(self, data_dir, make_anima):
        anima_dir = make_anima("filter_test2")
        chat_prompt = self._build("", anima_dir)
        task_prompt = self._build("task:test-id", anima_dir)
        assert len(task_prompt) < len(chat_prompt), (
            f"task prompt ({len(task_prompt)}) should be shorter than chat ({len(chat_prompt)})"
        )

    def test_inbox_trigger_shorter_than_chat(self, data_dir, make_anima):
        anima_dir = make_anima("filter_test3")
        chat_prompt = self._build("", anima_dir)
        inbox_prompt = self._build("inbox:alice", anima_dir)
        assert len(inbox_prompt) <= len(chat_prompt)

    def test_heartbeat_trigger_shorter_than_chat(self, data_dir, make_anima):
        anima_dir = make_anima("filter_test4")
        chat_prompt = self._build("", anima_dir)
        hb_prompt = self._build("heartbeat", anima_dir)
        assert len(hb_prompt) <= len(chat_prompt)

    def test_cron_trigger_shorter_than_chat(self, data_dir, make_anima):
        anima_dir = make_anima("filter_cron1")
        chat_prompt = self._build("", anima_dir)
        cron_prompt = self._build("cron:morning_plan", anima_dir)
        assert len(cron_prompt) <= len(chat_prompt), (
            f"cron prompt ({len(cron_prompt)}) should be <= chat ({len(chat_prompt)})"
        )

    def test_cron_trigger_similar_to_heartbeat(self, data_dir, make_anima):
        """Cron and heartbeat should produce similar-length prompts (same context tier)."""
        anima_dir = make_anima("filter_cron2")
        hb_prompt = self._build("heartbeat", anima_dir)
        cron_prompt = self._build("cron:task1", anima_dir)
        ratio = len(cron_prompt) / max(len(hb_prompt), 1)
        assert 0.5 < ratio < 2.0, (
            f"cron ({len(cron_prompt)}) and heartbeat ({len(hb_prompt)}) "
            f"should be similar length (ratio={ratio:.2f})"
        )

    def test_cron_trigger_excludes_emotion_metadata(self, data_dir, make_anima):
        """Cron trigger should not include emotion metadata instruction."""
        anima_dir = make_anima("filter_cron3")
        cron_prompt = self._build("cron:test", anima_dir)
        assert "表情メタデータ" not in cron_prompt

    def test_cron_trigger_longer_than_task(self, data_dir, make_anima):
        """Cron should have more context than task (minimal) trigger."""
        anima_dir = make_anima("filter_cron4")
        cron_prompt = self._build("cron:test", anima_dir)
        task_prompt = self._build("task:test-id", anima_dir)
        assert len(cron_prompt) > len(task_prompt), (
            f"cron ({len(cron_prompt)}) should be longer than task ({len(task_prompt)})"
        )


# ── Cron LLM Session ────────────────────────────────────────


class TestCronLLMSession:
    """Verify cron LLM sessions have heartbeat-equivalent context."""

    def test_build_cron_prompt_includes_description(self, data_dir, make_anima):
        """_build_cron_prompt should include the task description."""
        anima_dir = make_anima("cron_prompt1")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.ConversationMemory") as MockConv:
            MockConv.return_value.load.return_value = MagicMock(turns=[])
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            result = dp._build_cron_prompt("morning_plan", "今日のタスクを計画する")
            assert "morning_plan" in result
            assert "今日のタスクを計画する" in result

    def test_build_cron_prompt_includes_command_output(self, data_dir, make_anima):
        """_build_cron_prompt should inject command output when provided."""
        anima_dir = make_anima("cron_prompt2")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.ConversationMemory") as MockConv:
            MockConv.return_value.load.return_value = MagicMock(turns=[])
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            result = dp._build_cron_prompt(
                "log_check", "ログを確認して報告",
                command_output="ERROR: disk full",
            )
            assert "ERROR: disk full" in result
            assert "コマンド実行結果" in result

    def test_build_cron_prompt_no_output_section_without_command(self, data_dir, make_anima):
        """Without command_output, no command result section should appear."""
        anima_dir = make_anima("cron_prompt3")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.ConversationMemory") as MockConv:
            MockConv.return_value.load.return_value = MagicMock(turns=[])
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            result = dp._build_cron_prompt("weekly_review", "週次振り返り")
            assert "コマンド実行結果" not in result

    def test_build_cron_prompt_shares_background_context(self, data_dir, make_anima):
        """_build_cron_prompt should include shared background context (e.g. recovery note)."""
        anima_dir = make_anima("cron_prompt4")
        shared_dir = data_dir / "shared"

        recovery_path = anima_dir / "state" / "recovery_note.md"
        recovery_path.parent.mkdir(parents=True, exist_ok=True)
        recovery_path.write_text("前回のエラー情報", encoding="utf-8")

        with patch("core.anima.AgentCore"), \
             patch("core.anima.ConversationMemory") as MockConv:
            MockConv.return_value.load.return_value = MagicMock(turns=[])
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            result = dp._build_cron_prompt("task", "説明")
            assert "前回のエラー情報" in result
            assert not recovery_path.exists()

    async def test_run_cron_task_with_command_output(self, data_dir, make_anima):
        """run_cron_task should pass command_output to _build_cron_prompt."""
        anima_dir = make_anima("cron_exec1")
        shared_dir = data_dir / "shared"

        def _lp(name, **kw):
            if name == "fragments/command_output":
                return f"## Command Output\n\n{kw.get('output', '')}"
            return "prompt"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.ConversationMemory") as MockConv, \
             patch("core.anima.load_prompt", side_effect=_lp):
            MockConv.return_value.load.return_value = MagicMock(turns=[])
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            dp.agent._tool_handler.set_active_session_type = \
                lambda st: active_session_type.set(st)

            captured_prompt = None

            async def mock_run_cycle(prompt, trigger="manual", **kwargs):
                nonlocal captured_prompt
                captured_prompt = prompt
                from core.schemas import CycleResult
                return CycleResult(
                    trigger=trigger, action="responded",
                    summary="done", duration_ms=10,
                )

            dp.agent.run_cycle = mock_run_cycle
            await dp.run_cron_task(
                "log_check", "ログ確認",
                command_output="no errors found",
            )
            assert captured_prompt is not None
            assert "no errors found" in captured_prompt


# ── Heartbeat Plan-Focus ────────────────────────────────────


class TestHeartbeatPlanFocus:
    """Verify heartbeat no longer processes inbox and focuses on planning."""

    async def test_heartbeat_warns_on_unread(self, data_dir, make_anima, caplog):
        """Heartbeat should log a warning when unread messages exist."""
        anima_dir = make_anima("hb_plan")
        shared_dir = data_dir / "shared"

        from core.messenger import Messenger
        m = Messenger(shared_dir, "other")
        m.send("hb_plan", "Test message")

        with patch("core.anima.AgentCore"), \
             patch("core.anima.ConversationMemory") as MockConv, \
             patch("core.anima.load_prompt", return_value="prompt"):
            MockConv.return_value.load.return_value = MagicMock(turns=[])
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.replied_to = set()
            dp.agent._tool_handler.set_active_session_type = lambda st: active_session_type.set(st)

            async def mock_stream(prompt, trigger="manual", **kwargs):
                yield {"type": "cycle_done", "cycle_result": {
                    "trigger": "heartbeat", "action": "responded",
                    "summary": "HEARTBEAT_OK", "duration_ms": 50,
                }}

            dp.agent.run_cycle_streaming = mock_stream
            import logging
            with caplog.at_level(logging.WARNING):
                await dp.run_heartbeat()

            assert any("Unread messages found during heartbeat" in r.message for r in caplog.records)

    async def test_heartbeat_leaves_inbox_untouched(self, data_dir, make_anima):
        """Heartbeat should NOT archive inbox messages."""
        anima_dir = make_anima("hb_leave")
        shared_dir = data_dir / "shared"

        from core.messenger import Messenger
        m = Messenger(shared_dir, "sender")
        m.send("hb_leave", "Test message")

        inbox_dir = shared_dir / "inbox" / "hb_leave"
        assert len(list(inbox_dir.glob("*.json"))) == 1

        with patch("core.anima.AgentCore"), \
             patch("core.anima.ConversationMemory") as MockConv, \
             patch("core.anima.load_prompt", return_value="prompt"):
            MockConv.return_value.load.return_value = MagicMock(turns=[])
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.replied_to = set()
            dp.agent._tool_handler.set_active_session_type = lambda st: active_session_type.set(st)

            async def mock_stream(prompt, trigger="manual", **kwargs):
                yield {"type": "cycle_done", "cycle_result": {
                    "trigger": "heartbeat", "action": "responded",
                    "summary": "HEARTBEAT_OK", "duration_ms": 50,
                }}

            dp.agent.run_cycle_streaming = mock_stream
            await dp.run_heartbeat()

        assert len(list(inbox_dir.glob("*.json"))) == 1


# ── process_inbox_message ───────────────────────────────────


class TestProcessInboxMessage:
    """Verify process_inbox_message (Path A) works correctly."""

    async def test_no_messages_returns_idle(self, data_dir, make_anima):
        """No unread messages should return idle CycleResult."""
        anima_dir = make_anima("inbox_idle")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.ConversationMemory") as MockConv, \
             patch("core.anima.load_prompt", return_value="prompt"):
            MockConv.return_value.load.return_value = MagicMock(turns=[])
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)

            result = await dp.process_inbox_message()
            assert result.action == "idle"

    async def test_processes_and_archives_messages(self, data_dir, make_anima):
        """Messages should be processed and archived."""
        anima_dir = make_anima("inbox_proc")
        shared_dir = data_dir / "shared"

        from core.messenger import Messenger
        m = Messenger(shared_dir, "sender")
        m.send("inbox_proc", "Hello!")

        inbox_dir = shared_dir / "inbox" / "inbox_proc"
        assert len(list(inbox_dir.glob("*.json"))) == 1

        with patch("core.anima.AgentCore"), \
             patch("core.anima.ConversationMemory") as MockConv, \
             patch("core.anima.load_prompt", return_value="prompt"):
            MockConv.return_value.load.return_value = MagicMock(turns=[])
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.reset_posted_channels = MagicMock()
            dp.agent.replied_to = {"sender"}
            dp.agent._tool_handler.set_active_session_type = lambda st: active_session_type.set(st)

            async def mock_stream(prompt, trigger="manual", **kwargs):
                yield {"type": "cycle_done", "cycle_result": {
                    "trigger": trigger, "action": "responded",
                    "summary": "Replied", "duration_ms": 50,
                }}

            dp.agent.run_cycle_streaming = mock_stream
            result = await dp.process_inbox_message()

        assert result.action == "responded"
        assert len(list(inbox_dir.glob("*.json"))) == 0

    async def test_uses_inbox_lock(self, data_dir, make_anima):
        """process_inbox_message should acquire _inbox_lock when messages exist."""
        anima_dir = make_anima("inbox_lock")
        shared_dir = data_dir / "shared"

        from core.messenger import Messenger
        m = Messenger(shared_dir, "peer")
        m.send("inbox_lock", "Lock test")

        with patch("core.anima.AgentCore"), \
             patch("core.anima.ConversationMemory") as MockConv, \
             patch("core.anima.load_prompt", return_value="prompt"):
            MockConv.return_value.load.return_value = MagicMock(turns=[])
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.reset_posted_channels = MagicMock()
            dp.agent.replied_to = {"peer"}
            dp.agent._tool_handler.set_active_session_type = lambda st: active_session_type.set(st)

            async def mock_stream(prompt, trigger="manual", **kwargs):
                assert dp._inbox_lock.locked(), "_inbox_lock should be held during streaming"
                yield {"type": "cycle_done", "cycle_result": {
                    "trigger": trigger, "action": "responded",
                    "summary": "OK", "duration_ms": 50,
                }}

            dp.agent.run_cycle_streaming = mock_stream
            await dp.process_inbox_message()


# ── PendingTaskExecutor LLM ─────────────────────────────────


class TestPendingTaskExecutorLLM:
    """Verify PendingTaskExecutor handles LLM task type."""

    async def test_llm_task_dispatched(self, data_dir, make_anima):
        """task_type='llm' should call _execute_llm_task."""
        anima_dir = make_anima("exec_test")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.ConversationMemory"):
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)

        from core.supervisor.pending_executor import PendingTaskExecutor
        executor = PendingTaskExecutor(
            anima=dp,
            anima_name="exec_test",
            anima_dir=anima_dir,
            shutdown_event=asyncio.Event(),
        )
        executor._execute_llm_task = AsyncMock()

        await executor.execute_pending_task({"task_type": "llm", "task_id": "t1"})
        executor._execute_llm_task.assert_called_once()

    async def test_command_task_uses_existing_path(self, data_dir, make_anima):
        """task_type='command' should use existing BackgroundTaskManager dispatch."""
        anima_dir = make_anima("exec_test2")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.ConversationMemory"):
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            dp.agent.background_manager = MagicMock()

        from core.supervisor.pending_executor import PendingTaskExecutor
        executor = PendingTaskExecutor(
            anima=dp,
            anima_name="exec_test2",
            anima_dir=anima_dir,
            shutdown_event=asyncio.Event(),
        )

        await executor.execute_pending_task({
            "task_type": "command",
            "tool_name": "test_tool",
            "task_id": "c1",
        })
        dp.agent.background_manager.submit.assert_called_once()

    async def test_expired_llm_task_skipped(self, data_dir, make_anima):
        """LLM task older than 24h should be skipped."""
        anima_dir = make_anima("exec_test3")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.ConversationMemory"):
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)

        from core.supervisor.pending_executor import PendingTaskExecutor
        executor = PendingTaskExecutor(
            anima=dp,
            anima_name="exec_test3",
            anima_dir=anima_dir,
            shutdown_event=asyncio.Event(),
        )

        task = {
            "task_type": "llm",
            "task_id": "old-task",
            "submitted_at": "2020-01-01T00:00:00+09:00",
        }

        with patch.object(dp, "_background_lock", asyncio.Lock()):
            await executor.execute_pending_task(task)


# ── Runner IPC ──────────────────────────────────────────────


# ── primary_status / primary_task ────────────────────────────


class TestPrimaryStatusInbox:
    """Verify primary_status includes inbox slot."""

    def test_inbox_processing_shows_in_status(self, data_dir, make_anima):
        anima_dir = make_anima("status_test")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.ConversationMemory"):
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)

            assert dp.primary_status == "idle"

            dp._status_slots["inbox"] = "processing"
            assert dp.primary_status == "processing"

    def test_conversation_takes_priority_over_inbox(self, data_dir, make_anima):
        anima_dir = make_anima("status_test2")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.ConversationMemory"):
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)

            dp._status_slots["inbox"] = "processing"
            dp._status_slots["conversation"] = "chatting"
            assert dp.primary_status == "chatting"

    def test_inbox_task_shows_in_primary_task(self, data_dir, make_anima):
        anima_dir = make_anima("status_test3")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.ConversationMemory"):
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)

            dp._task_slots["inbox"] = "replying to alice"
            assert dp.primary_task == "replying to alice"


# ── State File Lock ─────────────────────────────────────────


class TestStateFileLock:
    """Verify _state_file_lock is passed to ToolHandler."""

    def test_tool_handler_receives_lock(self, data_dir, make_anima):
        """set_state_file_lock is called on ToolHandler with the lock."""
        anima_dir = make_anima("lock_pass_test")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.ConversationMemory"):
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)
            dp.agent._tool_handler.set_state_file_lock.assert_called_once_with(
                dp._state_file_lock
            )

    def test_is_state_file_detection(self, data_dir, make_anima):
        from core.tooling.handler import ToolHandler
        from core.memory.manager import MemoryManager
        anima_dir = make_anima("state_detect")
        mm = MemoryManager(anima_dir)
        handler = ToolHandler(anima_dir=anima_dir, memory=mm)
        state_dir = anima_dir / "state"
        state_dir.mkdir(exist_ok=True)
        assert handler._is_state_file(state_dir / "current_task.md")
        assert handler._is_state_file(state_dir / "pending.md")
        assert not handler._is_state_file(anima_dir / "identity.md")


# ── Wake Event ──────────────────────────────────────────────


class TestPendingExecutorWake:
    """Verify PendingTaskExecutor wake event."""

    def test_wake_event_exists(self, data_dir, make_anima):
        anima_dir = make_anima("wake_test")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.ConversationMemory"):
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)

        from core.supervisor.pending_executor import PendingTaskExecutor
        executor = PendingTaskExecutor(
            anima=dp, anima_name="wake_test",
            anima_dir=anima_dir, shutdown_event=asyncio.Event(),
        )
        assert hasattr(executor, "_wake_event")
        assert isinstance(executor._wake_event, asyncio.Event)

    def test_wake_sets_event(self, data_dir, make_anima):
        anima_dir = make_anima("wake_test2")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.ConversationMemory"):
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)

        from core.supervisor.pending_executor import PendingTaskExecutor
        executor = PendingTaskExecutor(
            anima=dp, anima_name="wake_test2",
            anima_dir=anima_dir, shutdown_event=asyncio.Event(),
        )
        assert not executor._wake_event.is_set()
        executor.wake()
        assert executor._wake_event.is_set()

    def test_trigger_calls_wake(self, data_dir, make_anima):
        """_trigger_pending_task_execution should call wake on executor."""
        anima_dir = make_anima("trigger_wake")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore"), \
             patch("core.anima.ConversationMemory"):
            from core.anima import DigitalAnima
            dp = DigitalAnima(anima_dir, shared_dir)

        pending_dir = anima_dir / "state" / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        (pending_dir / "test-task.json").write_text('{"task_type": "llm"}')

        mock_executor = MagicMock()
        dp._pending_executor = mock_executor
        dp._trigger_pending_task_execution()
        mock_executor.wake.assert_called_once()


# ── Runner IPC ──────────────────────────────────────────────


class TestRunnerIPC:
    """Verify runner has process_inbox IPC handler."""

    def test_process_inbox_handler_registered(self):
        from core.supervisor.runner import AnimaRunner
        runner = AnimaRunner.__new__(AnimaRunner)
        runner.anima = MagicMock()
        runner._streaming_handler = None
        handlers = runner._get_handler.__func__(runner, "process_inbox")
        assert handlers is not None
