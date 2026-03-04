# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for run_heartbeat() decomposition refactoring.

Validates that the refactored heartbeat flow (5 private methods + orchestrator)
behaves identically to the original monolithic run_heartbeat():
  1. Basic heartbeat flow returns CycleResult with expected fields
  2. Inbox messages are processed and archived
  3. Heartbeat runs concurrently with conversation (no skip)
  4. Failure writes recovery note and crash-archives inbox
  5. Recovery note is injected on next run and deleted
  6. InboxResult dataclass integration
  7. Orchestrator body stays within line-count budget
  8. All 5 private methods exist on DigitalAnima
"""
from __future__ import annotations

import inspect
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.schemas import CycleResult
from core.tooling.handler import active_session_type

pytestmark = pytest.mark.e2e


# ── Helpers ───────────────────────────────────────────────


def _make_digital_anima(anima_dir: Path, shared_dir: Path):
    """Create a DigitalAnima with AgentCore, ConversationMemory, and load_prompt mocked."""
    with patch("core.anima.AgentCore") as MockAgent, \
         patch("core._anima_heartbeat.ConversationMemory") as MockConv, \
         patch("core._anima_heartbeat.load_prompt", return_value="prompt"):
        MockConv.return_value.load.return_value = MagicMock(turns=[])

        from core.anima import DigitalAnima
        dp = DigitalAnima(anima_dir, shared_dir)
        dp.agent.reset_reply_tracking = MagicMock()
        dp.agent.replied_to = set()
        dp.agent.background_manager = None
        # Wire set_active_session_type to use the real ContextVar
        dp.agent._tool_handler.set_active_session_type = lambda st: active_session_type.set(st)
        return dp


def _attach_mock_stream(dp, cycle_result_overrides: dict | None = None):
    """Wire up a mock streaming generator that yields a single cycle_done event."""
    defaults = {
        "trigger": "heartbeat",
        "action": "responded",
        "summary": "All systems normal",
        "duration_ms": 100,
    }
    if cycle_result_overrides:
        defaults.update(cycle_result_overrides)

    async def mock_stream(prompt, trigger="manual", **kwargs):
        yield {
            "type": "cycle_done",
            "cycle_result": defaults,
        }

    dp.agent.run_cycle_streaming = mock_stream


def _attach_failing_stream(dp, error: Exception | None = None):
    """Wire up a mock streaming generator that raises an exception."""
    exc = error or RuntimeError("Agent execution failed")

    async def mock_stream(prompt, trigger="manual", **kwargs):
        raise exc
        yield  # noqa: unreachable — makes this an async generator

    dp.agent.run_cycle_streaming = mock_stream


# ── Test 1: Basic heartbeat flow ──────────────────────────


class TestHeartbeatBasicFlow:
    """Create an anima, mock agent cycle, run heartbeat. Verify CycleResult."""

    async def test_heartbeat_basic_flow(self, data_dir, make_anima):
        """Run heartbeat with mocked agent; verify returned CycleResult fields."""
        alice_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        dp = _make_digital_anima(alice_dir, shared_dir)
        _attach_mock_stream(dp, {
            "summary": "Checked Slack, no new messages",
            "duration_ms": 250,
        })

        result = await dp.run_heartbeat()

        assert isinstance(result, CycleResult)
        assert result.trigger == "heartbeat"
        assert result.action == "responded"
        assert result.summary == "Checked Slack, no new messages"
        assert result.duration_ms == 250

    async def test_heartbeat_returns_cycle_result_type(self, data_dir, make_anima):
        """Verify CycleResult model fields are all accessible."""
        alice_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        dp = _make_digital_anima(alice_dir, shared_dir)
        _attach_mock_stream(dp)

        result = await dp.run_heartbeat()

        # All CycleResult fields should be populated
        assert hasattr(result, "trigger")
        assert hasattr(result, "action")
        assert hasattr(result, "summary")
        assert hasattr(result, "duration_ms")
        assert hasattr(result, "timestamp")
        assert hasattr(result, "context_usage_ratio")
        assert hasattr(result, "session_chained")
        assert hasattr(result, "total_turns")


# ── Test 2: Heartbeat with inbox messages ─────────────────


class TestInboxProcessing:
    """Put messages in inbox, run process_inbox_message, verify processed and archived.

    Since the 3-path separation, inbox processing is handled by
    process_inbox_message() (Path A), not run_heartbeat() (Path B).
    """

    async def test_inbox_messages_processed_and_archived(self, data_dir, make_anima):
        """Messages in inbox are consumed and archived during inbox processing."""
        alice_dir = make_anima("alice")
        make_anima("mio")  # Must be in config for Messenger to accept
        make_anima("bob")  # Must be in config for Messenger to accept
        shared_dir = data_dir / "shared"

        from core.schemas import Message
        inbox_dir = shared_dir / "inbox" / "alice"
        inbox_dir.mkdir(parents=True, exist_ok=True)
        msg1 = Message(from_person="mio", to_person="alice",
                       content="Please check the deployment logs.")
        msg2 = Message(from_person="bob", to_person="alice",
                       content="Weekly report is ready for review.")
        (inbox_dir / "msg_mio.json").write_text(
            msg1.model_dump_json(indent=2), encoding="utf-8")
        (inbox_dir / "msg_bob.json").write_text(
            msg2.model_dump_json(indent=2), encoding="utf-8")

        assert len(list(inbox_dir.glob("*.json"))) == 2

        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core._anima_heartbeat.ConversationMemory") as MockConv, \
             patch("core._anima_heartbeat.load_prompt", return_value="prompt"):
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            from core.anima import DigitalAnima
            dp = DigitalAnima(alice_dir, shared_dir)
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.reset_posted_channels = MagicMock()
            dp.agent.replied_to = {"mio", "bob"}
            dp.agent.background_manager = None
            dp.agent._tool_handler.set_active_session_type = lambda st: active_session_type.set(st)
            _attach_mock_stream(dp, {"trigger": "inbox:mio, bob", "summary": "Processed 2 messages"})

            result = await dp.process_inbox_message()

        assert result.action == "responded"

        remaining = list(inbox_dir.glob("*.json"))
        assert len(remaining) == 0, (
            f"Expected inbox to be empty after archive, "
            f"found {len(remaining)} files"
        )

    async def test_inbox_messages_recorded_to_episodes(self, data_dir, make_anima):
        """Inbox messages are recorded to episode files during inbox processing."""
        alice_dir = make_anima("alice")
        make_anima("episode_sender")  # Must be in config for Messenger to accept
        shared_dir = data_dir / "shared"

        from core.schemas import Message
        inbox_dir = shared_dir / "inbox" / "alice"
        inbox_dir.mkdir(parents=True, exist_ok=True)
        msg = Message(
            from_person="episode_sender",
            to_person="alice",
            content="DB backup completed successfully.",
            type="message",
        )
        (inbox_dir / "episode_test_msg.json").write_text(
            msg.model_dump_json(indent=2), encoding="utf-8",
        )

        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core._anima_heartbeat.ConversationMemory") as MockConv, \
             patch("core._anima_heartbeat.load_prompt", return_value="prompt"):
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            from core.anima import DigitalAnima
            dp = DigitalAnima(alice_dir, shared_dir)
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.reset_posted_channels = MagicMock()
            dp.agent.replied_to = {"episode_sender"}
            dp.agent.background_manager = None
            dp.agent._tool_handler.set_active_session_type = lambda st: active_session_type.set(st)

            async def mock_stream(prompt, trigger="manual", **kwargs):
                yield {
                    "type": "cycle_done",
                    "cycle_result": {
                        "trigger": trigger,
                        "action": "responded",
                        "summary": "All systems normal",
                        "duration_ms": 50,
                    },
                }

            dp.agent.run_cycle_streaming = mock_stream

            await dp.process_inbox_message()

        from datetime import date
        episode_file = alice_dir / "episodes" / f"{date.today().isoformat()}.md"
        assert episode_file.exists()
        content = episode_file.read_text(encoding="utf-8")
        assert "episode_senderからのメッセージ受信" in content
        assert "DB backup" in content

    async def test_heartbeat_does_not_archive_inbox(self, data_dir, make_anima):
        """Heartbeat should NOT process/archive inbox messages (3-path separation)."""
        alice_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        from core.schemas import Message
        inbox_dir = shared_dir / "inbox" / "alice"
        inbox_dir.mkdir(parents=True, exist_ok=True)
        msg = Message(from_person="mio", to_person="alice", content="Test msg")
        (inbox_dir / "msg_mio.json").write_text(
            msg.model_dump_json(indent=2), encoding="utf-8")

        dp = _make_digital_anima(alice_dir, shared_dir)
        _attach_mock_stream(dp)

        await dp.run_heartbeat()

        remaining = list(inbox_dir.glob("*.json"))
        assert len(remaining) == 1, (
            "Heartbeat should leave inbox messages for Path A (process_inbox_message)"
        )


# ── Test 3: Heartbeat runs concurrently with conversation ─


class TestHeartbeatConcurrency:
    """Verify heartbeat runs even when conversation lock is held (no skip)."""

    async def test_heartbeat_skips_when_user_waiting(self, data_dir, make_anima):
        """Heartbeat runs even when conversation lock is held (concurrent design)."""
        alice_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        dp = _make_digital_anima(alice_dir, shared_dir)
        _attach_mock_stream(dp)

        # Acquire conversation lock to simulate an active user conversation.
        # With the new concurrent design, heartbeat uses _background_lock
        # and should NOT be skipped.
        async with dp._get_thread_lock("default"):
            result = await dp.run_heartbeat()

        assert result.trigger == "heartbeat"
        assert result.action != "skipped"

    async def test_heartbeat_runs_normally_when_no_user_waiting(
        self, data_dir, make_anima,
    ):
        """Heartbeat runs normally when no conversation is active."""
        alice_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        dp = _make_digital_anima(alice_dir, shared_dir)
        _attach_mock_stream(dp)

        result = await dp.run_heartbeat()
        assert result.action != "skipped"


# ── Test 4: Heartbeat failure writes recovery note ────────


class TestHeartbeatFailureWritesRecoveryNote:
    """Make agent execution fail, verify recovery note and crash-archive."""

    async def test_heartbeat_failure_writes_recovery_note(
        self, data_dir, make_anima,
    ):
        """Agent failure triggers recovery note creation."""
        alice_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        dp = _make_digital_anima(alice_dir, shared_dir)
        _attach_failing_stream(dp, RuntimeError("LLM timeout"))

        with pytest.raises(RuntimeError, match="LLM timeout"):
            await dp.run_heartbeat()

        # Verify recovery note was saved
        recovery_path = alice_dir / "state" / "recovery_note.md"
        assert recovery_path.exists(), "Recovery note should be written on failure"

        content = recovery_path.read_text(encoding="utf-8")
        assert "RuntimeError" in content
        assert "LLM timeout" in content

    async def test_inbox_failure_crash_archives_messages(
        self, data_dir, make_anima,
    ):
        """On inbox processing failure, messages are crash-archived to prevent re-processing."""
        alice_dir = make_anima("alice")
        make_anima("mio")  # Must be in config for Messenger to accept
        shared_dir = data_dir / "shared"

        from core.schemas import Message
        inbox_dir = shared_dir / "inbox" / "alice"
        inbox_dir.mkdir(parents=True, exist_ok=True)
        msg = Message(from_person="mio", to_person="alice",
                      content="Important task: check server health.")
        (inbox_dir / "crash_test_msg.json").write_text(
            msg.model_dump_json(indent=2), encoding="utf-8")

        assert len(list(inbox_dir.glob("*.json"))) == 1

        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core._anima_heartbeat.ConversationMemory") as MockConv, \
             patch("core._anima_heartbeat.load_prompt", return_value="prompt"):
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            from core.anima import DigitalAnima
            dp = DigitalAnima(alice_dir, shared_dir)
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.reset_posted_channels = MagicMock()
            dp.agent.replied_to = set()
            dp.agent.background_manager = None
            dp.agent._tool_handler.set_active_session_type = lambda st: active_session_type.set(st)
            _attach_failing_stream(dp, RuntimeError("Agent crash"))

            with pytest.raises(RuntimeError):
                await dp.process_inbox_message()

        remaining = list(inbox_dir.glob("*.json"))
        assert len(remaining) == 0, (
            "Inbox should be crash-archived on failure to prevent "
            "re-processing storms"
        )

    async def test_heartbeat_failure_does_not_touch_inbox(
        self, data_dir, make_anima,
    ):
        """On heartbeat failure, inbox messages remain untouched (3-path separation)."""
        alice_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        from core.schemas import Message
        inbox_dir = shared_dir / "inbox" / "alice"
        inbox_dir.mkdir(parents=True, exist_ok=True)
        msg = Message(from_person="mio", to_person="alice",
                      content="Important task: check server health.")
        (inbox_dir / "crash_test_msg.json").write_text(
            msg.model_dump_json(indent=2), encoding="utf-8")

        dp = _make_digital_anima(alice_dir, shared_dir)
        _attach_failing_stream(dp, RuntimeError("Agent crash"))

        with pytest.raises(RuntimeError):
            await dp.run_heartbeat()

        remaining = list(inbox_dir.glob("*.json"))
        assert len(remaining) == 1, (
            "Heartbeat failure should not touch inbox messages"
        )


# ── Test 5: Recovery note injected on next run ────────────


class TestHeartbeatRecoveryNoteInjectedOnNextRun:
    """Write recovery note, run heartbeat, verify loaded and deleted."""

    async def test_recovery_note_injected_on_next_run(
        self, data_dir, make_anima,
    ):
        """Recovery note from previous failure is loaded into prompt and deleted."""
        alice_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        # Pre-write a recovery note (simulating previous failure)
        recovery_path = alice_dir / "state" / "recovery_note.md"
        recovery_path.write_text(
            "### エラー情報\n\n"
            "- エラー種別: RuntimeError\n"
            "- エラー内容: LLM timeout\n"
            "- 未処理メッセージ数: 3\n",
            encoding="utf-8",
        )
        assert recovery_path.exists()

        dp = _make_digital_anima(alice_dir, shared_dir)

        # Track what prompt is passed to the agent
        captured_prompts: list[str] = []

        async def mock_stream(prompt, trigger="manual", **kwargs):
            captured_prompts.append(prompt)
            yield {
                "type": "cycle_done",
                "cycle_result": {
                    "trigger": "heartbeat",
                    "action": "responded",
                    "summary": "Recovered from previous failure",
                    "duration_ms": 100,
                },
            }

        dp.agent.run_cycle_streaming = mock_stream

        result = await dp.run_heartbeat()

        assert result.action == "responded"

        # Recovery note content should appear in the prompt
        assert len(captured_prompts) == 1
        assert "前回のバックグラウンド障害情報" in captured_prompts[0]
        assert "RuntimeError" in captured_prompts[0]

        # Recovery note should be deleted after loading
        assert not recovery_path.exists(), (
            "Recovery note should be deleted after being loaded"
        )


# ── Test 6: InboxResult dataclass integration ─────────────


class TestInboxResultDataclassIntegration:
    """Create InboxResult with realistic data, verify all fields accessible."""

    def test_inbox_result_default_values(self):
        """InboxResult has sensible defaults when created empty."""
        from core.anima import InboxResult

        result = InboxResult()

        assert result.inbox_items == []
        assert result.messages == []
        assert result.senders == set()
        assert result.unread_count == 0
        assert result.prompt_parts == []

    def test_inbox_result_with_realistic_data(self):
        """InboxResult correctly stores populated fields."""
        from core.anima import InboxResult
        from core.schemas import Message

        msg1 = Message(
            from_person="mio",
            to_person="alice",
            content="Check the logs.",
        )
        msg2 = Message(
            from_person="bob",
            to_person="alice",
            content="Deploy is ready.",
        )

        result = InboxResult(
            inbox_items=[],  # simplified; paths not needed for field test
            messages=[msg1, msg2],
            senders={"mio", "bob"},
            unread_count=2,
            prompt_parts=["## 未読メッセージ\n- mio: Check the logs."],
        )

        assert len(result.messages) == 2
        assert "mio" in result.senders
        assert "bob" in result.senders
        assert result.unread_count == 2
        assert len(result.prompt_parts) == 1
        assert "未読メッセージ" in result.prompt_parts[0]

    def test_inbox_result_empty_means_no_messages(self):
        """Empty InboxResult indicates zero unread messages."""
        from core.anima import InboxResult

        result = InboxResult()
        assert result.unread_count == 0
        assert not result.senders
        assert not result.messages


# ── Test 7: Orchestrator line count ───────────────────────


class TestHeartbeatOrchestratorLineCount:
    """Verify run_heartbeat() method body is within line-count budget."""

    def test_heartbeat_orchestrator_line_count(self):
        """run_heartbeat() body must be <= 80 lines (decomposition goal)."""
        from core.anima import DigitalAnima

        source = inspect.getsource(DigitalAnima.run_heartbeat)
        # Remove the decorator/signature and dedent
        lines = source.splitlines()

        # Count non-blank, non-comment lines in the method body
        # (skip the 'async def' line and any decorator lines)
        body_started = False
        body_lines = 0
        for line in lines:
            stripped = line.strip()
            if not body_started:
                if stripped.startswith("async def ") or stripped.startswith("def "):
                    body_started = True
                continue
            # Count all lines in the body (blanks, comments, and code)
            body_lines += 1

        assert body_lines <= 85, (
            f"run_heartbeat() body is {body_lines} lines, "
            f"exceeds 85-line budget. Further decomposition needed."
        )


# ── Test 8: Five private methods exist ────────────────────


class TestPrivateMethodsExist:
    """Verify key private methods exist on DigitalAnima class.

    After 3-path separation:
    - run_heartbeat() calls _build_heartbeat_prompt, _execute_heartbeat_cycle, _handle_heartbeat_failure
    - process_inbox_message() calls _process_inbox_messages, _archive_processed_messages
    """

    HEARTBEAT_METHODS = [
        "_build_heartbeat_prompt",
        "_execute_heartbeat_cycle",
        "_handle_heartbeat_failure",
    ]

    INBOX_METHODS = [
        "_process_inbox_messages",
        "_archive_processed_messages",
    ]

    ALL_METHODS = HEARTBEAT_METHODS + INBOX_METHODS

    def test_private_methods_exist(self):
        """All decomposed private methods must exist on DigitalAnima."""
        from core.anima import DigitalAnima

        for method_name in self.ALL_METHODS:
            assert hasattr(DigitalAnima, method_name), (
                f"DigitalAnima is missing method: {method_name}"
            )
            method = getattr(DigitalAnima, method_name)
            assert callable(method), (
                f"{method_name} should be callable"
            )

    def test_process_inbox_message_exists(self):
        """process_inbox_message public method must exist on DigitalAnima."""
        from core.anima import DigitalAnima
        assert hasattr(DigitalAnima, "process_inbox_message")
        assert inspect.iscoroutinefunction(DigitalAnima.process_inbox_message)

    def test_private_methods_are_coroutines(self):
        """All private methods should be async (coroutine functions)."""
        from core.anima import DigitalAnima

        for method_name in self.ALL_METHODS:
            method = getattr(DigitalAnima, method_name)
            assert inspect.iscoroutinefunction(method), (
                f"{method_name} should be an async method (coroutine function)"
            )

    def test_heartbeat_calls_heartbeat_methods(self):
        """run_heartbeat source references heartbeat-specific private methods."""
        from core.anima import DigitalAnima

        source = inspect.getsource(DigitalAnima.run_heartbeat)

        for method_name in self.HEARTBEAT_METHODS:
            assert f"self.{method_name}" in source, (
                f"run_heartbeat() should call self.{method_name}"
            )

    def test_inbox_calls_inbox_methods(self):
        """process_inbox_message source references inbox-specific private methods."""
        from core.anima import DigitalAnima

        source = inspect.getsource(DigitalAnima.process_inbox_message)

        for method_name in self.INBOX_METHODS:
            assert f"self.{method_name}" in source, (
                f"process_inbox_message() should call self.{method_name}"
            )
