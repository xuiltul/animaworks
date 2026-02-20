# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for run_heartbeat() decomposition refactoring.

Validates that the refactored heartbeat flow (5 private methods + orchestrator)
behaves identically to the original monolithic run_heartbeat():
  1. Basic heartbeat flow returns CycleResult with expected fields
  2. Inbox messages are processed and archived
  3. Heartbeat skips when user is waiting
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

pytestmark = pytest.mark.e2e


# ── Helpers ───────────────────────────────────────────────


def _make_digital_anima(anima_dir: Path, shared_dir: Path):
    """Create a DigitalAnima with AgentCore, ConversationMemory, and load_prompt mocked."""
    with patch("core.anima.AgentCore") as MockAgent, \
         patch("core.anima.ConversationMemory") as MockConv, \
         patch("core.anima.load_prompt", return_value="prompt"):
        MockConv.return_value.load.return_value = MagicMock(turns=[])

        from core.anima import DigitalAnima
        dp = DigitalAnima(anima_dir, shared_dir)
        dp.agent.reset_reply_tracking = MagicMock()
        dp.agent.replied_to = set()
        dp.agent.background_manager = None
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

    async def mock_stream(prompt, trigger="manual"):
        yield {
            "type": "cycle_done",
            "cycle_result": defaults,
        }

    dp.agent.run_cycle_streaming = mock_stream


def _attach_failing_stream(dp, error: Exception | None = None):
    """Wire up a mock streaming generator that raises an exception."""
    exc = error or RuntimeError("Agent execution failed")

    async def mock_stream(prompt, trigger="manual"):
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


class TestHeartbeatWithInboxMessages:
    """Put messages in inbox, run heartbeat, verify processed and archived."""

    async def test_heartbeat_with_inbox_messages(self, data_dir, make_anima):
        """Messages in inbox are consumed and archived during heartbeat."""
        alice_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        # Write messages directly to inbox (bypass CascadeLimiter singleton)
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

        # Verify messages are in inbox before heartbeat
        inbox_dir = shared_dir / "inbox" / "alice"
        assert len(list(inbox_dir.glob("*.json"))) == 2

        dp = _make_digital_anima(alice_dir, shared_dir)
        # Mark both senders as replied to so messages get archived
        dp.agent.replied_to = {"mio", "bob"}
        _attach_mock_stream(dp, {"summary": "Processed 2 messages"})

        result = await dp.run_heartbeat()

        assert result.trigger == "heartbeat"
        assert result.action == "responded"

        # After heartbeat, inbox should be empty (messages archived)
        remaining = list(inbox_dir.glob("*.json"))
        assert len(remaining) == 0, (
            f"Expected inbox to be empty after archive, "
            f"found {len(remaining)} files"
        )

    async def test_inbox_messages_recorded_to_episodes(self, data_dir, make_anima):
        """Inbox messages are recorded to episode files during heartbeat."""
        alice_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        # Write message directly to inbox (bypass CascadeLimiter singleton)
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

        # Keep patches active during heartbeat execution
        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core.anima.ConversationMemory") as MockConv, \
             patch("core.anima.load_prompt", return_value="prompt"):
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            from core.anima import DigitalAnima
            dp = DigitalAnima(alice_dir, shared_dir)
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.replied_to = {"episode_sender"}
            dp.agent.background_manager = None

            async def mock_stream(prompt, trigger="manual"):
                yield {
                    "type": "cycle_done",
                    "cycle_result": {
                        "trigger": "heartbeat",
                        "action": "responded",
                        "summary": "All systems normal",
                        "duration_ms": 50,
                    },
                }

            dp.agent.run_cycle_streaming = mock_stream

            await dp.run_heartbeat()

        from datetime import date
        episode_file = alice_dir / "episodes" / f"{date.today().isoformat()}.md"
        assert episode_file.exists()
        content = episode_file.read_text(encoding="utf-8")
        assert "episode_senderからのメッセージ受信" in content
        assert "DB backup" in content


# ── Test 3: Heartbeat skips when user waiting ─────────────


class TestHeartbeatSkipsWhenUserWaiting:
    """Set _user_waiting event, verify heartbeat returns skipped result."""

    async def test_heartbeat_skips_when_user_waiting(self, data_dir, make_anima):
        """Heartbeat defers to user messages when _user_waiting is set."""
        alice_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        dp = _make_digital_anima(alice_dir, shared_dir)
        _attach_mock_stream(dp)

        # Simulate a user waiting for the lock
        dp._user_waiting.set()

        result = await dp.run_heartbeat()

        assert result.trigger == "heartbeat"
        assert result.action == "skipped"
        assert "User message priority" in result.summary

    async def test_heartbeat_runs_normally_when_no_user_waiting(
        self, data_dir, make_anima,
    ):
        """Heartbeat runs normally when _user_waiting is NOT set."""
        alice_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        dp = _make_digital_anima(alice_dir, shared_dir)
        _attach_mock_stream(dp)

        # Default state: _user_waiting is NOT set
        assert not dp._user_waiting.is_set()

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

    async def test_heartbeat_failure_crash_archives_inbox(
        self, data_dir, make_anima,
    ):
        """On failure, inbox messages are crash-archived to prevent re-processing."""
        alice_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        # Write message directly to inbox (bypass CascadeLimiter singleton)
        from core.schemas import Message
        inbox_dir = shared_dir / "inbox" / "alice"
        inbox_dir.mkdir(parents=True, exist_ok=True)
        msg = Message(from_person="mio", to_person="alice",
                      content="Important task: check server health.")
        (inbox_dir / "crash_test_msg.json").write_text(
            msg.model_dump_json(indent=2), encoding="utf-8")

        inbox_dir = shared_dir / "inbox" / "alice"
        assert len(list(inbox_dir.glob("*.json"))) == 1

        dp = _make_digital_anima(alice_dir, shared_dir)
        _attach_failing_stream(dp, RuntimeError("Agent crash"))

        with pytest.raises(RuntimeError):
            await dp.run_heartbeat()

        # Inbox should be empty after crash-archive
        remaining = list(inbox_dir.glob("*.json"))
        assert len(remaining) == 0, (
            "Inbox should be crash-archived on failure to prevent "
            "re-processing storms"
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

        async def mock_stream(prompt, trigger="manual"):
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
        assert "前回のハートビート障害情報" in captured_prompts[0]
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

        assert body_lines <= 80, (
            f"run_heartbeat() body is {body_lines} lines, "
            f"exceeds 80-line budget. Further decomposition needed."
        )


# ── Test 8: Five private methods exist ────────────────────


class TestFivePrivateMethodsExist:
    """Verify all 5 private methods exist on DigitalAnima class."""

    EXPECTED_METHODS = [
        "_build_heartbeat_prompt",
        "_process_inbox_messages",
        "_execute_heartbeat_cycle",
        "_archive_processed_messages",
        "_handle_heartbeat_failure",
    ]

    def test_five_private_methods_exist(self):
        """All 5 decomposed private methods must exist on DigitalAnima."""
        from core.anima import DigitalAnima

        for method_name in self.EXPECTED_METHODS:
            assert hasattr(DigitalAnima, method_name), (
                f"DigitalAnima is missing method: {method_name}"
            )
            method = getattr(DigitalAnima, method_name)
            assert callable(method), (
                f"{method_name} should be callable"
            )

    def test_private_methods_are_coroutines(self):
        """All 5 private methods should be async (coroutine functions)."""
        from core.anima import DigitalAnima

        for method_name in self.EXPECTED_METHODS:
            method = getattr(DigitalAnima, method_name)
            assert inspect.iscoroutinefunction(method), (
                f"{method_name} should be an async method (coroutine function)"
            )

    def test_orchestrator_calls_private_methods(self):
        """run_heartbeat source references all 5 private methods."""
        from core.anima import DigitalAnima

        source = inspect.getsource(DigitalAnima.run_heartbeat)

        for method_name in self.EXPECTED_METHODS:
            assert f"self.{method_name}" in source, (
                f"run_heartbeat() should call self.{method_name}"
            )
