"""Unit and E2E tests for intent-based trigger filtering.

Covers the intent filter logic in both:
- core/lifecycle.py  (_message_triggered_heartbeat)
- core/supervisor/inbox_rate_limiter.py  (message_triggered_heartbeat)

Non-actionable messages (empty intent, ack, FYI) are deferred to the scheduled
heartbeat.  Actionable messages (delegation, report, question) and human-source
messages trigger an immediate heartbeat.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config.models import AnimaWorksConfig, HeartbeatConfig
from core.lifecycle import LifecycleManager
from core.schemas import Message
from core.supervisor.inbox_rate_limiter import InboxRateLimiter
from core.supervisor.scheduler_manager import SchedulerManager


# ── Helpers ───────────────────────────────────────────────


def _default_config() -> AnimaWorksConfig:
    """Return an AnimaWorksConfig with default HeartbeatConfig.

    actionable_intents defaults to ["delegation", "report", "question"].
    """
    return AnimaWorksConfig()


def _make_message(
    *,
    intent: str = "",
    source: str = "anima",
    from_person: str = "bob",
    to_person: str = "alice",
    content: str = "hello",
) -> Message:
    """Create a Message with the given intent and source."""
    return Message(
        from_person=from_person,
        to_person=to_person,
        content=content,
        intent=intent,
        source=source,
    )


def _setup_lifecycle(messages: list[Message]) -> LifecycleManager:
    """Create a LifecycleManager with a mock anima named 'alice'.

    The anima's messenger.receive() returns *messages* and
    run_heartbeat is an AsyncMock.  'alice' is pre-added to
    _pending_triggers to mimic the real trigger path.
    """
    lm = LifecycleManager()
    dp = MagicMock()
    dp.name = "alice"
    dp.run_heartbeat = AsyncMock(return_value=MagicMock())
    dp.run_heartbeat.return_value.model_dump.return_value = {}
    dp.messenger.receive.return_value = messages
    lm.animas["alice"] = dp
    lm._pending_triggers.add("alice")
    return lm


def _make_limiter(messages: list[Message], anima_name: str = "alice") -> InboxRateLimiter:
    """Create an InboxRateLimiter with a mock anima.

    The anima's messenger.receive() returns *messages* and
    run_heartbeat is an AsyncMock.
    """
    mock_anima = MagicMock()
    mock_anima.messenger = MagicMock()
    mock_anima.messenger.receive.return_value = messages
    mock_anima._lock = asyncio.Lock()
    mock_anima.run_heartbeat = AsyncMock(return_value=MagicMock())
    mock_anima.run_heartbeat.return_value.model_dump.return_value = {}

    mock_scheduler_mgr = MagicMock(spec=SchedulerManager)
    mock_scheduler_mgr.heartbeat_running = False

    limiter = InboxRateLimiter(
        anima=mock_anima,
        anima_name=anima_name,
        shutdown_event=asyncio.Event(),
        scheduler_mgr=mock_scheduler_mgr,
    )
    limiter._pending_trigger = True  # mimic the real trigger path
    return limiter


# ══════════════════════════════════════════════════════════
# Lifecycle — _message_triggered_heartbeat
# ══════════════════════════════════════════════════════════


class TestLifecycleIntentFilter:
    """Intent filtering in LifecycleManager._message_triggered_heartbeat."""

    async def test_lifecycle_delegation_triggers_heartbeat(self):
        """Message with intent='delegation' should trigger heartbeat."""
        messages = [_make_message(intent="delegation")]
        lm = _setup_lifecycle(messages)

        with patch("core.lifecycle.load_config", return_value=_default_config()):
            await lm._message_triggered_heartbeat("alice")

        lm.animas["alice"].run_heartbeat.assert_called_once()
        assert "alice" not in lm._pending_triggers

    async def test_lifecycle_report_triggers_heartbeat(self):
        """Message with intent='report' should trigger heartbeat."""
        messages = [_make_message(intent="report")]
        lm = _setup_lifecycle(messages)

        with patch("core.lifecycle.load_config", return_value=_default_config()):
            await lm._message_triggered_heartbeat("alice")

        lm.animas["alice"].run_heartbeat.assert_called_once()
        assert "alice" not in lm._pending_triggers

    async def test_lifecycle_question_triggers_heartbeat(self):
        """Message with intent='question' should trigger heartbeat."""
        messages = [_make_message(intent="question")]
        lm = _setup_lifecycle(messages)

        with patch("core.lifecycle.load_config", return_value=_default_config()):
            await lm._message_triggered_heartbeat("alice")

        lm.animas["alice"].run_heartbeat.assert_called_once()
        assert "alice" not in lm._pending_triggers

    async def test_lifecycle_empty_intent_skips_heartbeat(self):
        """Message with intent='' should NOT trigger heartbeat (deferred)."""
        messages = [_make_message(intent="")]
        lm = _setup_lifecycle(messages)

        with patch("core.lifecycle.load_config", return_value=_default_config()):
            await lm._message_triggered_heartbeat("alice")

        lm.animas["alice"].run_heartbeat.assert_not_called()
        assert "alice" not in lm._pending_triggers

    async def test_lifecycle_human_source_always_triggers(self):
        """Message with source='human' and intent='' should trigger.

        Human messages always bypass the intent filter regardless of intent.
        """
        messages = [_make_message(intent="", source="human")]
        lm = _setup_lifecycle(messages)

        with patch("core.lifecycle.load_config", return_value=_default_config()):
            await lm._message_triggered_heartbeat("alice")

        lm.animas["alice"].run_heartbeat.assert_called_once()
        assert "alice" not in lm._pending_triggers

    async def test_lifecycle_mixed_messages_actionable_wins(self):
        """If any message has an actionable intent, heartbeat triggers."""
        messages = [
            _make_message(intent=""),              # non-actionable
            _make_message(intent="report"),         # actionable
        ]
        lm = _setup_lifecycle(messages)

        with patch("core.lifecycle.load_config", return_value=_default_config()):
            await lm._message_triggered_heartbeat("alice")

        lm.animas["alice"].run_heartbeat.assert_called_once()
        assert "alice" not in lm._pending_triggers

    async def test_lifecycle_all_ack_messages_skip(self):
        """Multiple messages all with intent='' should NOT trigger."""
        messages = [
            _make_message(intent="", from_person="bob"),
            _make_message(intent="", from_person="carol"),
            _make_message(intent="", from_person="dave"),
        ]
        lm = _setup_lifecycle(messages)

        with patch("core.lifecycle.load_config", return_value=_default_config()):
            await lm._message_triggered_heartbeat("alice")

        lm.animas["alice"].run_heartbeat.assert_not_called()
        assert "alice" not in lm._pending_triggers


# ══════════════════════════════════════════════════════════
# InboxRateLimiter — message_triggered_heartbeat
# ══════════════════════════════════════════════════════════


class TestLimiterIntentFilter:
    """Intent filtering in InboxRateLimiter.message_triggered_heartbeat."""

    async def test_limiter_delegation_triggers_heartbeat(self):
        """Message with intent='delegation' should trigger heartbeat."""
        messages = [_make_message(intent="delegation")]
        limiter = _make_limiter(messages)

        with patch(
            "core.supervisor.inbox_rate_limiter.load_config",
            return_value=_default_config(),
        ):
            await limiter.message_triggered_heartbeat()

        limiter._anima.run_heartbeat.assert_called_once()
        assert limiter._pending_trigger is False

    async def test_limiter_empty_intent_skips_heartbeat(self):
        """Message with intent='' should NOT trigger heartbeat."""
        messages = [_make_message(intent="")]
        limiter = _make_limiter(messages)

        with patch(
            "core.supervisor.inbox_rate_limiter.load_config",
            return_value=_default_config(),
        ):
            await limiter.message_triggered_heartbeat()

        limiter._anima.run_heartbeat.assert_not_called()
        assert limiter._pending_trigger is False

    async def test_limiter_human_source_always_triggers(self):
        """Message with source='human' and intent='' should trigger.

        Human messages always bypass the intent filter.
        """
        messages = [_make_message(intent="", source="human")]
        limiter = _make_limiter(messages)

        with patch(
            "core.supervisor.inbox_rate_limiter.load_config",
            return_value=_default_config(),
        ):
            await limiter.message_triggered_heartbeat()

        limiter._anima.run_heartbeat.assert_called_once()
        assert limiter._pending_trigger is False

    async def test_limiter_mixed_messages_actionable_wins(self):
        """If any message has an actionable intent, heartbeat triggers."""
        messages = [
            _make_message(intent=""),              # non-actionable
            _make_message(intent="question"),       # actionable
        ]
        limiter = _make_limiter(messages)

        with patch(
            "core.supervisor.inbox_rate_limiter.load_config",
            return_value=_default_config(),
        ):
            await limiter.message_triggered_heartbeat()

        limiter._anima.run_heartbeat.assert_called_once()
        assert limiter._pending_trigger is False

    async def test_limiter_all_ack_messages_skip(self):
        """Multiple messages all with intent='' should NOT trigger."""
        messages = [
            _make_message(intent="", from_person="bob"),
            _make_message(intent="", from_person="carol"),
        ]
        limiter = _make_limiter(messages)

        with patch(
            "core.supervisor.inbox_rate_limiter.load_config",
            return_value=_default_config(),
        ):
            await limiter.message_triggered_heartbeat()

        limiter._anima.run_heartbeat.assert_not_called()
        assert limiter._pending_trigger is False


# ══════════════════════════════════════════════════════════
# E2E-like — Scheduled heartbeat processes all intents
# ══════════════════════════════════════════════════════════


class TestScheduledHeartbeatNoIntentFilter:
    """Verify that _heartbeat_wrapper (scheduled) does NOT filter by intent.

    Scheduled heartbeats handle all messages including those deferred by the
    message-triggered intent filter.  This confirms the division of
    responsibility: message-triggered filters, scheduled processes everything.
    """

    async def test_scheduled_heartbeat_processes_all_intents(self):
        """_heartbeat_wrapper calls run_heartbeat even for empty-intent messages.

        This is the complementary E2E test: non-actionable messages skipped by
        _message_triggered_heartbeat are picked up by the scheduled heartbeat.
        """
        lm = LifecycleManager()
        dp = MagicMock()
        dp.name = "alice"
        dp.run_heartbeat = AsyncMock(return_value=MagicMock())
        dp.run_heartbeat.return_value.model_dump.return_value = {}

        # All messages have empty intent — would be skipped by message-triggered
        inbox = [
            _make_message(intent="", from_person="bob"),
            _make_message(intent="", from_person="carol"),
        ]
        dp.messenger.receive.return_value = inbox
        lm.animas["alice"] = dp

        broadcast = AsyncMock()
        lm._ws_broadcast = broadcast

        with patch("core.lifecycle.load_config", return_value=_default_config()):
            await lm._heartbeat_wrapper("alice")

        # Scheduled heartbeat should ALWAYS call run_heartbeat regardless of intent
        dp.run_heartbeat.assert_called_once()
        broadcast.assert_called_once()
