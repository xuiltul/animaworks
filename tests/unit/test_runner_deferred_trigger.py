"""Unit tests for deferred trigger mechanism in InboxRateLimiter.

Verifies that messages arriving during cooldown or lock-held states
are guaranteed to be processed via deferred timer scheduling.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.supervisor.inbox_rate_limiter import InboxRateLimiter
from core.supervisor.scheduler_manager import SchedulerManager


def _make_limiter(tmp_path: Path) -> InboxRateLimiter:
    """Create an InboxRateLimiter with minimal dependencies."""
    mock_anima = MagicMock()
    mock_anima.messenger = MagicMock()
    mock_anima._lock = asyncio.Lock()

    mock_scheduler_mgr = MagicMock(spec=SchedulerManager)
    mock_scheduler_mgr.heartbeat_running = False

    return InboxRateLimiter(
        anima=mock_anima,
        anima_name="defer-test",
        shutdown_event=asyncio.Event(),
        scheduler_mgr=mock_scheduler_mgr,
    )


# ── schedule_deferred_trigger ──────────────────────────


class TestScheduleDeferredTrigger:
    """Verify deferred timer scheduling behavior."""

    def test_initial_deferred_timer_is_none(self, tmp_path):
        """InboxRateLimiter initializes with _deferred_timer=None."""
        limiter = _make_limiter(tmp_path)
        assert limiter._deferred_timer is None

    def test_schedules_timer(self, tmp_path):
        """schedule_deferred_trigger creates a timer handle."""
        limiter = _make_limiter(tmp_path)
        loop = asyncio.new_event_loop()
        try:
            with patch("asyncio.get_running_loop", return_value=loop):
                limiter.schedule_deferred_trigger()
            assert limiter._deferred_timer is not None
            limiter._deferred_timer.cancel()
        finally:
            loop.close()

    def test_noop_when_already_scheduled(self, tmp_path):
        """Second call is a no-op when timer already exists."""
        limiter = _make_limiter(tmp_path)
        sentinel = MagicMock()
        limiter._deferred_timer = sentinel

        limiter.schedule_deferred_trigger()
        # Timer should still be the same sentinel object
        assert limiter._deferred_timer is sentinel


# ── try_deferred_trigger ───────────────────────────────


class TestTryDeferredTrigger:
    """Verify deferred trigger execution logic."""

    @pytest.mark.asyncio
    async def test_triggers_heartbeat_when_ready(self, tmp_path):
        """Fires heartbeat when not in cooldown and lock not held."""
        limiter = _make_limiter(tmp_path)
        limiter._anima.messenger.has_unread.return_value = True
        limiter._anima._lock = MagicMock()
        limiter._anima._lock.locked.return_value = False
        limiter._deferred_timer = MagicMock()

        with patch.object(limiter, "is_in_cooldown", return_value=False), \
             patch.object(limiter, "message_triggered_heartbeat", new_callable=AsyncMock):
            await limiter.try_deferred_trigger()
            assert limiter._pending_trigger is True
            assert limiter._deferred_timer is None

    @pytest.mark.asyncio
    async def test_reschedules_when_in_cooldown(self, tmp_path):
        """Re-schedules if still in cooldown."""
        limiter = _make_limiter(tmp_path)
        limiter._anima.messenger.has_unread.return_value = True
        limiter._deferred_timer = MagicMock()

        with patch.object(limiter, "is_in_cooldown", return_value=True), \
             patch.object(limiter, "schedule_deferred_trigger") as mock_sched:
            await limiter.try_deferred_trigger()
            mock_sched.assert_called_once()
            assert limiter._pending_trigger is False

    @pytest.mark.asyncio
    async def test_reschedules_when_lock_held(self, tmp_path):
        """Re-schedules if anima lock is held."""
        limiter = _make_limiter(tmp_path)
        limiter._anima.messenger.has_unread.return_value = True
        limiter._anima._lock = MagicMock()
        limiter._anima._lock.locked.return_value = True
        limiter._deferred_timer = MagicMock()

        with patch.object(limiter, "is_in_cooldown", return_value=False), \
             patch.object(limiter, "schedule_deferred_trigger") as mock_sched:
            await limiter.try_deferred_trigger()
            mock_sched.assert_called_once()

    @pytest.mark.asyncio
    async def test_noop_when_no_unread(self, tmp_path):
        """Does nothing if inbox is empty."""
        limiter = _make_limiter(tmp_path)
        limiter._anima.messenger.has_unread.return_value = False
        limiter._deferred_timer = MagicMock()

        await limiter.try_deferred_trigger()
        assert limiter._pending_trigger is False
        assert limiter._deferred_timer is None

    @pytest.mark.asyncio
    async def test_noop_when_pending_trigger(self, tmp_path):
        """Does nothing if trigger already pending."""
        limiter = _make_limiter(tmp_path)
        limiter._anima.messenger.has_unread.return_value = True
        limiter._deferred_timer = MagicMock()
        limiter._pending_trigger = True

        await limiter.try_deferred_trigger()

    @pytest.mark.asyncio
    async def test_clears_timer_on_entry(self, tmp_path):
        """Timer reference is cleared at the start of execution."""
        limiter = _make_limiter(tmp_path)
        limiter._anima.messenger.has_unread.return_value = False
        limiter._deferred_timer = MagicMock()

        await limiter.try_deferred_trigger()
        assert limiter._deferred_timer is None


# ── on_anima_lock_released ────────────────────────────


class TestOnAnimaLockReleased:
    """Verify lock-released callback behavior."""

    @pytest.mark.asyncio
    async def test_schedules_deferred_when_unread(self, tmp_path):
        """Schedules deferred trigger when unread messages exist."""
        limiter = _make_limiter(tmp_path)
        limiter._anima.messenger.has_unread.return_value = True

        with patch.object(limiter, "schedule_deferred_trigger") as mock_sched:
            await limiter.on_anima_lock_released()
            mock_sched.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_action_when_no_unread(self, tmp_path):
        """Does nothing when no unread messages."""
        limiter = _make_limiter(tmp_path)
        limiter._anima.messenger.has_unread.return_value = False

        with patch.object(limiter, "schedule_deferred_trigger") as mock_sched:
            await limiter.on_anima_lock_released()
            mock_sched.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_action_when_pending_trigger(self, tmp_path):
        """Does nothing when trigger already pending."""
        limiter = _make_limiter(tmp_path)
        limiter._anima.messenger.has_unread.return_value = True
        limiter._pending_trigger = True

        with patch.object(limiter, "schedule_deferred_trigger") as mock_sched:
            await limiter.on_anima_lock_released()
            mock_sched.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_action_when_no_anima(self, tmp_path):
        """Does nothing when anima is None."""
        limiter = _make_limiter(tmp_path)
        limiter._anima = None

        # Should not raise
        await limiter.on_anima_lock_released()


# ── Cleanup ────────────────────────────────────────────


class TestDeferredTimerCleanup:
    """Verify timer is cleaned up properly."""

    def test_cancel_deferred_timer(self, tmp_path):
        """cancel_deferred_timer cancels and clears the timer."""
        limiter = _make_limiter(tmp_path)
        mock_timer = MagicMock()
        limiter._deferred_timer = mock_timer

        limiter.cancel_deferred_timer()
        mock_timer.cancel.assert_called_once()
        assert limiter._deferred_timer is None

    def test_cancel_noop_when_no_timer(self, tmp_path):
        """cancel_deferred_timer handles None timer gracefully."""
        limiter = _make_limiter(tmp_path)
        assert limiter._deferred_timer is None

        # Should not raise
        limiter.cancel_deferred_timer()


class TestAnimaRunnerCleanupDelegation:
    """Verify AnimaRunner._cleanup delegates to InboxRateLimiter."""

    @pytest.mark.asyncio
    async def test_cleanup_cancels_timer_via_limiter(self, tmp_path):
        """AnimaRunner._cleanup calls inbox_limiter.cancel_deferred_timer()."""
        from core.supervisor.runner import AnimaRunner

        animas_dir = tmp_path / "animas"
        animas_dir.mkdir(exist_ok=True)
        (animas_dir / "defer-test").mkdir(exist_ok=True)
        (animas_dir / "defer-test" / "identity.md").write_text("test")
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir(exist_ok=True)

        runner = AnimaRunner(
            anima_name="defer-test",
            socket_path=tmp_path / "test.sock",
            animas_dir=animas_dir,
            shared_dir=shared_dir,
        )

        mock_limiter = MagicMock(spec=InboxRateLimiter)
        runner._inbox_limiter = mock_limiter

        await runner._cleanup()
        mock_limiter.cancel_deferred_timer.assert_called_once()
