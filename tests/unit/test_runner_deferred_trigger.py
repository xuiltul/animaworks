"""Unit tests for deferred trigger mechanism in AnimaRunner.

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


def _make_runner(tmp_path: Path):
    """Create an AnimaRunner with minimal filesystem dependencies."""
    from core.supervisor.runner import AnimaRunner

    animas_dir = tmp_path / "animas"
    animas_dir.mkdir(exist_ok=True)
    anima_dir = animas_dir / "defer-test"
    anima_dir.mkdir(exist_ok=True)
    (anima_dir / "identity.md").write_text("test identity")
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(exist_ok=True)
    socket_path = tmp_path / "test.sock"

    return AnimaRunner(
        anima_name="defer-test",
        socket_path=socket_path,
        animas_dir=animas_dir,
        shared_dir=shared_dir,
    )


# ── _schedule_deferred_trigger ──────────────────────────


class TestScheduleDeferredTrigger:
    """Verify deferred timer scheduling behavior."""

    def test_initial_deferred_timer_is_none(self, tmp_path):
        """AnimaRunner initializes with _deferred_timer=None."""
        runner = _make_runner(tmp_path)
        assert runner._deferred_timer is None

    def test_schedules_timer(self, tmp_path):
        """_schedule_deferred_trigger creates a timer handle."""
        runner = _make_runner(tmp_path)
        loop = asyncio.new_event_loop()
        try:
            with patch("asyncio.get_running_loop", return_value=loop):
                runner._schedule_deferred_trigger()
            assert runner._deferred_timer is not None
            runner._deferred_timer.cancel()
        finally:
            loop.close()

    def test_noop_when_already_scheduled(self, tmp_path):
        """Second call is a no-op when timer already exists."""
        runner = _make_runner(tmp_path)
        sentinel = MagicMock()
        runner._deferred_timer = sentinel

        runner._schedule_deferred_trigger()
        # Timer should still be the same sentinel object
        assert runner._deferred_timer is sentinel


# ── _try_deferred_trigger ───────────────────────────────


class TestTryDeferredTrigger:
    """Verify deferred trigger execution logic."""

    @pytest.mark.asyncio
    async def test_triggers_heartbeat_when_ready(self, tmp_path):
        """Fires heartbeat when not in cooldown and lock not held."""
        runner = _make_runner(tmp_path)
        mock_anima = MagicMock()
        mock_anima.messenger.has_unread.return_value = True
        mock_anima._lock = MagicMock()
        mock_anima._lock.locked.return_value = False
        runner.anima = mock_anima
        runner._deferred_timer = MagicMock()

        with patch.object(runner, "_is_in_cooldown", return_value=False), \
             patch.object(runner, "_message_triggered_heartbeat", new_callable=AsyncMock):
            await runner._try_deferred_trigger()
            assert runner._pending_trigger is True
            assert runner._deferred_timer is None

    @pytest.mark.asyncio
    async def test_reschedules_when_in_cooldown(self, tmp_path):
        """Re-schedules if still in cooldown."""
        runner = _make_runner(tmp_path)
        mock_anima = MagicMock()
        mock_anima.messenger.has_unread.return_value = True
        runner.anima = mock_anima
        runner._deferred_timer = MagicMock()

        with patch.object(runner, "_is_in_cooldown", return_value=True), \
             patch.object(runner, "_schedule_deferred_trigger") as mock_sched:
            await runner._try_deferred_trigger()
            mock_sched.assert_called_once()
            assert runner._pending_trigger is False

    @pytest.mark.asyncio
    async def test_reschedules_when_lock_held(self, tmp_path):
        """Re-schedules if anima lock is held."""
        runner = _make_runner(tmp_path)
        mock_anima = MagicMock()
        mock_anima.messenger.has_unread.return_value = True
        mock_anima._lock = MagicMock()
        mock_anima._lock.locked.return_value = True
        runner.anima = mock_anima
        runner._deferred_timer = MagicMock()

        with patch.object(runner, "_is_in_cooldown", return_value=False), \
             patch.object(runner, "_schedule_deferred_trigger") as mock_sched:
            await runner._try_deferred_trigger()
            mock_sched.assert_called_once()

    @pytest.mark.asyncio
    async def test_noop_when_no_unread(self, tmp_path):
        """Does nothing if inbox is empty."""
        runner = _make_runner(tmp_path)
        mock_anima = MagicMock()
        mock_anima.messenger.has_unread.return_value = False
        runner.anima = mock_anima
        runner._deferred_timer = MagicMock()

        await runner._try_deferred_trigger()
        assert runner._pending_trigger is False
        assert runner._deferred_timer is None

    @pytest.mark.asyncio
    async def test_noop_when_pending_trigger(self, tmp_path):
        """Does nothing if trigger already pending."""
        runner = _make_runner(tmp_path)
        mock_anima = MagicMock()
        mock_anima.messenger.has_unread.return_value = True
        runner.anima = mock_anima
        runner._deferred_timer = MagicMock()
        runner._pending_trigger = True

        await runner._try_deferred_trigger()

    @pytest.mark.asyncio
    async def test_clears_timer_on_entry(self, tmp_path):
        """Timer reference is cleared at the start of execution."""
        runner = _make_runner(tmp_path)
        mock_anima = MagicMock()
        mock_anima.messenger.has_unread.return_value = False
        runner.anima = mock_anima
        runner._deferred_timer = MagicMock()

        await runner._try_deferred_trigger()
        assert runner._deferred_timer is None


# ── _on_anima_lock_released ────────────────────────────


class TestOnAnimaLockReleased:
    """Verify lock-released callback behavior."""

    @pytest.mark.asyncio
    async def test_schedules_deferred_when_unread(self, tmp_path):
        """Schedules deferred trigger when unread messages exist."""
        runner = _make_runner(tmp_path)
        mock_anima = MagicMock()
        mock_anima.messenger.has_unread.return_value = True
        runner.anima = mock_anima

        with patch.object(runner, "_schedule_deferred_trigger") as mock_sched:
            await runner._on_anima_lock_released()
            mock_sched.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_action_when_no_unread(self, tmp_path):
        """Does nothing when no unread messages."""
        runner = _make_runner(tmp_path)
        mock_anima = MagicMock()
        mock_anima.messenger.has_unread.return_value = False
        runner.anima = mock_anima

        with patch.object(runner, "_schedule_deferred_trigger") as mock_sched:
            await runner._on_anima_lock_released()
            mock_sched.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_action_when_pending_trigger(self, tmp_path):
        """Does nothing when trigger already pending."""
        runner = _make_runner(tmp_path)
        mock_anima = MagicMock()
        mock_anima.messenger.has_unread.return_value = True
        runner.anima = mock_anima
        runner._pending_trigger = True

        with patch.object(runner, "_schedule_deferred_trigger") as mock_sched:
            await runner._on_anima_lock_released()
            mock_sched.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_action_when_no_anima(self, tmp_path):
        """Does nothing when anima is None."""
        runner = _make_runner(tmp_path)
        assert runner.anima is None

        # Should not raise
        await runner._on_anima_lock_released()


# ── Cleanup ────────────────────────────────────────────


class TestDeferredTimerCleanup:
    """Verify timer is cleaned up properly."""

    @pytest.mark.asyncio
    async def test_cleanup_cancels_timer(self, tmp_path):
        """_cleanup cancels deferred timer."""
        runner = _make_runner(tmp_path)
        mock_timer = MagicMock()
        runner._deferred_timer = mock_timer

        await runner._cleanup()
        mock_timer.cancel.assert_called_once()
        assert runner._deferred_timer is None

    @pytest.mark.asyncio
    async def test_cleanup_noop_when_no_timer(self, tmp_path):
        """_cleanup handles None timer gracefully."""
        runner = _make_runner(tmp_path)
        assert runner._deferred_timer is None

        # Should not raise
        await runner._cleanup()
