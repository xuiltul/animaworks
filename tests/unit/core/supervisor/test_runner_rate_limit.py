"""Unit tests for rate limiting in core/supervisor/runner.py."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.supervisor.runner import (
    AnimaRunner,
    _CASCADE_THRESHOLD,
    _CASCADE_WINDOW_S,
    _MSG_HEARTBEAT_COOLDOWN_S,
)


# ── Helpers ──────────────────────────────────────────────────


def _make_runner() -> AnimaRunner:
    """Create a AnimaRunner with minimal config for unit testing."""
    runner = AnimaRunner(
        anima_name="test",
        socket_path=Path("/tmp/test.sock"),
        animas_dir=Path("/tmp/animas"),
        shared_dir=Path("/tmp/shared"),
    )
    runner.anima = MagicMock()
    runner.anima.messenger = MagicMock()
    runner.anima._lock = asyncio.Lock()
    return runner


# ── TestCooldown ─────────────────────────────────────────────


class TestCooldown:
    """Tests for _is_in_cooldown() method."""

    def test_no_cooldown_initially(self):
        """_is_in_cooldown() returns False when _last_msg_heartbeat_end is 0."""
        runner = _make_runner()
        # Default is 0.0, which is far in the past relative to monotonic()
        assert runner._is_in_cooldown() is False

    def test_in_cooldown_within_60s(self):
        """Returns True when heartbeat ended less than 60s ago."""
        runner = _make_runner()
        runner._last_msg_heartbeat_end = time.monotonic()
        assert runner._is_in_cooldown() is True

    def test_cooldown_expired(self):
        """Returns False when >60s has passed since last heartbeat end."""
        runner = _make_runner()
        runner._last_msg_heartbeat_end = (
            time.monotonic() - _MSG_HEARTBEAT_COOLDOWN_S - 1
        )
        assert runner._is_in_cooldown() is False


# ── TestCascadeDetection ─────────────────────────────────────


class TestCascadeDetection:
    """Tests for _check_cascade() method."""

    def test_no_cascade_below_threshold(self):
        """_check_cascade returns False when < 4 round-trips."""
        runner = _make_runner()
        now = time.monotonic()
        key = ("test", "alice")
        # Add entries below threshold
        runner._pair_heartbeat_times[key] = [
            now - 10, now - 5, now - 1,
        ]
        assert runner._check_cascade({"alice"}) is False

    def test_cascade_detected_at_threshold(self):
        """Returns True when exactly 4 round-trips in window."""
        runner = _make_runner()
        now = time.monotonic()
        key = ("test", "alice")
        runner._pair_heartbeat_times[key] = [
            now - 30, now - 20, now - 10, now - 1,
        ]
        assert runner._check_cascade({"alice"}) is True

    def test_cascade_entries_expire(self):
        """Old entries outside 600s window are evicted."""
        runner = _make_runner()
        now = time.monotonic()
        key = ("test", "alice")
        # All entries are older than the cascade window
        runner._pair_heartbeat_times[key] = [
            now - _CASCADE_WINDOW_S - 100,
            now - _CASCADE_WINDOW_S - 50,
            now - _CASCADE_WINDOW_S - 20,
            now - _CASCADE_WINDOW_S - 10,
        ]
        assert runner._check_cascade({"alice"}) is False
        # Expired entries should have been evicted
        assert key not in runner._pair_heartbeat_times


# ── TestRecordPairHeartbeat ──────────────────────────────────


class TestRecordPairHeartbeat:
    """Tests for _record_pair_heartbeat() method."""

    def test_records_pair_timestamp(self):
        """Adds entry to _pair_heartbeat_times dict."""
        runner = _make_runner()
        before = time.monotonic()
        runner._record_pair_heartbeat({"alice"})
        after = time.monotonic()

        key = ("test", "alice")
        assert key in runner._pair_heartbeat_times
        times = runner._pair_heartbeat_times[key]
        assert len(times) == 1
        assert before <= times[0] <= after

    def test_records_multiple_senders(self):
        """Records entries for each sender separately."""
        runner = _make_runner()
        runner._record_pair_heartbeat({"alice", "bob"})

        assert ("test", "alice") in runner._pair_heartbeat_times
        assert ("test", "bob") in runner._pair_heartbeat_times

    def test_appends_to_existing(self):
        """Appends timestamps to existing entries rather than replacing."""
        runner = _make_runner()
        runner._record_pair_heartbeat({"alice"})
        runner._record_pair_heartbeat({"alice"})

        key = ("test", "alice")
        assert len(runner._pair_heartbeat_times[key]) == 2


# ── TestInboxWatcher ─────────────────────────────────────────


class TestInboxWatcher:
    """Tests for _inbox_watcher_loop() method."""

    @pytest.mark.asyncio
    async def test_skips_when_pending_trigger(self):
        """Loop continues without triggering when _pending_trigger is True."""
        runner = _make_runner()
        runner._pending_trigger = True

        # Set shutdown after first iteration
        iteration_count = 0
        original_sleep = asyncio.sleep

        async def mock_sleep(duration):
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count >= 1:
                runner.shutdown_event.set()
            await original_sleep(0)

        with patch("asyncio.sleep", side_effect=mock_sleep):
            await runner._inbox_watcher_loop()

        # messenger.has_unread should never be called because
        # _pending_trigger short-circuits first
        runner.anima.messenger.has_unread.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_in_cooldown(self):
        """Loop continues without triggering when in cooldown."""
        runner = _make_runner()
        runner._last_msg_heartbeat_end = time.monotonic()  # Just ended
        runner.anima.messenger.has_unread.return_value = True

        iteration_count = 0
        original_sleep = asyncio.sleep

        async def mock_sleep(duration):
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count >= 1:
                runner.shutdown_event.set()
            await original_sleep(0)

        with patch("asyncio.sleep", side_effect=mock_sleep):
            await runner._inbox_watcher_loop()

        # _pending_trigger should remain False — no heartbeat was triggered
        assert runner._pending_trigger is False

    @pytest.mark.asyncio
    async def test_defers_when_lock_held(self):
        """Sets _deferred_inbox when anima._lock is locked."""
        runner = _make_runner()
        runner.anima.messenger.has_unread.return_value = True

        # Acquire the lock before the watcher checks it
        await runner.anima._lock.acquire()

        iteration_count = 0
        original_sleep = asyncio.sleep

        async def mock_sleep(duration):
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count >= 1:
                runner.shutdown_event.set()
            await original_sleep(0)

        with patch("asyncio.sleep", side_effect=mock_sleep):
            await runner._inbox_watcher_loop()

        assert runner._deferred_timer is not None
        assert runner._pending_trigger is False

        # Release the lock to clean up
        runner.anima._lock.release()
