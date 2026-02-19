from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Tests for _heartbeat_wrapper cascade detection in LifecycleManager.

Covers:
- No inbox messages -> no cascade suppression
- Cascade detected -> sender passed as suppressed
- No cascade -> None passed
- Pair exchanges recorded for processed (non-suppressed) senders
- Suppressed senders not recorded as pair exchanges
- Mixed senders: only cascading ones suppressed
"""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.lifecycle import LifecycleManager
from core.schemas import Message


class TestScheduledHBCascadeDetection:
    """Test _heartbeat_wrapper cascade detection for scheduled heartbeats."""

    def _make_manager(self) -> LifecycleManager:
        mgr = LifecycleManager()
        return mgr

    def _make_mock_anima(self, name: str = "test-anima") -> MagicMock:
        anima = MagicMock()
        anima.name = name
        anima.messenger = MagicMock()
        anima.run_heartbeat = AsyncMock()
        result_mock = MagicMock()
        result_mock.model_dump.return_value = {}
        anima.run_heartbeat.return_value = result_mock
        return anima

    @pytest.mark.asyncio
    async def test_no_inbox_messages_no_cascade(self):
        """No inbox messages -> no cascade suppression."""
        mgr = self._make_manager()
        anima = self._make_mock_anima()
        anima.messenger.receive.return_value = []
        mgr.animas["test-anima"] = anima

        await mgr._heartbeat_wrapper("test-anima")

        anima.run_heartbeat.assert_called_once_with(
            cascade_suppressed_senders=None,
        )

    @pytest.mark.asyncio
    async def test_cascade_detected_suppresses_sender(self):
        """When cascade is detected, the sender is passed as suppressed."""
        mgr = self._make_manager()
        anima = self._make_mock_anima()

        msg = Message(from_person="bob", to_person="test-anima", content="hi")
        anima.messenger.receive.return_value = [msg]
        mgr.animas["test-anima"] = anima

        # Pre-fill cascade history to trigger detection
        # _CASCADE_THRESHOLD = 3, so fill with 3 entries
        now = time.monotonic()
        mgr._pair_heartbeat_times[("test-anima", "bob")] = [now, now, now]

        await mgr._heartbeat_wrapper("test-anima")

        call_args = anima.run_heartbeat.call_args
        assert call_args.kwargs["cascade_suppressed_senders"] == {"bob"}

    @pytest.mark.asyncio
    async def test_no_cascade_passes_none(self):
        """When no cascade detected, passes None."""
        mgr = self._make_manager()
        anima = self._make_mock_anima()

        msg = Message(from_person="bob", to_person="test-anima", content="hi")
        anima.messenger.receive.return_value = [msg]
        mgr.animas["test-anima"] = anima
        # No pre-filled cascade history

        await mgr._heartbeat_wrapper("test-anima")

        anima.run_heartbeat.assert_called_once_with(
            cascade_suppressed_senders=None,
        )

    @pytest.mark.asyncio
    async def test_pair_exchange_recorded_for_processed_senders(self):
        """Pair exchanges are recorded for non-suppressed senders."""
        mgr = self._make_manager()
        anima = self._make_mock_anima()

        msg = Message(
            from_person="alice", to_person="test-anima", content="hi",
        )
        anima.messenger.receive.return_value = [msg]
        mgr.animas["test-anima"] = anima

        await mgr._heartbeat_wrapper("test-anima")

        # Check pair was recorded
        key = ("test-anima", "alice")
        assert key in mgr._pair_heartbeat_times
        assert len(mgr._pair_heartbeat_times[key]) == 1

    @pytest.mark.asyncio
    async def test_suppressed_senders_not_recorded_as_pair(self):
        """Cascade-suppressed senders are NOT recorded in pair exchanges."""
        mgr = self._make_manager()
        anima = self._make_mock_anima()

        msg = Message(from_person="bob", to_person="test-anima", content="hi")
        anima.messenger.receive.return_value = [msg]
        mgr.animas["test-anima"] = anima

        # Pre-fill cascade to trigger suppression
        now = time.monotonic()
        mgr._pair_heartbeat_times[("test-anima", "bob")] = [now, now, now]

        await mgr._heartbeat_wrapper("test-anima")

        # Bob was suppressed, so only the pre-existing entries should remain
        # (no new entry added)
        assert len(mgr._pair_heartbeat_times[("test-anima", "bob")]) == 3

    @pytest.mark.asyncio
    async def test_mixed_senders_partial_suppression(self):
        """With multiple senders, only cascading ones are suppressed."""
        mgr = self._make_manager()
        anima = self._make_mock_anima()

        msg1 = Message(
            from_person="bob", to_person="test-anima", content="hi",
        )
        msg2 = Message(
            from_person="alice", to_person="test-anima", content="hello",
        )
        anima.messenger.receive.return_value = [msg1, msg2]
        mgr.animas["test-anima"] = anima

        # Only bob is cascading
        now = time.monotonic()
        mgr._pair_heartbeat_times[("test-anima", "bob")] = [now, now, now]

        await mgr._heartbeat_wrapper("test-anima")

        call_args = anima.run_heartbeat.call_args
        assert call_args.kwargs["cascade_suppressed_senders"] == {"bob"}

        # Alice's pair should be recorded, bob's should not have new entries
        assert ("test-anima", "alice") in mgr._pair_heartbeat_times

    @pytest.mark.asyncio
    async def test_nonexistent_anima_returns_early(self):
        """If anima doesn't exist, _heartbeat_wrapper returns immediately."""
        mgr = self._make_manager()
        # No anima registered
        await mgr._heartbeat_wrapper("nonexistent")
        # No error raised
