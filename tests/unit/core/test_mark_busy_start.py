"""Unit tests for _mark_busy_start() — busy-hang false positive fix.

Verifies that _mark_busy_start() resets _last_progress_at on lock acquisition,
preventing the health monitor from seeing stale timestamps and falsely killing
processes at heartbeat boundaries.
"""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import os
from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

from core.time_utils import now_jst


def _make_digital_anima(anima_dir, shared_dir):
    """Create a DigitalAnima with mocked dependencies."""
    with (
        patch("core.anima.AgentCore"),
        patch("core.anima.MemoryManager") as MockMM,
        patch("core.anima.Messenger"),
    ):
        MockMM.return_value.read_model_config.return_value = MagicMock()
        from core.anima import DigitalAnima

        return DigitalAnima(anima_dir, shared_dir)


class TestMarkBusyStart:
    """Tests for DigitalAnima._mark_busy_start()."""

    def test_sets_last_progress_at(self, data_dir, make_anima):
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp = _make_digital_anima(anima_dir, shared_dir)

        assert dp._last_progress_at is None

        dp._mark_busy_start()

        assert dp._last_progress_at is not None
        elapsed = (now_jst() - dp._last_progress_at).total_seconds()
        assert elapsed < 2.0

    def test_sets_busy_since(self, data_dir, make_anima):
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp = _make_digital_anima(anima_dir, shared_dir)

        assert dp._busy_since is None

        dp._mark_busy_start()

        assert dp._busy_since is not None
        elapsed = (now_jst() - dp._busy_since).total_seconds()
        assert elapsed < 2.0

    def test_overwrites_stale_progress(self, data_dir, make_anima):
        """A stale _last_progress_at from a previous task is replaced."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp = _make_digital_anima(anima_dir, shared_dir)

        stale_time = now_jst() - timedelta(minutes=30)
        dp._last_progress_at = stale_time

        dp._mark_busy_start()

        assert dp._last_progress_at != stale_time
        elapsed = (now_jst() - dp._last_progress_at).total_seconds()
        assert elapsed < 2.0

    def test_last_progress_and_busy_since_are_same(self, data_dir, make_anima):
        """Both timestamps should be set to the same time."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp = _make_digital_anima(anima_dir, shared_dir)

        dp._mark_busy_start()

        assert dp._last_progress_at == dp._busy_since

    def test_writes_and_clears_busy_sidecar(self, data_dir, make_anima):
        """Runtime instances write an IPC-independent busy marker."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp = _make_digital_anima(anima_dir, shared_dir)

        dp._mark_busy_start()

        sidecar = data_dir / "run" / "animas" / "alice.busy.json"
        assert sidecar.exists()
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        assert data["anima"] == "alice"
        assert data["pid"] == os.getpid()
        assert data["is_busy"] is True

        dp._clear_busy_status_sidecar_if_idle()
        assert not sidecar.exists()

    def test_progress_callback_refreshes_busy_sidecar(self, data_dir, make_anima):
        """Streaming progress should refresh the IPC-independent busy marker."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp = _make_digital_anima(anima_dir, shared_dir)

        async def _hold_lock():
            async with dp._background_lock:
                dp._mark_busy_start()
                sidecar = data_dir / "run" / "animas" / "alice.busy.json"
                before = json.loads(sidecar.read_text(encoding="utf-8"))

                dp._last_progress_at = now_jst() - timedelta(minutes=20)
                dp._agent_progress_callback()

                after = json.loads(sidecar.read_text(encoding="utf-8"))
                assert after["last_progress_at"] != before["last_progress_at"]
                assert after["last_progress_at"] == dp._last_progress_at.isoformat()

        asyncio.run(_hold_lock())


class TestHeartbeatCallsMarkBusyStart:
    """Verify that heartbeat acquisition triggers _mark_busy_start()."""

    @pytest.mark.asyncio
    async def test_heartbeat_resets_progress(self, data_dir, make_anima):
        """run_heartbeat should call _mark_busy_start() on lock acquisition."""
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp = _make_digital_anima(anima_dir, shared_dir)

        stale_time = now_jst() - timedelta(minutes=29)
        dp._last_progress_at = stale_time

        mark_calls = []
        original_mark = dp._mark_busy_start

        def tracking_mark():
            mark_calls.append(True)
            original_mark()

        dp._mark_busy_start = tracking_mark

        from core.schemas import CycleResult

        mock_result = CycleResult(
            trigger="heartbeat", action="idle", summary="done", duration_ms=100,
        )
        with patch.object(dp, "_build_heartbeat_prompt", return_value=[]), \
             patch.object(dp, "_execute_heartbeat_cycle", return_value=mock_result), \
             patch.object(dp, "_build_prior_messages", return_value=[]), \
             patch.object(dp._activity, "log"), \
             patch.object(dp.messenger, "has_unread", return_value=False), \
             patch("core.tooling.handler.active_session_type"):
            try:
                await dp.run_heartbeat()
            except Exception:
                pass

        assert len(mark_calls) >= 1
        assert dp._last_progress_at != stale_time


class TestHealthCheckWithFreshProgress:
    """Integration: health check should NOT kill a process with fresh progress from _mark_busy_start()."""

    @pytest.mark.asyncio
    async def test_fresh_busy_start_prevents_false_positive_kill(self, tmp_path):
        """Simulates the exact race condition: stale progress + new lock acquisition.

        After _mark_busy_start(), the health check should see a fresh timestamp
        and NOT trigger a kill.
        """
        from core.supervisor._mgr_health import HealthMixin
        from core.supervisor.manager import HealthConfig
        from core.supervisor.process_handle import ProcessHandle, ProcessState, ProcessStats

        handle = ProcessHandle(
            anima_name="test-anima",
            socket_path=tmp_path / "test.sock",
            animas_dir=tmp_path / "animas",
            shared_dir=tmp_path / "shared",
        )
        handle.state = ProcessState.RUNNING
        handle.stats = ProcessStats(started_at=now_jst() - timedelta(minutes=60))
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.returncode = None
        handle.process = mock_proc

        now = now_jst()
        fresh_progress = now - timedelta(seconds=3)
        from unittest.mock import AsyncMock

        handle.ping = AsyncMock(return_value={
            "success": True,
            "is_busy": True,
            "last_progress_at": fresh_progress.isoformat(),
        })

        sup = object.__new__(HealthMixin)
        sup.health_config = HealthConfig()
        sup._shutdown = False
        sup._permanently_failed = set()
        sup._failed_log_times = {}
        sup._restarting = set()
        sup._restart_counts = {}
        sup.restart_policy = MagicMock()
        sup.restart_policy.max_retries = 5
        sup.restart_policy.backoff_base_sec = 2.0
        sup.restart_policy.backoff_max_sec = 60.0
        sup.restart_policy.reset_after_sec = 300.0
        sup._max_streaming_duration_sec = 1800
        sup.processes = {}

        hang_calls: list[str] = []

        async def mock_hang(name, h):
            hang_calls.append(name)

        sup._handle_process_hang = mock_hang

        await sup._check_process_health("test-anima", handle)

        assert len(hang_calls) == 0, (
            "Process with fresh _last_progress_at (3s ago) should NOT be killed"
        )

    @pytest.mark.asyncio
    async def test_stale_progress_still_triggers_kill(self, tmp_path):
        """Genuinely stale progress (>15min) should still trigger kill."""
        from unittest.mock import AsyncMock

        from core.supervisor._mgr_health import HealthMixin
        from core.supervisor.manager import HealthConfig
        from core.supervisor.process_handle import ProcessHandle, ProcessState, ProcessStats

        handle = ProcessHandle(
            anima_name="test-anima",
            socket_path=tmp_path / "test.sock",
            animas_dir=tmp_path / "animas",
            shared_dir=tmp_path / "shared",
        )
        handle.state = ProcessState.RUNNING
        handle.stats = ProcessStats(started_at=now_jst() - timedelta(minutes=60))
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.returncode = None
        handle.process = mock_proc

        stale_progress = now_jst() - timedelta(minutes=20)
        handle.ping = AsyncMock(return_value={
            "success": True,
            "is_busy": True,
            "last_progress_at": stale_progress.isoformat(),
        })

        sup = object.__new__(HealthMixin)
        sup.health_config = HealthConfig()
        sup._shutdown = False
        sup._permanently_failed = set()
        sup._failed_log_times = {}
        sup._restarting = set()
        sup._restart_counts = {}
        sup.restart_policy = MagicMock()
        sup.restart_policy.max_retries = 5
        sup.restart_policy.backoff_base_sec = 2.0
        sup.restart_policy.backoff_max_sec = 60.0
        sup.restart_policy.reset_after_sec = 300.0
        sup._max_streaming_duration_sec = 1800
        sup.processes = {}

        hang_calls: list[str] = []

        async def mock_hang(name, h):
            hang_calls.append(name)

        sup._handle_process_hang = mock_hang

        await sup._check_process_health("test-anima", handle)
        await asyncio.sleep(0)

        assert len(hang_calls) == 1, (
            "Process with genuinely stale progress (20min) should be killed"
        )


class TestPingReturnsBusySince:
    """Verify that the ping handler returns busy_since in its response."""

    @pytest.mark.asyncio
    async def test_ping_includes_busy_since(self, data_dir, make_anima):
        anima_dir = make_anima("alice")
        shared_dir = data_dir / "shared"
        dp = _make_digital_anima(anima_dir, shared_dir)

        dp._mark_busy_start()

        from core.supervisor.runner import AnimaRunner

        runner = object.__new__(AnimaRunner)
        runner.anima = dp
        runner.anima_name = "alice"
        runner._started_at = now_jst()
        runner._ready_event = asyncio.Event()
        runner._ready_event.set()

        result = await runner._handle_ping({})

        assert "busy_since" in result
        assert result["busy_since"] is not None
        assert result["last_progress_at"] is not None
