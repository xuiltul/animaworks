"""Unit tests for progress-aware busy hang detection.

Tests that the health check kills processes only when there has been
no LLM progress for the configured threshold (15 min default), and
never kills processes that are making progress.
"""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.supervisor._mgr_health import HealthMixin
from core.supervisor.manager import HealthConfig
from core.supervisor.process_handle import ProcessHandle, ProcessState, ProcessStats
from core.time_utils import now_jst

JST = timezone(timedelta(hours=9))


def _make_handle(tmp_path: Path, *, started_minutes_ago: int = 5) -> ProcessHandle:
    h = ProcessHandle(
        anima_name="test-anima",
        socket_path=tmp_path / "test.sock",
        animas_dir=tmp_path / "animas",
        shared_dir=tmp_path / "shared",
    )
    h.state = ProcessState.RUNNING
    h.stats = ProcessStats(started_at=now_jst() - timedelta(minutes=started_minutes_ago))
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.returncode = None
    h.process = mock_proc
    return h


def _make_supervisor(health_config: HealthConfig | None = None) -> HealthMixin:
    sup = object.__new__(HealthMixin)
    sup.health_config = health_config or HealthConfig()
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
    return sup


class TestProgressAwareBusyHang:
    """Tests for progress-aware busy hang detection."""

    @pytest.mark.asyncio
    async def test_busy_with_recent_progress_not_killed(self, tmp_path: Path):
        """Process busy with recent progress should NOT be killed."""
        handle = _make_handle(tmp_path)
        now = now_jst()
        handle.ping = AsyncMock(return_value={
            "success": True,
            "is_busy": True,
            "last_progress_at": (now - timedelta(seconds=30)).isoformat(),
        })

        sup = _make_supervisor()
        hang_calls: list[str] = []

        async def mock_hang(name, h):
            hang_calls.append(name)

        sup._handle_process_hang = mock_hang

        await sup._check_process_health("test-anima", handle)

        assert len(hang_calls) == 0
        assert handle.stats.missed_pings == 0

    @pytest.mark.asyncio
    async def test_busy_with_stale_progress_killed(self, tmp_path: Path):
        """Process busy with no progress for >15min should be killed."""
        handle = _make_handle(tmp_path)
        now = now_jst()
        handle.ping = AsyncMock(return_value={
            "success": True,
            "is_busy": True,
            "last_progress_at": (now - timedelta(minutes=16)).isoformat(),
        })

        sup = _make_supervisor()
        hang_calls: list[str] = []

        async def mock_hang(name, h):
            hang_calls.append(name)

        sup._handle_process_hang = mock_hang

        await sup._check_process_health("test-anima", handle)
        await asyncio.sleep(0)  # let create_task execute

        assert len(hang_calls) == 1
        assert hang_calls[0] == "test-anima"

    @pytest.mark.asyncio
    async def test_busy_without_progress_info_fallback(self, tmp_path: Path):
        """Process busy without last_progress_at should use fallback timer."""
        handle = _make_handle(tmp_path)
        handle.ping = AsyncMock(return_value={
            "success": True,
            "is_busy": True,
            "last_progress_at": None,
        })

        sup = _make_supervisor()
        hang_calls: list[str] = []

        async def mock_hang(name, h):
            hang_calls.append(name)

        sup._handle_process_hang = mock_hang

        # First call: sets last_busy_since
        await sup._check_process_health("test-anima", handle)
        assert handle.stats.last_busy_since is not None
        assert len(hang_calls) == 0

        # Simulate 16 minutes passing
        handle.stats.last_busy_since = now_jst() - timedelta(minutes=16)
        await sup._check_process_health("test-anima", handle)
        await asyncio.sleep(0)  # let create_task execute
        assert len(hang_calls) == 1

    @pytest.mark.asyncio
    async def test_busy_to_idle_resets_last_busy_since(self, tmp_path: Path):
        """When process goes from busy to idle, last_busy_since should reset."""
        handle = _make_handle(tmp_path)

        # First: busy (no progress info)
        handle.ping = AsyncMock(return_value={
            "success": True,
            "is_busy": True,
            "last_progress_at": None,
        })
        sup = _make_supervisor()
        sup._handle_process_hang = AsyncMock()
        await sup._check_process_health("test-anima", handle)
        assert handle.stats.last_busy_since is not None

        # Then: idle
        handle.ping = AsyncMock(return_value={
            "success": True,
            "is_busy": False,
        })
        await sup._check_process_health("test-anima", handle)
        assert handle.stats.last_busy_since is None

    @pytest.mark.asyncio
    async def test_custom_threshold_from_config(self, tmp_path: Path):
        """Custom busy_hang_threshold_sec should be respected."""
        handle = _make_handle(tmp_path)
        now = now_jst()
        handle.ping = AsyncMock(return_value={
            "success": True,
            "is_busy": True,
            "last_progress_at": (now - timedelta(minutes=6)).isoformat(),
        })

        # 5-minute threshold (300s)
        config = HealthConfig(busy_hang_threshold_sec=300.0)
        sup = _make_supervisor(config)
        hang_calls: list[str] = []

        async def mock_hang(name, h):
            hang_calls.append(name)

        sup._handle_process_hang = mock_hang

        await sup._check_process_health("test-anima", handle)
        await asyncio.sleep(0)  # let create_task execute
        assert len(hang_calls) == 1

    @pytest.mark.asyncio
    async def test_progress_at_boundary_not_killed(self, tmp_path: Path):
        """Process at exactly 15 min should NOT be killed (> not >=)."""
        handle = _make_handle(tmp_path)
        now = now_jst()
        handle.ping = AsyncMock(return_value={
            "success": True,
            "is_busy": True,
            "last_progress_at": (now - timedelta(seconds=899)).isoformat(),
        })

        sup = _make_supervisor()
        hang_calls: list[str] = []

        async def mock_hang(name, h):
            hang_calls.append(name)

        sup._handle_process_hang = mock_hang

        await sup._check_process_health("test-anima", handle)
        assert len(hang_calls) == 0

    @pytest.mark.asyncio
    async def test_ping_failure_resets_last_busy_since(self, tmp_path: Path):
        """Ping failure should reset last_busy_since."""
        handle = _make_handle(tmp_path)
        handle.stats.last_busy_since = now_jst() - timedelta(minutes=5)

        handle.ping = AsyncMock(return_value={
            "success": False,
        })

        sup = _make_supervisor()
        sup._handle_process_hang = AsyncMock()

        await sup._check_process_health("test-anima", handle)
        assert handle.stats.last_busy_since is None

    @pytest.mark.asyncio
    async def test_not_busy_recovered_log(self, tmp_path: Path):
        """When transitioning from busy to not-busy, 'recovered' is logged."""
        handle = _make_handle(tmp_path)
        handle.stats.last_busy_since = now_jst()

        handle.ping = AsyncMock(return_value={
            "success": True,
            "is_busy": False,
        })

        sup = _make_supervisor()
        sup._handle_process_hang = AsyncMock()

        with patch("core.supervisor._mgr_health.logger") as mock_logger:
            await sup._check_process_health("test-anima", handle)
            mock_logger.info.assert_any_call("Process recovered: %s", "test-anima")


class TestProcessStatsLastBusySince:
    """Tests for ProcessStats.last_busy_since field."""

    def test_default_is_none(self):
        stats = ProcessStats(started_at=now_jst())
        assert stats.last_busy_since is None

    def test_can_set_datetime(self):
        stats = ProcessStats(started_at=now_jst())
        now = now_jst()
        stats.last_busy_since = now
        assert stats.last_busy_since == now


class TestHealthConfigDefaults:
    """Tests for HealthConfig busy_hang_threshold_sec."""

    def test_default_threshold_is_900(self):
        config = HealthConfig()
        assert config.busy_hang_threshold_sec == 900.0

    def test_custom_threshold(self):
        config = HealthConfig(busy_hang_threshold_sec=600.0)
        assert config.busy_hang_threshold_sec == 600.0


class TestServerConfigBusyHangThreshold:
    """Tests for ServerConfig.busy_hang_threshold."""

    def test_default_is_900(self):
        from core.config.models import ServerConfig

        config = ServerConfig()
        assert config.busy_hang_threshold == 900

    def test_custom_value(self):
        from core.config.models import ServerConfig

        config = ServerConfig(busy_hang_threshold=1200)
        assert config.busy_hang_threshold == 1200
