# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for restart/reconciliation race condition fix.

Verifies that start_anima() and _reconcile() skip animas that are
currently in the _restarting set, preventing duplicate child processes.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.supervisor.manager import ProcessSupervisor, RestartPolicy, HealthConfig


@pytest.fixture
def supervisor(tmp_path: Path) -> ProcessSupervisor:
    """Create a ProcessSupervisor with test paths."""
    animas_dir = tmp_path / "animas"
    animas_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()

    return ProcessSupervisor(
        animas_dir=animas_dir,
        shared_dir=shared_dir,
        run_dir=run_dir,
    )


class TestStartAnimaRestartingGuard:
    """start_anima() should skip if anima is in _restarting set."""

    @pytest.mark.asyncio
    async def test_start_anima_proceeds_when_restarting(self, supervisor: ProcessSupervisor):
        """start_anima() should proceed even when anima is in _restarting (guard removed)."""
        supervisor._restarting.add("test-anima")

        # Mock ProcessHandle to avoid real process creation
        with patch("core.supervisor.manager.ProcessHandle") as MockHandle:
            mock_handle = AsyncMock()
            mock_handle.get_pid.return_value = 12345
            mock_handle.send_request = AsyncMock(
                return_value=MagicMock(error=None, result={"needs_bootstrap": False})
            )
            MockHandle.return_value = mock_handle

            await supervisor.start_anima("test-anima")

            assert "test-anima" in supervisor.processes
            mock_handle.start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_anima_proceeds_when_not_restarting(self, supervisor: ProcessSupervisor):
        """start_anima() proceeds normally when anima is not in _restarting."""
        # Mock ProcessHandle to avoid real process creation
        with patch("core.supervisor.manager.ProcessHandle") as MockHandle:
            mock_handle = AsyncMock()
            mock_handle.get_pid.return_value = 12345
            mock_handle.send_request = AsyncMock(
                return_value=MagicMock(error=None, result={"needs_bootstrap": False})
            )
            MockHandle.return_value = mock_handle

            await supervisor.start_anima("test-anima")

            assert "test-anima" in supervisor.processes
            mock_handle.start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_anima_skips_existing_process(self, supervisor: ProcessSupervisor):
        """start_anima() returns early when process already exists."""
        supervisor.processes["test-anima"] = MagicMock()

        # Should return without error
        await supervisor.start_anima("test-anima")

        # Verify the existing process wasn't replaced
        assert "test-anima" in supervisor.processes


class TestReconcileRestartingGuard:
    """_reconcile() should skip animas in _restarting set."""

    @pytest.mark.asyncio
    async def test_reconcile_skips_restarting_anima(self, supervisor: ProcessSupervisor):
        """_reconcile() should not start an anima that is currently restarting."""
        # Set up an anima directory on disk (enabled)
        anima_dir = supervisor.animas_dir / "test-anima"
        anima_dir.mkdir()
        (anima_dir / "identity.md").write_text("Test identity")

        # Mark as restarting
        supervisor._restarting.add("test-anima")

        # Patch start_anima to track calls
        supervisor.start_anima = AsyncMock()

        # Patch _reconcile_assets to skip
        with patch.object(supervisor, "_reconcile_assets", new_callable=AsyncMock):
            await supervisor._reconcile()

        # start_anima should NOT have been called
        supervisor.start_anima.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reconcile_starts_non_restarting_anima(self, supervisor: ProcessSupervisor):
        """_reconcile() should start an anima that is NOT restarting."""
        # Set up an anima directory on disk (enabled)
        anima_dir = supervisor.animas_dir / "test-anima"
        anima_dir.mkdir()
        (anima_dir / "identity.md").write_text("Test identity")
        (anima_dir / "status.json").write_text('{"enabled": true}')

        # NOT in _restarting set
        assert "test-anima" not in supervisor._restarting

        # Patch start_anima to track calls
        supervisor.start_anima = AsyncMock()

        with patch.object(supervisor, "_reconcile_assets", new_callable=AsyncMock):
            await supervisor._reconcile()

        # start_anima SHOULD have been called
        supervisor.start_anima.assert_awaited_once_with("test-anima")

    @pytest.mark.asyncio
    async def test_concurrent_restart_and_reconcile(self, supervisor: ProcessSupervisor):
        """Simulate race: _restarting prevents double start via _reconcile guard."""
        anima_dir = supervisor.animas_dir / "test-anima"
        anima_dir.mkdir()
        (anima_dir / "identity.md").write_text("Test identity")

        supervisor.start_anima = AsyncMock()

        # Simulate restart in progress
        supervisor._restarting.add("test-anima")

        with patch.object(supervisor, "_reconcile_assets", new_callable=AsyncMock):
            await supervisor._reconcile()

        # _reconcile should skip animas in _restarting (guard is in _reconcile itself)
        supervisor.start_anima.assert_not_awaited()
