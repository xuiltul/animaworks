# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for restart/reconciliation race condition fix.

Verifies that start_anima() and _reconcile() skip animas that are
currently in the _restarting or _starting set, preventing duplicate
child processes.
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


class TestStartAnimaStartingGuard:
    """start_anima() should skip if anima is already in _starting set."""

    @pytest.mark.asyncio
    async def test_start_anima_skips_when_already_starting(self, supervisor: ProcessSupervisor):
        """start_anima() returns early when the same anima is already starting."""
        supervisor._starting.add("test-anima")

        with patch("core.supervisor.manager.ProcessHandle") as MockHandle:
            await supervisor.start_anima("test-anima")

            # ProcessHandle should never be instantiated
            MockHandle.assert_not_called()

        assert "test-anima" not in supervisor.processes

    @pytest.mark.asyncio
    async def test_start_anima_sets_and_clears_starting_flag(self, supervisor: ProcessSupervisor):
        """start_anima() adds anima to _starting before spawn and removes after."""
        entered_starting: list[bool] = []

        original_start = ProcessSupervisor.start_anima

        with patch("core.supervisor.manager.ProcessHandle") as MockHandle:
            mock_handle = AsyncMock()
            mock_handle.get_pid.return_value = 99999
            mock_handle.send_request = AsyncMock(
                return_value=MagicMock(error=None, result={"needs_bootstrap": False})
            )

            async def capture_starting(*args, **kwargs):
                # Capture the flag state during handle.start()
                entered_starting.append("test-anima" in supervisor._starting)

            mock_handle.start = capture_starting
            MockHandle.return_value = mock_handle

            await supervisor.start_anima("test-anima")

        # Flag was set during handle.start()
        assert entered_starting == [True]
        # Flag is cleared after start_anima returns
        assert "test-anima" not in supervisor._starting

    @pytest.mark.asyncio
    async def test_start_anima_clears_starting_on_failure(self, supervisor: ProcessSupervisor):
        """_starting flag is cleared even when process start fails."""
        with patch("core.supervisor.manager.ProcessHandle") as MockHandle:
            mock_handle = AsyncMock()
            mock_handle.start = AsyncMock(side_effect=RuntimeError("spawn failed"))
            MockHandle.return_value = mock_handle

            with pytest.raises(RuntimeError, match="spawn failed"):
                await supervisor.start_anima("test-anima")

        # Flag must be cleared even after failure
        assert "test-anima" not in supervisor._starting
        assert "test-anima" not in supervisor.processes


class TestReconcileStartingGuard:
    """_reconcile() should skip animas in _starting set."""

    @pytest.mark.asyncio
    async def test_reconcile_skips_starting_anima(self, supervisor: ProcessSupervisor):
        """_reconcile() should not start an anima that is already being started."""
        anima_dir = supervisor.animas_dir / "test-anima"
        anima_dir.mkdir()
        (anima_dir / "identity.md").write_text("Test identity")
        (anima_dir / "status.json").write_text('{"enabled": true}')

        supervisor._starting.add("test-anima")
        supervisor.start_anima = AsyncMock()

        with patch.object(supervisor, "_reconcile_assets", new_callable=AsyncMock):
            await supervisor._reconcile()

        supervisor.start_anima.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reconcile_starts_when_not_starting(self, supervisor: ProcessSupervisor):
        """_reconcile() starts an anima that is NOT in _starting set."""
        anima_dir = supervisor.animas_dir / "test-anima"
        anima_dir.mkdir()
        (anima_dir / "identity.md").write_text("Test identity")
        (anima_dir / "status.json").write_text('{"enabled": true}')

        assert "test-anima" not in supervisor._starting
        supervisor.start_anima = AsyncMock()

        with patch.object(supervisor, "_reconcile_assets", new_callable=AsyncMock):
            await supervisor._reconcile()

        supervisor.start_anima.assert_awaited_once_with("test-anima")


class TestRestartAnimaRestartingGuard:
    """restart_anima() guards with _restarting during stop-start window."""

    @pytest.mark.asyncio
    async def test_restart_anima_sets_restarting_guard(self, supervisor: ProcessSupervisor):
        """restart_anima() adds to _restarting before stop and clears after start."""
        observed_states: list[bool] = []

        supervisor.stop_anima = AsyncMock()

        async def _capturing_start(name: str) -> None:
            observed_states.append(name in supervisor._restarting)
            supervisor.processes[name] = MagicMock()

        supervisor.start_anima = _capturing_start
        supervisor.processes["test-anima"] = MagicMock()

        await supervisor.restart_anima("test-anima")

        assert observed_states == [True]
        assert "test-anima" not in supervisor._restarting

    @pytest.mark.asyncio
    async def test_restart_anima_clears_restarting_on_failure(self, supervisor: ProcessSupervisor):
        """_restarting is cleared even when restart_anima() fails."""
        supervisor.stop_anima = AsyncMock()

        async def _failing_start(name: str) -> None:
            raise RuntimeError("start failed")

        supervisor.start_anima = _failing_start
        supervisor.processes["test-anima"] = MagicMock()

        with pytest.raises(RuntimeError, match="start failed"):
            await supervisor.restart_anima("test-anima")

        assert "test-anima" not in supervisor._restarting
