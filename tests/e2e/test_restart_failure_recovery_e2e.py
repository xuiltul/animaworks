# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for the restart bug fix (_handle_process_failure flow).

Verifies:
- _handle_process_failure calls restart_anima and populates processes[name]
- _restarting set is cleaned up in the finally block
- _handle_process_hang kills the process then delegates to _handle_process_failure
- Max retries exceeded sets state to FAILED without restarting
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.supervisor.manager import ProcessSupervisor, RestartPolicy, HealthConfig
from core.supervisor.process_handle import ProcessState


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


class TestFailureToRestartFlow:
    """Full _handle_process_failure -> restart_anima flow."""

    async def test_failure_triggers_restart_and_populates_processes(
        self, supervisor: ProcessSupervisor,
    ):
        """After _handle_process_failure, processes[name] holds the new handle."""
        name = "test-anima"

        # Set up old (crashed) handle in processes
        old_handle = MagicMock()
        old_handle.state = ProcessState.FAILED
        supervisor.processes[name] = old_handle

        # Create the new handle that restart_anima will install
        new_handle = MagicMock()
        new_handle.get_pid.return_value = 99999
        new_handle.state = ProcessState.RUNNING

        async def mock_restart(anima_name: str) -> None:
            supervisor.processes[anima_name] = new_handle

        supervisor.restart_anima = AsyncMock(side_effect=mock_restart)

        with patch("core.supervisor.manager.asyncio.sleep", new_callable=AsyncMock):
            await supervisor._handle_process_failure(name, old_handle)

        # restart_anima was called (not blocked by _restarting guard)
        supervisor.restart_anima.assert_awaited_once_with(name)

        # processes[name] has the new handle (no KeyError)
        assert supervisor.processes[name] is new_handle

        # _restarting set is cleaned up in finally block
        assert name not in supervisor._restarting

    async def test_restarting_set_is_cleaned_even_on_restart_error(
        self, supervisor: ProcessSupervisor,
    ):
        """_restarting is cleaned up even if restart_anima raises."""
        name = "crash-anima"

        old_handle = MagicMock()
        old_handle.state = ProcessState.RUNNING
        supervisor.processes[name] = old_handle

        supervisor.restart_anima = AsyncMock(
            side_effect=RuntimeError("restart failed"),
        )

        with patch("core.supervisor.manager.asyncio.sleep", new_callable=AsyncMock):
            await supervisor._handle_process_failure(name, old_handle)

        # _restarting must be empty even after error
        assert name not in supervisor._restarting

        # State should be FAILED after exception
        assert old_handle.state == ProcessState.FAILED

    async def test_duplicate_failure_is_skipped_by_restarting_guard(
        self, supervisor: ProcessSupervisor,
    ):
        """Second concurrent call to _handle_process_failure returns immediately."""
        name = "double-fail"

        # Pre-mark as restarting (simulating a concurrent first call)
        supervisor._restarting.add(name)

        handle = MagicMock()
        supervisor.restart_anima = AsyncMock()

        with patch("core.supervisor.manager.asyncio.sleep", new_callable=AsyncMock):
            await supervisor._handle_process_failure(name, handle)

        # restart_anima should NOT have been called
        supervisor.restart_anima.assert_not_awaited()


class TestHangToRestartFlow:
    """_handle_process_hang kills the process then delegates to _handle_process_failure."""

    async def test_hang_kills_process_then_restarts(
        self, supervisor: ProcessSupervisor,
    ):
        """Hung process is killed and then restarted via _handle_process_failure."""
        name = "hung-anima"

        hung_handle = MagicMock()
        hung_handle.kill = AsyncMock()
        hung_handle.state = ProcessState.RUNNING
        supervisor.processes[name] = hung_handle

        # Set up restart side-effect
        new_handle = MagicMock()
        new_handle.get_pid.return_value = 88888
        new_handle.state = ProcessState.RUNNING

        async def mock_restart(anima_name: str) -> None:
            supervisor.processes[anima_name] = new_handle

        supervisor.restart_anima = AsyncMock(side_effect=mock_restart)

        with patch("core.supervisor.manager.asyncio.sleep", new_callable=AsyncMock):
            await supervisor._handle_process_hang(name, hung_handle)

        # kill() was called on the hung handle
        hung_handle.kill.assert_awaited_once()

        # restart_anima was called (via _handle_process_failure)
        supervisor.restart_anima.assert_awaited_once_with(name)

        # New handle is in processes
        assert supervisor.processes[name] is new_handle

        # _restarting cleaned up
        assert name not in supervisor._restarting


class TestMaxRetriesExceeded:
    """When restart count reaches max, state should be FAILED without restart."""

    async def test_max_retries_sets_failed_without_restart(
        self, supervisor: ProcessSupervisor,
    ):
        """_handle_process_failure sets FAILED when max retries exceeded."""
        name = "exhausted-anima"

        handle = MagicMock()
        handle.state = ProcessState.RUNNING
        supervisor.processes[name] = handle

        # Set restart count to max (default max_retries=5)
        supervisor._restart_counts[name] = supervisor.restart_policy.max_retries

        supervisor.restart_anima = AsyncMock()

        with patch("core.supervisor.manager.asyncio.sleep", new_callable=AsyncMock):
            await supervisor._handle_process_failure(name, handle)

        # restart_anima should NOT be called
        supervisor.restart_anima.assert_not_awaited()

        # State should be FAILED
        assert handle.state == ProcessState.FAILED

        # _restarting cleaned up
        assert name not in supervisor._restarting

    async def test_incremental_restart_count(
        self, supervisor: ProcessSupervisor,
    ):
        """Each successful restart increments _restart_counts."""
        name = "counting-anima"

        handle = MagicMock()
        handle.state = ProcessState.RUNNING
        supervisor.processes[name] = handle

        new_handle = MagicMock()
        new_handle.get_pid.return_value = 77777
        new_handle.state = ProcessState.RUNNING

        async def mock_restart(anima_name: str) -> None:
            supervisor.processes[anima_name] = new_handle

        supervisor.restart_anima = AsyncMock(side_effect=mock_restart)

        # Start with count=0 (no previous restarts)
        assert supervisor._restart_counts.get(name, 0) == 0

        with patch("core.supervisor.manager.asyncio.sleep", new_callable=AsyncMock):
            await supervisor._handle_process_failure(name, handle)

        # Count should now be 1
        assert supervisor._restart_counts[name] == 1
