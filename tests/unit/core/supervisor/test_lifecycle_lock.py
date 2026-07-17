# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for per-anima lifecycle lock (start/stop serialization).

Covers TOCTOU start-vs-disable, handle-identity-safe pop on stop, and
respawn clean-skip when anima is disabled.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.supervisor.manager import ProcessSupervisor


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


def _write_status(animas_dir: Path, name: str, *, enabled: bool) -> Path:
    anima_dir = animas_dir / name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "status.json").write_text(
        json.dumps({"enabled": enabled}),
        encoding="utf-8",
    )
    return anima_dir


class TestStartStopLifecycleLock:
    """start_anima and stop_anima share per-anima lifecycle lock."""

    @pytest.mark.asyncio
    async def test_start_disabled_mid_start_does_not_register(
        self,
        supervisor: ProcessSupervisor,
    ) -> None:
        """TOCTOU: disable during slow handle.start → process not left running.

        start_anima holds lifecycle lock during start. disable writes
        enabled=false then stop_anima waits for the lock, then no-ops if
        start already refused — or stops if start registered after re-check.

        Scenario: start begins while enabled; under lock we re-check and
        if disabled mid-flight before processes[name]=handle, refuse.
        Here we disable *before* start so start refuses under lock.
        Concurrent case: slow start, disable stop waits, then cleans up.
        """
        name = "test-anima"
        _write_status(supervisor.animas_dir, name, enabled=True)

        start_entered = asyncio.Event()
        allow_start_finish = asyncio.Event()

        async def slow_start(*_args, **_kwargs):
            start_entered.set()
            await allow_start_finish.wait()

        with patch("core.supervisor.manager.ProcessHandle") as MockHandle:
            mock_handle = AsyncMock()
            mock_handle.start = AsyncMock(side_effect=slow_start)
            mock_handle.stop = AsyncMock()
            mock_handle.get_pid = MagicMock(return_value=111)
            mock_handle.send_request = AsyncMock(
                return_value=MagicMock(error=None, result={"needs_bootstrap": False}),
            )
            MockHandle.return_value = mock_handle

            start_task = asyncio.create_task(supervisor.start_anima(name))
            await asyncio.wait_for(start_entered.wait(), timeout=2.0)

            # While start holds lifecycle lock (inside handle.start), flip
            # disabled and request stop (queues on same lock).
            _write_status(supervisor.animas_dir, name, enabled=False)
            stop_task = asyncio.create_task(supervisor.stop_anima(name))
            await asyncio.sleep(0.05)

            allow_start_finish.set()
            results = await asyncio.gather(start_task, stop_task, return_exceptions=True)

        assert all(r is None for r in results), f"unexpected: {results}"
        # Final state: no live process (stop after start registration, or
        # start never registered — both acceptable; must not leave orphan).
        assert name not in supervisor.processes


class TestStopHandleIdentityGuard:
    """stop_anima must not remove a handle installed after the stop completes."""

    @pytest.mark.asyncio
    async def test_concurrent_stop_then_start_leaves_new_handle(
        self,
        supervisor: ProcessSupervisor,
    ) -> None:
        """Real lock order: stop holds lifecycle lock during drain; start waits.

        After stop pops the old handle and releases the lock, start registers a
        new handle. The new handle must remain (stop must not leave processes empty).
        """
        name = "test-anima"
        _write_status(supervisor.animas_dir, name, enabled=True)

        drain_started = asyncio.Event()
        allow_drain_finish = asyncio.Event()

        old_handle = MagicMock()

        async def slow_stop(*_args, **_kwargs):
            drain_started.set()
            await allow_drain_finish.wait()

        old_handle.stop = AsyncMock(side_effect=slow_stop)
        supervisor.processes[name] = old_handle

        with patch("core.supervisor.manager.ProcessHandle") as MockHandle:
            new_handle = AsyncMock()
            new_handle.start = AsyncMock()
            new_handle.stop = AsyncMock()
            new_handle.get_pid = MagicMock(return_value=222)
            new_handle.send_request = AsyncMock(
                return_value=MagicMock(error=None, result={"needs_bootstrap": False}),
            )
            MockHandle.return_value = new_handle

            stop_task = asyncio.create_task(supervisor.stop_anima(name))
            await asyncio.wait_for(drain_started.wait(), timeout=2.0)

            # start queues on the same lifecycle lock held by stop
            start_task = asyncio.create_task(supervisor.start_anima(name))
            await asyncio.sleep(0.05)
            # While stop drains under lock, processes still has old handle
            assert supervisor.processes.get(name) is old_handle
            assert not start_task.done()

            allow_drain_finish.set()
            results = await asyncio.gather(stop_task, start_task, return_exceptions=True)

        assert all(r is None for r in results), f"unexpected: {results}"
        old_handle.stop.assert_awaited_once()
        # New handle from start remains after stop's identity-safe pop of old
        assert supervisor.processes.get(name) is new_handle


class TestStartDuringShutdown:
    """start_anima must not register processes while supervisor is shutting down."""

    @pytest.mark.asyncio
    async def test_start_anima_skips_when_shutdown(
        self,
        supervisor: ProcessSupervisor,
    ) -> None:
        name = "test-anima"
        _write_status(supervisor.animas_dir, name, enabled=True)
        supervisor._shutdown = True

        with patch("core.supervisor.manager.ProcessHandle") as MockHandle:
            await supervisor.start_anima(name)

        MockHandle.assert_not_called()
        assert name not in supervisor.processes


class TestRespawnDisabledCleanSkip:
    """Disabled respawn must not pollute failure state."""

    @pytest.mark.asyncio
    async def test_respawn_disabled_returns_none_without_pollution(
        self,
        supervisor: ProcessSupervisor,
    ) -> None:
        name = "test-anima"
        _write_status(supervisor.animas_dir, name, enabled=False)
        supervisor.restart_policy.max_retries = 2

        # Pre-seed counters to prove they are not written by disabled path
        assert name not in supervisor._start_fail_counts
        assert name not in supervisor._failure_reasons
        assert name not in supervisor._permanently_failed

        with patch("core.supervisor.manager.ProcessHandle") as MockHandle:
            result = await supervisor._respawn_anima_transaction(name)

        assert result is None
        MockHandle.assert_not_called()
        assert name not in supervisor._start_fail_counts
        assert name not in supervisor._failure_reasons
        assert name not in supervisor._permanently_failed
        assert name not in supervisor.processes

    @pytest.mark.asyncio
    async def test_handle_process_failure_disabled_no_restart_count(
        self,
        supervisor: ProcessSupervisor,
    ) -> None:
        from core.supervisor.process_handle import ProcessState

        name = "test-anima"
        _write_status(supervisor.animas_dir, name, enabled=False)
        supervisor.restart_policy.max_retries = 3

        old_handle = MagicMock()
        old_handle.state = ProcessState.FAILED
        supervisor.processes[name] = old_handle

        async def mock_stop(anima_name: str, **_kwargs) -> None:
            supervisor.processes.pop(anima_name, None)

        supervisor.stop_anima = AsyncMock(side_effect=mock_stop)

        with patch("core.supervisor.manager.asyncio.sleep", new_callable=AsyncMock):
            await supervisor._handle_process_failure(name, old_handle)

        assert name not in supervisor.processes
        assert name not in supervisor._restarting
        assert name not in supervisor._restart_counts
        assert name not in supervisor._permanently_failed
        assert name not in supervisor._start_fail_counts
        assert name not in supervisor._failure_reasons

    @pytest.mark.asyncio
    async def test_handle_process_failure_disabled_at_max_retries_no_permanent(
        self,
        supervisor: ProcessSupervisor,
    ) -> None:
        """Disabled + already at max retries must not enter _permanently_failed."""
        from core.supervisor.process_handle import ProcessState

        name = "test-anima"
        _write_status(supervisor.animas_dir, name, enabled=False)
        supervisor.restart_policy.max_retries = 3
        # Already exhausted retries — previously would hit _mark_process_error first
        supervisor._restart_counts[name] = 3

        old_handle = MagicMock()
        old_handle.state = ProcessState.FAILED
        supervisor.processes[name] = old_handle

        async def mock_stop(anima_name: str, **_kwargs) -> None:
            supervisor.processes.pop(anima_name, None)

        supervisor.stop_anima = AsyncMock(side_effect=mock_stop)

        with patch("core.supervisor.manager.asyncio.sleep", new_callable=AsyncMock):
            await supervisor._handle_process_failure(name, old_handle)

        assert name not in supervisor.processes
        assert name not in supervisor._restarting
        assert name not in supervisor._permanently_failed
        assert name not in supervisor._failure_reasons
        supervisor.stop_anima.assert_awaited_once_with(name)


class TestShutdownDuringInFlightStart:
    """shutdown_all racing an in-flight start must not orphan the runner."""

    @pytest.mark.asyncio
    async def test_shutdown_during_slow_start_stops_and_unregisters(
        self,
        supervisor: ProcessSupervisor,
    ) -> None:
        """_shutdown set while handle.start() awaits → handle stopped, not registered."""
        name = "test-anima"
        _write_status(supervisor.animas_dir, name, enabled=True)

        start_entered = asyncio.Event()
        allow_start_finish = asyncio.Event()

        async def slow_start(*_args, **_kwargs):
            start_entered.set()
            await allow_start_finish.wait()

        with patch("core.supervisor.manager.ProcessHandle") as MockHandle:
            mock_handle = AsyncMock()
            mock_handle.start = AsyncMock(side_effect=slow_start)
            mock_handle.stop = AsyncMock()
            mock_handle.get_pid = MagicMock(return_value=222)
            MockHandle.return_value = mock_handle

            start_task = asyncio.create_task(supervisor.start_anima(name))
            await asyncio.wait_for(start_entered.wait(), timeout=2.0)

            # shutdown_all's snapshot happens while start is suspended: the
            # process is not yet registered, so only the flag protects us.
            supervisor._shutdown = True
            allow_start_finish.set()
            await start_task

        assert name not in supervisor.processes
        mock_handle.stop.assert_awaited_once()


class TestDisableMidRespawnCountRollback:
    """_restart_counts must not stay incremented on a disabled clean-skip."""

    @pytest.mark.asyncio
    async def test_disable_mid_respawn_rolls_back_restart_count(
        self,
        supervisor: ProcessSupervisor,
    ) -> None:
        """Disable lands between count increment and respawn → count rolled back."""
        from core.supervisor.process_handle import ProcessState

        name = "test-anima"
        _write_status(supervisor.animas_dir, name, enabled=True)
        supervisor.restart_policy.max_retries = 3
        supervisor._maybe_repair_rag_before_restart = AsyncMock(return_value=False)

        old_handle = MagicMock()
        old_handle.state = ProcessState.FAILED

        async def respawn_with_disable(anima_name: str):
            # Simulate disable arriving while respawn is in flight; the
            # production path then clean-skips and returns None.
            _write_status(supervisor.animas_dir, anima_name, enabled=False)
            return None

        supervisor._respawn_anima_transaction = AsyncMock(side_effect=respawn_with_disable)

        with patch("core.supervisor.manager.asyncio.sleep", new_callable=AsyncMock):
            await supervisor._handle_process_failure(name, old_handle)

        assert name not in supervisor._restart_counts
        assert name not in supervisor._permanently_failed
        assert name not in supervisor._restarting
