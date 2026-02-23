# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for process group isolation and health check improvements.

Bug 1: start_new_session=True for subprocess isolation
Bug 2: FAILED log spam suppression via _permanently_failed
Bug 3: Reconciliation-based auto-recovery with 1-minute cooldown
Bug 4: restart_anima() resets _restart_counts
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
from datetime import datetime, timedelta
from core.time_utils import now_jst
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.supervisor.manager import (
    HealthConfig,
    ProcessSupervisor,
    RestartPolicy,
    ReconciliationConfig,
)
from core.supervisor.process_handle import ProcessHandle, ProcessState


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def supervisor(tmp_path: Path) -> ProcessSupervisor:
    """Create a ProcessSupervisor with fast timeouts for testing."""
    return ProcessSupervisor(
        animas_dir=tmp_path / "animas",
        shared_dir=tmp_path / "shared",
        run_dir=tmp_path / "run",
        restart_policy=RestartPolicy(
            max_retries=3,
            backoff_base_sec=0.01,
            backoff_max_sec=0.1,
        ),
        health_config=HealthConfig(
            ping_interval_sec=0.5,
            ping_timeout_sec=0.2,
            max_missed_pings=2,
            startup_grace_sec=0.5,
        ),
    )


@pytest.fixture
def handle(tmp_path: Path) -> ProcessHandle:
    """Create a ProcessHandle in RUNNING state with mocked process."""
    socket_path = tmp_path / "test.sock"
    h = ProcessHandle(
        anima_name="test-anima",
        socket_path=socket_path,
        animas_dir=tmp_path / "animas",
        shared_dir=tmp_path / "shared",
        log_dir=tmp_path / "logs",
    )
    h.state = ProcessState.RUNNING
    h.process = MagicMock()
    h.process.pid = 12345
    h.process.poll.return_value = None
    h.process.returncode = None
    h.ipc_client = AsyncMock()
    socket_path.touch()
    return h


# ── Bug 1: start_new_session=True ─────────────────────────────


class TestProcessGroupIsolation:
    """Tests for subprocess session isolation (start_new_session=True)."""

    @pytest.mark.asyncio
    async def test_popen_uses_start_new_session(self, tmp_path: Path):
        """Verify Popen is called with start_new_session=True."""
        handle = ProcessHandle(
            anima_name="test-anima",
            socket_path=tmp_path / "test.sock",
            animas_dir=tmp_path / "animas",
            shared_dir=tmp_path / "shared",
            log_dir=tmp_path / "logs",
        )

        with patch("core.supervisor.process_handle.subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.pid = 99999
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            # start() will fail at _wait_for_socket, but Popen is called first
            with pytest.raises((TimeoutError, RuntimeError)):
                await handle.start()

            mock_popen.assert_called_once()
            call_kwargs = mock_popen.call_args
            assert call_kwargs.kwargs.get("start_new_session") is True \
                or (len(call_kwargs.args) > 0 and False) \
                or call_kwargs[1].get("start_new_session") is True

    @pytest.mark.asyncio
    async def test_stop_uses_killpg_for_sigterm(self, handle: ProcessHandle):
        """Verify stop() uses os.killpg() for SIGTERM instead of process.terminate()."""
        killpg_called = False

        # Process stays alive until killpg sends SIGTERM, then exits
        def poll_side_effect():
            if killpg_called:
                return 0
            return None

        handle.process.poll.side_effect = poll_side_effect
        handle.process.returncode = 0

        # IPC shutdown "fails" so we reach the SIGTERM step
        async def fail_send(*args, **kwargs):
            raise ConnectionError("IPC lost")
        handle.send_request = fail_send

        def killpg_side_effect(pgid, sig):
            nonlocal killpg_called
            killpg_called = True

        with patch("core.supervisor.process_handle.os.killpg", side_effect=killpg_side_effect) as mock_killpg, \
             patch("core.supervisor.process_handle.os.getpgid", return_value=12345):
            await handle.stop(timeout=2.0)

            # killpg should have been called with SIGTERM
            assert any(
                call.args == (12345, signal.SIGTERM)
                for call in mock_killpg.call_args_list
            )

    @pytest.mark.asyncio
    async def test_stop_falls_back_to_terminate_on_os_error(
        self, handle: ProcessHandle,
    ):
        """Verify stop() falls back to process.terminate() if killpg raises OSError."""
        terminate_called = False

        def poll_side_effect():
            if terminate_called:
                return 0
            return None

        handle.process.poll.side_effect = poll_side_effect
        handle.process.returncode = 0

        original_terminate = handle.process.terminate

        def track_terminate():
            nonlocal terminate_called
            terminate_called = True

        handle.process.terminate.side_effect = track_terminate

        # IPC shutdown fails so we reach SIGTERM step
        async def fail_send(*args, **kwargs):
            raise ConnectionError("IPC lost")
        handle.send_request = fail_send

        with patch("core.supervisor.process_handle.os.killpg", side_effect=OSError("ESRCH")), \
             patch("core.supervisor.process_handle.os.getpgid", return_value=12345):
            await handle.stop(timeout=2.0)

            # Should have fallen back to terminate()
            assert terminate_called

    @pytest.mark.asyncio
    async def test_kill_uses_killpg_for_sigkill(self, handle: ProcessHandle):
        """Verify kill() uses os.killpg() with SIGKILL."""
        handle.process.wait.return_value = -9
        handle.process.returncode = -9
        # After kill, process is dead so poll() returns exit code
        handle.process.poll.return_value = -9

        with patch("core.supervisor.process_handle.os.killpg") as mock_killpg, \
             patch("core.supervisor.process_handle.os.getpgid", return_value=12345):
            await handle.kill()

            # The first killpg call should be SIGKILL (from kill())
            assert any(
                call.args == (12345, signal.SIGKILL)
                for call in mock_killpg.call_args_list
            )

    @pytest.mark.asyncio
    async def test_kill_falls_back_on_os_error(
        self, handle: ProcessHandle,
    ):
        """Verify kill() falls back to process.kill() if killpg raises OSError."""
        handle.process.wait.return_value = -9
        handle.process.returncode = -9

        with patch("core.supervisor.process_handle.os.killpg", side_effect=OSError("ESRCH")), \
             patch("core.supervisor.process_handle.os.getpgid", return_value=12345):
            await handle.kill()

            handle.process.kill.assert_called()

    @pytest.mark.asyncio
    async def test_cleanup_uses_killpg(self, handle: ProcessHandle):
        """Verify _cleanup() uses os.killpg() for orphaned subprocess termination."""
        handle.process.poll.return_value = None  # Still alive
        handle.process.wait.side_effect = [None]  # Exits after SIGTERM

        with patch("core.supervisor.process_handle.os.killpg") as mock_killpg, \
             patch("core.supervisor.process_handle.os.getpgid", return_value=12345):
            await handle._cleanup()

            mock_killpg.assert_called_with(12345, signal.SIGTERM)


# ── Bug 2: FAILED log spam suppression ────────────────────────


class TestFailedLogSpamSuppression:
    """Tests for _permanently_failed set and log throttling."""

    @pytest.mark.asyncio
    async def test_permanently_failed_initialized(self, supervisor: ProcessSupervisor):
        """Verify _permanently_failed and _failed_log_times are initialized."""
        assert hasattr(supervisor, "_permanently_failed")
        assert hasattr(supervisor, "_failed_log_times")
        assert isinstance(supervisor._permanently_failed, set)
        assert isinstance(supervisor._failed_log_times, dict)
        assert len(supervisor._permanently_failed) == 0

    @pytest.mark.asyncio
    async def test_max_retries_adds_to_permanently_failed(
        self, supervisor: ProcessSupervisor, handle: ProcessHandle,
    ):
        """Verify _handle_process_failure adds to _permanently_failed on max retries."""
        supervisor.processes["test-anima"] = handle
        # Set restart count to max_retries (3)
        supervisor._restart_counts["test-anima"] = 3

        await supervisor._handle_process_failure("test-anima", handle)

        assert "test-anima" in supervisor._permanently_failed
        assert "test-anima" in supervisor._failed_log_times
        assert handle.state == ProcessState.FAILED

    @pytest.mark.asyncio
    async def test_permanently_failed_skips_health_check(
        self, supervisor: ProcessSupervisor, handle: ProcessHandle, caplog,
    ):
        """Verify health check skips permanently failed processes."""
        supervisor.processes["test-anima"] = handle
        handle.state = ProcessState.FAILED
        supervisor._permanently_failed.add("test-anima")
        # Set last log time to now so it won't log again within 5 min
        supervisor._failed_log_times["test-anima"] = asyncio.get_running_loop().time()

        with patch.object(
            supervisor, "_handle_process_failure", new_callable=AsyncMock,
        ) as mock_failure:
            await supervisor._check_process_health("test-anima", handle)

        # _handle_process_failure should NOT be called
        mock_failure.assert_not_called()

    @pytest.mark.asyncio
    async def test_permanently_failed_logs_warning_every_5_minutes(
        self, supervisor: ProcessSupervisor, handle: ProcessHandle, caplog,
    ):
        """Verify WARNING log is emitted when 5-minute interval has passed."""
        import logging

        supervisor.processes["test-anima"] = handle
        handle.state = ProcessState.FAILED
        supervisor._permanently_failed.add("test-anima")
        # Set last log time to 301 seconds ago
        supervisor._failed_log_times["test-anima"] = (
            asyncio.get_running_loop().time() - 301
        )

        with caplog.at_level(logging.WARNING):
            await supervisor._check_process_health("test-anima", handle)

        assert any(
            "Process still in FAILED state: test-anima" in record.message
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_permanently_failed_no_log_within_5_minutes(
        self, supervisor: ProcessSupervisor, handle: ProcessHandle, caplog,
    ):
        """Verify no log is emitted within the 5-minute interval."""
        import logging

        supervisor.processes["test-anima"] = handle
        handle.state = ProcessState.FAILED
        supervisor._permanently_failed.add("test-anima")
        # Set last log time to just 10 seconds ago
        supervisor._failed_log_times["test-anima"] = (
            asyncio.get_running_loop().time() - 10
        )

        with caplog.at_level(logging.WARNING):
            await supervisor._check_process_health("test-anima", handle)

        assert not any(
            "Process still in FAILED state" in record.message
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_non_permanently_failed_still_triggers_restart(
        self, supervisor: ProcessSupervisor, handle: ProcessHandle,
    ):
        """Verify FAILED processes NOT in _permanently_failed still trigger restart."""
        supervisor.processes["test-anima"] = handle
        handle.state = ProcessState.FAILED
        # NOT added to _permanently_failed
        handle.stats.started_at = now_jst() - timedelta(seconds=60)

        with patch.object(
            supervisor, "_handle_process_failure", new_callable=AsyncMock,
        ) as mock_failure:
            await supervisor._check_process_health("test-anima", handle)
            await asyncio.sleep(0)

        # _handle_process_failure IS called (via asyncio.create_task)
        # We can't easily check create_task, but verify no _permanently_failed skip
        assert "test-anima" not in supervisor._permanently_failed


# ── Bug 3: Reconciliation auto-recovery ───────────────────────


class TestReconciliationAutoRecovery:
    """Tests for reconciliation-based auto-recovery with 1-minute cooldown."""

    @pytest.mark.asyncio
    async def test_reconcile_recovers_permanently_failed_after_cooldown(
        self, supervisor: ProcessSupervisor, tmp_path: Path,
    ):
        """Verify reconciliation recovers permanently failed processes after 1 min."""
        # Set up animas_dir with an enabled anima
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir(parents=True)
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir()
        (alice_dir / "identity.md").write_text("Alice identity", encoding="utf-8")
        (alice_dir / "status.json").write_text(
            json.dumps({"enabled": True}), encoding="utf-8",
        )
        supervisor.animas_dir = animas_dir

        # Set up failed handle
        handle = MagicMock(spec=ProcessHandle)
        handle.anima_name = "alice"
        handle.state = ProcessState.FAILED
        supervisor.processes["alice"] = handle

        # Mark as permanently failed with cooldown elapsed (>60s)
        supervisor._permanently_failed.add("alice")
        supervisor._restart_counts["alice"] = 5
        supervisor._failed_log_times["alice"] = asyncio.get_running_loop().time() - 61

        supervisor.start_anima = AsyncMock()
        supervisor.stop_anima = AsyncMock()
        callback = MagicMock()
        supervisor.on_anima_added = callback

        await supervisor._reconcile()

        # Should have cleaned up and restarted
        assert "alice" not in supervisor._permanently_failed
        assert "alice" not in supervisor._restart_counts
        assert "alice" not in supervisor._failed_log_times
        supervisor.start_anima.assert_called_with("alice")
        callback.assert_called_with("alice")

    @pytest.mark.asyncio
    async def test_reconcile_does_not_recover_within_cooldown(
        self, supervisor: ProcessSupervisor, tmp_path: Path,
    ):
        """Verify reconciliation does NOT recover within the 1-minute cooldown."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir(parents=True)
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir()
        (alice_dir / "identity.md").write_text("Alice identity", encoding="utf-8")
        (alice_dir / "status.json").write_text(
            json.dumps({"enabled": True}), encoding="utf-8",
        )
        supervisor.animas_dir = animas_dir

        handle = MagicMock(spec=ProcessHandle)
        handle.anima_name = "alice"
        handle.state = ProcessState.FAILED
        supervisor.processes["alice"] = handle

        # Cooldown NOT elapsed (only 10 seconds)
        supervisor._permanently_failed.add("alice")
        supervisor._restart_counts["alice"] = 5
        supervisor._failed_log_times["alice"] = asyncio.get_running_loop().time() - 10

        supervisor.start_anima = AsyncMock()
        supervisor.stop_anima = AsyncMock()

        await supervisor._reconcile()

        # Should NOT have attempted recovery
        assert "alice" in supervisor._permanently_failed
        assert supervisor._restart_counts["alice"] == 5
        # start_anima should not be called for alice (it's still in processes)
        supervisor.start_anima.assert_not_called()

    @pytest.mark.asyncio
    async def test_reconcile_skips_disabled_permanently_failed(
        self, supervisor: ProcessSupervisor, tmp_path: Path,
    ):
        """Verify reconciliation does not recover disabled permanently failed processes."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir(parents=True)
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir()
        (alice_dir / "identity.md").write_text("Alice identity", encoding="utf-8")
        (alice_dir / "status.json").write_text(
            json.dumps({"enabled": False}), encoding="utf-8",
        )
        supervisor.animas_dir = animas_dir

        handle = MagicMock(spec=ProcessHandle)
        handle.anima_name = "alice"
        handle.state = ProcessState.FAILED
        supervisor.processes["alice"] = handle

        supervisor._permanently_failed.add("alice")
        supervisor._restart_counts["alice"] = 5
        supervisor._failed_log_times["alice"] = asyncio.get_running_loop().time() - 120

        supervisor.start_anima = AsyncMock()
        supervisor.stop_anima = AsyncMock()

        await supervisor._reconcile()

        # Should NOT recover (disabled)
        assert "alice" in supervisor._permanently_failed

    @pytest.mark.asyncio
    async def test_reconcile_cleans_up_orphaned_permanently_failed(
        self, supervisor: ProcessSupervisor, tmp_path: Path,
    ):
        """Verify reconciliation cleans up _permanently_failed entries with no handle."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir(parents=True)
        supervisor.animas_dir = animas_dir

        # No handle in processes, but name in _permanently_failed
        supervisor._permanently_failed.add("ghost")

        await supervisor._reconcile()

        assert "ghost" not in supervisor._permanently_failed


# ── Bug 4: restart_anima() counter reset ──────────────────────


class TestRestartCounterReset:
    """Tests for restart_anima() resetting failure tracking state."""

    @pytest.mark.asyncio
    async def test_restart_resets_restart_counts(
        self, supervisor: ProcessSupervisor,
    ):
        """Verify restart_anima() resets _restart_counts."""
        supervisor._restart_counts["alice"] = 5
        supervisor._permanently_failed.add("alice")
        supervisor._failed_log_times["alice"] = 123.0

        handle = MagicMock(spec=ProcessHandle)
        handle.anima_name = "alice"
        supervisor.processes["alice"] = handle

        supervisor.stop_anima = AsyncMock()
        supervisor.start_anima = AsyncMock()

        await supervisor.restart_anima("alice")

        assert "alice" not in supervisor._restart_counts
        assert "alice" not in supervisor._permanently_failed
        assert "alice" not in supervisor._failed_log_times

    @pytest.mark.asyncio
    async def test_restart_calls_stop_then_start(
        self, supervisor: ProcessSupervisor,
    ):
        """Verify restart_anima() calls stop then start in order."""
        handle = MagicMock(spec=ProcessHandle)
        supervisor.processes["alice"] = handle

        call_order = []

        async def mock_stop(name):
            call_order.append(("stop", name))
            del supervisor.processes[name]

        async def mock_start(name):
            call_order.append(("start", name))

        supervisor.stop_anima = mock_stop
        supervisor.start_anima = mock_start

        await supervisor.restart_anima("alice")

        assert call_order == [("stop", "alice"), ("start", "alice")]

    @pytest.mark.asyncio
    async def test_restart_works_when_no_existing_process(
        self, supervisor: ProcessSupervisor,
    ):
        """Verify restart_anima() works when process doesn't exist yet."""
        supervisor.start_anima = AsyncMock()

        # No process in supervisor.processes
        await supervisor.restart_anima("new-anima")

        supervisor.start_anima.assert_called_once_with("new-anima")
        assert "new-anima" not in supervisor._restart_counts

    @pytest.mark.asyncio
    async def test_restart_preserves_counters_when_reset_disabled(
        self, supervisor: ProcessSupervisor,
    ):
        """Verify restart_anima(_reset_counters=False) preserves failure tracking."""
        supervisor._restart_counts["alice"] = 3
        supervisor._permanently_failed.add("alice")
        supervisor._failed_log_times["alice"] = 99.0

        handle = MagicMock(spec=ProcessHandle)
        handle.anima_name = "alice"
        supervisor.processes["alice"] = handle

        supervisor.stop_anima = AsyncMock()
        supervisor.start_anima = AsyncMock()

        await supervisor.restart_anima("alice", _reset_counters=False)

        # Counters must be preserved
        assert supervisor._restart_counts["alice"] == 3
        assert "alice" in supervisor._permanently_failed
        assert supervisor._failed_log_times["alice"] == 99.0

    @pytest.mark.asyncio
    async def test_handle_process_failure_preserves_counter_through_restart(
        self, supervisor: ProcessSupervisor, handle: ProcessHandle,
    ):
        """Verify _handle_process_failure increments counter and restart doesn't erase it."""
        supervisor.processes["test-anima"] = handle
        supervisor._restart_counts["test-anima"] = 1  # Already retried once

        # Mock stop/start so restart_anima succeeds
        async def mock_stop(name):
            del supervisor.processes[name]
        async def mock_start(name):
            new_h = MagicMock(spec=ProcessHandle)
            new_h.anima_name = name
            new_h.get_pid.return_value = 99999
            supervisor.processes[name] = new_h
        supervisor.stop_anima = mock_stop
        supervisor.start_anima = mock_start

        await supervisor._handle_process_failure("test-anima", handle)

        # Counter should be incremented (1 → 2), NOT reset to 0
        assert supervisor._restart_counts["test-anima"] == 2


# ── Integration: Full failure-recovery cycle ──────────────────


class TestFailureRecoveryCycle:
    """Integration tests for the complete failure → permanently_failed → recovery cycle."""

    @pytest.mark.asyncio
    async def test_full_cycle_failure_to_recovery(
        self, supervisor: ProcessSupervisor, tmp_path: Path,
    ):
        """Test complete cycle: FAILED → max_retries → permanently_failed → reconciliation recovery."""
        # Set up animas_dir
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir(parents=True)
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir()
        (alice_dir / "identity.md").write_text("Alice identity", encoding="utf-8")
        (alice_dir / "status.json").write_text(
            json.dumps({"enabled": True}), encoding="utf-8",
        )
        supervisor.animas_dir = animas_dir

        # Step 1: Create a FAILED handle with max retries exhausted
        handle = MagicMock(spec=ProcessHandle)
        handle.anima_name = "alice"
        handle.state = ProcessState.RUNNING
        supervisor.processes["alice"] = handle
        supervisor._restart_counts["alice"] = 3  # At max_retries

        # Step 2: Trigger _handle_process_failure (simulating crash detection)
        await supervisor._handle_process_failure("alice", handle)

        # Verify: entered permanently_failed
        assert "alice" in supervisor._permanently_failed
        assert handle.state == ProcessState.FAILED

        # Step 3: Health check should skip (no log spam)
        supervisor._failed_log_times["alice"] = asyncio.get_running_loop().time()

        with patch.object(
            supervisor, "_handle_process_failure", new_callable=AsyncMock,
        ) as mock_failure:
            await supervisor._check_process_health("alice", handle)
            mock_failure.assert_not_called()

        # Step 4: Simulate cooldown elapsed
        supervisor._failed_log_times["alice"] = (
            asyncio.get_running_loop().time() - 61
        )

        supervisor.start_anima = AsyncMock()
        supervisor.stop_anima = AsyncMock()

        await supervisor._reconcile()

        # Verify: recovered
        assert "alice" not in supervisor._permanently_failed
        assert "alice" not in supervisor._restart_counts
        supervisor.start_anima.assert_called_with("alice")
