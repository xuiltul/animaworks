"""Unit tests for ProcessSupervisor zombie reaper loop."""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core.supervisor.manager import ProcessSupervisor


@pytest.fixture
def supervisor(tmp_path: Path) -> ProcessSupervisor:
    """Create a minimal ProcessSupervisor."""
    return ProcessSupervisor(
        animas_dir=tmp_path / "animas",
        shared_dir=tmp_path / "shared",
        run_dir=tmp_path / "run",
        log_dir=tmp_path / "logs",
    )


class TestZombieReaperLoop:
    """Tests for _zombie_reaper_loop() in ProcessSupervisor."""

    @pytest.mark.asyncio
    async def test_reaper_reaps_zombies(self, supervisor: ProcessSupervisor):
        """Only manager-owned Popen objects may consume their wait status."""
        exited = MagicMock(returncode=None)
        exited.poll.return_value = 23
        alive = MagicMock(returncode=None)
        alive.poll.return_value = None
        already_reaped = MagicMock(returncode=17)
        supervisor.processes = {
            "exited": SimpleNamespace(process=exited),
            "alive": SimpleNamespace(process=alive),
            "already_reaped": SimpleNamespace(process=already_reaped),
            "not_started": SimpleNamespace(process=None),
        }

        original_sleep = asyncio.sleep

        async def shutdown_after_one_cycle(duration):
            supervisor._shutdown = True
            await original_sleep(0)

        with (
            patch("os.waitpid") as waitpid,
            patch.object(asyncio, "sleep", side_effect=shutdown_after_one_cycle),
        ):
            await supervisor._zombie_reaper_loop()

        exited.poll.assert_called_once_with()
        alive.poll.assert_called_once_with()
        already_reaped.poll.assert_not_called()
        waitpid.assert_not_called()

    @pytest.mark.asyncio
    async def test_reaper_handles_no_children(self, supervisor: ProcessSupervisor):
        """An empty ownership map must never reap another component's child."""
        original_sleep = asyncio.sleep

        async def shutdown_after_one_cycle(duration):
            supervisor._shutdown = True
            await original_sleep(0)

        with (
            patch("os.waitpid") as waitpid,
            patch.object(asyncio, "sleep", side_effect=shutdown_after_one_cycle),
        ):
            await supervisor._zombie_reaper_loop()
        waitpid.assert_not_called()

    @pytest.mark.asyncio
    async def test_reaper_stops_on_cancel(self, supervisor: ProcessSupervisor):
        """Zombie reaper should exit cleanly on CancelledError."""

        async def cancel_sleep(_duration):
            raise asyncio.CancelledError()

        with patch.object(asyncio, "sleep", side_effect=cancel_sleep):
            await supervisor._zombie_reaper_loop()

    @pytest.mark.asyncio
    async def test_reaper_survives_unexpected_exception(self, supervisor: ProcessSupervisor):
        """One owner's polling error must not skip the other owners."""
        broken = MagicMock(returncode=None)
        broken.poll.side_effect = OSError("unexpected")
        healthy = MagicMock(returncode=None)
        healthy.poll.return_value = None
        supervisor.processes = {"broken": SimpleNamespace(process=broken), "healthy": SimpleNamespace(process=healthy)}
        cycle_count = 0
        original_sleep = asyncio.sleep

        async def counting_sleep(_duration):
            nonlocal cycle_count
            cycle_count += 1
            if cycle_count >= 2:
                supervisor._shutdown = True
            await original_sleep(0)

        with (
            patch.object(asyncio, "sleep", side_effect=counting_sleep),
        ):
            await supervisor._zombie_reaper_loop()

        assert cycle_count >= 2
        assert healthy.poll.call_count == cycle_count

    @pytest.mark.asyncio
    async def test_shutdown_all_cancels_reaper(self, supervisor: ProcessSupervisor):
        """shutdown_all() should cancel the zombie reaper task."""
        reaper_running = asyncio.Event()

        async def slow_reaper():
            reaper_running.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                pass

        supervisor._zombie_reaper_task = asyncio.create_task(slow_reaper())
        await reaper_running.wait()

        await supervisor.shutdown_all()

        assert supervisor._zombie_reaper_task.done()


@pytest.mark.skipif(not hasattr(os, "WNOWAIT"), reason="Requires non-consuming POSIX waitid")
@pytest.mark.parametrize("returncode", [0, 23, -signal.SIGTERM])
async def test_reaper_preserves_real_exit_status_and_foreign_child(supervisor, returncode):
    """A generic waitpid(-1) would turn both real statuses into false zeroes."""
    code = (
        f"raise SystemExit({returncode})"
        if returncode >= 0
        else "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"
    )
    original_sleep = asyncio.sleep

    async def one_cycle(_duration):
        supervisor._shutdown = True
        await original_sleep(0)

    with (
        subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as owned,
        subprocess.Popen(
            [sys.executable, "-c", "raise SystemExit(17)"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ) as foreign,
    ):
        # Observe death without consuming status, just as the old reaper would
        # encounter a zombie before its real subprocess owner polls it.
        os.waitid(os.P_PID, owned.pid, os.WEXITED | os.WNOWAIT)
        os.waitid(os.P_PID, foreign.pid, os.WEXITED | os.WNOWAIT)
        supervisor.processes = {"owned": SimpleNamespace(process=owned)}
        with patch.object(asyncio, "sleep", side_effect=one_cycle):
            await supervisor._zombie_reaper_loop()
        assert owned.returncode == returncode
        assert owned.wait(timeout=2) == returncode
        assert foreign.returncode is None
        assert foreign.wait(timeout=2) == 17
