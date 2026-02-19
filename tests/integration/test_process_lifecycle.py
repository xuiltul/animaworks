"""Integration tests for process lifecycle (start, stop, restart)."""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

psutil = pytest.importorskip("psutil")

from core.supervisor.manager import ProcessSupervisor
from core.supervisor.process_handle import ProcessState


@pytest.mark.asyncio
async def test_start_anima_process(data_dir: Path, make_anima):
    """Test starting a anima process."""
    # Create a test anima
    anima_dir = make_anima("test-anima")

    # Create supervisor
    supervisor = ProcessSupervisor(
        animas_dir=data_dir / "animas",
        shared_dir=data_dir / "shared",
        run_dir=data_dir / "run",
        log_dir=data_dir / "logs"
    )

    try:
        # Start anima
        await supervisor.start_anima("test-anima")

        # Verify process is running
        handle = supervisor.processes.get("test-anima")
        assert handle is not None
        assert handle.state == ProcessState.RUNNING
        assert handle.get_pid() > 0

        # Verify socket file exists
        socket_path = data_dir / "run" / "sockets" / "test-anima.sock"
        assert socket_path.exists()

    finally:
        # Cleanup
        await supervisor.shutdown_all()


@pytest.mark.asyncio
async def test_stop_anima_process(data_dir: Path, make_anima):
    """Test stopping a anima process."""
    make_anima("test-anima")

    supervisor = ProcessSupervisor(
        animas_dir=data_dir / "animas",
        shared_dir=data_dir / "shared",
        run_dir=data_dir / "run",
        log_dir=data_dir / "logs"
    )

    try:
        # Start anima
        await supervisor.start_anima("test-anima")
        handle = supervisor.processes.get("test-anima")
        assert handle is not None

        pid = handle.get_pid()

        # Stop anima
        await supervisor.stop_anima("test-anima")

        # Verify process stopped
        assert "test-anima" not in supervisor.processes
        assert handle.state == ProcessState.STOPPED

        # Verify process actually exited
        assert not psutil.pid_exists(pid)

    finally:
        await supervisor.shutdown_all()


@pytest.mark.asyncio
async def test_restart_anima_process(data_dir: Path, make_anima):
    """Test restarting a anima process."""
    make_anima("test-anima")

    supervisor = ProcessSupervisor(
        animas_dir=data_dir / "animas",
        shared_dir=data_dir / "shared",
        run_dir=data_dir / "run",
        log_dir=data_dir / "logs"
    )

    try:
        # Start anima
        await supervisor.start_anima("test-anima")
        handle = supervisor.processes.get("test-anima")
        original_pid = handle.get_pid()

        # Restart anima
        await supervisor.restart_anima("test-anima")

        # Verify new process started
        handle = supervisor.processes.get("test-anima")
        assert handle is not None
        assert handle.state == ProcessState.RUNNING

        new_pid = handle.get_pid()
        assert new_pid != original_pid  # Different process

    finally:
        await supervisor.shutdown_all()


@pytest.mark.asyncio
async def test_start_multiple_animas(data_dir: Path, make_anima):
    """Test starting multiple anima processes simultaneously."""
    make_anima("alice")
    make_anima("bob")
    make_anima("charlie")

    supervisor = ProcessSupervisor(
        animas_dir=data_dir / "animas",
        shared_dir=data_dir / "shared",
        run_dir=data_dir / "run",
        log_dir=data_dir / "logs"
    )

    try:
        # Start all animas
        await supervisor.start_all(["alice", "bob", "charlie"])

        # Verify all are running
        assert len(supervisor.processes) == 3
        for name in ["alice", "bob", "charlie"]:
            handle = supervisor.processes.get(name)
            assert handle is not None
            assert handle.state == ProcessState.RUNNING
            assert handle.get_pid() > 0

    finally:
        await supervisor.shutdown_all()


@pytest.mark.asyncio
async def test_graceful_shutdown_all(data_dir: Path, make_anima):
    """Test graceful shutdown of all processes."""
    make_anima("alice")
    make_anima("bob")

    supervisor = ProcessSupervisor(
        animas_dir=data_dir / "animas",
        shared_dir=data_dir / "shared",
        run_dir=data_dir / "run",
        log_dir=data_dir / "logs"
    )

    # Start animas
    await supervisor.start_all(["alice", "bob"])

    pids = [
        supervisor.processes["alice"].get_pid(),
        supervisor.processes["bob"].get_pid()
    ]

    # Shutdown all
    await supervisor.shutdown_all()

    # Verify all processes stopped
    assert len(supervisor.processes) == 0

    # Verify processes actually exited
    for pid in pids:
        assert not psutil.pid_exists(pid)
