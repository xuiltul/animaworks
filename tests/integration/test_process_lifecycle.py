"""Integration tests for process lifecycle (start, stop, restart)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

psutil = pytest.importorskip("psutil")

from core.supervisor.manager import ProcessSupervisor
from core.supervisor.process_handle import ProcessState


@pytest.mark.asyncio
async def test_start_person_process(data_dir: Path, make_person):
    """Test starting a person process."""
    # Create a test person
    person_dir = make_person("test-person")

    # Create supervisor
    supervisor = ProcessSupervisor(
        persons_dir=data_dir / "persons",
        shared_dir=data_dir / "shared",
        run_dir=data_dir / "run",
        log_dir=data_dir / "logs"
    )

    try:
        # Start person
        await supervisor.start_person("test-person")

        # Verify process is running
        handle = supervisor.processes.get("test-person")
        assert handle is not None
        assert handle.state == ProcessState.RUNNING
        assert handle.get_pid() > 0

        # Verify socket file exists
        socket_path = data_dir / "run" / "sockets" / "test-person.sock"
        assert socket_path.exists()

    finally:
        # Cleanup
        await supervisor.shutdown_all()


@pytest.mark.asyncio
async def test_stop_person_process(data_dir: Path, make_person):
    """Test stopping a person process."""
    make_person("test-person")

    supervisor = ProcessSupervisor(
        persons_dir=data_dir / "persons",
        shared_dir=data_dir / "shared",
        run_dir=data_dir / "run",
        log_dir=data_dir / "logs"
    )

    try:
        # Start person
        await supervisor.start_person("test-person")
        handle = supervisor.processes.get("test-person")
        assert handle is not None

        pid = handle.get_pid()

        # Stop person
        await supervisor.stop_person("test-person")

        # Verify process stopped
        assert "test-person" not in supervisor.processes
        assert handle.state == ProcessState.STOPPED

        # Verify process actually exited
        assert not psutil.pid_exists(pid)

    finally:
        await supervisor.shutdown_all()


@pytest.mark.asyncio
async def test_restart_person_process(data_dir: Path, make_person):
    """Test restarting a person process."""
    make_person("test-person")

    supervisor = ProcessSupervisor(
        persons_dir=data_dir / "persons",
        shared_dir=data_dir / "shared",
        run_dir=data_dir / "run",
        log_dir=data_dir / "logs"
    )

    try:
        # Start person
        await supervisor.start_person("test-person")
        handle = supervisor.processes.get("test-person")
        original_pid = handle.get_pid()

        # Restart person
        await supervisor.restart_person("test-person")

        # Verify new process started
        handle = supervisor.processes.get("test-person")
        assert handle is not None
        assert handle.state == ProcessState.RUNNING

        new_pid = handle.get_pid()
        assert new_pid != original_pid  # Different process

    finally:
        await supervisor.shutdown_all()


@pytest.mark.asyncio
async def test_start_multiple_persons(data_dir: Path, make_person):
    """Test starting multiple person processes simultaneously."""
    make_person("alice")
    make_person("bob")
    make_person("charlie")

    supervisor = ProcessSupervisor(
        persons_dir=data_dir / "persons",
        shared_dir=data_dir / "shared",
        run_dir=data_dir / "run",
        log_dir=data_dir / "logs"
    )

    try:
        # Start all persons
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
async def test_graceful_shutdown_all(data_dir: Path, make_person):
    """Test graceful shutdown of all processes."""
    make_person("alice")
    make_person("bob")

    supervisor = ProcessSupervisor(
        persons_dir=data_dir / "persons",
        shared_dir=data_dir / "shared",
        run_dir=data_dir / "run",
        log_dir=data_dir / "logs"
    )

    # Start persons
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
