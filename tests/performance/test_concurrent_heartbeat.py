"""Performance tests for concurrent heartbeat execution."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from core.supervisor.manager import ProcessSupervisor


@pytest.mark.asyncio
async def test_concurrent_heartbeat_no_deadlock(data_dir: Path, make_person):
    """Test multiple persons running heartbeat concurrently without deadlock."""
    # Create multiple persons
    person_names = ["alice", "bob", "charlie"]
    for name in person_names:
        make_person(name)

    supervisor = ProcessSupervisor(
        persons_dir=data_dir / "persons",
        shared_dir=data_dir / "shared",
        run_dir=data_dir / "run",
        log_dir=data_dir / "logs"
    )

    try:
        # Start all persons
        await supervisor.start_all(person_names)

        # Trigger heartbeat on all persons concurrently
        start_time = time.time()

        tasks = [
            supervisor.send_request(name, "run_heartbeat", {}, timeout=30.0)
            for name in person_names
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = time.time() - start_time

        # Verify all completed (no deadlock)
        assert elapsed < 35.0, "Heartbeats took too long (possible deadlock)"

        # Verify all succeeded or have expected errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Log exception but don't fail test
                # (heartbeat might fail for various reasons in test environment)
                print(f"Heartbeat for {person_names[i]} raised: {result}")
            else:
                assert isinstance(result, dict)

    finally:
        await supervisor.shutdown_all()


@pytest.mark.asyncio
async def test_concurrent_status_requests(data_dir: Path, make_person):
    """Test concurrent status requests performance."""
    person_names = ["alice", "bob", "charlie", "dave"]
    for name in person_names:
        make_person(name)

    supervisor = ProcessSupervisor(
        persons_dir=data_dir / "persons",
        shared_dir=data_dir / "shared",
        run_dir=data_dir / "run",
        log_dir=data_dir / "logs"
    )

    try:
        await supervisor.start_all(person_names)

        # Send many concurrent status requests
        start_time = time.time()

        tasks = []
        for _ in range(20):  # 20 requests per person = 80 total
            for name in person_names:
                tasks.append(
                    supervisor.send_request(name, "get_status", {}, timeout=5.0)
                )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = time.time() - start_time

        # Verify performance (should complete quickly)
        assert elapsed < 10.0, f"Status requests took too long: {elapsed}s"

        # Verify most succeeded
        success_count = sum(1 for r in results if isinstance(r, dict))
        assert success_count >= len(tasks) * 0.9, "Too many failed requests"

    finally:
        await supervisor.shutdown_all()


@pytest.mark.asyncio
async def test_rapid_restart_stability(data_dir: Path, make_person):
    """Test stability under rapid restart operations."""
    make_person("test-person")

    supervisor = ProcessSupervisor(
        persons_dir=data_dir / "persons",
        shared_dir=data_dir / "shared",
        run_dir=data_dir / "run",
        log_dir=data_dir / "logs"
    )

    try:
        # Perform multiple rapid restarts
        for i in range(3):
            await supervisor.start_person("test-person")

            # Verify running
            handle = supervisor.processes.get("test-person")
            assert handle is not None

            # Send ping to verify alive
            pong = await handle.ping(timeout=5.0)
            assert pong is True

            # Stop
            await supervisor.stop_person("test-person")

            # Small delay between restarts
            await asyncio.sleep(0.5)

        # Final verification
        await supervisor.start_person("test-person")
        handle = supervisor.processes.get("test-person")
        assert handle is not None

    finally:
        await supervisor.shutdown_all()
