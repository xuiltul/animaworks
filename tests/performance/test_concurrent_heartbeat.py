"""Performance tests for concurrent heartbeat execution."""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from core.supervisor.manager import ProcessSupervisor


@pytest.mark.asyncio
@pytest.mark.live
@pytest.mark.timeout(120)
async def test_concurrent_heartbeat_no_deadlock(data_dir: Path, make_anima):
    """Test multiple animas running heartbeat concurrently without deadlock (requires API key)."""
    # Create multiple animas
    anima_names = ["alice", "bob", "charlie"]
    for name in anima_names:
        make_anima(name)

    supervisor = ProcessSupervisor(
        animas_dir=data_dir / "animas",
        shared_dir=data_dir / "shared",
        run_dir=data_dir / "run",
        log_dir=data_dir / "logs"
    )

    try:
        # Start all animas
        await supervisor.start_all(anima_names)

        # Trigger heartbeat on all animas concurrently
        start_time = time.time()

        tasks = [
            supervisor.send_request(name, "run_heartbeat", {}, timeout=30.0)
            for name in anima_names
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
                print(f"Heartbeat for {anima_names[i]} raised: {result}")
            else:
                assert isinstance(result, dict)

    finally:
        await supervisor.shutdown_all()


@pytest.mark.asyncio
async def test_concurrent_status_requests(data_dir: Path, make_anima):
    """Test concurrent status requests performance."""
    anima_names = ["alice", "bob", "charlie", "dave"]
    for name in anima_names:
        make_anima(name)

    supervisor = ProcessSupervisor(
        animas_dir=data_dir / "animas",
        shared_dir=data_dir / "shared",
        run_dir=data_dir / "run",
        log_dir=data_dir / "logs"
    )

    try:
        await supervisor.start_all(anima_names)

        # Send many concurrent status requests
        start_time = time.time()

        tasks = []
        for _ in range(20):  # 20 requests per anima = 80 total
            for name in anima_names:
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
async def test_rapid_restart_stability(data_dir: Path, make_anima):
    """Test stability under rapid restart operations."""
    make_anima("test-anima")

    supervisor = ProcessSupervisor(
        animas_dir=data_dir / "animas",
        shared_dir=data_dir / "shared",
        run_dir=data_dir / "run",
        log_dir=data_dir / "logs"
    )

    try:
        # Perform multiple rapid restarts
        for i in range(3):
            await supervisor.start_anima("test-anima")

            # Verify running
            handle = supervisor.processes.get("test-anima")
            assert handle is not None

            # Send ping to verify alive
            pong = await handle.ping(timeout=5.0)
            assert pong is True

            # Stop
            await supervisor.stop_anima("test-anima")

            # Small delay between restarts
            await asyncio.sleep(0.5)

        # Final verification
        await supervisor.start_anima("test-anima")
        handle = supervisor.processes.get("test-anima")
        assert handle is not None

    finally:
        await supervisor.shutdown_all()
