"""Integration tests for IPC communication between parent and child processes."""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from core.supervisor.manager import ProcessSupervisor
from core.supervisor.process_handle import ProcessState


@pytest.mark.asyncio
async def test_ping_pong(data_dir: Path, make_anima):
    """Test ping/pong IPC communication."""
    make_anima("test-anima")

    supervisor = ProcessSupervisor(
        animas_dir=data_dir / "animas",
        shared_dir=data_dir / "shared",
        run_dir=data_dir / "run",
        log_dir=data_dir / "logs"
    )

    try:
        await supervisor.start_anima("test-anima")
        handle = supervisor.processes.get("test-anima")

        # Send ping
        pong = await handle.ping(timeout=5.0)

        # Verify pong received
        assert pong is True
        assert handle.stats.missed_pings == 0
        assert handle.stats.last_ping_at is not None

    finally:
        await supervisor.shutdown_all()


@pytest.mark.asyncio
async def test_get_status(data_dir: Path, make_anima):
    """Test get_status IPC request."""
    make_anima("test-anima")

    supervisor = ProcessSupervisor(
        animas_dir=data_dir / "animas",
        shared_dir=data_dir / "shared",
        run_dir=data_dir / "run",
        log_dir=data_dir / "logs"
    )

    try:
        await supervisor.start_anima("test-anima")

        # Send get_status request
        result = await supervisor.send_request("test-anima", "get_status", {})

        # Verify response
        assert "status" in result
        assert result["status"] in ["idle", "thinking", "working"]

    finally:
        await supervisor.shutdown_all()


@pytest.mark.asyncio
@pytest.mark.live
async def test_process_message(data_dir: Path, make_anima):
    """Test process_message IPC request (requires API key)."""
    make_anima(
        "test-anima",
        model="anthropic/claude-sonnet-4",
        identity="You are a helpful test assistant."
    )

    supervisor = ProcessSupervisor(
        animas_dir=data_dir / "animas",
        shared_dir=data_dir / "shared",
        run_dir=data_dir / "run",
        log_dir=data_dir / "logs"
    )

    try:
        await supervisor.start_anima("test-anima")

        # Send message
        result = await supervisor.send_request(
            "test-anima",
            "process_message",
            {"message": "Say hello in exactly 3 words.", "stream": False},
            timeout=60.0
        )

        # Verify response
        assert "response" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    finally:
        await supervisor.shutdown_all()


@pytest.mark.asyncio
@pytest.mark.live
@pytest.mark.timeout(90)
async def test_heartbeat_request(data_dir: Path, make_anima):
    """Test run_heartbeat IPC request (requires API key for LLM execution)."""
    make_anima("test-anima")

    supervisor = ProcessSupervisor(
        animas_dir=data_dir / "animas",
        shared_dir=data_dir / "shared",
        run_dir=data_dir / "run",
        log_dir=data_dir / "logs"
    )

    try:
        await supervisor.start_anima("test-anima")

        # Send heartbeat request
        result = await supervisor.send_request(
            "test-anima",
            "run_heartbeat",
            {},
            timeout=30.0
        )

        # Verify response
        assert "status" in result
        assert result["status"] == "completed"

    finally:
        await supervisor.shutdown_all()


@pytest.mark.asyncio
async def test_concurrent_requests(data_dir: Path, make_anima):
    """Test multiple concurrent IPC requests to same process."""
    make_anima("test-anima")

    supervisor = ProcessSupervisor(
        animas_dir=data_dir / "animas",
        shared_dir=data_dir / "shared",
        run_dir=data_dir / "run",
        log_dir=data_dir / "logs"
    )

    try:
        await supervisor.start_anima("test-anima")

        # Send multiple pings concurrently
        tasks = [
            supervisor.send_request("test-anima", "get_status", {})
            for _ in range(5)
        ]

        results = await asyncio.gather(*tasks)

        # Verify all succeeded
        assert len(results) == 5
        for result in results:
            assert "status" in result

    finally:
        await supervisor.shutdown_all()


@pytest.mark.asyncio
async def test_request_timeout(data_dir: Path, make_anima):
    """Test IPC request timeout handling."""
    make_anima("test-anima")

    supervisor = ProcessSupervisor(
        animas_dir=data_dir / "animas",
        shared_dir=data_dir / "shared",
        run_dir=data_dir / "run",
        log_dir=data_dir / "logs"
    )

    try:
        await supervisor.start_anima("test-anima")

        # Send request with very short timeout
        # (Should succeed since ping is fast)
        result = await supervisor.send_request(
            "test-anima",
            "ping",
            {},
            timeout=1.0
        )

        assert "status" in result

    finally:
        await supervisor.shutdown_all()
