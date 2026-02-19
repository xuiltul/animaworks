# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E test for the bootstrap background execution flow.

Tests the full bootstrap lifecycle using mock IPC (no actual subprocesses).
Verifies:
- ProcessSupervisor detects ``needs_bootstrap`` and launches background task
- ``is_bootstrapping()`` state transitions are correct
- WebSocket events (``anima.bootstrap``) are broadcast for started/completed
- Chat endpoints return busy response during bootstrap
- Chat endpoints work normally after bootstrap completes
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from core.supervisor.ipc import IPCResponse
from core.supervisor.manager import (
    HealthConfig,
    ProcessSupervisor,
    RestartPolicy,
)
from core.supervisor.process_handle import ProcessHandle, ProcessState, ProcessStats
from server.stream_registry import StreamRegistry


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def mock_ws_manager() -> MagicMock:
    """Create a mock WebSocketManager with an async broadcast method."""
    ws = MagicMock()
    ws.broadcast = AsyncMock()
    return ws


@pytest.fixture
def supervisor_dirs(tmp_path):
    """Create temporary directories for ProcessSupervisor."""
    animas_dir = tmp_path / "animas"
    shared_dir = tmp_path / "shared"
    run_dir = tmp_path / "run"
    log_dir = tmp_path / "logs"
    for d in (animas_dir, shared_dir, run_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)
    return animas_dir, shared_dir, run_dir, log_dir


@pytest.fixture
def supervisor(supervisor_dirs, mock_ws_manager) -> ProcessSupervisor:
    """Create a ProcessSupervisor with mock WebSocketManager."""
    animas_dir, shared_dir, run_dir, log_dir = supervisor_dirs
    return ProcessSupervisor(
        animas_dir=animas_dir,
        shared_dir=shared_dir,
        run_dir=run_dir,
        log_dir=log_dir,
        restart_policy=RestartPolicy(max_retries=0),
        health_config=HealthConfig(ping_interval_sec=999),
        ws_manager=mock_ws_manager,
    )


def _make_mock_handle(
    anima_name: str,
    needs_bootstrap: bool = True,
    bootstrap_delay: float = 0.0,
) -> MagicMock:
    """Create a mock ProcessHandle that simulates IPC responses.

    Args:
        anima_name: Name of the anima.
        needs_bootstrap: Whether ``get_status`` reports bootstrap needed.
        bootstrap_delay: Seconds to wait before returning from ``run_bootstrap``.
    """
    handle = MagicMock(spec=ProcessHandle)
    handle.anima_name = anima_name
    handle.state = ProcessState.RUNNING
    handle.stats = ProcessStats(started_at=datetime.now())
    handle._streaming = False

    handle.start = AsyncMock()
    handle.stop = AsyncMock()
    handle.kill = AsyncMock()
    handle.get_pid.return_value = 12345
    handle.is_alive.return_value = True

    async def _send_request(method: str, params: dict, timeout: float = 60.0) -> IPCResponse:
        if method == "get_status":
            return IPCResponse(
                id="status_check",
                result={"needs_bootstrap": needs_bootstrap, "status": "running"},
            )
        if method == "run_bootstrap":
            if bootstrap_delay > 0:
                await asyncio.sleep(bootstrap_delay)
            return IPCResponse(
                id="bootstrap",
                result={"duration_ms": 1234, "status": "completed"},
            )
        if method == "process_message":
            return IPCResponse(
                id="chat",
                result={"response": "Hello from test!", "replied_to": []},
            )
        if method == "ping":
            return IPCResponse(id="ping", result={"status": "ok"})
        if method == "shutdown":
            return IPCResponse(id="shutdown", result={"status": "ok"})
        return IPCResponse(id="unknown", result={})

    handle.send_request = AsyncMock(side_effect=_send_request)

    return handle


# ── Bootstrap Lifecycle Tests ────────────────────────────────


class TestBootstrapLifecycle:
    """Test the full bootstrap lifecycle through ProcessSupervisor."""

    async def test_bootstrap_detected_and_launched(
        self, supervisor: ProcessSupervisor, mock_ws_manager: MagicMock,
    ):
        """When start_anima finds needs_bootstrap=True, a background
        bootstrap task is launched and runs to completion."""
        bootstrap_started = asyncio.Event()
        bootstrap_proceed = asyncio.Event()
        mock_handle = _make_mock_handle("alice", needs_bootstrap=True)

        original_side_effect = mock_handle.send_request.side_effect

        async def _controlled_send_request(method, params, timeout=60.0):
            if method == "run_bootstrap":
                bootstrap_started.set()
                await bootstrap_proceed.wait()
                return IPCResponse(
                    id="bootstrap",
                    result={"duration_ms": 1234, "status": "completed"},
                )
            return await original_side_effect(method, params, timeout)

        mock_handle.send_request = AsyncMock(side_effect=_controlled_send_request)

        with patch(
            "core.supervisor.manager.ProcessHandle",
            return_value=mock_handle,
        ):
            await supervisor.start_anima("alice")

        # The handle should be registered
        assert "alice" in supervisor.processes

        # Wait for bootstrap to start
        await asyncio.wait_for(bootstrap_started.wait(), timeout=2.0)

        # is_bootstrapping should be True during execution
        assert supervisor.is_bootstrapping("alice") is True

        # Allow bootstrap to complete
        bootstrap_proceed.set()
        await asyncio.sleep(0.1)

        # is_bootstrapping should be False after completion
        assert supervisor.is_bootstrapping("alice") is False

        # Verify run_bootstrap was called via IPC
        bootstrap_calls = [
            call for call in mock_handle.send_request.call_args_list
            if call.args[0] == "run_bootstrap"
        ]
        assert len(bootstrap_calls) == 1

    async def test_bootstrap_websocket_events(
        self, supervisor: ProcessSupervisor, mock_ws_manager: MagicMock,
    ):
        """WebSocket broadcasts started and completed events during bootstrap."""
        mock_handle = _make_mock_handle("bob", needs_bootstrap=True, bootstrap_delay=0.05)

        with patch(
            "core.supervisor.manager.ProcessHandle",
            return_value=mock_handle,
        ):
            await supervisor.start_anima("bob")

        # Wait for bootstrap to complete
        await asyncio.sleep(0.3)

        # Collect all broadcast calls
        broadcast_calls = [
            call.args[0] for call in mock_ws_manager.broadcast.call_args_list
        ]

        # Filter for anima.bootstrap events
        bootstrap_events = [
            c for c in broadcast_calls if c.get("type") == "anima.bootstrap"
        ]

        assert len(bootstrap_events) == 2, (
            f"Expected 2 bootstrap events (started + completed), got {len(bootstrap_events)}: "
            f"{bootstrap_events}"
        )

        statuses = [e["data"]["status"] for e in bootstrap_events]
        assert statuses[0] == "started"
        assert statuses[1] == "completed"

        # Both events should reference the anima name
        for event in bootstrap_events:
            assert event["data"]["name"] == "bob"

    async def test_is_bootstrapping_transitions(
        self, supervisor: ProcessSupervisor,
    ):
        """is_bootstrapping() transitions from False -> True -> False."""
        bootstrap_event = asyncio.Event()
        bootstrap_proceed = asyncio.Event()

        mock_handle = _make_mock_handle("carol", needs_bootstrap=True)

        # Override send_request to wait for a signal before returning
        original_side_effect = mock_handle.send_request.side_effect

        async def _controlled_send_request(method, params, timeout=60.0):
            if method == "run_bootstrap":
                bootstrap_event.set()  # Signal that bootstrap has started
                await bootstrap_proceed.wait()  # Wait for test to allow completion
                return IPCResponse(
                    id="bootstrap",
                    result={"duration_ms": 100, "status": "completed"},
                )
            return await original_side_effect(method, params, timeout)

        mock_handle.send_request = AsyncMock(side_effect=_controlled_send_request)

        # Before start: should not be bootstrapping
        assert supervisor.is_bootstrapping("carol") is False

        with patch(
            "core.supervisor.manager.ProcessHandle",
            return_value=mock_handle,
        ):
            await supervisor.start_anima("carol")

        # Wait until bootstrap has actually started
        await asyncio.wait_for(bootstrap_event.wait(), timeout=2.0)

        # During bootstrap: should be True
        assert supervisor.is_bootstrapping("carol") is True

        # Allow bootstrap to complete
        bootstrap_proceed.set()
        await asyncio.sleep(0.1)

        # After bootstrap: should be False
        assert supervisor.is_bootstrapping("carol") is False

    async def test_no_bootstrap_when_not_needed(
        self, supervisor: ProcessSupervisor, mock_ws_manager: MagicMock,
    ):
        """When needs_bootstrap=False, no bootstrap task is launched."""
        mock_handle = _make_mock_handle("dave", needs_bootstrap=False)

        with patch(
            "core.supervisor.manager.ProcessHandle",
            return_value=mock_handle,
        ):
            await supervisor.start_anima("dave")

        await asyncio.sleep(0.1)

        # Should never be bootstrapping
        assert supervisor.is_bootstrapping("dave") is False

        # run_bootstrap should never have been called
        bootstrap_calls = [
            call for call in mock_handle.send_request.call_args_list
            if call.args[0] == "run_bootstrap"
        ]
        assert len(bootstrap_calls) == 0

        # No anima.bootstrap events should have been broadcast
        bootstrap_events = [
            call.args[0]
            for call in mock_ws_manager.broadcast.call_args_list
            if call.args[0].get("type") == "anima.bootstrap"
        ]
        assert len(bootstrap_events) == 0

    async def test_bootstrap_failure_broadcasts_failed(
        self, supervisor: ProcessSupervisor, mock_ws_manager: MagicMock,
    ):
        """When bootstrap IPC returns an error, a 'failed' event is broadcast."""
        mock_handle = _make_mock_handle("eve", needs_bootstrap=True)

        # Override send_request to return an error for run_bootstrap
        original_side_effect = mock_handle.send_request.side_effect

        async def _failing_send_request(method, params, timeout=60.0):
            if method == "run_bootstrap":
                return IPCResponse(
                    id="bootstrap",
                    error={"code": "BOOTSTRAP_ERROR", "message": "LLM call failed"},
                )
            return await original_side_effect(method, params, timeout)

        mock_handle.send_request = AsyncMock(side_effect=_failing_send_request)

        with patch(
            "core.supervisor.manager.ProcessHandle",
            return_value=mock_handle,
        ):
            await supervisor.start_anima("eve")

        await asyncio.sleep(0.2)

        # Should no longer be bootstrapping after failure
        assert supervisor.is_bootstrapping("eve") is False

        # Check that failed event was broadcast
        broadcast_calls = [
            call.args[0] for call in mock_ws_manager.broadcast.call_args_list
        ]
        bootstrap_events = [
            c for c in broadcast_calls if c.get("type") == "anima.bootstrap"
        ]

        statuses = [e["data"]["status"] for e in bootstrap_events]
        assert "started" in statuses
        assert "failed" in statuses

    async def test_get_process_status_during_bootstrap(
        self, supervisor: ProcessSupervisor,
    ):
        """get_process_status returns 'bootstrapping' status during bootstrap."""
        bootstrap_proceed = asyncio.Event()
        mock_handle = _make_mock_handle("frank", needs_bootstrap=True)

        original_side_effect = mock_handle.send_request.side_effect

        async def _pausing_send_request(method, params, timeout=60.0):
            if method == "run_bootstrap":
                await bootstrap_proceed.wait()
                return IPCResponse(
                    id="bootstrap",
                    result={"duration_ms": 100, "status": "completed"},
                )
            return await original_side_effect(method, params, timeout)

        mock_handle.send_request = AsyncMock(side_effect=_pausing_send_request)

        with patch(
            "core.supervisor.manager.ProcessHandle",
            return_value=mock_handle,
        ):
            await supervisor.start_anima("frank")

        await asyncio.sleep(0.05)

        # During bootstrap
        status = supervisor.get_process_status("frank")
        assert status["status"] == "bootstrapping"
        assert status["bootstrapping"] is True

        # Release bootstrap
        bootstrap_proceed.set()
        await asyncio.sleep(0.1)

        # After bootstrap
        status = supervisor.get_process_status("frank")
        assert status["status"] == "running"
        assert status["bootstrapping"] is False


# ── Chat Guard Tests ─────────────────────────────────────────


def _make_test_app_with_supervisor(supervisor: ProcessSupervisor) -> "FastAPI":
    """Build a test FastAPI app wired to the given supervisor."""
    from fastapi import FastAPI

    from server.routes.chat import create_chat_router

    app = FastAPI()
    app.state.ws_manager = supervisor.ws_manager
    app.state.supervisor = supervisor
    app.state.stream_registry = StreamRegistry()

    router = create_chat_router()
    app.include_router(router, prefix="/api")
    return app


class TestChatGuardDuringBootstrap:
    """Test that chat endpoints reject requests during bootstrap."""

    async def test_chat_stream_returns_busy_during_bootstrap(
        self, supervisor: ProcessSupervisor,
    ):
        """POST /animas/{name}/chat/stream returns bootstrap busy SSE
        while anima is bootstrapping."""
        bootstrap_proceed = asyncio.Event()
        mock_handle = _make_mock_handle("alice", needs_bootstrap=True)

        original_side_effect = mock_handle.send_request.side_effect

        async def _pausing_send_request(method, params, timeout=60.0):
            if method == "run_bootstrap":
                await bootstrap_proceed.wait()
                return IPCResponse(
                    id="bootstrap",
                    result={"duration_ms": 100, "status": "completed"},
                )
            return await original_side_effect(method, params, timeout)

        mock_handle.send_request = AsyncMock(side_effect=_pausing_send_request)

        with patch(
            "core.supervisor.manager.ProcessHandle",
            return_value=mock_handle,
        ):
            await supervisor.start_anima("alice")

        await asyncio.sleep(0.05)
        assert supervisor.is_bootstrapping("alice") is True

        # Build app and make request during bootstrap
        app = _make_test_app_with_supervisor(supervisor)
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test",
        ) as client:
            resp = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "Hello"},
            )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        body = resp.text
        assert "event: bootstrap" in body
        assert '"busy"' in body

        # Cleanup
        bootstrap_proceed.set()
        await asyncio.sleep(0.1)

    async def test_chat_returns_503_during_bootstrap(
        self, supervisor: ProcessSupervisor,
    ):
        """POST /animas/{name}/chat returns 503 while anima is bootstrapping."""
        bootstrap_proceed = asyncio.Event()
        mock_handle = _make_mock_handle("alice", needs_bootstrap=True)

        original_side_effect = mock_handle.send_request.side_effect

        async def _pausing_send_request(method, params, timeout=60.0):
            if method == "run_bootstrap":
                await bootstrap_proceed.wait()
                return IPCResponse(
                    id="bootstrap",
                    result={"duration_ms": 100, "status": "completed"},
                )
            return await original_side_effect(method, params, timeout)

        mock_handle.send_request = AsyncMock(side_effect=_pausing_send_request)

        with patch(
            "core.supervisor.manager.ProcessHandle",
            return_value=mock_handle,
        ):
            await supervisor.start_anima("alice")

        await asyncio.sleep(0.05)
        assert supervisor.is_bootstrapping("alice") is True

        # Build app and make request during bootstrap
        app = _make_test_app_with_supervisor(supervisor)
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test",
        ) as client:
            resp = await client.post(
                "/api/animas/alice/chat",
                json={"message": "Hello"},
            )

        assert resp.status_code == 503
        body = resp.json()
        assert "error" in body

        # Cleanup
        bootstrap_proceed.set()
        await asyncio.sleep(0.1)

    async def test_chat_stream_works_after_bootstrap(
        self, supervisor: ProcessSupervisor, mock_ws_manager: MagicMock,
    ):
        """After bootstrap completes, chat/stream works normally."""
        mock_handle = _make_mock_handle("alice", needs_bootstrap=True, bootstrap_delay=0.05)

        # Set up streaming for the chat request
        async def _send_request_stream(method, params, timeout=120.0):
            yield IPCResponse(
                id="chat_stream",
                stream=True,
                chunk=json.dumps(
                    {"type": "text_delta", "text": "Hello!"},
                    ensure_ascii=False,
                ),
            )
            yield IPCResponse(
                id="chat_stream",
                stream=True,
                done=True,
                result={
                    "response": "Hello!",
                    "replied_to": [],
                    "cycle_result": {"summary": "Hello!"},
                },
            )

        mock_handle.send_request_stream = _send_request_stream

        with patch(
            "core.supervisor.manager.ProcessHandle",
            return_value=mock_handle,
        ):
            await supervisor.start_anima("alice")

        # Wait for bootstrap to complete
        await asyncio.sleep(0.3)
        assert supervisor.is_bootstrapping("alice") is False

        # Now chat should work
        app = _make_test_app_with_supervisor(supervisor)

        # We need to wire supervisor.send_request_stream to delegate properly
        # The supervisor's send_request_stream calls handle.send_request_stream
        # and our mock handle has it set up above.

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test",
        ) as client:
            resp = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "Hello"},
            )

        assert resp.status_code == 200
        body = resp.text

        # Should NOT contain bootstrap busy
        assert '"busy"' not in body

        # Should contain normal chat events
        assert "event: text_delta" in body or "event: done" in body

    async def test_chat_returns_normal_after_bootstrap(
        self, supervisor: ProcessSupervisor,
    ):
        """After bootstrap completes, POST /animas/{name}/chat returns
        a normal 200 response."""
        mock_handle = _make_mock_handle("alice", needs_bootstrap=True, bootstrap_delay=0.05)

        with patch(
            "core.supervisor.manager.ProcessHandle",
            return_value=mock_handle,
        ):
            await supervisor.start_anima("alice")

        # Wait for bootstrap to complete
        await asyncio.sleep(0.3)
        assert supervisor.is_bootstrapping("alice") is False

        app = _make_test_app_with_supervisor(supervisor)
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test",
        ) as client:
            resp = await client.post(
                "/api/animas/alice/chat",
                json={"message": "Hello"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["anima"] == "alice"
        assert body["response"] == "Hello from test!"


# ── Multiple Animas Tests ───────────────────────────────────


class TestMultipleAnimasBootstrap:
    """Test bootstrap with multiple animas running concurrently."""

    async def test_independent_bootstrap_tracking(
        self, supervisor: ProcessSupervisor,
    ):
        """Each anima's bootstrap state is tracked independently."""
        alice_proceed = asyncio.Event()
        bob_proceed = asyncio.Event()

        alice_handle = _make_mock_handle("alice", needs_bootstrap=True)
        bob_handle = _make_mock_handle("bob", needs_bootstrap=True)

        # Alice pauses in bootstrap, Bob completes immediately
        alice_original = alice_handle.send_request.side_effect

        async def _alice_send(method, params, timeout=60.0):
            if method == "run_bootstrap":
                await alice_proceed.wait()
                return IPCResponse(
                    id="bootstrap",
                    result={"duration_ms": 100, "status": "completed"},
                )
            return await alice_original(method, params, timeout)

        alice_handle.send_request = AsyncMock(side_effect=_alice_send)

        bob_original = bob_handle.send_request.side_effect

        async def _bob_send(method, params, timeout=60.0):
            if method == "run_bootstrap":
                await bob_proceed.wait()
                return IPCResponse(
                    id="bootstrap",
                    result={"duration_ms": 50, "status": "completed"},
                )
            return await bob_original(method, params, timeout)

        bob_handle.send_request = AsyncMock(side_effect=_bob_send)

        # Patch ProcessHandle in manager to return different mocks per call
        with patch(
            "core.supervisor.manager.ProcessHandle",
            side_effect=[alice_handle, bob_handle],
        ):
            await supervisor.start_anima("alice")
            await supervisor.start_anima("bob")

        await asyncio.sleep(0.05)

        # Both should be bootstrapping
        assert supervisor.is_bootstrapping("alice") is True
        assert supervisor.is_bootstrapping("bob") is True

        # Complete Bob's bootstrap only
        bob_proceed.set()
        await asyncio.sleep(0.1)

        # Alice still bootstrapping, Bob done
        assert supervisor.is_bootstrapping("alice") is True
        assert supervisor.is_bootstrapping("bob") is False

        # Complete Alice's bootstrap
        alice_proceed.set()
        await asyncio.sleep(0.1)

        # Both done
        assert supervisor.is_bootstrapping("alice") is False
        assert supervisor.is_bootstrapping("bob") is False
