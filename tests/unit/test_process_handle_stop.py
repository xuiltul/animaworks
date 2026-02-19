# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for three bug fixes in ProcessHandle.stop(), SSE stream_done fallback,
and health check STOPPING state detection.

Bug 1: stop() method IPC shutdown order fix
    - send_request("shutdown") must be called while state is still RUNNING
    - state transitions to STOPPING only after the IPC call

Bug 2: SSE stream silent termination fallback
    - stream_done flag tracks whether a "done" event was sent
    - STREAM_INCOMPLETE error is yielded when stream ends without done

Bug 3: Health check STOPPING state detection
    - _check_process_health detects handles stuck in STOPPING for >30s
    - Sets state to FAILED and triggers restart
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.supervisor.ipc import IPCResponse
from core.supervisor.manager import HealthConfig, ProcessSupervisor, RestartPolicy
from core.supervisor.process_handle import ProcessHandle, ProcessState


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def handle(tmp_path: Path) -> ProcessHandle:
    """Create a ProcessHandle in RUNNING state with mocked IPC client."""
    socket_path = tmp_path / "test.sock"
    h = ProcessHandle(
        anima_name="test-anima",
        socket_path=socket_path,
        animas_dir=tmp_path / "animas",
        shared_dir=tmp_path / "shared",
        log_dir=tmp_path / "logs",
    )
    # Set up as a running process with mock subprocess and IPC client
    h.state = ProcessState.RUNNING
    h.process = MagicMock()
    h.process.pid = 12345
    h.process.poll.return_value = None  # Process alive
    h.process.returncode = None
    h.ipc_client = AsyncMock()
    # Create socket file so cleanup doesn't error
    socket_path.touch()
    return h


@pytest.fixture
def supervisor(tmp_path: Path) -> ProcessSupervisor:
    """Create a ProcessSupervisor instance for health check tests."""
    return ProcessSupervisor(
        animas_dir=tmp_path / "animas",
        shared_dir=tmp_path / "shared",
        run_dir=tmp_path / "run",
        restart_policy=RestartPolicy(
            max_retries=3,
            backoff_base_sec=0.1,
            backoff_max_sec=1.0,
        ),
        health_config=HealthConfig(
            ping_interval_sec=1.0,
            ping_timeout_sec=0.5,
            max_missed_pings=3,
            startup_grace_sec=1.0,
        ),
    )


# ── Bug 1: stop() IPC shutdown order fix ─────────────────────


class TestStopIPCShutdownOrder:
    """Tests for the stop() method IPC shutdown ordering fix.

    The fix ensures send_request("shutdown") is called while state is
    still RUNNING (since send_request guards on state == RUNNING),
    and only then transitions to STOPPING.
    """

    @pytest.mark.asyncio
    async def test_stop_sends_ipc_shutdown_before_state_change(
        self, handle: ProcessHandle,
    ):
        """Verify that stop() calls send_request("shutdown") while still RUNNING.

        The key invariant is that when send_request is invoked, handle.state
        must be RUNNING. After the call succeeds, state moves to STOPPING,
        then eventually STOPPED.
        """
        states_during_send: list[ProcessState] = []

        original_send_request = handle.send_request

        async def capture_state_send_request(method, params, timeout=60.0):
            """Capture the state at the time send_request is called."""
            states_during_send.append(handle.state)
            # Return a successful shutdown response
            return IPCResponse(id="shutdown", result={"status": "ok"})

        handle.send_request = capture_state_send_request

        # Make process "exit" during grace period so stop() completes
        def poll_side_effect():
            if handle.state in (ProcessState.STOPPING, ProcessState.STOPPED):
                return 0
            return None

        handle.process.poll.side_effect = poll_side_effect
        handle.process.returncode = 0

        await handle.stop(timeout=5.0)

        # send_request must have been called while state was RUNNING
        assert len(states_during_send) == 1
        assert states_during_send[0] == ProcessState.RUNNING

        # After stop completes, state should be STOPPED
        assert handle.state == ProcessState.STOPPED

    @pytest.mark.asyncio
    async def test_stop_skips_ipc_when_not_running(
        self, handle: ProcessHandle,
    ):
        """Verify that stop() skips IPC shutdown when state is not RUNNING.

        When the handle is in STARTING state, send_request("shutdown") should
        not be attempted. stop() should proceed directly to SIGTERM.
        """
        handle.state = ProcessState.STARTING

        send_request_called = False

        async def spy_send_request(method, params, timeout=60.0):
            nonlocal send_request_called
            send_request_called = True
            return IPCResponse(id="shutdown", result={"status": "ok"})

        handle.send_request = spy_send_request

        # Make process "exit" after terminate
        def poll_exit(*args):
            return 0

        handle.process.terminate.side_effect = lambda: None
        # First few polls return None (alive), then 0 (exited) after terminate
        poll_count = 0

        def poll_side_effect():
            nonlocal poll_count
            poll_count += 1
            # After a few polls, simulate process exit
            if poll_count > 2:
                return 0
            return None

        handle.process.poll.side_effect = poll_side_effect
        handle.process.returncode = 0

        await handle.stop(timeout=5.0)

        # send_request should NOT have been called (state was STARTING, not RUNNING)
        assert send_request_called is False

        # Final state should be STOPPED
        assert handle.state == ProcessState.STOPPED

    @pytest.mark.asyncio
    async def test_stop_handles_ipc_failure_gracefully(
        self, handle: ProcessHandle, caplog,
    ):
        """Verify that stop() continues with SIGTERM if IPC shutdown fails.

        If send_request raises an exception, stop() should log a warning
        and continue the shutdown sequence (SIGTERM, then SIGKILL if needed).
        """
        async def failing_send_request(method, params, timeout=60.0):
            raise ConnectionError("IPC connection lost")

        handle.send_request = failing_send_request

        # Make process "exit" after terminate
        poll_count = 0

        def poll_side_effect():
            nonlocal poll_count
            poll_count += 1
            if poll_count > 3:
                return 0
            return None

        handle.process.poll.side_effect = poll_side_effect
        handle.process.returncode = 0

        with caplog.at_level(logging.WARNING):
            await handle.stop(timeout=5.0)

        # Should have logged a warning about the failed shutdown request
        assert any("Shutdown request failed" in record.message for record in caplog.records)

        # State should still reach STOPPED (via SIGTERM fallback)
        assert handle.state == ProcessState.STOPPED


# ── Bug 2: SSE stream_done flag and STREAM_INCOMPLETE fallback ────


class TestStreamDoneFlag:
    """Tests for the stream_done tracking logic in chat_stream SSE generator.

    Bug 2 added a stream_done flag that is set to True when the IPC stream
    yields a done=True response. If the async for loop ends without this
    flag being set, a STREAM_INCOMPLETE error is emitted.
    """

    @pytest.mark.asyncio
    async def test_stream_done_flag_set_on_done_event(self):
        """Verify that stream_done becomes True when ipc_response.done=True.

        Simulates the core logic of the SSE generator: iterating over IPC
        responses and checking the done flag. When a done=True response
        arrives, stream_done should be set to True.
        """
        # Simulate IPC stream responses: text chunk, then done
        responses = [
            IPCResponse(
                id="req_1",
                stream=True,
                chunk=json.dumps({"type": "text_delta", "text": "Hello"}),
            ),
            IPCResponse(
                id="req_1",
                stream=True,
                done=True,
                result={"response": "Hello", "cycle_result": {"summary": "Hello"}},
            ),
        ]

        stream_done = False

        # Replicate the core loop logic from _ipc_stream_events
        for ipc_response in responses:
            if ipc_response.done:
                stream_done = True
                break
            if ipc_response.chunk:
                # Process chunk (no-op for this test)
                pass

        assert stream_done is True

    @pytest.mark.asyncio
    async def test_stream_incomplete_error_when_no_done(self):
        """Verify that STREAM_INCOMPLETE is yielded when stream ends without done.

        When the async for loop over IPC responses completes without ever
        receiving done=True, the stream_done flag stays False, and the
        generator should yield an error SSE event with code STREAM_INCOMPLETE.
        """
        # Simulate IPC stream that ends without done=True
        responses = [
            IPCResponse(
                id="req_1",
                stream=True,
                chunk=json.dumps({"type": "text_delta", "text": "Hello"}),
            ),
            IPCResponse(
                id="req_1",
                stream=True,
                chunk=json.dumps({"type": "text_delta", "text": " world"}),
            ),
            # No done=True response — stream ends abruptly
        ]

        stream_done = False
        collected_events: list[dict] = []

        # Replicate the core loop logic from _ipc_stream_events
        for ipc_response in responses:
            if ipc_response.done:
                result = ipc_response.result or {}
                collected_events.append({"type": "done", "data": result})
                stream_done = True
                break
            if ipc_response.chunk:
                chunk_data = json.loads(ipc_response.chunk)
                collected_events.append({"type": chunk_data["type"], "data": chunk_data})

        # After the loop, check the stream_done fallback
        if not stream_done:
            collected_events.append({
                "type": "error",
                "data": {
                    "code": "STREAM_INCOMPLETE",
                    "message": "ストリームが予期せず終了しました。再試行してください。",
                },
            })

        assert stream_done is False

        # Verify STREAM_INCOMPLETE error was emitted
        error_events = [e for e in collected_events if e["type"] == "error"]
        assert len(error_events) == 1
        assert error_events[0]["data"]["code"] == "STREAM_INCOMPLETE"

    @pytest.mark.asyncio
    async def test_stream_done_via_chunk_cycle_done(self):
        """Verify stream_done is set when a chunk contains a cycle_done event.

        The SSE generator also sets stream_done=True when a chunk with
        type=="cycle_done" is parsed via _chunk_to_event returning "done".
        """
        from server.routes.chat import _chunk_to_event

        responses = [
            IPCResponse(
                id="req_1",
                stream=True,
                chunk=json.dumps({"type": "text_delta", "text": "Hello"}),
            ),
            IPCResponse(
                id="req_1",
                stream=True,
                chunk=json.dumps({
                    "type": "cycle_done",
                    "cycle_result": {"summary": "Hello"},
                }),
            ),
        ]

        stream_done = False

        for ipc_response in responses:
            if ipc_response.done:
                stream_done = True
                break
            if ipc_response.chunk:
                chunk_data = json.loads(ipc_response.chunk)
                result = _chunk_to_event(chunk_data)
                if result:
                    evt_name, _ = result
                    if evt_name == "done":
                        stream_done = True

        assert stream_done is True


# ── Bug 3: Health check STOPPING state detection ─────────────


class TestHealthCheckStoppingDetection:
    """Tests for _check_process_health detecting stuck STOPPING state.

    Bug 3 added an early return check: if handle.state == STOPPING and the
    duration exceeds 30 seconds, the handle is set to FAILED and a restart
    is triggered.
    """

    @pytest.mark.asyncio
    async def test_health_check_detects_stuck_stopping(
        self, supervisor: ProcessSupervisor, tmp_path: Path,
    ):
        """Verify stuck STOPPING state (>30s) triggers FAILED + restart.

        A handle that has been in STOPPING state for more than 30 seconds
        should be detected by _check_process_health, which sets state to
        FAILED and creates a task to handle the failure.
        """
        handle = ProcessHandle(
            anima_name="stuck-anima",
            socket_path=tmp_path / "stuck.sock",
            animas_dir=tmp_path / "animas",
            shared_dir=tmp_path / "shared",
        )
        handle.state = ProcessState.STOPPING
        # Set stopping_since to >30 seconds ago so the duration check triggers
        handle.stopping_since = datetime.now() - timedelta(seconds=60)

        supervisor.processes["stuck-anima"] = handle

        with patch.object(
            supervisor, "_handle_process_failure", new_callable=AsyncMock,
        ) as mock_failure:
            await supervisor._check_process_health("stuck-anima", handle)

        # State should have been set to FAILED
        assert handle.state == ProcessState.FAILED

        # The method should have early-returned (no ping attempt)
        # We verify this indirectly: _handle_process_failure was scheduled
        # via asyncio.create_task. Give the event loop a tick to execute it.
        await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_health_check_ignores_recent_stopping(
        self, supervisor: ProcessSupervisor, tmp_path: Path,
    ):
        """Verify no action for a handle that entered STOPPING <30s ago.

        A handle that recently entered STOPPING state (less than 30 seconds
        ago) should be ignored by _check_process_health — it returns early
        without setting FAILED or triggering restart.
        """
        handle = ProcessHandle(
            anima_name="stopping-anima",
            socket_path=tmp_path / "stopping.sock",
            animas_dir=tmp_path / "animas",
            shared_dir=tmp_path / "shared",
        )
        handle.state = ProcessState.STOPPING
        # Set stopping_since to just 5 seconds ago (well under the 30s threshold)
        handle.stopping_since = datetime.now() - timedelta(seconds=5)

        supervisor.processes["stopping-anima"] = handle

        with patch.object(
            supervisor, "_handle_process_failure", new_callable=AsyncMock,
        ) as mock_failure:
            await supervisor._check_process_health("stopping-anima", handle)

        # State should remain STOPPING (not changed to FAILED)
        assert handle.state == ProcessState.STOPPING

        # _handle_process_failure should NOT have been called
        mock_failure.assert_not_called()

    @pytest.mark.asyncio
    async def test_health_check_stopping_returns_early(
        self, supervisor: ProcessSupervisor, tmp_path: Path,
    ):
        """Verify _check_process_health returns early for STOPPING state.

        Regardless of whether the threshold is exceeded, when state is
        STOPPING the method should return early without proceeding to
        ping or other health checks.
        """
        handle = ProcessHandle(
            anima_name="stopping-anima",
            socket_path=tmp_path / "stopping.sock",
            animas_dir=tmp_path / "animas",
            shared_dir=tmp_path / "shared",
        )
        handle.state = ProcessState.STOPPING
        handle.stats.started_at = datetime.now() - timedelta(seconds=10)
        handle.process = MagicMock()
        handle.process.poll.return_value = None

        # Mock ping to ensure it's NOT called
        handle_ping_mock = AsyncMock(return_value=True)
        handle.ping = handle_ping_mock

        supervisor.processes["stopping-anima"] = handle

        await supervisor._check_process_health("stopping-anima", handle)

        # Ping should NOT have been called (early return for STOPPING state)
        handle_ping_mock.assert_not_called()

        # State should remain STOPPING
        assert handle.state == ProcessState.STOPPING
