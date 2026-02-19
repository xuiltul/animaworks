# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for bootstrap background execution feature.

Tests cover:
1. DigitalAnima.run_bootstrap() — skip, status transitions, error handling
2. AnimaRunner._handle_run_bootstrap() — IPC handler dispatch and get_status
3. ProcessSupervisor bootstrap management — is_bootstrapping, broadcasts, set cleanup
4. Chat endpoint bootstrap guard — stream SSE, non-streaming 503, greet 503
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from core.schemas import CycleResult
from core.supervisor.ipc import IPCRequest, IPCResponse
from core.supervisor.manager import (
    HealthConfig,
    ProcessSupervisor,
    RestartPolicy,
)
from core.supervisor.process_handle import ProcessHandle, ProcessState
from server.stream_registry import StreamRegistry


# ── Helpers ─────────────────────────────────────────────────────────


def _make_anima_dir(tmp_path: Path, *, with_bootstrap: bool = True) -> Path:
    """Create a minimal anima directory for testing."""
    anima_dir = tmp_path / "animas" / "test-anima"
    anima_dir.mkdir(parents=True)
    (anima_dir / "identity.md").write_text("# Test Anima", encoding="utf-8")
    if with_bootstrap:
        (anima_dir / "bootstrap.md").write_text(
            "# Bootstrap instructions", encoding="utf-8"
        )
    for sub in [
        "episodes", "knowledge", "procedures", "skills",
        "state", "shortterm", "shortterm/archive", "transcripts",
    ]:
        (anima_dir / sub).mkdir(parents=True, exist_ok=True)
    (anima_dir / "state" / "current_task.md").write_text(
        "status: idle\n", encoding="utf-8"
    )
    (anima_dir / "state" / "pending.md").write_text("", encoding="utf-8")
    return anima_dir


def _make_shared_dir(tmp_path: Path, anima_name: str = "test-anima") -> Path:
    """Create a minimal shared directory for testing."""
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True)
    (shared_dir / "inbox" / anima_name).mkdir(parents=True)
    (shared_dir / "users").mkdir(parents=True)
    return shared_dir


def _make_digital_anima(tmp_path: Path, *, with_bootstrap: bool = True):
    """Create a DigitalAnima with mocked heavy dependencies."""
    from core.anima import DigitalAnima

    anima_dir = _make_anima_dir(tmp_path, with_bootstrap=with_bootstrap)
    shared_dir = _make_shared_dir(tmp_path)

    with (
        patch("core.anima.MemoryManager"),
        patch("core.anima.AgentCore"),
        patch("core.anima.Messenger"),
    ):
        dp = DigitalAnima(anima_dir, shared_dir)

    return dp


def _make_supervisor(tmp_path: Path, *, ws_manager=None) -> ProcessSupervisor:
    """Create a ProcessSupervisor for testing."""
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
            ping_interval_sec=0.5,
            ping_timeout_sec=0.2,
            max_missed_pings=2,
            startup_grace_sec=0.5,
        ),
        ws_manager=ws_manager,
    )


def _make_mock_handle(anima_name: str = "test-anima") -> MagicMock:
    """Create a mock ProcessHandle."""
    handle = MagicMock(spec=ProcessHandle)
    handle.anima_name = anima_name
    handle.state = ProcessState.RUNNING
    handle.get_pid.return_value = 12345
    handle.stats = MagicMock()
    handle.stats.started_at = datetime.now()
    handle.stats.restart_count = 0
    handle.stats.missed_pings = 0
    handle.stats.last_ping_at = None
    return handle


# ════════════════════════════════════════════════════════════════════
# 1. DigitalAnima.run_bootstrap()
# ════════════════════════════════════════════════════════════════════


class TestDigitalAnimaRunBootstrap:
    """Tests for DigitalAnima.run_bootstrap()."""

    @pytest.mark.asyncio
    async def test_run_bootstrap_skips_when_no_bootstrap_md(self, tmp_path: Path):
        """When needs_bootstrap is False, should return CycleResult with action='skipped'."""
        dp = _make_digital_anima(tmp_path, with_bootstrap=False)

        assert dp.needs_bootstrap is False

        result = await dp.run_bootstrap()

        assert isinstance(result, CycleResult)
        assert result.action == "skipped"
        assert result.trigger == "bootstrap"
        assert "not needed" in result.summary.lower() or "not needed" in result.summary

    @pytest.mark.asyncio
    async def test_run_bootstrap_sets_status_to_bootstrapping(self, tmp_path: Path):
        """During run_bootstrap, self._status should be set to 'bootstrapping'."""
        dp = _make_digital_anima(tmp_path, with_bootstrap=True)

        observed_statuses: list[str] = []

        async def mock_run_cycle(prompt, trigger=""):
            # Capture status while inside the cycle
            observed_statuses.append(dp._status)
            return CycleResult(
                trigger=trigger,
                action="completed",
                summary="Bootstrap done",
                duration_ms=100,
            )

        dp.agent.run_cycle = mock_run_cycle

        with patch("core.anima.ConversationMemory") as mock_conv_cls:
            mock_conv = MagicMock()
            mock_conv.build_chat_prompt = MagicMock(return_value="prompt")
            mock_conv_cls.return_value = mock_conv

            await dp.run_bootstrap()

        assert "bootstrapping" in observed_statuses

    @pytest.mark.asyncio
    async def test_run_bootstrap_resets_status_on_completion(self, tmp_path: Path):
        """After run_bootstrap completes successfully, status should be 'idle'."""
        dp = _make_digital_anima(tmp_path, with_bootstrap=True)

        dp.agent.run_cycle = AsyncMock(
            return_value=CycleResult(
                trigger="bootstrap",
                action="completed",
                summary="Done",
                duration_ms=50,
            )
        )

        with patch("core.anima.ConversationMemory") as mock_conv_cls:
            mock_conv = MagicMock()
            mock_conv.build_chat_prompt = MagicMock(return_value="prompt")
            mock_conv_cls.return_value = mock_conv

            result = await dp.run_bootstrap()

        assert dp._status == "idle"
        assert result.action == "completed"

    @pytest.mark.asyncio
    async def test_run_bootstrap_resets_status_on_failure(self, tmp_path: Path):
        """Even if agent.run_cycle raises, status should be reset to 'idle'."""
        dp = _make_digital_anima(tmp_path, with_bootstrap=True)

        dp.agent.run_cycle = AsyncMock(
            side_effect=RuntimeError("LLM API error")
        )

        with patch("core.anima.ConversationMemory") as mock_conv_cls:
            mock_conv = MagicMock()
            mock_conv.build_chat_prompt = MagicMock(return_value="prompt")
            mock_conv_cls.return_value = mock_conv

            with pytest.raises(RuntimeError, match="LLM API error"):
                await dp.run_bootstrap()

        assert dp._status == "idle"
        assert dp._current_task == ""


# ════════════════════════════════════════════════════════════════════
# 2. AnimaRunner._handle_run_bootstrap()
# ════════════════════════════════════════════════════════════════════


class TestAnimaRunnerBootstrap:
    """Tests for AnimaRunner bootstrap handler dispatch and get_status."""

    @pytest.mark.asyncio
    async def test_runner_run_bootstrap_handler(self, tmp_path: Path):
        """AnimaRunner should have 'run_bootstrap' in its handler dispatch."""
        from core.supervisor.runner import AnimaRunner

        runner = AnimaRunner(
            anima_name="test-anima",
            socket_path=tmp_path / "test.sock",
            animas_dir=tmp_path / "animas",
            shared_dir=tmp_path / "shared",
        )

        handler = runner._get_handler("run_bootstrap")
        assert handler is not None
        assert handler == runner._handle_run_bootstrap

    @pytest.mark.asyncio
    async def test_runner_run_bootstrap_calls_anima(self, tmp_path: Path):
        """_handle_run_bootstrap should call dp.run_bootstrap() and return result."""
        from core.supervisor.runner import AnimaRunner

        runner = AnimaRunner(
            anima_name="test-anima",
            socket_path=tmp_path / "test.sock",
            animas_dir=tmp_path / "animas",
            shared_dir=tmp_path / "shared",
        )

        mock_anima = MagicMock()
        mock_anima.run_bootstrap = AsyncMock(
            return_value=CycleResult(
                trigger="bootstrap",
                action="completed",
                summary="Bootstrap finished",
                duration_ms=200,
            )
        )
        runner.anima = mock_anima

        result = await runner._handle_run_bootstrap({})

        mock_anima.run_bootstrap.assert_awaited_once()
        assert result["status"] == "completed"
        assert result["summary"] == "Bootstrap finished"
        assert result["duration_ms"] == 200

    @pytest.mark.asyncio
    async def test_runner_get_status_includes_needs_bootstrap(self, tmp_path: Path):
        """get_status response should include needs_bootstrap field."""
        from core.supervisor.runner import AnimaRunner

        runner = AnimaRunner(
            anima_name="test-anima",
            socket_path=tmp_path / "test.sock",
            animas_dir=tmp_path / "animas",
            shared_dir=tmp_path / "shared",
        )

        mock_anima = MagicMock()
        mock_anima._status = "idle"
        mock_anima._current_task = ""
        mock_anima.needs_bootstrap = True
        runner.anima = mock_anima

        result = await runner._handle_get_status({})

        assert "needs_bootstrap" in result
        assert result["needs_bootstrap"] is True

    @pytest.mark.asyncio
    async def test_runner_get_status_needs_bootstrap_false(self, tmp_path: Path):
        """get_status should show needs_bootstrap=False when no bootstrap.md."""
        from core.supervisor.runner import AnimaRunner

        runner = AnimaRunner(
            anima_name="test-anima",
            socket_path=tmp_path / "test.sock",
            animas_dir=tmp_path / "animas",
            shared_dir=tmp_path / "shared",
        )

        mock_anima = MagicMock()
        mock_anima._status = "idle"
        mock_anima._current_task = ""
        mock_anima.needs_bootstrap = False
        runner.anima = mock_anima

        result = await runner._handle_get_status({})

        assert result["needs_bootstrap"] is False


# ════════════════════════════════════════════════════════════════════
# 3. ProcessSupervisor bootstrap management
# ════════════════════════════════════════════════════════════════════


class TestSupervisorBootstrap:
    """Tests for ProcessSupervisor bootstrap tracking and broadcasting."""

    @pytest.mark.asyncio
    async def test_supervisor_is_bootstrapping_default_false(self, tmp_path: Path):
        """is_bootstrapping should return False for unknown anima."""
        supervisor = _make_supervisor(tmp_path)

        assert supervisor.is_bootstrapping("nonexistent") is False

    @pytest.mark.asyncio
    async def test_supervisor_is_bootstrapping_during_bootstrap(self, tmp_path: Path):
        """When anima is in _bootstrapping set, is_bootstrapping returns True."""
        supervisor = _make_supervisor(tmp_path)

        supervisor._bootstrapping.add("alice")

        assert supervisor.is_bootstrapping("alice") is True
        assert supervisor.is_bootstrapping("bob") is False

    @pytest.mark.asyncio
    async def test_supervisor_run_bootstrap_broadcasts_started(self, tmp_path: Path):
        """_run_bootstrap should broadcast a 'started' event via ws_manager."""
        ws_manager = MagicMock()
        ws_manager.broadcast = AsyncMock()
        supervisor = _make_supervisor(tmp_path, ws_manager=ws_manager)

        # Set up a mock handle that returns a successful response
        handle = _make_mock_handle("alice")
        handle.send_request = AsyncMock(
            return_value=IPCResponse(
                id="test",
                result={"status": "completed", "duration_ms": 100},
            )
        )
        supervisor.processes["alice"] = handle

        await supervisor._run_bootstrap("alice")

        # Check that broadcast was called with "started"
        calls = ws_manager.broadcast.call_args_list
        started_calls = [
            c for c in calls
            if c[0][0].get("data", {}).get("status") == "started"
        ]
        assert len(started_calls) >= 1
        started_data = started_calls[0][0][0]
        assert started_data["type"] == "anima.bootstrap"
        assert started_data["data"]["name"] == "alice"

    @pytest.mark.asyncio
    async def test_supervisor_run_bootstrap_broadcasts_completed(self, tmp_path: Path):
        """On success, _run_bootstrap broadcasts a 'completed' event."""
        ws_manager = MagicMock()
        ws_manager.broadcast = AsyncMock()
        supervisor = _make_supervisor(tmp_path, ws_manager=ws_manager)

        handle = _make_mock_handle("alice")
        handle.send_request = AsyncMock(
            return_value=IPCResponse(
                id="test",
                result={"status": "completed", "duration_ms": 100},
            )
        )
        supervisor.processes["alice"] = handle

        await supervisor._run_bootstrap("alice")

        # Check completed broadcast
        calls = ws_manager.broadcast.call_args_list
        completed_calls = [
            c for c in calls
            if c[0][0].get("data", {}).get("status") == "completed"
        ]
        assert len(completed_calls) >= 1
        completed_data = completed_calls[0][0][0]
        assert completed_data["type"] == "anima.bootstrap"
        assert completed_data["data"]["name"] == "alice"

    @pytest.mark.asyncio
    async def test_supervisor_run_bootstrap_broadcasts_failed_on_error(
        self, tmp_path: Path
    ):
        """On error, _run_bootstrap broadcasts a 'failed' event."""
        ws_manager = MagicMock()
        ws_manager.broadcast = AsyncMock()
        supervisor = _make_supervisor(tmp_path, ws_manager=ws_manager)

        handle = _make_mock_handle("alice")
        handle.send_request = AsyncMock(
            return_value=IPCResponse(
                id="test",
                error={"code": "EXECUTION_ERROR", "message": "Boom"},
            )
        )
        supervisor.processes["alice"] = handle

        await supervisor._run_bootstrap("alice")

        # Check failed broadcast
        calls = ws_manager.broadcast.call_args_list
        failed_calls = [
            c for c in calls
            if c[0][0].get("data", {}).get("status") == "failed"
        ]
        assert len(failed_calls) >= 1
        failed_data = failed_calls[0][0][0]
        assert failed_data["type"] == "anima.bootstrap"
        assert failed_data["data"]["name"] == "alice"

    @pytest.mark.asyncio
    async def test_supervisor_run_bootstrap_broadcasts_failed_on_exception(
        self, tmp_path: Path
    ):
        """When send_request raises an exception, broadcasts 'failed'."""
        ws_manager = MagicMock()
        ws_manager.broadcast = AsyncMock()
        supervisor = _make_supervisor(tmp_path, ws_manager=ws_manager)

        handle = _make_mock_handle("alice")
        handle.send_request = AsyncMock(side_effect=Exception("Connection lost"))
        supervisor.processes["alice"] = handle

        await supervisor._run_bootstrap("alice")

        calls = ws_manager.broadcast.call_args_list
        failed_calls = [
            c for c in calls
            if c[0][0].get("data", {}).get("status") == "failed"
        ]
        assert len(failed_calls) >= 1

    @pytest.mark.asyncio
    async def test_supervisor_run_bootstrap_removes_from_set_on_success(
        self, tmp_path: Path
    ):
        """After successful completion, anima is removed from _bootstrapping set."""
        supervisor = _make_supervisor(tmp_path)

        handle = _make_mock_handle("alice")
        handle.send_request = AsyncMock(
            return_value=IPCResponse(
                id="test",
                result={"status": "completed", "duration_ms": 100},
            )
        )
        supervisor.processes["alice"] = handle

        await supervisor._run_bootstrap("alice")

        assert "alice" not in supervisor._bootstrapping

    @pytest.mark.asyncio
    async def test_supervisor_run_bootstrap_removes_from_set_on_failure(
        self, tmp_path: Path
    ):
        """After failure, anima is removed from _bootstrapping set."""
        supervisor = _make_supervisor(tmp_path)

        handle = _make_mock_handle("alice")
        handle.send_request = AsyncMock(
            return_value=IPCResponse(
                id="test",
                error={"code": "EXECUTION_ERROR", "message": "Boom"},
            )
        )
        supervisor.processes["alice"] = handle

        await supervisor._run_bootstrap("alice")

        assert "alice" not in supervisor._bootstrapping

    @pytest.mark.asyncio
    async def test_supervisor_run_bootstrap_removes_from_set_on_exception(
        self, tmp_path: Path
    ):
        """After exception, anima is removed from _bootstrapping set."""
        supervisor = _make_supervisor(tmp_path)

        handle = _make_mock_handle("alice")
        handle.send_request = AsyncMock(side_effect=Exception("Crash"))
        supervisor.processes["alice"] = handle

        await supervisor._run_bootstrap("alice")

        assert "alice" not in supervisor._bootstrapping

    @pytest.mark.asyncio
    async def test_supervisor_process_status_shows_bootstrapping(
        self, tmp_path: Path
    ):
        """get_process_status returns 'bootstrapping' when anima is bootstrapping."""
        supervisor = _make_supervisor(tmp_path)

        handle = _make_mock_handle("alice")
        supervisor.processes["alice"] = handle
        supervisor._bootstrapping.add("alice")

        status = supervisor.get_process_status("alice")

        assert status["status"] == "bootstrapping"

    @pytest.mark.asyncio
    async def test_supervisor_process_status_includes_bootstrapping_flag(
        self, tmp_path: Path
    ):
        """get_process_status includes bootstrapping: True/False flag."""
        supervisor = _make_supervisor(tmp_path)

        handle = _make_mock_handle("alice")
        supervisor.processes["alice"] = handle

        # Not bootstrapping
        status_normal = supervisor.get_process_status("alice")
        assert status_normal["bootstrapping"] is False

        # Bootstrapping
        supervisor._bootstrapping.add("alice")
        status_boot = supervisor.get_process_status("alice")
        assert status_boot["bootstrapping"] is True

    @pytest.mark.asyncio
    async def test_supervisor_process_status_not_bootstrapping_shows_running(
        self, tmp_path: Path
    ):
        """When not bootstrapping, status shows the handle's actual state."""
        supervisor = _make_supervisor(tmp_path)

        handle = _make_mock_handle("alice")
        handle.state = ProcessState.RUNNING
        supervisor.processes["alice"] = handle

        status = supervisor.get_process_status("alice")

        assert status["status"] == "running"
        assert status["bootstrapping"] is False


# ════════════════════════════════════════════════════════════════════
# 4. Chat endpoint bootstrap guard
# ════════════════════════════════════════════════════════════════════


def _make_test_app(*, is_bootstrapping: bool = False):
    """Create a minimal FastAPI test app with mocked supervisor."""
    from fastapi import FastAPI
    from server.routes.chat import create_chat_router

    app = FastAPI()
    app.state.ws_manager = MagicMock()
    app.state.ws_manager.broadcast = AsyncMock()
    app.state.stream_registry = StreamRegistry()

    supervisor = MagicMock()
    supervisor.processes = {"alice": MagicMock()}
    supervisor.is_bootstrapping = MagicMock(return_value=is_bootstrapping)

    # Non-streaming send_request
    async def _send_request(anima_name, method, params, timeout=60.0):
        if anima_name not in supervisor.processes:
            raise KeyError(anima_name)
        return {
            "response": "Hello from alice",
            "replied_to": [],
            "emotion": "neutral",
            "cached": False,
        }

    supervisor.send_request = _send_request

    # Streaming send_request_stream
    async def _send_request_stream(anima_name, method, params, timeout=120.0):
        if anima_name not in supervisor.processes:
            raise KeyError(anima_name)
        yield IPCResponse(
            id="test",
            stream=True,
            chunk=json.dumps(
                {"type": "text_delta", "text": "Hello"},
                ensure_ascii=False,
            ),
        )
        yield IPCResponse(
            id="test",
            stream=True,
            done=True,
            result={
                "response": "Hello",
                "replied_to": [],
                "cycle_result": {"summary": "Hello"},
            },
        )

    supervisor.send_request_stream = _send_request_stream

    app.state.supervisor = supervisor

    router = create_chat_router()
    app.include_router(router, prefix="/api")
    return app


class TestChatEndpointBootstrapGuard:
    """Tests for chat endpoint bootstrap guards."""

    @pytest.mark.asyncio
    async def test_chat_stream_returns_busy_when_bootstrapping(self):
        """When supervisor.is_bootstrapping returns True, chat/stream returns
        a bootstrap busy SSE event."""
        app = _make_test_app(is_bootstrapping=True)

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "hello"},
            )

        assert response.status_code == 200
        # Should be SSE content type
        assert "text/event-stream" in response.headers["content-type"]
        body = response.text
        # Should contain bootstrap busy event
        assert "event: bootstrap" in body
        assert "busy" in body

    @pytest.mark.asyncio
    async def test_chat_returns_503_when_bootstrapping(self):
        """Non-streaming chat returns 503 when bootstrapping."""
        app = _make_test_app(is_bootstrapping=True)

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/animas/alice/chat",
                json={"message": "hello"},
            )

        assert response.status_code == 503
        data = response.json()
        assert "error" in data

    @pytest.mark.asyncio
    async def test_greet_returns_503_when_bootstrapping(self):
        """Greet returns 503 when bootstrapping."""
        app = _make_test_app(is_bootstrapping=True)

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post("/api/animas/alice/greet")

        assert response.status_code == 503
        data = response.json()
        assert "error" in data

    @pytest.mark.asyncio
    async def test_chat_succeeds_when_not_bootstrapping(self):
        """Non-streaming chat succeeds when not bootstrapping."""
        app = _make_test_app(is_bootstrapping=False)

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/animas/alice/chat",
                json={"message": "hello"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "response" in data

    @pytest.mark.asyncio
    async def test_chat_stream_succeeds_when_not_bootstrapping(self):
        """Streaming chat succeeds when not bootstrapping."""
        app = _make_test_app(is_bootstrapping=False)

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "hello"},
            )

        assert response.status_code == 200
        body = response.text
        # Should contain normal SSE events, not bootstrap busy
        assert "event: text_delta" in body or "event: done" in body
        assert "bootstrap" not in body or '"busy"' not in body

    @pytest.mark.asyncio
    async def test_greet_succeeds_when_not_bootstrapping(self):
        """Greet succeeds when not bootstrapping."""
        app = _make_test_app(is_bootstrapping=False)

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post("/api/animas/alice/greet")

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
