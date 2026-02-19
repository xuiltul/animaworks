# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for error handling through the full FastAPI application stack.

Uses httpx AsyncClient + ASGITransport to test create_app() with a mocked
ProcessSupervisor, verifying that errors surface correctly in HTTP/SSE responses.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import httpx
import pytest

from core.config import invalidate_cache
from core.supervisor.ipc import IPCResponse
from server.app import create_app
from tests.helpers.filesystem import create_anima_dir, create_test_data_dir


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def e2e_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create an isolated data directory for E2E tests."""
    d = create_test_data_dir(tmp_path)
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(d))
    invalidate_cache()
    yield d
    invalidate_cache()


@pytest.fixture
def e2e_anima_dir(e2e_data_dir: Path) -> Path:
    """Create a test anima directory."""
    invalidate_cache()
    anima_dir = create_anima_dir(e2e_data_dir, "alice")
    invalidate_cache()
    return anima_dir


@pytest.fixture
def mock_supervisor() -> MagicMock:
    """Create a MagicMock mimicking ProcessSupervisor."""
    sup = MagicMock()
    sup.processes = {"alice": MagicMock()}
    sup.is_bootstrapping = MagicMock(return_value=False)
    sup.send_request = AsyncMock()
    sup.send_request_stream = MagicMock()
    sup.start_all = AsyncMock()
    sup.shutdown_all = AsyncMock()
    return sup


@pytest.fixture
def app(e2e_data_dir: Path, e2e_anima_dir: Path, mock_supervisor: MagicMock, monkeypatch: pytest.MonkeyPatch):
    """Create the FastAPI app with mocked supervisor."""
    animas_dir = e2e_data_dir / "animas"
    shared_dir = e2e_data_dir / "shared"

    # Mock load_auth to return local_trust mode so auth_guard middleware passes
    auth_cfg = MagicMock()
    auth_cfg.auth_mode = "local_trust"
    monkeypatch.setattr("server.app.load_auth", lambda: auth_cfg)

    application = create_app(animas_dir, shared_dir)

    # Override the supervisor and setup_complete state
    application.state.supervisor = mock_supervisor
    application.state.anima_names = ["alice"]
    application.state.setup_complete = True

    return application


@pytest.fixture
async def client(app):
    """Create an async httpx client bound to the ASGI app."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ── Test 1: Streaming SSE error event ─────────────────────────────────


async def test_streaming_chat_error_returns_sse_error_event(
    client: httpx.AsyncClient, mock_supervisor: MagicMock,
):
    """When supervisor.send_request_stream raises mid-stream, the SSE
    response should contain an event: error frame with a code field."""

    async def _failing_stream(*args: Any, **kwargs: Any):
        # Yield one text chunk, then raise
        yield IPCResponse(
            id="req-1",
            stream=True,
            chunk=json.dumps({"type": "text_delta", "text": "Hello"}),
        )
        raise RuntimeError("LLM backend crashed")

    mock_supervisor.send_request_stream.return_value = _failing_stream()

    resp = await client.post(
        "/api/animas/alice/chat/stream",
        json={"message": "Hi", "from_person": "human"},
    )

    assert resp.status_code == 200
    body = resp.text

    # Must contain SSE error event
    assert "event: error" in body

    # Parse the error data line
    error_data = None
    for line in body.splitlines():
        if line.startswith("data: ") and error_data is None:
            # Only capture the data line right after "event: error"
            pass
    # More robust: find the error event block
    blocks = body.split("\n\n")
    for block in blocks:
        if "event: error" in block:
            for line in block.splitlines():
                if line.startswith("data: "):
                    error_data = json.loads(line[len("data: "):])
                    break

    assert error_data is not None, f"No error data found in SSE body: {body!r}"
    assert "code" in error_data, f"Error data missing 'code' field: {error_data}"


# ── Test 2: Non-streaming timeout returns 504 ─────────────────────────


async def test_non_streaming_chat_timeout_returns_504(
    client: httpx.AsyncClient, mock_supervisor: MagicMock,
):
    """When supervisor.send_request raises TimeoutError, the response
    should be HTTP 504."""

    mock_supervisor.send_request.side_effect = asyncio.TimeoutError()

    resp = await client.post(
        "/api/animas/alice/chat",
        json={"message": "Hi", "from_person": "human"},
    )

    assert resp.status_code == 504
    data = resp.json()
    assert "error" in data


# ── Test 3: Memory route file I/O error returns 500 ───────────────────


async def test_memory_episode_io_error_returns_500(
    client: httpx.AsyncClient, e2e_anima_dir: Path,
):
    """When an episode file has no read permission, GET episodes/{date}
    should return HTTP 500."""

    # Create an episode file with restricted permissions
    episodes_dir = e2e_anima_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    episode_file = episodes_dir / "2026-01-01.md"
    episode_file.write_text("# Episode content", encoding="utf-8")

    # Remove read permission
    episode_file.chmod(0o000)

    try:
        resp = await client.get("/api/animas/alice/episodes/2026-01-01")
        assert resp.status_code == 500
        data = resp.json()
        assert "detail" in data or "error" in data
    finally:
        # Restore permissions for cleanup
        episode_file.chmod(stat.S_IRUSR | stat.S_IWUSR)


# ── Test 4: Global exception handler catches unhandled exceptions ──────


async def test_global_exception_handler(
    e2e_data_dir: Path, e2e_anima_dir: Path, mock_supervisor: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
    """Unhandled exceptions in routes should be caught by the global
    exception handler and return a 500 JSON response."""
    from fastapi import APIRouter

    animas_dir = e2e_data_dir / "animas"
    shared_dir = e2e_data_dir / "shared"

    # Mock load_auth to return local_trust mode so auth_guard middleware passes
    auth_cfg = MagicMock()
    auth_cfg.auth_mode = "local_trust"
    monkeypatch.setattr("server.app.load_auth", lambda: auth_cfg)

    application = create_app(animas_dir, shared_dir)
    application.state.supervisor = mock_supervisor
    application.state.anima_names = ["alice"]
    application.state.setup_complete = True

    # Add a test route that raises an unhandled exception.
    # Insert it BEFORE static file mounts so it takes precedence.
    test_router = APIRouter()

    @test_router.get("/api/test-unhandled-error")
    async def _raise_unhandled():
        raise ValueError("This is an unhandled test exception")

    # Insert the router's routes before the catch-all static mount
    application.include_router(test_router)
    # Move the newly added route before the static mounts
    new_route = application.routes.pop()
    application.routes.insert(0, new_route)

    transport = httpx.ASGITransport(app=application, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/api/test-unhandled-error")

    assert resp.status_code == 500
    data = resp.json()
    assert "error" in data
