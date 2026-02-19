"""Unit tests for stop/restart endpoints in server/routes/animas.py."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient


# ── Helper ───────────────────────────────────────────────


def _create_app(
    anima_names: list[str] | None = None,
    processes: dict | None = None,
    process_status: dict | None = None,
):
    """Build a minimal FastAPI app with the animas router and mocked supervisor.

    Parameters
    ----------
    anima_names:
        List of known anima names (set on ``app.state.anima_names``).
    processes:
        Dict simulating ``supervisor.processes`` (running anima entries).
    process_status:
        Dict returned by ``supervisor.get_process_status()``.
    """
    from fastapi import FastAPI

    from server.routes.animas import create_animas_router

    app = FastAPI()
    app.state.animas_dir = Path("/tmp/fake/animas")
    app.state.anima_names = anima_names or []

    supervisor = MagicMock()
    supervisor.processes = processes if processes is not None else {}
    supervisor.stop_anima = AsyncMock()
    supervisor.restart_anima = AsyncMock()
    supervisor.get_process_status.return_value = process_status or {
        "status": "running",
        "pid": 12345,
    }
    app.state.supervisor = supervisor

    router = create_animas_router()
    app.include_router(router, prefix="/api")
    return app


# ── POST /api/animas/{name}/stop ─────────────────────────


class TestStopAnima:
    """Tests for the POST /api/animas/{name}/stop endpoint."""

    async def test_stop_anima_success(self) -> None:
        """Stopping a running anima returns status='stopped' and calls stop_anima."""
        app = _create_app(
            anima_names=["alice"],
            processes={"alice": MagicMock()},
        )
        supervisor = app.state.supervisor

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/stop")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "stopped"
        assert data["name"] == "alice"

        supervisor.stop_anima.assert_awaited_once_with("alice")

    async def test_stop_anima_already_stopped(self) -> None:
        """Stopping an anima not in supervisor.processes returns 'already_stopped'."""
        app = _create_app(
            anima_names=["alice"],
            processes={},  # alice is NOT in processes
        )
        supervisor = app.state.supervisor

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/stop")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "already_stopped"
        assert data["name"] == "alice"

        # stop_anima should NOT be called
        supervisor.stop_anima.assert_not_awaited()

    async def test_stop_anima_not_found(self) -> None:
        """Stopping an unknown anima returns 404."""
        app = _create_app(anima_names=[])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/nonexistent/stop")

        assert resp.status_code == 404
        assert "Anima not found" in resp.json()["detail"]


# ── POST /api/animas/{name}/restart ──────────────────────


class TestRestartAnima:
    """Tests for the POST /api/animas/{name}/restart endpoint."""

    async def test_restart_anima_success(self) -> None:
        """Restarting a known anima returns status='restarted', name, and pid."""
        expected_pid = 99999
        app = _create_app(
            anima_names=["bob"],
            processes={"bob": MagicMock()},
            process_status={"status": "running", "pid": expected_pid},
        )
        supervisor = app.state.supervisor

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/bob/restart")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "restarted"
        assert data["name"] == "bob"
        assert data["pid"] == expected_pid

        supervisor.restart_anima.assert_awaited_once_with("bob")

    async def test_restart_anima_not_found(self) -> None:
        """Restarting an unknown anima returns 404."""
        app = _create_app(anima_names=[])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/nonexistent/restart")

        assert resp.status_code == 404
        assert "Anima not found" in resp.json()["detail"]
