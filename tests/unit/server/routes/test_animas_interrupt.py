"""Unit tests for POST /api/animas/{name}/interrupt endpoint."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient


def _make_test_app(
    anima_names: list[str] | None = None,
    supervisor: MagicMock | None = None,
):
    from fastapi import FastAPI
    from server.routes.animas import create_animas_router

    app = FastAPI()
    app.state.animas_dir = Path("/tmp/fake/animas")
    app.state.anima_names = anima_names or []

    if supervisor is None:
        supervisor = MagicMock()
        supervisor.get_process_status.return_value = {"status": "running", "pid": 1234}
        supervisor.processes = {n: MagicMock() for n in (anima_names or [])}
        supervisor.send_request = AsyncMock(
            return_value={"status": "interrupted", "name": "test"},
        )
    app.state.supervisor = supervisor

    router = create_animas_router()
    app.include_router(router, prefix="/api")
    return app


class TestInterruptEndpoint:
    """Tests for POST /api/animas/{name}/interrupt."""

    async def test_interrupt_success(self):
        """Interrupt a running anima should return interrupted status."""
        app = _make_test_app(anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/interrupt")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "interrupted"

    async def test_interrupt_not_found(self):
        """Interrupt a non-existent anima should return 404."""
        app = _make_test_app(anima_names=[])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/nobody/interrupt")
        assert resp.status_code == 404

    async def test_interrupt_not_running(self):
        """Interrupt a stopped anima should return not_running."""
        sup = MagicMock()
        sup.processes = {}  # No running processes
        sup.send_request = AsyncMock()
        app = _make_test_app(anima_names=["bob"], supervisor=sup)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/bob/interrupt")
        assert resp.status_code == 200
        assert resp.json()["status"] == "not_running"

    async def test_interrupt_timeout(self):
        """Interrupt that times out should return timeout status."""
        sup = MagicMock()
        sup.processes = {"carol": MagicMock()}
        sup.send_request = AsyncMock(side_effect=asyncio.TimeoutError())
        app = _make_test_app(anima_names=["carol"], supervisor=sup)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/carol/interrupt")
        assert resp.status_code == 200
        assert resp.json()["status"] == "timeout"

    async def test_interrupt_error(self):
        """Interrupt that raises exception should return 500."""
        sup = MagicMock()
        sup.processes = {"dave": MagicMock()}
        sup.send_request = AsyncMock(side_effect=RuntimeError("IPC failed"))
        app = _make_test_app(anima_names=["dave"], supervisor=sup)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/dave/interrupt")
        assert resp.status_code == 500


class TestStartStopRestartEndpoints:
    """Tests for existing start/stop/restart endpoints to ensure they still work."""

    async def test_stop_success(self):
        """Stop a running anima."""
        sup = MagicMock()
        sup.processes = {"alice": MagicMock()}
        sup.stop_anima = AsyncMock()
        app = _make_test_app(anima_names=["alice"], supervisor=sup)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/stop")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"

    async def test_start_already_running(self):
        """Start an already running anima."""
        sup = MagicMock()
        sup.get_process_status.return_value = {"status": "running"}
        sup.processes = {"alice": MagicMock()}
        sup.start_anima = AsyncMock()
        app = _make_test_app(anima_names=["alice"], supervisor=sup)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/start")
        assert resp.status_code == 200
        assert resp.json()["status"] == "already_running"

    async def test_restart_success(self):
        """Restart a running anima."""
        sup = MagicMock()
        sup.processes = {"alice": MagicMock()}
        sup.restart_anima = AsyncMock()
        sup.get_process_status.return_value = {"status": "running", "pid": 5678}
        app = _make_test_app(anima_names=["alice"], supervisor=sup)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/restart")
        assert resp.status_code == 200
        assert resp.json()["status"] == "restarted"
