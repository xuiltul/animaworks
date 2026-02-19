# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for bootstrap UI feature in server/routes/animas.py.

Tests the bootstrapping flag in list_animas and the POST /animas/{name}/start
endpoint added for anima lifecycle visual states.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from httpx import ASGITransport, AsyncClient


# ── Helper to build a minimal FastAPI app with animas router ──


def _make_test_app(
    animas_dir: Path | None = None,
    anima_names: list[str] | None = None,
):
    from fastapi import FastAPI

    from server.routes.animas import create_animas_router

    app = FastAPI()
    app.state.animas_dir = animas_dir or Path("/tmp/fake/animas")
    app.state.anima_names = anima_names or []

    # Mock supervisor
    supervisor = MagicMock()
    supervisor.get_process_status.return_value = {"status": "running", "pid": 1234}
    supervisor.send_request = AsyncMock(return_value={"action": "skip", "summary": "ok"})
    app.state.supervisor = supervisor

    router = create_animas_router()
    app.include_router(router, prefix="/api")
    return app


# ── GET /animas — bootstrapping flag ────────────────────


class TestListAnimasBootstrapFlag:
    """Verify that list_animas includes the bootstrapping flag from proc_status."""

    async def test_list_animas_includes_bootstrapping_flag(self, tmp_path):
        """When supervisor reports bootstrapping=True, API response should reflect it."""
        animas_dir = tmp_path / "animas"
        (animas_dir / "alice").mkdir(parents=True)

        app = _make_test_app(
            animas_dir=animas_dir,
            anima_names=["alice"],
        )
        app.state.supervisor.get_process_status.return_value = {
            "status": "bootstrapping",
            "bootstrapping": True,
            "pid": 5678,
        }

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "alice"
        assert data[0]["status"] == "bootstrapping"
        assert data[0]["bootstrapping"] is True

    async def test_list_animas_bootstrapping_false_by_default(self, tmp_path):
        """When supervisor omits bootstrapping key, API defaults to False."""
        animas_dir = tmp_path / "animas"
        (animas_dir / "alice").mkdir(parents=True)

        app = _make_test_app(
            animas_dir=animas_dir,
            anima_names=["alice"],
        )
        # Supervisor returns status without bootstrapping key
        app.state.supervisor.get_process_status.return_value = {
            "status": "running",
            "pid": 1234,
        }

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["bootstrapping"] is False


# ── POST /animas/{name}/start ───────────────────────────


class TestStartAnima:
    """Verify the POST /animas/{name}/start endpoint."""

    async def test_start_stopped_anima(self, tmp_path):
        """Starting an anima with status 'not_found' should call start_anima."""
        animas_dir = tmp_path / "animas"
        (animas_dir / "alice").mkdir(parents=True)

        app = _make_test_app(
            animas_dir=animas_dir,
            anima_names=["alice"],
        )
        app.state.supervisor.get_process_status.return_value = {
            "status": "not_found",
        }
        app.state.supervisor.start_anima = AsyncMock()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/start")

        assert resp.status_code == 200
        data = resp.json()
        assert data == {"status": "started", "name": "alice"}
        app.state.supervisor.start_anima.assert_awaited_once_with("alice")

    async def test_start_already_running_anima(self, tmp_path):
        """Starting an anima already running should return already_running without calling start."""
        animas_dir = tmp_path / "animas"
        (animas_dir / "alice").mkdir(parents=True)

        app = _make_test_app(
            animas_dir=animas_dir,
            anima_names=["alice"],
        )
        app.state.supervisor.get_process_status.return_value = {
            "status": "running",
            "pid": 1234,
        }
        app.state.supervisor.start_anima = AsyncMock()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/start")

        assert resp.status_code == 200
        data = resp.json()
        assert data == {"status": "already_running", "current_status": "running"}
        app.state.supervisor.start_anima.assert_not_awaited()

    async def test_start_unknown_anima(self):
        """Starting an anima not in anima_names should return 404."""
        app = _make_test_app(anima_names=[])
        app.state.supervisor.start_anima = AsyncMock()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/nobody/start")

        assert resp.status_code == 404
        assert "Anima not found" in resp.json()["detail"]
        app.state.supervisor.start_anima.assert_not_awaited()

    async def test_start_stopped_anima_status_stopped(self, tmp_path):
        """Starting an anima with status 'stopped' should call start_anima."""
        animas_dir = tmp_path / "animas"
        (animas_dir / "alice").mkdir(parents=True)

        app = _make_test_app(
            animas_dir=animas_dir,
            anima_names=["alice"],
        )
        app.state.supervisor.get_process_status.return_value = {
            "status": "stopped",
        }
        app.state.supervisor.start_anima = AsyncMock()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/start")

        assert resp.status_code == 200
        data = resp.json()
        assert data == {"status": "started", "name": "alice"}
        app.state.supervisor.start_anima.assert_awaited_once_with("alice")
