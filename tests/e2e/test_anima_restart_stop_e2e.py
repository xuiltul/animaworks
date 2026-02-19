# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for anima restart and stop API endpoints.

Validates the following endpoints through the full FastAPI app stack:

1. POST /api/animas/{name}/stop — stops a specific anima process
2. POST /api/animas/{name}/restart — restarts a specific anima process

These tests exercise the complete request lifecycle including the
setup-guard middleware, route dispatch, and supervisor interactions.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


# ── Helpers ──────────────────────────────────────────────


def _create_app(
    tmp_path: Path,
    anima_names: list[str] | None = None,
    processes: dict[str, object] | None = None,
):
    """Build a real FastAPI app via create_app with mocked externals.

    Returns an app whose setup_complete flag is True so the setup-guard
    middleware lets API requests through.

    Parameters
    ----------
    tmp_path:
        Temporary directory for animas/shared dirs.
    anima_names:
        List of anima names to register. Defaults to empty.
    processes:
        Dict to use as ``supervisor.processes``. If *None*, defaults to
        an entry per *anima_names* key so that ``name in processes``
        evaluates to True for every known anima.
    """
    animas_dir = tmp_path / "animas"
    animas_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

    names = anima_names if anima_names is not None else []

    with (
        patch("server.app.ProcessSupervisor") as mock_sup_cls,
        patch("server.app.load_config") as mock_cfg,
        patch("server.app.WebSocketManager") as mock_ws_cls,
        patch("server.app.load_auth") as mock_auth,
    ):
        cfg = MagicMock()
        cfg.setup_complete = True
        mock_cfg.return_value = cfg

        auth_cfg = MagicMock()
        auth_cfg.auth_mode = "local_trust"
        mock_auth.return_value = auth_cfg

        supervisor = MagicMock()
        supervisor.get_all_status.return_value = {}
        supervisor.get_process_status.return_value = {
            "status": "running",
            "pid": 12345,
            "bootstrapping": False,
            "uptime_sec": 60,
        }
        supervisor.is_scheduler_running.return_value = False
        supervisor.scheduler = None

        # Async methods
        supervisor.stop_anima = AsyncMock()
        supervisor.restart_anima = AsyncMock()
        supervisor.start_anima = AsyncMock()

        # Default processes dict: one entry per anima name
        if processes is not None:
            supervisor.processes = processes
        else:
            supervisor.processes = {n: MagicMock() for n in names}

        mock_sup_cls.return_value = supervisor

        ws_manager = MagicMock()
        ws_manager.active_connections = []
        mock_ws_cls.return_value = ws_manager

        from server.app import create_app

        app = create_app(animas_dir, shared_dir)

    # Persist auth mock beyond the with-block for request-time middleware
    import server.app as _sa
    _auth = MagicMock()
    _auth.auth_mode = "local_trust"
    _sa.load_auth = lambda: _auth

    # Override anima_names
    app.state.anima_names = list(names)

    return app


# ── Test 1: Restart then verify status ───────────────────


class TestRestartThenVerifyStatus:
    """Restart an anima, then call GET /api/animas to verify it is listed."""

    async def test_restart_then_verify_status(self, tmp_path: Path) -> None:
        """After restart, GET /api/animas should show the anima as running."""
        app = _create_app(tmp_path, anima_names=["sakura"])

        # After restart, get_process_status should report running
        app.state.supervisor.get_process_status.return_value = {
            "status": "running",
            "pid": 99999,
            "bootstrapping": False,
            "uptime_sec": 1,
        }

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Restart the anima
            restart_resp = await client.post("/api/animas/sakura/restart")
            assert restart_resp.status_code == 200
            restart_data = restart_resp.json()
            assert restart_data["status"] == "restarted"
            assert restart_data["name"] == "sakura"

            # Verify status via list endpoint
            with patch("server.routes.animas.load_config") as mock_route_cfg:
                mock_route_cfg.return_value = MagicMock(animas={})
                list_resp = await client.get("/api/animas")

            assert list_resp.status_code == 200
            animas = list_resp.json()
            assert len(animas) == 1
            assert animas[0]["name"] == "sakura"
            assert animas[0]["status"] == "running"

        # Verify restart_anima was called exactly once
        app.state.supervisor.restart_anima.assert_awaited_once_with("sakura")


# ── Test 2: Stop then start lifecycle ────────────────────


class TestStopThenStartLifecycle:
    """Stop an anima, verify it is stopped, then start it and verify running."""

    async def test_stop_then_start_lifecycle(self, tmp_path: Path) -> None:
        """Full stop -> verify -> start -> verify lifecycle."""
        app = _create_app(tmp_path, anima_names=["sakura"])
        supervisor = app.state.supervisor

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Step 1: Stop the anima
            stop_resp = await client.post("/api/animas/sakura/stop")
            assert stop_resp.status_code == 200
            stop_data = stop_resp.json()
            assert stop_data["status"] == "stopped"
            assert stop_data["name"] == "sakura"
            supervisor.stop_anima.assert_awaited_once_with("sakura")

            # Step 2: Simulate stopped state — remove from processes
            supervisor.processes = {}
            supervisor.get_process_status.return_value = {
                "status": "stopped",
                "pid": None,
                "bootstrapping": False,
                "uptime_sec": None,
            }

            # Verify it reports stopped
            with patch("server.routes.animas.load_config") as mock_route_cfg:
                mock_route_cfg.return_value = MagicMock(animas={})
                list_resp = await client.get("/api/animas")
            assert list_resp.status_code == 200
            animas = list_resp.json()
            assert animas[0]["status"] == "stopped"

            # Step 3: Start the anima again
            supervisor.get_process_status.return_value = {
                "status": "not_found",
                "pid": None,
                "bootstrapping": False,
                "uptime_sec": None,
            }
            start_resp = await client.post("/api/animas/sakura/start")
            assert start_resp.status_code == 200
            start_data = start_resp.json()
            assert start_data["status"] == "started"
            assert start_data["name"] == "sakura"
            supervisor.start_anima.assert_awaited_once_with("sakura")

            # Step 4: Simulate running state
            supervisor.processes = {"sakura": MagicMock()}
            supervisor.get_process_status.return_value = {
                "status": "running",
                "pid": 54321,
                "bootstrapping": False,
                "uptime_sec": 2,
            }

            with patch("server.routes.animas.load_config") as mock_route_cfg:
                mock_route_cfg.return_value = MagicMock(animas={})
                list_resp = await client.get("/api/animas")
            assert list_resp.status_code == 200
            animas = list_resp.json()
            assert animas[0]["status"] == "running"


# ── Test 3: Restart does not affect other animas ─────────


class TestRestartDoesNotAffectOtherAnimas:
    """With two animas, restarting one should not affect the other."""

    async def test_restart_does_not_affect_other_animas(
        self, tmp_path: Path,
    ) -> None:
        """Restarting sakura should not call restart on kotoha."""
        app = _create_app(tmp_path, anima_names=["sakura", "kotoha"])
        supervisor = app.state.supervisor

        # Per-anima status tracking
        def _get_status(name: str) -> dict:
            return {
                "status": "running",
                "pid": 111 if name == "sakura" else 222,
                "bootstrapping": False,
                "uptime_sec": 60,
            }

        supervisor.get_process_status.side_effect = _get_status

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Restart only sakura
            resp = await client.post("/api/animas/sakura/restart")
            assert resp.status_code == 200
            assert resp.json()["status"] == "restarted"
            assert resp.json()["name"] == "sakura"

            # Verify list shows both animas still running
            with patch("server.routes.animas.load_config") as mock_route_cfg:
                mock_route_cfg.return_value = MagicMock(animas={})
                list_resp = await client.get("/api/animas")

            assert list_resp.status_code == 200
            animas = list_resp.json()
            assert len(animas) == 2

            statuses = {a["name"]: a["status"] for a in animas}
            assert statuses["sakura"] == "running"
            assert statuses["kotoha"] == "running"

        # Key assertion: restart_anima was called ONLY with "sakura"
        supervisor.restart_anima.assert_awaited_once_with("sakura")
        supervisor.stop_anima.assert_not_awaited()


# ── Test 4: Stop nonexistent returns 404 ─────────────────


class TestStopNonexistentReturns404:
    """Stopping a nonexistent anima should return HTTP 404."""

    async def test_stop_nonexistent_returns_404(self, tmp_path: Path) -> None:
        """POST /api/animas/ghost/stop should return 404."""
        app = _create_app(tmp_path, anima_names=["sakura"])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/ghost/stop")

        assert resp.status_code == 404
        data = resp.json()
        assert "ghost" in data["detail"]

        # Supervisor methods should NOT have been called
        app.state.supervisor.stop_anima.assert_not_awaited()


# ── Test 5: Restart nonexistent returns 404 ──────────────


class TestRestartNonexistentReturns404:
    """Restarting a nonexistent anima should return HTTP 404."""

    async def test_restart_nonexistent_returns_404(self, tmp_path: Path) -> None:
        """POST /api/animas/ghost/restart should return 404."""
        app = _create_app(tmp_path, anima_names=["sakura"])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/ghost/restart")

        assert resp.status_code == 404
        data = resp.json()
        assert "ghost" in data["detail"]

        # Supervisor methods should NOT have been called
        app.state.supervisor.restart_anima.assert_not_awaited()
