# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for /api/system/status endpoint with real FastAPI app.

Verifies the complete response structure including the scheduler_running
field through the actual application stack (minus process supervisor startup).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


def _create_real_app(tmp_path: Path) -> "FastAPI":  # noqa: F821
    """Build a real FastAPI app via create_app with mocked externals.

    Returns an app whose setup_complete flag is True so the setup-guard
    middleware lets API requests through.
    """
    animas_dir = tmp_path / "animas"
    animas_dir.mkdir(parents=True)
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True)

    with (
        patch("server.app.ProcessSupervisor") as mock_sup_cls,
        patch("server.app.load_config") as mock_cfg,
        patch("server.app.WebSocketManager") as mock_ws_cls,
        patch("server.app.load_auth") as mock_auth,
    ):
        # Configure mock config
        cfg = MagicMock()
        cfg.setup_complete = True
        mock_cfg.return_value = cfg

        # Configure mock auth
        auth_cfg = MagicMock()
        auth_cfg.auth_mode = "local_trust"
        mock_auth.return_value = auth_cfg

        # Configure mock supervisor
        supervisor = MagicMock()
        supervisor.get_all_status.return_value = {}
        supervisor.is_scheduler_running.return_value = False
        supervisor.scheduler = None
        mock_sup_cls.return_value = supervisor

        # Configure mock ws_manager
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

    return app


class TestSystemStatusE2E:
    """E2E tests for /api/system/status through the full app stack."""

    async def test_system_status_response_structure(self, tmp_path: Path) -> None:
        """Response from real app must contain animas, processes, and scheduler_running."""
        app = _create_real_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/status")

        assert resp.status_code == 200
        data = resp.json()
        assert "animas" in data
        assert "processes" in data
        assert "scheduler_running" in data

    async def test_scheduler_running_reflects_cron_files(
        self, tmp_path: Path,
    ) -> None:
        """scheduler_running should be True when cron.md files define jobs."""
        app = _create_real_app(tmp_path)

        # Create an anima with cron.md
        animas_dir = app.state.animas_dir
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True, exist_ok=True)
        (alice_dir / "cron.md").write_text(
            "# Cron: alice\n\n"
            "## Morning Planning (Daily 9:00 JST)\n"
            "type: llm\n"
            "Plan daily tasks.\n",
            encoding="utf-8",
        )
        app.state.anima_names = ["alice"]
        app.state.supervisor.is_scheduler_running.return_value = True

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/status")

        data = resp.json()
        assert data["scheduler_running"] is True

    async def test_scheduler_running_false_without_cron(
        self, tmp_path: Path,
    ) -> None:
        """scheduler_running should be False when no cron.md files exist."""
        app = _create_real_app(tmp_path)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/status")

        data = resp.json()
        assert data["scheduler_running"] is False
