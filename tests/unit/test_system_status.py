"""Unit tests for scheduler_running field in /api/system/status endpoint."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient


def _make_test_app(
    anima_names: list[str] | None = None,
    scheduler: MagicMock | None | object = ...,
) -> "FastAPI":  # noqa: F821
    """Create a minimal FastAPI app with the system router for testing.

    Args:
        anima_names: List of anima names to register.
        scheduler: Mock scheduler to attach to supervisor.
            Use ``...`` (sentinel) to leave default MagicMock attribute,
            ``None`` to explicitly delete the attribute (no scheduler),
            or a ``MagicMock`` instance for a present scheduler.
    """
    from fastapi import FastAPI

    from server.routes.system import create_system_router

    app = FastAPI()
    app.state.animas_dir = Path("/tmp/fake/animas")
    app.state.shared_dir = Path("/tmp/fake/shared")
    app.state.anima_names = anima_names if anima_names is not None else []

    # Mock supervisor
    supervisor = MagicMock()
    supervisor.get_all_status.return_value = {}
    app.state.supervisor = supervisor

    # Mock ws_manager
    ws_manager = MagicMock()
    ws_manager.active_connections = []
    app.state.ws_manager = ws_manager

    # Configure scheduler presence
    if scheduler is ...:
        # Default: scheduler not running
        supervisor.is_scheduler_running.return_value = False
    elif scheduler is None:
        # Explicitly no scheduler
        supervisor.is_scheduler_running.return_value = False
        supervisor.scheduler = None
    else:
        supervisor.is_scheduler_running.return_value = True
        supervisor.scheduler = scheduler

    router = create_system_router()
    app.include_router(router, prefix="/api")
    return app


# ── scheduler_running field ─────────────────────────────


class TestSystemStatusSchedulerRunning:
    """Tests for the scheduler_running field in /api/system/status."""

    async def test_response_contains_scheduler_running_field(self) -> None:
        """The response must include scheduler_running regardless of state."""
        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/status")
        data = resp.json()
        assert "scheduler_running" in data

    async def test_scheduler_running_true_when_scheduler_exists(self) -> None:
        """scheduler_running should be True when supervisor has a scheduler."""
        mock_scheduler = MagicMock()
        app = _make_test_app(scheduler=mock_scheduler)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/status")
        data = resp.json()
        assert data["scheduler_running"] is True

    async def test_scheduler_running_false_when_no_scheduler(self) -> None:
        """scheduler_running should be False when supervisor has no scheduler."""
        app = _make_test_app(scheduler=None)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/status")
        data = resp.json()
        assert data["scheduler_running"] is False

    async def test_response_contains_animas_and_processes(self) -> None:
        """The response should also include animas and processes fields."""
        app = _make_test_app(anima_names=["alice", "bob"])
        app.state.supervisor.get_all_status.return_value = {
            "alice": {"status": "running", "pid": 1234},
            "bob": {"status": "stopped", "pid": None},
        }
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/status")
        data = resp.json()
        assert data["animas"] == 2
        assert "processes" in data
        assert "alice" in data["processes"]
        assert "bob" in data["processes"]
