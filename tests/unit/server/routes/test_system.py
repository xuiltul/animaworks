"""Unit tests for server/routes/system.py — System endpoints."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


def _make_test_app(
    persons: dict | None = None,
    persons_dir: Path | None = None,
    shared_dir: Path | None = None,
    person_names: list[str] | None = None,
):
    from fastapi import FastAPI
    from server.routes.system import create_system_router

    app = FastAPI()
    app.state.persons = persons or {}
    app.state.persons_dir = persons_dir or Path("/tmp/fake/persons")
    app.state.shared_dir = shared_dir or Path("/tmp/fake/shared")
    app.state.person_names = (
        person_names if person_names is not None
        else list((persons or {}).keys())
    )

    # Mock supervisor
    supervisor = MagicMock()
    supervisor.get_all_status.return_value = {}
    supervisor.get_process_status.return_value = {"status": "running", "pid": 1234}
    supervisor.start_person = AsyncMock()
    supervisor.stop_person = AsyncMock()
    supervisor.restart_person = AsyncMock()
    app.state.supervisor = supervisor

    # Mock ws_manager
    ws_manager = MagicMock()
    ws_manager.active_connections = []
    app.state.ws_manager = ws_manager

    router = create_system_router()
    app.include_router(router, prefix="/api")
    return app


# ── GET /shared/users ────────────────────────────────────


class TestListSharedUsers:
    async def test_no_users_dir(self, tmp_path):
        shared_dir = tmp_path / "shared"
        # Don't create users dir
        app = _make_test_app(shared_dir=shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/shared/users")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_with_users(self, tmp_path):
        shared_dir = tmp_path / "shared"
        users_dir = shared_dir / "users"
        users_dir.mkdir(parents=True)
        (users_dir / "alice").mkdir()
        (users_dir / "bob").mkdir()
        # Also create a file (should be ignored)
        (users_dir / "readme.txt").write_text("ignore", encoding="utf-8")

        app = _make_test_app(shared_dir=shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/shared/users")
        data = resp.json()
        assert "alice" in data
        assert "bob" in data
        assert "readme.txt" not in data


# ── GET /system/status ───────────────────────────────────


class TestSystemStatus:
    async def test_status(self):
        app = _make_test_app(person_names=["alice"])
        app.state.supervisor.get_all_status.return_value = {
            "alice": {"status": "running", "pid": 1234},
        }
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/status")
        data = resp.json()
        assert data["persons"] == 1
        assert "processes" in data

    async def test_status_empty(self):
        app = _make_test_app(person_names=[])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/status")
        data = resp.json()
        assert data["persons"] == 0


# ── POST /system/reload ─────────────────────────────────


class TestReloadPersons:
    async def test_reload_adds_new_persons(self, tmp_path):
        persons_dir = tmp_path / "persons"
        persons_dir.mkdir()
        shared_dir = tmp_path / "shared"

        # Create a new person on disk
        alice_dir = persons_dir / "alice"
        alice_dir.mkdir()
        (alice_dir / "identity.md").write_text("# Alice", encoding="utf-8")

        app = _make_test_app(
            persons={},
            persons_dir=persons_dir,
            shared_dir=shared_dir,
            person_names=[],
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/system/reload")

        data = resp.json()
        assert "alice" in data["added"]
        assert data["total"] == 1

    async def test_reload_removes_deleted_persons(self, tmp_path):
        persons_dir = tmp_path / "persons"
        persons_dir.mkdir()
        shared_dir = tmp_path / "shared"

        app = _make_test_app(
            persons={},
            persons_dir=persons_dir,
            shared_dir=shared_dir,
            person_names=["deleted"],
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/system/reload")

        data = resp.json()
        assert "deleted" in data["removed"]
        assert data["total"] == 0

    async def test_reload_refreshes_existing(self, tmp_path):
        persons_dir = tmp_path / "persons"
        persons_dir.mkdir()
        shared_dir = tmp_path / "shared"

        alice_dir = persons_dir / "alice"
        alice_dir.mkdir()
        (alice_dir / "identity.md").write_text("# Alice", encoding="utf-8")

        app = _make_test_app(
            persons={},
            persons_dir=persons_dir,
            shared_dir=shared_dir,
            person_names=["alice"],
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/system/reload")

        data = resp.json()
        assert "alice" in data["refreshed"]

    async def test_reload_no_persons_dir(self, tmp_path):
        persons_dir = tmp_path / "nonexistent"
        shared_dir = tmp_path / "shared"

        app = _make_test_app(
            persons={},
            persons_dir=persons_dir,
            shared_dir=shared_dir,
            person_names=[],
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/system/reload")
        data = resp.json()
        assert data["total"] == 0


# ── GET /activity/recent ─────────────────────────────────


class TestRecentActivity:
    async def test_activity_no_persons(self):
        app = _make_test_app(persons={})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent")
        data = resp.json()
        assert data["events"] == []

    async def test_activity_with_hours_param(self):
        app = _make_test_app(persons={})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?hours=1")
        assert resp.status_code == 200

    async def test_activity_with_person_filter(self):
        app = _make_test_app(persons={"alice": MagicMock()})
        # Mock the person's person_dir to avoid filesystem access
        alice = app.state.persons["alice"]
        alice.person_dir = Path("/tmp/nonexistent")
        mc = MagicMock()
        mc.model = "test"
        alice.model_config = mc

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?person=alice")
        assert resp.status_code == 200


# ── GET /system/connections ──────────────────────────────


class TestSystemConnections:
    async def test_connections_with_active_clients(self):
        app = _make_test_app(person_names=["alice", "bob"])
        # Simulate 3 active websocket connections
        app.state.ws_manager.active_connections = [
            MagicMock(), MagicMock(), MagicMock(),
        ]

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/connections")
        data = resp.json()
        assert data["websocket"]["connected_clients"] == 3
        assert "alice" in data["processes"]
        assert "bob" in data["processes"]

    async def test_connections_without_active_connections_attr(self):
        app = _make_test_app(person_names=["alice"])
        # Remove active_connections attribute
        del app.state.ws_manager.active_connections

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/connections")
        data = resp.json()
        assert data["websocket"]["connected_clients"] == 0

    async def test_connections_empty(self):
        app = _make_test_app(person_names=[])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/connections")
        data = resp.json()
        assert data["websocket"]["connected_clients"] == 0
        assert data["processes"] == {}


# ── GET /system/scheduler ────────────────────────────────


class TestSystemScheduler:
    async def test_no_scheduler(self):
        app = _make_test_app()
        # Supervisor has no scheduler attribute
        del app.state.supervisor.scheduler

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/scheduler")
        data = resp.json()
        assert data["running"] is False
        assert data["jobs"] == []

    async def test_scheduler_with_jobs(self):
        app = _make_test_app()

        mock_job = MagicMock()
        mock_job.id = "hb-alice"
        mock_job.name = "heartbeat:alice"
        mock_job.next_run_time = "2026-01-01T12:00:00"
        mock_job.trigger = "interval[0:05:00]"

        scheduler = MagicMock()
        scheduler.get_jobs.return_value = [mock_job]
        app.state.supervisor.scheduler = scheduler

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/scheduler")
        data = resp.json()
        assert data["running"] is True
        assert len(data["jobs"]) == 1
        assert data["jobs"][0]["id"] == "hb-alice"
        assert data["jobs"][0]["name"] == "heartbeat:alice"
        assert data["jobs"][0]["next_run"] == "2026-01-01T12:00:00"
        assert "interval" in data["jobs"][0]["trigger"]

    async def test_scheduler_with_no_jobs(self):
        app = _make_test_app()

        scheduler = MagicMock()
        scheduler.get_jobs.return_value = []
        app.state.supervisor.scheduler = scheduler

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/scheduler")
        data = resp.json()
        assert data["running"] is True
        assert data["jobs"] == []

    async def test_scheduler_job_without_next_run(self):
        app = _make_test_app()

        mock_job = MagicMock()
        mock_job.id = "cron-bob"
        mock_job.name = "cron:bob"
        mock_job.next_run_time = None
        mock_job.trigger = "cron[hour=9]"

        scheduler = MagicMock()
        scheduler.get_jobs.return_value = [mock_job]
        app.state.supervisor.scheduler = scheduler

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/scheduler")
        data = resp.json()
        assert data["jobs"][0]["next_run"] is None
