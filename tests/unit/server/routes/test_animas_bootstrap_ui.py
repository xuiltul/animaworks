"""Unit tests for bootstrap UI feature in server/routes/persons.py.

Tests the bootstrapping flag in list_persons and the POST /persons/{name}/start
endpoint added for person lifecycle visual states.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from httpx import ASGITransport, AsyncClient


# ── Helper to build a minimal FastAPI app with persons router ──


def _make_test_app(
    persons_dir: Path | None = None,
    person_names: list[str] | None = None,
):
    from fastapi import FastAPI

    from server.routes.persons import create_persons_router

    app = FastAPI()
    app.state.persons_dir = persons_dir or Path("/tmp/fake/persons")
    app.state.person_names = person_names or []

    # Mock supervisor
    supervisor = MagicMock()
    supervisor.get_process_status.return_value = {"status": "running", "pid": 1234}
    supervisor.send_request = AsyncMock(return_value={"action": "skip", "summary": "ok"})
    app.state.supervisor = supervisor

    router = create_persons_router()
    app.include_router(router, prefix="/api")
    return app


# ── GET /persons — bootstrapping flag ────────────────────


class TestListPersonsBootstrapFlag:
    """Verify that list_persons includes the bootstrapping flag from proc_status."""

    async def test_list_persons_includes_bootstrapping_flag(self, tmp_path):
        """When supervisor reports bootstrapping=True, API response should reflect it."""
        persons_dir = tmp_path / "persons"
        (persons_dir / "alice").mkdir(parents=True)

        app = _make_test_app(
            persons_dir=persons_dir,
            person_names=["alice"],
        )
        app.state.supervisor.get_process_status.return_value = {
            "status": "bootstrapping",
            "bootstrapping": True,
            "pid": 5678,
        }

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "alice"
        assert data[0]["status"] == "bootstrapping"
        assert data[0]["bootstrapping"] is True

    async def test_list_persons_bootstrapping_false_by_default(self, tmp_path):
        """When supervisor omits bootstrapping key, API defaults to False."""
        persons_dir = tmp_path / "persons"
        (persons_dir / "alice").mkdir(parents=True)

        app = _make_test_app(
            persons_dir=persons_dir,
            person_names=["alice"],
        )
        # Supervisor returns status without bootstrapping key
        app.state.supervisor.get_process_status.return_value = {
            "status": "running",
            "pid": 1234,
        }

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["bootstrapping"] is False


# ── POST /persons/{name}/start ───────────────────────────


class TestStartPerson:
    """Verify the POST /persons/{name}/start endpoint."""

    async def test_start_stopped_person(self, tmp_path):
        """Starting a person with status 'not_found' should call start_person."""
        persons_dir = tmp_path / "persons"
        (persons_dir / "alice").mkdir(parents=True)

        app = _make_test_app(
            persons_dir=persons_dir,
            person_names=["alice"],
        )
        app.state.supervisor.get_process_status.return_value = {
            "status": "not_found",
        }
        app.state.supervisor.start_person = AsyncMock()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/persons/alice/start")

        assert resp.status_code == 200
        data = resp.json()
        assert data == {"status": "started", "name": "alice"}
        app.state.supervisor.start_person.assert_awaited_once_with("alice")

    async def test_start_already_running_person(self, tmp_path):
        """Starting a person already running should return already_running without calling start."""
        persons_dir = tmp_path / "persons"
        (persons_dir / "alice").mkdir(parents=True)

        app = _make_test_app(
            persons_dir=persons_dir,
            person_names=["alice"],
        )
        app.state.supervisor.get_process_status.return_value = {
            "status": "running",
            "pid": 1234,
        }
        app.state.supervisor.start_person = AsyncMock()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/persons/alice/start")

        assert resp.status_code == 200
        data = resp.json()
        assert data == {"status": "already_running", "current_status": "running"}
        app.state.supervisor.start_person.assert_not_awaited()

    async def test_start_unknown_person(self):
        """Starting a person not in person_names should return 404."""
        app = _make_test_app(person_names=[])
        app.state.supervisor.start_person = AsyncMock()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/persons/nobody/start")

        assert resp.status_code == 404
        assert "Person not found" in resp.json()["detail"]
        app.state.supervisor.start_person.assert_not_awaited()

    async def test_start_stopped_person_status_stopped(self, tmp_path):
        """Starting a person with status 'stopped' should call start_person."""
        persons_dir = tmp_path / "persons"
        (persons_dir / "alice").mkdir(parents=True)

        app = _make_test_app(
            persons_dir=persons_dir,
            person_names=["alice"],
        )
        app.state.supervisor.get_process_status.return_value = {
            "status": "stopped",
        }
        app.state.supervisor.start_person = AsyncMock()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/persons/alice/start")

        assert resp.status_code == 200
        data = resp.json()
        assert data == {"status": "started", "name": "alice"}
        app.state.supervisor.start_person.assert_awaited_once_with("alice")
