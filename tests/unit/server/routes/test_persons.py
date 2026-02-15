"""Unit tests for server/routes/persons.py — Person API endpoints."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from server.routes.persons import _read_appearance


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


# ── _read_appearance ─────────────────────────────────────


class TestReadAppearance:
    def test_no_file(self, tmp_path):
        result = _read_appearance(tmp_path)
        assert result is None

    def test_valid_json(self, tmp_path):
        (tmp_path / "appearance.json").write_text(
            json.dumps({"hair": "black"}), encoding="utf-8"
        )
        result = _read_appearance(tmp_path)
        assert result == {"hair": "black"}

    def test_empty_json(self, tmp_path):
        (tmp_path / "appearance.json").write_text("{}", encoding="utf-8")
        result = _read_appearance(tmp_path)
        assert result is None

    def test_invalid_json(self, tmp_path):
        (tmp_path / "appearance.json").write_text("not json", encoding="utf-8")
        result = _read_appearance(tmp_path)
        assert result is None


# ── GET /persons ─────────────────────────────────────────


class TestListPersons:
    async def test_list_empty(self):
        app = _make_test_app(person_names=[])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_list_with_persons(self, tmp_path):
        persons_dir = tmp_path / "persons"
        alice_dir = persons_dir / "alice"
        alice_dir.mkdir(parents=True)

        app = _make_test_app(
            persons_dir=persons_dir,
            person_names=["alice"],
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "alice"
        assert data[0]["status"] == "running"


# ── GET /persons/{name} ─────────────────────────────────


class TestGetPerson:
    async def test_found(self, tmp_path):
        persons_dir = tmp_path / "persons"
        alice_dir = persons_dir / "alice"
        alice_dir.mkdir(parents=True)

        with patch("core.memory.manager.MemoryManager") as MockMM:
            mock_mm = MagicMock()
            mock_mm.read_identity.return_value = "# Identity"
            mock_mm.read_injection.return_value = ""
            mock_mm.read_current_state.return_value = "idle"
            mock_mm.read_pending.return_value = ""
            mock_mm.list_knowledge_files.return_value = ["topic1.md"]
            mock_mm.list_episode_files.return_value = ["2026-01-01.md"]
            mock_mm.list_procedure_files.return_value = []
            MockMM.return_value = mock_mm

            app = _make_test_app(
                persons_dir=persons_dir,
                person_names=["alice"],
            )
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/persons/alice")
        data = resp.json()
        assert "status" in data
        assert "identity" in data
        assert data["knowledge_files"] == ["topic1.md"]

    async def test_not_found(self):
        app = _make_test_app(person_names=[])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons/nonexistent")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Person not found: nonexistent"


# ── POST /persons/{name}/trigger ─────────────────────────


class TestTriggerHeartbeat:
    async def test_success(self, tmp_path):
        persons_dir = tmp_path / "persons"
        alice_dir = persons_dir / "alice"
        alice_dir.mkdir(parents=True)

        app = _make_test_app(
            persons_dir=persons_dir,
            person_names=["alice"],
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/persons/alice/trigger")
        data = resp.json()
        assert data["action"] == "skip"

    async def test_not_found(self):
        app = _make_test_app(person_names=[])
        # Make supervisor raise KeyError for unknown person
        app.state.supervisor.send_request = AsyncMock(
            side_effect=KeyError("nobody")
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/persons/nobody/trigger")
        assert resp.status_code == 404
        assert "Person not found" in resp.json()["detail"]


# ── GET /persons/{name}/config ───────────────────────────


class TestGetPersonConfig:
    async def test_person_not_found(self):
        app = _make_test_app(person_names=[])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons/nobody/config")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Person not found: nobody"

    async def test_success(self, tmp_path):
        persons_dir = tmp_path / "persons"
        alice_dir = persons_dir / "alice"
        alice_dir.mkdir(parents=True)

        mock_resolved = MagicMock()
        mock_resolved.model = "claude-sonnet-4-20250514"
        mock_resolved.execution_mode = "a1"
        mock_resolved.model_dump.return_value = {
            "model": "claude-sonnet-4-20250514",
            "execution_mode": "a1",
        }

        with patch(
            "core.config.models.load_config",
            return_value=MagicMock(),
        ), patch(
            "core.config.models.resolve_person_config",
            return_value=(mock_resolved, "anthropic"),
        ):
            app = _make_test_app(
                persons_dir=persons_dir,
                person_names=["alice"],
            )
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.get("/api/persons/alice/config")

        assert resp.status_code == 200
        data = resp.json()
        assert data["person"] == "alice"
        assert data["model"] == "claude-sonnet-4-20250514"
        assert data["execution_mode"] == "a1"
        assert "config" in data


# ── POST /persons/{name}/enable ─────────────────────────


class TestEnablePerson:
    async def test_enable_success(self, tmp_path):
        """Enable a person: status.json written, process started, name added."""
        persons_dir = tmp_path / "persons"
        alice_dir = persons_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "identity.md").write_text("# Alice", encoding="utf-8")

        app = _make_test_app(persons_dir=persons_dir, person_names=[])
        supervisor = app.state.supervisor
        supervisor.processes = {}
        supervisor.start_person = AsyncMock()
        supervisor.stop_person = AsyncMock()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/persons/alice/enable")

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "alice"
        assert data["enabled"] is True

        # Verify status.json was written
        status_file = alice_dir / "status.json"
        assert status_file.exists()
        status_data = json.loads(status_file.read_text(encoding="utf-8"))
        assert status_data["enabled"] is True

        # Process was started and name was added
        supervisor.start_person.assert_awaited_once_with("alice")
        assert "alice" in app.state.person_names

    async def test_enable_already_running(self, tmp_path):
        """When person is already running, status.json is written but start_person is NOT called."""
        persons_dir = tmp_path / "persons"
        alice_dir = persons_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "identity.md").write_text("# Alice", encoding="utf-8")

        app = _make_test_app(persons_dir=persons_dir, person_names=["alice"])
        supervisor = app.state.supervisor
        supervisor.processes = {"alice": MagicMock()}
        supervisor.start_person = AsyncMock()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/persons/alice/enable")

        assert resp.status_code == 200

        # status.json still written
        status_file = alice_dir / "status.json"
        assert status_file.exists()
        status_data = json.loads(status_file.read_text(encoding="utf-8"))
        assert status_data["enabled"] is True

        # start_person NOT called because already running
        supervisor.start_person.assert_not_awaited()

    async def test_enable_not_found(self):
        """Enable a nonexistent person returns 404."""
        app = _make_test_app(person_names=[])
        supervisor = app.state.supervisor
        supervisor.processes = {}

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/persons/nonexistent/enable")

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    async def test_enable_no_identity(self, tmp_path):
        """Directory exists but no identity.md returns 404."""
        persons_dir = tmp_path / "persons"
        alice_dir = persons_dir / "alice"
        alice_dir.mkdir(parents=True)
        # No identity.md created

        app = _make_test_app(persons_dir=persons_dir, person_names=[])
        supervisor = app.state.supervisor
        supervisor.processes = {}

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/persons/alice/enable")

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]


# ── POST /persons/{name}/disable ────────────────────────


class TestDisablePerson:
    async def test_disable_success(self, tmp_path):
        """Disable a running person: status.json written, process stopped, name removed."""
        persons_dir = tmp_path / "persons"
        alice_dir = persons_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "identity.md").write_text("# Alice", encoding="utf-8")

        app = _make_test_app(persons_dir=persons_dir, person_names=["alice"])
        supervisor = app.state.supervisor
        supervisor.processes = {"alice": MagicMock()}
        supervisor.start_person = AsyncMock()
        supervisor.stop_person = AsyncMock()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/persons/alice/disable")

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "alice"
        assert data["enabled"] is False

        # Verify status.json was written
        status_file = alice_dir / "status.json"
        assert status_file.exists()
        status_data = json.loads(status_file.read_text(encoding="utf-8"))
        assert status_data["enabled"] is False

        # Process was stopped and name was removed
        supervisor.stop_person.assert_awaited_once_with("alice")
        assert "alice" not in app.state.person_names

    async def test_disable_not_running(self, tmp_path):
        """When person is not running, status.json is written but stop_person is NOT called."""
        persons_dir = tmp_path / "persons"
        alice_dir = persons_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "identity.md").write_text("# Alice", encoding="utf-8")

        app = _make_test_app(persons_dir=persons_dir, person_names=[])
        supervisor = app.state.supervisor
        supervisor.processes = {}
        supervisor.stop_person = AsyncMock()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/persons/alice/disable")

        assert resp.status_code == 200

        # status.json still written
        status_file = alice_dir / "status.json"
        assert status_file.exists()
        status_data = json.loads(status_file.read_text(encoding="utf-8"))
        assert status_data["enabled"] is False

        # stop_person NOT called because not running
        supervisor.stop_person.assert_not_awaited()

    async def test_disable_not_found(self):
        """Disable a nonexistent person returns 404."""
        app = _make_test_app(person_names=[])
        supervisor = app.state.supervisor
        supervisor.processes = {}

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/persons/nonexistent/disable")

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]
