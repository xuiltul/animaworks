"""Unit tests for server/routes/persons.py — Person API endpoints."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from server.routes.persons import _read_appearance


# ── Helper to build a minimal FastAPI app with persons router ──


def _make_test_app(persons: dict | None = None):
    from fastapi import FastAPI
    from server.routes.persons import create_persons_router

    app = FastAPI()
    app.state.persons = persons or {}
    router = create_persons_router()
    app.include_router(router, prefix="/api")
    return app


def _make_mock_person(
    name: str = "alice",
    supervisor: str | None = None,
    person_dir: Path | None = None,
):
    person = MagicMock()
    person.name = name
    person.person_dir = person_dir or Path("/tmp/fake/persons") / name

    status = MagicMock()
    status.model_dump.return_value = {"name": name, "status": "idle"}
    person.status = status

    mc = MagicMock()
    mc.supervisor = supervisor
    person.model_config = mc

    memory = MagicMock()
    memory.read_identity.return_value = "# Identity"
    memory.read_injection.return_value = ""
    memory.read_current_state.return_value = "idle"
    memory.read_pending.return_value = ""
    memory.list_knowledge_files.return_value = ["topic1.md"]
    memory.list_episode_files.return_value = ["2026-01-01.md"]
    memory.list_procedure_files.return_value = []
    person.memory = memory

    person.run_heartbeat = AsyncMock()

    return person


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
        app = _make_test_app({})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_list_with_persons(self):
        alice = _make_mock_person("alice")
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "alice"
        assert data[0]["supervisor"] is None


# ── GET /persons/{name} ─────────────────────────────────


class TestGetPerson:
    async def test_found(self):
        alice = _make_mock_person("alice")
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons/alice")
        data = resp.json()
        assert "status" in data
        assert "identity" in data
        assert data["knowledge_files"] == ["topic1.md"]

    async def test_not_found(self):
        app = _make_test_app({})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons/nonexistent")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Person not found: nonexistent"


# ── POST /persons/{name}/trigger ─────────────────────────


class TestTriggerHeartbeat:
    async def test_success(self):
        alice = _make_mock_person("alice")
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"action": "skip", "summary": "ok"}
        alice.run_heartbeat.return_value = mock_result

        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/persons/alice/trigger")
        data = resp.json()
        assert data["action"] == "skip"

    async def test_not_found(self):
        app = _make_test_app({})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/persons/nobody/trigger")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Person not found: nobody"
