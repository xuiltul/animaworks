"""Unit tests for server/routes/memory_routes.py — Memory API endpoints."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


def _make_test_app(persons: dict | None = None):
    from fastapi import FastAPI
    from server.routes.memory_routes import create_memory_router

    app = FastAPI()
    app.state.persons = persons or {}
    router = create_memory_router()
    app.include_router(router, prefix="/api")
    return app


def _make_mock_person(name: str = "alice", tmp_path: Path | None = None):
    person = MagicMock()
    person.name = name

    person_dir = tmp_path or Path("/tmp/fake/persons") / name
    person.person_dir = person_dir

    mc = MagicMock()
    mc.model = "test-model"
    person.model_config = mc

    memory = MagicMock()
    memory.episodes_dir = person_dir / "episodes"
    memory.knowledge_dir = person_dir / "knowledge"
    memory.procedures_dir = person_dir / "procedures"
    memory.list_episode_files.return_value = ["2026-01-01.md"]
    memory.list_knowledge_files.return_value = ["topic1.md"]
    memory.list_procedure_files.return_value = ["proc1.md"]
    person.memory = memory

    return person


# ── Episodes ─────────────────────────────────────────────


class TestEpisodes:
    async def test_list_episodes(self):
        alice = _make_mock_person("alice")
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons/alice/episodes")
        assert resp.status_code == 200
        data = resp.json()
        assert data["files"] == ["2026-01-01.md"]

    async def test_list_episodes_not_found(self):
        app = _make_test_app({})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons/nobody/episodes")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Person not found: nobody"

    async def test_get_episode_success(self, tmp_path):
        person_dir = tmp_path / "alice"
        person_dir.mkdir()
        episodes_dir = person_dir / "episodes"
        episodes_dir.mkdir()
        (episodes_dir / "2026-01-01.md").write_text("Today was good.", encoding="utf-8")

        alice = _make_mock_person("alice", tmp_path=person_dir)
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons/alice/episodes/2026-01-01")
        data = resp.json()
        assert data["date"] == "2026-01-01"
        assert data["content"] == "Today was good."

    async def test_get_episode_not_found(self, tmp_path):
        person_dir = tmp_path / "alice"
        person_dir.mkdir()
        episodes_dir = person_dir / "episodes"
        episodes_dir.mkdir()

        alice = _make_mock_person("alice", tmp_path=person_dir)
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons/alice/episodes/9999-01-01")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Episode not found"


# ── Knowledge ────────────────────────────────────────────


class TestKnowledge:
    async def test_list_knowledge(self):
        alice = _make_mock_person("alice")
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons/alice/knowledge")
        assert resp.json()["files"] == ["topic1.md"]

    async def test_get_knowledge_success(self, tmp_path):
        person_dir = tmp_path / "alice"
        person_dir.mkdir()
        knowledge_dir = person_dir / "knowledge"
        knowledge_dir.mkdir()
        (knowledge_dir / "python.md").write_text("Python is great.", encoding="utf-8")

        alice = _make_mock_person("alice", tmp_path=person_dir)
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons/alice/knowledge/python")
        data = resp.json()
        assert data["topic"] == "python"
        assert data["content"] == "Python is great."

    async def test_get_knowledge_not_found(self, tmp_path):
        person_dir = tmp_path / "alice"
        person_dir.mkdir()
        knowledge_dir = person_dir / "knowledge"
        knowledge_dir.mkdir()

        alice = _make_mock_person("alice", tmp_path=person_dir)
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons/alice/knowledge/nonexistent")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Knowledge not found"


# ── Procedures ───────────────────────────────────────────


class TestProcedures:
    async def test_list_procedures(self):
        alice = _make_mock_person("alice")
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons/alice/procedures")
        assert resp.json()["files"] == ["proc1.md"]

    async def test_get_procedure_success(self, tmp_path):
        person_dir = tmp_path / "alice"
        person_dir.mkdir()
        proc_dir = person_dir / "procedures"
        proc_dir.mkdir()
        (proc_dir / "deploy.md").write_text("Step 1: ...", encoding="utf-8")

        alice = _make_mock_person("alice", tmp_path=person_dir)
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons/alice/procedures/deploy")
        data = resp.json()
        assert data["name"] == "deploy"
        assert data["content"] == "Step 1: ..."

    async def test_get_procedure_not_found(self, tmp_path):
        person_dir = tmp_path / "alice"
        person_dir.mkdir()
        proc_dir = person_dir / "procedures"
        proc_dir.mkdir()

        alice = _make_mock_person("alice", tmp_path=person_dir)
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons/alice/procedures/nonexistent")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Procedure not found"


# ── Conversation ─────────────────────────────────────────


class TestConversation:
    async def test_get_conversation_person_not_found(self):
        app = _make_test_app({})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons/nobody/conversation")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Person not found: nobody"

    @patch("server.routes.memory_routes.ConversationMemory", create=True)
    async def test_get_conversation_success(self, mock_conv_cls):
        alice = _make_mock_person("alice")

        mock_turn = MagicMock()
        mock_turn.role = "user"
        mock_turn.content = "Hello"
        mock_turn.timestamp = "2026-01-01T00:00:00"
        mock_turn.token_estimate = 10

        mock_state = MagicMock()
        mock_state.total_turn_count = 5
        mock_state.turns = [mock_turn]
        mock_state.compressed_turn_count = 0
        mock_state.compressed_summary = ""
        mock_state.total_token_estimate = 100

        mock_conv = MagicMock()
        mock_conv.load.return_value = mock_state

        # Patch the import inside the endpoint
        with patch(
            "core.memory.conversation.ConversationMemory", return_value=mock_conv
        ):
            app = _make_test_app({"alice": alice})
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.get("/api/persons/alice/conversation")

        assert resp.status_code == 200
        data = resp.json()
        assert data["person"] == "alice"

    async def test_delete_conversation_person_not_found(self):
        app = _make_test_app({})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete("/api/persons/nobody/conversation")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Person not found: nobody"

    async def test_compress_conversation_person_not_found(self):
        app = _make_test_app({})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/persons/nobody/conversation/compress")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Person not found: nobody"
