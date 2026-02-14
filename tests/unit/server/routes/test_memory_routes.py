"""Unit tests for server/routes/memory_routes.py — Memory API endpoints."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


def _make_test_app(persons_dir: Path | None = None):
    from fastapi import FastAPI
    from server.routes.memory_routes import create_memory_router

    app = FastAPI()
    app.state.persons_dir = persons_dir or Path("/tmp/fake/persons")
    router = create_memory_router()
    app.include_router(router, prefix="/api")
    return app


# ── Episodes ─────────────────────────────────────────────


class TestEpisodes:
    async def test_list_episodes(self, tmp_path):
        persons_dir = tmp_path / "persons"
        person_dir = persons_dir / "alice"
        person_dir.mkdir(parents=True)
        episodes_dir = person_dir / "episodes"
        episodes_dir.mkdir()
        (episodes_dir / "2026-01-01.md").write_text("Today.", encoding="utf-8")

        with patch("server.routes.memory_routes.MemoryManager") as MockMM:
            mock_mm = MagicMock()
            mock_mm.list_episode_files.return_value = ["2026-01-01.md"]
            MockMM.return_value = mock_mm

            app = _make_test_app(persons_dir=persons_dir)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/persons/alice/episodes")
        assert resp.status_code == 200
        data = resp.json()
        assert data["files"] == ["2026-01-01.md"]

    async def test_list_episodes_not_found(self):
        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons/nobody/episodes")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Person not found: nobody"

    async def test_get_episode_success(self, tmp_path):
        persons_dir = tmp_path / "persons"
        person_dir = persons_dir / "alice"
        person_dir.mkdir(parents=True)
        episodes_dir = person_dir / "episodes"
        episodes_dir.mkdir()
        (episodes_dir / "2026-01-01.md").write_text("Today was good.", encoding="utf-8")

        with patch("server.routes.memory_routes.MemoryManager") as MockMM:
            mock_mm = MagicMock()
            mock_mm.episodes_dir = episodes_dir
            MockMM.return_value = mock_mm

            app = _make_test_app(persons_dir=persons_dir)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/persons/alice/episodes/2026-01-01")
        data = resp.json()
        assert data["date"] == "2026-01-01"
        assert data["content"] == "Today was good."

    async def test_get_episode_not_found(self, tmp_path):
        persons_dir = tmp_path / "persons"
        person_dir = persons_dir / "alice"
        person_dir.mkdir(parents=True)
        episodes_dir = person_dir / "episodes"
        episodes_dir.mkdir()

        with patch("server.routes.memory_routes.MemoryManager") as MockMM:
            mock_mm = MagicMock()
            mock_mm.episodes_dir = episodes_dir
            MockMM.return_value = mock_mm

            app = _make_test_app(persons_dir=persons_dir)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/persons/alice/episodes/9999-01-01")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Episode not found"


# ── Knowledge ────────────────────────────────────────────


class TestKnowledge:
    async def test_list_knowledge(self, tmp_path):
        persons_dir = tmp_path / "persons"
        person_dir = persons_dir / "alice"
        person_dir.mkdir(parents=True)

        with patch("server.routes.memory_routes.MemoryManager") as MockMM:
            mock_mm = MagicMock()
            mock_mm.list_knowledge_files.return_value = ["topic1.md"]
            MockMM.return_value = mock_mm

            app = _make_test_app(persons_dir=persons_dir)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/persons/alice/knowledge")
        assert resp.json()["files"] == ["topic1.md"]

    async def test_get_knowledge_success(self, tmp_path):
        persons_dir = tmp_path / "persons"
        person_dir = persons_dir / "alice"
        person_dir.mkdir(parents=True)
        knowledge_dir = person_dir / "knowledge"
        knowledge_dir.mkdir()
        (knowledge_dir / "python.md").write_text("Python is great.", encoding="utf-8")

        with patch("server.routes.memory_routes.MemoryManager") as MockMM:
            mock_mm = MagicMock()
            mock_mm.knowledge_dir = knowledge_dir
            MockMM.return_value = mock_mm

            app = _make_test_app(persons_dir=persons_dir)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/persons/alice/knowledge/python")
        data = resp.json()
        assert data["topic"] == "python"
        assert data["content"] == "Python is great."

    async def test_get_knowledge_not_found(self, tmp_path):
        persons_dir = tmp_path / "persons"
        person_dir = persons_dir / "alice"
        person_dir.mkdir(parents=True)
        knowledge_dir = person_dir / "knowledge"
        knowledge_dir.mkdir()

        with patch("server.routes.memory_routes.MemoryManager") as MockMM:
            mock_mm = MagicMock()
            mock_mm.knowledge_dir = knowledge_dir
            MockMM.return_value = mock_mm

            app = _make_test_app(persons_dir=persons_dir)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/persons/alice/knowledge/nonexistent")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Knowledge not found"


# ── Procedures ───────────────────────────────────────────


class TestProcedures:
    async def test_list_procedures(self, tmp_path):
        persons_dir = tmp_path / "persons"
        person_dir = persons_dir / "alice"
        person_dir.mkdir(parents=True)

        with patch("server.routes.memory_routes.MemoryManager") as MockMM:
            mock_mm = MagicMock()
            mock_mm.list_procedure_files.return_value = ["proc1.md"]
            MockMM.return_value = mock_mm

            app = _make_test_app(persons_dir=persons_dir)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/persons/alice/procedures")
        assert resp.json()["files"] == ["proc1.md"]

    async def test_get_procedure_success(self, tmp_path):
        persons_dir = tmp_path / "persons"
        person_dir = persons_dir / "alice"
        person_dir.mkdir(parents=True)
        proc_dir = person_dir / "procedures"
        proc_dir.mkdir()
        (proc_dir / "deploy.md").write_text("Step 1: ...", encoding="utf-8")

        with patch("server.routes.memory_routes.MemoryManager") as MockMM:
            mock_mm = MagicMock()
            mock_mm.procedures_dir = proc_dir
            MockMM.return_value = mock_mm

            app = _make_test_app(persons_dir=persons_dir)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/persons/alice/procedures/deploy")
        data = resp.json()
        assert data["name"] == "deploy"
        assert data["content"] == "Step 1: ..."

    async def test_get_procedure_not_found(self, tmp_path):
        persons_dir = tmp_path / "persons"
        person_dir = persons_dir / "alice"
        person_dir.mkdir(parents=True)
        proc_dir = person_dir / "procedures"
        proc_dir.mkdir()

        with patch("server.routes.memory_routes.MemoryManager") as MockMM:
            mock_mm = MagicMock()
            mock_mm.procedures_dir = proc_dir
            MockMM.return_value = mock_mm

            app = _make_test_app(persons_dir=persons_dir)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/persons/alice/procedures/nonexistent")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Procedure not found"


# ── Conversation ─────────────────────────────────────────


class TestConversation:
    async def test_get_conversation_person_not_found(self):
        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons/nobody/conversation")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Person not found: nobody"

    async def test_get_conversation_success(self, tmp_path):
        persons_dir = tmp_path / "persons"
        person_dir = persons_dir / "alice"
        person_dir.mkdir(parents=True)

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

        import core.config.models as config_mod

        with patch(
            "server.routes.memory_routes.ConversationMemory",
            return_value=mock_conv,
        ), patch.object(
            config_mod,
            "load_model_config",
            create=True,
            return_value=MagicMock(),
        ):
            app = _make_test_app(persons_dir=persons_dir)
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.get("/api/persons/alice/conversation")

        assert resp.status_code == 200
        data = resp.json()
        assert data["person"] == "alice"

    async def test_delete_conversation_person_not_found(self):
        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete("/api/persons/nobody/conversation")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Person not found: nobody"

    async def test_compress_conversation_person_not_found(self):
        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/persons/nobody/conversation/compress")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Person not found: nobody"


# ── Memory Stats ─────────────────────────────────────────


class TestMemoryStats:
    async def test_stats_person_not_found(self):
        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons/nobody/memory/stats")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Person not found: nobody"

    async def test_stats_with_files(self, tmp_path):
        persons_dir = tmp_path / "persons"
        person_dir = persons_dir / "alice"
        person_dir.mkdir(parents=True)

        # Create memory directories with .md files
        episodes_dir = person_dir / "episodes"
        episodes_dir.mkdir()
        (episodes_dir / "2026-01-01.md").write_text(
            "Episode 1 content", encoding="utf-8"
        )
        (episodes_dir / "2026-01-02.md").write_text(
            "Episode 2 content here", encoding="utf-8"
        )

        knowledge_dir = person_dir / "knowledge"
        knowledge_dir.mkdir()
        (knowledge_dir / "python.md").write_text(
            "Python knowledge", encoding="utf-8"
        )

        procedures_dir = person_dir / "procedures"
        procedures_dir.mkdir()

        with patch("server.routes.memory_routes.MemoryManager") as MockMM:
            mock_mm = MagicMock()
            mock_mm.episodes_dir = episodes_dir
            mock_mm.knowledge_dir = knowledge_dir
            mock_mm.procedures_dir = procedures_dir
            MockMM.return_value = mock_mm

            app = _make_test_app(persons_dir=persons_dir)
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.get("/api/persons/alice/memory/stats")

        assert resp.status_code == 200
        data = resp.json()
        assert data["person"] == "alice"
        assert data["episodes"]["count"] == 2
        assert data["episodes"]["total_bytes"] > 0
        assert data["knowledge"]["count"] == 1
        assert data["knowledge"]["total_bytes"] > 0
        assert data["procedures"]["count"] == 0
        assert data["procedures"]["total_bytes"] == 0

    async def test_stats_empty_directories(self, tmp_path):
        persons_dir = tmp_path / "persons"
        person_dir = persons_dir / "alice"
        person_dir.mkdir(parents=True)

        episodes_dir = person_dir / "episodes"
        episodes_dir.mkdir()
        knowledge_dir = person_dir / "knowledge"
        knowledge_dir.mkdir()
        procedures_dir = person_dir / "procedures"
        procedures_dir.mkdir()

        with patch("server.routes.memory_routes.MemoryManager") as MockMM:
            mock_mm = MagicMock()
            mock_mm.episodes_dir = episodes_dir
            mock_mm.knowledge_dir = knowledge_dir
            mock_mm.procedures_dir = procedures_dir
            MockMM.return_value = mock_mm

            app = _make_test_app(persons_dir=persons_dir)
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.get("/api/persons/alice/memory/stats")

        assert resp.status_code == 200
        data = resp.json()
        assert data["episodes"]["count"] == 0
        assert data["episodes"]["total_bytes"] == 0
        assert data["knowledge"]["count"] == 0
        assert data["knowledge"]["total_bytes"] == 0
        assert data["procedures"]["count"] == 0
        assert data["procedures"]["total_bytes"] == 0

    async def test_stats_nonexistent_memory_dirs(self, tmp_path):
        persons_dir = tmp_path / "persons"
        person_dir = persons_dir / "alice"
        person_dir.mkdir(parents=True)

        # Directories don't exist on disk
        with patch("server.routes.memory_routes.MemoryManager") as MockMM:
            mock_mm = MagicMock()
            mock_mm.episodes_dir = person_dir / "episodes"
            mock_mm.knowledge_dir = person_dir / "knowledge"
            mock_mm.procedures_dir = person_dir / "procedures"
            MockMM.return_value = mock_mm

            app = _make_test_app(persons_dir=persons_dir)
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.get("/api/persons/alice/memory/stats")

        assert resp.status_code == 200
        data = resp.json()
        assert data["episodes"]["count"] == 0
        assert data["episodes"]["total_bytes"] == 0
