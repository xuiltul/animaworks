"""Unit tests for file I/O error handling in server/routes/memory_routes.py."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


def _make_test_app(animas_dir: Path | None = None):
    from fastapi import FastAPI
    from server.routes.memory_routes import create_memory_router

    app = FastAPI()
    app.state.animas_dir = animas_dir or Path("/tmp/fake/animas")
    router = create_memory_router()
    app.include_router(router, prefix="/api")
    return app


class TestMemoryRoutesIOError:
    async def test_episode_read_os_error_returns_500(self, tmp_path):
        animas_dir = tmp_path / "animas"
        anima_dir = animas_dir / "alice"
        anima_dir.mkdir(parents=True)
        episodes_dir = anima_dir / "episodes"
        episodes_dir.mkdir()
        ep_file = episodes_dir / "2026-01-01.md"
        ep_file.write_text("content", encoding="utf-8")

        with patch("server.routes.memory_routes.MemoryManager") as MockMM:
            mock_mm = MagicMock()
            mock_mm.episodes_dir = episodes_dir
            MockMM.return_value = mock_mm

            with patch.object(Path, "read_text", side_effect=OSError("Permission denied")):
                app = _make_test_app(animas_dir=animas_dir)
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.get("/api/animas/alice/episodes/2026-01-01")
        assert resp.status_code == 500

    async def test_episode_read_unicode_error_returns_500(self, tmp_path):
        animas_dir = tmp_path / "animas"
        anima_dir = animas_dir / "alice"
        anima_dir.mkdir(parents=True)
        episodes_dir = anima_dir / "episodes"
        episodes_dir.mkdir()
        ep_file = episodes_dir / "2026-01-01.md"
        ep_file.write_text("content", encoding="utf-8")

        with patch("server.routes.memory_routes.MemoryManager") as MockMM:
            mock_mm = MagicMock()
            mock_mm.episodes_dir = episodes_dir
            MockMM.return_value = mock_mm

            with patch.object(
                Path, "read_text",
                side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "bad byte"),
            ):
                app = _make_test_app(animas_dir=animas_dir)
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.get("/api/animas/alice/episodes/2026-01-01")
        assert resp.status_code == 500

    async def test_knowledge_read_os_error_returns_500(self, tmp_path):
        animas_dir = tmp_path / "animas"
        anima_dir = animas_dir / "alice"
        anima_dir.mkdir(parents=True)
        knowledge_dir = anima_dir / "knowledge"
        knowledge_dir.mkdir()
        k_file = knowledge_dir / "test.md"
        k_file.write_text("content", encoding="utf-8")

        with patch("server.routes.memory_routes.MemoryManager") as MockMM:
            mock_mm = MagicMock()
            mock_mm.knowledge_dir = knowledge_dir
            MockMM.return_value = mock_mm

            with patch.object(Path, "read_text", side_effect=OSError("Disk error")):
                app = _make_test_app(animas_dir=animas_dir)
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.get("/api/animas/alice/knowledge/test")
        assert resp.status_code == 500

    async def test_procedure_read_os_error_returns_500(self, tmp_path):
        animas_dir = tmp_path / "animas"
        anima_dir = animas_dir / "alice"
        anima_dir.mkdir(parents=True)
        procedures_dir = anima_dir / "procedures"
        procedures_dir.mkdir()
        p_file = procedures_dir / "test.md"
        p_file.write_text("content", encoding="utf-8")

        with patch("server.routes.memory_routes.MemoryManager") as MockMM:
            mock_mm = MagicMock()
            mock_mm.procedures_dir = procedures_dir
            MockMM.return_value = mock_mm

            with patch.object(Path, "read_text", side_effect=OSError("Disk error")):
                app = _make_test_app(animas_dir=animas_dir)
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.get("/api/animas/alice/procedures/test")
        assert resp.status_code == 500
