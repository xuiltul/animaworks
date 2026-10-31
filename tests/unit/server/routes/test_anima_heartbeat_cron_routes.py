"""Unit tests for heartbeat/cron config endpoints in server/routes/animas.py."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from httpx import ASGITransport, AsyncClient

# ── Helper ───────────────────────────────────────────────


def _create_app(tmp_path: Path, anima_names: list[str] | None = None):
    """Build a minimal FastAPI app with the animas router and mocked supervisor."""
    from fastapi import FastAPI

    from server.routes.animas import create_animas_router

    app = FastAPI()
    app.state.animas_dir = tmp_path
    app.state.anima_names = anima_names or []

    supervisor = MagicMock()
    supervisor.processes = {}
    supervisor.get_process_status.return_value = {"status": "running", "pid": 1}
    app.state.supervisor = supervisor

    router = create_animas_router()
    app.include_router(router, prefix="/api")
    return app


# ── GET /api/animas/{name}/heartbeat ─────────────────────


class TestGetAnimaHeartbeat:
    """Tests for the GET /api/animas/{name}/heartbeat endpoint."""

    async def test_heartbeat_with_content(self, tmp_path: Path) -> None:
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        hb_content = "# Heartbeat: Alice\n## 活動時間\n9:00 - 22:00\n"
        (anima_dir / "HEARTBEAT.md").write_text(hb_content, encoding="utf-8")

        app = _create_app(tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/alice/heartbeat")

        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == hb_content

    async def test_heartbeat_file_missing(self, tmp_path: Path) -> None:
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()

        app = _create_app(tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/alice/heartbeat")

        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == ""

    async def test_heartbeat_anima_not_found(self, tmp_path: Path) -> None:
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/nonexistent/heartbeat")

        assert resp.status_code == 404


# ── GET /api/animas/{name}/cron ──────────────────────────


class TestGetAnimaCron:
    """Tests for the GET /api/animas/{name}/cron endpoint."""

    async def test_cron_with_content(self, tmp_path: Path) -> None:
        anima_dir = tmp_path / "bob"
        anima_dir.mkdir()
        cron_content = "## Morning Plan\nschedule: 0 9 * * *\ntype: llm\nPlan the day.\n"
        (anima_dir / "cron.md").write_text(cron_content, encoding="utf-8")

        app = _create_app(tmp_path, anima_names=["bob"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/bob/cron")

        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == cron_content

    async def test_cron_file_missing(self, tmp_path: Path) -> None:
        anima_dir = tmp_path / "bob"
        anima_dir.mkdir()

        app = _create_app(tmp_path, anima_names=["bob"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/bob/cron")

        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == ""

    async def test_cron_anima_not_found(self, tmp_path: Path) -> None:
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/nonexistent/cron")

        assert resp.status_code == 404
