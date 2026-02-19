# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for frontend log ingestion and viewer endpoints.

Covers:
  - POST /api/system/frontend-logs — receiving log entries
  - GET /api/system/frontend-logs — reading log entries with filters
  - _get_frontend_logger — TimedRotatingFileHandler setup
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


def _make_test_app(tmp_path: Path):
    """Build a minimal FastAPI app with the system router."""
    from fastapi import FastAPI

    from server.routes.system import create_system_router

    app = FastAPI()
    app.state.animas_dir = tmp_path / "animas"
    app.state.shared_dir = tmp_path / "shared"
    app.state.anima_names = []

    supervisor = MagicMock()
    supervisor.get_all_status.return_value = {}
    supervisor.is_scheduler_running.return_value = False
    supervisor.scheduler = None
    app.state.supervisor = supervisor

    ws_manager = MagicMock()
    ws_manager.active_connections = []
    app.state.ws_manager = ws_manager

    router = create_system_router()
    app.include_router(router, prefix="/api")
    return app


# ── POST /api/system/frontend-logs ────────────────────────


class TestReceiveFrontendLogs:
    """POST endpoint: receive and persist frontend log entries."""

    async def test_valid_json_array(self, tmp_path: Path) -> None:
        """Valid JSON array should be accepted and stored."""
        import server.routes.system as mod

        mod._frontend_logger = None
        mod._frontend_log_dir = None
        log_dir = tmp_path / "logs" / "frontend"
        with patch("core.paths.get_data_dir", return_value=tmp_path):
            app = _make_test_app(tmp_path)
            transport = ASGITransport(app=app)
            entries = [
                {"ts": "2026-02-17T12:00:00Z", "level": "INFO", "module": "api", "msg": "test"},
            ]
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/system/frontend-logs",
                    content=json.dumps(entries),
                    headers={"Content-Type": "application/json"},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["count"] == 1

        # Verify file was created with fixed base name
        log_file = log_dir / "frontend.jsonl"
        assert log_file.exists()
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        stored = json.loads(lines[0])
        assert stored["module"] == "api"
        assert stored["msg"] == "test"

        mod._frontend_logger = None
        mod._frontend_log_dir = None

    async def test_text_plain_content_type(self, tmp_path: Path) -> None:
        """Body sent as text/plain (e.g. from sendBeacon) should still parse."""
        import server.routes.system as mod

        mod._frontend_logger = None
        mod._frontend_log_dir = None
        with patch("core.paths.get_data_dir", return_value=tmp_path):
            app = _make_test_app(tmp_path)
            transport = ASGITransport(app=app)
            entries = [{"ts": "2026-02-17T12:00:00Z", "level": "WARN", "module": "ws", "msg": "reconnect"}]
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/system/frontend-logs",
                    content=json.dumps(entries),
                    headers={"Content-Type": "text/plain"},
                )

        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        mod._frontend_logger = None
        mod._frontend_log_dir = None

    async def test_invalid_json_returns_400(self, tmp_path: Path) -> None:
        """Non-JSON body should return 400."""
        app = _make_test_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/system/frontend-logs",
                content="not json{{{",
                headers={"Content-Type": "application/json"},
            )

        assert resp.status_code == 400
        assert "Invalid JSON" in resp.json()["error"]

    async def test_non_array_returns_400(self, tmp_path: Path) -> None:
        """JSON object (not array) should return 400."""
        app = _make_test_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/system/frontend-logs",
                content=json.dumps({"level": "INFO"}),
                headers={"Content-Type": "application/json"},
            )

        assert resp.status_code == 400
        assert "Expected a JSON array" in resp.json()["error"]

    async def test_too_many_entries_returns_400(self, tmp_path: Path) -> None:
        """More than 500 entries should be rejected."""
        app = _make_test_app(tmp_path)
        transport = ASGITransport(app=app)
        entries = [{"level": "INFO", "msg": f"e{i}"} for i in range(501)]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/system/frontend-logs",
                content=json.dumps(entries),
                headers={"Content-Type": "application/json"},
            )

        assert resp.status_code == 400
        assert "Too many entries" in resp.json()["error"]

    async def test_multiple_entries_stored(self, tmp_path: Path) -> None:
        """Multiple entries in one batch should all be stored."""
        import server.routes.system as mod

        mod._frontend_logger = None
        mod._frontend_log_dir = None
        with patch("core.paths.get_data_dir", return_value=tmp_path):
            app = _make_test_app(tmp_path)
            transport = ASGITransport(app=app)
            entries = [
                {"ts": "2026-02-17T12:00:00Z", "level": "INFO", "module": "api", "msg": "one"},
                {"ts": "2026-02-17T12:00:01Z", "level": "ERROR", "module": "ws", "msg": "two"},
                {"ts": "2026-02-17T12:00:02Z", "level": "WARN", "module": "chat", "msg": "three"},
            ]
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/system/frontend-logs",
                    content=json.dumps(entries),
                    headers={"Content-Type": "application/json"},
                )

        assert resp.status_code == 200
        assert resp.json()["count"] == 3

        log_file = tmp_path / "logs" / "frontend" / "frontend.jsonl"
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3

        mod._frontend_logger = None
        mod._frontend_log_dir = None

    async def test_non_dict_entries_skipped(self, tmp_path: Path) -> None:
        """Non-dict entries in the array should be silently skipped."""
        import server.routes.system as mod

        mod._frontend_logger = None
        mod._frontend_log_dir = None
        with patch("core.paths.get_data_dir", return_value=tmp_path):
            app = _make_test_app(tmp_path)
            transport = ASGITransport(app=app)
            entries = [
                {"ts": "2026-02-17T12:00:00Z", "level": "INFO", "module": "api", "msg": "valid"},
                "string entry",
                42,
                None,
            ]
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/system/frontend-logs",
                    content=json.dumps(entries),
                    headers={"Content-Type": "application/json"},
                )

        assert resp.status_code == 200
        # count includes all entries, not just dicts
        assert resp.json()["count"] == 4

        log_file = tmp_path / "logs" / "frontend" / "frontend.jsonl"
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        # Only the dict entry should be stored
        assert len(lines) == 1

        mod._frontend_logger = None
        mod._frontend_log_dir = None


# ── GET /api/system/frontend-logs ─────────────────────────


class TestViewFrontendLogs:
    """GET endpoint: read and filter stored frontend logs."""

    def _write_log_entries(
        self, log_dir: Path, entries: list[dict], filename: str = "frontend.jsonl",
    ) -> None:
        """Write JSONL entries to the specified log file."""
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / filename
        lines = [json.dumps(e, ensure_ascii=False) for e in entries]
        log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    async def test_today_reads_active_file(self, tmp_path: Path) -> None:
        """Requesting today's logs should read the active frontend.jsonl file."""
        log_dir = tmp_path / "logs" / "frontend"
        self._write_log_entries(log_dir, [
            {"ts": "2026-02-17T12:00:00Z", "level": "INFO", "module": "api", "msg": "hello"},
        ])

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            app = _make_test_app(tmp_path)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/system/frontend-logs")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["entries"][0]["msg"] == "hello"

    async def test_past_date_reads_rotated_file(self, tmp_path: Path) -> None:
        """Requesting a past date should read frontend.jsonl.YYYYMMDD."""
        log_dir = tmp_path / "logs" / "frontend"
        self._write_log_entries(
            log_dir,
            [{"ts": "2026-02-16T12:00:00Z", "level": "WARN", "module": "ws", "msg": "old"}],
            filename="frontend.jsonl.20260216",
        )

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            app = _make_test_app(tmp_path)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/system/frontend-logs?date=20260216")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["entries"][0]["msg"] == "old"
        assert data["date"] == "20260216"

    async def test_legacy_date_filename_fallback(self, tmp_path: Path) -> None:
        """Should fall back to legacy YYYYMMDD.jsonl format."""
        log_dir = tmp_path / "logs" / "frontend"
        self._write_log_entries(
            log_dir,
            [{"ts": "2026-02-15T12:00:00Z", "level": "INFO", "module": "api", "msg": "legacy"}],
            filename="20260215.jsonl",
        )

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            app = _make_test_app(tmp_path)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/system/frontend-logs?date=20260215")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["entries"][0]["msg"] == "legacy"

    async def test_no_file_returns_empty(self, tmp_path: Path) -> None:
        """Non-existent date should return empty results."""
        with patch("core.paths.get_data_dir", return_value=tmp_path):
            app = _make_test_app(tmp_path)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/system/frontend-logs?date=20260101")

        assert resp.status_code == 200
        data = resp.json()
        assert data["entries"] == []
        assert data["total"] == 0

    async def test_filter_by_level(self, tmp_path: Path) -> None:
        """Level filter should only return matching entries."""
        log_dir = tmp_path / "logs" / "frontend"
        self._write_log_entries(log_dir, [
            {"ts": "2026-02-17T12:00:00Z", "level": "INFO", "module": "api", "msg": "info msg"},
            {"ts": "2026-02-17T12:00:01Z", "level": "ERROR", "module": "api", "msg": "error msg"},
            {"ts": "2026-02-17T12:00:02Z", "level": "INFO", "module": "ws", "msg": "info2"},
        ])

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            app = _make_test_app(tmp_path)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/system/frontend-logs?level=ERROR")

        data = resp.json()
        assert data["total"] == 1
        assert data["entries"][0]["msg"] == "error msg"

    async def test_filter_by_module(self, tmp_path: Path) -> None:
        """Module filter should only return matching entries."""
        log_dir = tmp_path / "logs" / "frontend"
        self._write_log_entries(log_dir, [
            {"ts": "2026-02-17T12:00:00Z", "level": "INFO", "module": "api", "msg": "api msg"},
            {"ts": "2026-02-17T12:00:01Z", "level": "INFO", "module": "ws", "msg": "ws msg"},
        ])

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            app = _make_test_app(tmp_path)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/system/frontend-logs?module=ws")

        data = resp.json()
        assert data["total"] == 1
        assert data["entries"][0]["module"] == "ws"

    async def test_limit_parameter(self, tmp_path: Path) -> None:
        """Limit parameter should cap the number of returned entries."""
        log_dir = tmp_path / "logs" / "frontend"
        entries = [
            {"ts": f"2026-02-17T12:00:{i:02d}Z", "level": "INFO", "module": "api", "msg": f"msg{i}"}
            for i in range(10)
        ]
        self._write_log_entries(log_dir, entries)

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            app = _make_test_app(tmp_path)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/system/frontend-logs?limit=3")

        data = resp.json()
        assert data["total"] == 10
        assert len(data["entries"]) == 3


# ── _get_frontend_logger setup ────────────────────────────


class TestFrontendLoggerSetup:
    """Verify _get_frontend_logger creates correct handler config."""

    def test_uses_fixed_base_filename(self, tmp_path: Path) -> None:
        """Logger should write to frontend.jsonl, not a date-encoded filename."""
        import server.routes.system as mod

        # Reset module-level cached logger
        mod._frontend_logger = None
        mod._frontend_log_dir = None

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            fe_logger = mod._get_frontend_logger()

        # Check handler filename
        handlers = fe_logger.handlers
        assert len(handlers) >= 1
        handler = handlers[-1]
        assert handler.baseFilename.endswith("frontend.jsonl")
        assert handler.suffix == "%Y%m%d"

        # Clean up cached logger for other tests
        mod._frontend_logger = None
        mod._frontend_log_dir = None

    def test_creates_directory(self, tmp_path: Path) -> None:
        """Logger init should create the logs/frontend/ directory."""
        import server.routes.system as mod

        mod._frontend_logger = None
        mod._frontend_log_dir = None

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            mod._get_frontend_logger()

        assert (tmp_path / "logs" / "frontend").is_dir()

        mod._frontend_logger = None
        mod._frontend_log_dir = None
