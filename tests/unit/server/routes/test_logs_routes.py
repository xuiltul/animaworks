"""Unit tests for server/routes/logs_routes.py — Log viewing endpoints."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from server.routes.logs_routes import _validate_filename


# ── Helper ──────────────────────────────────────────────────


def _make_test_app(log_dirs: list[Path] | None = None):
    from fastapi import FastAPI
    from server.routes.logs_routes import create_logs_router

    app = FastAPI()
    router = create_logs_router()
    app.include_router(router, prefix="/api")

    # Patch _LOG_SEARCH_DIRS to point at test directories
    if log_dirs is not None:
        import server.routes.logs_routes as mod
        mod._LOG_SEARCH_DIRS = log_dirs

    return app


# ── _validate_filename ──────────────────────────────────────


class TestValidateFilename:
    def test_valid_filename_passes(self):
        # Should not raise
        _validate_filename("animaworks.log")
        _validate_filename("server-2026-01-01.log")

    def test_rejects_double_dot(self):
        with pytest.raises(HTTPException) as exc_info:
            _validate_filename("../etc/passwd")
        assert exc_info.value.status_code == 400

    def test_rejects_forward_slash(self):
        with pytest.raises(HTTPException) as exc_info:
            _validate_filename("path/to/file.log")
        assert exc_info.value.status_code == 400

    def test_rejects_backslash(self):
        with pytest.raises(HTTPException) as exc_info:
            _validate_filename("path\\to\\file.log")
        assert exc_info.value.status_code == 400

    def test_rejects_dot_prefix(self):
        with pytest.raises(HTTPException) as exc_info:
            _validate_filename(".hidden.log")
        assert exc_info.value.status_code == 400

    def test_rejects_just_dots(self):
        with pytest.raises(HTTPException) as exc_info:
            _validate_filename("..")
        assert exc_info.value.status_code == 400


# ── GET /system/logs ────────────────────────────────────────


class TestListLogs:
    async def test_empty_when_no_log_dir(self, tmp_path):
        # Point to a nonexistent directory
        log_dir = tmp_path / "logs"
        app = _make_test_app(log_dirs=[log_dir])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/logs")
        assert resp.status_code == 200
        assert resp.json()["files"] == []

    async def test_returns_log_files(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "app.log").write_text("line 1\nline 2\n", encoding="utf-8")
        (log_dir / "error.log").write_text("error\n", encoding="utf-8")
        # Non-.log files should be ignored
        (log_dir / "readme.txt").write_text("not a log", encoding="utf-8")

        app = _make_test_app(log_dirs=[log_dir])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/logs")
        data = resp.json()
        names = [f["name"] for f in data["files"]]
        assert "app.log" in names
        assert "error.log" in names
        assert "readme.txt" not in names

    async def test_log_file_metadata(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        content = "line 1\nline 2\nline 3\n"
        (log_dir / "test.log").write_text(content, encoding="utf-8")

        app = _make_test_app(log_dirs=[log_dir])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/logs")
        data = resp.json()
        assert len(data["files"]) == 1
        f = data["files"][0]
        assert f["name"] == "test.log"
        assert f["size_bytes"] > 0
        assert "modified" in f
        assert "path" in f


# ── GET /system/logs/{filename} ─────────────────────────────


class TestReadLog:
    async def test_404_for_nonexistent_file(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        app = _make_test_app(log_dirs=[log_dir])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/logs/nonexistent.log")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    async def test_400_for_path_traversal(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        app = _make_test_app(log_dirs=[log_dir])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/logs/..secret.log")
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Invalid filename"

    async def test_successful_read_with_pagination(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        # Create a log file with known lines
        lines = [f"Line {i}" for i in range(50)]
        (log_dir / "test.log").write_text(
            "\n".join(lines), encoding="utf-8"
        )

        app = _make_test_app(log_dirs=[log_dir])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/system/logs/test.log?offset=10&limit=5"
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "test.log"
        assert data["total_lines"] == 50
        assert data["offset"] == 10
        assert data["limit"] == 5
        assert len(data["lines"]) == 5
        assert data["lines"][0] == "Line 10"
        assert data["lines"][4] == "Line 14"

    async def test_default_pagination(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        lines = [f"L{i}" for i in range(10)]
        (log_dir / "small.log").write_text(
            "\n".join(lines), encoding="utf-8"
        )

        app = _make_test_app(log_dirs=[log_dir])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/logs/small.log")
        data = resp.json()
        assert data["offset"] == 0
        assert data["limit"] == 200
        assert len(data["lines"]) == 10

    async def test_read_offset_beyond_total(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "tiny.log").write_text("one line", encoding="utf-8")

        app = _make_test_app(log_dirs=[log_dir])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/logs/tiny.log?offset=999")
        data = resp.json()
        assert data["total_lines"] == 1
        assert data["lines"] == []


# ── GET /system/logs/stream ─────────────────────────────────


class TestStreamLogs:
    async def test_404_for_nonexistent_file(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        app = _make_test_app(log_dirs=[log_dir])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/system/logs/stream?file=nonexistent.log"
            )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    async def test_400_for_invalid_filename(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        app = _make_test_app(log_dirs=[log_dir])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/system/logs/stream?file=..%2Fetc%2Fpasswd"
            )
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Invalid filename"

    async def test_stream_valid_file_exists(self, tmp_path):
        """Verify that a valid log file doesn't trigger 400/404.

        The actual SSE streaming is an infinite generator, so we only
        test error paths at unit level.  Integration tests cover the
        full streaming behaviour.
        """
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "app.log").write_text("line\n", encoding="utf-8")

        # Confirm the file resolves (no 404/400) by checking _resolve_log_path
        from server.routes.logs_routes import _resolve_log_path

        import server.routes.logs_routes as mod
        original = mod._LOG_SEARCH_DIRS
        mod._LOG_SEARCH_DIRS = [log_dir]
        try:
            resolved = _resolve_log_path("app.log")
            assert resolved is not None
            assert resolved.name == "app.log"
        finally:
            mod._LOG_SEARCH_DIRS = original
