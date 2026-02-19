"""Tests for RequestLoggingMiddleware and frontend log endpoints."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.app import RequestLoggingMiddleware


# ── RequestLoggingMiddleware ──────────────────────────────


class TestRequestLoggingMiddleware:
    @pytest.fixture()
    def app(self):
        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)

        @app.get("/api/test")
        async def test_endpoint():
            return {"ok": True}

        @app.get("/api/system/health")
        async def health():
            return {"status": "ok"}

        return app

    @pytest.fixture()
    def client(self, app):
        return TestClient(app)

    def test_adds_request_id_header(self, client):
        response = client.get("/api/test")
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) > 0

    def test_respects_existing_request_id(self, client):
        response = client.get(
            "/api/test",
            headers={"X-Request-ID": "custom-id-123"},
        )
        assert response.headers["X-Request-ID"] == "custom-id-123"

    def test_noisy_path_not_logged(self, client, caplog):
        """Health check endpoint should not produce request log."""
        with caplog.at_level(logging.INFO, logger="animaworks.request"):
            client.get("/api/system/health")
        # The health endpoint is in _NOISY_PATHS
        assert not any("system/health" in rec.message for rec in caplog.records)

    def test_normal_path_logged(self, client, caplog):
        """Normal endpoints should produce request log."""
        with caplog.at_level(logging.INFO, logger="animaworks.request"):
            client.get("/api/test")
        assert any("/api/test" in rec.message for rec in caplog.records)


# ── Frontend Log Endpoints ───────────────────────────────


class TestFrontendLogEndpoints:
    @pytest.fixture()
    def app(self, tmp_path):
        """Create a minimal FastAPI app with system routes for testing."""
        from server.routes.system import create_system_router, _get_frontend_logger
        import server.routes.system as sys_module

        # Reset the frontend logger so it gets re-created with tmp_path
        sys_module._frontend_logger = None
        sys_module._frontend_log_dir = None

        app = FastAPI()
        app.state.anima_names = []
        app.state.animas_dir = tmp_path / "animas"
        app.state.shared_dir = tmp_path / "shared"

        # Mock supervisor
        class FakeSupervisor:
            scheduler = None
            def is_scheduler_running(self):
                return False
            def get_all_status(self):
                return {}

        app.state.supervisor = FakeSupervisor()

        router = create_system_router()
        app.include_router(router, prefix="/api")

        # Patch get_data_dir to use tmp_path
        with patch("core.paths.get_data_dir", return_value=tmp_path):
            yield app

        # Cleanup
        sys_module._frontend_logger = None
        sys_module._frontend_log_dir = None

    @pytest.fixture()
    def client(self, app):
        return TestClient(app)

    def test_receive_frontend_logs(self, client, tmp_path):
        """POST /api/system/frontend-logs should accept a JSON array."""
        entries = [
            {"ts": "2026-02-17T10:00:00Z", "level": "ERROR", "module": "websocket", "msg": "Connection lost"},
            {"ts": "2026-02-17T10:00:01Z", "level": "INFO", "module": "chat", "msg": "Stream started"},
        ]
        with patch("core.paths.get_data_dir", return_value=tmp_path):
            response = client.post("/api/system/frontend-logs", json=entries)

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["count"] == 2

    def test_receive_invalid_json(self, client):
        """POST with non-array should return 400."""
        response = client.post(
            "/api/system/frontend-logs",
            content='{"not": "an array"}',
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400

    def test_view_frontend_logs_empty(self, client, tmp_path):
        """GET should return empty when no logs exist."""
        with patch("core.paths.get_data_dir", return_value=tmp_path):
            response = client.get("/api/system/frontend-logs?date=19700101")
        assert response.status_code == 200
        data = response.json()
        assert data["entries"] == []
        assert data["total"] == 0

    def test_view_frontend_logs_with_filter(self, client, tmp_path):
        """GET with level filter should filter entries."""
        from datetime import datetime

        log_dir = tmp_path / "logs" / "frontend"
        log_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / "frontend.jsonl"
        log_file.write_text(
            '{"level":"ERROR","module":"ws","msg":"err1"}\n'
            '{"level":"INFO","module":"ws","msg":"info1"}\n'
            '{"level":"ERROR","module":"chat","msg":"err2"}\n',
            encoding="utf-8",
        )

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            response = client.get(f"/api/system/frontend-logs?date={today}&level=ERROR")

        data = response.json()
        assert data["total"] == 2
        assert all(e["level"] == "ERROR" for e in data["entries"])

    def test_view_frontend_logs_with_module_filter(self, client, tmp_path):
        """GET with module filter should filter entries."""
        from datetime import datetime

        log_dir = tmp_path / "logs" / "frontend"
        log_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / "frontend.jsonl"
        log_file.write_text(
            '{"level":"ERROR","module":"websocket","msg":"err1"}\n'
            '{"level":"INFO","module":"chat","msg":"info1"}\n',
            encoding="utf-8",
        )

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            response = client.get(f"/api/system/frontend-logs?date={today}&module=websocket")

        data = response.json()
        assert data["total"] == 1
        assert data["entries"][0]["module"] == "websocket"


# ── Dynamic Log Level ────────────────────────────────────


class TestLogLevelAPI:
    @pytest.fixture()
    def app(self, tmp_path):
        from server.routes.system import create_system_router

        app = FastAPI()
        app.state.anima_names = []
        app.state.animas_dir = tmp_path / "animas"
        app.state.shared_dir = tmp_path / "shared"

        class FakeSupervisor:
            scheduler = None
            def is_scheduler_running(self):
                return False
            def get_all_status(self):
                return {}

        app.state.supervisor = FakeSupervisor()

        router = create_system_router()
        app.include_router(router, prefix="/api")
        return app

    @pytest.fixture()
    def client(self, app):
        return TestClient(app)

    def test_get_log_level(self, client):
        response = client.get("/api/system/log-level")
        assert response.status_code == 200
        data = response.json()
        assert "level" in data

    def test_set_root_log_level(self, client):
        response = client.post(
            "/api/system/log-level",
            json={"level": "DEBUG"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["level"] == "DEBUG"
        assert data["logger"] == "root"

        # Verify it actually changed
        root = logging.getLogger()
        assert root.level == logging.DEBUG

        # Reset
        root.setLevel(logging.WARNING)

    def test_set_named_logger_level(self, client):
        response = client.post(
            "/api/system/log-level",
            json={"level": "DEBUG", "logger_name": "animaworks.test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["level"] == "DEBUG"
        assert data["logger"] == "animaworks.test"

    def test_invalid_level_rejected(self, client):
        response = client.post(
            "/api/system/log-level",
            json={"level": "INVALID"},
        )
        assert response.status_code == 400
