# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for frontend log delivery through the full FastAPI stack.

Validates:
  1. POST /api/system/frontend-logs — logs arrive through create_app stack
  2. GET /api/system/frontend-logs — read back stored logs
  3. Round-trip: POST then GET returns consistent data
  4. setup_guard middleware does NOT block frontend-logs when setup is complete
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


def _create_app(tmp_path: Path):
    """Build a real FastAPI app via create_app with mocked externals.

    Uses setup_complete=True so the setup_guard middleware allows requests.
    """
    animas_dir = tmp_path / "animas"
    animas_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch("server.app.ProcessSupervisor") as mock_sup_cls,
        patch("server.app.load_config") as mock_cfg,
        patch("server.app.WebSocketManager") as mock_ws_cls,
        patch("server.app.load_auth") as mock_auth,
    ):
        cfg = MagicMock()
        cfg.setup_complete = True
        mock_cfg.return_value = cfg

        auth_cfg = MagicMock()
        auth_cfg.auth_mode = "local_trust"
        mock_auth.return_value = auth_cfg

        supervisor = MagicMock()
        supervisor.get_all_status.return_value = {}
        supervisor.is_scheduler_running.return_value = False
        supervisor.scheduler = None
        mock_sup_cls.return_value = supervisor

        ws_manager = MagicMock()
        ws_manager.active_connections = []
        mock_ws_cls.return_value = ws_manager

        from server.app import create_app

        app = create_app(animas_dir, shared_dir)

    # Persist auth mock beyond the with-block for request-time middleware
    import server.app as _sa
    _auth = MagicMock()
    _auth.auth_mode = "local_trust"
    _sa.load_auth = lambda: _auth

    return app


class TestFrontendLogDeliveryE2E:
    """Full-stack POST → file → GET round-trip tests."""

    @pytest.fixture(autouse=True)
    def _reset_frontend_logger(self):
        """Reset cached frontend logger between tests."""
        import server.routes.system as mod

        mod._frontend_logger = None
        mod._frontend_log_dir = None
        yield
        mod._frontend_logger = None
        mod._frontend_log_dir = None

    async def test_post_creates_log_file(self, tmp_path: Path) -> None:
        """POST should create frontend.jsonl in the data directory."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        with patch("core.paths.get_data_dir", return_value=data_dir):
            app = _create_app(tmp_path)
            transport = ASGITransport(app=app)
            entries = [
                {"ts": "2026-02-17T12:00:00Z", "level": "INFO", "module": "websocket", "msg": "Connected"},
            ]
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/system/frontend-logs",
                    content=json.dumps(entries),
                    headers={"Content-Type": "application/json"},
                )

        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        log_file = data_dir / "logs" / "frontend" / "frontend.jsonl"
        assert log_file.exists()

    async def test_round_trip_post_then_get(self, tmp_path: Path) -> None:
        """Posted entries should be retrievable via GET."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        with patch("core.paths.get_data_dir", return_value=data_dir):
            app = _create_app(tmp_path)
            transport = ASGITransport(app=app)
            entries = [
                {"ts": "2026-02-17T10:00:00Z", "level": "INFO", "module": "api", "msg": "request"},
                {"ts": "2026-02-17T10:00:01Z", "level": "ERROR", "module": "ws", "msg": "disconnect"},
            ]
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                post_resp = await client.post(
                    "/api/system/frontend-logs",
                    content=json.dumps(entries),
                    headers={"Content-Type": "application/json"},
                )
                assert post_resp.status_code == 200

                get_resp = await client.get("/api/system/frontend-logs")

        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["total"] == 2
        # Most recent first
        msgs = [e["msg"] for e in data["entries"]]
        assert "disconnect" in msgs
        assert "request" in msgs

    async def test_text_plain_through_full_stack(self, tmp_path: Path) -> None:
        """text/plain Content-Type (like sendBeacon) should work through the full stack."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        with patch("core.paths.get_data_dir", return_value=data_dir):
            app = _create_app(tmp_path)
            transport = ASGITransport(app=app)
            entries = [{"ts": "2026-02-17T12:00:00Z", "level": "WARN", "module": "chat", "msg": "slow"}]
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/system/frontend-logs",
                    content=json.dumps(entries),
                    headers={"Content-Type": "text/plain"},
                )

        assert resp.status_code == 200
        assert resp.json()["count"] == 1

    async def test_get_with_level_filter(self, tmp_path: Path) -> None:
        """GET with level filter should return only matching entries."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        with patch("core.paths.get_data_dir", return_value=data_dir):
            app = _create_app(tmp_path)
            transport = ASGITransport(app=app)
            entries = [
                {"ts": "2026-02-17T10:00:00Z", "level": "INFO", "module": "api", "msg": "ok"},
                {"ts": "2026-02-17T10:00:01Z", "level": "ERROR", "module": "api", "msg": "fail"},
                {"ts": "2026-02-17T10:00:02Z", "level": "INFO", "module": "ws", "msg": "connected"},
            ]
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                await client.post(
                    "/api/system/frontend-logs",
                    content=json.dumps(entries),
                    headers={"Content-Type": "application/json"},
                )
                get_resp = await client.get("/api/system/frontend-logs?level=ERROR")

        data = get_resp.json()
        assert data["total"] == 1
        assert data["entries"][0]["level"] == "ERROR"

    async def test_get_with_module_filter(self, tmp_path: Path) -> None:
        """GET with module filter should return only matching module entries."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        with patch("core.paths.get_data_dir", return_value=data_dir):
            app = _create_app(tmp_path)
            transport = ASGITransport(app=app)
            entries = [
                {"ts": "2026-02-17T10:00:00Z", "level": "INFO", "module": "api", "msg": "api log"},
                {"ts": "2026-02-17T10:00:01Z", "level": "INFO", "module": "websocket", "msg": "ws log"},
            ]
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                await client.post(
                    "/api/system/frontend-logs",
                    content=json.dumps(entries),
                    headers={"Content-Type": "application/json"},
                )
                get_resp = await client.get("/api/system/frontend-logs?module=websocket")

        data = get_resp.json()
        assert data["total"] == 1
        assert data["entries"][0]["module"] == "websocket"

    async def test_multiple_batches_accumulate(self, tmp_path: Path) -> None:
        """Multiple POST batches should accumulate in the same file."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        with patch("core.paths.get_data_dir", return_value=data_dir):
            app = _create_app(tmp_path)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                # Batch 1
                await client.post(
                    "/api/system/frontend-logs",
                    content=json.dumps([
                        {"ts": "2026-02-17T10:00:00Z", "level": "INFO", "module": "api", "msg": "batch1"},
                    ]),
                    headers={"Content-Type": "application/json"},
                )
                # Batch 2
                await client.post(
                    "/api/system/frontend-logs",
                    content=json.dumps([
                        {"ts": "2026-02-17T10:00:01Z", "level": "INFO", "module": "ws", "msg": "batch2"},
                    ]),
                    headers={"Content-Type": "application/json"},
                )

                get_resp = await client.get("/api/system/frontend-logs")

        data = get_resp.json()
        assert data["total"] == 2


class TestSetupGuardFrontendLogs:
    """Verify setup_guard middleware behaviour for frontend-logs."""

    async def test_blocked_when_setup_incomplete(self, tmp_path: Path) -> None:
        """Frontend logs should be blocked (503) when setup is not complete."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir(parents=True)
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir(parents=True)

        with (
            patch("server.app.ProcessSupervisor") as mock_sup_cls,
            patch("server.app.load_config") as mock_cfg,
            patch("server.app.WebSocketManager") as mock_ws_cls,
            patch("server.app.load_auth") as mock_auth,
        ):
            cfg = MagicMock()
            cfg.setup_complete = False  # Setup NOT complete
            mock_cfg.return_value = cfg

            auth_cfg = MagicMock()
            auth_cfg.auth_mode = "local_trust"
            mock_auth.return_value = auth_cfg

            mock_sup_cls.return_value = MagicMock()
            mock_ws_cls.return_value = MagicMock()

            from server.app import create_app

            app = create_app(animas_dir, shared_dir)

        # Persist auth mock beyond the with-block for request-time middleware
        import server.app as _sa
        _auth = MagicMock()
        _auth.auth_mode = "local_trust"
        _sa.load_auth = lambda: _auth

        transport = ASGITransport(app=app)
        entries = [{"ts": "2026-02-17T12:00:00Z", "level": "INFO", "module": "api", "msg": "test"}]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/system/frontend-logs",
                content=json.dumps(entries),
                headers={"Content-Type": "application/json"},
            )

        # setup_guard returns 503 for all /api/* when setup incomplete
        assert resp.status_code == 503
