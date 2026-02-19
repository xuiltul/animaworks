# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for logger.js crypto.randomUUID fallback.

Tests the full application stack to verify that logger.js is correctly
served via the static file mount and contains the required fallback
for non-secure contexts (HTTP + LAN IP).

The crypto.randomUUID API requires a secure context (HTTPS or localhost).
Without a fallback, all frontend modules crash on HTTP + LAN IP access
because every log call triggers _getSessionId() → crypto.randomUUID().
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


# ── Helpers ──────────────────────────────────────────────────────


def _create_app_with_static(tmp_path: Path):
    """Create a FastAPI app with full static file mounting.

    Uses create_app() to include the StaticFiles mount so we can
    fetch logger.js through the actual serving pipeline.
    """
    animas_dir = tmp_path / "animas"
    animas_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch("server.app.ProcessSupervisor") as mock_sup_cls,
        patch("server.app.load_config") as mock_cfg,
        patch("server.app.WebSocketManager") as mock_ws_cls,
    ):
        cfg = MagicMock()
        cfg.setup_complete = True
        mock_cfg.return_value = cfg

        supervisor = MagicMock()
        supervisor.get_all_status.return_value = {}
        supervisor.is_scheduler_running.return_value = False
        supervisor.scheduler = None
        mock_sup_cls.return_value = supervisor

        ws_manager = MagicMock()
        ws_manager.active_connections = []
        mock_ws_cls.return_value = ws_manager

        from server.app import create_app

        app = create_app(
            animas_dir=animas_dir,
            shared_dir=shared_dir,
        )
    return app


# ── Tests ────────────────────────────────────────────────────────


class TestLoggerJsServedCorrectly:
    """Verify logger.js is accessible and contains the fallback."""

    @pytest.fixture
    def app(self, tmp_path: Path):
        return _create_app_with_static(tmp_path)

    @pytest.mark.asyncio
    async def test_logger_js_is_served(self, app) -> None:
        """GET /shared/logger.js returns 200 with JavaScript content."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/shared/logger.js")
        assert resp.status_code == 200
        assert "javascript" in resp.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_logger_js_has_randomuuid_guard(self, app) -> None:
        """The served logger.js must check typeof crypto.randomUUID."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/shared/logger.js")
        body = resp.text
        assert "typeof crypto.randomUUID === 'function'" in body, (
            "logger.js must feature-detect crypto.randomUUID"
        )

    @pytest.mark.asyncio
    async def test_logger_js_has_getRandomValues_fallback(self, app) -> None:
        """The served logger.js must include crypto.getRandomValues fallback."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/shared/logger.js")
        body = resp.text
        assert "crypto.getRandomValues" in body, (
            "logger.js must fall back to crypto.getRandomValues"
        )

    @pytest.mark.asyncio
    async def test_logger_js_no_unguarded_randomuuid(self, app) -> None:
        """The served logger.js must not call crypto.randomUUID without a guard."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/shared/logger.js")
        body = resp.text

        # Count occurrences of crypto.randomUUID
        # Should appear in: typeof check + guarded call = at most 2 in the
        # code, plus once in the comment
        lines = body.splitlines()
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("*"):
                continue
            if "typeof crypto.randomUUID" in stripped:
                continue
            if "crypto.randomUUID()" in stripped:
                # Must be indented (inside conditional block)
                assert line.startswith("      ") or line.startswith("\t"), (
                    f"Unguarded crypto.randomUUID() call found: {stripped}"
                )
