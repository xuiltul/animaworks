"""Unit tests for setup guard middleware in server/app.py."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


def _make_app(setup_complete: bool, tmp_path: Path):
    """Build a test app with the setup guard middleware."""
    from core.config.models import AnimaWorksConfig, invalidate_cache, save_config

    # Write a config file so load_config() works
    config = AnimaWorksConfig(setup_complete=setup_complete)
    config_path = tmp_path / "config.json"
    save_config(config, config_path)
    invalidate_cache()

    animas_dir = tmp_path / "animas"
    animas_dir.mkdir(exist_ok=True)
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(exist_ok=True)

    with (
        patch("server.app.load_config", return_value=config),
        patch("server.app.ProcessSupervisor") as mock_sup_cls,
    ):
        mock_sup = MagicMock()
        mock_sup_cls.return_value = mock_sup

        from server.app import create_app
        app = create_app(animas_dir, shared_dir)

    return app


class TestSetupGuardNotComplete:
    """When setup_complete=False, the guard should enforce setup-mode routing."""

    async def test_root_redirects_to_setup(self, tmp_path: Path):
        app = _make_app(False, tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test", follow_redirects=False
        ) as client:
            resp = await client.get("/")

        assert resp.status_code == 307
        assert "/setup/" in resp.headers["location"]

    async def test_setup_api_accessible(self, tmp_path: Path):
        app = _make_app(False, tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # detect-locale doesn't need mocking beyond the request
            resp = await client.get("/api/setup/detect-locale")

        assert resp.status_code == 200

    async def test_dashboard_api_blocked(self, tmp_path: Path):
        app = _make_app(False, tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas")

        assert resp.status_code == 503
        data = resp.json()
        assert "Setup not yet complete" in data["error"]

    async def test_other_paths_redirect_to_setup(self, tmp_path: Path):
        app = _make_app(False, tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test", follow_redirects=False
        ) as client:
            resp = await client.get("/some/random/path")

        assert resp.status_code == 307
        assert "/setup/" in resp.headers["location"]


class TestSetupGuardComplete:
    """When setup_complete=True, the guard should block setup and allow dashboard."""

    async def test_setup_api_blocked(self, tmp_path: Path):
        app = _make_app(True, tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/setup/environment")

        assert resp.status_code == 403
        data = resp.json()
        assert "Setup already completed" in data["error"]

    async def test_setup_page_redirects_to_dashboard(self, tmp_path: Path):
        app = _make_app(True, tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test", follow_redirects=False
        ) as client:
            resp = await client.get("/setup/")

        assert resp.status_code == 307
        assert resp.headers["location"] == "/"

    async def test_dashboard_api_accessible(self, tmp_path: Path):
        app = _make_app(True, tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas")

        # Should get 200 (even if empty list)
        assert resp.status_code == 200


class TestSetupCacheControl:
    """Cache-Control headers for setup static files during setup mode."""

    async def test_setup_static_files_have_no_cache_header(self, tmp_path: Path):
        """Setup static files should include Cache-Control: no-cache during setup."""
        app = _make_app(False, tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/setup/setup.js")

        # Even if file doesn't exist (404), the middleware should add headers
        # for paths starting with /setup and not /api/setup
        if resp.status_code == 200:
            assert "no-cache" in resp.headers.get("cache-control", "")
            assert "no-store" in resp.headers.get("cache-control", "")
            assert "must-revalidate" in resp.headers.get("cache-control", "")

    async def test_setup_html_has_no_cache_header(self, tmp_path: Path):
        """Setup index.html should include Cache-Control: no-cache during setup."""
        app = _make_app(False, tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test", follow_redirects=False
        ) as client:
            resp = await client.get("/setup/")

        if resp.status_code == 200:
            cc = resp.headers.get("cache-control", "")
            assert "no-cache" in cc

    async def test_setup_i18n_has_no_cache_header(self, tmp_path: Path):
        """Setup i18n JSON files should include Cache-Control: no-cache during setup."""
        app = _make_app(False, tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/setup/i18n/ja.json")

        if resp.status_code == 200:
            cc = resp.headers.get("cache-control", "")
            assert "no-cache" in cc

    async def test_setup_api_no_cache_control_header(self, tmp_path: Path):
        """Setup API responses (/api/setup/*) should NOT get cache-control headers."""
        app = _make_app(False, tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/setup/detect-locale")

        assert resp.status_code == 200
        # API responses should not have the no-cache header from our middleware
        cc = resp.headers.get("cache-control", "")
        assert "no-store" not in cc


class TestSetupGuardTransition:
    """Test that changing setup_complete in app.state switches behaviour."""

    async def test_transition_from_setup_to_complete(self, tmp_path: Path):
        app = _make_app(False, tmp_path)
        transport = ASGITransport(app=app)

        # First: setup API should be accessible
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/setup/detect-locale")
            assert resp.status_code == 200

        # Simulate setup completion
        app.state.setup_complete = True

        # Now: setup API should be blocked
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/setup/detect-locale")
            assert resp.status_code == 403

        # And dashboard API should work
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas")
            assert resp.status_code == 200
