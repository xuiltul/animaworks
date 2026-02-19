"""Tests for asset route HTTP caching (ETag, Cache-Control, 304)."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.routes.assets import create_assets_router


@pytest.fixture
def app_with_assets(tmp_path):
    """Create a FastAPI app with asset routes and a test anima."""
    app = FastAPI()
    app.state.animas_dir = tmp_path

    # Create test anima with a test asset
    anima_dir = tmp_path / "test-anima"
    assets_dir = anima_dir / "assets"
    assets_dir.mkdir(parents=True)

    # Create a small test GLB file
    test_glb = assets_dir / "avatar_chibi.glb"
    test_glb.write_bytes(b"fake-glb-data-for-testing")

    # Create identity.md
    (anima_dir / "identity.md").write_text("# Test Anima\n")

    router = create_assets_router()
    app.include_router(router, prefix="/api")

    # Patch the emit function to avoid event errors
    with patch("server.routes.assets.emit", new_callable=AsyncMock):
        yield app, test_glb


class TestAssetCacheHeaders:
    """Tests for Cache-Control and ETag headers on asset responses."""

    def test_returns_no_cache_control(self, app_with_assets):
        """Asset response should have no-cache Cache-Control (always revalidate via ETag)."""
        app, _ = app_with_assets
        client = TestClient(app)

        resp = client.get("/api/animas/test-anima/assets/avatar_chibi.glb")
        assert resp.status_code == 200
        assert "no-cache" in resp.headers["cache-control"]

    def test_returns_etag_header(self, app_with_assets):
        """Asset response should include an ETag header."""
        app, _ = app_with_assets
        client = TestClient(app)

        resp = client.get("/api/animas/test-anima/assets/avatar_chibi.glb")
        assert resp.status_code == 200
        assert "etag" in resp.headers
        etag = resp.headers["etag"]
        assert etag.startswith('"') and etag.endswith('"')

    def test_304_on_matching_etag(self, app_with_assets):
        """Should return 304 Not Modified when If-None-Match matches ETag."""
        app, _ = app_with_assets
        client = TestClient(app)

        # First request to get the ETag
        resp1 = client.get("/api/animas/test-anima/assets/avatar_chibi.glb")
        etag = resp1.headers["etag"]

        # Second request with matching If-None-Match
        resp2 = client.get(
            "/api/animas/test-anima/assets/avatar_chibi.glb",
            headers={"If-None-Match": etag},
        )
        assert resp2.status_code == 304
        assert resp2.headers["etag"] == etag

    def test_200_on_mismatched_etag(self, app_with_assets):
        """Should return 200 when If-None-Match does not match."""
        app, _ = app_with_assets
        client = TestClient(app)

        resp = client.get(
            "/api/animas/test-anima/assets/avatar_chibi.glb",
            headers={"If-None-Match": '"old-etag"'},
        )
        assert resp.status_code == 200

    def test_head_request_returns_etag(self, app_with_assets):
        """HEAD request should also return ETag header."""
        app, _ = app_with_assets
        client = TestClient(app)

        resp = client.head("/api/animas/test-anima/assets/avatar_chibi.glb")
        assert "etag" in resp.headers
        assert "no-cache" in resp.headers["cache-control"]

    def test_etag_changes_on_file_modification(self, app_with_assets):
        """ETag should change when the file is modified."""
        app, test_glb = app_with_assets
        client = TestClient(app)

        resp1 = client.get("/api/animas/test-anima/assets/avatar_chibi.glb")
        etag1 = resp1.headers["etag"]

        # Modify the file
        import time
        time.sleep(0.01)
        test_glb.write_bytes(b"modified-glb-data")

        resp2 = client.get("/api/animas/test-anima/assets/avatar_chibi.glb")
        etag2 = resp2.headers["etag"]

        assert etag1 != etag2
