"""E2E tests for 3D model caching and asset delivery optimization."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.routes.assets import create_assets_router


@pytest.fixture
def e2e_app(tmp_path):
    """Create a full app with asset routes for E2E testing."""
    app = FastAPI()
    app.state.animas_dir = tmp_path

    anima_dir = tmp_path / "sakura"
    assets_dir = anima_dir / "assets"
    assets_dir.mkdir(parents=True)
    (anima_dir / "identity.md").write_text(
        "# サクラ\nイメージカラー: #FF69B4\n"
    )

    # Create realistic test assets
    (assets_dir / "avatar_chibi.glb").write_bytes(b"x" * 1024)
    (assets_dir / "avatar_chibi_rigged.glb").write_bytes(b"y" * 2048)
    (assets_dir / "anim_idle.glb").write_bytes(b"z" * 512)
    (assets_dir / "avatar_bustup.png").write_bytes(b"\x89PNG" + b"p" * 256)

    router = create_assets_router()
    app.include_router(router, prefix="/api")

    with patch("server.routes.assets.emit", new_callable=AsyncMock):
        yield app


@pytest.mark.e2e
class TestAssetCacheE2E:
    """E2E tests for the complete asset caching flow."""

    def test_full_cache_cycle_glb(self, e2e_app):
        """Test complete cache cycle: GET -> ETag -> conditional GET -> 304."""
        client = TestClient(e2e_app)

        # 1. Initial GET - should return 200 with cache headers
        resp1 = client.get("/api/animas/sakura/assets/avatar_chibi_rigged.glb")
        assert resp1.status_code == 200
        assert resp1.headers["content-type"] == "model/gltf-binary"
        assert "no-cache" in resp1.headers["cache-control"]
        etag = resp1.headers["etag"]

        # 2. Conditional GET with ETag - should return 304
        resp2 = client.get(
            "/api/animas/sakura/assets/avatar_chibi_rigged.glb",
            headers={"If-None-Match": etag},
        )
        assert resp2.status_code == 304
        assert len(resp2.content) == 0  # No body in 304

        # 3. Request different file - should get 200 with different ETag
        resp3 = client.get("/api/animas/sakura/assets/anim_idle.glb")
        assert resp3.status_code == 200
        assert resp3.headers["etag"] != etag

    def test_png_assets_also_cached(self, e2e_app):
        """PNG assets should also get cache headers."""
        client = TestClient(e2e_app)

        resp = client.get("/api/animas/sakura/assets/avatar_bustup.png")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert "no-cache" in resp.headers["cache-control"]
        assert "etag" in resp.headers

    def test_metadata_includes_animation_sizes(self, e2e_app):
        """Metadata endpoint should list all animation files with sizes."""
        client = TestClient(e2e_app)

        resp = client.get("/api/animas/sakura/assets/metadata")
        assert resp.status_code == 200
        data = resp.json()
        assert "idle" in data["animations"]
        assert data["animations"]["idle"]["size"] == 512

    def test_404_for_nonexistent_anima(self, e2e_app):
        """Should return 404 for nonexistent anima."""
        client = TestClient(e2e_app)
        resp = client.get("/api/animas/nonexistent/assets/avatar_chibi.glb")
        assert resp.status_code == 404

    def test_404_for_nonexistent_asset(self, e2e_app):
        """Should return 404 for nonexistent asset file."""
        client = TestClient(e2e_app)
        resp = client.get("/api/animas/sakura/assets/nonexistent.glb")
        assert resp.status_code == 404
