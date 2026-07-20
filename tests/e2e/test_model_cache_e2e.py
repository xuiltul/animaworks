"""E2E tests for 3D model caching and asset delivery optimization."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
        assert "max-age=" in resp1.headers["cache-control"]
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
        assert "max-age=" in resp.headers["cache-control"]
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


@pytest.mark.e2e
class TestAssetThumbnailE2E:
    """E2E: server-side avatar thumbnail via ?size= query."""

    @pytest.fixture
    def thumb_app(self, tmp_path):
        """App with a real large PNG icon for thumbnail generation."""
        from PIL import Image

        app = FastAPI()
        app.state.animas_dir = tmp_path
        anima_dir = tmp_path / "sakura"
        assets_dir = anima_dir / "assets"
        assets_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("# サクラ\n", encoding="utf-8")
        img = Image.new("RGB", (1024, 1024), color=(80, 160, 240))
        img.save(assets_dir / "icon.png", format="PNG")

        router = create_assets_router()
        app.include_router(router, prefix="/api")
        with patch("server.routes.assets.emit", new_callable=AsyncMock):
            yield app

    def test_size_s_webp_and_cache(self, thumb_app, tmp_path):
        client = TestClient(thumb_app)
        r1 = client.get("/api/animas/sakura/assets/icon.png?size=S")
        assert r1.status_code == 200
        assert "image/webp" in r1.headers["content-type"]
        assert "max-age=3600" in r1.headers["cache-control"]

        from io import BytesIO

        from PIL import Image

        thumb = Image.open(BytesIO(r1.content))
        assert thumb.size == (96, 96)

        cache = tmp_path / "sakura" / "assets" / ".thumbs" / "icon.png.S.webp"
        assert cache.is_file()
        mtime1 = cache.stat().st_mtime_ns

        r2 = client.get("/api/animas/sakura/assets/icon.png?size=S")
        assert r2.status_code == 200
        assert r2.content == r1.content
        assert cache.stat().st_mtime_ns == mtime1

    def test_invalid_size_serves_original_png(self, thumb_app):
        client = TestClient(thumb_app)
        resp = client.get("/api/animas/sakura/assets/icon.png?size=huge")
        assert resp.status_code == 200
        assert "image/png" in resp.headers["content-type"]
