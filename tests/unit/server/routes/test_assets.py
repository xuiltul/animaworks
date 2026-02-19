"""Unit tests for server/routes/assets.py — Asset serving endpoints."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


def _make_test_app(animas_dir: Path | None = None):
    from fastapi import FastAPI
    from server.routes.assets import create_assets_router

    app = FastAPI()
    app.state.animas_dir = animas_dir or Path("/tmp/fake/animas")
    app.state.ws_manager = MagicMock()
    app.state.ws_manager.broadcast = AsyncMock()
    router = create_assets_router()
    app.include_router(router, prefix="/api")
    return app


# ── GET /animas/{name}/assets ───────────────────────────


class TestListAssets:
    async def test_anima_not_found(self, tmp_path):
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/nobody/assets")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Anima not found: nobody"

    async def test_no_assets_dir(self, tmp_path):
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/alice/assets")
        assert resp.status_code == 200
        assert resp.json()["assets"] == []

    async def test_list_assets(self, tmp_path):
        animas_dir = tmp_path / "animas"
        anima_dir = animas_dir / "alice"
        anima_dir.mkdir(parents=True)
        assets_dir = anima_dir / "assets"
        assets_dir.mkdir()
        (assets_dir / "avatar.png").write_bytes(b"\x89PNG")
        (assets_dir / "model.glb").write_bytes(b"\x00")

        app = _make_test_app(animas_dir=animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/alice/assets")
        data = resp.json()
        assert len(data["assets"]) == 2
        names = [a["name"] for a in data["assets"]]
        assert "avatar.png" in names
        assert "model.glb" in names


# ── GET /animas/{name}/assets/metadata ──────────────────


class TestGetAssetMetadata:
    async def test_anima_not_found(self, tmp_path):
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/nobody/assets/metadata")
        assert resp.status_code == 404

    async def test_metadata_no_assets(self, tmp_path):
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        (anima_dir / "identity.md").write_text("# Alice", encoding="utf-8")

        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/alice/assets/metadata")
        data = resp.json()
        assert data["name"] == "alice"
        assert data["assets"] == {}

    async def test_metadata_with_color(self, tmp_path):
        animas_dir = tmp_path / "animas"
        anima_dir = animas_dir / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text(
            "# Alice\nイメージカラー: ピンク #FF69B4\n", encoding="utf-8"
        )
        (anima_dir / "assets").mkdir()

        app = _make_test_app(animas_dir=animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/alice/assets/metadata")
        data = resp.json()
        assert data["colors"] == {"image_color": "#FF69B4"}

    async def test_metadata_with_assets_and_animations(self, tmp_path):
        animas_dir = tmp_path / "animas"
        anima_dir = animas_dir / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("# Alice", encoding="utf-8")
        assets_dir = anima_dir / "assets"
        assets_dir.mkdir()
        (assets_dir / "avatar_fullbody.png").write_bytes(b"\x89PNG")
        (assets_dir / "avatar_chibi.glb").write_bytes(b"\x00")
        (assets_dir / "anim_idle.glb").write_bytes(b"\x00")
        (assets_dir / "anim_walk.glb").write_bytes(b"\x00")

        app = _make_test_app(animas_dir=animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/alice/assets/metadata")
        data = resp.json()
        assert "avatar_fullbody" in data["assets"]
        assert "model_chibi" in data["assets"]
        assert "idle" in data["animations"]
        assert "walk" in data["animations"]


# ── GET/HEAD /animas/{name}/assets/{filename} ───────────


class TestGetAsset:
    async def test_anima_not_found(self, tmp_path):
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/nobody/assets/file.png")
        assert resp.status_code == 404

    async def test_invalid_filename(self, tmp_path):
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/alice/assets/..%2Fetc%2Fpasswd")
        assert resp.status_code in (400, 404)

    async def test_asset_not_found(self, tmp_path):
        animas_dir = tmp_path / "animas"
        anima_dir = animas_dir / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "assets").mkdir()
        app = _make_test_app(animas_dir=animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/alice/assets/missing.png")
        assert resp.status_code == 404

    async def test_serve_png(self, tmp_path):
        animas_dir = tmp_path / "animas"
        anima_dir = animas_dir / "alice"
        anima_dir.mkdir(parents=True)
        assets_dir = anima_dir / "assets"
        assets_dir.mkdir()
        (assets_dir / "avatar.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        app = _make_test_app(animas_dir=animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/alice/assets/avatar.png")
        assert resp.status_code == 200
        assert "image/png" in resp.headers.get("content-type", "")

    async def test_serve_glb(self, tmp_path):
        animas_dir = tmp_path / "animas"
        anima_dir = animas_dir / "alice"
        anima_dir.mkdir(parents=True)
        assets_dir = anima_dir / "assets"
        assets_dir.mkdir()
        (assets_dir / "model.glb").write_bytes(b"\x00\x00\x00\x00")

        app = _make_test_app(animas_dir=animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/alice/assets/model.glb")
        assert resp.status_code == 200
        assert "model/gltf-binary" in resp.headers.get("content-type", "")

    async def test_head_request(self, tmp_path):
        animas_dir = tmp_path / "animas"
        anima_dir = animas_dir / "alice"
        anima_dir.mkdir(parents=True)
        assets_dir = anima_dir / "assets"
        assets_dir.mkdir()
        (assets_dir / "avatar.png").write_bytes(b"\x89PNG")

        app = _make_test_app(animas_dir=animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.head("/api/animas/alice/assets/avatar.png")
        assert resp.status_code == 200


# ── POST /animas/{name}/assets/generate ─────────────────


class TestGenerateAssets:
    async def test_anima_not_found(self, tmp_path):
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/nobody/assets/generate",
                json={"prompt": "a character"},
            )
        assert resp.status_code == 404

    async def test_missing_prompt(self, tmp_path):
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/assets/generate",
                json={},
            )
        assert resp.status_code == 400
        assert "prompt is required" in resp.json()["detail"]

    @patch("core.tools.image_gen.ImageGenPipeline")
    async def test_generate_success(self, mock_pipeline_cls, tmp_path):
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()

        mock_result = MagicMock()
        mock_result.fullbody_path = Path("/tmp/fb.png")
        mock_result.bustup_path = None
        mock_result.chibi_path = None
        mock_result.model_path = None
        mock_result.rigged_model_path = None
        mock_result.animation_paths = {}
        mock_result.errors = []
        mock_result.to_dict.return_value = {"status": "done"}

        mock_pipeline = MagicMock()
        mock_pipeline.generate_all.return_value = mock_result
        mock_pipeline_cls.return_value = mock_pipeline

        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/assets/generate",
                json={"prompt": "anime girl"},
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "done"
