"""Unit tests for server/routes/assets.py — Asset serving endpoints."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import ASGITransport, AsyncClient

from core.config.models import ImageGenConfig
from core.integrations.image_gen import PipelineResult


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
        (anima_dir / "IDENTITY.md").write_text("# Alice", encoding="utf-8")

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
        (anima_dir / "IDENTITY.md").write_text("# Alice\nイメージカラー: ピンク #FF69B4\n", encoding="utf-8")
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
        (anima_dir / "IDENTITY.md").write_text("# Alice", encoding="utf-8")
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

    async def test_attachment_content_disposition_and_nosniff(self, tmp_path):
        anima_dir = tmp_path / "alice"
        attachment_dir = anima_dir / "attachments"
        attachment_dir.mkdir(parents=True)
        for filename in ("document.pdf", "page.html", "vector.svg", "preview.png", "notes..txt"):
            (attachment_dir / filename).write_bytes(b"content")

        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            responses = {
                filename: await client.get(f"/api/animas/alice/attachments/{filename}")
                for filename in ("document.pdf", "page.html", "vector.svg", "preview.png", "notes..txt")
            }

        for filename in ("document.pdf", "page.html", "vector.svg", "preview.png"):
            assert responses[filename].status_code == 200
            assert responses[filename].headers.get("x-content-type-options") == "nosniff"
        for filename in ("document.pdf", "page.html", "vector.svg"):
            disposition = responses[filename].headers.get("content-disposition", "")
            assert disposition.startswith("attachment; filename*=UTF-8''")
            assert filename in disposition
        assert "content-disposition" not in responses["preview.png"].headers
        assert "image/png" in responses["preview.png"].headers.get("content-type", "")
        assert responses["notes..txt"].status_code == 200
        assert responses["notes..txt"].headers.get("content-disposition", "").startswith("attachment;")

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


def _write_test_png(path: Path, size: int = 256) -> Path:
    """Write a solid-color PNG of *size*×*size* for thumbnail tests."""
    from PIL import Image

    img = Image.new("RGB", (size, size), color=(200, 100, 50))
    img.save(path, format="PNG")
    return path


class TestAssetThumbnail:
    """Server-side ?size=S|M|L WebP thumbnail generation."""

    async def test_portrait_sizes_preserve_aspect_ratio_and_use_requested_edge(self, tmp_path):
        from io import BytesIO

        from PIL import Image

        animas_dir = tmp_path / "animas"
        assets_dir = animas_dir / "alice" / "assets"
        assets_dir.mkdir(parents=True)
        Image.new("RGBA", (1000, 1500), color=(200, 100, 50, 128)).save(assets_dir / "bustup.png")

        app = _make_test_app(animas_dir=animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            panel = await client.get("/api/animas/alice/assets/bustup.png?size=P")
            call = await client.get("/api/animas/alice/assets/bustup.png?size=C")

        assert panel.status_code == call.status_code == 200
        panel_image = Image.open(BytesIO(panel.content))
        call_image = Image.open(BytesIO(call.content))
        assert panel_image.format == call_image.format == "WEBP"
        assert panel_image.size == (360, 540)
        assert call_image.size == (733, 1100)
        assert panel_image.mode == call_image.mode == "RGBA"
        assert (assets_dir / ".thumbs" / "bustup.png.P.webp").is_file()
        assert (assets_dir / ".thumbs" / "bustup.png.C.webp").is_file()

    async def test_portrait_sizes_regenerate_after_source_update(self, tmp_path):
        import os
        import time

        from PIL import Image

        animas_dir = tmp_path / "animas"
        assets_dir = animas_dir / "alice" / "assets"
        assets_dir.mkdir(parents=True)
        source = assets_dir / "bustup.png"
        Image.new("RGB", (1000, 1500), color=(200, 100, 50)).save(source)

        app = _make_test_app(animas_dir=animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            for key in ("P", "C"):
                response = await client.get(f"/api/animas/alice/assets/bustup.png?size={key}")
                assert response.status_code == 200
            caches = {key: assets_dir / ".thumbs" / f"bustup.png.{key}.webp" for key in ("P", "C")}
            mtimes = {key: cache.stat().st_mtime_ns for key, cache in caches.items()}

            time.sleep(0.02)
            Image.new("RGB", (1100, 1600), color=(80, 120, 200)).save(source)
            source_stat = source.stat()
            os.utime(source, ns=(source_stat.st_atime_ns, max(mtimes.values()) + 1_000_000))

            for key in ("P", "C"):
                response = await client.get(f"/api/animas/alice/assets/bustup.png?size={key}")
                assert response.status_code == 200

        assert all(caches[key].stat().st_mtime_ns > mtimes[key] for key in caches)

    async def test_size_s_returns_96px_webp(self, tmp_path):
        animas_dir = tmp_path / "animas"
        assets_dir = animas_dir / "alice" / "assets"
        assets_dir.mkdir(parents=True)
        src = _write_test_png(assets_dir / "icon.png", size=512)
        original_size = src.stat().st_size

        app = _make_test_app(animas_dir=animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/alice/assets/icon.png?size=S")

        assert resp.status_code == 200
        assert "image/webp" in resp.headers.get("content-type", "")
        assert "must-revalidate" in resp.headers.get("cache-control", "")
        assert "etag" in {k.lower() for k in resp.headers}

        from io import BytesIO

        from PIL import Image

        thumb = Image.open(BytesIO(resp.content))
        assert thumb.size == (96, 96)
        assert thumb.format == "WEBP"
        assert len(resp.content) < original_size

        # Disk cache created
        cache = assets_dir / ".thumbs" / "icon.png.S.webp"
        assert cache.is_file()

    async def test_cache_hit_on_second_request(self, tmp_path):
        animas_dir = tmp_path / "animas"
        assets_dir = animas_dir / "alice" / "assets"
        assets_dir.mkdir(parents=True)
        _write_test_png(assets_dir / "icon.png", size=256)

        app = _make_test_app(animas_dir=animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r1 = await client.get("/api/animas/alice/assets/icon.png?size=M")
            cache = assets_dir / ".thumbs" / "icon.png.M.webp"
            assert cache.is_file()
            mtime1 = cache.stat().st_mtime_ns
            content1 = r1.content

            r2 = await client.get("/api/animas/alice/assets/icon.png?size=M")
            mtime2 = cache.stat().st_mtime_ns

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r2.content == content1
        assert mtime2 == mtime1  # cache not regenerated

    async def test_regenerate_when_source_updated(self, tmp_path):
        import os
        import time

        animas_dir = tmp_path / "animas"
        assets_dir = animas_dir / "alice" / "assets"
        assets_dir.mkdir(parents=True)
        src = _write_test_png(assets_dir / "icon.png", size=200)

        app = _make_test_app(animas_dir=animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r1 = await client.get("/api/animas/alice/assets/icon.png?size=S")
            cache = assets_dir / ".thumbs" / "icon.png.S.webp"
            mtime1 = cache.stat().st_mtime_ns

            # Update source so mtime is newer than cache
            time.sleep(0.02)
            _write_test_png(src, size=300)
            # Ensure source mtime is strictly newer (some FS have coarse resolution)
            src_stat = src.stat()
            os.utime(src, ns=(src_stat.st_atime_ns, mtime1 + 1_000_000))

            r2 = await client.get("/api/animas/alice/assets/icon.png?size=S")
            mtime2 = cache.stat().st_mtime_ns

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert mtime2 > mtime1

    async def test_invalid_size_falls_back_to_original(self, tmp_path):
        animas_dir = tmp_path / "animas"
        assets_dir = animas_dir / "alice" / "assets"
        assets_dir.mkdir(parents=True)
        src = _write_test_png(assets_dir / "icon.png", size=128)
        original = src.read_bytes()

        app = _make_test_app(animas_dir=animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/alice/assets/icon.png?size=XL")

        assert resp.status_code == 200
        assert "image/png" in resp.headers.get("content-type", "")
        assert resp.content == original
        assert not (assets_dir / ".thumbs").exists()

    async def test_corrupt_image_falls_back_to_original(self, tmp_path):
        animas_dir = tmp_path / "animas"
        assets_dir = animas_dir / "alice" / "assets"
        assets_dir.mkdir(parents=True)
        corrupt = b"\x89PNG\r\n\x1a\nnot-a-real-image"
        (assets_dir / "broken.png").write_bytes(corrupt)

        app = _make_test_app(animas_dir=animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/alice/assets/broken.png?size=S")

        assert resp.status_code == 200
        assert "image/png" in resp.headers.get("content-type", "")
        assert resp.content == corrupt

    async def test_pillow_import_error_falls_back(self, tmp_path):
        import sys

        animas_dir = tmp_path / "animas"
        assets_dir = animas_dir / "alice" / "assets"
        assets_dir.mkdir(parents=True)
        src = _write_test_png(assets_dir / "icon.png", size=128)
        original = src.read_bytes()

        app = _make_test_app(animas_dir=animas_dir)
        transport = ASGITransport(app=app)
        # sys.modules[PIL]=None makes `from PIL import Image` raise ImportError
        blocked = {k: None for k in list(sys.modules) if k == "PIL" or k.startswith("PIL.")}
        with patch.dict(sys.modules, blocked):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/animas/alice/assets/icon.png?size=S")

        assert resp.status_code == 200
        assert "image/png" in resp.headers.get("content-type", "")
        assert resp.content == original

    async def test_omit_size_serves_original(self, tmp_path):
        animas_dir = tmp_path / "animas"
        assets_dir = animas_dir / "alice" / "assets"
        assets_dir.mkdir(parents=True)
        src = _write_test_png(assets_dir / "icon.png", size=128)
        original = src.read_bytes()

        app = _make_test_app(animas_dir=animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/alice/assets/icon.png")

        assert resp.status_code == 200
        assert "image/png" in resp.headers.get("content-type", "")
        assert resp.content == original

    async def test_size_l_is_400px(self, tmp_path):
        animas_dir = tmp_path / "animas"
        assets_dir = animas_dir / "alice" / "assets"
        assets_dir.mkdir(parents=True)
        _write_test_png(assets_dir / "icon.png", size=800)

        app = _make_test_app(animas_dir=animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/alice/assets/icon.png?size=L")

        from io import BytesIO

        from PIL import Image

        thumb = Image.open(BytesIO(resp.content))
        assert thumb.size == (400, 400)


class TestGetAttachment:
    async def test_serve_attachment(self, tmp_path):
        animas_dir = tmp_path / "animas"
        anima_dir = animas_dir / "alice"
        attachment_dir = anima_dir / "attachments"
        attachment_dir.mkdir(parents=True)
        (attachment_dir / "uploaded.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        app = _make_test_app(animas_dir=animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/alice/attachments/uploaded.png")
        assert resp.status_code == 200
        assert "image/png" in resp.headers.get("content-type", "")


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

    @patch("core.integrations.image_gen.ImageGenPipeline")
    async def test_generate_success(self, mock_pipeline_cls, tmp_path):
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()

        mock_result = MagicMock()
        mock_result.fullbody_path = Path("/tmp/fb.png")
        mock_result.bustup_path = None
        mock_result.icon_path = None
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

    @patch("core.integrations.image_gen.ImageGenPipeline")
    @patch("core.config.models.load_config")
    async def test_generate_uses_global_image_backend(self, mock_load_config, mock_pipeline_cls, tmp_path):
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()

        mock_result = MagicMock()
        mock_result.fullbody_path = None
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

        mock_load_config.return_value = MagicMock(image_gen=ImageGenConfig(backend="diffusers", image_style="anime"))
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/assets/generate",
                json={"prompt": "anime girl", "image_style": "realistic"},
            )

        assert resp.status_code == 200
        config = mock_pipeline_cls.call_args.kwargs["config"]
        assert config.backend == "diffusers"
        assert config.image_style == "realistic"


# ── GET /animas/{name}/assets/metadata (icon cross-mode fallback) ──


class TestGetAssetMetadataIconFallback:
    """icon.png / icon_realistic.png fallback for the opposite mode's gallery keys."""

    async def test_only_icon_png_fills_realistic_slot(self, tmp_path):
        anima_dir = tmp_path / "alice"
        assets_dir = anima_dir / "assets"
        assets_dir.mkdir(parents=True)
        (assets_dir / "icon.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 8)

        mock_cfg = MagicMock()
        mock_cfg.image_gen.image_style = "anime"

        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        with patch("core.config.models.load_config", return_value=mock_cfg):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/animas/alice/assets/metadata")
        assert resp.status_code == 200
        data = resp.json()
        assert data["assets"]["avatar_icon"]["filename"] == "icon.png"
        assert data["assets_realistic"]["avatar_icon_realistic"]["filename"] == "icon.png"
        assert data["assets_realistic"]["avatar_icon_realistic"]["url"].endswith("/icon.png")

    async def test_only_icon_realistic_fills_anime_slot(self, tmp_path):
        anima_dir = tmp_path / "bob"
        assets_dir = anima_dir / "assets"
        assets_dir.mkdir(parents=True)
        (assets_dir / "icon_realistic.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"y" * 8)

        mock_cfg = MagicMock()
        mock_cfg.image_gen.image_style = "realistic"

        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        with patch("core.config.models.load_config", return_value=mock_cfg):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/animas/bob/assets/metadata")
        assert resp.status_code == 200
        data = resp.json()
        assert data["assets_realistic"]["avatar_icon_realistic"]["filename"] == "icon_realistic.png"
        assert data["assets"]["avatar_icon"]["filename"] == "icon_realistic.png"
        assert data["assets"]["avatar_icon"]["url"].endswith("/icon_realistic.png")


# ── POST /animas/{name}/assets/regenerate-step ───────────


class TestRegenerateAssetStep:
    async def test_anima_not_found(self, tmp_path):
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/nobody/assets/regenerate-step",
                json={"step": "icon"},
            )
        assert resp.status_code == 404

    async def test_fullbody_requires_prompt_or_prompt_txt(self, tmp_path):
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        (anima_dir / "assets").mkdir()

        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        with patch("core.anima.asset_reconciler._resolve_prompt", return_value=""):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/animas/alice/assets/regenerate-step",
                    json={"step": "fullbody", "prompt": ""},
                )
        assert resp.status_code == 400
        assert "prompt is required" in resp.json()["detail"]

    @patch("server.routes.assets.emit", new_callable=AsyncMock)
    @patch("core.integrations.image_gen.ImageGenPipeline")
    async def test_icon_step_ok_and_emits(self, mock_pipeline_cls, mock_emit, tmp_path):
        anima_dir = tmp_path / "alice"
        (anima_dir / "assets").mkdir(parents=True)

        icon_path = anima_dir / "assets" / "icon.png"
        mock_pipeline = MagicMock()
        mock_pipeline.generate_all.return_value = PipelineResult(
            icon_path=icon_path,
            errors=[],
        )
        mock_pipeline_cls.return_value = mock_pipeline
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/assets/regenerate-step",
                json={"step": "icon"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["icon"] == str(icon_path)
        mock_emit.assert_awaited()
        call_kw = mock_emit.call_args[0]
        assert call_kw[1] == "anima.assets_updated"
        assert call_kw[2]["assets"] == ["icon.png"]
        assert call_kw[2]["errors"] == []

        mock_pipeline.generate_all.assert_called_once()
        ga_kw = mock_pipeline.generate_all.call_args[1]
        assert ga_kw["steps"] == ["icon"]
        assert ga_kw["skip_existing"] is False


# ── POST /animas/{name}/assets/remake-preview ───────────


class TestRemakePreview:
    @patch("core.integrations.image_gen.ImageGenPipeline")
    @patch("core.config.models.load_config")
    async def test_without_style_from_ignores_global_style_reference(
        self,
        mock_load_config,
        mock_pipeline_cls,
        tmp_path,
    ):
        import asyncio

        animas_dir = tmp_path / "animas"
        anima_dir = animas_dir / "alice"
        assets_dir = anima_dir / "assets"
        assets_dir.mkdir(parents=True)
        (anima_dir / "IDENTITY.md").write_text("# Alice\n", encoding="utf-8")
        (assets_dir / "avatar_fullbody_realistic.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        global_style_ref = tmp_path / "global-style.png"
        global_style_ref.write_bytes(b"\x89PNG\r\n\x1a\n")
        mock_load_config.return_value = MagicMock(
            image_gen=ImageGenConfig(
                backend="diffusers",
                image_style="anime",
                style_reference=str(global_style_ref),
            ),
        )

        mock_pipeline = MagicMock()
        mock_pipeline.generate_all.return_value = MagicMock(errors=[])
        mock_pipeline_cls.return_value = mock_pipeline

        app = _make_test_app(animas_dir=animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/assets/remake-preview",
                json={"image_style": "realistic", "prompt": "portrait"},
            )

        assert resp.status_code == 202

        config = mock_pipeline_cls.call_args.kwargs["config"]
        assert config.image_style == "realistic"
        assert config.style_reference is None

        await asyncio.sleep(0.05)
        mock_pipeline.generate_all.assert_called_once()
        ga_kw = mock_pipeline.generate_all.call_args.kwargs
        assert "vibe_image" not in ga_kw


# ── POST /animas/{name}/assets/upload-fullbody ──────────


class TestUploadFullbodyAsset:
    _PNG_MIN = b"\x89PNG\r\n\x1a\n" + b"x" * 8

    async def test_anima_not_found(self, tmp_path):
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/nobody/assets/upload-fullbody",
                files={"file": ("f.png", self._PNG_MIN, "image/png")},
                data={"image_style": "anime"},
            )
        assert resp.status_code == 404

    async def test_rejects_too_short_file(self, tmp_path):
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()

        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/assets/upload-fullbody",
                files={"file": ("f.png", b"short", "image/png")},
                data={"image_style": "anime"},
            )
        assert resp.status_code == 400
        assert "invalid" in resp.json()["detail"].lower()

    async def test_rejects_non_image_bytes(self, tmp_path):
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()

        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/assets/upload-fullbody",
                files={"file": ("f.bin", b"\x00" * 16, "application/octet-stream")},
                data={"image_style": "anime"},
            )
        assert resp.status_code == 400
        assert "PNG or JPEG" in resp.json()["detail"]

    @patch("server.routes.assets.emit", new_callable=AsyncMock)
    async def test_writes_anime_fullbody_png(self, mock_emit, tmp_path):
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()

        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/assets/upload-fullbody",
                files={"file": ("fb.png", self._PNG_MIN, "image/png")},
                data={"image_style": "anime"},
            )
        assert resp.status_code == 200
        out = anima_dir / "assets" / "avatar_fullbody.png"
        assert out.is_file()
        assert out.read_bytes() == self._PNG_MIN
        data = resp.json()
        assert data["filename"] == "avatar_fullbody.png"
        assert data["url"].endswith("/avatar_fullbody.png")
        mock_emit.assert_awaited_once()

    @patch("server.routes.assets.emit", new_callable=AsyncMock)
    async def test_writes_realistic_fullbody_png(self, mock_emit, tmp_path):
        anima_dir = tmp_path / "bob"
        anima_dir.mkdir()

        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        jpeg = b"\xff\xd8\xff" + b"z" * 16
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/bob/assets/upload-fullbody",
                files={"file": ("fb.jpg", jpeg, "image/jpeg")},
                data={"image_style": "realistic"},
            )
        assert resp.status_code == 200
        out = anima_dir / "assets" / "avatar_fullbody_realistic.png"
        assert out.read_bytes() == jpeg
        assert resp.json()["filename"] == "avatar_fullbody_realistic.png"


# ── GET /animas/{name}/assets/remake-preview ────────────


class TestListRemakePreviews:
    async def test_anima_not_found(self, tmp_path):
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/nobody/assets/remake-preview")
        assert resp.status_code == 404

    async def test_empty(self, tmp_path):
        (tmp_path / "alice").mkdir()
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/alice/assets/remake-preview")
        assert resp.status_code == 200
        assert resp.json() == {"previews": [], "backup_id": None}

    async def test_lists_previews_and_latest_backup(self, tmp_path):
        anima_dir = tmp_path / "alice"
        assets_dir = anima_dir / "assets"
        assets_dir.mkdir(parents=True)
        (assets_dir / "_preview_001_realistic.png").write_bytes(b"\x89PNG")
        (assets_dir / "_preview_002_realistic.png").write_bytes(b"\x89PNG")
        (assets_dir / "avatar_fullbody_realistic.png").write_bytes(b"\x89PNG")
        (anima_dir / "assets_backup_20260101_000000").mkdir()
        (anima_dir / "assets_backup_20260808_113051").mkdir()

        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/alice/assets/remake-preview")
        data = resp.json()
        assert data["previews"] == ["_preview_001_realistic.png", "_preview_002_realistic.png"]
        assert data["backup_id"] == "assets_backup_20260808_113051"


class TestFaceAssets:
    @staticmethod
    def _write_region_png(path: Path, face_color: tuple[int, int, int]) -> Path:
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (100, 100), (220, 20, 20))
        ImageDraw.Draw(image).rectangle((24, 0, 75, 51), fill=face_color)
        image.save(path, format="PNG")
        return path

    async def test_face_crop_cache_and_size(self, tmp_path):
        from io import BytesIO

        from PIL import Image

        assets_dir = tmp_path / "alice" / "assets"
        assets_dir.mkdir(parents=True)
        source = self._write_region_png(assets_dir / "avatar_bustup_smile.png", (15, 180, 40))
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            first = await client.get("/api/animas/alice/assets/face/smile?size=S&style=anime")
            second = await client.get("/api/animas/alice/assets/face/smile?size=S&style=anime")

        assert first.status_code == 200
        assert first.headers["content-type"].startswith("image/webp")
        image = Image.open(BytesIO(first.content))
        assert image.size == (96, 96)
        assert image.getpixel((48, 48))[1] > 150  # the checked upper-center face crop
        cache = assets_dir / ".thumbs" / "face2_smile_anime.S.webp"
        assert cache.is_file()
        assert second.content == first.content
        assert cache.stat().st_mtime_ns >= source.stat().st_mtime_ns

    async def test_face_cache_rebuilds_when_source_changes(self, tmp_path):
        import os

        assets_dir = tmp_path / "alice" / "assets"
        assets_dir.mkdir(parents=True)
        source = self._write_region_png(assets_dir / "avatar_bustup_troubled.png", (15, 180, 40))
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.get("/api/animas/alice/assets/face/troubled?size=M")
            cache = assets_dir / ".thumbs" / "face2_troubled_anime.M.webp"
            old_mtime = cache.stat().st_mtime_ns
            self._write_region_png(source, (20, 30, 210))
            source_stat = source.stat()
            os.utime(source, ns=(source_stat.st_atime_ns, old_mtime + 1_000_000))
            response = await client.get("/api/animas/alice/assets/face/troubled?size=M")

        assert response.status_code == 200
        assert cache.stat().st_mtime_ns > old_mtime

    async def test_face_crop_uses_style_specific_configuration(self, tmp_path):
        from io import BytesIO

        from PIL import Image, ImageDraw

        assets_dir = tmp_path / "alice" / "assets"
        assets_dir.mkdir(parents=True)
        source = Image.new("RGB", (100, 100), (220, 20, 20))
        ImageDraw.Draw(source).rectangle((70, 50, 89, 69), fill=(15, 180, 40))
        source.save(assets_dir / "avatar_bustup_smile.png", format="PNG")
        (assets_dir / "face_crop.json").write_text(
            '{"anime":{"x":0.7,"y":0.5,"w":0.2},"realistic":{"x":0.0,"y":0.0,"w":0.2}}',
            encoding="utf-8",
        )

        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/animas/alice/assets/face/smile?size=S&style=anime")

        assert response.status_code == 200
        image = Image.open(BytesIO(response.content))
        assert image.getpixel((48, 48))[1] > 150

    async def test_face_uses_style_icon_fallback_when_bustup_missing(self, tmp_path):
        from io import BytesIO

        from PIL import Image

        assets_dir = tmp_path / "alice" / "assets"
        assets_dir.mkdir(parents=True)
        _write_test_png(assets_dir / "icon_realistic.png", size=256)
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/animas/alice/assets/face/laugh?size=M&style=realistic")

        assert response.status_code == 200
        image = Image.open(BytesIO(response.content))
        assert image.format == "WEBP"
        assert image.size == (192, 192)
        assert (assets_dir / ".thumbs" / "face2_laugh_realistic.M.webp").is_file()

    async def test_invalid_emotion_returns_400(self, tmp_path):
        (tmp_path / "alice" / "assets").mkdir(parents=True)
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/animas/alice/assets/face/angry")
        assert response.status_code == 400


class TestAttachmentInlineDelivery:
    async def test_inline_allowed_and_attachment_forced_types(self, tmp_path):
        anima_dir = tmp_path / "alice"
        attachment_dir = anima_dir / "attachments"
        attachment_dir.mkdir(parents=True)
        filenames = ("report.pdf", "track.mp3", "clip.mp4", "notes.txt", "readme.md", "data.csv", "feed.xml")
        forced = ("page.html", "vector.svg", "doc.docx", "mystery.bin")
        for filename in (*filenames, *forced):
            (attachment_dir / filename).write_bytes(b"content")

        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            responses = {
                filename: await client.get(f"/api/animas/alice/attachments/{filename}?inline=1")
                for filename in (*filenames, *forced)
            }

        for filename in filenames:
            disposition = responses[filename].headers.get("content-disposition", "")
            assert disposition.startswith("inline; filename*=UTF-8''"), filename
        for filename in forced:
            disposition = responses[filename].headers.get("content-disposition", "")
            assert disposition.startswith("attachment; filename*=UTF-8''"), filename

        assert responses["readme.md"].headers["content-type"] == "text/plain; charset=utf-8"
        assert responses["notes.txt"].headers["content-type"] == "text/plain; charset=utf-8"
        assert responses["feed.xml"].headers["content-type"] == "text/plain; charset=utf-8"
        assert responses["data.csv"].headers["content-type"] == "text/csv; charset=utf-8"
        assert responses["page.html"].headers["content-type"] == "text/html; charset=utf-8"
        for filename in filenames:
            assert responses[filename].headers["x-content-type-options"] == "nosniff"
            assert responses[filename].headers["cache-control"] == "no-store"

    async def test_inline_csp_except_pdf(self, tmp_path):
        anima_dir = tmp_path / "alice"
        attachment_dir = anima_dir / "attachments"
        attachment_dir.mkdir(parents=True)
        for filename in ("report.pdf", "notes.txt", "track.mp3"):
            (attachment_dir / filename).write_bytes(b"content")

        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            pdf = await client.get("/api/animas/alice/attachments/report.pdf?inline=1")
            text = await client.get("/api/animas/alice/attachments/notes.txt?inline=1")
            audio = await client.get("/api/animas/alice/attachments/track.mp3?inline=1")

        assert "content-security-policy" not in pdf.headers
        csp = text.headers["content-security-policy"]
        assert "sandbox" in csp
        assert "default-src 'none'" in csp
        assert "sandbox" in audio.headers["content-security-policy"]

    async def test_range_request_returns_206(self, tmp_path):
        anima_dir = tmp_path / "alice"
        attachment_dir = anima_dir / "attachments"
        attachment_dir.mkdir(parents=True)
        (attachment_dir / "notes.txt").write_bytes(b"0123456789")

        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/api/animas/alice/attachments/notes.txt",
                headers={"Range": "bytes=0-1"},
            )

        assert response.status_code == 206
        assert response.headers["content-range"] == "bytes 0-1/10"
        assert response.headers["accept-ranges"] == "bytes"
        assert response.content == b"01"

    async def test_image_without_inline_keeps_existing_behaviour(self, tmp_path):
        anima_dir = tmp_path / "alice"
        attachment_dir = anima_dir / "attachments"
        attachment_dir.mkdir(parents=True)
        (attachment_dir / "preview.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/animas/alice/attachments/preview.png")

        assert response.status_code == 200
        assert "content-disposition" not in response.headers
