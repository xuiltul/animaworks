"""Tests for core/tools/image_gen.py — Image generation pipeline."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import httpx
import pytest

from core.tools._base import ToolConfigError
from core.tools.image_gen import (
    FluxKontextClient,
    ImageGenPipeline,
    MeshyClient,
    NovelAIClient,
    PipelineResult,
    _image_to_data_uri,
    _retry,
    get_tool_schemas,
)


# ── _image_to_data_uri ───────────────────────────────────────────


class TestImageToDataUri:
    def test_basic(self):
        data = b"\x89PNG\r\n\x1a\n"
        result = _image_to_data_uri(data)
        assert result.startswith("data:image/png;base64,")
        decoded = base64.b64decode(result.split(",", 1)[1])
        assert decoded == data

    def test_custom_mime(self):
        result = _image_to_data_uri(b"test", mime="image/jpeg")
        assert result.startswith("data:image/jpeg;base64,")


# ── _retry ────────────────────────────────────────────────────────


class TestRetry:
    def test_success_first_try(self):
        fn = MagicMock(return_value="ok")
        result = _retry(fn)
        assert result == "ok"
        assert fn.call_count == 1

    def test_retry_on_retryable_status(self):
        resp_429 = MagicMock()
        resp_429.status_code = 429
        req = MagicMock()
        req.url = "http://test"
        error = httpx.HTTPStatusError("429", request=req, response=resp_429)

        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise error
            return "ok"

        with patch("core.tools.image_gen.time.sleep"):
            result = _retry(fn, max_retries=3, delay=0.01)
        assert result == "ok"
        assert call_count == 3

    def test_no_retry_on_non_retryable(self):
        resp_400 = MagicMock()
        resp_400.status_code = 400
        req = MagicMock()
        error = httpx.HTTPStatusError("400", request=req, response=resp_400)

        fn = MagicMock(side_effect=error)
        with pytest.raises(httpx.HTTPStatusError):
            _retry(fn, max_retries=3)
        assert fn.call_count == 1

    def test_retry_on_connect_error(self):
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.ConnectError("connection refused")
            return "ok"

        with patch("core.tools.image_gen.time.sleep"):
            result = _retry(fn, max_retries=2, delay=0.01)
        assert result == "ok"


# ── NovelAIClient ─────────────────────────────────────────────────


class TestNovelAIClient:
    @pytest.fixture(autouse=True)
    def _set_token(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("NOVELAI_TOKEN", "test-nai-token")

    def test_init(self):
        client = NovelAIClient()
        assert client._token == "test-nai-token"

    def test_missing_token(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("NOVELAI_TOKEN", raising=False)
        with pytest.raises(ToolConfigError):
            NovelAIClient()

    def test_extract_png_from_zip(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("image.png", b"fake-png-data")
        raw = buf.getvalue()
        result = NovelAIClient._extract_png(raw)
        assert result == b"fake-png-data"

    def test_extract_png_no_png(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("image.jpg", b"data")
        with pytest.raises(ValueError, match="no PNG"):
            NovelAIClient._extract_png(buf.getvalue())

    def test_generate_fullbody(self):
        # Create a valid ZIP response
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("output.png", b"PNG-BYTES")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = zip_buf.getvalue()
        mock_resp.raise_for_status = MagicMock()

        with patch("core.tools.image_gen.httpx.post", return_value=mock_resp):
            client = NovelAIClient()
            result = client.generate_fullbody("1girl, black hair")
        assert result == b"PNG-BYTES"


# ── FluxKontextClient ─────────────────────────────────────────────


class TestFluxKontextClient:
    @pytest.fixture(autouse=True)
    def _set_key(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("FAL_KEY", "test-fal-key")

    def test_init(self):
        client = FluxKontextClient()
        assert client._key == "test-fal-key"

    def test_missing_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("FAL_KEY", raising=False)
        with pytest.raises(ToolConfigError):
            FluxKontextClient()


# ── MeshyClient ───────────────────────────────────────────────────


class TestMeshyClient:
    @pytest.fixture(autouse=True)
    def _set_key(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("MESHY_API_KEY", "test-meshy-key")

    def test_init(self):
        client = MeshyClient()
        assert client._key == "test-meshy-key"

    def test_missing_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("MESHY_API_KEY", raising=False)
        with pytest.raises(ToolConfigError):
            MeshyClient()

    def test_headers(self):
        client = MeshyClient()
        headers = client._headers()
        assert headers["Authorization"] == "Bearer test-meshy-key"

    def test_download_model_missing_format(self):
        client = MeshyClient()
        task = {"model_urls": {"glb": "http://test/model.glb"}}
        with pytest.raises(ValueError, match="Format 'fbx' not available"):
            client.download_model(task, fmt="fbx")

    def test_download_rigged_model_missing_key(self):
        client = MeshyClient()
        task = {"result": {}}
        with pytest.raises(ValueError, match="missing"):
            client.download_rigged_model(task, fmt="glb")

    def test_download_animation_missing_key(self):
        client = MeshyClient()
        task = {"result": {}}
        with pytest.raises(ValueError, match="missing"):
            client.download_animation(task, fmt="glb")


# ── PipelineResult ────────────────────────────────────────────────


class TestPipelineResult:
    def test_defaults(self):
        r = PipelineResult()
        assert r.fullbody_path is None
        assert r.errors == []
        assert r.skipped == []

    def test_to_dict(self, tmp_path: Path):
        r = PipelineResult(
            fullbody_path=tmp_path / "fb.png",
            errors=["err1"],
            skipped=["bustup"],
        )
        d = r.to_dict()
        assert "fb.png" in d["fullbody"]
        assert d["errors"] == ["err1"]
        assert d["skipped"] == ["bustup"]

    def test_to_dict_empty(self):
        r = PipelineResult()
        d = r.to_dict()
        assert d["fullbody"] is None
        assert d["model"] is None
        assert d["animations"] == {}


# ── ImageGenPipeline ──────────────────────────────────────────────


class TestImageGenPipeline:
    def test_init(self, tmp_path: Path):
        pipe = ImageGenPipeline(tmp_path)
        assert pipe._anima_dir == tmp_path

    def test_generate_all_skips_existing(self, tmp_path: Path):
        assets = tmp_path / "assets"
        assets.mkdir()
        (assets / "avatar_fullbody.png").write_bytes(b"existing")

        pipe = ImageGenPipeline(tmp_path)
        # Only run fullbody step
        result = pipe.generate_all(
            prompt="test",
            skip_existing=True,
            steps=["fullbody"],
        )
        assert "fullbody" in result.skipped
        assert result.fullbody_path is not None

    def test_generate_all_no_fullbody_fails(self, tmp_path: Path):
        pipe = ImageGenPipeline(tmp_path)
        # Skip fullbody step -> no reference image
        result = pipe.generate_all(prompt="test", steps=["bustup"])
        assert len(result.errors) > 0

    def test_init_with_config(self, tmp_path: Path):
        from core.config.models import ImageGenConfig

        cfg = ImageGenConfig(style_prefix="anime, ", vibe_strength=0.7)
        pipe = ImageGenPipeline(tmp_path, config=cfg)
        assert pipe._config is cfg
        assert pipe._config.style_prefix == "anime, "
        assert pipe._config.vibe_strength == 0.7

    def test_init_without_config_uses_default(self, tmp_path: Path):
        from core.config.models import ImageGenConfig

        pipe = ImageGenPipeline(tmp_path)
        assert isinstance(pipe._config, ImageGenConfig)
        assert pipe._config.style_reference is None
        assert pipe._config.style_prefix == ""
        assert pipe._config.vibe_strength == 0.6

    def test_generate_all_applies_style_prefix_suffix(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("NOVELAI_TOKEN", "test-token")
        from core.config.models import ImageGenConfig

        cfg = ImageGenConfig(
            style_prefix="anime coloring, ",
            style_suffix=", high quality",
        )
        pipe = ImageGenPipeline(tmp_path, config=cfg)

        with patch("core.tools.image_gen.NovelAIClient") as mock_nai_cls:
            mock_client = MagicMock()
            mock_client.generate_fullbody.return_value = b"PNG-DATA"
            mock_nai_cls.return_value = mock_client

            pipe.generate_all(
                prompt="1girl, black hair",
                skip_existing=False,
                steps=["fullbody"],
            )

            call_kwargs = mock_client.generate_fullbody.call_args[1]
            assert call_kwargs["prompt"] == "anime coloring, 1girl, black hair, high quality"

    def test_generate_all_applies_negative_prompt_extra(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("NOVELAI_TOKEN", "test-token")
        from core.config.models import ImageGenConfig

        cfg = ImageGenConfig(negative_prompt_extra="realistic, 3d render")
        pipe = ImageGenPipeline(tmp_path, config=cfg)

        with patch("core.tools.image_gen.NovelAIClient") as mock_nai_cls:
            mock_client = MagicMock()
            mock_client.generate_fullbody.return_value = b"PNG-DATA"
            mock_nai_cls.return_value = mock_client

            pipe.generate_all(
                prompt="1girl",
                negative_prompt="lowres, bad anatomy",
                skip_existing=False,
                steps=["fullbody"],
            )

            call_kwargs = mock_client.generate_fullbody.call_args[1]
            assert call_kwargs["negative_prompt"] == "lowres, bad anatomy, realistic, 3d render"

    def test_generate_all_applies_negative_prompt_extra_empty(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("NOVELAI_TOKEN", "test-token")
        from core.config.models import ImageGenConfig

        cfg = ImageGenConfig(negative_prompt_extra="realistic")
        pipe = ImageGenPipeline(tmp_path, config=cfg)

        with patch("core.tools.image_gen.NovelAIClient") as mock_nai_cls:
            mock_client = MagicMock()
            mock_client.generate_fullbody.return_value = b"PNG-DATA"
            mock_nai_cls.return_value = mock_client

            pipe.generate_all(
                prompt="1girl",
                negative_prompt="",
                skip_existing=False,
                steps=["fullbody"],
            )

            call_kwargs = mock_client.generate_fullbody.call_args[1]
            assert call_kwargs["negative_prompt"] == "realistic"

    def test_generate_all_loads_style_reference(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("NOVELAI_TOKEN", "test-token")
        from core.config.models import ImageGenConfig

        style_ref = tmp_path / "style_ref.png"
        style_ref.write_bytes(b"STYLE-IMAGE-DATA")

        cfg = ImageGenConfig(style_reference=str(style_ref))
        pipe = ImageGenPipeline(tmp_path, config=cfg)

        with patch("core.tools.image_gen.NovelAIClient") as mock_nai_cls:
            mock_client = MagicMock()
            mock_client.generate_fullbody.return_value = b"PNG-DATA"
            mock_nai_cls.return_value = mock_client

            pipe.generate_all(
                prompt="1girl",
                skip_existing=False,
                steps=["fullbody"],
            )

            call_kwargs = mock_client.generate_fullbody.call_args[1]
            assert call_kwargs["vibe_image"] == b"STYLE-IMAGE-DATA"

    def test_generate_all_warns_missing_style_reference(self, tmp_path: Path, monkeypatch, caplog):
        monkeypatch.setenv("NOVELAI_TOKEN", "test-token")
        import logging
        from core.config.models import ImageGenConfig

        cfg = ImageGenConfig(style_reference="/nonexistent/path/style.png")
        pipe = ImageGenPipeline(tmp_path, config=cfg)

        with patch("core.tools.image_gen.NovelAIClient") as mock_nai_cls:
            mock_client = MagicMock()
            mock_client.generate_fullbody.return_value = b"PNG-DATA"
            mock_nai_cls.return_value = mock_client

            with caplog.at_level(logging.WARNING):
                pipe.generate_all(
                    prompt="1girl",
                    skip_existing=False,
                    steps=["fullbody"],
                )

            call_kwargs = mock_client.generate_fullbody.call_args[1]
            assert call_kwargs["vibe_image"] is None
            assert any("Style reference not found" in r.message for r in caplog.records)

    def test_generate_all_passes_vibe_params(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("NOVELAI_TOKEN", "test-token")
        from core.config.models import ImageGenConfig

        cfg = ImageGenConfig(vibe_strength=0.3, vibe_info_extracted=0.5)
        pipe = ImageGenPipeline(tmp_path, config=cfg)

        with patch("core.tools.image_gen.NovelAIClient") as mock_nai_cls:
            mock_client = MagicMock()
            mock_client.generate_fullbody.return_value = b"PNG-DATA"
            mock_nai_cls.return_value = mock_client

            pipe.generate_all(
                prompt="1girl",
                skip_existing=False,
                steps=["fullbody"],
            )

            call_kwargs = mock_client.generate_fullbody.call_args[1]
            assert call_kwargs["vibe_strength"] == 0.3
            assert call_kwargs["vibe_info_extracted"] == 0.5


# ── get_tool_schemas ──────────────────────────────────────────────


class TestGetToolSchemas:
    def test_returns_schemas(self):
        schemas = get_tool_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) == 7
        names = {s["name"] for s in schemas}
        expected = {
            "generate_character_assets", "generate_fullbody",
            "generate_bustup", "generate_chibi",
            "generate_3d_model", "generate_rigged_model",
            "generate_animations",
        }
        assert names == expected

    def test_generate_character_assets_requires_prompt(self):
        schemas = get_tool_schemas()
        s = [s for s in schemas if s["name"] == "generate_character_assets"][0]
        assert "prompt" in s["input_schema"]["required"]
