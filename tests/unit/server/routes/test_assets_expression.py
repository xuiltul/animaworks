"""Unit tests for expression generation endpoint in server/routes/assets.py."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from server.routes.assets import ExpressionGenerateRequest


def _make_test_app(animas_dir: Path):
    from fastapi import FastAPI
    from server.routes.assets import create_assets_router

    app = FastAPI()
    app.state.animas_dir = animas_dir
    app.state.ws_manager = MagicMock()
    app.state.ws_manager.broadcast = AsyncMock()
    router = create_assets_router()
    app.include_router(router, prefix="/api")
    return app


class TestExpressionGenerateRequest:
    """Tests for the ExpressionGenerateRequest Pydantic model."""

    def test_valid_expression(self):
        req = ExpressionGenerateRequest(expression="smile")
        assert req.expression == "smile"

    @pytest.mark.parametrize("expr", [
        "neutral", "smile", "laugh", "troubled",
        "surprised", "thinking", "embarrassed",
    ])
    def test_all_valid_expressions(self, expr):
        req = ExpressionGenerateRequest(expression=expr)
        assert req.expression == expr

    def test_invalid_expression_rejected(self):
        with pytest.raises(Exception):
            ExpressionGenerateRequest(expression="angry")

    def test_invalid_random_string_rejected(self):
        with pytest.raises(Exception):
            ExpressionGenerateRequest(expression="foobar")


class TestGenerateExpressionEndpoint:
    """Tests for POST /animas/{name}/assets/generate-expression."""

    async def test_anima_not_found(self, tmp_path):
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        app = _make_test_app(animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/nobody/assets/generate-expression",
                json={"expression": "smile"},
            )
        assert resp.status_code == 404

    async def test_no_reference_image(self, tmp_path):
        animas_dir = tmp_path / "animas"
        anima_dir = animas_dir / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "assets").mkdir()
        app = _make_test_app(animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/assets/generate-expression",
                json={"expression": "smile"},
            )
        assert resp.status_code == 404
        assert "reference" in resp.json()["detail"].lower()

    async def test_invalid_expression_rejected(self, tmp_path):
        animas_dir = tmp_path / "animas"
        anima_dir = animas_dir / "alice"
        (anima_dir / "assets").mkdir(parents=True)
        (anima_dir / "assets" / "avatar_fullbody.png").write_bytes(b"fake")
        app = _make_test_app(animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/assets/generate-expression",
                json={"expression": "angry"},
            )
        assert resp.status_code == 422

    @patch("core.tools.image_gen.ImageGenPipeline")
    async def test_success_generates_expression(self, mock_pipeline_cls, tmp_path):
        animas_dir = tmp_path / "animas"
        anima_dir = animas_dir / "alice"
        assets_dir = anima_dir / "assets"
        assets_dir.mkdir(parents=True)
        (assets_dir / "avatar_fullbody.png").write_bytes(b"fake-image")

        mock_pipeline = MagicMock()
        mock_result_path = assets_dir / "avatar_bustup_smile.png"
        mock_result_path.write_bytes(b"generated-image")
        mock_pipeline.generate_bustup_expression.return_value = mock_result_path
        mock_pipeline_cls.return_value = mock_pipeline

        app = _make_test_app(animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/assets/generate-expression",
                json={"expression": "smile"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["expression"] == "smile"
        assert data["path"] is not None
