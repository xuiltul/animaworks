"""Unit tests for POST /api/internal/embed endpoint."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient


def _make_test_app():
    """Create a minimal FastAPI app with internal routes."""
    from fastapi import FastAPI

    from server.routes.internal import create_internal_router

    app = FastAPI()
    app.state.shared_dir = Path("/tmp/test-shared")
    app.include_router(create_internal_router(), prefix="/api")
    return app


@pytest.fixture
def app():
    return _make_test_app()


# ── Tests ────────────────────────────────────────────────────────


class TestInternalEmbed:
    @pytest.mark.anyio
    async def test_returns_embeddings(self, app):
        """Valid request returns correct embeddings."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        with patch(
            "core.memory.rag.singleton.get_embedding_model",
            return_value=mock_model,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.post(
                    "/api/internal/embed",
                    json={"texts": ["hello", "world"]},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["embeddings"]) == 2
        assert data["embeddings"][0] == pytest.approx([0.1, 0.2, 0.3])

    @pytest.mark.anyio
    async def test_empty_texts_returns_empty(self, app):
        """Empty texts list returns empty embeddings without loading model."""
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/api/internal/embed",
                json={"texts": []},
            )

        assert resp.status_code == 200
        assert resp.json() == {"embeddings": []}

    @pytest.mark.anyio
    async def test_rejects_over_1000_texts(self, app):
        """Requests with >1000 texts should be rejected."""
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/api/internal/embed",
                json={"texts": [f"text_{i}" for i in range(1001)]},
            )

        assert resp.status_code == 400
        assert "Max 1000" in resp.json()["detail"]

    @pytest.mark.anyio
    async def test_single_text(self, app):
        """Single text input should work correctly."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.5, 0.6]])

        with patch(
            "core.memory.rag.singleton.get_embedding_model",
            return_value=mock_model,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.post(
                    "/api/internal/embed",
                    json={"texts": ["single"]},
                )

        assert resp.status_code == 200
        assert len(resp.json()["embeddings"]) == 1

    @pytest.mark.anyio
    async def test_exactly_1000_texts_accepted(self, app):
        """Exactly 1000 texts should be accepted."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1]] * 1000)

        with patch(
            "core.memory.rag.singleton.get_embedding_model",
            return_value=mock_model,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.post(
                    "/api/internal/embed",
                    json={"texts": [f"t{i}" for i in range(1000)]},
                )

        assert resp.status_code == 200
        assert len(resp.json()["embeddings"]) == 1000
