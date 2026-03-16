"""E2E tests for centralized embedding service.

Validates the full pipeline:
  server endpoint → HTTP client → generate_embeddings() → correct results.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset singletons before/after each test."""
    from core.memory.rag.singleton import _reset_for_testing

    _reset_for_testing()
    yield
    _reset_for_testing()


def _make_embed_app():
    """Create a FastAPI app with the embed endpoint."""
    from fastapi import FastAPI

    from server.routes.internal import create_internal_router

    app = FastAPI()
    app.state.shared_dir = Path("/tmp/test-shared")
    app.include_router(create_internal_router(), prefix="/api")
    return app


class TestEmbedCentralizationE2E:
    """End-to-end: generate_embeddings() → HTTP → server endpoint → result."""

    @pytest.mark.anyio
    async def test_http_mode_round_trip(self):
        """generate_embeddings() in HTTP mode should call the server endpoint
        and return correct embeddings."""
        app = _make_embed_app()

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ])

        with patch(
            "core.memory.rag.singleton.get_embedding_model",
            return_value=mock_model,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://testserver",
            ) as client:
                resp = await client.post(
                    "/api/internal/embed",
                    json={"texts": ["hello world", "foo bar"]},
                )

        assert resp.status_code == 200
        embeddings = resp.json()["embeddings"]
        assert len(embeddings) == 2
        assert embeddings[0] == pytest.approx([0.1, 0.2, 0.3])
        assert embeddings[1] == pytest.approx([0.4, 0.5, 0.6])

    @pytest.mark.anyio
    async def test_generate_embeddings_http_with_real_server(self, monkeypatch):
        """Full round-trip: generate_embeddings() HTTP client → live ASGI app."""
        monkeypatch.setenv("ANIMAWORKS_EMBED_URL", "http://testserver/api/internal/embed")

        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[1.0, 2.0, 3.0]]}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response):
            from core.memory.rag.singleton import generate_embeddings

            result = generate_embeddings(["test text"])

        assert len(result) == 1
        assert result[0] == pytest.approx([1.0, 2.0, 3.0])

    @pytest.mark.anyio
    async def test_local_mode_fallback(self, tmp_path, monkeypatch):
        """Without ANIMAWORKS_EMBED_URL, generate_embeddings uses local model."""
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
        monkeypatch.delenv("ANIMAWORKS_EMBED_URL", raising=False)

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[7.0, 8.0]])

        with patch(
            "core.memory.rag.singleton.get_embedding_model",
            return_value=mock_model,
        ):
            from core.memory.rag.singleton import generate_embeddings

            result = generate_embeddings(["local test"])

        assert result == [pytest.approx([7.0, 8.0])]
        mock_model.encode.assert_called_once()

    @pytest.mark.anyio
    async def test_batch_splitting_e2e(self, monkeypatch):
        """Large batch (>1000) is split and reassembled correctly."""
        monkeypatch.setenv("ANIMAWORKS_EMBED_URL", "http://testserver/api/internal/embed")

        call_count = 0

        def mock_post(url, **kwargs):
            nonlocal call_count
            call_count += 1
            texts = kwargs.get("json", {}).get("texts", [])
            resp = MagicMock()
            resp.json.return_value = {
                "embeddings": [[float(call_count)] * 3] * len(texts)
            }
            resp.raise_for_status = MagicMock()
            return resp

        with patch("httpx.post", side_effect=mock_post):
            from core.memory.rag.singleton import generate_embeddings

            texts = [f"text_{i}" for i in range(1500)]
            result = generate_embeddings(texts)

        assert len(result) == 1500
        assert call_count == 2

    @pytest.mark.anyio
    async def test_indexer_uses_generate_embeddings(self, tmp_path):
        """MemoryIndexer._generate_embeddings() delegates to generate_embeddings()."""
        mock_store = MagicMock()
        mock_model = MagicMock()

        anima_dir = tmp_path / "test-anima"
        anima_dir.mkdir(parents=True)

        expected = [[0.1, 0.2], [0.3, 0.4]]

        with patch(
            "core.memory.rag.singleton.generate_embeddings",
            return_value=expected,
        ) as mock_gen:
            from core.memory.rag.indexer import MemoryIndexer

            indexer = MemoryIndexer(
                vector_store=mock_store,
                anima_name="test-anima",
                anima_dir=anima_dir,
                embedding_model=mock_model,
            )
            result = indexer._generate_embeddings(["a", "b"])

        assert result == expected
        mock_gen.assert_called_once_with(["a", "b"])

    @pytest.mark.anyio
    async def test_env_var_not_set_in_server_process(self, monkeypatch):
        """Without ANIMAWORKS_EMBED_URL, generate_embeddings uses local mode
        (server process behavior)."""
        monkeypatch.delenv("ANIMAWORKS_EMBED_URL", raising=False)

        from core.memory.rag.singleton import generate_embeddings

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0]])

        with patch(
            "core.memory.rag.singleton.get_embedding_model",
            return_value=mock_model,
        ):
            result = generate_embeddings(["server-side"])

        assert result == [pytest.approx([1.0])]
        mock_model.encode.assert_called_once()
