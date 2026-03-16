"""Unit tests for generate_embeddings() — HTTP/local routing in singleton.py."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset singletons and module-level _EMBED_URL before/after each test."""
    from core.memory.rag.singleton import _reset_for_testing

    _reset_for_testing()
    yield
    _reset_for_testing()


@pytest.fixture
def mock_sentence_transformers():
    """Inject a mock sentence_transformers module."""
    mock_cls = MagicMock()
    mock_module = types.ModuleType("sentence_transformers")
    mock_module.SentenceTransformer = mock_cls  # type: ignore[attr-defined]

    already_present = "sentence_transformers" in sys.modules
    original = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = mock_module
    yield mock_cls
    if already_present:
        sys.modules["sentence_transformers"] = original  # type: ignore[assignment]
    else:
        sys.modules.pop("sentence_transformers", None)


# ── Local mode ───────────────────────────────────────────────────


class TestGenerateEmbeddingsLocal:
    def test_empty_input(self, monkeypatch):
        """Empty list should return empty list without touching model."""
        monkeypatch.delenv("ANIMAWORKS_EMBED_URL", raising=False)

        from core.memory.rag.singleton import generate_embeddings

        result = generate_embeddings([])
        assert result == []

    def test_local_mode_uses_model(
        self, tmp_path, monkeypatch, mock_sentence_transformers
    ):
        """When ANIMAWORKS_EMBED_URL is not set, generate_embeddings uses local model."""
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
        monkeypatch.delenv("ANIMAWORKS_EMBED_URL", raising=False)

        import numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_sentence_transformers.return_value = mock_model

        from core.memory.rag.singleton import generate_embeddings

        result = generate_embeddings(["hello", "world"])

        assert len(result) == 2
        assert result[0] == pytest.approx([0.1, 0.2])
        assert result[1] == pytest.approx([0.3, 0.4])
        mock_model.encode.assert_called_once_with(
            ["hello", "world"],
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def test_local_mode_returns_lists(
        self, tmp_path, monkeypatch, mock_sentence_transformers
    ):
        """Local mode should return plain lists, not numpy arrays."""
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
        monkeypatch.delenv("ANIMAWORKS_EMBED_URL", raising=False)

        import numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 2.0]])
        mock_sentence_transformers.return_value = mock_model

        from core.memory.rag.singleton import generate_embeddings

        result = generate_embeddings(["test"])
        assert isinstance(result[0], list)


# ── HTTP mode ────────────────────────────────────────────────────


class TestGenerateEmbeddingsHTTP:
    def test_http_mode_calls_endpoint(self, monkeypatch):
        """When ANIMAWORKS_EMBED_URL is set, generate_embeddings calls HTTP endpoint."""
        monkeypatch.setenv("ANIMAWORKS_EMBED_URL", "http://127.0.0.1:18500/api/internal/embed")

        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response) as mock_post:
            from core.memory.rag.singleton import generate_embeddings

            result = generate_embeddings(["hello", "world"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_post.assert_called_once_with(
            "http://127.0.0.1:18500/api/internal/embed",
            json={"texts": ["hello", "world"]},
            timeout=30.0,
        )

    def test_http_mode_batches_large_requests(self, monkeypatch):
        """Requests over _BATCH_LIMIT should be split into batches."""
        monkeypatch.setenv("ANIMAWORKS_EMBED_URL", "http://localhost/embed")

        batch1_resp = MagicMock()
        batch1_resp.json.return_value = {"embeddings": [[i] for i in range(1000)]}
        batch1_resp.raise_for_status = MagicMock()

        batch2_resp = MagicMock()
        batch2_resp.json.return_value = {"embeddings": [[1000], [1001]]}
        batch2_resp.raise_for_status = MagicMock()

        with patch("httpx.post", side_effect=[batch1_resp, batch2_resp]) as mock_post:
            from core.memory.rag.singleton import generate_embeddings

            texts = [f"text_{i}" for i in range(1002)]
            result = generate_embeddings(texts)

        assert len(result) == 1002
        assert mock_post.call_count == 2

    def test_http_mode_propagates_errors(self, monkeypatch):
        """HTTP errors should propagate as exceptions (no local fallback)."""
        import httpx

        monkeypatch.setenv("ANIMAWORKS_EMBED_URL", "http://localhost/embed")

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )

        with (
            patch("httpx.post", return_value=mock_response),
            pytest.raises(httpx.HTTPStatusError),
        ):
            from core.memory.rag.singleton import generate_embeddings

            generate_embeddings(["test"])

    def test_empty_input_skips_http(self, monkeypatch):
        """Empty list should return immediately without HTTP call."""
        monkeypatch.setenv("ANIMAWORKS_EMBED_URL", "http://localhost/embed")

        with patch("httpx.post") as mock_post:
            from core.memory.rag.singleton import generate_embeddings

            result = generate_embeddings([])

        assert result == []
        mock_post.assert_not_called()


# ── Indexer delegation ───────────────────────────────────────────


class TestIndexerDelegation:
    def test_indexer_delegates_to_generate_embeddings(self, tmp_path):
        """MemoryIndexer._generate_embeddings() should delegate to singleton."""
        mock_store = MagicMock()
        mock_model = MagicMock()

        anima_dir = tmp_path / "test-anima"
        anima_dir.mkdir(parents=True)

        with patch(
            "core.memory.rag.singleton.generate_embeddings",
            return_value=[[0.1, 0.2]],
        ) as mock_gen:
            from core.memory.rag.indexer import MemoryIndexer

            indexer = MemoryIndexer(
                vector_store=mock_store,
                anima_name="test-anima",
                anima_dir=anima_dir,
                embedding_model=mock_model,
            )
            result = indexer._generate_embeddings(["hello"])

        assert result == [[0.1, 0.2]]
        mock_gen.assert_called_once_with(["hello"])

    def test_indexer_skips_model_init_when_embed_url_set(self, tmp_path, monkeypatch):
        """When ANIMAWORKS_EMBED_URL is set, indexer skips model loading."""
        monkeypatch.setenv("ANIMAWORKS_EMBED_URL", "http://localhost/embed")

        mock_store = MagicMock()
        anima_dir = tmp_path / "test-anima"
        anima_dir.mkdir(parents=True)

        with patch(
            "core.memory.rag.indexer.MemoryIndexer._init_embedding_model"
        ) as mock_init:
            from core.memory.rag.indexer import MemoryIndexer

            indexer = MemoryIndexer(
                vector_store=mock_store,
                anima_name="test-anima",
                anima_dir=anima_dir,
            )

            mock_init.assert_not_called()
            assert indexer.embedding_model is None
