"""Unit tests for core/memory/rag/singleton.py — RAG component singletons."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset singletons before and after each test for isolation."""
    from core.memory.rag.singleton import _reset_for_testing

    _reset_for_testing()
    yield
    _reset_for_testing()


@pytest.fixture
def mock_sentence_transformers():
    """Inject a mock sentence_transformers module into sys.modules.

    This allows patching SentenceTransformer even when the real
    sentence_transformers package is not installed.
    """
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


# ── get_vector_store ──────────────────────────────────────────────


class TestGetVectorStore:
    def test_returns_same_instance(self):
        """get_vector_store() should return the same instance on repeated calls."""
        mock_store = MagicMock()
        with patch(
            "core.memory.rag.store.ChromaVectorStore",
            return_value=mock_store,
        ):
            from core.memory.rag.singleton import get_vector_store

            store1 = get_vector_store()
            store2 = get_vector_store()

        assert store1 is store2
        assert store1 is mock_store

    def test_creates_only_once(self):
        """ChromaVectorStore constructor should be called exactly once."""
        mock_cls = MagicMock()
        with patch(
            "core.memory.rag.store.ChromaVectorStore",
            mock_cls,
        ):
            from core.memory.rag.singleton import get_vector_store

            get_vector_store()
            get_vector_store()
            get_vector_store()

        mock_cls.assert_called_once()


# ── get_embedding_model ──────────────────────────────────────────


class TestGetEmbeddingModel:
    def test_returns_same_instance(
        self, tmp_path, monkeypatch, mock_sentence_transformers
    ):
        """get_embedding_model() should return the same instance on repeated calls."""
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))

        mock_model = MagicMock()
        mock_sentence_transformers.return_value = mock_model

        from core.memory.rag.singleton import get_embedding_model

        model1 = get_embedding_model()
        model2 = get_embedding_model()

        assert model1 is model2
        assert model1 is mock_model

    def test_creates_only_once(
        self, tmp_path, monkeypatch, mock_sentence_transformers
    ):
        """SentenceTransformer constructor should be called exactly once."""
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))

        from core.memory.rag.singleton import get_embedding_model

        get_embedding_model()
        get_embedding_model()
        get_embedding_model()

        mock_sentence_transformers.assert_called_once()

    def test_creates_cache_dir(
        self, tmp_path, monkeypatch, mock_sentence_transformers
    ):
        """get_embedding_model() should create the models cache directory."""
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))

        from core.memory.rag.singleton import get_embedding_model

        get_embedding_model()

        assert (tmp_path / "models").is_dir()

    def test_reads_model_from_config(
        self, tmp_path, monkeypatch, mock_sentence_transformers
    ):
        """get_embedding_model() with no args should read model from config."""
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))

        mock_model = MagicMock()
        mock_sentence_transformers.return_value = mock_model

        with patch(
            "core.memory.rag.singleton._get_configured_model_name",
            return_value="cl-nagoya/ruri-small",
        ):
            from core.memory.rag.singleton import get_embedding_model

            get_embedding_model()

        mock_sentence_transformers.assert_called_once_with(
            "cl-nagoya/ruri-small",
            cache_folder=str(tmp_path / "models"),
        )

    def test_explicit_model_name_overrides_config(
        self, tmp_path, monkeypatch, mock_sentence_transformers
    ):
        """Explicit model_name parameter should override config value."""
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))

        mock_model = MagicMock()
        mock_sentence_transformers.return_value = mock_model

        with patch(
            "core.memory.rag.singleton._get_configured_model_name",
            return_value="intfloat/multilingual-e5-small",
        ):
            from core.memory.rag.singleton import get_embedding_model

            get_embedding_model("pkshatech/RoSEtta-base-ja")

        mock_sentence_transformers.assert_called_once_with(
            "pkshatech/RoSEtta-base-ja",
            cache_folder=str(tmp_path / "models"),
        )

    def test_model_switch_reloads(
        self, tmp_path, monkeypatch, mock_sentence_transformers
    ):
        """Requesting a different model name should discard cache and reload."""
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))

        model_a = MagicMock(name="model_a")
        model_b = MagicMock(name="model_b")
        mock_sentence_transformers.side_effect = [model_a, model_b]

        from core.memory.rag.singleton import get_embedding_model

        result_a = get_embedding_model("model-a")
        result_b = get_embedding_model("model-b")

        assert result_a is model_a
        assert result_b is model_b
        assert mock_sentence_transformers.call_count == 2

    def test_same_model_does_not_reload(
        self, tmp_path, monkeypatch, mock_sentence_transformers
    ):
        """Requesting the same model name should return cached instance."""
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))

        mock_model = MagicMock()
        mock_sentence_transformers.return_value = mock_model

        from core.memory.rag.singleton import get_embedding_model

        m1 = get_embedding_model("model-x")
        m2 = get_embedding_model("model-x")

        assert m1 is m2
        mock_sentence_transformers.assert_called_once()


# ── get_embedding_dimension ──────────────────────────────────────


class TestGetEmbeddingDimension:
    def test_returns_model_dimension(
        self, tmp_path, monkeypatch, mock_sentence_transformers
    ):
        """get_embedding_dimension() should return model's embedding dimension."""
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_sentence_transformers.return_value = mock_model

        from core.memory.rag.singleton import get_embedding_dimension

        dim = get_embedding_dimension()
        assert dim == 768
        mock_model.get_sentence_embedding_dimension.assert_called_once()


# ── get_embedding_model_name ─────────────────────────────────────


class TestGetEmbeddingModelName:
    def test_returns_loaded_model_name(
        self, tmp_path, monkeypatch, mock_sentence_transformers
    ):
        """After loading, get_embedding_model_name() returns the loaded model name."""
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))

        mock_model = MagicMock()
        mock_sentence_transformers.return_value = mock_model

        from core.memory.rag.singleton import (
            get_embedding_model,
            get_embedding_model_name,
        )

        get_embedding_model("cl-nagoya/ruri-small")
        assert get_embedding_model_name() == "cl-nagoya/ruri-small"

    def test_returns_config_when_not_loaded(self):
        """Before loading, get_embedding_model_name() falls back to config."""
        with patch(
            "core.memory.rag.singleton._get_configured_model_name",
            return_value="custom/model",
        ):
            from core.memory.rag.singleton import get_embedding_model_name

            assert get_embedding_model_name() == "custom/model"


# ── _get_configured_model_name ───────────────────────────────────


class TestGetConfiguredModelName:
    def test_reads_from_config(self, tmp_path, monkeypatch):
        """Should read rag.embedding_model from config.json."""
        import json

        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps({
                "rag": {"embedding_model": "cl-nagoya/ruri-small"},
            }),
            encoding="utf-8",
        )
        # Invalidate config cache
        from core.config import invalidate_cache
        invalidate_cache()

        from core.memory.rag.singleton import _get_configured_model_name

        result = _get_configured_model_name()
        assert result == "cl-nagoya/ruri-small"

    def test_fallback_on_missing_config(self, tmp_path, monkeypatch):
        """Should fall back to default when config is unavailable."""
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
        # No config.json exists → load_config returns defaults
        from core.config import invalidate_cache
        invalidate_cache()

        from core.memory.rag.singleton import _get_configured_model_name

        result = _get_configured_model_name()
        assert result == "intfloat/multilingual-e5-small"


# ── _reset_for_testing ───────────────────────────────────────────


class TestResetForTesting:
    def test_reset_clears_singletons(self, tmp_path, monkeypatch):
        """_reset_for_testing() should allow re-creation of singletons."""
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))

        mock_store_1 = MagicMock()
        mock_store_2 = MagicMock()

        from core.memory.rag.singleton import (
            _reset_for_testing,
            get_vector_store,
        )

        with patch(
            "core.memory.rag.store.ChromaVectorStore",
            return_value=mock_store_1,
        ):
            store1 = get_vector_store()

        _reset_for_testing()

        with patch(
            "core.memory.rag.store.ChromaVectorStore",
            return_value=mock_store_2,
        ):
            store2 = get_vector_store()

        assert store1 is not store2
        assert store1 is mock_store_1
        assert store2 is mock_store_2

    def test_reset_clears_model_name(
        self, tmp_path, monkeypatch, mock_sentence_transformers
    ):
        """_reset_for_testing() should clear _embedding_model_name."""
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))

        mock_model = MagicMock()
        mock_sentence_transformers.return_value = mock_model

        from core.memory.rag.singleton import (
            _reset_for_testing,
            get_embedding_model,
            _embedding_model_name,
        )
        import core.memory.rag.singleton as singleton_mod

        get_embedding_model("test-model")
        assert singleton_mod._embedding_model_name == "test-model"

        _reset_for_testing()
        assert singleton_mod._embedding_model_name is None


# ── Thread safety ────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_get_vector_store(self):
        """Multiple threads calling get_vector_store() concurrently
        should all receive the same instance."""
        mock_store = MagicMock()
        results: list[object] = []
        errors: list[Exception] = []

        with patch(
            "core.memory.rag.store.ChromaVectorStore",
            return_value=mock_store,
        ):
            from core.memory.rag.singleton import get_vector_store

            def worker():
                try:
                    store = get_vector_store()
                    results.append(store)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=worker) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

        assert not errors, f"Thread errors: {errors}"
        assert len(results) == 10
        assert all(r is mock_store for r in results)

    def test_concurrent_get_embedding_model(
        self, tmp_path, monkeypatch, mock_sentence_transformers
    ):
        """Multiple threads calling get_embedding_model() concurrently
        should all receive the same instance."""
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))

        mock_model = MagicMock()
        mock_sentence_transformers.return_value = mock_model

        results: list[object] = []
        errors: list[Exception] = []

        from core.memory.rag.singleton import get_embedding_model

        def worker():
            try:
                model = get_embedding_model()
                results.append(model)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Thread errors: {errors}"
        assert len(results) == 10
        assert all(r is mock_model for r in results)


# ── MemoryIndexer integration ────────────────────────────────────


class TestMemoryIndexerEmbeddingInjection:
    def test_accepts_external_embedding_model(self, tmp_path):
        """MemoryIndexer should accept an externally provided embedding_model."""
        mock_store = MagicMock()
        mock_model = MagicMock()

        anima_dir = tmp_path / "test-anima"
        anima_dir.mkdir(parents=True)

        from core.memory.rag.indexer import MemoryIndexer

        indexer = MemoryIndexer(
            vector_store=mock_store,
            anima_name="test-anima",
            anima_dir=anima_dir,
            embedding_model=mock_model,
        )

        assert indexer.embedding_model is mock_model

    def test_skips_init_when_model_provided(self, tmp_path):
        """When embedding_model is provided, _init_embedding_model() should not be called."""
        mock_store = MagicMock()
        mock_model = MagicMock()

        anima_dir = tmp_path / "test-anima"
        anima_dir.mkdir(parents=True)

        with patch(
            "core.memory.rag.indexer.MemoryIndexer._init_embedding_model"
        ) as mock_init:
            from core.memory.rag.indexer import MemoryIndexer

            MemoryIndexer(
                vector_store=mock_store,
                anima_name="test-anima",
                anima_dir=anima_dir,
                embedding_model=mock_model,
            )

            mock_init.assert_not_called()

    def test_calls_singleton_when_no_model_provided(self, tmp_path, monkeypatch):
        """When no embedding_model is provided, _init_embedding_model() should
        use the singleton get_embedding_model()."""
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))

        mock_store = MagicMock()
        mock_model = MagicMock()

        anima_dir = tmp_path / "test-anima"
        anima_dir.mkdir(parents=True)

        with patch(
            "core.memory.rag.singleton.get_embedding_model",
            return_value=mock_model,
        ) as mock_get:
            from core.memory.rag.indexer import MemoryIndexer

            indexer = MemoryIndexer(
                vector_store=mock_store,
                anima_name="test-anima",
                anima_dir=anima_dir,
            )

            mock_get.assert_called_once()
            assert indexer.embedding_model is mock_model

    def test_embedding_model_name_default_is_none(self, tmp_path):
        """MemoryIndexer with no embedding_model_name should pass None to singleton."""
        mock_store = MagicMock()
        mock_model = MagicMock()

        anima_dir = tmp_path / "test-anima"
        anima_dir.mkdir(parents=True)

        with patch(
            "core.memory.rag.singleton.get_embedding_model",
            return_value=mock_model,
        ) as mock_get:
            from core.memory.rag.indexer import MemoryIndexer

            MemoryIndexer(
                vector_store=mock_store,
                anima_name="test-anima",
                anima_dir=anima_dir,
            )

            # Should be called with None (config-resolved)
            mock_get.assert_called_once_with(None)
