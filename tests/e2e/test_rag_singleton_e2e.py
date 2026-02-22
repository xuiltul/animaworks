"""E2E tests for RAG singleton integration with MemoryManager."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset singletons before and after each test for isolation."""
    from core.memory.rag.singleton import _reset_for_testing

    _reset_for_testing()
    yield
    _reset_for_testing()


@pytest.fixture
def anima_dir(tmp_path, monkeypatch):
    """Create a minimal anima directory structure."""
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))

    # Create required directories
    animas_dir = tmp_path / "animas"
    animas_dir.mkdir()
    anima_path = animas_dir / "test-anima"
    anima_path.mkdir()
    for sub in ("episodes", "knowledge", "procedures", "skills", "state"):
        (anima_path / sub).mkdir()
    (tmp_path / "company").mkdir()
    (tmp_path / "company" / "vision.md").write_text("# Vision", encoding="utf-8")
    (tmp_path / "common_skills").mkdir()
    (tmp_path / "common_knowledge").mkdir()
    (tmp_path / "shared" / "users").mkdir(parents=True)

    return anima_path


class TestMemoryManagerSingleton:
    def test_init_indexer_uses_singleton(self, anima_dir):
        """MemoryManager._init_indexer() should use get_vector_store() singleton."""
        mock_store = MagicMock()
        mock_model = MagicMock()

        with (
            patch(
                "core.memory.rag.singleton.get_vector_store",
                return_value=mock_store,
            ) as mock_get_store,
            patch(
                "core.memory.rag.singleton.get_embedding_model",
                return_value=mock_model,
            ),
        ):
            from core.memory.manager import MemoryManager

            mgr = MemoryManager(anima_dir)
            # Trigger lazy initialization
            indexer = mgr._get_indexer()

            mock_get_store.assert_called_once_with("test-anima")
            assert indexer is not None
            assert indexer.vector_store is mock_store

    def test_different_animas_get_per_anima_vector_store(self, anima_dir, tmp_path):
        """Different animas should get separate vector stores (per-anima isolation)."""
        mock_store_1 = MagicMock(name="store-1")
        mock_store_2 = MagicMock(name="store-2")
        mock_model = MagicMock()

        # Create a second anima dir
        anima2 = tmp_path / "animas" / "test-anima-2"
        anima2.mkdir()
        for sub in ("episodes", "knowledge", "procedures", "skills", "state"):
            (anima2 / sub).mkdir()

        def per_anima_store(anima_name=None):
            if anima_name == "test-anima":
                return mock_store_1
            return mock_store_2

        with (
            patch(
                "core.memory.rag.singleton.get_vector_store",
                side_effect=per_anima_store,
            ) as mock_get_store,
            patch(
                "core.memory.rag.singleton.get_embedding_model",
                return_value=mock_model,
            ),
        ):
            from core.memory.manager import MemoryManager

            mgr1 = MemoryManager(anima_dir)
            mgr2 = MemoryManager(anima2)

            indexer1 = mgr1._get_indexer()
            indexer2 = mgr2._get_indexer()

            # Different animas should get different vector stores
            assert indexer1.vector_store is not indexer2.vector_store
            assert indexer1.vector_store is mock_store_1
            assert indexer2.vector_store is mock_store_2

            # get_vector_store called once per anima with anima name
            assert mock_get_store.call_count == 2
            mock_get_store.assert_any_call("test-anima")
            mock_get_store.assert_any_call("test-anima-2")

    def test_multiple_managers_share_embedding_model(self, anima_dir, tmp_path):
        """Multiple MemoryManager instances should share the same embedding model."""
        mock_store = MagicMock()
        mock_model = MagicMock()

        # Create a second anima dir
        anima2 = tmp_path / "animas" / "test-anima-2"
        anima2.mkdir()
        for sub in ("episodes", "knowledge", "procedures", "skills", "state"):
            (anima2 / sub).mkdir()

        with (
            patch(
                "core.memory.rag.singleton.get_vector_store",
                return_value=mock_store,
            ),
            patch(
                "core.memory.rag.singleton.get_embedding_model",
                return_value=mock_model,
            ),
        ):
            from core.memory.manager import MemoryManager

            mgr1 = MemoryManager(anima_dir)
            mgr2 = MemoryManager(anima2)

            indexer1 = mgr1._get_indexer()
            indexer2 = mgr2._get_indexer()

            # Both should share the same embedding model instance
            assert indexer1.embedding_model is indexer2.embedding_model
            assert indexer1.embedding_model is mock_model
