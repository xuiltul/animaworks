# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Unit tests for ConsolidationEngine rag_store injection (Fix N2).

Verifies that when a ``rag_store`` is injected via the constructor,
RAG-consuming methods (``_update_rag_index``, ``_rebuild_rag_index``)
use the injected store instead of calling the ``get_vector_store``
singleton.

Note: ``_fetch_related_knowledge`` was removed in the consolidation
refactor; its tests have been removed accordingly.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.memory.consolidation import ConsolidationEngine


# ── Fixtures ────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    """Create a minimal anima directory structure."""
    d = tmp_path / "animas" / "test_anima"
    d.mkdir(parents=True)
    return d


# ── Constructor Tests ───────────────────────────────────


class TestConsolidationRagStoreConstructor:
    """Tests for rag_store parameter handling in __init__."""

    def test_default_rag_store_is_none(self, anima_dir: Path):
        engine = ConsolidationEngine(anima_dir, "test")
        assert engine._rag_store is None

    def test_injected_rag_store_is_stored(self, anima_dir: Path):
        mock_store = MagicMock()
        engine = ConsolidationEngine(anima_dir, "test", rag_store=mock_store)
        assert engine._rag_store is mock_store

    def test_rag_store_keyword_only(self, anima_dir: Path):
        """rag_store must be passed as keyword argument."""
        mock_store = MagicMock()
        # Positional should fail since rag_store is keyword-only (after *)
        with pytest.raises(TypeError):
            ConsolidationEngine(anima_dir, "test", mock_store)  # type: ignore[misc]


# ── _update_rag_index Tests ─────────────────────────────


class TestUpdateRagIndexInjection:
    """Tests for _update_rag_index using injected vs singleton store."""

    def test_uses_injected_store(self, anima_dir: Path):
        """When rag_store is injected, get_vector_store should NOT be called."""
        mock_store = MagicMock()
        engine = ConsolidationEngine(anima_dir, "test", rag_store=mock_store)

        # Create a test knowledge file
        (anima_dir / "knowledge" / "test.md").write_text(
            "test content", encoding="utf-8",
        )

        with patch("core.memory.rag.MemoryIndexer") as MockIndexer, \
             patch(
                 "core.memory.rag.singleton.get_vector_store",
             ) as mock_get_vs:
            mock_indexer_inst = MagicMock()
            MockIndexer.return_value = mock_indexer_inst

            engine._update_rag_index(["test.md"])

            # Should NOT call get_vector_store since we injected one
            mock_get_vs.assert_not_called()
            # Should use the injected store
            MockIndexer.assert_called_once_with(mock_store, "test", anima_dir)
            mock_indexer_inst.index_file.assert_called_once()

    def test_falls_back_to_singleton(self, anima_dir: Path):
        """When rag_store is None, get_vector_store() IS called."""
        engine = ConsolidationEngine(anima_dir, "test")  # No rag_store

        (anima_dir / "knowledge" / "test.md").write_text(
            "test content", encoding="utf-8",
        )

        with patch("core.memory.rag.MemoryIndexer") as MockIndexer, \
             patch(
                 "core.memory.rag.singleton.get_vector_store",
             ) as mock_get_vs:
            singleton_store = MagicMock()
            mock_get_vs.return_value = singleton_store
            mock_indexer_inst = MagicMock()
            MockIndexer.return_value = mock_indexer_inst

            engine._update_rag_index(["test.md"])

            # Should call get_vector_store since no injected store
            mock_get_vs.assert_called_once_with("test")
            MockIndexer.assert_called_once_with(
                singleton_store, "test", anima_dir,
            )

    def test_noop_when_filenames_empty(self, anima_dir: Path):
        """When filenames is empty, nothing should happen."""
        mock_store = MagicMock()
        engine = ConsolidationEngine(anima_dir, "test", rag_store=mock_store)

        with patch("core.memory.rag.MemoryIndexer") as MockIndexer:
            engine._update_rag_index([])
            MockIndexer.assert_not_called()


# ── _rebuild_rag_index Tests ────────────────────────────


class TestRebuildRagIndexInjection:
    """Tests for _rebuild_rag_index using injected vs singleton store."""

    def test_uses_injected_store(self, anima_dir: Path):
        """When rag_store is injected, get_vector_store should NOT be called."""
        mock_store = MagicMock()
        engine = ConsolidationEngine(anima_dir, "test", rag_store=mock_store)

        # Create test files to index
        (anima_dir / "knowledge" / "k1.md").write_text(
            "knowledge", encoding="utf-8",
        )
        (anima_dir / "episodes" / "2026-02-19.md").write_text(
            "episode", encoding="utf-8",
        )

        with patch("core.memory.rag.MemoryIndexer") as MockIndexer, \
             patch(
                 "core.memory.rag.singleton.get_vector_store",
             ) as mock_get_vs:
            mock_indexer_inst = MagicMock()
            MockIndexer.return_value = mock_indexer_inst

            engine._rebuild_rag_index()

            mock_get_vs.assert_not_called()
            MockIndexer.assert_called_once_with(mock_store, "test", anima_dir)
            # Should index both knowledge and episode files
            assert mock_indexer_inst.index_file.call_count == 2

    def test_falls_back_to_singleton(self, anima_dir: Path):
        """When rag_store is None, get_vector_store() IS called."""
        engine = ConsolidationEngine(anima_dir, "test")

        (anima_dir / "knowledge" / "k1.md").write_text(
            "knowledge", encoding="utf-8",
        )

        with patch("core.memory.rag.MemoryIndexer") as MockIndexer, \
             patch(
                 "core.memory.rag.singleton.get_vector_store",
             ) as mock_get_vs:
            singleton_store = MagicMock()
            mock_get_vs.return_value = singleton_store
            mock_indexer_inst = MagicMock()
            MockIndexer.return_value = mock_indexer_inst

            engine._rebuild_rag_index()

            mock_get_vs.assert_called_once_with("test")
            MockIndexer.assert_called_once_with(
                singleton_store, "test", anima_dir,
            )
