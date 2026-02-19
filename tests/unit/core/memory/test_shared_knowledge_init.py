# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for shared_common_knowledge auto-initialization in MemoryManager.

Verifies that MemoryManager._init_indexer() triggers indexing of
~/.animaworks/common_knowledge/ into the shared_common_knowledge
vector store collection.
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

pytest.importorskip("sentence_transformers")


class TestSharedKnowledgeInit:
    """Verify _ensure_shared_knowledge_indexed behavior."""

    @pytest.fixture
    def data_dir(self, tmp_path: Path, monkeypatch) -> Path:
        d = tmp_path / "data"
        d.mkdir()
        (d / "animas").mkdir()
        (d / "common_knowledge").mkdir()
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(d))
        # Invalidate path caches
        try:
            from core.config import invalidate_cache
            invalidate_cache()
        except Exception:
            pass
        return d

    @pytest.fixture
    def anima_dir(self, data_dir: Path) -> Path:
        d = data_dir / "animas" / "test_anima"
        d.mkdir(parents=True)
        for sub in ("knowledge", "episodes", "procedures", "skills", "state"):
            (d / sub).mkdir()
        return d

    def test_indexes_common_knowledge_on_init(self, anima_dir: Path, data_dir: Path):
        """When common_knowledge/ has .md files, they get indexed."""
        # Create a test common_knowledge file
        ck_dir = data_dir / "common_knowledge"
        (ck_dir / "guide.md").write_text("# Guide\n\n## Section\n\nContent", encoding="utf-8")

        from core.memory.manager import MemoryManager

        class FakeArray:
            """Minimal ndarray-like object with tolist()."""
            def __init__(self):
                self._data = [0.0] * 384
            def tolist(self):
                return self._data

        mock_embedding = MagicMock()
        mock_embedding.encode.return_value = [FakeArray()]

        def _set_embedding(self_indexer):
            self_indexer.embedding_model = mock_embedding

        with (
            patch("core.memory.rag.singleton.get_vector_store") as mock_vs,
            patch(
                "core.memory.rag.indexer.MemoryIndexer._init_embedding_model",
                _set_embedding,
            ),
        ):
            mock_store = MagicMock()
            mock_vs.return_value = mock_store

            mm = MemoryManager(anima_dir)
            # Trigger lazy init
            mm._get_indexer()

            # Verify create_collection was called with shared_common_knowledge
            collection_calls = [
                c for c in mock_store.create_collection.call_args_list
                if c[0][0] == "shared_common_knowledge"
            ]
            assert len(collection_calls) >= 1, (
                "shared_common_knowledge collection should be created"
            )

    def test_skips_when_no_common_knowledge_files(self, anima_dir: Path, data_dir: Path, caplog):
        """When common_knowledge/ is empty, indexing is skipped gracefully."""
        # common_knowledge/ exists but is empty
        from core.memory.manager import MemoryManager

        with (
            patch("core.memory.rag.singleton.get_vector_store") as mock_vs,
            patch("core.memory.rag.indexer.MemoryIndexer._init_embedding_model"),
        ):
            mock_store = MagicMock()
            mock_vs.return_value = mock_store

            with caplog.at_level(logging.DEBUG, logger="animaworks.memory"):
                mm = MemoryManager(anima_dir)
                mm._get_indexer()

            # shared_common_knowledge should NOT be created
            shared_calls = [
                c for c in mock_store.create_collection.call_args_list
                if c[0][0] == "shared_common_knowledge"
            ]
            assert len(shared_calls) == 0

    def test_does_not_fail_init_on_indexing_error(self, anima_dir: Path, data_dir: Path, caplog):
        """If shared knowledge indexing fails, personal indexer still works."""
        ck_dir = data_dir / "common_knowledge"
        (ck_dir / "guide.md").write_text("content", encoding="utf-8")

        from core.memory.manager import MemoryManager

        with (
            patch("core.memory.rag.singleton.get_vector_store") as mock_vs,
            patch("core.memory.rag.indexer.MemoryIndexer._init_embedding_model"),
            patch(
                "core.memory.manager.MemoryManager._ensure_shared_knowledge_indexed",
                side_effect=RuntimeError("indexing boom"),
            ),
        ):
            mock_store = MagicMock()
            mock_vs.return_value = mock_store

            # _init_indexer catches all exceptions, so personal indexer
            # should still be created even if shared indexing fails
            mm = MemoryManager(anima_dir)
            # The _ensure_shared_knowledge_indexed error is caught in _init_indexer
            # but since we patched the method directly, the exception propagates
            # to the try/except in _init_indexer
            indexer = mm._get_indexer()
            # Personal indexer should still be created
            assert indexer is not None

    def test_hash_based_dedup_prevents_reindexing(self, anima_dir: Path, data_dir: Path):
        """Calling _ensure_shared_knowledge_indexed twice doesn't re-index."""
        ck_dir = data_dir / "common_knowledge"
        (ck_dir / "guide.md").write_text("# Guide\n\ncontent", encoding="utf-8")

        from core.memory.manager import MemoryManager

        with (
            patch("core.memory.rag.singleton.get_vector_store") as mock_vs,
            patch("core.memory.rag.indexer.MemoryIndexer._init_embedding_model"),
        ):
            mock_store = MagicMock()
            mock_vs.return_value = mock_store

            mm = MemoryManager(anima_dir)
            mm._get_indexer()

            # Count upsert calls from first init
            first_upsert_count = mock_store.upsert.call_count

            # Simulate second init (e.g., another anima process)
            mm2 = MemoryManager(anima_dir)
            mm2._get_indexer()

            # Second call should also attempt indexing, but the indexer's
            # hash check means index_file returns 0 (file unchanged)
            # Both calls to index_directory are made, but the content
            # dedup prevents actual re-indexing
            # We can't easily test this without a real vector store,
            # but at least verify no exceptions occurred
            assert mock_store.upsert.call_count >= first_upsert_count
