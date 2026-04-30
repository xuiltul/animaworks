# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MemoryIndexer collection-existence recovery.

Verifies the fix for the bug where a wiped/recreated vectordb would
leave per-anima ``index_meta.json`` hashes intact, causing
``index_file()`` to silently skip indexing and never recreate the
missing collection.

Regression: 2026-04-30 weekly bug investigation (Bug B).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "test_anima"
    d.mkdir()
    (d / "knowledge").mkdir()
    return d


def _make_indexer(anima_dir: Path):
    """Create a MemoryIndexer with the embedding model patched out."""
    from core.memory.rag.indexer import MemoryIndexer

    with patch.object(MemoryIndexer, "_init_embedding_model"):
        idx = MemoryIndexer(
            MagicMock(),
            anima_name=anima_dir.name,
            anima_dir=anima_dir,
        )
    # Patch embedding generation to return deterministic vectors
    idx._generate_embeddings = MagicMock(return_value=[[0.1] * 4])
    return idx


class TestCollectionExistenceCache:
    """Verify the lazy-init collection-existence cache."""

    def test_first_call_populates_cache(self, anima_dir: Path):
        idx = _make_indexer(anima_dir)
        idx.vector_store.list_collections.return_value = ["foo", "bar"]
        assert idx._collection_exists("foo") is True
        assert idx.vector_store.list_collections.call_count == 1
        # Second call uses cache
        assert idx._collection_exists("bar") is True
        assert idx.vector_store.list_collections.call_count == 1

    def test_missing_returns_false(self, anima_dir: Path):
        idx = _make_indexer(anima_dir)
        idx.vector_store.list_collections.return_value = ["foo"]
        assert idx._collection_exists("missing") is False

    def test_listing_failure_is_conservative(self, anima_dir: Path):
        """When list_collections raises, we should NOT trigger spurious re-index."""
        idx = _make_indexer(anima_dir)
        idx.vector_store.list_collections.side_effect = RuntimeError("transient")
        # Be conservative: assume the collection exists rather than
        # forcing a full re-index of everything on a transient error.
        assert idx._collection_exists("any") is True

    def test_mark_known_adds_to_cache(self, anima_dir: Path):
        idx = _make_indexer(anima_dir)
        idx.vector_store.list_collections.return_value = []
        # Initially not present
        assert idx._collection_exists("new") is False
        # After marking known, present without re-listing
        idx._mark_collection_known("new")
        assert idx._collection_exists("new") is True


class TestIndexFileCollectionRecovery:
    """Verify ``index_file`` re-indexes when collection is missing.

    The bug: when the vectordb was wiped but per-anima ``index_meta.json``
    retained file hashes, ``index_file`` would skip indexing forever and
    the collection would never be recreated.
    """

    # Markdown content with a ``## `` heading so chunker yields chunks.
    _SAMPLE_MD = "# Title\n\n## Section A\n\n" + ("body sentence with enough length to chunk. " * 5)

    def test_skips_when_hash_matches_and_collection_exists(self, anima_dir: Path):
        idx = _make_indexer(anima_dir)
        idx.vector_store.list_collections.return_value = ["test_anima_knowledge"]
        idx.vector_store.upsert.return_value = True

        f = anima_dir / "knowledge" / "topic.md"
        f.write_text(self._SAMPLE_MD, encoding="utf-8")

        # First index: hash gets stored
        chunks = idx.index_file(f, "knowledge")
        assert chunks > 0
        assert idx.vector_store.upsert.called

        # Second index: hash matches, collection exists → skip
        idx.vector_store.upsert.reset_mock()
        chunks = idx.index_file(f, "knowledge")
        assert chunks == 0
        assert not idx.vector_store.upsert.called

    def test_force_reindex_when_hash_matches_but_collection_missing(self, anima_dir: Path):
        """Recovery scenario: vectordb was wiped after the first index.

        Hash still matches the previously indexed file, but the
        collection no longer exists in the vector store.  The framework
        must detect this and force re-index rather than silently skipping.
        """
        idx = _make_indexer(anima_dir)
        idx.vector_store.upsert.return_value = True

        f = anima_dir / "knowledge" / "topic.md"
        f.write_text(self._SAMPLE_MD, encoding="utf-8")

        # First index — collection ends up created
        idx.vector_store.list_collections.return_value = []
        chunks = idx.index_file(f, "knowledge")
        assert chunks > 0
        assert idx.vector_store.upsert.called

        # Simulate vectordb wipe: collection no longer present, but
        # index_meta.json still holds the file hash from the first index.
        idx.vector_store.upsert.reset_mock()
        idx._known_collections = None  # invalidate cache (simulate fresh process)
        idx.vector_store.list_collections.return_value = []  # nothing exists

        chunks = idx.index_file(f, "knowledge")
        assert chunks > 0, "should have re-indexed despite hash match"
        assert idx.vector_store.upsert.called, "must recreate collection via upsert"


class TestIndexConversationSummaryRecovery:
    """Same recovery semantics for ``index_conversation_summary``."""

    def test_force_reindex_when_collection_missing(self, anima_dir: Path):
        idx = _make_indexer(anima_dir)
        idx.vector_store.upsert.return_value = True

        state_dir = anima_dir / "state"
        state_dir.mkdir()
        (state_dir / "conversation.json").write_text(
            '{"compressed_summary": "### Section A\\n\\n' + ("hello world " * 10) + '"}',
            encoding="utf-8",
        )

        # First index
        idx.vector_store.list_collections.return_value = []
        idx._generate_embeddings = MagicMock(return_value=[[0.1] * 4])
        chunks = idx.index_conversation_summary(state_dir, "test_anima")
        assert chunks > 0
        assert idx.vector_store.upsert.called

        # Simulate vectordb wipe → re-index expected
        idx.vector_store.upsert.reset_mock()
        idx._known_collections = None
        idx.vector_store.list_collections.return_value = []
        chunks = idx.index_conversation_summary(state_dir, "test_anima")
        assert chunks > 0
        assert idx.vector_store.upsert.called
