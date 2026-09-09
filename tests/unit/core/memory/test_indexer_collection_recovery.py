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

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory.rag.store import CollectionExistence


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
    idx.vector_store.create_collection.return_value = True
    idx.vector_store.list_collections_checked.side_effect = lambda: _checked_collections(idx.vector_store)
    return idx


def _checked_collections(vector_store):
    try:
        return vector_store.list_collections()
    except Exception:
        return None


class TestCollectionExistenceCache:
    """Verify the lazy-init collection-existence cache."""

    def test_first_call_populates_cache(self, anima_dir: Path):
        idx = _make_indexer(anima_dir)
        idx.vector_store.list_collections.return_value = ["foo", "bar"]
        assert idx._collection_exists("foo") is CollectionExistence.EXISTS
        assert idx.vector_store.list_collections.call_count == 1
        # Second call uses cache
        assert idx._collection_exists("bar") is CollectionExistence.EXISTS
        assert idx.vector_store.list_collections.call_count == 1

    def test_missing_returns_false(self, anima_dir: Path):
        idx = _make_indexer(anima_dir)
        idx.vector_store.list_collections.return_value = ["foo"]
        assert idx._collection_exists("missing") is CollectionExistence.MISSING

    def test_listing_failure_is_conservative(self, anima_dir: Path):
        """When list_collections raises, we should NOT trigger spurious re-index."""
        idx = _make_indexer(anima_dir)
        idx.vector_store.list_collections.side_effect = RuntimeError("transient")
        # Be conservative: assume the collection exists rather than
        # forcing a full re-index of everything on a transient error.
        assert idx._collection_exists("any") is CollectionExistence.UNAVAILABLE

    def test_mark_known_adds_to_cache(self, anima_dir: Path):
        idx = _make_indexer(anima_dir)
        idx.vector_store.list_collections.return_value = []
        # Initially not present
        assert idx._collection_exists("new") is CollectionExistence.MISSING
        # After marking known, present without re-listing
        idx._mark_collection_known("new")
        assert idx._collection_exists("new") is CollectionExistence.EXISTS


class TestIndexFileCollectionRecovery:
    """Verify ``index_file`` re-indexes when collection is missing.

    The bug: when the vectordb was wiped but per-anima ``index_meta.json``
    retained file hashes, ``index_file`` would skip indexing forever and
    the collection would never be recreated.
    """

    # Markdown content with a ``## `` heading so chunker yields chunks.
    _SAMPLE_MD = "# Title\n\n## Section A\n\n" + ("body sentence with enough length to chunk. " * 5)

    def test_directory_write_gate_stops_before_embedding(self, anima_dir: Path):
        idx = _make_indexer(anima_dir)
        files = [anima_dir / "knowledge" / f"{name}.md" for name in ("a", "b", "c")]
        for file_path in files:
            file_path.write_text(self._SAMPLE_MD, encoding="utf-8")
        idx.vector_store.create_collection.return_value = False
        idx._generate_embeddings.reset_mock()
        idx._reconcile_stale_entries = MagicMock(return_value=0)
        idx._save_index_meta = MagicMock()

        result = idx.index_directory(anima_dir / "knowledge", "knowledge")

        assert result.files_failed == 1
        assert result.transient_failures == 1
        assert result.files_unprocessed == 2
        assert result.files_reconciled == 0
        assert result.transient is True
        assert result.failed_sources == ("a.md",)
        idx.vector_store.create_collection.assert_called_once_with("test_anima_knowledge")
        idx._generate_embeddings.assert_not_called()
        idx.vector_store.get_by_metadata.assert_not_called()
        idx.vector_store.upsert.assert_not_called()
        idx._reconcile_stale_entries.assert_not_called()
        idx._save_index_meta.assert_not_called()
        assert all(str(file_path.relative_to(anima_dir)) not in idx.index_meta for file_path in files)

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

    def test_unavailable_collection_check_does_not_reindex_or_embed(self, anima_dir: Path):
        idx = _make_indexer(anima_dir)
        idx.vector_store.upsert.return_value = True
        f = anima_dir / "knowledge" / "topic.md"
        f.write_text(self._SAMPLE_MD, encoding="utf-8")

        assert idx.index_file(f, "knowledge") > 0
        idx._known_collections = None
        idx.vector_store.list_collections_checked.side_effect = None
        idx.vector_store.list_collections_checked.return_value = None
        idx.vector_store.upsert.reset_mock()
        idx._generate_embeddings.reset_mock()

        assert idx.index_file(f, "knowledge") == 0
        assert idx._last_index_file_outcome.status == "failed"
        assert idx._last_index_file_outcome.transient is True
        idx._generate_embeddings.assert_not_called()
        idx.vector_store.upsert.assert_not_called()

    def test_directory_unavailable_collection_is_retryable_not_unchanged(self, anima_dir: Path):
        idx = _make_indexer(anima_dir)
        idx.vector_store.upsert.return_value = True
        for name in ("a", "b"):
            file_path = anima_dir / "knowledge" / f"{name}.md"
            file_path.write_text(self._SAMPLE_MD, encoding="utf-8")
            assert idx.index_file(file_path, "knowledge") > 0
        idx._known_collections = None
        idx.vector_store.list_collections_checked.side_effect = None
        idx.vector_store.list_collections_checked.return_value = None
        idx._generate_embeddings.reset_mock()
        idx._reconcile_stale_entries = MagicMock()

        result = idx.index_directory(anima_dir / "knowledge", "knowledge")

        assert result.files_failed == 1
        assert result.transient_failures == 1
        assert result.files_unprocessed == 1
        assert result.files_unchanged == 0
        assert result.transient is True
        idx._generate_embeddings.assert_not_called()
        idx._reconcile_stale_entries.assert_not_called()


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

    def test_create_failure_skips_embedding_and_metadata_update(self, anima_dir: Path):
        idx = _make_indexer(anima_dir)
        state_dir = anima_dir / "state"
        state_dir.mkdir()
        (state_dir / "conversation.json").write_text(
            '{"compressed_summary": "### Section A\\n\\n' + ("hello world " * 10) + '"}',
            encoding="utf-8",
        )
        idx.vector_store.create_collection.return_value = False
        idx._generate_embeddings.reset_mock()
        idx._save_index_meta = MagicMock()

        chunks = idx.index_conversation_summary(state_dir, "test_anima")

        assert chunks == 0
        assert idx._last_index_file_outcome.status == "failed"
        assert idx._last_index_file_outcome.transient is True
        idx.vector_store.create_collection.assert_called_once_with("test_anima_conversation_summary")
        idx._generate_embeddings.assert_not_called()
        idx.vector_store.upsert.assert_not_called()
        idx._save_index_meta.assert_not_called()
        assert "conversation_summary" not in idx.index_meta

    @pytest.mark.parametrize("contents", [None, {}, {"compressed_summary": "short"}])
    def test_skipped_summary_replaces_previous_outcome(self, anima_dir: Path, contents):
        idx = _make_indexer(anima_dir)
        idx._finish_index_file(0, "failed", transient=True)
        state_dir = anima_dir / "state"
        state_dir.mkdir()
        if contents is not None:
            (state_dir / "conversation.json").write_text(json.dumps(contents), encoding="utf-8")

        assert idx.index_conversation_summary(state_dir, "test_anima") == 0
        assert idx._last_index_file_outcome.status == "skipped"
        assert idx._last_index_file_outcome.transient is False
        idx._generate_embeddings.assert_not_called()
        idx.vector_store.upsert.assert_not_called()

    def test_malformed_summary_reports_failure(self, anima_dir: Path):
        idx = _make_indexer(anima_dir)
        idx._finish_index_file(1, "indexed")
        state_dir = anima_dir / "state"
        state_dir.mkdir()
        (state_dir / "conversation.json").write_text("{invalid", encoding="utf-8")

        assert idx.index_conversation_summary(state_dir, "test_anima") == 0
        assert idx._last_index_file_outcome.status == "failed"
        assert idx._last_index_file_outcome.transient is False
        idx._generate_embeddings.assert_not_called()

    @pytest.mark.parametrize("available", [True, False])
    def test_unchanged_summary_requires_confirmed_collection(self, anima_dir: Path, available: bool):
        idx = _make_indexer(anima_dir)
        state_dir = anima_dir / "state"
        state_dir.mkdir()
        (state_dir / "conversation.json").write_text(
            json.dumps({"compressed_summary": "### Section A\n\n" + "hello world " * 10}), encoding="utf-8"
        )
        idx.vector_store.upsert.return_value = True
        assert idx.index_conversation_summary(state_dir, "test_anima") == 1
        assert idx._last_index_file_outcome.status == "indexed"
        idx._known_collections = None
        idx.vector_store.list_collections_checked.side_effect = None
        idx.vector_store.list_collections_checked.return_value = (
            ["test_anima_conversation_summary"] if available else None
        )
        idx._generate_embeddings.reset_mock()
        idx.vector_store.upsert.reset_mock()

        assert idx.index_conversation_summary(state_dir, "test_anima") == 0
        assert idx._last_index_file_outcome.status == ("unchanged" if available else "failed")
        assert idx._last_index_file_outcome.transient is (not available)
        idx._generate_embeddings.assert_not_called()
        idx.vector_store.upsert.assert_not_called()

    @pytest.mark.parametrize("transient", [True, False])
    def test_upsert_failure_reports_outcome_without_committing_hash(self, anima_dir: Path, transient: bool):
        idx = _make_indexer(anima_dir)
        state_dir = anima_dir / "state"
        state_dir.mkdir()
        (state_dir / "conversation.json").write_text(
            json.dumps({"compressed_summary": "### Section A\n\n" + "hello world " * 10}), encoding="utf-8"
        )
        idx.vector_store.upsert.return_value = False
        idx.vector_store.is_transient_write_failure.return_value = transient
        idx._save_index_meta = MagicMock()

        assert idx.index_conversation_summary(state_dir, "test_anima") == 0
        assert idx._last_index_file_outcome.status == "failed"
        assert idx._last_index_file_outcome.transient is transient
        idx._save_index_meta.assert_not_called()
        assert "conversation_summary" not in idx.index_meta

    def test_embedding_exception_cannot_reuse_previous_success_outcome(self, anima_dir: Path):
        idx = _make_indexer(anima_dir)
        idx._finish_index_file(1, "indexed")
        state_dir = anima_dir / "state"
        state_dir.mkdir()
        (state_dir / "conversation.json").write_text(
            json.dumps({"compressed_summary": "### Section A\n\n" + "hello world " * 10}), encoding="utf-8"
        )
        idx._generate_embeddings.side_effect = RuntimeError("embedding unavailable")

        with pytest.raises(RuntimeError, match="embedding unavailable"):
            idx.index_conversation_summary(state_dir, "test_anima")
        assert idx._last_index_file_outcome.status == "failed"
        assert idx._last_index_file_outcome.transient is False
        idx.vector_store.upsert.assert_not_called()

    def test_empty_chunk_result_is_skipped_not_failed(self, anima_dir: Path):
        idx = _make_indexer(anima_dir)
        state_dir = anima_dir / "state"
        state_dir.mkdir()
        (state_dir / "conversation.json").write_text(
            json.dumps({"compressed_summary": "### Section A\n\n" + "hello world " * 10}), encoding="utf-8"
        )
        idx._chunk_markdown_text = MagicMock(return_value=[])

        assert idx.index_conversation_summary(state_dir, "test_anima") == 0
        assert idx._last_index_file_outcome.status == "skipped"
        assert idx._last_index_file_outcome.transient is False
        idx.vector_store.create_collection.assert_not_called()
        idx._generate_embeddings.assert_not_called()
