from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Tests for memory access frequency features.

Tests cover:
- Score adjustments with frequency boost in MemoryRetriever
- Access recording via record_access()
- Metadata fields added by MemoryIndexer
- ChromaVectorStore.update_metadata()
"""

import math
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory.rag.retriever import (
    MemoryRetriever,
    RetrievalResult,
    WEIGHT_FREQUENCY,
    WEIGHT_RECENCY,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def mock_vector_store():
    """Create a mock VectorStore instance."""
    store = MagicMock()
    store.update_metadata = MagicMock()
    store.query = MagicMock(return_value=[])
    return store


@pytest.fixture
def retriever(mock_vector_store):
    """Create a MemoryRetriever with mocked dependencies."""
    mock_indexer = MagicMock()
    knowledge_dir = Path("/tmp/test_knowledge")
    return MemoryRetriever(
        vector_store=mock_vector_store,
        indexer=mock_indexer,
        knowledge_dir=knowledge_dir,
    )


def _make_result(
    doc_id: str = "doc1",
    content: str = "test content",
    score: float = 0.5,
    access_count: int = 0,
    updated_at: str | None = None,
    memory_type: str = "knowledge",
    anima: str = "test_anima",
    **extra_metadata,
) -> RetrievalResult:
    """Helper to create a RetrievalResult with common defaults."""
    metadata: dict = {
        "memory_type": memory_type,
        "anima": anima,
        "access_count": access_count,
        "last_accessed_at": "",
    }
    if updated_at is not None:
        metadata["updated_at"] = updated_at
    metadata.update(extra_metadata)
    return RetrievalResult(
        doc_id=doc_id,
        content=content,
        score=score,
        metadata=metadata,
        source_scores={"vector": score},
    )


# ── Score Adjustment Tests ──────────────────────────────────────────


class TestScoreAdjustmentsFrequency:
    """Test _apply_score_adjustments frequency boost component."""

    def test_score_adjustments_with_frequency_boost(self, retriever):
        """Test that _apply_score_adjustments adds both recency AND frequency scores.

        A RetrievalResult with access_count=10 should receive a higher
        frequency boost than one with access_count=0.
        """
        now = datetime.now()
        recent = (now - timedelta(days=1)).isoformat()

        high_access = _make_result(
            doc_id="high", score=0.5, access_count=10, updated_at=recent,
        )
        zero_access = _make_result(
            doc_id="zero", score=0.5, access_count=0, updated_at=recent,
        )

        results = retriever._apply_score_adjustments([high_access, zero_access])

        high_result = next(r for r in results if r.doc_id == "high")
        zero_result = next(r for r in results if r.doc_id == "zero")

        # Both should have recency and frequency scores
        assert "recency" in high_result.source_scores
        assert "frequency" in high_result.source_scores
        assert "recency" in zero_result.source_scores
        assert "frequency" in zero_result.source_scores

        # High-access doc should have a higher frequency boost
        assert high_result.source_scores["frequency"] > zero_result.source_scores["frequency"]

        # Overall score with high access should be higher (same base score + recency)
        assert high_result.score > zero_result.score

    def test_score_adjustments_frequency_zero_access(self, retriever):
        """Test that access_count=0 gives frequency_boost=0 (log1p(0) == 0)."""
        result = _make_result(
            doc_id="zero", score=0.5, access_count=0,
            updated_at=datetime.now().isoformat(),
        )

        adjusted = retriever._apply_score_adjustments([result])

        assert adjusted[0].source_scores["frequency"] == 0.0
        # Verify the math: WEIGHT_FREQUENCY * log1p(0) = WEIGHT_FREQUENCY * 0 = 0
        assert adjusted[0].source_scores["frequency"] == WEIGHT_FREQUENCY * math.log1p(0)

    def test_score_adjustments_frequency_logarithmic_scaling(self, retriever):
        """Test that frequency boost scales logarithmically.

        100 accesses should NOT give 100x the boost of 1 access.
        The ratio should be much smaller due to log scaling.
        """
        now_iso = datetime.now().isoformat()

        one_access = _make_result(
            doc_id="one", score=0.5, access_count=1, updated_at=now_iso,
        )
        hundred_access = _make_result(
            doc_id="hundred", score=0.5, access_count=100, updated_at=now_iso,
        )

        results = retriever._apply_score_adjustments([one_access, hundred_access])

        one_freq = next(r for r in results if r.doc_id == "one").source_scores["frequency"]
        hundred_freq = next(r for r in results if r.doc_id == "hundred").source_scores["frequency"]

        # Verify logarithmic scaling: ratio should be far less than 100
        assert one_freq > 0
        assert hundred_freq > one_freq
        ratio = hundred_freq / one_freq
        assert ratio < 10, (
            f"Expected logarithmic scaling (ratio < 10), got ratio={ratio:.2f}"
        )

        # Verify exact expected values
        expected_one = WEIGHT_FREQUENCY * math.log1p(1)
        expected_hundred = WEIGHT_FREQUENCY * math.log1p(100)
        assert abs(one_freq - expected_one) < 1e-9
        assert abs(hundred_freq - expected_hundred) < 1e-9


# ── Record Access Tests ─────────────────────────────────────────────


class TestRecordAccess:
    """Test record_access() method on MemoryRetriever."""

    def test_record_access_updates_metadata(self, retriever, mock_vector_store):
        """Test that record_access calls update_metadata with correct arguments.

        Should increment access_count and set last_accessed_at to current ISO
        timestamp.
        """
        result = _make_result(
            doc_id="doc1", access_count=3,
            memory_type="knowledge", anima="test_anima",
        )

        fixed_now = datetime(2026, 2, 15, 12, 0, 0)
        with patch("core.memory.rag.retriever.now_iso", return_value=fixed_now.isoformat()):
            retriever.record_access([result], "test_anima")

        mock_vector_store.update_metadata.assert_called_once()
        call_args = mock_vector_store.update_metadata.call_args
        collection_arg = call_args[0][0]
        ids_arg = call_args[0][1]
        metas_arg = call_args[0][2]

        assert collection_arg == "test_anima_knowledge"
        assert ids_arg == ["doc1"]
        assert len(metas_arg) == 1
        assert metas_arg[0]["access_count"] == 4  # 3 + 1
        assert metas_arg[0]["last_accessed_at"] == fixed_now.isoformat()

    def test_record_access_groups_by_collection(self, retriever, mock_vector_store):
        """Test that record_access groups results by collection.

        Results from different memory types should produce separate
        update_metadata calls.
        """
        knowledge_result = _make_result(
            doc_id="k1", access_count=0,
            memory_type="knowledge", anima="test_anima",
        )
        episode_result = _make_result(
            doc_id="e1", access_count=2,
            memory_type="episodes", anima="test_anima",
        )

        retriever.record_access([knowledge_result, episode_result], "test_anima")

        # Should have been called twice: once for knowledge, once for episodes
        assert mock_vector_store.update_metadata.call_count == 2

        # Collect collection names from calls
        collections_called = set()
        for call in mock_vector_store.update_metadata.call_args_list:
            collections_called.add(call[0][0])

        assert "test_anima_knowledge" in collections_called
        assert "test_anima_episodes" in collections_called

    def test_record_access_handles_errors_gracefully(self, retriever, mock_vector_store):
        """Test that record_access doesn't propagate errors.

        When update_metadata raises an exception, record_access should log
        a warning but not re-raise.
        """
        mock_vector_store.update_metadata.side_effect = RuntimeError("DB error")

        result = _make_result(
            doc_id="doc1", access_count=0,
            memory_type="knowledge", anima="test_anima",
        )

        # Should not raise
        retriever.record_access([result], "test_anima")

        # Verify the call was attempted
        mock_vector_store.update_metadata.assert_called_once()

    def test_record_access_empty_results(self, retriever, mock_vector_store):
        """Test that record_access returns early for empty list."""
        retriever.record_access([], "test_anima")

        mock_vector_store.update_metadata.assert_not_called()


# ── Indexer Metadata Tests ──────────────────────────────────────────


class TestIndexerMetadata:
    """Test that MemoryIndexer._extract_metadata includes access fields."""

    def test_indexer_metadata_includes_access_fields(self, tmp_path):
        """Test that _extract_metadata includes access_count, last_accessed_at,
        activation_level, and low_activation_since fields.
        """
        from core.memory.rag.indexer import MemoryIndexer

        anima_dir = tmp_path / "test_anima"
        anima_dir.mkdir()
        knowledge_dir = anima_dir / "knowledge"
        knowledge_dir.mkdir()

        test_file = knowledge_dir / "test.md"
        test_file.write_text("# Test\n\nSome content here.", encoding="utf-8")

        # Create indexer with a mock vector store and mock embedding model
        mock_store = MagicMock()
        mock_embedding_model = MagicMock()

        indexer = MemoryIndexer(
            vector_store=mock_store,
            anima_name="test_anima",
            anima_dir=anima_dir,
            embedding_model=mock_embedding_model,
        )

        metadata = indexer._extract_metadata(
            file_path=test_file,
            content="# Test\n\nSome content here.",
            memory_type="knowledge",
            chunk_index=0,
            total_chunks=1,
        )

        # Verify access frequency fields
        assert metadata["access_count"] == 0
        assert metadata["last_accessed_at"] == ""

        # Verify activation level fields
        assert metadata["activation_level"] == "normal"
        assert metadata["low_activation_since"] == ""

        # Verify standard fields are also present
        assert metadata["memory_type"] == "knowledge"
        assert metadata["anima"] == "test_anima"
        assert "importance" in metadata


# ── ChromaVectorStore.update_metadata Integration Test ──────────────


class TestUpdateMetadataOnVectorStore:
    """Integration test for ChromaVectorStore.update_metadata."""

    def test_update_metadata_on_vector_store(self, tmp_path):
        """Test upserting a document, updating its metadata, and verifying.

        Creates a ChromaVectorStore, upserts a document, calls
        update_metadata to change a field, then queries back and verifies
        the metadata was updated.
        """
        chromadb = pytest.importorskip(
            "chromadb",
            reason="ChromaDB not installed. Install with: pip install 'animaworks[rag]'",
        )

        from core.memory.rag.store import ChromaVectorStore, Document

        vectordb_dir = tmp_path / "vectordb"
        vectordb_dir.mkdir()

        store = ChromaVectorStore(persist_dir=vectordb_dir)
        store.create_collection("test_col", dimension=3)

        # Upsert a document
        doc = Document(
            id="doc1",
            content="Test document",
            embedding=[0.1, 0.2, 0.3],
            metadata={
                "access_count": 0,
                "last_accessed_at": "",
                "activation_level": "normal",
            },
        )
        store.upsert("test_col", [doc])

        # Verify initial metadata
        results = store.query("test_col", embedding=[0.1, 0.2, 0.3], top_k=1)
        assert len(results) == 1
        assert results[0].document.metadata["access_count"] == 0

        # Update metadata
        new_ts = "2026-02-15T12:00:00"
        store.update_metadata(
            "test_col",
            ids=["doc1"],
            metadatas=[{
                "access_count": 5,
                "last_accessed_at": new_ts,
            }],
        )

        # Query back and verify
        results = store.query("test_col", embedding=[0.1, 0.2, 0.3], top_k=1)
        assert len(results) == 1
        assert results[0].document.metadata["access_count"] == 5
        assert results[0].document.metadata["last_accessed_at"] == new_ts
        # Unchanged field should still be present
        assert results[0].document.metadata["activation_level"] == "normal"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
