# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for min_score parameter in MemoryRetriever.search().

Verifies:
- min_score=None returns all results (default behavior)
- min_score filters out results with vector score < threshold
- min_score keeps results with vector score >= threshold
- min_score that filters ALL results returns empty list
"""

from __future__ import annotations

from pathlib import Path

from core.memory.rag.retriever import MemoryRetriever
from core.memory.rag.store import Document, SearchResult

# ── Mock fixtures ────────────────────────────────────────────────────


class MockVectorStoreWithScores:
    """Mock vector store that returns configurable scores."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def query(self, collection, embedding, top_k, filter_metadata=None):
        return [
            SearchResult(
                document=Document(
                    id=f"anima/knowledge/doc{i}.md#0",
                    content=f"Content {i}",
                    metadata={"source_file": f"knowledge/doc{i}.md"},
                ),
                score=score,
            )
            for i, score in enumerate(self._scores[:top_k])
        ]


class MockIndexer:
    """Mock indexer for embedding generation."""

    def _generate_embeddings(self, texts, **_kwargs):
        return [[0.1] * 384 for _ in texts]


# ── min_score=None returns all ────────────────────────────────────────


class TestMinScoreNone:
    def test_min_score_none_returns_all_results(
        self,
        tmp_path: Path,
    ) -> None:
        """search() with min_score=None returns all results (default behavior)."""
        vector_store = MockVectorStoreWithScores([0.9, 0.5, 0.2])
        indexer = MockIndexer()
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()

        retriever = MemoryRetriever(vector_store, indexer, knowledge_dir)

        results = retriever.search(
            "query",
            "test_anima",
            memory_type="knowledge",
            top_k=5,
            min_score=None,
        )

        assert len(results) == 3
        assert [r.source_scores.get("vector") for r in results] == [0.9, 0.5, 0.2]


# ── min_score filters low scores ─────────────────────────────────────


class TestMinScoreFilters:
    def test_min_score_filters_out_low_results(
        self,
        tmp_path: Path,
    ) -> None:
        """search() with min_score=0.3 filters out results with vector score < 0.3."""
        vector_store = MockVectorStoreWithScores([0.9, 0.5, 0.2, 0.1])
        indexer = MockIndexer()
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()

        retriever = MemoryRetriever(vector_store, indexer, knowledge_dir)

        results = retriever.search(
            "query",
            "test_anima",
            memory_type="knowledge",
            top_k=5,
            min_score=0.3,
        )

        assert len(results) == 2
        assert all(r.source_scores.get("vector", 0) >= 0.3 for r in results)

    def test_min_score_keeps_results_at_threshold(
        self,
        tmp_path: Path,
    ) -> None:
        """search() with min_score=0.3 keeps results with vector score >= 0.3."""
        vector_store = MockVectorStoreWithScores([0.9, 0.3, 0.299])
        indexer = MockIndexer()
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()

        retriever = MemoryRetriever(vector_store, indexer, knowledge_dir)

        results = retriever.search(
            "query",
            "test_anima",
            memory_type="knowledge",
            top_k=5,
            min_score=0.3,
        )

        assert len(results) == 2
        scores = [r.source_scores.get("vector") for r in results]
        assert 0.9 in scores
        assert 0.3 in scores
        assert 0.299 not in scores

    def test_min_score_filters_all_returns_empty(
        self,
        tmp_path: Path,
    ) -> None:
        """search() with min_score that filters ALL results returns empty list."""
        vector_store = MockVectorStoreWithScores([0.1, 0.2])
        indexer = MockIndexer()
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()

        retriever = MemoryRetriever(vector_store, indexer, knowledge_dir)

        results = retriever.search(
            "query",
            "test_anima",
            memory_type="knowledge",
            top_k=5,
            min_score=0.5,
        )

        assert results == []
