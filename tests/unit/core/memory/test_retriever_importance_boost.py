# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for importance boost in MemoryRetriever._apply_score_adjustments().

Verifies:
- importance=="important" chunks receive WEIGHT_IMPORTANCE (+0.20) boost
- importance=="normal" chunks receive no boost
- Missing importance metadata receives no boost
- Importance boost is recorded in source_scores["importance"]
- Importance boost interacts correctly with other adjustments (recency, frequency)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from core.memory.rag.retriever import (
    WEIGHT_IMPORTANCE,
    MemoryRetriever,
)
from core.memory.rag.store import Document, SearchResult

# ── Mock fixtures ────────────────────────────────────────────────────


class MockVectorStore:
    """Mock vector store returning results with configurable metadata."""

    def __init__(self, items: list[tuple[float, dict]]) -> None:
        self._items = items

    def query(self, collection, embedding, top_k, filter_metadata=None):
        return [
            SearchResult(
                document=Document(
                    id=f"anima/knowledge/doc{i}.md#0",
                    content=f"Content {i}",
                    metadata=meta,
                ),
                score=score,
            )
            for i, (score, meta) in enumerate(self._items[:top_k])
        ]


class MockIndexer:
    def _generate_embeddings(self, texts):
        return [[0.1] * 384 for _ in texts]


class _MockConfigDisabled:
    class _RAG:
        enable_spreading_activation = False
        spreading_memory_types = ["knowledge", "episodes"]

    rag = _RAG()


def _make_retriever(tmp_path: Path, items: list[tuple[float, dict]]) -> MemoryRetriever:
    vector_store = MockVectorStore(items)
    indexer = MockIndexer()
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir(exist_ok=True)
    with patch.object(
        MemoryRetriever,
        "_load_config",
        staticmethod(lambda: _MockConfigDisabled()),
    ):
        return MemoryRetriever(vector_store, indexer, knowledge_dir)


# ── WEIGHT_IMPORTANCE constant ───────────────────────────────────────


class TestWeightImportanceConstant:
    def test_weight_importance_value(self) -> None:
        assert WEIGHT_IMPORTANCE == 0.20


# ── Importance boost applied ─────────────────────────────────────────


class TestImportanceBoostApplied:
    def test_important_chunk_receives_boost(self, tmp_path: Path) -> None:
        """Chunk with importance='important' gets +WEIGHT_IMPORTANCE score."""
        items = [(0.8, {"source_file": "knowledge/lesson.md", "importance": "important"})]
        retriever = _make_retriever(tmp_path, items)

        results = retriever.search("query", "test_anima", memory_type="knowledge", top_k=5)

        assert len(results) == 1
        assert results[0].source_scores.get("importance") == WEIGHT_IMPORTANCE

    def test_normal_chunk_no_boost(self, tmp_path: Path) -> None:
        """Chunk with importance='normal' gets no importance boost."""
        items = [(0.8, {"source_file": "knowledge/doc.md", "importance": "normal"})]
        retriever = _make_retriever(tmp_path, items)

        results = retriever.search("query", "test_anima", memory_type="knowledge", top_k=5)

        assert len(results) == 1
        assert "importance" not in results[0].source_scores

    def test_missing_importance_no_boost(self, tmp_path: Path) -> None:
        """Chunk with no importance metadata gets no importance boost."""
        items = [(0.8, {"source_file": "knowledge/doc.md"})]
        retriever = _make_retriever(tmp_path, items)

        results = retriever.search("query", "test_anima", memory_type="knowledge", top_k=5)

        assert len(results) == 1
        assert "importance" not in results[0].source_scores


# ── Boost affects ranking ────────────────────────────────────────────


class TestImportanceBoostRanking:
    def test_important_chunk_ranks_higher(self, tmp_path: Path) -> None:
        """An important chunk with slightly lower base score outranks normal chunk."""
        items = [
            (0.7, {"source_file": "knowledge/normal.md", "importance": "normal"}),
            (0.6, {"source_file": "knowledge/important.md", "importance": "important"}),
        ]
        retriever = _make_retriever(tmp_path, items)

        results = retriever.search("query", "test_anima", memory_type="knowledge", top_k=5)

        assert len(results) == 2
        scores = {r.doc_id: r.score for r in results}
        important_score = scores["anima/knowledge/doc1.md#0"]
        normal_score = scores["anima/knowledge/doc0.md#0"]
        assert important_score > normal_score, (
            f"Important chunk ({important_score:.3f}) should outrank normal chunk ({normal_score:.3f})"
        )


# ── Boost value correctness ─────────────────────────────────────────


class TestImportanceBoostValue:
    def test_boost_exact_value(self, tmp_path: Path) -> None:
        """Importance boost adds exactly WEIGHT_IMPORTANCE to the score."""
        important_items = [(0.5, {"source_file": "knowledge/a.md", "importance": "important"})]
        normal_items = [(0.5, {"source_file": "knowledge/b.md", "importance": "normal"})]

        retriever_imp = _make_retriever(tmp_path, important_items)
        retriever_norm = _make_retriever(tmp_path, normal_items)

        results_imp = retriever_imp.search("query", "test_anima", memory_type="knowledge", top_k=5)
        results_norm = retriever_norm.search("query", "test_anima", memory_type="knowledge", top_k=5)

        diff = results_imp[0].score - results_norm[0].score
        assert abs(diff - WEIGHT_IMPORTANCE) < 1e-6, (
            f"Score difference ({diff:.6f}) should equal WEIGHT_IMPORTANCE ({WEIGHT_IMPORTANCE})"
        )
