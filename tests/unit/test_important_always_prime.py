# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for always-prime [IMPORTANT] knowledge channel (C0).

Tests ChromaVectorStore.get_by_metadata, MemoryRetriever.get_important_chunks,
PrimingEngine._channel_c0_important_knowledge, and _extract_summary.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory.priming import (
    _BUDGET_IMPORTANT_KNOWLEDGE,
    _CHARS_PER_TOKEN,
    PrimingEngine,
)
from core.memory.rag.retriever import MemoryRetriever
from core.memory.rag.store import ChromaVectorStore, Document, SearchResult

# ── ChromaVectorStore.get_by_metadata ────────────────────────────────────────


class TestChromaVectorStoreGetByMetadata:
    """ChromaVectorStore.get_by_metadata filters by importance: important."""

    def test_get_by_metadata_importance_filter(self):
        """get_by_metadata with importance: important returns matching documents."""
        with patch("chromadb.PersistentClient"):
            store = ChromaVectorStore.__new__(ChromaVectorStore)
            store.client = MagicMock()
            store.persist_dir = MagicMock()

            mock_coll = MagicMock()
            mock_coll.get.return_value = {
                "ids": ["chunk-1", "chunk-2"],
                "documents": [
                    "Content for important chunk 1",
                    "Content for important chunk 2",
                ],
                "metadatas": [
                    {"importance": "important", "source_file": "rule-a.md"},
                    {"importance": "important", "source_file": "rule-b.md"},
                ],
            }
            store.client.get_collection.return_value = mock_coll

            results = store.get_by_metadata(
                "test_knowledge",
                {"importance": "important"},
                limit=20,
            )

            assert len(results) == 2
            mock_coll.get.assert_called_once()
            call_kwargs = mock_coll.get.call_args[1]
            assert call_kwargs["where"] == {"importance": "important"}
            assert call_kwargs["limit"] == 20
            assert results[0].document.metadata.get("importance") == "important"
            assert results[1].document.metadata.get("importance") == "important"

    def test_get_by_metadata_collection_not_found_returns_empty(self):
        """get_by_metadata returns [] when collection does not exist."""
        with patch("chromadb.PersistentClient"):
            store = ChromaVectorStore.__new__(ChromaVectorStore)
            store.client = MagicMock()
            store.persist_dir = MagicMock()
            store.client.get_collection.side_effect = Exception("Collection not found")

            results = store.get_by_metadata(
                "nonexistent_collection",
                {"importance": "important"},
                limit=20,
            )

            assert results == []


# ── MemoryRetriever.get_important_chunks ────────────────────────────────────


class TestMemoryRetrieverGetImportantChunks:
    """MemoryRetriever.get_important_chunks merges personal + shared with dedup."""

    def test_get_important_chunks_merge_and_dedup(self):
        """Personal and shared chunks are merged; duplicates by id are excluded."""
        mock_store = MagicMock()
        mock_indexer = MagicMock()

        doc_p1 = Document(
            id="p1",
            content="Personal important",
            metadata={"importance": "important", "source_file": "p.md"},
        )
        doc_p2 = Document(
            id="p2",
            content="Personal important 2",
            metadata={"importance": "important", "source_file": "p2.md"},
        )
        doc_s1 = Document(
            id="s1",
            content="Shared important",
            metadata={"importance": "important", "source_file": "shared/x.md"},
        )
        doc_dup = Document(
            id="p1",
            content="Same id as personal",
            metadata={"importance": "important", "source_file": "shared/dup.md"},
        )

        mock_store.get_by_metadata.side_effect = [
            [SearchResult(document=doc_p1, score=1.0), SearchResult(document=doc_p2, score=1.0)],
            [
                SearchResult(document=doc_s1, score=1.0),
                SearchResult(document=doc_dup, score=1.0),
            ],
        ]

        retriever = MemoryRetriever(
            vector_store=mock_store,
            indexer=mock_indexer,
            knowledge_dir=Path("/tmp/knowledge"),
        )

        results = retriever.get_important_chunks("test_anima", include_shared=True)

        assert len(results) == 3
        ids = [r.document.id for r in results]
        assert "p1" in ids
        assert "p2" in ids
        assert "s1" in ids
        assert ids.count("p1") == 1

        mock_store.get_by_metadata.assert_any_call(
            "test_anima_knowledge",
            {"importance": "important"},
            limit=20,
        )
        mock_store.get_by_metadata.assert_any_call(
            "shared_common_knowledge",
            {"importance": "important"},
            limit=20,
        )

    def test_get_important_chunks_personal_only_when_include_shared_false(self):
        """When include_shared=False, only personal collection is queried."""
        mock_store = MagicMock()
        mock_store.get_by_metadata.return_value = []

        retriever = MemoryRetriever(
            vector_store=mock_store,
            indexer=MagicMock(),
            knowledge_dir=Path("/tmp/knowledge"),
        )

        retriever.get_important_chunks("test_anima", include_shared=False)

        assert mock_store.get_by_metadata.call_count == 1
        mock_store.get_by_metadata.assert_called_once_with(
            "test_anima_knowledge",
            {"importance": "important"},
            limit=20,
        )


# ── PrimingEngine._channel_c0_important_knowledge ────────────────────────────


@pytest.fixture
def temp_anima_dir():
    """Create a temporary anima directory with knowledge subdir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        anima_dir = Path(tmpdir) / "animas" / "test"
        (anima_dir / "knowledge").mkdir(parents=True)
        (anima_dir / "episodes").mkdir(parents=True)
        (anima_dir / "skills").mkdir(parents=True)
        yield anima_dir


class TestChannelC0ImportantKnowledge:
    """PrimingEngine._channel_c0_important_knowledge output format and budget."""

    @pytest.mark.asyncio
    async def test_output_pointer_format(self, temp_anima_dir):
        """Output is summary pointer format with read_memory_file path."""
        doc = Document(
            id="c1",
            content="# My Important Rule\n\nDetails here.",
            metadata={
                "importance": "important",
                "source_file": "knowledge/rule-a.md",
                "anima": "test",
            },
        )

        mock_retriever = MagicMock()
        mock_retriever.get_important_chunks.return_value = [SearchResult(document=doc, score=1.0)]

        with patch.object(
            PrimingEngine,
            "_get_or_create_retriever",
            return_value=mock_retriever,
        ):
            engine = PrimingEngine(temp_anima_dir)
            result = await engine._channel_c0_important_knowledge()

        assert "### [IMPORTANT] Knowledge (summary pointers)" in result
        assert "My Important Rule" in result
        assert 'read_memory_file(path="knowledge/rule-a.md")' in result
        assert "📌" in result

    @pytest.mark.asyncio
    async def test_budget_trims_when_over_300_tokens(self, temp_anima_dir):
        """When total exceeds 300 tokens (~1200 chars), output is trimmed."""
        budget_chars = _BUDGET_IMPORTANT_KNOWLEDGE * _CHARS_PER_TOKEN
        assert budget_chars == 1200

        long_summary = "A" * 100
        docs = []
        for i in range(30):
            rel_path = f"knowledge/rule-{i}.md"
            doc = Document(
                id=f"c{i}",
                content="# " + long_summary,
                metadata={
                    "importance": "important",
                    "source_file": rel_path,
                    "anima": "test",
                },
            )
            docs.append(SearchResult(document=doc, score=1.0))

        mock_retriever = MagicMock()
        mock_retriever.get_important_chunks.return_value = docs

        with patch.object(
            PrimingEngine,
            "_get_or_create_retriever",
            return_value=mock_retriever,
        ):
            engine = PrimingEngine(temp_anima_dir)
            result = await engine._channel_c0_important_knowledge()

        lines = result.split("\n")
        content_lines = [line for line in lines if line.startswith("📌")]
        total_len = len(result)
        assert total_len <= budget_chars + 50
        assert len(content_lines) < 30

    @pytest.mark.asyncio
    async def test_empty_when_no_knowledge_dir(self, tmp_path):
        """Returns empty string when knowledge dir does not exist."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        (anima_dir / "episodes").mkdir()
        (anima_dir / "skills").mkdir()

        engine = PrimingEngine(anima_dir)
        result = await engine._channel_c0_important_knowledge()

        assert result == ""


# ── _extract_summary ─────────────────────────────────────────────────────────


class TestExtractSummary:
    """_extract_summary fallback: summary → # heading → filename."""

    def test_prefers_summary_field(self, tmp_path):
        """summary metadata is used when present."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        (anima_dir / "knowledge").mkdir()
        (anima_dir / "episodes").mkdir()
        (anima_dir / "skills").mkdir()
        engine = PrimingEngine(anima_dir)
        content = "Some content"
        meta = {"summary": "Custom summary text"}
        assert engine._extract_summary(content, meta) == "Custom summary text"

    def test_fallback_to_h1_heading(self, tmp_path):
        """First # heading is used when summary is absent."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        (anima_dir / "knowledge").mkdir()
        (anima_dir / "episodes").mkdir()
        (anima_dir / "skills").mkdir()
        engine = PrimingEngine(anima_dir)
        content = "# Main Title\n\nBody text"
        meta = {}
        assert engine._extract_summary(content, meta) == "Main Title"

    def test_fallback_to_filename_stem(self, tmp_path):
        """source_file stem is used when no summary or heading."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        (anima_dir / "knowledge").mkdir()
        (anima_dir / "episodes").mkdir()
        (anima_dir / "skills").mkdir()
        engine = PrimingEngine(anima_dir)
        content = "Plain content without heading"
        meta = {"source_file": "knowledge/my-rule-file.md"}
        result = engine._extract_summary(content, meta)
        assert result == "my rule file"

    def test_empty_when_no_source(self, tmp_path):
        """Returns empty when no summary, heading, or source_file."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        (anima_dir / "knowledge").mkdir()
        (anima_dir / "episodes").mkdir()
        (anima_dir / "skills").mkdir()
        engine = PrimingEngine(anima_dir)
        content = "Plain content"
        meta = {}
        assert engine._extract_summary(content, meta) == ""
