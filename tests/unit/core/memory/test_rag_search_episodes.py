from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for episodes scope in RAGMemorySearch vector search."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory.rag_search import RAGMemorySearch


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "alice"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def common_knowledge_dir(tmp_path: Path) -> Path:
    d = tmp_path / "common_knowledge"
    d.mkdir()
    return d


@pytest.fixture
def common_skills_dir(tmp_path: Path) -> Path:
    d = tmp_path / "common_skills"
    d.mkdir()
    return d


@pytest.fixture
def rag(
    anima_dir: Path,
    common_knowledge_dir: Path,
    common_skills_dir: Path,
) -> RAGMemorySearch:
    return RAGMemorySearch(anima_dir, common_knowledge_dir, common_skills_dir)


@pytest.fixture
def knowledge_dir(anima_dir: Path) -> Path:
    d = anima_dir / "knowledge"
    d.mkdir()
    return d


@pytest.fixture
def episodes_dir(anima_dir: Path) -> Path:
    d = anima_dir / "episodes"
    d.mkdir()
    return d


@pytest.fixture
def procedures_dir(anima_dir: Path) -> Path:
    d = anima_dir / "procedures"
    d.mkdir()
    return d


# ── _resolve_search_types ────────────────────────────────


class TestResolveSearchTypes:
    def test_episodes_scope(self) -> None:
        assert RAGMemorySearch._resolve_search_types("episodes") == ["episodes"]

    def test_all_scope_includes_episodes(self) -> None:
        types = RAGMemorySearch._resolve_search_types("all")
        assert "episodes" in types
        assert "knowledge" in types
        assert "procedures" in types
        assert "skills" in types
        assert "conversation_summary" in types

    def test_knowledge_scope(self) -> None:
        assert RAGMemorySearch._resolve_search_types("knowledge") == ["knowledge"]

    def test_procedures_scope(self) -> None:
        assert RAGMemorySearch._resolve_search_types("procedures") == ["procedures"]


# ── search_memory_text with episodes ─────────────────────


class TestSearchMemoryTextEpisodesKeyword:
    def test_keyword_search_includes_episodes(
        self,
        rag: RAGMemorySearch,
        knowledge_dir: Path,
        episodes_dir: Path,
        procedures_dir: Path,
        common_knowledge_dir: Path,
    ) -> None:
        """scope='episodes' keyword fallback only returns episode files."""
        (knowledge_dir / "python.md").write_text(
            "Python keyword", encoding="utf-8",
        )
        (episodes_dir / "2026-03-01.md").write_text(
            "Deployed service with keyword fix",
            encoding="utf-8",
        )

        with patch.object(rag, "_get_indexer", return_value=None):
            results = rag.search_memory_text(
                "keyword",
                scope="episodes",
                knowledge_dir=knowledge_dir,
                episodes_dir=episodes_dir,
                procedures_dir=procedures_dir,
                common_knowledge_dir=common_knowledge_dir,
            )

        sources = [r["source_file"] for r in results]
        assert any("2026-03-01.md" in s for s in sources)
        assert not any("python.md" in s for s in sources)


class TestSearchMemoryTextEpisodesVector:
    def test_vector_search_augments_episodes(
        self,
        rag: RAGMemorySearch,
        knowledge_dir: Path,
        episodes_dir: Path,
        procedures_dir: Path,
        common_knowledge_dir: Path,
    ) -> None:
        """Vector search augmentation should run for episodes scope."""
        mock_indexer = MagicMock()
        rag._indexer = mock_indexer
        rag._indexer_initialized = True

        (episodes_dir / "2026-03-01.md").write_text(
            "No match here", encoding="utf-8",
        )

        with patch.object(
            rag,
            "_vector_search_primary",
            return_value=[
                {
                    "source_file": "episodes/2026-02-20.md",
                    "content": "vector hit line",
                    "score": 0.9,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "memory_type": "episodes",
                    "search_method": "vector",
                },
            ],
        ) as mock_vs:
            results = rag.search_memory_text(
                "deploy error",
                scope="episodes",
                knowledge_dir=knowledge_dir,
                episodes_dir=episodes_dir,
                procedures_dir=procedures_dir,
                common_knowledge_dir=common_knowledge_dir,
            )

        mock_vs.assert_called_once()
        sources = [r["source_file"] for r in results]
        assert any("episodes/2026-02-20.md" in s for s in sources)

    def test_vector_search_all_scope_includes_episodes(
        self,
        rag: RAGMemorySearch,
        knowledge_dir: Path,
        episodes_dir: Path,
        procedures_dir: Path,
        common_knowledge_dir: Path,
    ) -> None:
        """scope='all' vector search should iterate through episodes type."""
        mock_indexer = MagicMock()
        rag._indexer = mock_indexer
        rag._indexer_initialized = True

        with patch.object(rag, "_vector_search_primary", return_value=[]) as mock_vs:
            rag.search_memory_text(
                "query",
                scope="all",
                knowledge_dir=knowledge_dir,
                episodes_dir=episodes_dir,
                procedures_dir=procedures_dir,
                common_knowledge_dir=common_knowledge_dir,
            )

        mock_vs.assert_called_once()
