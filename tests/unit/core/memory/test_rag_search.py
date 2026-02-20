"""Unit tests for core/memory/rag_search.py — RAGMemorySearch."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory.rag_search import RAGMemorySearch


# ── Fixtures ─────────────────────────────────────────────


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


# ── _get_indexer / _init_indexer ─────────────────────────


class TestGetIndexerLazyInit:
    def test_get_indexer_lazy_init(self, rag: RAGMemorySearch) -> None:
        """_get_indexer() calls _init_indexer() on first call."""
        assert rag._indexer_initialized is False

        with patch.object(rag, "_init_indexer") as mock_init:
            rag._get_indexer()
            mock_init.assert_called_once()

    def test_get_indexer_only_inits_once(self, rag: RAGMemorySearch) -> None:
        """Second call to _get_indexer() does not call _init_indexer() again."""
        def _fake_init():
            rag._indexer_initialized = True

        with patch.object(rag, "_init_indexer", side_effect=_fake_init) as mock_init:
            rag._get_indexer()
            rag._get_indexer()
            mock_init.assert_called_once()


class TestGetIndexerDependencyMissing:
    def test_get_indexer_returns_none_when_deps_missing(
        self, rag: RAGMemorySearch,
    ) -> None:
        """When ImportError is raised, indexer stays None."""
        def _simulate_import_error():
            rag._indexer_initialized = True
            # indexer stays None — no assignment

        with patch.object(rag, "_init_indexer", side_effect=_simulate_import_error):
            result = rag._get_indexer()

        assert result is None
        assert rag._indexer is None


# ── search_memory_text ───────────────────────────────────


class TestSearchMemoryTextKeywordOnly:
    def test_search_memory_text_keyword_only(
        self,
        rag: RAGMemorySearch,
        knowledge_dir: Path,
        episodes_dir: Path,
        procedures_dir: Path,
        common_knowledge_dir: Path,
    ) -> None:
        """Keyword search works without RAG (indexer is None)."""
        (knowledge_dir / "python.md").write_text(
            "Python is a great language\nJava is also fine",
            encoding="utf-8",
        )
        (episodes_dir / "2026-01-01.md").write_text(
            "Learned Python today",
            encoding="utf-8",
        )

        assert rag._indexer is None

        results = rag.search_memory_text(
            "python",
            scope="all",
            knowledge_dir=knowledge_dir,
            episodes_dir=episodes_dir,
            procedures_dir=procedures_dir,
            common_knowledge_dir=common_knowledge_dir,
        )

        assert len(results) >= 2
        filenames = [r[0] for r in results]
        assert "python.md" in filenames
        assert "2026-01-01.md" in filenames


class TestSearchMemoryTextRespectsScope:
    def test_search_memory_text_respects_scope(
        self,
        rag: RAGMemorySearch,
        knowledge_dir: Path,
        episodes_dir: Path,
        procedures_dir: Path,
        common_knowledge_dir: Path,
    ) -> None:
        """scope='knowledge' only searches the knowledge directory."""
        (knowledge_dir / "topic.md").write_text(
            "keyword in knowledge", encoding="utf-8",
        )
        (episodes_dir / "2026-02-01.md").write_text(
            "keyword in episodes", encoding="utf-8",
        )
        (procedures_dir / "deploy.md").write_text(
            "keyword in procedures", encoding="utf-8",
        )

        results = rag.search_memory_text(
            "keyword",
            scope="knowledge",
            knowledge_dir=knowledge_dir,
            episodes_dir=episodes_dir,
            procedures_dir=procedures_dir,
            common_knowledge_dir=common_knowledge_dir,
        )

        filenames = [r[0] for r in results]
        assert "topic.md" in filenames
        assert "2026-02-01.md" not in filenames
        assert "deploy.md" not in filenames


class TestSearchMemoryTextEmptyQuery:
    def test_search_memory_text_empty_query(
        self,
        rag: RAGMemorySearch,
        knowledge_dir: Path,
        episodes_dir: Path,
        procedures_dir: Path,
        common_knowledge_dir: Path,
    ) -> None:
        """Returns matching lines for empty-ish queries.

        An empty string matches every line via ``"" in line.lower()``.
        """
        (knowledge_dir / "info.md").write_text(
            "Line one\nLine two", encoding="utf-8",
        )

        results = rag.search_memory_text(
            "",
            scope="knowledge",
            knowledge_dir=knowledge_dir,
            episodes_dir=episodes_dir,
            procedures_dir=procedures_dir,
            common_knowledge_dir=common_knowledge_dir,
        )

        assert len(results) == 2
        lines = [r[1] for r in results]
        assert "Line one" in lines
        assert "Line two" in lines


# ── search_knowledge ─────────────────────────────────────


class TestSearchKnowledgeKeyword:
    def test_search_knowledge_keyword(
        self, rag: RAGMemorySearch, knowledge_dir: Path,
    ) -> None:
        """search_knowledge finds matching lines."""
        (knowledge_dir / "api-design.md").write_text(
            "REST API best practices\nGraphQL overview",
            encoding="utf-8",
        )
        (knowledge_dir / "testing.md").write_text(
            "Unit testing strategies", encoding="utf-8",
        )

        results = rag.search_knowledge("api", knowledge_dir)

        assert len(results) == 1
        assert results[0][0] == "api-design.md"
        assert "REST API best practices" in results[0][1]


class TestSearchKnowledgeNoResults:
    def test_search_knowledge_no_results(
        self, rag: RAGMemorySearch, knowledge_dir: Path,
    ) -> None:
        """Returns empty list for a non-matching query."""
        (knowledge_dir / "topic.md").write_text(
            "Existing content here", encoding="utf-8",
        )

        results = rag.search_knowledge("nonexistent_xyz_query", knowledge_dir)

        assert results == []


# ── index_file ───────────────────────────────────────────


class TestIndexFileDelegatesToIndexer:
    def test_index_file_delegates_to_indexer(
        self, rag: RAGMemorySearch, knowledge_dir: Path,
    ) -> None:
        """When indexer exists, index_file calls indexer.index_file."""
        mock_indexer = MagicMock()
        rag._indexer = mock_indexer
        rag._indexer_initialized = True

        test_path = knowledge_dir / "new_topic.md"
        test_path.write_text("New knowledge", encoding="utf-8")

        rag.index_file(test_path, "knowledge")

        mock_indexer.index_file.assert_called_once_with(test_path, "knowledge")


class TestIndexFileNoIndexerNoError:
    def test_index_file_no_indexer_no_error(
        self, rag: RAGMemorySearch, knowledge_dir: Path,
    ) -> None:
        """When indexer is None, index_file doesn't crash."""
        rag._indexer = None
        rag._indexer_initialized = True

        test_path = knowledge_dir / "topic.md"
        test_path.write_text("Content", encoding="utf-8")

        # Should not raise
        rag.index_file(test_path, "knowledge")
