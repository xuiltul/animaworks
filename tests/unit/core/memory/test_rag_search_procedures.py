"""Unit tests for procedures vector search enablement in rag_search.py."""
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


# ── _resolve_search_types ────────────────────────────────


class TestResolveSearchTypesKnowledge:
    def test_resolve_search_types_knowledge(self) -> None:
        """scope='knowledge' -> ['knowledge']."""
        assert RAGMemorySearch._resolve_search_types("knowledge") == ["knowledge"]


class TestResolveSearchTypesProcedures:
    def test_resolve_search_types_procedures(self) -> None:
        """scope='procedures' -> ['procedures']."""
        assert RAGMemorySearch._resolve_search_types("procedures") == ["procedures"]


class TestResolveSearchTypesAll:
    def test_resolve_search_types_all(self) -> None:
        """scope='all' -> ['knowledge', 'procedures']."""
        assert RAGMemorySearch._resolve_search_types("all") == ["knowledge", "procedures"]


class TestResolveSearchTypesCommonKnowledge:
    def test_resolve_search_types_common_knowledge(self) -> None:
        """scope='common_knowledge' -> ['knowledge']."""
        assert RAGMemorySearch._resolve_search_types("common_knowledge") == ["knowledge"]


class TestResolveSearchTypesDefault:
    def test_resolve_search_types_default(self) -> None:
        """scope='unknown' -> ['knowledge'] (fallback)."""
        assert RAGMemorySearch._resolve_search_types("unknown") == ["knowledge"]


# ── search_memory_text with procedures scope ─────────────


class TestSearchMemoryTextProceduresWithVector:
    def test_search_memory_text_procedures_with_vector(
        self,
        rag: RAGMemorySearch,
        knowledge_dir: Path,
        episodes_dir: Path,
        procedures_dir: Path,
        common_knowledge_dir: Path,
    ) -> None:
        """scope='procedures' triggers vector search when indexer is available."""
        (procedures_dir / "deploy.md").write_text(
            "Deploy to production server", encoding="utf-8",
        )

        # Set up a mock indexer to enable vector search path
        mock_indexer = MagicMock()
        rag._indexer = mock_indexer
        rag._indexer_initialized = True

        with patch.object(
            rag, "_vector_search_memory", return_value=[],
        ) as mock_vector:
            results = rag.search_memory_text(
                "deploy",
                scope="procedures",
                knowledge_dir=knowledge_dir,
                episodes_dir=episodes_dir,
                procedures_dir=procedures_dir,
                common_knowledge_dir=common_knowledge_dir,
            )

            # Keyword search should find the file
            filenames = [r[0] for r in results]
            assert "deploy.md" in filenames

            # Vector search should have been attempted for procedures scope
            mock_vector.assert_called_once_with("deploy", "procedures", knowledge_dir)


# ── _init_indexer indexes procedures ─────────────────────


class TestInitIndexerIndexesProcedures:
    def test_init_indexer_indexes_procedures(
        self,
        rag: RAGMemorySearch,
        anima_dir: Path,
    ) -> None:
        """Verify _init_indexer() calls index_directory for procedures/."""
        procedures_dir = anima_dir / "procedures"
        procedures_dir.mkdir(exist_ok=True)
        (procedures_dir / "test_proc.md").write_text(
            "Test procedure content", encoding="utf-8",
        )

        mock_indexer = MagicMock()
        mock_indexer.index_directory.return_value = 1

        mock_vector_store = MagicMock()

        # Patch where the imports happen (inside _init_indexer's local scope)
        with patch(
            "core.memory.rag.singleton.get_vector_store",
            return_value=mock_vector_store,
        ), patch(
            "core.memory.rag.MemoryIndexer",
            return_value=mock_indexer,
        ):
            rag._init_indexer()

            # Verify index_directory was called with procedures dir
            calls = mock_indexer.index_directory.call_args_list
            proc_calls = [
                c for c in calls
                if c.args[0] == procedures_dir and c.args[1] == "procedures"
            ]
            assert len(proc_calls) == 1, (
                f"Expected index_directory call for procedures, "
                f"got calls: {calls}"
            )
