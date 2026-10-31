"""Unit tests for core/memory/retrieval/rag_search.py — RAGMemorySearch."""

from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory.rag.retriever import RetrievalResult
from core.memory.retrieval.rag_search import (
    RAGMemorySearch,
    _keyword_token_matches,
    _read_keyword_file,
    _read_keyword_file_by_signature,
)

# ── Fixtures ─────────────────────────────────────────────


@pytest.mark.parametrize(
    ("token", "content", "expected"),
    [
        ("foobar", "foo release", False),
        ("foobar", "xxfoo release", False),
        ("foobar", "xxfoobarxx", True),
        ("週次圧縮について", "週次圧縮", True),
        ("長い日本語入力" * 100, "日本語入力", True),
        ("foo", "fo release", False),
    ],
)
def test_keyword_token_matches(token: str, content: str, expected: bool) -> None:
    assert _keyword_token_matches(token, content) is expected


def test_keyword_file_cache_invalidates_on_change(tmp_path: Path) -> None:
    path = tmp_path / "memory.md"
    path.write_text("first", encoding="utf-8")
    _read_keyword_file_by_signature.cache_clear()

    first, _ = _read_keyword_file(path)
    second, _ = _read_keyword_file(path)
    path.write_text("changed", encoding="utf-8")
    changed, _ = _read_keyword_file(path)

    assert first == second == "first"
    assert changed == "changed"
    assert _read_keyword_file_by_signature.cache_info().hits == 1


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

    def test_task_runner_search_skips_auto_indexing_but_reads_existing_data(
        self,
        anima_dir: Path,
        common_knowledge_dir: Path,
        common_skills_dir: Path,
        knowledge_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ANIMAWORKS_TASK_IPC_PATH", "/tmp/task-runner.sock")
        (knowledge_dir / "pending.md").write_text("not indexed yet", encoding="utf-8")
        rag = RAGMemorySearch(anima_dir, common_knowledge_dir, common_skills_dir)
        existing = RetrievalResult(
            doc_id="alice/knowledge/existing#0",
            content="existing indexed memory",
            score=0.8,
            metadata={"source_file": "knowledge/existing.md", "memory_type": "knowledge"},
            source_scores={},
        )

        with (
            patch("core.memory.rag.vector_registry.get_vector_store", return_value=object()),
            patch("core.memory.rag.MemoryIndexer") as indexer_cls,
            patch("core.memory.rag.retriever.MemoryRetriever") as retriever_cls,
            patch.object(rag, "_check_shared_collections") as check_shared,
        ):
            retriever_cls.return_value.search.return_value = [existing]
            rag._get_indexer()
            results = rag._vector_search_primary("existing", "knowledge", 0, knowledge_dir)

        indexer_cls.return_value.index_directory.assert_not_called()
        indexer_cls.return_value.index_conversation_summary.assert_not_called()
        check_shared.assert_not_called()
        assert results[0]["content"] == "existing indexed memory"

    def test_root_search_keeps_auto_indexing(
        self,
        anima_dir: Path,
        common_knowledge_dir: Path,
        common_skills_dir: Path,
        knowledge_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("ANIMAWORKS_TASK_IPC_PATH", raising=False)
        (knowledge_dir / "pending.md").write_text("index me", encoding="utf-8")
        rag = RAGMemorySearch(anima_dir, common_knowledge_dir, common_skills_dir)

        with (
            patch("core.memory.rag.vector_registry.get_vector_store", return_value=object()),
            patch("core.memory.rag.MemoryIndexer") as indexer_cls,
            patch.object(rag, "_check_shared_collections") as check_shared,
        ):
            indexer_cls.return_value.index_directory.return_value.chunks_indexed = 0
            rag._get_indexer()

        indexer_cls.return_value.index_directory.assert_any_call(knowledge_dir, "knowledge")
        check_shared.assert_called_once()


class TestLongtermBM25Refresh:
    def test_index_file_updates_longterm_bm25_without_rebuilding(
        self,
        rag: RAGMemorySearch,
        anima_dir: Path,
        knowledge_dir: Path,
    ) -> None:
        from core.memory.retrieval.bm25 import (
            is_longterm_bm25_dirty,
            rebuild_longterm_bm25_index,
            search_longterm_memory_bm25,
        )

        (knowledge_dir / "old.md").write_text("# Old\n\nBaseline memo.", encoding="utf-8")
        rebuild_longterm_bm25_index(anima_dir)
        assert is_longterm_bm25_dirty(anima_dir) is False

        new_file = knowledge_dir / "new.md"
        new_file.write_text("# New\n\nZephyrNova launchpad checklist.", encoding="utf-8")
        with patch.object(rag, "_get_indexer", return_value=None):
            rag.index_file(new_file, "knowledge")

        assert is_longterm_bm25_dirty(anima_dir) is True
        hits = search_longterm_memory_bm25(
            anima_dir,
            "ZephyrNova",
            memory_types=("knowledge",),
            top_k=5,
        )
        assert hits[0]["source_file"] == "knowledge/new.md"

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
        self,
        rag: RAGMemorySearch,
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
    def test_activity_time_range_is_forwarded_to_bm25(
        self,
        rag: RAGMemorySearch,
        knowledge_dir: Path,
        episodes_dir: Path,
        procedures_dir: Path,
        common_knowledge_dir: Path,
    ) -> None:
        with patch("core.memory.retrieval.rag_search.search_activity_log", return_value=[]) as search:
            rag.search_memory_text(
                "meeting",
                scope="activity_log",
                knowledge_dir=knowledge_dir,
                episodes_dir=episodes_dir,
                procedures_dir=procedures_dir,
                common_knowledge_dir=common_knowledge_dir,
                time_start="2026-07-01",
                time_end="2026-07-02",
            )

        assert search.call_args.kwargs["time_start"] == "2026-07-01"
        assert search.call_args.kwargs["time_end"] == "2026-07-02"
        assert search.call_args.kwargs["all_time"] is True

    def test_explicit_time_range_is_forwarded_to_unified_search(
        self,
        rag: RAGMemorySearch,
        knowledge_dir: Path,
        episodes_dir: Path,
        procedures_dir: Path,
        common_knowledge_dir: Path,
    ) -> None:
        with patch("core.memory.retrieval.unified_search.UnifiedMemorySearch") as search_cls:
            searcher = search_cls.return_value
            searcher.search.return_value = []
            searcher.last_search_meta = {}

            rag.search_memory_text(
                "meeting",
                scope="episodes",
                knowledge_dir=knowledge_dir,
                episodes_dir=episodes_dir,
                procedures_dir=procedures_dir,
                common_knowledge_dir=common_knowledge_dir,
                time_start="2026-07-01",
                time_end="2026-07-18",
            )

        assert searcher.search.call_args.kwargs["time_start"] == "2026-07-01"
        assert searcher.search.call_args.kwargs["time_end"] == "2026-07-18"

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

        with patch.object(rag, "_get_indexer", return_value=None):
            results = rag.search_memory_text(
                "python",
                scope="all",
                knowledge_dir=knowledge_dir,
                episodes_dir=episodes_dir,
                procedures_dir=procedures_dir,
                common_knowledge_dir=common_knowledge_dir,
            )

        assert len(results) >= 2
        sources = [r["source_file"] for r in results]
        assert any("python.md" in s for s in sources)
        assert any("2026-01-01.md" in s for s in sources)

    def test_search_memory_text_keyword_excludes_superseded_knowledge(
        self,
        rag: RAGMemorySearch,
        knowledge_dir: Path,
        episodes_dir: Path,
        procedures_dir: Path,
        common_knowledge_dir: Path,
    ) -> None:
        (knowledge_dir / "active.md").write_text(
            "---\nvalid_until: ''\n---\n\nkeyword current deployment policy",
            encoding="utf-8",
        )
        (knowledge_dir / "superseded.md").write_text(
            "---\nvalid_until: '2026-06-10T00:00:00+09:00'\n---\n\nkeyword obsolete deployment policy",
            encoding="utf-8",
        )

        with patch.object(rag, "_get_indexer", return_value=None):
            results = rag.search_memory_text(
                "keyword deployment",
                scope="knowledge",
                knowledge_dir=knowledge_dir,
                episodes_dir=episodes_dir,
                procedures_dir=procedures_dir,
                common_knowledge_dir=common_knowledge_dir,
            )

        sources = [r["source_file"] for r in results]
        assert any("active.md" in s for s in sources)
        assert not any("superseded.md" in s for s in sources)


class TestSearchMemoryTextRespectsScope:
    def test_search_memory_text_respects_scope(
        self,
        rag: RAGMemorySearch,
        knowledge_dir: Path,
        episodes_dir: Path,
        procedures_dir: Path,
        common_knowledge_dir: Path,
    ) -> None:
        """scope='knowledge' only searches the knowledge directory (keyword fallback)."""
        (knowledge_dir / "topic.md").write_text(
            "keyword in knowledge",
            encoding="utf-8",
        )
        (episodes_dir / "2026-02-01.md").write_text(
            "keyword in episodes",
            encoding="utf-8",
        )
        (procedures_dir / "deploy.md").write_text(
            "keyword in procedures",
            encoding="utf-8",
        )

        with patch.object(rag, "_get_indexer", return_value=None):
            results = rag.search_memory_text(
                "keyword",
                scope="knowledge",
                knowledge_dir=knowledge_dir,
                episodes_dir=episodes_dir,
                procedures_dir=procedures_dir,
                common_knowledge_dir=common_knowledge_dir,
            )

        sources = [r["source_file"] for r in results]
        assert any("topic.md" in s for s in sources)
        assert not any("2026-02-01.md" in s for s in sources)
        assert not any("deploy.md" in s for s in sources)


class TestSearchMemoryTextEmptyQuery:
    def test_search_memory_text_empty_query_returns_nothing(
        self,
        rag: RAGMemorySearch,
        knowledge_dir: Path,
        episodes_dir: Path,
        procedures_dir: Path,
        common_knowledge_dir: Path,
    ) -> None:
        """Empty query returns no results (guard against vacuous match)."""
        (knowledge_dir / "info.md").write_text(
            "Line one\nLine two",
            encoding="utf-8",
        )

        with patch.object(rag, "_get_indexer", return_value=None):
            results = rag.search_memory_text(
                "",
                scope="knowledge",
                knowledge_dir=knowledge_dir,
                episodes_dir=episodes_dir,
                procedures_dir=procedures_dir,
                common_knowledge_dir=common_knowledge_dir,
            )

        assert results == []


class TestSearchMemoryTextOrSplit:
    def test_or_split_matches_either_token(
        self,
        rag: RAGMemorySearch,
        knowledge_dir: Path,
        episodes_dir: Path,
        procedures_dir: Path,
        common_knowledge_dir: Path,
    ) -> None:
        """Space-separated tokens are OR-matched: a line containing
        any single token is returned."""
        (knowledge_dir / "langs.md").write_text(
            "Python is great\nJava is fine\nRust is fast",
            encoding="utf-8",
        )

        with patch.object(rag, "_get_indexer", return_value=None):
            results = rag.search_memory_text(
                "Python Rust",
                scope="knowledge",
                knowledge_dir=knowledge_dir,
                episodes_dir=episodes_dir,
                procedures_dir=procedures_dir,
                common_knowledge_dir=common_knowledge_dir,
            )

        assert len(results) >= 1
        content = results[0]["content"]
        assert "Python is great" in content
        assert "Rust is fast" in content

    def test_or_split_whitespace_only_returns_empty(
        self,
        rag: RAGMemorySearch,
        knowledge_dir: Path,
        episodes_dir: Path,
        procedures_dir: Path,
        common_knowledge_dir: Path,
    ) -> None:
        """Query with only whitespace returns no results."""
        (knowledge_dir / "info.md").write_text("content", encoding="utf-8")

        results = rag.search_memory_text(
            "   ",
            scope="knowledge",
            knowledge_dir=knowledge_dir,
            episodes_dir=episodes_dir,
            procedures_dir=procedures_dir,
            common_knowledge_dir=common_knowledge_dir,
        )

        assert results == []


# ── search_knowledge ─────────────────────────────────────


class TestSearchKnowledgeKeyword:
    def test_search_knowledge_keyword(
        self,
        rag: RAGMemorySearch,
        knowledge_dir: Path,
    ) -> None:
        """search_knowledge finds matching lines."""
        (knowledge_dir / "api-design.md").write_text(
            "REST API best practices\nGraphQL overview",
            encoding="utf-8",
        )
        (knowledge_dir / "testing.md").write_text(
            "Unit testing strategies",
            encoding="utf-8",
        )

        results = rag.search_knowledge("api", knowledge_dir)

        assert len(results) == 1
        assert results[0][0] == "api-design.md"
        assert "REST API best practices" in results[0][1]


class TestSearchKnowledgeOrSplit:
    def test_search_knowledge_or_split(
        self,
        rag: RAGMemorySearch,
        knowledge_dir: Path,
    ) -> None:
        """search_knowledge uses OR-split for multi-word queries."""
        (knowledge_dir / "mixed.md").write_text(
            "REST API guide\nGraphQL overview\nDeploy instructions",
            encoding="utf-8",
        )

        results = rag.search_knowledge("api deploy", knowledge_dir)

        lines = [r[1] for r in results]
        assert "REST API guide" in lines
        assert "Deploy instructions" in lines
        assert "GraphQL overview" not in lines

    def test_search_knowledge_empty_returns_nothing(
        self,
        rag: RAGMemorySearch,
        knowledge_dir: Path,
    ) -> None:
        """Empty query returns no results."""
        (knowledge_dir / "info.md").write_text("content", encoding="utf-8")

        results = rag.search_knowledge("", knowledge_dir)
        assert results == []


class TestSearchKnowledgeNoResults:
    def test_search_knowledge_no_results(
        self,
        rag: RAGMemorySearch,
        knowledge_dir: Path,
    ) -> None:
        """Returns empty list for a non-matching query."""
        (knowledge_dir / "topic.md").write_text(
            "Existing content here",
            encoding="utf-8",
        )

        results = rag.search_knowledge("nonexistent_xyz_query", knowledge_dir)

        assert results == []


# ── index_file ───────────────────────────────────────────


class TestIndexFileDelegatesToIndexer:
    def test_index_file_delegates_to_indexer(
        self,
        rag: RAGMemorySearch,
        knowledge_dir: Path,
    ) -> None:
        """When indexer exists, index_file calls indexer.index_file."""
        mock_indexer = MagicMock()
        rag._indexer = mock_indexer
        rag._indexer_initialized = True

        test_path = knowledge_dir / "new_topic.md"
        test_path.write_text("New knowledge", encoding="utf-8")

        rag.index_file(test_path, "knowledge")

        mock_indexer.index_file.assert_called_once_with(test_path, "knowledge", force=False, origin="")


class TestIndexFileNoIndexerNoError:
    def test_index_file_no_indexer_no_error(
        self,
        rag: RAGMemorySearch,
        knowledge_dir: Path,
    ) -> None:
        """When indexer is None, index_file doesn't crash."""
        rag._indexer = None
        rag._indexer_initialized = True

        test_path = knowledge_dir / "topic.md"
        test_path.write_text("Content", encoding="utf-8")

        # Should not raise
        rag.index_file(test_path, "knowledge")
