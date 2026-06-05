from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for dual-query strategy and language-agnostic keyword extraction.

Covers:
  - _build_dual_queries(): query construction for message + keyword paths
  - _search_and_merge(): max-score deduplication across multiple queries
  - _extract_keywords(): language-agnostic keyword extraction (CJK, Latin, Korean)
  - _meets_min_length(): character-category-based minimum length filter
  - Semantic dilution regression: multi-topic messages must surface minority keywords
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core.memory.priming import PrimingEngine


@pytest.fixture
def anima_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        anima_dir = Path(tmpdir) / "animas" / "test"
        anima_dir.mkdir(parents=True)
        (anima_dir / "knowledge").mkdir()
        (anima_dir / "episodes").mkdir()
        (anima_dir / "skills").mkdir()
        yield anima_dir


@pytest.fixture
def anima_dir_with_knowledge(anima_dir):
    (anima_dir / "knowledge" / "malaysia-travel.md").write_text(
        "# マレーシア旅行\n\nKL観光とランカウイの予定。\n",
        encoding="utf-8",
    )
    (anima_dir / "knowledge" / "debugging-guide.md").write_text(
        "# デバッグガイド\n\nサーバーデバッグの手順。\n",
        encoding="utf-8",
    )
    return anima_dir


# ── _build_dual_queries ──────────────────────────────────


class TestBuildDualQueries:
    def test_both_message_and_keywords(self) -> None:
        queries = PrimingEngine._build_dual_queries(
            "Hello world, how are you?",
            ["hello", "world"],
        )
        assert len(queries) == 2
        assert queries[0] == "Hello world, how are you?"
        assert queries[1] == "hello world"

    def test_message_only(self) -> None:
        queries = PrimingEngine._build_dual_queries("Some message", [])
        assert len(queries) == 1
        assert queries[0] == "Some message"

    def test_keywords_only(self) -> None:
        queries = PrimingEngine._build_dual_queries("", ["alpha", "beta"])
        assert len(queries) == 1
        assert queries[0] == "alpha beta"

    def test_empty_both(self) -> None:
        queries = PrimingEngine._build_dual_queries("", [])
        assert queries == []

    def test_dedup_identical(self) -> None:
        queries = PrimingEngine._build_dual_queries("test", ["test"])
        assert len(queries) == 1

    def test_long_message_truncated_to_300(self) -> None:
        long_msg = "a" * 500
        queries = PrimingEngine._build_dual_queries(long_msg, ["kw"])
        assert len(queries[0]) == 300

    def test_max_5_keywords(self) -> None:
        kws = ["a", "b", "c", "d", "e", "f", "g"]
        queries = PrimingEngine._build_dual_queries("msg", kws)
        assert queries[1] == "a b c d e"


# ── _search_and_merge ─────────────────────────────────────


class TestSearchAndMerge:
    def test_merge_deduplicates_by_doc_id(self, anima_dir) -> None:
        engine = PrimingEngine(anima_dir)

        r1 = MagicMock(doc_id="doc1", score=0.8, content="result 1")
        r2 = MagicMock(doc_id="doc2", score=0.6, content="result 2")
        r3 = MagicMock(doc_id="doc1", score=0.9, content="result 1 better")

        mock_retriever = MagicMock()
        mock_retriever.search.side_effect = [[r1, r2], [r3]]

        results = engine._search_and_merge(
            mock_retriever,
            ["query1", "query2"],
            "test",
            memory_type="knowledge",
            top_k=5,
        )

        assert len(results) == 2
        assert results[0].doc_id == "doc1"
        assert results[0].score == 0.9
        assert results[1].doc_id == "doc2"

    def test_merge_respects_top_k(self, anima_dir) -> None:
        engine = PrimingEngine(anima_dir)

        results_a = [MagicMock(doc_id=f"a{i}", score=0.9 - i * 0.1) for i in range(5)]
        results_b = [MagicMock(doc_id=f"b{i}", score=0.85 - i * 0.1) for i in range(5)]

        mock_retriever = MagicMock()
        mock_retriever.search.side_effect = [results_a, results_b]

        results = engine._search_and_merge(
            mock_retriever,
            ["q1", "q2"],
            "test",
            memory_type="knowledge",
            top_k=3,
        )

        assert len(results) == 3

    def test_single_query_works(self, anima_dir) -> None:
        engine = PrimingEngine(anima_dir)
        r1 = MagicMock(doc_id="doc1", score=0.7)
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [r1]

        results = engine._search_and_merge(
            mock_retriever,
            ["single"],
            "test",
            memory_type="episodes",
            top_k=3,
        )

        assert len(results) == 1
        mock_retriever.search.assert_called_once()


# ── _meets_min_length ─────────────────────────────────────


class TestMeetsMinLength:
    def test_single_cjk_kanji(self) -> None:
        assert PrimingEngine._meets_min_length("裏") is True
        assert PrimingEngine._meets_min_length("金") is True
        assert PrimingEngine._meets_min_length("型") is True

    def test_cjk_two_chars(self) -> None:
        assert PrimingEngine._meets_min_length("実装") is True
        assert PrimingEngine._meets_min_length("검색") is True  # Korean

    def test_katakana(self) -> None:
        assert PrimingEngine._meets_min_length("マレーシア") is True
        assert PrimingEngine._meets_min_length("ア") is True  # single katakana

    def test_latin_short_rejected(self) -> None:
        assert PrimingEngine._meets_min_length("a") is False
        assert PrimingEngine._meets_min_length("to") is False

    def test_latin_3_chars_accepted(self) -> None:
        assert PrimingEngine._meets_min_length("RAG") is True
        assert PrimingEngine._meets_min_length("the") is True  # length OK, stopword filters separately

    def test_mixed_cjk_latin(self) -> None:
        assert PrimingEngine._meets_min_length("Python3") is True
        assert PrimingEngine._meets_min_length("型A") is True  # has CJK -> threshold 1

    def test_thai(self) -> None:
        assert PrimingEngine._meets_min_length("ก") is True  # single Thai char
        assert PrimingEngine._meets_min_length("กร") is True

    def test_korean_hangul(self) -> None:
        assert PrimingEngine._meets_min_length("검") is True
        assert PrimingEngine._meets_min_length("검색") is True


# ── _extract_keywords: language-agnostic ──────────────────


class TestExtractKeywordsMultiLang:
    def test_english_keywords(self, anima_dir) -> None:
        engine = PrimingEngine(anima_dir)
        kws = engine._extract_keywords("Deploy the application to production server")
        lower = [k.lower() for k in kws]
        assert "deploy" in lower or "application" in lower or "production" in lower
        assert "the" not in lower
        assert "to" not in lower

    def test_japanese_keywords(self, anima_dir) -> None:
        engine = PrimingEngine(anima_dir)
        kws = engine._extract_keywords("サーバー の デバッグ を 修正 した")
        assert "サーバー" in kws
        assert "デバッグ" in kws
        assert "修正" in kws
        assert "の" not in kws
        assert "を" not in kws

    def test_korean_keywords(self, anima_dir) -> None:
        engine = PrimingEngine(anima_dir)
        kws = engine._extract_keywords("서버 배포 오류 수정")
        assert len(kws) > 0
        assert all(len(k) >= 1 for k in kws)

    def test_chinese_keywords(self, anima_dir) -> None:
        engine = PrimingEngine(anima_dir)
        kws = engine._extract_keywords("部署 服务器 修复 错误")
        assert len(kws) > 0

    def test_mixed_language(self, anima_dir) -> None:
        engine = PrimingEngine(anima_dir)
        kws = engine._extract_keywords("ChromaDB ベクトル 検索 engine")
        assert "ChromaDB" in kws
        assert "ベクトル" in kws or "検索" in kws
        assert "engine" in kws

    def test_knowledge_entity_priority(self, anima_dir_with_knowledge) -> None:
        engine = PrimingEngine(anima_dir_with_knowledge)
        kws = engine._extract_keywords("malaysia travel plan for next month")
        lower = [k.lower() for k in kws]
        assert "malaysia-travel" in lower or any("malaysia" in k for k in lower)

    def test_max_10_keywords(self, anima_dir) -> None:
        long_msg = " ".join(f"word{i}" for i in range(50))
        engine = PrimingEngine(anima_dir)
        kws = engine._extract_keywords(long_msg)
        assert len(kws) <= 10

    def test_truncation(self, anima_dir) -> None:
        engine = PrimingEngine(anima_dir)
        long_msg = "keyword " * 2000
        kws = engine._extract_keywords(long_msg)
        assert len(kws) > 0
        assert len(kws) <= 10


# ── Semantic dilution regression ──────────────────────────


class TestSemanticDilutionRegression:
    @pytest.mark.asyncio
    async def test_multi_topic_message_finds_minority_keyword(
        self,
        anima_dir_with_knowledge,
    ) -> None:
        """The original bug: multi-topic messages where the minority keyword
        (e.g. 'マレーシア') was drowned out by the dominant topic ('デバッグ').

        Dual query issues two searches. The first (message-context) returns
        debugging-focused results; the second (keyword-only) returns malaysia
        results. After merging, both topics are represented.
        """
        engine = PrimingEngine(anima_dir_with_knowledge)

        mock_searcher = SimpleNamespace(
            last_search_meta={},
            search_many=MagicMock(
                return_value=[
                    {
                        "doc_id": "debugging-guide#0",
                        "score": 0.85,
                        "content": "デバッグ手順",
                        "source_file": "knowledge/debugging-guide.md",
                        "origin": "consolidation",
                        "anima": "test",
                    },
                    {
                        "doc_id": "malaysia-travel#0",
                        "score": 0.9,
                        "content": "マレーシア旅行の計画",
                        "source_file": "knowledge/malaysia-travel.md",
                        "origin": "consolidation",
                        "anima": "test",
                    },
                ]
            ),
        )

        with patch("core.memory.priming.channel_c.UnifiedMemorySearch", return_value=mock_searcher):
            result = await engine._channel_c_related_knowledge(
                keywords=["マレーシア", "デバッグ"],
                message="サーバーのデバッグに問題があったので修正した。マレーシア",
            )

        mock_searcher.search_many.assert_called_once()
        queries = mock_searcher.search_many.call_args.args[0]
        assert len(queries) == 2, "Dual query should include message and keyword searches"
        medium_text, _ = result
        assert 'read_memory_file(path="knowledge/malaysia-travel.md")' in medium_text, (
            f"Malaysia pointer should be in results: {medium_text}"
        )
        assert 'read_memory_file(path="knowledge/debugging-guide.md")' in medium_text, (
            f"Debug pointer should also be in results: {medium_text}"
        )
        assert "マレーシア旅行の計画" not in medium_text
        assert "デバッグ手順" not in medium_text

    @pytest.mark.asyncio
    async def test_single_topic_still_works(self, anima_dir) -> None:
        """Single-topic messages should still work correctly (no regression)."""
        engine = PrimingEngine(anima_dir)

        mock_searcher = SimpleNamespace(
            last_search_meta={},
            search_many=MagicMock(
                return_value=[
                    {
                        "doc_id": "doc1",
                        "score": 0.9,
                        "content": "relevant content",
                        "source_file": "knowledge/test.md",
                        "origin": "consolidation",
                        "anima": "test",
                    },
                ]
            ),
        )

        with patch("core.memory.priming.channel_c.UnifiedMemorySearch", return_value=mock_searcher):
            result = await engine._channel_c_related_knowledge(
                keywords=["マレーシア"],
                message="マレーシア旅行について教えて",
            )

        mock_searcher.search_many.assert_called_once()
        assert len(mock_searcher.search_many.call_args.args[0]) == 2
        medium_text, _ = result
        assert 'read_memory_file(path="knowledge/test.md")' in medium_text
        assert "relevant content" not in medium_text
