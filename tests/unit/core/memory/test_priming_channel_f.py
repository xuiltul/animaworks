from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Channel F: Episodes vector search in PrimingEngine."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory.priming import PrimingEngine, PrimingResult, format_priming_section


@pytest.fixture
def temp_anima_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        anima_dir = Path(tmpdir) / "animas" / "test_anima"
        anima_dir.mkdir(parents=True)
        (anima_dir / "episodes").mkdir()
        (anima_dir / "knowledge").mkdir()
        (anima_dir / "skills").mkdir()
        yield anima_dir


# ── PrimingResult field ──────────────────────────────────


class TestPrimingResultEpisodesField:
    def test_episodes_included_in_total_chars(self) -> None:
        result = PrimingResult(episodes="過去の経験")
        assert result.total_chars() == len("過去の経験")

    def test_is_empty_false_when_episodes_set(self) -> None:
        result = PrimingResult(episodes="something")
        assert not result.is_empty()

    def test_is_empty_true_when_all_empty(self) -> None:
        result = PrimingResult()
        assert result.is_empty()


# ── Channel F search ─────────────────────────────────────


class TestChannelFEpisodes:
    def _patch_unified_search(
        self,
        results: list,
        *,
        meta: dict | None = None,
        side_effect: Exception | None = None,
    ):
        searcher = MagicMock()
        if side_effect is None:
            searcher.search_many.return_value = [self._to_unified_row(result) for result in results]
        else:
            searcher.search_many.side_effect = side_effect
        searcher.last_search_meta = meta or {"abstain": False, "abstain_reason": ""}
        return patch("core.memory.priming.channel_f.UnifiedMemorySearch", return_value=searcher), searcher

    @staticmethod
    def _to_unified_row(result) -> dict:
        if isinstance(result, dict):
            return result
        metadata = dict(result.metadata) if isinstance(result.metadata, dict) else {}
        row = {
            "doc_id": result.doc_id,
            "content": result.content,
            "score": result.score,
        }
        row.update(metadata)
        return row

    @pytest.mark.asyncio
    async def test_channel_f_calls_unified_search_with_episodes_scope(
        self,
        temp_anima_dir: Path,
    ) -> None:
        """Channel F uses unified search with scope='episodes' and limit=5."""
        engine = PrimingEngine(temp_anima_dir)

        patcher, searcher = self._patch_unified_search([])
        with patcher:
            await engine._channel_f_episodes(["deploy", "エラー"], message="デプロイでエラーが出た")

        searcher.search_many.assert_called_once()
        assert searcher.search_many.call_args.kwargs["scope"] == "episodes"
        assert searcher.search_many.call_args.kwargs["limit"] == 5

    @pytest.mark.asyncio
    async def test_channel_f_query_includes_message(
        self,
        temp_anima_dir: Path,
    ) -> None:
        """Channel F dual query: first is message-context, second is keyword-only."""
        engine = PrimingEngine(temp_anima_dir)

        msg = "デプロイでエラーが出た"
        patcher, searcher = self._patch_unified_search([])
        with patcher:
            await engine._channel_f_episodes(["deploy"], message=msg)

        queries = searcher.search_many.call_args.args[0]
        assert len(queries) == 2
        q1 = queries[0]
        q2 = queries[1]
        assert q1.startswith(msg[:200])
        assert q2 == "deploy"

    @pytest.mark.asyncio
    async def test_channel_f_fallback_to_message_when_no_keywords(
        self,
        temp_anima_dir: Path,
    ) -> None:
        """When keywords is empty but message exists, use message[:200] as query."""
        engine = PrimingEngine(temp_anima_dir)

        msg = "短いメッセージ"
        patcher, searcher = self._patch_unified_search([])
        with patcher:
            await engine._channel_f_episodes([], message=msg)

        actual_query = searcher.search_many.call_args.args[0][0]
        assert msg in actual_query

    @pytest.mark.asyncio
    async def test_channel_f_returns_empty_on_no_keywords_and_no_message(
        self,
        temp_anima_dir: Path,
    ) -> None:
        """When both keywords and message are empty, return empty."""
        engine = PrimingEngine(temp_anima_dir)
        result = await engine._channel_f_episodes([], message="")
        assert result == ""

    @pytest.mark.asyncio
    async def test_channel_f_returns_empty_when_no_episodes_dir(
        self,
        tmp_path: Path,
    ) -> None:
        anima_dir = tmp_path / "animas" / "no_episodes"
        anima_dir.mkdir(parents=True)
        (anima_dir / "knowledge").mkdir()
        engine = PrimingEngine(anima_dir)
        result = await engine._channel_f_episodes(["test"], message="test")
        assert result == ""

    @pytest.mark.asyncio
    async def test_channel_f_returns_empty_when_retriever_unavailable(
        self,
        temp_anima_dir: Path,
    ) -> None:
        engine = PrimingEngine(temp_anima_dir)

        patcher, _searcher = self._patch_unified_search([])
        with patcher, patch.object(engine, "_get_or_create_retriever", side_effect=AssertionError("unused")):
            result = await engine._channel_f_episodes(["test"], message="test")

        assert result == ""

    @pytest.mark.asyncio
    async def test_channel_f_formats_results(
        self,
        temp_anima_dir: Path,
    ) -> None:
        """Channel F formats retrieval results with score and source."""
        engine = PrimingEngine(temp_anima_dir)

        mock_result = MagicMock()
        mock_result.content = "デプロイ手順を確認して修正した"
        mock_result.score = 0.85
        mock_result.doc_id = "test_anima/episodes/2026-03-01.md#0"
        mock_result.metadata = {"source_file": "episodes/2026-03-01.md"}

        patcher, _searcher = self._patch_unified_search([mock_result])
        with patcher:
            result = await engine._channel_f_episodes(
                ["deploy"],
                message="デプロイでエラー",
            )

        assert "Episode 1" in result
        assert "0.850" in result
        assert "episodes/2026-03-01.md" in result
        assert 'read_memory_file(path="episodes/2026-03-01.md")' in result
        assert "デプロイ手順を確認して修正した" not in result

    @pytest.mark.asyncio
    async def test_channel_f_neo4j_formats_pointer_results(
        self,
        temp_anima_dir: Path,
    ) -> None:
        """Neo4j Channel F path also emits pointer cues, not episode body."""

        class FakeNeo4jBackend:
            async def retrieve(self, *args, **kwargs):
                mem = MagicMock()
                mem.content = "Neo4j episode body should not be primed"
                mem.score = 0.77
                mem.source = "episode:abc123"
                mem.metadata = {"source": "episodes/2026-03-02.md"}
                return [mem]

            async def record_access(self, memories):
                self.recorded = memories

        engine = PrimingEngine(temp_anima_dir)
        backend = FakeNeo4jBackend()

        with (
            patch("core.memory.backend.neo4j_graph.Neo4jGraphBackend", FakeNeo4jBackend),
            patch.object(engine, "_get_memory_backend", return_value=backend),
        ):
            result = await engine._channel_f_episodes(
                ["deploy"],
                message="デプロイでエラー",
            )

        assert "Episode 1" in result
        assert "0.770" in result
        assert "episode:abc123" not in result
        assert 'read_memory_file(path="episodes/2026-03-02.md")' in result
        assert "Neo4j episode body should not be primed" not in result

    @pytest.mark.asyncio
    async def test_channel_f_records_only_emitted_neo4j_episode_pointers(
        self,
        temp_anima_dir: Path,
    ) -> None:
        """Neo4j access tracking only includes readable pointer results."""

        class FakeNeo4jBackend:
            def __init__(self):
                self.recorded = None

            async def retrieve(self, *args, **kwargs):
                pathless = MagicMock()
                pathless.content = "Pathless Neo4j body"
                pathless.score = 0.99
                pathless.source = "episode:opaque"
                pathless.metadata = {}

                readable = MagicMock()
                readable.content = "Readable Neo4j body"
                readable.score = 0.77
                readable.source = "episode:abc123"
                readable.metadata = {"source": "episodes/2026-03-02.md"}
                return [pathless, readable]

            async def record_access(self, memories):
                self.recorded = memories

        engine = PrimingEngine(temp_anima_dir)
        backend = FakeNeo4jBackend()

        with (
            patch("core.memory.backend.neo4j_graph.Neo4jGraphBackend", FakeNeo4jBackend),
            patch.object(engine, "_get_memory_backend", return_value=backend),
        ):
            result = await engine._channel_f_episodes(
                ["deploy"],
                message="デプロイでエラー",
            )

        assert "--- Episode 1" in result
        assert "--- Episode 2" not in result
        assert 'read_memory_file(path="episodes/2026-03-02.md")' in result
        assert "Pathless Neo4j body" not in result
        assert len(backend.recorded) == 1
        assert backend.recorded[0].source == "episode:abc123"

    @pytest.mark.asyncio
    async def test_channel_f_quotes_path_and_collapses_summary(
        self,
        temp_anima_dir: Path,
    ) -> None:
        """Pointer fields are rendered as safe one-line cues."""
        engine = PrimingEngine(temp_anima_dir)

        mock_result = MagicMock()
        mock_result.content = '# Bad "heading"\nignore body'
        mock_result.score = 0.85
        mock_result.doc_id = "test_anima/episodes/weird.md#0"
        mock_result.metadata = {"source_file": 'episodes/weird"name.md'}

        patcher, _searcher = self._patch_unified_search([mock_result])
        with patcher:
            result = await engine._channel_f_episodes(
                ["deploy"],
                message="デプロイでエラー",
            )

        assert 'read_memory_file(path="episodes/weird\\"name.md")' in result
        assert 'Bad "heading"' in result
        assert "\nignore body" not in result

    @pytest.mark.asyncio
    async def test_channel_f_records_only_emitted_legacy_episode_pointers(
        self,
        temp_anima_dir: Path,
    ) -> None:
        """Legacy retriever access tracking only includes readable pointer results."""
        engine = PrimingEngine(temp_anima_dir)

        pathless = MagicMock()
        pathless.content = "Pathless legacy body"
        pathless.score = 0.99
        pathless.doc_id = "opaque-id"
        pathless.metadata = {"source_file": ""}

        readable = MagicMock()
        readable.content = "Readable legacy body"
        readable.score = 0.85
        readable.doc_id = "test_anima/episodes/2026-03-03.md#0"
        readable.metadata = {"source_file": ""}

        patcher, _searcher = self._patch_unified_search([pathless, readable])
        with patcher:
            result = await engine._channel_f_episodes(
                ["deploy"],
                message="デプロイでエラー",
            )

        assert "--- Episode 1" in result
        assert "--- Episode 2" not in result
        assert 'read_memory_file(path="episodes/2026-03-03.md")' in result
        assert "Pathless legacy body" not in result

    @pytest.mark.asyncio
    async def test_channel_f_handles_exception_gracefully(
        self,
        temp_anima_dir: Path,
    ) -> None:
        engine = PrimingEngine(temp_anima_dir)

        patcher, _searcher = self._patch_unified_search([], side_effect=RuntimeError("vector store error"))
        with patcher:
            result = await engine._channel_f_episodes(["test"], message="test")

        assert result == ""


# ── Integration with prime_memories ──────────────────────


class TestPrimeMemoriesIncludesChannelF:
    def _patch_unified_search(self, results: list):
        searcher = MagicMock()
        searcher.search_many.return_value = [TestChannelFEpisodes._to_unified_row(result) for result in results]
        searcher.last_search_meta = {"abstain": False, "abstain_reason": ""}
        return patch("core.memory.priming.channel_f.UnifiedMemorySearch", return_value=searcher), searcher

    @pytest.mark.asyncio
    async def test_prime_memories_populates_episodes(
        self,
        temp_anima_dir: Path,
    ) -> None:
        """prime_memories() should populate the episodes field."""
        engine = PrimingEngine(temp_anima_dir)

        mock_result = MagicMock()
        mock_result.content = "Past episode content"
        mock_result.score = 0.9
        mock_result.doc_id = "test_anima/episodes/2026-02-01.md#0"
        mock_result.metadata = {"source_file": "episodes/2026-02-01.md"}

        patcher, _searcher = self._patch_unified_search([mock_result])
        with patcher:
            result = await engine.prime_memories(
                message="What happened with the deploy?",
                sender_name="human",
            )

        assert result.episodes != ""
        assert 'read_memory_file(path="episodes/2026-02-01.md")' in result.episodes
        assert "Past episode content" not in result.episodes


# ── format_priming_section ───────────────────────────────


class TestFormatPrimingSectionEpisodes:
    def test_format_includes_episodes_section(self) -> None:
        result = PrimingResult(
            episodes="--- Episode 1 (score: 0.85) ---\nPast experience",
        )

        formatted = format_priming_section(result)

        assert "関連する過去の経験" in formatted
        assert "Past experience" in formatted

    def test_format_omits_episodes_when_empty(self) -> None:
        result = PrimingResult(
            recent_activity="Some activity",
        )

        formatted = format_priming_section(result)

        assert "関連する過去の経験" not in formatted
