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
    @pytest.mark.asyncio
    async def test_channel_f_calls_retriever_with_episodes_type(
        self, temp_anima_dir: Path,
    ) -> None:
        """Channel F must search with memory_type='episodes' and top_k=3."""
        engine = PrimingEngine(temp_anima_dir)

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []

        with patch.object(engine, "_get_or_create_retriever", return_value=mock_retriever):
            await engine._channel_f_episodes(["deploy", "エラー"], message="デプロイでエラーが出た")

        mock_retriever.search.assert_called_once()
        call_kwargs = mock_retriever.search.call_args
        assert call_kwargs.kwargs.get("memory_type") == "episodes"
        assert call_kwargs.kwargs.get("top_k") == 3

    @pytest.mark.asyncio
    async def test_channel_f_query_includes_message(
        self, temp_anima_dir: Path,
    ) -> None:
        """Channel F query prepends message[:200] to keywords."""
        engine = PrimingEngine(temp_anima_dir)

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []

        msg = "デプロイでエラーが出た"
        with patch.object(engine, "_get_or_create_retriever", return_value=mock_retriever):
            await engine._channel_f_episodes(["deploy"], message=msg)

        call_kwargs = mock_retriever.search.call_args
        actual_query = call_kwargs.kwargs.get("query")
        assert actual_query.startswith(msg[:200])
        assert "deploy" in actual_query

    @pytest.mark.asyncio
    async def test_channel_f_returns_empty_on_no_keywords(
        self, temp_anima_dir: Path,
    ) -> None:
        engine = PrimingEngine(temp_anima_dir)
        result = await engine._channel_f_episodes([], message="hello")
        assert result == ""

    @pytest.mark.asyncio
    async def test_channel_f_returns_empty_when_no_episodes_dir(
        self, tmp_path: Path,
    ) -> None:
        anima_dir = tmp_path / "animas" / "no_episodes"
        anima_dir.mkdir(parents=True)
        (anima_dir / "knowledge").mkdir()
        engine = PrimingEngine(anima_dir)
        result = await engine._channel_f_episodes(["test"], message="test")
        assert result == ""

    @pytest.mark.asyncio
    async def test_channel_f_returns_empty_when_retriever_unavailable(
        self, temp_anima_dir: Path,
    ) -> None:
        engine = PrimingEngine(temp_anima_dir)

        with patch.object(engine, "_get_or_create_retriever", return_value=None):
            result = await engine._channel_f_episodes(["test"], message="test")

        assert result == ""

    @pytest.mark.asyncio
    async def test_channel_f_formats_results(
        self, temp_anima_dir: Path,
    ) -> None:
        """Channel F formats retrieval results with score and source."""
        engine = PrimingEngine(temp_anima_dir)

        mock_result = MagicMock()
        mock_result.content = "デプロイ手順を確認して修正した"
        mock_result.score = 0.85
        mock_result.doc_id = "test_anima/episodes/2026-03-01.md#0"
        mock_result.metadata = {"source_file": "episodes/2026-03-01.md"}

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [mock_result]

        with patch.object(engine, "_get_or_create_retriever", return_value=mock_retriever):
            result = await engine._channel_f_episodes(
                ["deploy"], message="デプロイでエラー",
            )

        assert "Episode 1" in result
        assert "0.850" in result
        assert "episodes/2026-03-01.md" in result
        assert "デプロイ手順を確認して修正した" in result
        mock_retriever.record_access.assert_called_once()

    @pytest.mark.asyncio
    async def test_channel_f_handles_exception_gracefully(
        self, temp_anima_dir: Path,
    ) -> None:
        engine = PrimingEngine(temp_anima_dir)

        mock_retriever = MagicMock()
        mock_retriever.search.side_effect = RuntimeError("vector store error")

        with patch.object(engine, "_get_or_create_retriever", return_value=mock_retriever):
            result = await engine._channel_f_episodes(["test"], message="test")

        assert result == ""


# ── Integration with prime_memories ──────────────────────


class TestPrimeMemoriesIncludesChannelF:
    @pytest.mark.asyncio
    async def test_prime_memories_populates_episodes(
        self, temp_anima_dir: Path,
    ) -> None:
        """prime_memories() should populate the episodes field."""
        engine = PrimingEngine(temp_anima_dir)

        mock_result = MagicMock()
        mock_result.content = "Past episode content"
        mock_result.score = 0.9
        mock_result.doc_id = "test_anima/episodes/2026-02-01.md#0"
        mock_result.metadata = {"source_file": "episodes/2026-02-01.md"}

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [mock_result]

        with patch.object(engine, "_get_or_create_retriever", return_value=mock_retriever):
            result = await engine.prime_memories(
                message="What happened with the deploy?",
                sender_name="human",
            )

        assert result.episodes != ""
        assert "Past episode content" in result.episodes


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
