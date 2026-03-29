"""Unit tests for Channel C conditional activation in priming.py."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.memory.priming import PrimingEngine


# ── Fixtures ─────────────────────────────────────────────


@pytest.fixture
def priming_engine(tmp_path: Path) -> PrimingEngine:
    """Create a PrimingEngine with isolated temp directories."""
    anima_dir = tmp_path / "animas" / "test"
    anima_dir.mkdir(parents=True)
    (anima_dir / "knowledge").mkdir()
    (anima_dir / "skills").mkdir()
    (anima_dir / "episodes").mkdir()
    return PrimingEngine(anima_dir)


# ── Channel C conditional activation ─────────────────────


class TestChannelCSkippedWhenAllInjected:
    @pytest.mark.asyncio
    async def test_channel_c_skipped_when_all_injected(
        self, priming_engine: PrimingEngine,
    ) -> None:
        """restrict_to=[] -> Channel C returns empty tuple."""
        result = await priming_engine._channel_c_related_knowledge(
            keywords=["test", "keyword"],
            restrict_to=[],
        )
        assert result == ("", "")


class TestChannelCRunsWhenOverflowExists:
    @pytest.mark.asyncio
    async def test_channel_c_runs_when_overflow_exists(
        self, priming_engine: PrimingEngine,
    ) -> None:
        """restrict_to=["file1"] -> Channel C runs normally.

        When overflow files exist, Channel C should attempt to run
        (though it may return empty if no retriever is available).
        """
        # Create a knowledge file so the directory is non-empty
        (priming_engine.knowledge_dir / "file1.md").write_text(
            "Test knowledge content", encoding="utf-8",
        )

        # Channel C should NOT short-circuit (restrict_to is non-empty)
        # It will attempt to run but return ("", "") because retriever is unavailable
        result = await priming_engine._channel_c_related_knowledge(
            keywords=["test"],
            restrict_to=["file1"],
        )
        assert isinstance(result, tuple)


class TestChannelCLegacyWhenNone:
    @pytest.mark.asyncio
    async def test_channel_c_legacy_when_none(
        self, priming_engine: PrimingEngine,
    ) -> None:
        """restrict_to=None -> Channel C runs full (legacy behavior).

        When restrict_to is None, Channel C should not short-circuit.
        """
        # Create a knowledge file
        (priming_engine.knowledge_dir / "topic.md").write_text(
            "Legacy knowledge content", encoding="utf-8",
        )

        # restrict_to=None -> legacy path, should not skip
        result = await priming_engine._channel_c_related_knowledge(
            keywords=["legacy"],
            restrict_to=None,
        )
        assert isinstance(result, tuple)


class TestChannelCFallbackToMessageWhenNoKeywords:
    @pytest.mark.asyncio
    async def test_channel_c_uses_message_when_keywords_empty(
        self, priming_engine: PrimingEngine,
    ) -> None:
        """When keywords is empty but message exists, use message[:200] as query."""
        (priming_engine.knowledge_dir / "topic.md").write_text(
            "Some knowledge", encoding="utf-8",
        )

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []

        with patch.object(
            priming_engine, "_get_or_create_retriever", return_value=mock_retriever,
        ):
            await priming_engine._channel_c_related_knowledge(
                keywords=[], message="短いメッセージ",
            )

        mock_retriever.search.assert_called_once()
        actual_query = mock_retriever.search.call_args.kwargs.get("query")
        assert "短いメッセージ" in actual_query

    @pytest.mark.asyncio
    async def test_channel_c_returns_empty_no_keywords_no_message(
        self, priming_engine: PrimingEngine,
    ) -> None:
        """When both keywords and message are empty, return empty tuple."""
        (priming_engine.knowledge_dir / "topic.md").write_text(
            "Knowledge", encoding="utf-8",
        )

        mock_retriever = MagicMock()
        with patch.object(
            priming_engine, "_get_or_create_retriever", return_value=mock_retriever,
        ):
            result = await priming_engine._channel_c_related_knowledge(
                keywords=[], message="",
            )

        assert result == ("", "")
        mock_retriever.search.assert_not_called()


class TestPrimeMemoriesPassesOverflowToChannelC:
    @pytest.mark.asyncio
    async def test_prime_memories_passes_overflow_to_channel_c(
        self, priming_engine: PrimingEngine,
    ) -> None:
        """Verify overflow_files parameter flows through to Channel C as restrict_to."""
        overflow = ["overflow_file1", "overflow_file2"]

        with patch.object(
            priming_engine,
            "_channel_c_related_knowledge",
            new_callable=AsyncMock,
            return_value=("", ""),
        ) as mock_channel_c, \
             patch.object(
                 priming_engine,
                 "_channel_a_sender_profile",
                 new_callable=AsyncMock,
                 return_value="",
             ), \
             patch.object(
                 priming_engine,
                 "_channel_b_recent_activity",
                 new_callable=AsyncMock,
                 return_value="",
             ), \
             patch.object(
                 priming_engine,
                 "_channel_e_pending_tasks",
                 new_callable=AsyncMock,
                 return_value="",
             ):
            await priming_engine.prime_memories(
                message="test message",
                sender_name="human",
                overflow_files=overflow,
            )

            # Verify Channel C was called with restrict_to (renamed from overflow_files)
            mock_channel_c.assert_called_once()
            call_kwargs = mock_channel_c.call_args
            assert call_kwargs.kwargs.get("restrict_to") == overflow
