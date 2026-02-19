# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for consolidation prompt improvements and LLM response logging.

Verifies:
- Updated prompt contains extraction requirements for specific info types
- LLM response is logged at INFO level after _summarize_episodes
- Empty response triggers WARNING log
- Zero-file extraction triggers WARNING log
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.memory.consolidation import ConsolidationEngine


class TestConsolidationPromptContent:
    """Verify the consolidation prompt includes updated extraction requirements."""

    @pytest.fixture
    def engine(self, tmp_path: Path) -> ConsolidationEngine:
        anima_dir = tmp_path / "animas" / "sakura"
        anima_dir.mkdir(parents=True)
        return ConsolidationEngine(anima_dir, "sakura")

    @pytest.mark.asyncio
    async def test_prompt_requires_specific_info_extraction(self, engine):
        """The prompt must instruct LLM to extract IDs, credentials, procedures, etc."""
        episodes = [
            {"date": "2026-02-16", "time": "10:00", "content": "Setup API key at /home/.token"},
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = ""
            mock_llm.return_value = mock_response

            await engine._summarize_episodes(episodes, [], "test-model")

            # Extract the prompt sent to the LLM
            call_args = mock_llm.call_args
            prompt = call_args.kwargs["messages"][0]["content"]

            # Verify key extraction requirements are in the prompt
            assert "APIキー" in prompt or "設定値" in prompt
            assert "識別情報" in prompt
            assert "手順" in prompt or "ワークフロー" in prompt
            assert "チーム構成" in prompt or "役割分担" in prompt
            assert "技術的な判断" in prompt

    @pytest.mark.asyncio
    async def test_prompt_no_longer_dismisses_routine_exchanges(self, engine):
        """The old instruction '些細な会話や定型的なやり取りは知識化不要' should be removed."""
        episodes = [
            {"date": "2026-02-16", "time": "10:00", "content": "test"},
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = ""
            mock_llm.return_value = mock_response

            await engine._summarize_episodes(episodes, [], "test-model")

            prompt = mock_llm.call_args.kwargs["messages"][0]["content"]
            # Old broad dismissal should be gone
            assert "些細な会話や定型的なやり取りは知識化不要です" not in prompt


class TestConsolidationLLMResponseLogging:
    """Verify LLM responses are logged for observability."""

    @pytest.fixture
    def engine(self, tmp_path: Path) -> ConsolidationEngine:
        anima_dir = tmp_path / "animas" / "sakura"
        anima_dir.mkdir(parents=True)
        return ConsolidationEngine(anima_dir, "sakura")

    @pytest.mark.asyncio
    async def test_llm_response_logged_at_info(self, engine, caplog):
        """LLM response should be logged at INFO level."""
        episodes = [
            {"date": "2026-02-16", "time": "10:00", "content": "test content"},
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "## 新規ファイル作成\n- ファイル名: test.md\n  内容: test"
            mock_llm.return_value = mock_response

            with caplog.at_level(logging.INFO, logger="animaworks.consolidation"):
                result = await engine._summarize_episodes(episodes, [], "test-model")

            assert result != ""
            assert any("Consolidation LLM response for sakura" in r.message for r in caplog.records)


class TestMergeToKnowledgeLogging:
    """Verify _merge_to_knowledge logs warnings for empty/zero results."""

    @pytest.fixture
    def engine(self, tmp_path: Path) -> ConsolidationEngine:
        anima_dir = tmp_path / "animas" / "sakura"
        anima_dir.mkdir(parents=True)
        return ConsolidationEngine(anima_dir, "sakura")

    def test_empty_response_logs_warning(self, engine, caplog):
        with caplog.at_level(logging.WARNING, logger="animaworks.consolidation"):
            created, updated = engine._merge_to_knowledge("")

        assert created == []
        assert updated == []
        assert any("Empty consolidation LLM response" in r.message for r in caplog.records)

    def test_whitespace_only_response_logs_warning(self, engine, caplog):
        with caplog.at_level(logging.WARNING, logger="animaworks.consolidation"):
            created, updated = engine._merge_to_knowledge("   \n\n  ")

        assert created == []
        assert updated == []
        assert any("Empty consolidation LLM response" in r.message for r in caplog.records)

    def test_no_parseable_entries_logs_warning(self, engine, caplog):
        """LLM returns text but no parseable file entries."""
        response = "この会話は知識化する必要はありません。日常的なやり取りのみです。"

        with caplog.at_level(logging.WARNING, logger="animaworks.consolidation"):
            created, updated = engine._merge_to_knowledge(response)

        assert created == []
        assert updated == []
        assert any("No knowledge files extracted" in r.message for r in caplog.records)

    def test_successful_extraction_no_warning(self, engine, caplog):
        """When files are successfully created, no zero-result warning should appear."""
        response = (
            "## 新規ファイル作成\n"
            "- ファイル名: knowledge/test-info.md\n"
            "  内容: # Test Info\n\nSome useful knowledge."
        )

        with caplog.at_level(logging.WARNING, logger="animaworks.consolidation"):
            created, updated = engine._merge_to_knowledge(response)

        assert len(created) == 1
        assert "test-info.md" in created
        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert not any("No knowledge files extracted" in m for m in warning_msgs)
