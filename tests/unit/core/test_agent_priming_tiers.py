"""Unit tests for _run_priming tier-aware behaviour in AgentCore."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.prompt.builder import (
    TIER_FULL,
    TIER_LIGHT,
    TIER_MICRO,
    TIER_MINIMAL,
    TIER_STANDARD,
)
from core.schemas import ModelConfig


# ── Helper ────────────────────────────────────────────────


def _make_agent(anima_dir: Path, model: str = "claude-sonnet-4-20250514"):
    """Create AgentCore with all external dependencies mocked."""
    mc = ModelConfig(
        model=model,
        api_key="test-key",
    )
    memory = MagicMock()
    memory.read_permissions.return_value = ""
    memory.anima_dir = anima_dir
    memory.collect_distilled_knowledge.return_value = []
    messenger = MagicMock()

    with patch("core.agent.ToolHandler"), \
         patch("core.agent.AgentCore._check_sdk", return_value=False), \
         patch("core.agent.AgentCore._init_tool_registry", return_value=[]), \
         patch("core.agent.AgentCore._discover_personal_tools", return_value={}), \
         patch("core.agent.AgentCore._create_executor") as mock_create:
        mock_executor = MagicMock()
        mock_create.return_value = mock_executor
        from core.agent import AgentCore
        agent = AgentCore(anima_dir, memory, mc, messenger)
        agent._executor = mock_executor
    return agent


def _make_priming_result(*, sender_profile: str = "", recent_activity: str = ""):
    """Create a mock PrimingResult."""
    from core.memory.priming import PrimingResult
    return PrimingResult(
        sender_profile=sender_profile,
        recent_activity=recent_activity,
    )


# ── T4 Minimal: priming skipped entirely ──────────────────


class TestPrimingTierMinimal:
    """T4 (< 16k): _run_priming must return ('', '') without calling prime_memories."""

    @pytest.mark.asyncio
    async def test_minimal_returns_empty(self, tmp_path):
        agent = _make_agent(tmp_path)
        result = await agent._run_priming(
            "hello", "message:human",
            prompt_tier=TIER_MINIMAL,
        )
        assert result == ("", "")

    @pytest.mark.asyncio
    async def test_minimal_does_not_call_priming_engine(self, tmp_path):
        agent = _make_agent(tmp_path)
        with patch("core.memory.priming.PrimingEngine") as mock_pe:
            await agent._run_priming(
                "hello", "message:human",
                prompt_tier=TIER_MINIMAL,
            )
            mock_pe.assert_not_called()


# ── T5 Micro: priming skipped (same as minimal) ──────────


class TestPrimingTierMicro:
    """T5 (< 8k): _run_priming must return ('', '') like minimal."""

    @pytest.mark.asyncio
    async def test_micro_returns_empty(self, tmp_path):
        agent = _make_agent(tmp_path)
        result = await agent._run_priming(
            "hello", "message:human",
            prompt_tier=TIER_MICRO,
        )
        assert result == ("", "")

    @pytest.mark.asyncio
    async def test_micro_does_not_call_priming_engine(self, tmp_path):
        agent = _make_agent(tmp_path)
        with patch("core.memory.priming.PrimingEngine") as mock_pe:
            await agent._run_priming(
                "hello", "message:human",
                prompt_tier=TIER_MICRO,
            )
            mock_pe.assert_not_called()


# ── T3 Light: sender_profile only ─────────────────────────


class TestPrimingTierLight:
    """T3 (16k–32k): only sender_profile section is returned."""

    @pytest.mark.asyncio
    async def test_light_returns_sender_profile_only(self, tmp_path):
        agent = _make_agent(tmp_path)
        priming_result = _make_priming_result(
            sender_profile="This is sender info",
            recent_activity="Recent activity data",
        )

        mock_engine = AsyncMock()
        mock_engine.prime_memories = AsyncMock(return_value=priming_result)
        agent._priming_engine = mock_engine

        section, notifications = await agent._run_priming(
            "hello", "message:yamada",
            prompt_tier=TIER_LIGHT,
        )

        assert "This is sender info" in section
        assert "yamada" in section
        assert "Recent activity" not in section

    @pytest.mark.asyncio
    async def test_light_empty_profile_returns_empty(self, tmp_path):
        agent = _make_agent(tmp_path)
        priming_result = _make_priming_result(
            sender_profile="",
            recent_activity="Some activity",
        )

        mock_engine = AsyncMock()
        mock_engine.prime_memories = AsyncMock(return_value=priming_result)
        agent._priming_engine = mock_engine

        section, notifications = await agent._run_priming(
            "hello", "message:human",
            prompt_tier=TIER_LIGHT,
        )

        assert section == ""


# ── T2 Standard: truncation to 4000 chars ─────────────────


class TestPrimingTierStandard:
    """T2 (32k–128k): formatted priming truncated to 4000 chars."""

    @pytest.mark.asyncio
    async def test_standard_truncates_long_priming(self, tmp_path):
        agent = _make_agent(tmp_path)
        priming_result = _make_priming_result(
            sender_profile="x" * 2000,
            recent_activity="y" * 3000,
        )

        mock_engine = AsyncMock()
        mock_engine.prime_memories = AsyncMock(return_value=priming_result)
        agent._priming_engine = mock_engine

        section, notifications = await agent._run_priming(
            "hello", "message:human",
            prompt_tier=TIER_STANDARD,
        )

        assert len(section) <= 4000 + len("\n\n（以降省略）")
        assert "（以降省略）" in section

    @pytest.mark.asyncio
    async def test_standard_short_priming_not_truncated(self, tmp_path):
        agent = _make_agent(tmp_path)
        priming_result = _make_priming_result(
            sender_profile="short profile",
        )

        mock_engine = AsyncMock()
        mock_engine.prime_memories = AsyncMock(return_value=priming_result)
        agent._priming_engine = mock_engine

        section, notifications = await agent._run_priming(
            "hello", "message:human",
            prompt_tier=TIER_STANDARD,
        )

        assert "（以降省略）" not in section
        assert "short profile" in section


# ── T1 Full: no change to existing behaviour ──────────────


class TestPrimingTierFull:
    """T1 (128k+): default behaviour, no truncation."""

    @pytest.mark.asyncio
    async def test_full_returns_complete_priming(self, tmp_path):
        agent = _make_agent(tmp_path)
        priming_result = _make_priming_result(
            sender_profile="x" * 2000,
            recent_activity="y" * 3000,
        )

        mock_engine = AsyncMock()
        mock_engine.prime_memories = AsyncMock(return_value=priming_result)
        agent._priming_engine = mock_engine

        section, notifications = await agent._run_priming(
            "hello", "message:human",
            prompt_tier=TIER_FULL,
        )

        assert "（以降省略）" not in section
        assert "x" * 100 in section
        assert "y" * 100 in section
