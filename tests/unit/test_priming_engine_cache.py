"""Unit tests for PrimingEngine caching in AgentCore."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.schemas import ModelConfig


# ── Helper to construct AgentCore with mocked dependencies ─────


def _make_agent(anima_dir: Path):
    """Create AgentCore with all external dependencies mocked."""
    mc = ModelConfig(
        model="claude-sonnet-4-20250514",
        resolved_mode="A2",
        api_key="test-key",
    )
    memory = MagicMock()
    memory.read_permissions.return_value = ""
    memory.anima_dir = anima_dir
    messenger = MagicMock()

    with (
        patch("core.agent.ToolHandler"),
        patch("core.agent.AgentCore._check_sdk", return_value=False),
        patch("core.agent.AgentCore._init_tool_registry", return_value=[]),
        patch("core.agent.AgentCore._discover_personal_tools", return_value={}),
        patch("core.agent.AgentCore._create_executor") as mock_create,
    ):
        mock_executor = MagicMock()
        mock_create.return_value = mock_executor
        from core.agent import AgentCore

        agent = AgentCore(anima_dir, memory, mc, messenger)
        agent._executor = mock_executor
    return agent


# ── Tests ─────────────────────────────────────────────────────────


class TestPrimingEngineCache:
    @pytest.mark.asyncio
    async def test_priming_engine_cached_on_agent(self, tmp_path):
        """AgentCore should cache PrimingEngine across _run_priming calls."""
        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)
        (anima_dir / "knowledge").mkdir()
        (anima_dir / "episodes").mkdir()
        (anima_dir / "skills").mkdir()

        agent = _make_agent(anima_dir)

        mock_result = MagicMock()
        mock_result.is_empty.return_value = True

        with patch(
            "core.memory.priming.PrimingEngine.prime_memories",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            await agent._run_priming("hello", "message:user")
            await agent._run_priming("world", "message:user")

        # Verify PrimingEngine was created only once (cached on self)
        assert hasattr(agent, "_priming_engine")

    @pytest.mark.asyncio
    async def test_priming_engine_same_instance(self, tmp_path):
        """Multiple _run_priming calls should use the same PrimingEngine instance."""
        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)
        (anima_dir / "knowledge").mkdir()
        (anima_dir / "episodes").mkdir()
        (anima_dir / "skills").mkdir()

        agent = _make_agent(anima_dir)

        engines_seen: list[object] = []

        original_prime = AsyncMock()
        original_prime.return_value = MagicMock(is_empty=MagicMock(return_value=True))

        with patch(
            "core.memory.priming.PrimingEngine.__init__",
            return_value=None,
        ) as mock_init:
            with patch(
                "core.memory.priming.PrimingEngine.prime_memories",
                original_prime,
            ):
                await agent._run_priming("hello", "message:user")
                engine_1 = agent._priming_engine
                engines_seen.append(engine_1)

                await agent._run_priming("world", "message:user")
                engine_2 = agent._priming_engine
                engines_seen.append(engine_2)

                await agent._run_priming("test", "message:user")
                engine_3 = agent._priming_engine
                engines_seen.append(engine_3)

        # PrimingEngine.__init__ should only be called once
        mock_init.assert_called_once()

        # All engines should be the same object
        assert engines_seen[0] is engines_seen[1]
        assert engines_seen[1] is engines_seen[2]
