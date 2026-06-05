"""Unit tests for PrimingMixin query quality (sender_name, REFLECTION, recent_human_messages)."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.prompt.builder import TIER_FULL

# ── Helper ────────────────────────────────────────────────


def _make_agent(anima_dir: Path, model: str = "claude-sonnet-4-20250514"):
    """Create AgentCore with all external dependencies mocked."""
    from core.schemas import ModelConfig

    mc = ModelConfig(
        model=model,
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


def _make_priming_result(*, sender_profile: str = "", recent_activity: str = ""):
    """Create a mock PrimingResult."""
    from core.memory.priming import PrimingResult

    return PrimingResult(
        sender_profile=sender_profile,
        recent_activity=recent_activity,
    )


# ── Inbox sender_name extraction ─────────────────────────


class TestInboxSenderNameExtraction:
    """sender_name extracted correctly from trigger for inbox vs message."""

    @pytest.mark.asyncio
    async def test_inbox_single_sender(self, tmp_path):
        """trigger='inbox:sakura' → sender_name should be 'sakura' (not 'human')."""
        agent = _make_agent(tmp_path)
        mock_engine = AsyncMock()
        mock_engine.prime_memories = AsyncMock(
            return_value=_make_priming_result(sender_profile="")
        )
        agent._priming_engine = mock_engine

        await agent._run_priming(
            "DM content",
            "inbox:sakura",
            prompt_tier=TIER_FULL,
        )

        mock_engine.prime_memories.assert_called_once()
        call_args = mock_engine.prime_memories.call_args
        assert call_args.args[1] == "sakura"

    @pytest.mark.asyncio
    async def test_inbox_multiple_senders_first_used(self, tmp_path):
        """trigger='inbox:sakura,hana' → sender_name should be 'sakura' (first sender)."""
        agent = _make_agent(tmp_path)
        mock_engine = AsyncMock()
        mock_engine.prime_memories = AsyncMock(
            return_value=_make_priming_result(sender_profile="")
        )
        agent._priming_engine = mock_engine

        await agent._run_priming(
            "DM content",
            "inbox:sakura,hana",
            prompt_tier=TIER_FULL,
        )

        mock_engine.prime_memories.assert_called_once()
        call_args = mock_engine.prime_memories.call_args
        assert call_args.args[1] == "sakura"

    @pytest.mark.asyncio
    async def test_message_trigger_sender_unchanged(self, tmp_path):
        """trigger='message:yamada' → sender_name should be 'yamada'."""
        agent = _make_agent(tmp_path)
        mock_engine = AsyncMock()
        mock_engine.prime_memories = AsyncMock(
            return_value=_make_priming_result(sender_profile="")
        )
        agent._priming_engine = mock_engine

        await agent._run_priming(
            "Chat message",
            "message:yamada",
            prompt_tier=TIER_FULL,
        )

        mock_engine.prime_memories.assert_called_once()
        call_args = mock_engine.prime_memories.call_args
        assert call_args.args[1] == "yamada"

    @pytest.mark.asyncio
    async def test_inbox_empty_sender_fallback_to_human(self, tmp_path):
        """trigger='inbox:' → sender_name should fallback to 'human'."""
        agent = _make_agent(tmp_path)
        mock_engine = AsyncMock()
        mock_engine.prime_memories = AsyncMock(
            return_value=_make_priming_result(sender_profile="")
        )
        agent._priming_engine = mock_engine

        await agent._run_priming(
            "DM content",
            "inbox:",
            prompt_tier=TIER_FULL,
        )

        mock_engine.prime_memories.assert_called_once()
        call_args = mock_engine.prime_memories.call_args
        assert call_args.args[1] == "human"

    @pytest.mark.asyncio
    async def test_heartbeat_sender_human(self, tmp_path):
        """trigger='heartbeat' → sender_name should be 'human'."""
        agent = _make_agent(tmp_path)
        mock_engine = AsyncMock()
        mock_engine.prime_memories = AsyncMock(
            return_value=_make_priming_result(sender_profile="")
        )
        agent._priming_engine = mock_engine

        await agent._run_priming(
            "Heartbeat prompt",
            "heartbeat",
            prompt_tier=TIER_FULL,
        )

        mock_engine.prime_memories.assert_called_once()
        call_args = mock_engine.prime_memories.call_args
        assert call_args.args[1] == "human"


# ── Heartbeat REFLECTION query ────────────────────────────


class TestHeartbeatReflectionQuery:
    """When channel is heartbeat, message comes from reflections, not prompt."""

    @pytest.mark.asyncio
    async def test_heartbeat_message_from_reflections(self, tmp_path):
        """When channel is 'heartbeat', message should come from reflections, not prompt."""
        agent = _make_agent(tmp_path)
        mock_engine = AsyncMock()
        mock_engine.prime_memories = AsyncMock(
            return_value=_make_priming_result(sender_profile="")
        )
        agent._priming_engine = mock_engine

        with patch.object(
            agent,
            "_get_recent_reflections_text",
            return_value="[REFLECTION]今日の振り返り内容[/REFLECTION]",
        ):
            await agent._run_priming(
                "Heartbeat prompt template text",
                "heartbeat",
                prompt_tier=TIER_FULL,
            )

        mock_engine.prime_memories.assert_called_once()
        call_args = mock_engine.prime_memories.call_args
        # First positional arg is message
        assert call_args.args[0] == "[REFLECTION]今日の振り返り内容[/REFLECTION]"
        assert "Heartbeat prompt template text" not in call_args.args[0]

    @pytest.mark.asyncio
    async def test_heartbeat_no_reflections_empty_message(self, tmp_path):
        """When no reflections exist, message should be empty string."""
        agent = _make_agent(tmp_path)
        mock_engine = AsyncMock()
        mock_engine.prime_memories = AsyncMock(
            return_value=_make_priming_result(sender_profile="")
        )
        agent._priming_engine = mock_engine

        with patch.object(agent, "_get_recent_reflections_text", return_value=""):
            await agent._run_priming(
                "Heartbeat prompt",
                "heartbeat",
                prompt_tier=TIER_FULL,
            )

        mock_engine.prime_memories.assert_called_once()
        call_args = mock_engine.prime_memories.call_args
        assert call_args.args[0] == ""


# ── Chat recent human messages ───────────────────────────


class TestChatRecentHumanMessages:
    """recent_human_messages populated for chat, empty for heartbeat, graceful on error."""

    @pytest.mark.asyncio
    async def test_message_human_populates_recent_human_messages(self, tmp_path):
        """When trigger is 'message:human', recent_human_messages should be populated."""
        agent = _make_agent(tmp_path)
        agent.model_config = MagicMock()
        mock_engine = AsyncMock()
        mock_engine.prime_memories = AsyncMock(
            return_value=_make_priming_result(sender_profile="")
        )
        agent._priming_engine = mock_engine

        with patch.object(
            agent,
            "_get_recent_human_messages",
            return_value=["First human msg", "Second human msg"],
        ):
            await agent._run_priming(
                "Latest message",
                "message:human",
                prompt_tier=TIER_FULL,
            )

        mock_engine.prime_memories.assert_called_once()
        call_kw = mock_engine.prime_memories.call_args
        assert call_kw.kwargs.get("recent_human_messages") == [
            "First human msg",
            "Second human msg",
        ]

    @pytest.mark.asyncio
    async def test_heartbeat_recent_human_messages_empty(self, tmp_path):
        """When trigger is 'heartbeat', recent_human_messages should be empty list."""
        agent = _make_agent(tmp_path)
        mock_engine = AsyncMock()
        mock_engine.prime_memories = AsyncMock(
            return_value=_make_priming_result(sender_profile="")
        )
        agent._priming_engine = mock_engine

        await agent._run_priming(
            "Heartbeat",
            "heartbeat",
            prompt_tier=TIER_FULL,
        )

        mock_engine.prime_memories.assert_called_once()
        call_kw = mock_engine.prime_memories.call_args
        assert call_kw.kwargs.get("recent_human_messages") == []

    @pytest.mark.asyncio
    async def test_conversation_memory_exception_returns_empty_list(self, tmp_path):
        """When ConversationMemory raises exception, should return empty list gracefully."""
        agent = _make_agent(tmp_path)
        mock_engine = AsyncMock()
        mock_engine.prime_memories = AsyncMock(
            return_value=_make_priming_result(sender_profile="")
        )
        agent._priming_engine = mock_engine

        with patch("core.memory.conversation.ConversationMemory") as MockConv:
            mock_instance = MagicMock()
            mock_instance.load.side_effect = OSError("Conversation load failed")
            MockConv.return_value = mock_instance

            await agent._run_priming(
                "Latest message",
                "message:human",
                prompt_tier=TIER_FULL,
            )

        mock_engine.prime_memories.assert_called_once()
        call_kw = mock_engine.prime_memories.call_args
        assert call_kw.kwargs.get("recent_human_messages") == []
