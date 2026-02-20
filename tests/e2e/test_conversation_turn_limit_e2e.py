# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for conversation history turn limit feature.

Verifies the full workflow of the turn limit mechanism:
  1. _format_history() caps output to the last 20 turns (_MAX_DISPLAY_TURNS)
  2. needs_compression() triggers when turns exceed _MAX_TURNS_BEFORE_COMPRESS (50)
  3. compress_if_needed() reduces stored turns to 20 (_MAX_DISPLAY_TURNS)

Uses real file I/O with tmp_path; only the LLM call and context window
resolution are mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from core.memory.conversation import ConversationMemory, ConversationTurn
from core.schemas import ModelConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conv_memory(
    anima_dir: Path,
    *,
    model: str = "claude-sonnet-4-20250514",
    conversation_history_threshold: float = 0.30,
) -> ConversationMemory:
    """Create a ConversationMemory backed by a real temp directory."""
    (anima_dir / "state").mkdir(parents=True, exist_ok=True)
    (anima_dir / "transcripts").mkdir(parents=True, exist_ok=True)

    model_config = ModelConfig(
        model=model,
        conversation_history_threshold=conversation_history_threshold,
    )
    return ConversationMemory(anima_dir, model_config)


def _populate_turns(conv: ConversationMemory, n: int) -> None:
    """Append *n* alternating human/assistant turns and persist to disk."""
    state = conv.load()
    for i in range(n):
        role = "human" if i % 2 == 0 else "assistant"
        conv.append_turn(role, f"Turn {i}: short message")
    conv.save()


def _read_conversation_json(anima_dir: Path) -> dict:
    """Load the raw conversation.json from disk."""
    path = anima_dir / "state" / "conversation.json"
    assert path.exists(), f"conversation.json not found at {path}"
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConversationTurnLimitE2E:
    """E2E tests for the conversation turn limit feature."""

    def test_format_history_limits_to_20_turns_e2e(self, tmp_path: Path):
        """Append 100 turns, build_chat_prompt, verify only last 20 turns
        appear in the formatted output."""
        anima_dir = tmp_path / "animas" / "turn-limit-format"
        conv = _make_conv_memory(anima_dir)

        # Append 100 turns
        _populate_turns(conv, 100)

        # Verify all 100 turns are stored on disk
        data = _read_conversation_json(anima_dir)
        assert len(data["turns"]) == 100

        # Build prompt - should only include last 20 turns
        prompt = conv.build_chat_prompt("New message", from_person="human")

        # The last turn appended was Turn 99 (assistant).
        # The 20 display turns are indices 80..99.
        # Turn 80 (human) should be the oldest visible turn.
        assert "Turn 80:" in prompt
        assert "Turn 99:" in prompt

        # Turn 79 should NOT appear (it's the 21st from the end)
        assert "Turn 79:" not in prompt

        # Verify a few more boundary turns are absent
        assert "Turn 0:" not in prompt
        assert "Turn 50:" not in prompt

    @pytest.mark.asyncio
    async def test_compression_triggers_at_51_turns_e2e(self, tmp_path: Path):
        """Append 51 turns, verify needs_compression() is True, run
        compress_if_needed(), verify only 20 turns remain."""
        anima_dir = tmp_path / "animas" / "turn-limit-compress"
        conv = _make_conv_memory(anima_dir)

        # Append 51 turns (exceeds _MAX_TURNS_BEFORE_COMPRESS = 50)
        _populate_turns(conv, 51)

        # Verify stored count
        data = _read_conversation_json(anima_dir)
        assert len(data["turns"]) == 51

        # needs_compression() should trigger on the turn-count rule
        with patch(
            "core.prompt.context.resolve_context_window",
            return_value=200_000,
        ):
            assert conv.needs_compression() is True

        # Run compression with mocked LLM
        with (
            patch(
                "core.prompt.context.resolve_context_window",
                return_value=200_000,
            ),
            patch.object(
                conv,
                "_call_compression_llm",
                new_callable=AsyncMock,
                return_value="Summary of older conversation turns.",
            ),
        ):
            result = await conv.compress_if_needed()

        assert result is True

        # Reload from disk and verify
        conv_fresh = _make_conv_memory(anima_dir)
        state = conv_fresh.load()

        # After compression, exactly 20 recent turns are kept
        assert len(state.turns) == 20
        # Compressed summary should be set
        assert "Summary" in state.compressed_summary
        # Compressed turn count should reflect the removed turns
        assert state.compressed_turn_count == 31  # 51 - 20

        # The kept turns should be the last 20 (Turn 31..50)
        assert "Turn 31:" in state.turns[0].content
        assert "Turn 50:" in state.turns[-1].content

    @pytest.mark.asyncio
    async def test_no_compression_at_49_turns_e2e(self, tmp_path: Path):
        """Append 49 turns with short content, verify needs_compression()
        is False because neither turn count nor token budget is exceeded."""
        anima_dir = tmp_path / "animas" / "turn-limit-no-compress"
        conv = _make_conv_memory(anima_dir)

        # Append 49 turns (below _MAX_TURNS_BEFORE_COMPRESS = 50)
        _populate_turns(conv, 49)

        # Verify stored count
        data = _read_conversation_json(anima_dir)
        assert len(data["turns"]) == 49

        # With a 200K context window and short messages, 49 turns
        # won't exceed the token budget either
        with patch(
            "core.prompt.context.resolve_context_window",
            return_value=200_000,
        ):
            assert conv.needs_compression() is False

        # compress_if_needed should be a no-op
        with patch(
            "core.prompt.context.resolve_context_window",
            return_value=200_000,
        ):
            result = await conv.compress_if_needed()

        assert result is False

        # All 49 turns should still be intact
        conv_fresh = _make_conv_memory(anima_dir)
        state = conv_fresh.load()
        assert len(state.turns) == 49
