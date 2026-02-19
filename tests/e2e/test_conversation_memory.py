# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for conversation memory (append + compression).

ConversationMemory manages a rolling conversation history per DigitalAnima.
When the token count exceeds the threshold, older turns are compressed.
"""

from __future__ import annotations

import json

import pytest

from core.memory.conversation import ConversationMemory, ConversationTurn
from core.schemas import ModelConfig
from tests.helpers.mocks import (
    make_litellm_response,
    patch_anthropic_compression,
    patch_litellm,
)


class TestConversationMemory:
    """Conversation memory tests."""

    async def test_append_and_persist(self, make_digital_anima):
        """process_message appends turns and persists to disk."""
        dp = make_digital_anima(
            name="conv-append",
            model="openai/gpt-4o",
            execution_mode="assisted",
        )

        # First message
        main_resp1 = make_litellm_response(content="Hello, nice to meet you!")
        extract_resp1 = make_litellm_response(content="なし")

        with patch_litellm(main_resp1, extract_resp1):
            await dp.process_message("Hi there", from_person="human")

        # Second message
        main_resp2 = make_litellm_response(content="I'm doing great, thanks!")
        extract_resp2 = make_litellm_response(content="なし")

        with patch_litellm(main_resp2, extract_resp2):
            await dp.process_message("How are you?", from_person="human")

        # Verify conversation.json
        conv_path = dp.anima_dir / "state" / "conversation.json"
        assert conv_path.exists()

        data = json.loads(conv_path.read_text(encoding="utf-8"))
        turns = data.get("turns", [])

        # Should have 4 turns: human, assistant, human, assistant
        assert len(turns) == 4
        assert turns[0]["role"] == "human"
        assert turns[1]["role"] == "assistant"
        assert turns[2]["role"] == "human"
        assert turns[3]["role"] == "assistant"

    async def test_compression_triggered(self, make_anima, data_dir):
        """Conversation compression fires when threshold is exceeded."""
        anima_dir = make_anima(
            name="conv-compress",
            model="claude-sonnet-4-20250514",
            conversation_history_threshold=0.001,  # Very low threshold
        )

        model_config = ModelConfig(
            model="claude-sonnet-4-20250514",
            conversation_history_threshold=0.001,
        )
        conv_mem = ConversationMemory(anima_dir, model_config)

        # Pre-populate with many turns to exceed threshold
        state = conv_mem.load()
        for i in range(20):
            state.turns.append(
                ConversationTurn(
                    role="human" if i % 2 == 0 else "assistant",
                    content=f"Turn {i}: " + "x" * 500,
                )
            )
        conv_mem.save()

        assert conv_mem.needs_compression()

        # Mock the compression LLM call
        with patch_anthropic_compression(
            summary_text="Summary of 20 conversation turns about various topics."
        ):
            compressed = await conv_mem.compress_if_needed()

        assert compressed is True

        # Verify state after compression
        fresh = ConversationMemory(anima_dir, model_config)
        state = fresh.load()
        assert state.compressed_summary
        assert "Summary" in state.compressed_summary
        assert state.compressed_turn_count > 0
        # Fewer turns remain after compression (kept 25% = 5 turns)
        assert len(state.turns) < 20
