"""Unit tests for core/memory/conversation.py — ConversationMemory."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.memory.conversation import (
    ConversationMemory,
    ConversationState,
    ConversationTurn,
    _CHARS_PER_TOKEN,
    _MAX_RESPONSE_CHARS_IN_HISTORY,
)
from core.schemas import ModelConfig


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "alice"
    (d / "state").mkdir(parents=True)
    (d / "transcripts").mkdir(parents=True)
    return d


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        model="claude-sonnet-4-20250514",
        conversation_history_threshold=0.30,
    )


@pytest.fixture
def conv(anima_dir: Path, model_config: ModelConfig) -> ConversationMemory:
    return ConversationMemory(anima_dir, model_config)


# ── ConversationTurn ──────────────────────────────────────


class TestConversationTurn:
    def test_defaults(self):
        turn = ConversationTurn(role="human", content="Hello")
        assert turn.role == "human"
        assert turn.content == "Hello"
        assert turn.timestamp != ""
        assert turn.token_estimate == len("Hello") // _CHARS_PER_TOKEN

    def test_custom_values(self):
        turn = ConversationTurn(
            role="assistant",
            content="Response",
            timestamp="2026-01-01T12:00:00",
            token_estimate=100,
        )
        assert turn.timestamp == "2026-01-01T12:00:00"
        assert turn.token_estimate == 100


# ── ConversationState ─────────────────────────────────────


class TestConversationState:
    def test_defaults(self):
        state = ConversationState()
        assert state.anima_name == ""
        assert state.turns == []
        assert state.compressed_summary == ""
        assert state.compressed_turn_count == 0

    def test_total_token_estimate(self):
        state = ConversationState(
            compressed_summary="x" * 400,  # 400/4 = 100 tokens
            turns=[
                ConversationTurn(role="human", content="x" * 200),  # 50 tokens
            ],
        )
        assert state.total_token_estimate == 100 + 50

    def test_total_turn_count(self):
        state = ConversationState(
            compressed_turn_count=10,
            turns=[
                ConversationTurn(role="human", content="a"),
                ConversationTurn(role="assistant", content="b"),
            ],
        )
        assert state.total_turn_count == 12


# ── Load / Save ───────────────────────────────────────────


class TestLoadSave:
    def test_load_fresh(self, conv):
        state = conv.load()
        assert isinstance(state, ConversationState)
        assert state.anima_name == "alice"
        assert state.turns == []

    def test_save_and_load(self, conv):
        conv.append_turn("human", "Hello")
        conv.append_turn("assistant", "Hi there")
        conv.save()

        # Create new instance to test loading from disk
        conv2 = ConversationMemory(conv.anima_dir, conv.model_config)
        state = conv2.load()
        assert len(state.turns) == 2
        assert state.turns[0].role == "human"
        assert state.turns[1].content == "Hi there"

    def test_load_cached(self, conv):
        s1 = conv.load()
        s2 = conv.load()
        assert s1 is s2

    def test_load_malformed_json(self, conv, anima_dir):
        (anima_dir / "state" / "conversation.json").write_text(
            "not valid json", encoding="utf-8"
        )
        state = conv.load()
        assert state.turns == []

    def test_save_creates_state_dir(self, tmp_path, model_config):
        anima_dir = tmp_path / "animas" / "bob"
        anima_dir.mkdir(parents=True)
        conv = ConversationMemory(anima_dir, model_config)
        conv.append_turn("human", "test")
        conv.save()
        assert (anima_dir / "state" / "conversation.json").exists()


# ── Transcript ────────────────────────────────────────────


class TestTranscript:
    def test_list_transcript_dates(self, conv, anima_dir):
        (anima_dir / "transcripts" / "2026-01-15.jsonl").write_text(
            '{"role":"human","content":"a","timestamp":"ts"}\n', encoding="utf-8"
        )
        (anima_dir / "transcripts" / "2026-01-16.jsonl").write_text(
            '{"role":"human","content":"b","timestamp":"ts"}\n', encoding="utf-8"
        )
        dates = conv.list_transcript_dates()
        assert dates == ["2026-01-16", "2026-01-15"]

    def test_list_transcript_dates_empty(self, conv, anima_dir):
        assert conv.list_transcript_dates() == []

    def test_load_transcript(self, conv, anima_dir):
        (anima_dir / "transcripts" / "2026-01-15.jsonl").write_text(
            '{"role":"human","content":"msg1","timestamp":"ts1"}\n'
            '{"role":"assistant","content":"msg2","timestamp":"ts2"}\n',
            encoding="utf-8",
        )
        messages = conv.load_transcript("2026-01-15")
        assert len(messages) == 2
        assert messages[0]["content"] == "msg1"

    def test_load_transcript_invalid_date(self, conv):
        assert conv.load_transcript("invalid") == []

    def test_load_transcript_missing(self, conv):
        assert conv.load_transcript("2099-01-01") == []

    def test_load_transcript_malformed_lines(self, conv, anima_dir):
        (anima_dir / "transcripts" / "2026-01-15.jsonl").write_text(
            '{"role":"human","content":"ok","timestamp":"ts"}\n'
            'not json\n'
            '{"role":"assistant","content":"ok2","timestamp":"ts2"}\n',
            encoding="utf-8",
        )
        messages = conv.load_transcript("2026-01-15")
        assert len(messages) == 2

    def test_valid_date(self):
        assert ConversationMemory._valid_date("2026-01-15") is True
        assert ConversationMemory._valid_date("not-a-date") is False
        assert ConversationMemory._valid_date("2026-1-5") is False


# ── Mutation ──────────────────────────────────────────────


class TestAppendTurn:
    def test_appends(self, conv):
        conv.append_turn("human", "Hello")
        state = conv.load()
        assert len(state.turns) == 1
        assert state.turns[0].role == "human"

    def test_multiple_appends(self, conv):
        conv.append_turn("human", "Q1")
        conv.append_turn("assistant", "A1")
        conv.append_turn("human", "Q2")
        state = conv.load()
        assert len(state.turns) == 3


class TestClear:
    def test_clears_state(self, conv, anima_dir):
        conv.append_turn("human", "Hello")
        conv.save()
        conv.clear()
        assert not (anima_dir / "state" / "conversation.json").exists()
        state = conv.load()
        assert state.turns == []


# ── Prompt building ───────────────────────────────────────


class TestBuildChatPrompt:
    def test_no_history(self, conv, anima_dir):
        with patch("core.paths.load_prompt") as mock_load:
            mock_load.return_value = "prompt text"
            result = conv.build_chat_prompt("Hello", from_person="human")
            mock_load.assert_called_once_with(
                "chat_message", from_person="human", content="Hello"
            )

    def test_with_history(self, conv, anima_dir):
        conv.append_turn("human", "Previous question")
        conv.append_turn("assistant", "Previous answer")

        with patch("core.paths.load_prompt") as mock_load:
            mock_load.return_value = "prompt with history"
            result = conv.build_chat_prompt("New question", from_person="bob")
            mock_load.assert_called_once()
            call_args = mock_load.call_args
            assert call_args[0][0] == "chat_message_with_history"


class TestFormatHistory:
    def test_empty_history(self, conv):
        state = ConversationState(anima_name="alice")
        result = conv._format_history(state)
        assert result == ""

    def test_with_summary_only(self, conv):
        state = ConversationState(
            anima_name="alice",
            compressed_summary="Summary of past conversations",
            compressed_turn_count=10,
        )
        result = conv._format_history(state)
        assert "会話の要約" in result
        assert "Summary of past conversations" in result

    def test_with_turns_only(self, conv):
        state = ConversationState(
            anima_name="alice",
            turns=[
                ConversationTurn(
                    role="human", content="Q", timestamp="2026-01-15T10:00:00"
                ),
                ConversationTurn(
                    role="assistant", content="A", timestamp="2026-01-15T10:01:00"
                ),
            ],
        )
        result = conv._format_history(state)
        assert "human" in result
        assert "あなた" in result  # assistant label

    def test_long_assistant_response_truncated(self, conv):
        long_response = "x" * (_MAX_RESPONSE_CHARS_IN_HISTORY + 500)
        state = ConversationState(
            anima_name="alice",
            turns=[
                ConversationTurn(
                    role="assistant", content=long_response,
                    timestamp="2026-01-15T10:00:00",
                ),
            ],
        )
        result = conv._format_history(state)
        assert "..." in result


# ── Compression ───────────────────────────────────────────


class TestNeedsCompression:
    def test_few_turns_no_compression(self, conv):
        conv.append_turn("human", "Q")
        conv.append_turn("assistant", "A")
        assert conv.needs_compression() is False

    def test_many_turns_large_content(self, conv):
        # Add many large turns to exceed threshold.
        # 200k window * 0.30 threshold = 60k tokens needed.
        # Content is truncated at _MAX_STORED_CONTENT_CHARS (3000 chars),
        # so each stored turn is ~3050 chars / 4 = ~762 tokens.
        # Need at least 79 turns to exceed 60k tokens. Use 90 turns.
        for i in range(45):
            conv.append_turn("human", "x" * 8000)
            conv.append_turn("assistant", "y" * 8000)
        assert conv.needs_compression() is True


class TestCompressIfNeeded:
    async def test_no_compression_needed(self, conv):
        conv.append_turn("human", "Hello")
        result = await conv.compress_if_needed()
        assert result is False

    async def test_compression_performed(self, conv):
        # Add enough turns to trigger compression.
        # Content is truncated at _MAX_STORED_CONTENT_CHARS (3000 chars),
        # so each stored turn is ~762 tokens. Need 90 turns to exceed 60k threshold.
        for i in range(45):
            conv.append_turn("human", "x" * 8000)
            conv.append_turn("assistant", "y" * 8000)

        with patch.object(conv, "_call_compression_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Compressed summary"
            result = await conv.compress_if_needed()
            assert result is True
            state = conv.load()
            assert state.compressed_summary == "Compressed summary"
            assert state.compressed_turn_count > 0

    async def test_compression_failure_keeps_turns(self, conv):
        # Add enough turns to trigger compression (same reasoning as above).
        for i in range(45):
            conv.append_turn("human", "x" * 8000)
            conv.append_turn("assistant", "y" * 8000)

        original_count = len(conv.load().turns)
        with patch.object(conv, "_call_compression_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = RuntimeError("API error")
            result = await conv.compress_if_needed()
            assert result is True  # compression was attempted
            # Turns should be preserved on failure
            assert len(conv.load().turns) == original_count


class TestFormatTurnsForCompression:
    def test_formats(self, conv):
        turns = [
            ConversationTurn(role="human", content="Q1", timestamp="2026-01-15T10:00"),
            ConversationTurn(role="assistant", content="A1", timestamp="2026-01-15T10:01"),
        ]
        result = conv._format_turns_for_compression(turns)
        assert "human" in result
        assert "あなた" in result
        assert "Q1" in result
        assert "A1" in result
