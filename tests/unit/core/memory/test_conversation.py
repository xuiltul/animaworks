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
    ToolRecord,
    _CHARS_PER_TOKEN,
    _MAX_DISPLAY_TURNS,
    _MAX_RESPONSE_CHARS_IN_HISTORY,
    _MAX_TURNS_BEFORE_COMPRESS,
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
        with patch("core.memory.conversation.load_prompt") as mock_load:
            mock_load.return_value = "prompt text"
            result = conv.build_chat_prompt("Hello", from_person="human")
            mock_load.assert_called_once_with(
                "chat_message", from_person="human", content="Hello"
            )

    def test_with_history(self, conv, anima_dir):
        conv.append_turn("human", "Previous question")
        conv.append_turn("assistant", "Previous answer")

        with patch("core.memory.conversation.load_prompt") as mock_load:
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


# ── Compression auto-scale ─────────────────────────────────


class TestNeedsCompressionAutoScale:
    """Verify that needs_compression() auto-scales threshold for small windows."""

    def _make_conv(self, anima_dir: Path, configured_threshold: float = 0.30) -> ConversationMemory:
        """Create a ConversationMemory with a specific configured threshold."""
        mc = ModelConfig(
            model="claude-sonnet-4-20250514",
            conversation_history_threshold=configured_threshold,
        )
        return ConversationMemory(anima_dir, mc)

    def _add_turns_with_tokens(self, conv: ConversationMemory, total_tokens: int) -> None:
        """Add enough turns to reach approximately total_tokens estimate.

        Each turn: content is 4 * token_estimate chars long (since _CHARS_PER_TOKEN=4).
        We need at least 4 turns so needs_compression doesn't bail early.
        """
        turns_count = max(4, 8)  # Enough turns to pass the min-4 check
        tokens_per_turn = total_tokens // turns_count
        chars_per_turn = tokens_per_turn * _CHARS_PER_TOKEN
        for i in range(turns_count):
            role = "human" if i % 2 == 0 else "assistant"
            # Use short chars to avoid _MAX_STORED_CONTENT_CHARS truncation (3000)
            # If chars_per_turn > 3000, stored content is truncated to ~3050
            conv.append_turn(role, "x" * min(chars_per_turn, 2900))

    def test_small_model_auto_scales_down(self, anima_dir: Path):
        """16K window → auto_threshold = max(0.10, 16000/64000*0.30) = max(0.10, 0.075) = 0.10.

        effective = min(0.30, 0.10) = 0.10.
        threshold_tokens = 16000 * 0.10 = 1600.
        """
        conv = self._make_conv(anima_dir, configured_threshold=0.30)
        with patch(
            "core.prompt.context.resolve_context_window", return_value=16_000,
        ):
            # Add turns totalling ~2000 tokens → exceeds 1600 threshold
            self._add_turns_with_tokens(conv, 2000)
            assert conv.needs_compression() is True

            # Reset and add turns totalling ~1000 tokens → below 1600 threshold
            conv._state = None
            conv2 = self._make_conv(anima_dir, configured_threshold=0.30)
            # 4 turns with minimal content: 4 * (10/4) = 10 tokens
            for i in range(4):
                conv2.append_turn("human" if i % 2 == 0 else "assistant", "x" * 40)
            # 4 turns * 10 tokens = 40 tokens total, well below 1600
            assert conv2.needs_compression() is False

    def test_medium_model_auto_scales(self, anima_dir: Path):
        """32K window → auto = max(0.10, 32000/64000*0.30) = 0.15.

        effective = min(0.30, 0.15) = 0.15.
        threshold_tokens = 32000 * 0.15 = 4800.
        """
        conv = self._make_conv(anima_dir, configured_threshold=0.30)
        with patch(
            "core.prompt.context.resolve_context_window", return_value=32_000,
        ):
            # Add turns totalling ~5500 tokens → exceeds 4800 threshold
            self._add_turns_with_tokens(conv, 5500)
            assert conv.needs_compression() is True

    def test_large_model_uses_configured(self, anima_dir: Path):
        """128K window (>= 64K) → effective = configured = 0.30.

        threshold_tokens = 128000 * 0.30 = 38400.
        """
        conv = self._make_conv(anima_dir, configured_threshold=0.30)
        with patch(
            "core.prompt.context.resolve_context_window", return_value=128_000,
        ):
            # 8 turns * ~700 tokens each ≈ 5600 tokens, well below 38400
            self._add_turns_with_tokens(conv, 5600)
            assert conv.needs_compression() is False

    def test_configured_lower_than_auto_uses_configured(self, anima_dir: Path):
        """configured=0.05, 32K window → auto=0.15, effective = min(0.05, 0.15) = 0.05.

        threshold_tokens = 32000 * 0.05 = 1600.
        """
        conv = self._make_conv(anima_dir, configured_threshold=0.05)
        with patch(
            "core.prompt.context.resolve_context_window", return_value=32_000,
        ):
            # Add turns totalling ~2000 tokens → exceeds 1600 threshold
            self._add_turns_with_tokens(conv, 2000)
            assert conv.needs_compression() is True


# ── Display turn limit (_format_history) ─────────────────


class TestFormatHistoryDisplayLimit:
    """Verify that _format_history() only includes the last _MAX_DISPLAY_TURNS turns."""

    @staticmethod
    def _make_state(n_turns: int) -> ConversationState:
        """Create a ConversationState with *n_turns* turns."""
        turns = [
            ConversationTurn(
                role="human" if i % 2 == 0 else "assistant",
                content=f"turn-{i}",
                timestamp=f"2026-01-15T10:{i:02d}:00",
            )
            for i in range(n_turns)
        ]
        return ConversationState(anima_name="alice", turns=turns)

    def test_zero_turns_returns_empty(self, conv):
        state = self._make_state(0)
        assert conv._format_history(state) == ""

    def test_under_limit_all_displayed(self, conv):
        state = self._make_state(5)
        result = conv._format_history(state)
        for i in range(5):
            assert f"turn-{i}" in result

    def test_at_limit_all_displayed(self, conv):
        state = self._make_state(_MAX_DISPLAY_TURNS)
        result = conv._format_history(state)
        for i in range(_MAX_DISPLAY_TURNS):
            assert f"turn-{i}" in result

    def test_over_limit_only_last_n_displayed(self, conv):
        n = 25
        state = self._make_state(n)
        result = conv._format_history(state)
        # Turns 0..4 (earliest 5) should be excluded.
        # Use newline boundary to avoid substring matches (e.g. "turn-1" in "turn-10").
        for i in range(n - _MAX_DISPLAY_TURNS):
            assert f"\nturn-{i}\n" not in result
        # Last 20 turns should be present
        for i in range(n - _MAX_DISPLAY_TURNS, n):
            assert f"turn-{i}" in result

    def test_large_count_only_last_n_displayed(self, conv):
        n = 300
        state = self._make_state(n)
        result = conv._format_history(state)
        # Only the last _MAX_DISPLAY_TURNS turns should appear
        for i in range(n - _MAX_DISPLAY_TURNS, n):
            assert f"turn-{i}" in result
        # An early turn should NOT appear
        assert "turn-0" not in result
        assert "turn-100" not in result


# ── Turn-count trigger (needs_compression) ───────────────


class TestNeedsCompressionTurnCount:
    """Verify that needs_compression() triggers on turn count > _MAX_TURNS_BEFORE_COMPRESS."""

    def test_three_turns_false(self, conv):
        """3 turns is below the min-4 early exit; always False."""
        for i in range(3):
            conv.append_turn("human" if i % 2 == 0 else "assistant", "x")
        assert conv.needs_compression() is False

    def test_49_turns_no_turn_count_trigger(self, conv):
        """49 turns: below _MAX_TURNS_BEFORE_COMPRESS, should NOT trigger the turn-count path.

        With a large context window and small content, token budget is not exceeded either.
        """
        for i in range(49):
            conv.append_turn("human" if i % 2 == 0 else "assistant", "short")
        with patch(
            "core.prompt.context.resolve_context_window", return_value=200_000,
        ):
            assert conv.needs_compression() is False

    def test_51_turns_true_regardless_of_budget(self, conv):
        """51 turns > _MAX_TURNS_BEFORE_COMPRESS → True, even with a huge context window."""
        for i in range(51):
            conv.append_turn("human" if i % 2 == 0 else "assistant", "tiny")
        # No need to mock resolve_context_window — the turn-count path
        # returns True before the token-budget check is reached.
        assert conv.needs_compression() is True

    def test_100_turns_true(self, conv):
        """100 turns >> _MAX_TURNS_BEFORE_COMPRESS → True."""
        for i in range(100):
            conv.append_turn("human" if i % 2 == 0 else "assistant", "x")
        assert conv.needs_compression() is True


# ── Fixed keep count (_compress) ─────────────────────────


class TestCompressKeepCount:
    """Verify that _compress() uses keep_count = min(_MAX_DISPLAY_TURNS, len-1)."""

    @pytest.mark.asyncio
    async def test_51_turns_keeps_max_display_compresses_rest(self, conv):
        """51 turns → keep _MAX_DISPLAY_TURNS, compress the rest."""
        for i in range(51):
            conv.append_turn("human" if i % 2 == 0 else "assistant", f"turn-{i}")

        keep = _MAX_DISPLAY_TURNS
        compress = 51 - keep
        with patch.object(conv, "_call_compression_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = f"Compressed summary of {compress} turns"
            await conv._compress()

            state = conv.load()
            assert len(state.turns) == keep
            assert state.compressed_turn_count == compress
            assert state.compressed_summary == f"Compressed summary of {compress} turns"
            # The kept turns should be the last _MAX_DISPLAY_TURNS
            assert state.turns[0].content == f"turn-{compress}"
            assert state.turns[-1].content == "turn-50"

    @pytest.mark.asyncio
    async def test_25_turns_keeps_max_display_compresses_rest(self, conv):
        """25 turns → keep min(_MAX_DISPLAY_TURNS, 24), compress the rest."""
        for i in range(25):
            conv.append_turn("human" if i % 2 == 0 else "assistant", f"turn-{i}")

        keep = min(_MAX_DISPLAY_TURNS, 24)
        compress = 25 - keep
        with patch.object(conv, "_call_compression_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = f"Compressed summary of {compress} turns"
            await conv._compress()

            state = conv.load()
            assert len(state.turns) == keep
            assert state.compressed_turn_count == compress
            assert state.compressed_summary == f"Compressed summary of {compress} turns"
            # The kept turns should be the last `keep`
            assert state.turns[0].content == f"turn-{compress}"
            assert state.turns[-1].content == "turn-24"


# ── ToolRecord ───────────────────────────────────────────


class TestToolRecord:
    """Test ToolRecord dataclass including is_error field."""

    def test_is_error_defaults_to_false(self):
        """is_error should default to False when not specified."""
        record = ToolRecord(tool_name="web_search")
        assert record.is_error is False

    def test_is_error_can_be_set_to_true(self):
        """is_error should accept True as an explicit value."""
        record = ToolRecord(tool_name="web_search", is_error=True)
        assert record.is_error is True

    def test_from_dict_with_is_error_true(self):
        """from_dict should correctly set is_error=True from dict."""
        d = {
            "tool_name": "Bash",
            "tool_id": "toolu_123",
            "input_summary": "ls -la",
            "result_summary": "error output",
            "is_error": True,
        }
        record = ToolRecord.from_dict(d)
        assert record.tool_name == "Bash"
        assert record.tool_id == "toolu_123"
        assert record.input_summary == "ls -la"
        assert record.result_summary == "error output"
        assert record.is_error is True

    def test_from_dict_with_is_error_false(self):
        """from_dict should correctly set is_error=False from dict."""
        d = {
            "tool_name": "Read",
            "is_error": False,
        }
        record = ToolRecord.from_dict(d)
        assert record.is_error is False

    def test_from_dict_without_is_error_key(self):
        """from_dict should default is_error to False when key is absent (backward compat)."""
        d = {
            "tool_name": "web_search",
            "tool_id": "toolu_456",
            "input_summary": "query=test",
            "result_summary": "3 results",
        }
        record = ToolRecord.from_dict(d)
        assert record.is_error is False

    def test_from_dict_empty_dict(self):
        """from_dict with empty dict should use all defaults."""
        record = ToolRecord.from_dict({})
        assert record.tool_name == ""
        assert record.tool_id == ""
        assert record.input_summary == ""
        assert record.result_summary == ""
        assert record.is_error is False
