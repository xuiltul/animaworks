"""Unit tests for _build_recent_tool_section() in core/prompt/builder.py.

Tests the Recent Tool Results section that summarizes tool call records
from ConversationMemory for injection into the system prompt.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory.conversation import (
    ConversationState,
    ConversationTurn,
    ToolRecord,
)
from core.prompt.builder import _build_recent_tool_section
from core.schemas import ModelConfig


# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "sakura"
    (d / "state").mkdir(parents=True)
    (d / "transcripts").mkdir(parents=True)
    return d


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        model="claude-sonnet-4-20250514",
        conversation_history_threshold=0.30,
    )


def _make_state(turns: list[ConversationTurn]) -> ConversationState:
    """Helper to build a ConversationState with given turns."""
    return ConversationState(anima_name="sakura", turns=turns)


def _make_turn(
    role: str = "assistant",
    content: str = "done",
    tool_records: list[ToolRecord] | None = None,
) -> ConversationTurn:
    """Helper to build a ConversationTurn."""
    return ConversationTurn(
        role=role,
        content=content,
        tool_records=tool_records or [],
    )


def _make_tool_record(
    tool_name: str = "search",
    result_summary: str = "found 5 results",
    input_summary: str = "",
) -> ToolRecord:
    """Helper to build a ToolRecord."""
    return ToolRecord(
        tool_name=tool_name,
        tool_id="",
        input_summary=input_summary,
        result_summary=result_summary,
    )


# ── Empty tool records ──────────────────────────────────────


class TestEmptyToolRecords:
    """Returns empty string when no tool records exist."""

    def test_no_turns_returns_empty(self, anima_dir: Path, model_config: ModelConfig):
        """Empty conversation state produces empty string."""
        state = _make_state(turns=[])
        with patch(
            "core.memory.conversation.ConversationMemory"
        ) as MockConvMem:
            mock_instance = MockConvMem.return_value
            mock_instance.load.return_value = state
            result = _build_recent_tool_section(anima_dir, model_config)
        assert result == ""

    def test_turns_without_tool_records_returns_empty(
        self, anima_dir: Path, model_config: ModelConfig,
    ):
        """Turns that have no tool_records produce empty string."""
        state = _make_state(turns=[
            _make_turn(role="human", content="hello"),
            _make_turn(role="assistant", content="hi there"),
        ])
        with patch(
            "core.memory.conversation.ConversationMemory"
        ) as MockConvMem:
            mock_instance = MockConvMem.return_value
            mock_instance.load.return_value = state
            result = _build_recent_tool_section(anima_dir, model_config)
        assert result == ""

    def test_tool_records_without_result_summary_returns_empty(
        self, anima_dir: Path, model_config: ModelConfig,
    ):
        """Tool records with empty result_summary are skipped."""
        record = ToolRecord(
            tool_name="search",
            tool_id="t1",
            input_summary="query=test",
            result_summary="",  # empty — should be skipped
        )
        state = _make_state(turns=[
            _make_turn(tool_records=[record]),
        ])
        with patch(
            "core.memory.conversation.ConversationMemory"
        ) as MockConvMem:
            mock_instance = MockConvMem.return_value
            mock_instance.load.return_value = state
            result = _build_recent_tool_section(anima_dir, model_config)
        assert result == ""

    def test_conversation_memory_load_exception_returns_empty(
        self, anima_dir: Path, model_config: ModelConfig,
    ):
        """Exception during ConversationMemory init/load returns empty."""
        with patch(
            "core.memory.conversation.ConversationMemory",
            side_effect=RuntimeError("disk error"),
        ):
            result = _build_recent_tool_section(anima_dir, model_config)
        assert result == ""


# ── Single tool record ──────────────────────────────────────


class TestSingleToolRecord:
    """Correctly formats one record with tool name and result summary."""

    def test_single_record_format(
        self, anima_dir: Path, model_config: ModelConfig,
    ):
        """A single tool record produces the expected header and line."""
        record = _make_tool_record(
            tool_name="web_search",
            result_summary="Found 3 pages about Python asyncio",
        )
        state = _make_state(turns=[
            _make_turn(tool_records=[record]),
        ])
        with patch(
            "core.memory.conversation.ConversationMemory"
        ) as MockConvMem:
            mock_instance = MockConvMem.return_value
            mock_instance.load.return_value = state
            result = _build_recent_tool_section(anima_dir, model_config)

        assert result.startswith("## Recent Tool Results\n\n")
        assert "- web_search: Found 3 pages about Python asyncio" in result

    def test_result_summary_truncated_at_500_chars(
        self, anima_dir: Path, model_config: ModelConfig,
    ):
        """result_summary longer than 500 chars is truncated in the line."""
        long_summary = "x" * 800
        record = _make_tool_record(
            tool_name="big_tool",
            result_summary=long_summary,
        )
        state = _make_state(turns=[
            _make_turn(tool_records=[record]),
        ])
        with patch(
            "core.memory.conversation.ConversationMemory"
        ) as MockConvMem:
            mock_instance = MockConvMem.return_value
            mock_instance.load.return_value = state
            result = _build_recent_tool_section(anima_dir, model_config)

        assert "## Recent Tool Results" in result
        # The line should contain at most 500 chars of the summary
        line = result.split("\n")[-1]
        # "- big_tool: " prefix + up to 500 chars of summary
        summary_in_line = line.split(": ", 1)[1]
        assert len(summary_in_line) <= 500


# ── Multiple tool records ───────────────────────────────────


class TestMultipleToolRecords:
    """Shows latest records from recent turns."""

    def test_records_from_multiple_turns(
        self, anima_dir: Path, model_config: ModelConfig,
    ):
        """Records from multiple turns are all included."""
        turn1 = _make_turn(tool_records=[
            _make_tool_record("search", "result A"),
        ])
        turn2 = _make_turn(tool_records=[
            _make_tool_record("post_channel", "posted to general"),
        ])
        state = _make_state(turns=[turn1, turn2])

        with patch(
            "core.memory.conversation.ConversationMemory"
        ) as MockConvMem:
            mock_instance = MockConvMem.return_value
            mock_instance.load.return_value = state
            result = _build_recent_tool_section(anima_dir, model_config)

        assert "- search: result A" in result
        assert "- post_channel: posted to general" in result

    def test_multiple_records_in_same_turn(
        self, anima_dir: Path, model_config: ModelConfig,
    ):
        """Multiple tool records within a single turn are all shown."""
        records = [
            _make_tool_record("tool_a", "result A"),
            _make_tool_record("tool_b", "result B"),
            _make_tool_record("tool_c", "result C"),
        ]
        state = _make_state(turns=[_make_turn(tool_records=records)])

        with patch(
            "core.memory.conversation.ConversationMemory"
        ) as MockConvMem:
            mock_instance = MockConvMem.return_value
            mock_instance.load.return_value = state
            result = _build_recent_tool_section(anima_dir, model_config)

        assert "- tool_a: result A" in result
        assert "- tool_b: result B" in result
        assert "- tool_c: result C" in result

    def test_only_last_three_turns_used(
        self, anima_dir: Path, model_config: ModelConfig,
    ):
        """Only state.turns[-3:] are examined (the last 3 turns)."""
        turns = [
            _make_turn(tool_records=[
                _make_tool_record("old_tool", "old result"),
            ]),
        ]
        # Add 3 more recent turns — only these should be used
        for i in range(3):
            turns.append(_make_turn(tool_records=[
                _make_tool_record(f"recent_{i}", f"recent result {i}"),
            ]))
        state = _make_state(turns=turns)

        with patch(
            "core.memory.conversation.ConversationMemory"
        ) as MockConvMem:
            mock_instance = MockConvMem.return_value
            mock_instance.load.return_value = state
            result = _build_recent_tool_section(anima_dir, model_config)

        # old_tool is in turns[0] which is outside [-3:]
        assert "old_tool" not in result
        assert "recent_0" in result
        assert "recent_1" in result
        assert "recent_2" in result

    def test_max_five_records_per_turn(
        self, anima_dir: Path, model_config: ModelConfig,
    ):
        """At most 5 tool_records per turn are rendered (inner [:5] slice)."""
        records = [
            _make_tool_record(f"tool_{i}", f"result {i}")
            for i in range(8)
        ]
        state = _make_state(turns=[_make_turn(tool_records=records)])

        with patch(
            "core.memory.conversation.ConversationMemory"
        ) as MockConvMem:
            mock_instance = MockConvMem.return_value
            mock_instance.load.return_value = state
            result = _build_recent_tool_section(anima_dir, model_config)

        # First 5 should appear
        for i in range(5):
            assert f"tool_{i}" in result
        # Records beyond 5 should not appear
        for i in range(5, 8):
            assert f"tool_{i}" not in result

    def test_skips_records_without_result_summary_among_others(
        self, anima_dir: Path, model_config: ModelConfig,
    ):
        """Records without result_summary are skipped; others still show."""
        records = [
            _make_tool_record("has_result", "some result"),
            ToolRecord(tool_name="no_result", result_summary=""),
            _make_tool_record("also_has_result", "another result"),
        ]
        state = _make_state(turns=[_make_turn(tool_records=records)])

        with patch(
            "core.memory.conversation.ConversationMemory"
        ) as MockConvMem:
            mock_instance = MockConvMem.return_value
            mock_instance.load.return_value = state
            result = _build_recent_tool_section(anima_dir, model_config)

        assert "- has_result: some result" in result
        assert "- also_has_result: another result" in result
        assert "no_result" not in result


# ── Truncation (budget limiting) ────────────────────────────


class TestBudgetTruncation:
    """Records are limited by the ~2000 token budget."""

    def test_budget_stops_adding_records(
        self, anima_dir: Path, model_config: ModelConfig,
    ):
        """When budget is exhausted, remaining records are dropped.

        Budget is 2000 tokens (~len(line)//4 per line).
        result_summary[:500] caps each summary at 500 chars in the line.
        Using long tool names to inflate line length and exhaust the budget:
          line ≈ "- {200-char name}: {500-char summary}" = ~704 chars
          704 // 4 = 176 tokens per line
          2000 / 176 ≈ 11 lines before budget runs out
        With 3 turns * 5 records = 15 total, some should be dropped.
        """
        long_name_prefix = "t" * 200  # 200-char tool name
        # Build 3 turns with 5 records each = 15 possible records
        turns = []
        for t in range(3):
            records = [
                _make_tool_record(
                    f"{long_name_prefix}_{t}_{i}",
                    "A" * 500,
                )
                for i in range(5)
            ]
            turns.append(_make_turn(tool_records=records))
        state = _make_state(turns=turns)

        with patch(
            "core.memory.conversation.ConversationMemory"
        ) as MockConvMem:
            mock_instance = MockConvMem.return_value
            mock_instance.load.return_value = state
            result = _build_recent_tool_section(anima_dir, model_config)

        # Should have some records but not all 15
        lines = [l for l in result.split("\n") if l.startswith("- ")]
        assert len(lines) > 0
        assert len(lines) < 15  # Budget should prevent all 15 from fitting

    def test_short_records_all_fit_within_budget(
        self, anima_dir: Path, model_config: ModelConfig,
    ):
        """Short records all fit within the 2000 token budget."""
        records = [
            _make_tool_record(f"tool_{i}", f"ok {i}")
            for i in range(5)
        ]
        state = _make_state(turns=[_make_turn(tool_records=records)])

        with patch(
            "core.memory.conversation.ConversationMemory"
        ) as MockConvMem:
            mock_instance = MockConvMem.return_value
            mock_instance.load.return_value = state
            result = _build_recent_tool_section(anima_dir, model_config)

        lines = [l for l in result.split("\n") if l.startswith("- ")]
        assert len(lines) == 5


# ── Integration with build_system_prompt ────────────────────


class TestBuildSystemPromptIntegration:
    """Verify the tool section appears in the system prompt."""

    def _make_memory_mock(self, anima_dir: Path, model_config: ModelConfig) -> MagicMock:
        """Create a standard MemoryManager mock."""
        memory = MagicMock()
        memory.anima_dir = anima_dir
        memory.read_company_vision.return_value = ""
        memory.read_identity.return_value = "I am Sakura"
        memory.read_injection.return_value = ""
        memory.read_permissions.return_value = ""
        memory.read_specialty_prompt.return_value = ""
        memory.read_current_state.return_value = ""
        memory.read_pending.return_value = ""
        memory.read_bootstrap.return_value = ""
        memory.list_knowledge_files.return_value = []
        memory.list_episode_files.return_value = []
        memory.list_procedure_files.return_value = []
        memory.list_skill_summaries.return_value = []
        memory.list_common_skill_summaries.return_value = []
        memory.list_skill_metas.return_value = []
        memory.list_common_skill_metas.return_value = []
        memory.list_procedure_metas.return_value = []
        memory.list_shared_users.return_value = []
        memory.read_model_config.return_value = model_config
        return memory

    def test_tool_section_in_system_prompt(
        self, anima_dir: Path, model_config: ModelConfig, data_dir,
    ):
        """When conversation has tool records, the section appears in the prompt."""
        record = _make_tool_record("web_search", "Found 3 results")
        state = _make_state(turns=[
            _make_turn(tool_records=[record]),
        ])
        memory = self._make_memory_mock(anima_dir, model_config)

        with patch("core.prompt.builder.load_prompt", return_value="section"), \
             patch(
                 "core.memory.conversation.ConversationMemory",
             ) as MockConvMem:
            mock_instance = MockConvMem.return_value
            mock_instance.load.return_value = state

            from core.prompt.builder import build_system_prompt
            result = build_system_prompt(memory)

        assert "## Recent Tool Results" in result
        assert "- web_search: Found 3 results" in result

    def test_no_tool_section_when_no_records(
        self, anima_dir: Path, model_config: ModelConfig, data_dir,
    ):
        """When conversation has no tool records, the section is absent."""
        state = _make_state(turns=[
            _make_turn(role="human", content="hello"),
            _make_turn(role="assistant", content="hi"),
        ])
        memory = self._make_memory_mock(anima_dir, model_config)

        with patch("core.prompt.builder.load_prompt", return_value="section"), \
             patch(
                 "core.memory.conversation.ConversationMemory",
             ) as MockConvMem:
            mock_instance = MockConvMem.return_value
            mock_instance.load.return_value = state

            from core.prompt.builder import build_system_prompt
            result = build_system_prompt(memory)

        assert "## Recent Tool Results" not in result

    def test_tool_section_exception_does_not_break_prompt(
        self, anima_dir: Path, model_config: ModelConfig, data_dir,
    ):
        """If _build_recent_tool_section raises, the prompt is still built."""
        memory = self._make_memory_mock(anima_dir, model_config)

        with patch("core.prompt.builder.load_prompt", return_value="section"), \
             patch(
                 "core.memory.conversation.ConversationMemory",
                 side_effect=RuntimeError("broken"),
             ):
            from core.prompt.builder import build_system_prompt
            result = build_system_prompt(memory)

        # Prompt should still be valid (just missing the tool section)
        assert "## Recent Tool Results" not in result
        assert len(result) > 0
