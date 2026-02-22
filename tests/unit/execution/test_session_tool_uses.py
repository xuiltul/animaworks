# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for session chaining tool_uses parameter and shortterm _render_markdown changes.

Covers:
- handle_session_chaining() backward compatibility with tool_uses=None
- handle_session_chaining() passing tool_uses through to SessionState
- SessionState including tool_uses in rendered markdown
- _render_markdown() tool entries limit (20), input truncation (500 chars),
  and result display
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.execution._session import handle_session_chaining
from core.memory.shortterm import (
    SessionState,
    ShortTermMemory,
    _MAX_RESPONSE_CHARS,
)
from core.prompt.builder import BuildResult


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    """Create an isolated anima directory with shortterm subdirectory."""
    d = tmp_path / "animas" / "test-anima"
    (d / "shortterm" / "archive").mkdir(parents=True)
    return d


@pytest.fixture
def shortterm(anima_dir: Path) -> ShortTermMemory:
    """Return a ShortTermMemory instance rooted at the test anima dir."""
    return ShortTermMemory(anima_dir)


@pytest.fixture
def mock_tracker() -> MagicMock:
    """Return a mock ContextTracker with threshold exceeded."""
    tracker = MagicMock()
    tracker.threshold_exceeded = True
    tracker.usage_ratio = 0.85
    return tracker


@pytest.fixture
def mock_tracker_below() -> MagicMock:
    """Return a mock ContextTracker with threshold NOT exceeded."""
    tracker = MagicMock()
    tracker.threshold_exceeded = False
    tracker.usage_ratio = 0.30
    return tracker


@pytest.fixture
def mock_memory() -> MagicMock:
    """Return a mock MemoryManager."""
    return MagicMock()


@pytest.fixture
def system_prompt_builder() -> MagicMock:
    """Return a callable that produces a BuildResult."""
    builder = MagicMock()
    builder.return_value = BuildResult(system_prompt="base system prompt")
    return builder


@pytest.fixture
def sample_tool_uses() -> list[dict[str, Any]]:
    """Return representative tool_uses list."""
    return [
        {"name": "web_search", "input": "latest news", "result": "Found 5 articles"},
        {"name": "read_file", "input": "/path/to/file.txt", "result": "File contents here"},
        {"name": "bash", "input": "ls -la", "result": "total 42\ndrwxr-xr-x ..."},
    ]


# ── handle_session_chaining with tool_uses=None ────────────────


class TestHandleSessionChainingToolUsesNone:
    """Backward compatibility: tool_uses=None produces empty tool_uses."""

    @pytest.mark.asyncio
    async def test_no_tool_uses_stored_when_none(
        self,
        mock_tracker: MagicMock,
        shortterm: ShortTermMemory,
        mock_memory: MagicMock,
        system_prompt_builder: MagicMock,
    ) -> None:
        """When tool_uses is not passed, SessionState.tool_uses should be []."""
        with patch("core.execution._session.now_iso", return_value="2026-02-22T10:00:00"):
            result, chain_count = await handle_session_chaining(
                tracker=mock_tracker,
                shortterm=shortterm,
                memory=mock_memory,
                current_text="some response",
                system_prompt_builder=system_prompt_builder,
                max_chains=3,
                chain_count=0,
                session_id="test-sess",
                trigger="heartbeat",
                original_prompt="Do something",
            )

        assert result is not None
        assert chain_count == 1

        # Verify the saved state has empty tool_uses
        loaded = shortterm.load()
        # load() returns None because clear() was called, so check archive
        # Actually, clear() is called after inject, so let's check the
        # JSON was saved with empty tool_uses by inspecting the archive
        archive_dir = shortterm._archive_dir
        json_files = list(archive_dir.glob("*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text(encoding="utf-8"))
        assert data["tool_uses"] == []

    @pytest.mark.asyncio
    async def test_backward_compatible_signature(
        self,
        mock_tracker: MagicMock,
        shortterm: ShortTermMemory,
        mock_memory: MagicMock,
        system_prompt_builder: MagicMock,
    ) -> None:
        """Calling without tool_uses keyword at all still works."""
        with patch("core.execution._session.now_iso", return_value="2026-02-22T10:00:00"):
            # Call without tool_uses parameter — should not raise
            result, chain_count = await handle_session_chaining(
                tracker=mock_tracker,
                shortterm=shortterm,
                memory=mock_memory,
                current_text="response text",
                system_prompt_builder=system_prompt_builder,
                max_chains=5,
                chain_count=0,
            )

        assert result is not None
        assert chain_count == 1

    @pytest.mark.asyncio
    async def test_no_chaining_when_shortterm_is_none(
        self,
        mock_tracker: MagicMock,
        mock_memory: MagicMock,
        system_prompt_builder: MagicMock,
    ) -> None:
        """When shortterm=None, no chaining occurs regardless of tool_uses."""
        result, chain_count = await handle_session_chaining(
            tracker=mock_tracker,
            shortterm=None,
            memory=mock_memory,
            current_text="text",
            system_prompt_builder=system_prompt_builder,
            max_chains=3,
            chain_count=0,
            tool_uses=[{"name": "search", "input": "q"}],
        )

        assert result is None
        assert chain_count == 0

    @pytest.mark.asyncio
    async def test_no_chaining_below_threshold(
        self,
        mock_tracker_below: MagicMock,
        shortterm: ShortTermMemory,
        mock_memory: MagicMock,
        system_prompt_builder: MagicMock,
    ) -> None:
        """When threshold is not exceeded, tool_uses are irrelevant."""
        result, chain_count = await handle_session_chaining(
            tracker=mock_tracker_below,
            shortterm=shortterm,
            memory=mock_memory,
            current_text="text",
            system_prompt_builder=system_prompt_builder,
            max_chains=3,
            chain_count=0,
            tool_uses=[{"name": "search", "input": "q"}],
        )

        assert result is None
        assert chain_count == 0


# ── handle_session_chaining with tool_uses list ────────────────


class TestHandleSessionChainingWithToolUses:
    """tool_uses are passed through to SessionState when chaining occurs."""

    @pytest.mark.asyncio
    async def test_tool_uses_stored_in_session_state(
        self,
        mock_tracker: MagicMock,
        shortterm: ShortTermMemory,
        mock_memory: MagicMock,
        system_prompt_builder: MagicMock,
        sample_tool_uses: list[dict[str, Any]],
    ) -> None:
        """tool_uses passed to handle_session_chaining appear in saved state."""
        with patch("core.execution._session.now_iso", return_value="2026-02-22T10:00:00"):
            result, chain_count = await handle_session_chaining(
                tracker=mock_tracker,
                shortterm=shortterm,
                memory=mock_memory,
                current_text="completed work",
                system_prompt_builder=system_prompt_builder,
                max_chains=3,
                chain_count=0,
                session_id="sess-with-tools",
                trigger="message",
                original_prompt="Use some tools",
                tool_uses=sample_tool_uses,
            )

        assert result is not None
        assert chain_count == 1

        # Check archived JSON has the tool_uses
        archive_dir = shortterm._archive_dir
        json_files = list(archive_dir.glob("*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text(encoding="utf-8"))
        assert len(data["tool_uses"]) == 3
        assert data["tool_uses"][0]["name"] == "web_search"
        assert data["tool_uses"][1]["name"] == "read_file"
        assert data["tool_uses"][2]["name"] == "bash"

    @pytest.mark.asyncio
    async def test_tool_uses_appear_in_new_system_prompt(
        self,
        mock_tracker: MagicMock,
        shortterm: ShortTermMemory,
        mock_memory: MagicMock,
        system_prompt_builder: MagicMock,
        sample_tool_uses: list[dict[str, Any]],
    ) -> None:
        """The rebuilt system prompt includes tool_uses from the session state."""
        with patch("core.execution._session.now_iso", return_value="2026-02-22T10:00:00"):
            result, _ = await handle_session_chaining(
                tracker=mock_tracker,
                shortterm=shortterm,
                memory=mock_memory,
                current_text="response",
                system_prompt_builder=system_prompt_builder,
                max_chains=3,
                chain_count=0,
                session_id="sess-tools",
                trigger="message",
                original_prompt="Do work",
                tool_uses=sample_tool_uses,
            )

        assert result is not None
        # The system prompt should contain tool names from the session state
        # (inject_shortterm loads the markdown which includes tool entries)
        assert "web_search" in result
        assert "read_file" in result
        assert "bash" in result

    @pytest.mark.asyncio
    async def test_empty_list_treated_as_no_tools(
        self,
        mock_tracker: MagicMock,
        shortterm: ShortTermMemory,
        mock_memory: MagicMock,
        system_prompt_builder: MagicMock,
    ) -> None:
        """Explicitly passing tool_uses=[] behaves the same as None."""
        with patch("core.execution._session.now_iso", return_value="2026-02-22T10:00:00"):
            result, chain_count = await handle_session_chaining(
                tracker=mock_tracker,
                shortterm=shortterm,
                memory=mock_memory,
                current_text="response",
                system_prompt_builder=system_prompt_builder,
                max_chains=3,
                chain_count=0,
                tool_uses=[],
            )

        assert result is not None
        # The rendered markdown should show "(なし)" for tools
        assert "(なし)" in result

    @pytest.mark.asyncio
    async def test_accumulated_response_includes_current_text(
        self,
        mock_tracker: MagicMock,
        shortterm: ShortTermMemory,
        mock_memory: MagicMock,
        system_prompt_builder: MagicMock,
    ) -> None:
        """current_text is appended to accumulated_response in saved state."""
        with patch("core.execution._session.now_iso", return_value="2026-02-22T10:00:00"):
            await handle_session_chaining(
                tracker=mock_tracker,
                shortterm=shortterm,
                memory=mock_memory,
                current_text="new fragment",
                system_prompt_builder=system_prompt_builder,
                max_chains=3,
                chain_count=0,
                accumulated_response="previous text",
                tool_uses=[{"name": "tool1", "input": "i1"}],
            )

        archive_dir = shortterm._archive_dir
        json_files = list(archive_dir.glob("*.json"))
        data = json.loads(json_files[0].read_text(encoding="utf-8"))
        assert "previous text" in data["accumulated_response"]
        assert "new fragment" in data["accumulated_response"]

    @pytest.mark.asyncio
    async def test_max_chains_prevents_chaining(
        self,
        mock_tracker: MagicMock,
        shortterm: ShortTermMemory,
        mock_memory: MagicMock,
        system_prompt_builder: MagicMock,
    ) -> None:
        """When chain_count >= max_chains, no chaining occurs even with tool_uses."""
        result, chain_count = await handle_session_chaining(
            tracker=mock_tracker,
            shortterm=shortterm,
            memory=mock_memory,
            current_text="text",
            system_prompt_builder=system_prompt_builder,
            max_chains=3,
            chain_count=3,
            tool_uses=[{"name": "search", "input": "q"}],
        )

        assert result is None
        assert chain_count == 3


# ── SessionState includes tool_uses in markdown render ─────────


class TestSessionStateToolUsesInMarkdown:
    """When tool_uses are present, they appear in the rendered session state."""

    def test_tool_names_in_markdown(self, shortterm: ShortTermMemory) -> None:
        """Tool names from tool_uses appear in the rendered markdown."""
        state = SessionState(
            session_id="sess-md",
            timestamp="2026-02-22T10:00:00",
            trigger="message",
            original_prompt="Test prompt",
            accumulated_response="Test response",
            tool_uses=[
                {"name": "web_search", "input": "query", "result": "results"},
                {"name": "bash", "input": "ls", "result": "files"},
            ],
            context_usage_ratio=0.75,
            turn_count=10,
        )
        md = shortterm._render_markdown(state)

        assert "web_search" in md
        assert "bash" in md

    def test_tool_results_in_markdown(self, shortterm: ShortTermMemory) -> None:
        """Tool results appear with arrow prefix in rendered markdown."""
        state = SessionState(
            tool_uses=[
                {"name": "search", "input": "query", "result": "Found 3 items"},
            ],
        )
        md = shortterm._render_markdown(state)

        assert "Found 3 items" in md
        # Result is prefixed with arrow
        assert "\u2192 Found 3 items" in md  # → character

    def test_tool_input_in_markdown(self, shortterm: ShortTermMemory) -> None:
        """Tool input values appear in rendered markdown."""
        state = SessionState(
            tool_uses=[
                {"name": "read_file", "input": "/path/to/important/file.py"},
            ],
        )
        md = shortterm._render_markdown(state)

        assert "/path/to/important/file.py" in md

    def test_empty_tool_uses_shows_none(self, shortterm: ShortTermMemory) -> None:
        """Empty tool_uses renders (なし) in the markdown."""
        state = SessionState(tool_uses=[])
        md = shortterm._render_markdown(state)

        assert "(なし)" in md

    def test_tool_uses_section_header_present(
        self, shortterm: ShortTermMemory
    ) -> None:
        """The markdown includes the tool section header."""
        state = SessionState(
            tool_uses=[{"name": "tool1", "input": "x"}],
        )
        md = shortterm._render_markdown(state)

        assert "使用したツール" in md

    def test_no_result_field_omits_arrow_line(
        self, shortterm: ShortTermMemory
    ) -> None:
        """When a tool entry has no result field, no arrow line is added."""
        state = SessionState(
            tool_uses=[
                {"name": "search", "input": "query"},
            ],
        )
        md = shortterm._render_markdown(state)

        assert "search" in md
        assert "query" in md
        # No arrow line because result is missing (empty string is falsy)
        assert "\u2192" not in md

    def test_empty_result_string_omits_arrow_line(
        self, shortterm: ShortTermMemory
    ) -> None:
        """When result is an empty string, no arrow line is rendered."""
        state = SessionState(
            tool_uses=[
                {"name": "tool", "input": "x", "result": ""},
            ],
        )
        md = shortterm._render_markdown(state)

        # Empty string is falsy, so arrow line should not appear
        assert "\u2192" not in md


# ── _render_markdown tool entries limit and truncation ─────────


class TestRenderMarkdownToolLimits:
    """_render_markdown changes: limit 20 entries, input truncation at 500, result display."""

    def test_tool_entries_limited_to_20(self, shortterm: ShortTermMemory) -> None:
        """Only the last 20 tool entries are included in the markdown."""
        # Create 30 tool entries
        tools = [
            {"name": f"tool_{i:02d}", "input": f"input_{i}", "result": f"result_{i}"}
            for i in range(30)
        ]
        state = SessionState(tool_uses=tools)
        md = shortterm._render_markdown(state)

        # First 10 tools (indices 0-9) should NOT appear
        for i in range(10):
            assert f"tool_{i:02d}" not in md, f"tool_{i:02d} should not be in markdown"

        # Last 20 tools (indices 10-29) should appear
        for i in range(10, 30):
            assert f"tool_{i:02d}" in md, f"tool_{i:02d} should be in markdown"

    def test_exactly_20_tools_all_shown(self, shortterm: ShortTermMemory) -> None:
        """When exactly 20 tool entries exist, all are shown."""
        tools = [
            {"name": f"tool_{i:02d}", "input": f"input_{i}"}
            for i in range(20)
        ]
        state = SessionState(tool_uses=tools)
        md = shortterm._render_markdown(state)

        for i in range(20):
            assert f"tool_{i:02d}" in md

    def test_fewer_than_20_tools_all_shown(self, shortterm: ShortTermMemory) -> None:
        """When fewer than 20 tool entries exist, all are shown."""
        tools = [
            {"name": f"tool_{i}", "input": f"input_{i}"}
            for i in range(5)
        ]
        state = SessionState(tool_uses=tools)
        md = shortterm._render_markdown(state)

        for i in range(5):
            assert f"tool_{i}" in md

    def test_input_truncated_at_500_chars(self, shortterm: ShortTermMemory) -> None:
        """Tool input longer than 500 chars is truncated."""
        long_input = "A" * 800
        state = SessionState(
            tool_uses=[{"name": "tool", "input": long_input}],
        )
        md = shortterm._render_markdown(state)

        # The full 800-char input should NOT appear
        assert long_input not in md
        # But the first 500 chars should appear
        assert "A" * 500 in md

    def test_input_at_exactly_500_not_truncated(
        self, shortterm: ShortTermMemory
    ) -> None:
        """Tool input of exactly 500 chars is kept as-is."""
        exact_input = "B" * 500
        state = SessionState(
            tool_uses=[{"name": "tool", "input": exact_input}],
        )
        md = shortterm._render_markdown(state)

        assert exact_input in md

    def test_result_truncated_at_500_chars(self, shortterm: ShortTermMemory) -> None:
        """Tool result longer than 500 chars is truncated."""
        long_result = "R" * 800
        state = SessionState(
            tool_uses=[{"name": "tool", "input": "x", "result": long_result}],
        )
        md = shortterm._render_markdown(state)

        # The full 800-char result should NOT appear
        assert long_result not in md
        # But the first 500 chars of the result should appear
        assert "R" * 500 in md

    def test_result_at_exactly_500_not_truncated(
        self, shortterm: ShortTermMemory
    ) -> None:
        """Tool result of exactly 500 chars is kept as-is."""
        exact_result = "S" * 500
        state = SessionState(
            tool_uses=[{"name": "tool", "input": "x", "result": exact_result}],
        )
        md = shortterm._render_markdown(state)

        assert exact_result in md

    def test_result_display_with_arrow_prefix(
        self, shortterm: ShortTermMemory
    ) -> None:
        """Tool results are displayed with arrow prefix on indented line."""
        state = SessionState(
            tool_uses=[
                {"name": "bash", "input": "echo hello", "result": "hello"},
            ],
        )
        md = shortterm._render_markdown(state)

        # Check the format: "  → result"
        assert "  \u2192 hello" in md  # "  → hello"

    def test_multiple_tools_with_mixed_results(
        self, shortterm: ShortTermMemory
    ) -> None:
        """Mix of tools with and without results renders correctly."""
        state = SessionState(
            tool_uses=[
                {"name": "search", "input": "query", "result": "found items"},
                {"name": "write", "input": "file.txt"},
                {"name": "bash", "input": "pwd", "result": "/home/user"},
            ],
        )
        md = shortterm._render_markdown(state)

        # All tool names present
        assert "search" in md
        assert "write" in md
        assert "bash" in md
        # Results with values have arrow lines
        assert "\u2192 found items" in md
        assert "\u2192 /home/user" in md

    def test_non_string_input_converted(self, shortterm: ShortTermMemory) -> None:
        """Non-string input values are converted to string before truncation."""
        state = SessionState(
            tool_uses=[
                {"name": "tool", "input": {"key": "value"}, "result": 12345},
            ],
        )
        md = shortterm._render_markdown(state)

        # dict input is stringified
        assert "key" in md
        assert "value" in md
        # int result is stringified
        assert "12345" in md

    def test_missing_name_defaults_to_question_mark(
        self, shortterm: ShortTermMemory
    ) -> None:
        """Tool entry without 'name' key shows '?' as the name."""
        state = SessionState(
            tool_uses=[{"input": "some input"}],
        )
        md = shortterm._render_markdown(state)

        assert "?" in md
        assert "some input" in md
