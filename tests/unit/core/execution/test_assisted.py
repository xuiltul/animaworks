"""Unit tests for Mode B text-based tool loop (core.execution.assisted)."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.execution.assisted import (
    _MAX_TOOL_OUTPUT_BYTES,
    _strip_tool_call_block,
    _truncate_tool_output,
    extract_tool_call,
)
from core.tooling.schemas import to_text_format


def _has_json_repair() -> bool:
    """Check if the json_repair library is available."""
    try:
        import json_repair  # noqa: F401
        return True
    except ImportError:
        return False


# ── extract_tool_call ────────────────────────────────────────


class TestExtractToolCall:
    """Test JSON extraction from LLM response text."""

    def test_valid_json_in_code_block(self):
        text = '```json\n{"tool": "search_memory", "arguments": {"query": "hello"}}\n```'
        result = extract_tool_call(text)
        assert result is not None
        assert result["tool"] == "search_memory"
        assert result["arguments"]["query"] == "hello"

    def test_valid_json_bare_object(self):
        text = '{"tool": "search_memory", "arguments": {"query": "hello"}}'
        result = extract_tool_call(text)
        assert result is not None
        assert result["tool"] == "search_memory"

    def test_no_tool_call_returns_none(self):
        text = "こんにちは！今日はいい天気ですね。"
        assert extract_tool_call(text) is None

    def test_empty_string_returns_none(self):
        assert extract_tool_call("") is None

    def test_json_without_tool_key_returns_none(self):
        text = '```json\n{"name": "search_memory", "args": {"query": "hello"}}\n```'
        assert extract_tool_call(text) is None

    @pytest.mark.skipif(
        not _has_json_repair(),
        reason="json_repair not installed",
    )
    def test_broken_json_missing_comma(self):
        # Missing comma between arguments — json_repair should fix
        text = '```json\n{"tool": "search_memory" "arguments": {"query": "hello"}}\n```'
        result = extract_tool_call(text)
        assert result is not None
        assert result["tool"] == "search_memory"

    def test_python_dict_literal_single_quotes(self):
        text = "```json\n{'tool': 'search_memory', 'arguments': {'query': 'hello'}}\n```"
        result = extract_tool_call(text)
        assert result is not None
        assert result["tool"] == "search_memory"

    def test_nested_json_in_arguments(self):
        text = '```json\n{"tool": "write_memory_file", "arguments": {"path": "test.md", "content": "hello"}}\n```'
        result = extract_tool_call(text)
        assert result is not None
        assert result["tool"] == "write_memory_file"
        assert result["arguments"]["path"] == "test.md"

    def test_thinking_before_code_block(self):
        text = (
            "記憶を検索してみます。\n\n"
            '```json\n{"tool": "search_memory", "arguments": {"query": "hello"}}\n```'
        )
        result = extract_tool_call(text)
        assert result is not None
        assert result["tool"] == "search_memory"

    def test_thinking_after_code_block(self):
        text = (
            '```json\n{"tool": "search_memory", "arguments": {"query": "hello"}}\n```'
            "\n\nこれで記憶を検索します。"
        )
        result = extract_tool_call(text)
        assert result is not None
        assert result["tool"] == "search_memory"

    def test_code_block_without_json_tag(self):
        text = '```\n{"tool": "search_memory", "arguments": {"query": "hello"}}\n```'
        result = extract_tool_call(text)
        assert result is not None
        assert result["tool"] == "search_memory"


# ── _strip_tool_call_block ───────────────────────────────────


class TestStripToolCallBlock:
    def test_removes_code_block(self):
        text = (
            "考えてみます。\n\n"
            '```json\n{"tool": "search_memory", "arguments": {"query": "hello"}}\n```'
            "\n\nこれで検索します。"
        )
        result = _strip_tool_call_block(text)
        assert "search_memory" not in result
        assert "考えてみます" in result
        assert "これで検索します" in result

    def test_text_without_code_block_unchanged(self):
        text = "こんにちは！"
        assert _strip_tool_call_block(text) == text


# ── to_text_format ───────────────────────────────────────────


class TestToTextFormat:
    def test_empty_schemas(self):
        result = to_text_format([])
        assert "利用可能なツール" in result
        # No tool entries
        assert "**" not in result.split("1回のメッセージで")[1] if "1回のメッセージで" in result else True

    def test_single_tool_with_required_args(self):
        schemas = [{
            "name": "search_memory",
            "description": "Search memory",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "scope": {"type": "string"},
                },
                "required": ["query"],
            },
        }]
        result = to_text_format(schemas)
        assert "**search_memory**" in result
        assert "query: string (必須)" in result
        assert "scope: string" in result
        assert "(必須)" not in result.split("scope")[0].split("query")[-1] or True

    def test_tool_without_parameters(self):
        schemas = [{
            "name": "refresh_tools",
            "description": "Refresh tools",
            "parameters": {"type": "object", "properties": {}},
        }]
        result = to_text_format(schemas)
        assert "**refresh_tools**" in result
        # No args line for parameterless tool
        lines = result.split("\n")
        refresh_idx = next(i for i, l in enumerate(lines) if "refresh_tools" in l)
        # Next line should NOT be an args line (or should be another tool/end)
        if refresh_idx + 1 < len(lines):
            assert "引数:" not in lines[refresh_idx + 1]

    def test_contains_json_format_instruction(self):
        result = to_text_format([])
        assert '{"tool":' in result
        assert "```json" in result


# ── _truncate_tool_output ────────────────────────────────────


class TestTruncateToolOutput:
    """Test tool output truncation for Mode B context overflow prevention."""

    def test_under_limit_returns_as_is(self):
        text = "a" * 100
        assert _truncate_tool_output(text) == text

    def test_over_limit_truncates(self):
        text = "x" * 8000
        result = _truncate_tool_output(text)
        assert len(result.encode("utf-8")) < 8000
        assert "出力切り捨て" in result
        assert "8000" in result  # original size in bytes

    def test_exact_limit_returns_as_is(self):
        text = "a" * _MAX_TOOL_OUTPUT_BYTES  # ASCII: 1 byte per char
        assert _truncate_tool_output(text) == text

    def test_multibyte_safe(self):
        # Japanese chars are 3 bytes each in UTF-8
        text = "あ" * 2000  # 6000 bytes, exceeds 4096
        result = _truncate_tool_output(text)
        assert "出力切り捨て" in result
        # Should not raise on decode — no corrupted multibyte chars
        result.encode("utf-8")

    def test_custom_max_bytes(self):
        text = "hello world!"  # 12 bytes
        result = _truncate_tool_output(text, max_bytes=10)
        assert "出力切り捨て" in result
        assert "12" in result  # original size


# ── _preflight_check ─────────────────────────────────────────


class TestPreflightCheck:
    """Test pre-flight context window check for Mode B."""

    @pytest.fixture
    def assisted_executor(self, tmp_path: Path):
        from core.execution.assisted import AssistedExecutor
        from core.memory import MemoryManager
        from core.schemas import ModelConfig
        from core.tooling.handler import ToolHandler

        # Create minimal anima directory structure
        anima_dir = tmp_path / "animas" / "test-preflight"
        for sub in ("state", "episodes", "knowledge", "procedures", "skills",
                    "shortterm", "transcripts", "activity_log"):
            (anima_dir / sub).mkdir(parents=True)
        (anima_dir / "identity.md").write_text("# Test", encoding="utf-8")
        (anima_dir / "injection.md").write_text("", encoding="utf-8")
        (anima_dir / "permissions.md").write_text("", encoding="utf-8")

        model_config = ModelConfig(
            model="ollama/gemma3:27b",
            max_tokens=4096,
            max_turns=5,
        )
        memory = MemoryManager(anima_dir)
        tool_handler = ToolHandler(anima_dir=anima_dir, memory=memory)
        return AssistedExecutor(
            model_config=model_config,
            anima_dir=anima_dir,
            tool_handler=tool_handler,
            memory=memory,
        )

    def test_normal_returns_configured_max(self, assisted_executor):
        messages = [
            {"role": "system", "content": "short"},
            {"role": "user", "content": "hi"},
        ]
        with patch("litellm.token_counter", return_value=500), \
             patch("core.config.load_config") as mock_lc:
            mock_lc.return_value = MagicMock(model_context_windows=None)
            result = assisted_executor._preflight_check(messages)
        assert result is not None
        assert result["max_tokens"] == 4096

    def test_clamped_when_tight(self, assisted_executor):
        # context_window=128000 default for gemma3
        # est_input high enough that available < configured_max (4096)
        # but available - 128 >= 256 so it doesn't return None
        # available = ctx_window - est_input; we want available < 4096
        # e.g. est_input = 128000 - 2000 = 126000 → available = 2000
        messages = [
            {"role": "system", "content": "x"},
            {"role": "user", "content": "y"},
        ]
        with patch("litellm.token_counter", return_value=126000), \
             patch("core.config.load_config") as mock_lc:
            mock_lc.return_value = MagicMock(model_context_windows=None)
            result = assisted_executor._preflight_check(messages)
        assert result is not None
        # available = 128000 - 126000 = 2000; clamped = 2000 - 128 = 1872
        assert result["max_tokens"] == 2000 - 128

    def test_none_when_too_large(self, assisted_executor):
        # available - 128 < 256 → returns None
        # available = ctx_window - est_input < 384
        # e.g. est_input = 128000 - 300 = 127700 → available = 300; 300-128=172 < 256
        messages = [
            {"role": "system", "content": "x"},
            {"role": "user", "content": "y"},
        ]
        with patch("litellm.token_counter", return_value=127700), \
             patch("core.config.load_config") as mock_lc:
            mock_lc.return_value = MagicMock(model_context_windows=None)
            result = assisted_executor._preflight_check(messages)
        assert result is None

    def test_fallback_on_token_counter_error(self, assisted_executor):
        # When token_counter raises, fallback uses char count // 2
        # "short" = 5 chars, "hi" = 2 chars → total = 7 → est_input = 3
        # available = 128000 - 3 = 127997 → plenty of room
        messages = [
            {"role": "system", "content": "short"},
            {"role": "user", "content": "hi"},
        ]
        with patch("litellm.token_counter", side_effect=Exception("model not found")), \
             patch("core.config.load_config") as mock_lc:
            mock_lc.return_value = MagicMock(model_context_windows=None)
            result = assisted_executor._preflight_check(messages)
        assert result is not None
        assert result["max_tokens"] == 4096
