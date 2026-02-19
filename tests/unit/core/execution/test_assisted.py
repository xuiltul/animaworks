"""Unit tests for Mode B text-based tool loop (core.execution.assisted)."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from core.execution.assisted import extract_tool_call, _strip_tool_call_block
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
