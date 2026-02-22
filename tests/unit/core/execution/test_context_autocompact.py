"""Unit tests for A1 mid-session context auto-compact (core.execution.agent_sdk)."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ── Mock claude_agent_sdk before importing agent_sdk module ───
# The SDK may not be installed in the test environment, so we inject
# lightweight mock modules that satisfy the import at module level.

_mock_sdk = MagicMock()

# HookInput / HookContext are dict-like in the real SDK.
# SyncHookJSONOutput is a TypedDict.  For testing we just need them
# importable and usable as plain dicts / constructors.
_mock_types = MagicMock()


def _sync_hook_json_output(**kwargs: Any) -> dict[str, Any]:
    """Mimic SyncHookJSONOutput as a plain dict."""
    return dict(kwargs)


_mock_types.SyncHookJSONOutput = _sync_hook_json_output
_mock_types.PreToolUseHookSpecificOutput = dict
_mock_types.HookInput = dict
_mock_types.HookContext = dict

sys.modules.setdefault("claude_agent_sdk", _mock_sdk)
sys.modules.setdefault("claude_agent_sdk.types", _mock_types)

from core.execution.agent_sdk import (  # noqa: E402
    _build_pre_tool_hook,
    _CONTEXT_AUTOCOMPACT_SAFETY,
    _tool_result_content_len,
)
from core.execution.base import ExecutionResult  # noqa: E402
from core.prompt.context import CHARS_PER_TOKEN  # noqa: E402


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    """Create a minimal anima directory for hook construction."""
    d = tmp_path / "animas" / "test-autocompact"
    for sub in ("state", "episodes", "knowledge", "procedures",
                "skills", "shortterm", "activity_log"):
        (d / sub).mkdir(parents=True)
    (d / "identity.md").write_text("# Test", encoding="utf-8")
    (d / "injection.md").write_text("", encoding="utf-8")
    (d / "permissions.md").write_text("", encoding="utf-8")
    return d


# ── _tool_result_content_len ─────────────────────────────────


class TestToolResultContentLen:
    """Test the _tool_result_content_len helper for byte-length estimation."""

    def test_list_content_multiple_text_items(self):
        """List content with multiple text dicts sums their text lengths."""
        block = SimpleNamespace(content=[
            {"text": "hello"},
            {"text": "world!"},
        ])
        assert _tool_result_content_len(block) == len("hello") + len("world!")

    def test_list_content_single_item(self):
        """List content with a single text dict returns its length."""
        block = SimpleNamespace(content=[{"text": "abc"}])
        assert _tool_result_content_len(block) == 3

    def test_list_content_with_non_dict_items(self):
        """Non-dict items in the list are ignored."""
        block = SimpleNamespace(content=[
            {"text": "valid"},
            "not a dict",
            42,
            {"text": "also valid"},
        ])
        assert _tool_result_content_len(block) == len("valid") + len("also valid")

    def test_list_content_with_missing_text_key(self):
        """Dict items without a 'text' key contribute zero length."""
        block = SimpleNamespace(content=[
            {"text": "ok"},
            {"data": "no text key"},
        ])
        # Second item: str(dict.get("text", "")) == "" -> len 0
        assert _tool_result_content_len(block) == 2

    def test_string_content(self):
        """String content returns its string length."""
        block = SimpleNamespace(content="hello world")
        assert _tool_result_content_len(block) == len("hello world")

    def test_none_content(self):
        """None content returns 0."""
        block = SimpleNamespace(content=None)
        assert _tool_result_content_len(block) == 0

    def test_empty_string_content(self):
        """Empty string content returns the length of str('') == 0."""
        block = SimpleNamespace(content="")
        # str("") is "" -> len 0
        assert _tool_result_content_len(block) == 0

    def test_empty_list_content(self):
        """Empty list content returns 0."""
        block = SimpleNamespace(content=[])
        assert _tool_result_content_len(block) == 0

    def test_numeric_content(self):
        """Numeric content is str()-ified and its length returned."""
        block = SimpleNamespace(content=12345)
        assert _tool_result_content_len(block) == len("12345")


# ── _build_pre_tool_hook: context budget check ───────────────


class TestPreToolHookContextBudget:
    """Test the PreToolUse hook's mid-session context budget logic."""

    @pytest.mark.asyncio
    async def test_no_session_stats_does_not_trigger(self, anima_dir: Path):
        """When session_stats is None, hook should NOT return continue_=False."""
        hook = _build_pre_tool_hook(
            anima_dir,
            max_tokens=4096,
            context_window=200_000,
            session_stats=None,
        )
        input_data = {"tool_name": "Read", "tool_input": {"file_path": "/tmp/x"}}
        result = await hook(input_data, "tool-123", {})
        # Should not have continue_=False (no autocompact without stats)
        assert result.get("continue_") is not False

    @pytest.mark.asyncio
    async def test_plenty_of_room_does_not_trigger(self, anima_dir: Path):
        """When context has plenty of room, hook should NOT trigger autocompact."""
        session_stats: dict[str, Any] = {
            "tool_call_count": 0,
            "total_result_bytes": 1000,
            "system_prompt_tokens": 500,
            "user_prompt_tokens": 0,
            "force_chain": False,
        }
        hook = _build_pre_tool_hook(
            anima_dir,
            max_tokens=4096,
            context_window=200_000,
            session_stats=session_stats,
        )
        input_data = {"tool_name": "Grep", "tool_input": {"pattern": "test"}}
        result = await hook(input_data, "tool-456", {})

        # Estimated: 500 + 1000/CHARS_PER_TOKEN = 500 + 250 = 750 tokens
        # Remaining: 200_000 - 750 = 199_250 >> 4096 * 2 = 8192
        assert result.get("continue_") is not False
        assert session_stats["force_chain"] is False
        # tool_call_count should have incremented
        assert session_stats["tool_call_count"] == 1

    @pytest.mark.asyncio
    async def test_exceeds_budget_triggers_autocompact(self, anima_dir: Path):
        """When estimated tokens exceed context_window - max_tokens*2, trigger."""
        max_tokens = 4096
        context_window = 200_000
        budget = max_tokens * _CONTEXT_AUTOCOMPACT_SAFETY  # 8192

        # We need: estimated_tokens > context_window - budget
        # estimated = system_prompt_tokens + total_result_bytes // CHARS_PER_TOKEN
        # Set system_prompt_tokens high enough to breach the limit.
        system_prompt_tokens = context_window - budget + 100  # 191_908
        session_stats: dict[str, Any] = {
            "tool_call_count": 0,
            "total_result_bytes": 0,
            "system_prompt_tokens": system_prompt_tokens,
            "user_prompt_tokens": 0,
            "force_chain": False,
        }
        hook = _build_pre_tool_hook(
            anima_dir,
            max_tokens=max_tokens,
            context_window=context_window,
            session_stats=session_stats,
        )
        input_data = {"tool_name": "Bash", "tool_input": {"command": "echo hi"}}
        result = await hook(input_data, "tool-789", {})

        # Estimated: 191908 + 0 = 191908.  Remaining = 200000 - 191908 = 8092 < 8192
        assert result.get("continue_") is False
        assert session_stats["force_chain"] is True
        assert "stopReason" in result
        assert "auto-compact" in result["stopReason"].lower() or "auto" in result["stopReason"].lower()

    @pytest.mark.asyncio
    async def test_total_result_bytes_contributes_to_estimate(self, anima_dir: Path):
        """large total_result_bytes pushes estimated tokens over the limit."""
        max_tokens = 4096
        context_window = 200_000
        budget = max_tokens * _CONTEXT_AUTOCOMPACT_SAFETY  # 8192

        # Set total_result_bytes large enough to trigger
        # estimated = 1000 + total_result_bytes // 4
        # We need estimated > 200000 - 8192 = 191808
        # So total_result_bytes // 4 > 190808 -> total_result_bytes > 763232
        session_stats: dict[str, Any] = {
            "tool_call_count": 5,
            "total_result_bytes": 770_000,
            "system_prompt_tokens": 1000,
            "user_prompt_tokens": 0,
            "force_chain": False,
        }
        hook = _build_pre_tool_hook(
            anima_dir,
            max_tokens=max_tokens,
            context_window=context_window,
            session_stats=session_stats,
        )
        input_data = {"tool_name": "Read", "tool_input": {"file_path": "/tmp/x"}}
        result = await hook(input_data, "tool-aaa", {})

        # estimated = 1000 + 770000//4 = 1000 + 192500 = 193500
        # remaining = 200000 - 193500 = 6500 < 8192
        assert result.get("continue_") is False
        assert session_stats["force_chain"] is True

    @pytest.mark.asyncio
    async def test_force_chain_already_true_still_blocks(self, anima_dir: Path):
        """When force_chain is already True (re-entry), subsequent calls still block."""
        max_tokens = 4096
        context_window = 200_000

        session_stats: dict[str, Any] = {
            "tool_call_count": 10,
            "total_result_bytes": 800_000,
            "system_prompt_tokens": 5000,
            "user_prompt_tokens": 0,
            "force_chain": True,  # already set from a previous trigger
        }
        hook = _build_pre_tool_hook(
            anima_dir,
            max_tokens=max_tokens,
            context_window=context_window,
            session_stats=session_stats,
        )
        input_data = {"tool_name": "Glob", "tool_input": {"pattern": "*.py"}}
        result = await hook(input_data, "tool-bbb", {})

        # estimated = 5000 + 800000//4 = 5000 + 200000 = 205000
        # remaining = 200000 - 205000 = -5000 < 8192 -> still triggers
        assert result.get("continue_") is False
        assert session_stats["force_chain"] is True

    @pytest.mark.asyncio
    async def test_tool_call_count_increments(self, anima_dir: Path):
        """Each hook invocation increments tool_call_count in session_stats."""
        session_stats: dict[str, Any] = {
            "tool_call_count": 0,
            "total_result_bytes": 0,
            "system_prompt_tokens": 100,
            "user_prompt_tokens": 0,
            "force_chain": False,
        }
        hook = _build_pre_tool_hook(
            anima_dir,
            max_tokens=4096,
            context_window=200_000,
            session_stats=session_stats,
        )
        input_data = {"tool_name": "Read", "tool_input": {"file_path": "/tmp/a"}}

        await hook(input_data, "t1", {})
        assert session_stats["tool_call_count"] == 1

        await hook(input_data, "t2", {})
        assert session_stats["tool_call_count"] == 2

        await hook(input_data, "t3", {})
        assert session_stats["tool_call_count"] == 3

    @pytest.mark.asyncio
    async def test_exact_boundary_does_not_trigger(self, anima_dir: Path):
        """When remaining == budget exactly, hook should NOT trigger (not strictly less)."""
        max_tokens = 4096
        context_window = 200_000
        budget = max_tokens * _CONTEXT_AUTOCOMPACT_SAFETY  # 8192

        # estimated = system_prompt_tokens + 0
        # We want remaining = context_window - estimated == budget
        # So estimated = context_window - budget = 191808
        session_stats: dict[str, Any] = {
            "tool_call_count": 0,
            "total_result_bytes": 0,
            "system_prompt_tokens": context_window - budget,
            "user_prompt_tokens": 0,
            "force_chain": False,
        }
        hook = _build_pre_tool_hook(
            anima_dir,
            max_tokens=max_tokens,
            context_window=context_window,
            session_stats=session_stats,
        )
        input_data = {"tool_name": "Read", "tool_input": {"file_path": "/tmp/b"}}
        result = await hook(input_data, "tool-ccc", {})

        # remaining = budget exactly -> not < budget -> should NOT trigger
        assert result.get("continue_") is not False
        assert session_stats["force_chain"] is False


# ── ExecutionResult.force_chain default ──────────────────────


class TestExecutionResultForceChain:
    """Test the force_chain field on ExecutionResult."""

    def test_default_is_false(self):
        """ExecutionResult() has force_chain=False by default."""
        result = ExecutionResult(text="hello")
        assert result.force_chain is False

    def test_explicit_true(self):
        """ExecutionResult(force_chain=True) works correctly."""
        result = ExecutionResult(text="hello", force_chain=True)
        assert result.force_chain is True

    def test_explicit_false(self):
        """ExecutionResult(force_chain=False) is explicitly False."""
        result = ExecutionResult(text="hello", force_chain=False)
        assert result.force_chain is False


# ── _tool_result_content_len with various block types ────────


class TestToolResultContentLenVariousBlocks:
    """Test _tool_result_content_len with various block structures for bytes tracking."""

    def test_large_text_content(self):
        """Large string content returns accurate length."""
        big_text = "x" * 100_000
        block = SimpleNamespace(content=big_text)
        assert _tool_result_content_len(block) == 100_000

    def test_list_with_many_items(self):
        """List with many text items sums all lengths correctly."""
        items = [{"text": f"item_{i}"} for i in range(100)]
        block = SimpleNamespace(content=items)
        expected = sum(len(f"item_{i}") for i in range(100))
        assert _tool_result_content_len(block) == expected

    def test_unicode_content(self):
        """Unicode string content returns character count (not byte count)."""
        # _tool_result_content_len uses len(str(...)), which counts characters
        text = "日本語テスト"
        block = SimpleNamespace(content=text)
        assert _tool_result_content_len(block) == len(text)  # 6 chars

    def test_list_with_unicode_text(self):
        """List items with unicode text sum character lengths."""
        block = SimpleNamespace(content=[
            {"text": "Hello"},
            {"text": "日本語"},
        ])
        assert _tool_result_content_len(block) == len("Hello") + len("日本語")

    def test_mixed_content_types_in_list(self):
        """Mixed dict/non-dict items in list only counts dict items."""
        block = SimpleNamespace(content=[
            {"text": "abc"},
            None,
            {"text": "def"},
            123,
            {"image": "base64data"},  # dict but no "text" key -> len("")=0
        ])
        assert _tool_result_content_len(block) == 3 + 3 + 0

    def test_boolean_content(self):
        """Boolean content is str()-ified: str(True) = 'True' -> len 4."""
        block = SimpleNamespace(content=True)
        assert _tool_result_content_len(block) == len("True")

    def test_dict_content_stringified(self):
        """Non-list, non-string content (e.g. dict) is str()-ified."""
        block = SimpleNamespace(content={"key": "value"})
        expected_len = len(str({"key": "value"}))
        assert _tool_result_content_len(block) == expected_len
