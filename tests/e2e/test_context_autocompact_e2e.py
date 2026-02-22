# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for A1 mid-session context auto-compact feature.

Verifies the full flow: PreToolUse hook context budget check,
session_stats accumulation, force_chain propagation through
ExecutionResult to ContextTracker, and boundary conditions across
different context window sizes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

# ── Mock claude_agent_sdk before importing agent_sdk module ───
# The SDK may not be installed in the test environment, so we inject
# lightweight mock modules that satisfy the import at module level.

_mock_sdk = MagicMock()
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
from core.prompt.context import CHARS_PER_TOKEN, ContextTracker  # noqa: E402


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    """Create a minimal anima directory structure."""
    d = tmp_path / "animas" / "test-autocompact"
    for sub in (
        "state", "episodes", "knowledge", "procedures",
        "skills", "shortterm", "activity_log",
    ):
        (d / sub).mkdir(parents=True)
    (d / "identity.md").write_text("# Test Anima", encoding="utf-8")
    return d


# ── Helpers ──────────────────────────────────────────────────


def _make_session_stats(
    *,
    tool_call_count: int = 0,
    total_result_bytes: int = 0,
    system_prompt_tokens: int = 0,
    user_prompt_tokens: int = 0,
    force_chain: bool = False,
) -> dict[str, Any]:
    """Create a session_stats dict with standard keys."""
    return {
        "tool_call_count": tool_call_count,
        "total_result_bytes": total_result_bytes,
        "system_prompt_tokens": system_prompt_tokens,
        "user_prompt_tokens": user_prompt_tokens,
        "force_chain": force_chain,
    }


# ---------------------------------------------------------------------------
# Test 1: PreToolUse hook full flow with realistic parameters
# ---------------------------------------------------------------------------


class TestPreToolHookContextAutoCompact:
    """Test the full PreToolUse hook flow with realistic parameters."""

    @pytest.mark.asyncio
    async def test_under_budget_proceeds_normally(self, anima_dir: Path):
        """When well under budget, the hook allows the tool call."""
        session_stats = _make_session_stats(
            system_prompt_tokens=2000,
            total_result_bytes=40_000,
        )
        hook = _build_pre_tool_hook(
            anima_dir,
            max_tokens=4096,
            context_window=200_000,
            session_stats=session_stats,
        )
        input_data = {"tool_name": "Read", "tool_input": {"file_path": "/some/path"}}
        result = await hook(input_data, "tool-001", {})

        # estimated = 2000 + 40000/4 = 12000.  remaining = 188000 >> 8192
        assert result.get("continue_") is not False
        assert session_stats["force_chain"] is False

    @pytest.mark.asyncio
    async def test_exceeds_budget_returns_continue_false(self, anima_dir: Path):
        """When estimated tokens exceed budget, hook returns continue_=False."""
        max_tokens = 4096
        context_window = 200_000
        budget = max_tokens * _CONTEXT_AUTOCOMPACT_SAFETY  # 8192

        # Set values so remaining < budget
        # estimated = 195_000 + 0 = 195_000.  remaining = 5000 < 8192
        session_stats = _make_session_stats(system_prompt_tokens=195_000)
        hook = _build_pre_tool_hook(
            anima_dir,
            max_tokens=max_tokens,
            context_window=context_window,
            session_stats=session_stats,
        )
        input_data = {"tool_name": "Grep", "tool_input": {"pattern": "test"}}
        result = await hook(input_data, "tool-002", {})

        assert result.get("continue_") is False
        assert session_stats["force_chain"] is True
        assert "stopReason" in result

    @pytest.mark.asyncio
    async def test_force_chain_set_to_true(self, anima_dir: Path):
        """When budget is exceeded, session_stats['force_chain'] is set True."""
        session_stats = _make_session_stats(
            system_prompt_tokens=198_000,
        )
        hook = _build_pre_tool_hook(
            anima_dir,
            max_tokens=4096,
            context_window=200_000,
            session_stats=session_stats,
        )
        input_data = {"tool_name": "Bash", "tool_input": {"command": "ls"}}
        result = await hook(input_data, "tool-003", {})

        assert result.get("continue_") is False
        assert session_stats["force_chain"] is True

    @pytest.mark.asyncio
    async def test_security_checks_still_work_after_budget_check(
        self, anima_dir: Path,
    ):
        """Write path security check still blocks even when budget is fine."""
        session_stats = _make_session_stats(
            system_prompt_tokens=1000,
            total_result_bytes=4000,
        )
        hook = _build_pre_tool_hook(
            anima_dir,
            max_tokens=4096,
            context_window=200_000,
            session_stats=session_stats,
        )

        # Attempt to write to a protected file (identity.md)
        protected_path = str(anima_dir / "identity.md")
        input_data = {
            "tool_name": "Write",
            "tool_input": {"file_path": protected_path, "content": "hacked"},
        }
        result = await hook(input_data, "tool-004", {})

        # Should be denied by the security check, not by budget
        assert session_stats["force_chain"] is False
        hook_output = result.get("hookSpecificOutput", {})
        if isinstance(hook_output, dict):
            assert hook_output.get("permissionDecision") == "deny"

    @pytest.mark.asyncio
    async def test_security_deny_for_other_anima_read(self, anima_dir: Path):
        """Reading another anima's directory is blocked."""
        session_stats = _make_session_stats(system_prompt_tokens=1000)
        hook = _build_pre_tool_hook(
            anima_dir,
            max_tokens=4096,
            context_window=200_000,
            session_stats=session_stats,
        )

        # Try to read from a sibling anima directory
        other_anima_path = str(anima_dir.parent / "other-anima" / "identity.md")
        input_data = {
            "tool_name": "Read",
            "tool_input": {"file_path": other_anima_path},
        }
        result = await hook(input_data, "tool-005", {})

        hook_output = result.get("hookSpecificOutput", {})
        if isinstance(hook_output, dict):
            assert hook_output.get("permissionDecision") == "deny"

    @pytest.mark.asyncio
    async def test_result_bytes_accumulation_triggers_compact(
        self, anima_dir: Path,
    ):
        """Large total_result_bytes alone can trigger auto-compact."""
        max_tokens = 4096
        context_window = 200_000
        budget = max_tokens * _CONTEXT_AUTOCOMPACT_SAFETY  # 8192

        # estimated = 1000 + 780_000/4 = 1000 + 195_000 = 196_000
        # remaining = 200_000 - 196_000 = 4000 < 8192
        session_stats = _make_session_stats(
            system_prompt_tokens=1000,
            total_result_bytes=780_000,
        )
        hook = _build_pre_tool_hook(
            anima_dir,
            max_tokens=max_tokens,
            context_window=context_window,
            session_stats=session_stats,
        )
        input_data = {"tool_name": "Glob", "tool_input": {"pattern": "*.py"}}
        result = await hook(input_data, "tool-006", {})

        assert result.get("continue_") is False
        assert session_stats["force_chain"] is True

    @pytest.mark.asyncio
    async def test_multiple_calls_accumulate_tool_call_count(
        self, anima_dir: Path,
    ):
        """Each hook call increments tool_call_count."""
        session_stats = _make_session_stats(system_prompt_tokens=100)
        hook = _build_pre_tool_hook(
            anima_dir,
            max_tokens=4096,
            context_window=200_000,
            session_stats=session_stats,
        )
        input_data = {"tool_name": "Read", "tool_input": {"file_path": "/tmp/x"}}

        for i in range(5):
            await hook(input_data, f"tool-{i}", {})

        assert session_stats["tool_call_count"] == 5


# ---------------------------------------------------------------------------
# Test 2: session_stats accumulation across the flow
# ---------------------------------------------------------------------------


class TestSessionStatsAccumulation:
    """Test that session_stats properly accumulates data across the flow."""

    def test_tool_result_content_len_with_list_content(self):
        """_tool_result_content_len correctly sums list content text lengths."""
        block = SimpleNamespace(content=[
            {"text": "line 1 of output\n"},
            {"text": "line 2 of output\n"},
            {"text": "line 3 of output\n"},
        ])
        expected = sum(
            len(t["text"]) for t in block.content
        )
        assert _tool_result_content_len(block) == expected

    def test_tool_result_content_len_with_string_content(self):
        """_tool_result_content_len returns len for string content."""
        big_result = "x" * 50_000
        block = SimpleNamespace(content=big_result)
        assert _tool_result_content_len(block) == 50_000

    def test_tool_result_content_len_none_returns_zero(self):
        """_tool_result_content_len returns 0 for None content."""
        block = SimpleNamespace(content=None)
        assert _tool_result_content_len(block) == 0

    def test_accumulation_across_multiple_blocks(self):
        """Simulating multiple tool result blocks accumulating bytes."""
        session_stats = _make_session_stats(system_prompt_tokens=3000)

        blocks = [
            SimpleNamespace(content="A" * 10_000),
            SimpleNamespace(content=[{"text": "B" * 5_000}]),
            SimpleNamespace(content="C" * 20_000),
            SimpleNamespace(content=[
                {"text": "D" * 3_000},
                {"text": "E" * 7_000},
            ]),
        ]

        for block in blocks:
            session_stats["total_result_bytes"] += _tool_result_content_len(block)

        # 10000 + 5000 + 20000 + 3000 + 7000 = 45000
        assert session_stats["total_result_bytes"] == 45_000

    def test_system_prompt_tokens_calculation(self):
        """system_prompt_tokens is correctly calculated from system_prompt length."""
        system_prompt = "x" * 80_000  # 80K chars
        system_prompt_tokens = len(system_prompt) // CHARS_PER_TOKEN
        assert system_prompt_tokens == 20_000

    def test_estimated_tokens_formula(self):
        """The estimation formula matches the hook's implementation."""
        system_prompt_tokens = 5_000
        user_prompt_tokens = 2_000
        total_result_bytes = 200_000

        estimated = (
            system_prompt_tokens
            + user_prompt_tokens
            + total_result_bytes // CHARS_PER_TOKEN
        )
        assert estimated == 5_000 + 2_000 + 50_000
        assert estimated == 57_000

    def test_realistic_session_accumulation(self):
        """Simulate a realistic session with multiple tool calls accumulating."""
        # Start with a 20K token system prompt (80K chars)
        system_prompt = "S" * 80_000
        session_stats = _make_session_stats(
            system_prompt_tokens=len(system_prompt) // CHARS_PER_TOKEN,
        )
        assert session_stats["system_prompt_tokens"] == 20_000

        # Simulate 10 Read tool calls returning ~8K chars each
        for _ in range(10):
            block = SimpleNamespace(content="R" * 8_000)
            session_stats["total_result_bytes"] += _tool_result_content_len(block)
            session_stats["tool_call_count"] += 1

        assert session_stats["total_result_bytes"] == 80_000
        assert session_stats["tool_call_count"] == 10

        # estimated = 20_000 + 0 + 80_000/4 = 20_000 + 20_000 = 40_000 tokens
        estimated = (
            session_stats["system_prompt_tokens"]
            + session_stats["user_prompt_tokens"]
            + session_stats["total_result_bytes"] // CHARS_PER_TOKEN
        )
        assert estimated == 40_000


# ---------------------------------------------------------------------------
# Test 3: force_chain flag flows through the full chain
# ---------------------------------------------------------------------------


class TestForceChainPropagation:
    """Test the force_chain flag flows through the full chain."""

    def test_execution_result_force_chain_accessible(self):
        """ExecutionResult with force_chain=True is accessible."""
        result = ExecutionResult(text="response", force_chain=True)
        assert result.force_chain is True

    def test_execution_result_force_chain_default_false(self):
        """ExecutionResult defaults force_chain to False."""
        result = ExecutionResult(text="response")
        assert result.force_chain is False

    def test_context_tracker_threshold_hit_triggers_exceeded(self):
        """force_threshold() makes threshold_exceeded return True."""
        tracker = ContextTracker(model="claude-sonnet-4-20250514", threshold=0.5)
        assert tracker.threshold_exceeded is False

        tracker.force_threshold()
        assert tracker.threshold_exceeded is True

    def test_force_chain_propagation_logic(self):
        """The agent.py logic: if force_chain and not exceeded -> force_threshold()."""
        # Simulate the agent.py logic from lines ~937-942
        result = ExecutionResult(text="response", force_chain=True)
        tracker = ContextTracker(model="claude-sonnet-4-20250514", threshold=0.5)

        # Before: threshold not exceeded
        assert not tracker.threshold_exceeded

        # Apply the agent.py logic
        if result.force_chain and not tracker.threshold_exceeded:
            tracker.force_threshold()

        # After: threshold is now exceeded
        assert tracker.threshold_exceeded is True

    def test_force_chain_false_does_not_affect_tracker(self):
        """When force_chain is False, tracker state is unchanged."""
        result = ExecutionResult(text="response", force_chain=False)
        tracker = ContextTracker(model="claude-sonnet-4-20250514", threshold=0.5)

        if result.force_chain and not tracker.threshold_exceeded:
            tracker.force_threshold()

        assert tracker.threshold_exceeded is False

    def test_already_exceeded_not_double_set(self):
        """When threshold is already exceeded, force_chain=True is a no-op."""
        result = ExecutionResult(text="response", force_chain=True)
        tracker = ContextTracker(model="claude-sonnet-4-20250514", threshold=0.5)

        # Simulate threshold already hit via normal usage
        tracker.force_threshold()
        assert tracker.threshold_exceeded is True

        # force_chain logic: condition `not tracker.threshold_exceeded` is False
        # so force_threshold should NOT be called (it's already True)
        if result.force_chain and not tracker.threshold_exceeded:
            # This branch should NOT execute
            tracker._threshold_hit = False  # would break if reached

        # Still True (the branch was skipped)
        assert tracker.threshold_exceeded is True

    def test_context_tracker_reset_clears_threshold(self):
        """ContextTracker.reset() clears threshold set by force_threshold()."""
        tracker = ContextTracker(model="claude-sonnet-4-20250514", threshold=0.5)
        tracker.force_threshold()
        assert tracker.threshold_exceeded is True

        tracker.reset()
        assert tracker.threshold_exceeded is False


# ---------------------------------------------------------------------------
# Test 4: Boundary conditions with various context window sizes
# ---------------------------------------------------------------------------


class TestAutoCompactBoundaryConditions:
    """Test edge cases with various context window sizes and model configs."""

    @pytest.mark.asyncio
    async def test_small_context_window_8k(self, anima_dir: Path):
        """Small context window (8K, Ollama-like) triggers earlier."""
        max_tokens = 512
        context_window = 8_000
        budget = max_tokens * _CONTEXT_AUTOCOMPACT_SAFETY  # 1024

        # estimated = 6500 + 2000/4 = 6500 + 500 = 7000
        # remaining = 8000 - 7000 = 1000 < 1024 -> triggers
        session_stats = _make_session_stats(
            system_prompt_tokens=6500,
            total_result_bytes=2000,
        )
        hook = _build_pre_tool_hook(
            anima_dir,
            max_tokens=max_tokens,
            context_window=context_window,
            session_stats=session_stats,
        )
        input_data = {"tool_name": "Read", "tool_input": {"file_path": "/tmp/x"}}
        result = await hook(input_data, "tool-boundary-1", {})

        assert result.get("continue_") is False
        assert session_stats["force_chain"] is True

    @pytest.mark.asyncio
    async def test_large_context_window_200k(self, anima_dir: Path):
        """Large context window (200K, Claude-like) requires much more data."""
        max_tokens = 8192
        context_window = 200_000
        budget = max_tokens * _CONTEXT_AUTOCOMPACT_SAFETY  # 16384

        # estimated = 50_000 + 400_000/4 = 50_000 + 100_000 = 150_000
        # remaining = 200_000 - 150_000 = 50_000 >> 16384 -> does NOT trigger
        session_stats = _make_session_stats(
            system_prompt_tokens=50_000,
            total_result_bytes=400_000,
        )
        hook = _build_pre_tool_hook(
            anima_dir,
            max_tokens=max_tokens,
            context_window=context_window,
            session_stats=session_stats,
        )
        input_data = {"tool_name": "Grep", "tool_input": {"pattern": "error"}}
        result = await hook(input_data, "tool-boundary-2", {})

        assert result.get("continue_") is not False
        assert session_stats["force_chain"] is False

    @pytest.mark.asyncio
    async def test_exact_boundary_does_not_trigger(self, anima_dir: Path):
        """When remaining == budget exactly, should NOT trigger (not strictly less)."""
        max_tokens = 4096
        context_window = 200_000
        budget = max_tokens * _CONTEXT_AUTOCOMPACT_SAFETY  # 8192

        # estimated = context_window - budget = 191808
        # remaining = budget exactly -> NOT < budget -> should NOT trigger
        session_stats = _make_session_stats(
            system_prompt_tokens=context_window - budget,
        )
        hook = _build_pre_tool_hook(
            anima_dir,
            max_tokens=max_tokens,
            context_window=context_window,
            session_stats=session_stats,
        )
        input_data = {"tool_name": "Read", "tool_input": {"file_path": "/tmp/x"}}
        result = await hook(input_data, "tool-boundary-3", {})

        assert result.get("continue_") is not False
        assert session_stats["force_chain"] is False

    @pytest.mark.asyncio
    async def test_one_token_over_boundary_triggers(self, anima_dir: Path):
        """When remaining == budget - 1, should trigger (strictly less)."""
        max_tokens = 4096
        context_window = 200_000
        budget = max_tokens * _CONTEXT_AUTOCOMPACT_SAFETY  # 8192

        # estimated = context_window - budget + 1 = 191809
        # remaining = budget - 1 = 8191 < 8192 -> triggers
        session_stats = _make_session_stats(
            system_prompt_tokens=context_window - budget + 1,
        )
        hook = _build_pre_tool_hook(
            anima_dir,
            max_tokens=max_tokens,
            context_window=context_window,
            session_stats=session_stats,
        )
        input_data = {"tool_name": "Bash", "tool_input": {"command": "echo test"}}
        result = await hook(input_data, "tool-boundary-4", {})

        assert result.get("continue_") is False
        assert session_stats["force_chain"] is True

    def test_budget_formula_verification(self):
        """Verify the budget formula matches the implementation."""
        test_cases = [
            # (max_tokens, context_window, system_prompt_tokens, user_prompt_tokens, total_bytes, should_trigger)
            (4096, 200_000, 191_808, 0, 0, False),    # remaining == budget exactly
            (4096, 200_000, 191_809, 0, 0, True),      # remaining == budget - 1
            (512, 8_000, 6_976, 0, 0, False),          # 8K: remaining == 8000-6976=1024 == budget
            (512, 8_000, 6_977, 0, 0, True),           # 8K: remaining == 8000-6977=1023 < 1024
            (8192, 200_000, 183_616, 0, 0, False),     # large max_tokens: remaining==16384
            (8192, 200_000, 183_617, 0, 0, True),      # large max_tokens: remaining==16383
            (4096, 200_000, 190_000, 1_808, 0, False),  # user_prompt contributes to boundary
            (4096, 200_000, 190_000, 1_809, 0, True),   # user_prompt pushes over boundary
        ]

        for max_tokens, cw, spt, upt, total_bytes, expected in test_cases:
            budget = max_tokens * _CONTEXT_AUTOCOMPACT_SAFETY
            estimated = spt + upt + total_bytes // CHARS_PER_TOKEN
            remaining = cw - estimated
            should_trigger = remaining < budget
            assert should_trigger == expected, (
                f"max_tokens={max_tokens} cw={cw} spt={spt} upt={upt} bytes={total_bytes}: "
                f"estimated={estimated} remaining={remaining} budget={budget} "
                f"expected={expected} got={should_trigger}"
            )

    @pytest.mark.asyncio
    async def test_zero_max_tokens(self, anima_dir: Path):
        """max_tokens=0 means budget=0, so remaining is always >= budget."""
        session_stats = _make_session_stats(system_prompt_tokens=199_999)
        hook = _build_pre_tool_hook(
            anima_dir,
            max_tokens=0,
            context_window=200_000,
            session_stats=session_stats,
        )
        input_data = {"tool_name": "Read", "tool_input": {"file_path": "/tmp/x"}}
        result = await hook(input_data, "tool-boundary-5", {})

        # budget = 0 * 2 = 0.  remaining = 1 >= 0 -> does NOT trigger
        assert result.get("continue_") is not False
        assert session_stats["force_chain"] is False

    @pytest.mark.asyncio
    async def test_estimated_exceeds_context_window(self, anima_dir: Path):
        """When estimated > context_window (negative remaining), triggers."""
        session_stats = _make_session_stats(
            system_prompt_tokens=150_000,
            total_result_bytes=300_000,  # 75_000 additional tokens
        )
        hook = _build_pre_tool_hook(
            anima_dir,
            max_tokens=4096,
            context_window=200_000,
            session_stats=session_stats,
        )
        input_data = {"tool_name": "Bash", "tool_input": {"command": "pwd"}}
        result = await hook(input_data, "tool-boundary-6", {})

        # estimated = 150_000 + 75_000 = 225_000 > context_window
        # remaining = 200_000 - 225_000 = -25_000 < 8192
        assert result.get("continue_") is False
        assert session_stats["force_chain"] is True
