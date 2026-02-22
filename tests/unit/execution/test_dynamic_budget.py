"""Unit tests for dynamic tool-record budget functions in core.execution.base.

The dynamic budget system scales per-tool character budgets proportionally
to the model's context window size, with clamping at ``_BUDGET_SCALE_MIN``
(0.25x) and ``_BUDGET_SCALE_MAX`` (2.0x), and a hard floor of
``_BUDGET_FLOOR`` (300 chars) for result budgets / 200 chars for input budgets.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from core.execution.base import tool_result_save_budget, tool_input_save_budget


# ── Reference context (128K) ────────────────────────────────
# At the reference window the scale factor is 1.0, so budget == base value.

_REF = 128_000


class TestReferenceContext:
    """Budget equals the base value at the 128K reference context window."""

    def test_read_tool_base_budget(self):
        assert tool_result_save_budget("Read", _REF) == 4000

    def test_grep_tool_base_budget(self):
        assert tool_result_save_budget("Grep", _REF) == 4000

    def test_glob_tool_base_budget(self):
        assert tool_result_save_budget("Glob", _REF) == 4000

    def test_bash_tool_base_budget(self):
        assert tool_result_save_budget("Bash", _REF) == 2000

    def test_web_search_base_budget(self):
        assert tool_result_save_budget("web_search", _REF) == 1500

    def test_x_search_base_budget(self):
        assert tool_result_save_budget("x_search", _REF) == 1500

    def test_write_file_base_budget(self):
        assert tool_result_save_budget("write_file", _REF) == 500

    def test_edit_file_base_budget(self):
        assert tool_result_save_budget("edit_file", _REF) == 500

    def test_search_memory_base_budget(self):
        assert tool_result_save_budget("search_memory", _REF) == 1500

    def test_read_file_base_budget(self):
        assert tool_result_save_budget("read_file", _REF) == 4000


# ── Large context (256K) ────────────────────────────────────
# 256K / 128K = 2.0x, which equals _BUDGET_SCALE_MAX.


class TestLargeContext:
    """Budget scales up by 2x at 256K (hits _BUDGET_SCALE_MAX)."""

    def test_read_tool_doubled(self):
        assert tool_result_save_budget("Read", 256_000) == 8000

    def test_bash_tool_doubled(self):
        assert tool_result_save_budget("Bash", 256_000) == 4000

    def test_web_search_doubled(self):
        assert tool_result_save_budget("web_search", 256_000) == 3000

    def test_write_file_doubled(self):
        assert tool_result_save_budget("write_file", 256_000) == 1000

    def test_unknown_tool_doubled(self):
        # Default budget 1000 * 2.0 = 2000
        assert tool_result_save_budget("unknown_tool", 256_000) == 2000


# ── Small context (32K) ─────────────────────────────────────
# 32K / 128K = 0.25x, which equals _BUDGET_SCALE_MIN.


class TestSmallContext:
    """Budget scales down by 0.25x at 32K (hits _BUDGET_SCALE_MIN)."""

    def test_read_tool_quarter(self):
        assert tool_result_save_budget("Read", 32_000) == 1000

    def test_bash_tool_quarter(self):
        assert tool_result_save_budget("Bash", 32_000) == 500

    def test_web_search_quarter(self):
        assert tool_result_save_budget("web_search", 32_000) == 375

    def test_write_file_quarter(self):
        # 500 * 0.25 = 125, but _BUDGET_FLOOR = 300 kicks in
        assert tool_result_save_budget("write_file", 32_000) == 300

    def test_edit_file_quarter(self):
        # 500 * 0.25 = 125, but _BUDGET_FLOOR = 300 kicks in
        assert tool_result_save_budget("edit_file", 32_000) == 300


# ── Very large context (1M) ─────────────────────────────────
# 1M / 128K = 7.8125x, capped at _BUDGET_SCALE_MAX = 2.0x.


class TestVeryLargeContext:
    """Budget capped at _BUDGET_SCALE_MAX (2.0x) for 1M+ windows."""

    def test_read_tool_capped_at_2x(self):
        assert tool_result_save_budget("Read", 1_000_000) == 8000

    def test_bash_tool_capped_at_2x(self):
        assert tool_result_save_budget("Bash", 1_000_000) == 4000

    def test_write_file_capped_at_2x(self):
        assert tool_result_save_budget("write_file", 1_000_000) == 1000

    def test_unknown_tool_capped_at_2x(self):
        assert tool_result_save_budget("some_future_tool", 1_000_000) == 2000


# ── Very small context (8K) ─────────────────────────────────
# 8K / 128K = 0.0625x, clamped to _BUDGET_SCALE_MIN = 0.25x.
# Even at 0.25x, small base budgets are lifted to _BUDGET_FLOOR (300).


class TestVerySmallContext:
    """Budget clamped at _BUDGET_SCALE_MIN (0.25x), floor enforced."""

    def test_read_tool_clamped(self):
        # 4000 * 0.25 = 1000 (above floor)
        assert tool_result_save_budget("Read", 8_000) == 1000

    def test_bash_tool_clamped(self):
        # 2000 * 0.25 = 500 (above floor)
        assert tool_result_save_budget("Bash", 8_000) == 500

    def test_write_file_clamped_to_floor(self):
        # 500 * 0.25 = 125, floor = 300
        assert tool_result_save_budget("write_file", 8_000) == 300

    def test_edit_file_clamped_to_floor(self):
        # 500 * 0.25 = 125, floor = 300
        assert tool_result_save_budget("edit_file", 8_000) == 300

    def test_web_search_clamped(self):
        # 1500 * 0.25 = 375 (above floor)
        assert tool_result_save_budget("web_search", 8_000) == 375


# ── Unknown tool name ────────────────────────────────────────
# Falls back to _TOOL_RESULT_DEFAULT_BUDGET (1000).


class TestUnknownToolName:
    """Unknown tool names use the default budget of 1000."""

    def test_unknown_tool_at_reference(self):
        assert tool_result_save_budget("nonexistent_tool", _REF) == 1000

    def test_unknown_tool_scales_up(self):
        # 1000 * 2.0 = 2000
        assert tool_result_save_budget("custom_widget", 256_000) == 2000

    def test_unknown_tool_scales_down(self):
        # 1000 * 0.25 = 250, floor = 300
        assert tool_result_save_budget("custom_widget", 32_000) == 300

    def test_empty_string_tool_name(self):
        assert tool_result_save_budget("", _REF) == 1000


# ── Tool input budget ────────────────────────────────────────
# Uses _TOOL_INPUT_BASE_BUDGET = 500, same scaling, floor = 200.


class TestToolInputBudget:
    """tool_input_save_budget uses base=500 and floor=200."""

    def test_reference_context(self):
        assert tool_input_save_budget(_REF) == 500

    def test_large_context_doubled(self):
        # 500 * 2.0 = 1000
        assert tool_input_save_budget(256_000) == 1000

    def test_small_context_quarter(self):
        # 500 * 0.25 = 125, floor = 200
        assert tool_input_save_budget(32_000) == 200

    def test_very_large_context_capped(self):
        # 500 * 2.0 = 1000 (scale capped at 2.0)
        assert tool_input_save_budget(1_000_000) == 1000

    def test_very_small_context_floor(self):
        # 500 * 0.25 = 125, floor = 200
        assert tool_input_save_budget(8_000) == 200

    def test_mid_range_context(self):
        # 64K / 128K = 0.5 → 500 * 0.5 = 250 (above floor)
        assert tool_input_save_budget(64_000) == 250


# ── Floor enforcement ────────────────────────────────────────
# Even extreme inputs produce at least _BUDGET_FLOOR (300) for results
# and 200 for inputs.


class TestFloorEnforcement:
    """Hard floor ensures minimum budgets even with tiny context windows."""

    def test_result_budget_never_below_floor_small_base(self):
        # write_file base=500, scale clamped to 0.25 → 125 → floor 300
        assert tool_result_save_budget("write_file", 1) == 300

    def test_result_budget_never_below_floor_default_base(self):
        # default base=1000, scale clamped to 0.25 → 250 → floor 300
        assert tool_result_save_budget("unknown", 1) == 300

    def test_result_budget_floor_with_zero_context(self):
        # context_window=0 → scale=0, clamped to 0.25 → 4000*0.25=1000
        assert tool_result_save_budget("Read", 0) == 1000

    def test_result_budget_floor_with_zero_context_small_base(self):
        # write_file base=500, scale clamped to 0.25 → 125 → floor 300
        assert tool_result_save_budget("write_file", 0) == 300

    def test_input_budget_never_below_floor(self):
        # 500 * 0.25 = 125, floor = 200
        assert tool_input_save_budget(1) == 200

    def test_input_budget_floor_with_zero_context(self):
        assert tool_input_save_budget(0) == 200

    @pytest.mark.parametrize("tool_name", [
        "Read", "Grep", "Glob", "Bash", "web_search", "x_search",
        "write_file", "edit_file", "search_memory", "read_file",
    ])
    def test_all_known_tools_respect_floor(self, tool_name: str):
        """Every known tool produces at least _BUDGET_FLOOR with a tiny context."""
        budget = tool_result_save_budget(tool_name, 1)
        assert budget >= 300, f"{tool_name} budget {budget} < floor 300"
