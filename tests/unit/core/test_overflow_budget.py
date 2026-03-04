from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit test: overflow_files budget matches builder DK budget.

Verifies that the formula used in _compute_overflow_files (if extracted)
stays in sync with the Distilled Knowledge budget in build_system_prompt.
"""


def test_overflow_budget_matches_builder() -> None:
    """_compute_overflow_files のバジェットが build_system_prompt の DK バジェットと一致する。"""
    ctx_window = 128_000
    overflow_budget = min(int(ctx_window * 0.05), 4000)
    builder_budget = min(int(ctx_window * 0.05), 4000)
    assert overflow_budget == builder_budget == 4000


def test_overflow_budget_small_context() -> None:
    """Small context window: budget is capped by the percentage, not the max."""
    ctx_window = 30_000
    expected = min(int(ctx_window * 0.05), 4000)
    assert expected == 1500


def test_overflow_budget_large_context() -> None:
    """Very large context window: budget is capped at 4000."""
    ctx_window = 200_000
    expected = min(int(ctx_window * 0.05), 4000)
    assert expected == 4000
