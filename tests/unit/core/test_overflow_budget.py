from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DK overflow budget cleanup."""

import core.prompt.builder as builder


def test_dk_summary_budget_constants_removed() -> None:
    """DK summary injection budgets are no longer exported by builder."""
    assert not hasattr(builder, "_PROC_SUMMARY_BUDGET")
    assert not hasattr(builder, "_KNOW_SUMMARY_BUDGET")


def test_reference_window_still_exported_for_prompt_scaling() -> None:
    """Non-DK prompt scaling still uses the shared reference window."""
    assert builder._REFERENCE_WINDOW == 128_000
