from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for intent-aware priming budget behavior."""

from core.memory.priming import PrimingEngine


def test_intent_budget_mapping_e2e(tmp_path) -> None:
    """Intent should map to stable dynamic budgets in the real PrimingEngine."""
    anima_dir = tmp_path / "animas" / "sakura"
    (anima_dir / "episodes").mkdir(parents=True)
    (anima_dir / "knowledge").mkdir()
    (anima_dir / "skills").mkdir()
    engine = PrimingEngine(anima_dir)

    delegation_budget = engine._adjust_token_budget("ok", "chat", intent="delegation")
    report_budget = engine._adjust_token_budget("ok", "chat", intent="report")
    question_budget = engine._adjust_token_budget("ok", "chat", intent="question")
    fallback_budget = engine._adjust_token_budget("こんにちは", "chat", intent="")

    assert delegation_budget == 3000
    assert report_budget == 1500
    assert question_budget == 1500
    assert fallback_budget == 500
