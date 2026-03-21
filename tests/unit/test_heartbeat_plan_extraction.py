from __future__ import annotations

import pytest

from core._anima_heartbeat import _extract_plan_summary, _MAX_PLAN_SUMMARY_CHARS


class TestExtractPlanSummary:
    """Unit tests for _extract_plan_summary — extracting the Plan section
    from heartbeat model output for plan-outcome tracking."""

    def test_basic_extraction(self):
        text = (
            "## Observe\nAll clear.\n\n"
            "## Plan\nDelegate Issue #2183 to hinata.\n"
            "Follow up on STALE tasks.\n\n"
            "## Reflect\nNothing to add."
        )
        result = _extract_plan_summary(text)
        assert "Delegate Issue #2183" in result
        assert "Follow up on STALE" in result

    def test_plan_with_japanese_heading(self):
        text = (
            "## Observe（観察）\n確認済み。\n\n"
            "## Plan（計画）\nhinataにIssue #2183を委譲する。\n\n"
            "## Reflect（振り返り）\n特になし。"
        )
        result = _extract_plan_summary(text)
        assert "hinataにIssue #2183" in result

    def test_no_plan_section_returns_empty(self):
        text = "## Observe\nAll clear.\n## Reflect\nNothing."
        assert _extract_plan_summary(text) == ""

    def test_empty_string_returns_empty(self):
        assert _extract_plan_summary("") == ""

    def test_none_like_empty(self):
        assert _extract_plan_summary("") == ""

    def test_plan_at_end_of_text(self):
        text = "## Observe\nAll clear.\n\n## Plan\nSubmit 3 tasks."
        result = _extract_plan_summary(text)
        assert "Submit 3 tasks" in result

    def test_truncation_at_max_chars(self):
        long_plan = "A" * 1000
        text = f"## Observe\nOk.\n\n## Plan\n{long_plan}\n\n## Reflect\nDone."
        result = _extract_plan_summary(text)
        assert len(result) <= _MAX_PLAN_SUMMARY_CHARS

    def test_multiline_plan_content(self):
        text = (
            "## Observe\nChecked board.\n\n"
            "## Plan\n"
            "1. delegate_task to natsume for PR #2185\n"
            "2. submit_tasks for STALE follow-up\n"
            "3. send_message to sakura (report)\n\n"
            "## Reflect\n[REFLECTION]\nGood progress.\n[/REFLECTION]"
        )
        result = _extract_plan_summary(text)
        assert "delegate_task" in result
        assert "submit_tasks" in result
        assert "send_message" in result

    def test_plan_without_other_sections(self):
        text = "## Plan\nJust plan, nothing else."
        result = _extract_plan_summary(text)
        assert "Just plan" in result

    def test_does_not_capture_reflect_content(self):
        text = (
            "## Plan\nDelegate to hinata.\n\n"
            "## Reflect\nThis should not be in plan."
        )
        result = _extract_plan_summary(text)
        assert "should not be in plan" not in result
        assert "Delegate to hinata" in result


class TestPlanSummaryInHeartbeatMeta:
    """Verify that _extract_plan_summary integrates with activity log meta."""

    def test_non_empty_plan_produces_meta(self):
        text = "## Observe\nOk.\n\n## Plan\nAction item X.\n\n## Reflect\nDone."
        plan = _extract_plan_summary(text)
        meta = {"plan_summary": plan} if plan else None
        assert meta is not None
        assert meta["plan_summary"] == "Action item X."

    def test_missing_plan_produces_no_meta(self):
        text = "## Observe\nOk.\n## Reflect\nDone."
        plan = _extract_plan_summary(text)
        meta = {"plan_summary": plan} if plan else None
        assert meta is None
