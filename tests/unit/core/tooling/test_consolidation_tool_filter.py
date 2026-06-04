"""Tests for consolidation-trigger tool filtering in build_unified_tool_list."""

from __future__ import annotations

from core.tooling.schemas import (
    _CONSOLIDATION_BLOCKED_TOOLS,
    build_tool_list,
    build_unified_tool_list,
)


def _tool_names(tools: list[dict]) -> set[str]:
    return {t["name"] for t in tools}


class TestConsolidationToolFilter:
    """Verify delegation/messaging tools are excluded during consolidation."""

    def test_normal_trigger_excludes_submit_tasks(self):
        tools = build_unified_tool_list(
            include_supervisor_tools=True,
            include_create_skill=False,
            trigger="chat",
        )
        names = _tool_names(tools)
        assert "delegate_task" in names
        assert "submit_tasks" not in names
        assert "send_message" in names

    def test_background_trigger_includes_submit_tasks(self):
        tools = build_unified_tool_list(
            include_supervisor_tools=True,
            include_create_skill=False,
            trigger="background:manual",
        )
        names = _tool_names(tools)
        assert "submit_tasks" in names

    def test_consolidation_trigger_excludes_blocked_tools(self):
        tools = build_unified_tool_list(
            include_supervisor_tools=True,
            include_create_skill=False,
            trigger="consolidation:daily",
        )
        names = _tool_names(tools)
        for blocked in _CONSOLIDATION_BLOCKED_TOOLS:
            assert blocked not in names, f"{blocked} should be hidden during consolidation"

    def test_consolidation_weekly_also_excludes(self):
        tools = build_unified_tool_list(
            include_supervisor_tools=True,
            include_create_skill=False,
            trigger="consolidation:weekly",
        )
        names = _tool_names(tools)
        assert "delegate_task" not in names
        assert "submit_tasks" not in names

    def test_consolidation_keeps_memory_tools(self):
        tools = build_unified_tool_list(
            include_create_skill=False,
            trigger="consolidation:daily",
        )
        names = _tool_names(tools)
        assert "search_memory" in names
        assert "read_memory_file" in names
        assert "write_memory_file" in names
        assert "report_procedure_outcome" in names
        assert "report_knowledge_outcome" in names

    def test_empty_trigger_excludes_submit_tasks(self):
        tools = build_unified_tool_list(
            include_supervisor_tools=True,
            include_create_skill=False,
            trigger="",
        )
        names = _tool_names(tools)
        assert "delegate_task" in names
        assert "submit_tasks" not in names
        assert "send_message" in names

    def test_build_tool_list_consolidation_filter(self):
        """build_tool_list (Anthropic fallback) also respects consolidation."""
        tools = build_tool_list(
            include_supervisor_tools=True,
            include_submit_tasks=True,
            trigger="consolidation:daily",
        )
        names = _tool_names(tools)
        for blocked in _CONSOLIDATION_BLOCKED_TOOLS:
            assert blocked not in names, f"{blocked} should be hidden in build_tool_list"

    def test_build_tool_list_submit_tasks_requires_background_trigger(self):
        normal = build_tool_list(include_submit_tasks=True, trigger="chat")
        background = build_tool_list(include_submit_tasks=True, trigger="background:manual")
        assert "submit_tasks" not in _tool_names(normal)
        assert "submit_tasks" in _tool_names(background)

    def test_trust_skill_only_available_for_human_triggers(self):
        human = build_unified_tool_list(trigger="message:user")
        heartbeat = build_unified_tool_list(trigger="heartbeat")
        cron = build_unified_tool_list(trigger="cron:daily")
        consolidation = build_unified_tool_list(trigger="consolidation:daily")

        assert "trust_skill" in _tool_names(human)
        assert "trust_skill" not in _tool_names(heartbeat)
        assert "trust_skill" not in _tool_names(cron)
        assert "trust_skill" not in _tool_names(consolidation)

    def test_build_tool_list_hides_trust_skill_for_background_trigger(self):
        tools = build_tool_list(include_create_skill=True, trigger="background:manual")
        assert "create_skill" in _tool_names(tools)
        assert "trust_skill" not in _tool_names(tools)
