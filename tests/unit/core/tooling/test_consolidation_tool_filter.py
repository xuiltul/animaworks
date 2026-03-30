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

    def test_normal_trigger_includes_delegation_tools(self):
        tools = build_unified_tool_list(
            include_supervisor_tools=True,
            include_create_skill=False,
            trigger="chat",
        )
        names = _tool_names(tools)
        assert "delegate_task" in names
        assert "submit_tasks" in names
        assert "send_message" in names

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

    def test_empty_trigger_includes_all(self):
        tools = build_unified_tool_list(
            include_supervisor_tools=True,
            include_create_skill=False,
            trigger="",
        )
        names = _tool_names(tools)
        assert "delegate_task" in names
        assert "submit_tasks" in names
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
