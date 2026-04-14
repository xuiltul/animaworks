"""Tests for compact tool filtering in build_unified_tool_list."""

from __future__ import annotations

from core.tooling.schemas import (
    _COMPACT_COMM_TOOLS,
    build_unified_tool_list,
)


def _tool_names(tools: list[dict]) -> set[str]:
    return {t["name"] for t in tools}


class TestCompactToolFilter:
    """Verify compact mode returns exactly the expected 12-tool subset."""

    def test_compact_returns_12_tools(self):
        tools = build_unified_tool_list(compact=True, include_create_skill=False)
        assert len(tools) == 12

    def test_compact_tool_names_match(self):
        tools = build_unified_tool_list(compact=True, include_create_skill=False)
        names = _tool_names(tools)
        assert names == _COMPACT_COMM_TOOLS

    def test_compact_includes_all_expected(self):
        tools = build_unified_tool_list(compact=True, include_create_skill=False)
        names = _tool_names(tools)
        for expected in [
            "Read", "Write", "Edit", "Bash", "Grep", "Glob",
            "search_memory", "read_memory_file", "write_memory_file",
            "send_message", "post_channel", "completion_gate",
        ]:
            assert expected in names, f"{expected} missing from compact tools"

    def test_compact_excludes_heavy_tools(self):
        tools = build_unified_tool_list(compact=True, include_create_skill=False)
        names = _tool_names(tools)
        for excluded in ["submit_tasks", "update_task", "session_todo", "create_skill"]:
            assert excluded not in names, f"{excluded} should be excluded in compact mode"

    def test_non_compact_returns_more_tools(self):
        compact = build_unified_tool_list(compact=True, include_create_skill=False)
        full = build_unified_tool_list(compact=False, include_create_skill=False)
        assert len(full) > len(compact)

    def test_compact_with_consolidation_trigger(self):
        """compact + consolidation: both filters apply."""
        tools = build_unified_tool_list(
            compact=True, include_create_skill=False, trigger="consolidation:daily",
        )
        names = _tool_names(tools)
        assert "send_message" not in names
        assert "post_channel" not in names

    def test_full_mode_unchanged_by_compact_false(self):
        """compact=False (default) should not alter existing behavior."""
        tools_default = build_unified_tool_list(include_create_skill=False)
        tools_explicit = build_unified_tool_list(compact=False, include_create_skill=False)
        assert _tool_names(tools_default) == _tool_names(tools_explicit)

    def test_compact_with_supervisor_still_filters(self):
        """Supervisor tools should be excluded by compact filter."""
        tools = build_unified_tool_list(
            compact=True, include_supervisor_tools=True, include_create_skill=False,
        )
        names = _tool_names(tools)
        assert "delegate_task" not in names
        assert "ping_subordinate" not in names
        assert len(names) == 12
