"""Tests for ToolHandler dispatch dict pattern.

Verifies:
1. All built-in tool names are registered in self._dispatch
2. Dict lookup dispatches correctly to handler methods
3. Unknown tools fall through to external dispatch
4. _ACTIVITY_TYPE_MAP covers the expected tools
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.tooling.handler import ToolHandler


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "test-anima"
    d.mkdir(parents=True)
    (d / "permissions.md").write_text("", encoding="utf-8")
    return d


@pytest.fixture
def handler(anima_dir: Path) -> ToolHandler:
    memory = MagicMock()
    memory.read_permissions.return_value = ""
    memory.search_memory_text.return_value = []
    return ToolHandler(
        anima_dir=anima_dir,
        memory=memory,
        messenger=None,
        tool_registry=[],
    )


# ── Dispatch dict completeness ────────────────────────────────


# All 21 built-in tool names that must be in the dispatch dict
EXPECTED_BUILTIN_TOOLS = frozenset({
    "search_memory",
    "read_memory_file",
    "write_memory_file",
    "send_message",
    "post_channel",
    "read_channel",
    "read_dm_history",
    "read_file",
    "write_file",
    "edit_file",
    "execute_command",
    "search_code",
    "list_directory",
    "call_human",
    "create_anima",
    "refresh_tools",
    "share_tool",
    "report_procedure_outcome",
    "report_knowledge_outcome",
    "add_task",
    "update_task",
    "list_tasks",
})


class TestDispatchDictCompleteness:
    """Verify dispatch dict contains all built-in tools."""

    def test_all_builtin_tools_registered(self, handler: ToolHandler):
        """Every expected built-in tool name must be a key in _dispatch."""
        registered = set(handler._dispatch.keys())
        missing = EXPECTED_BUILTIN_TOOLS - registered
        assert missing == set(), f"Missing tools in dispatch dict: {missing}"

    def test_no_extra_tools_in_dispatch(self, handler: ToolHandler):
        """Dispatch dict should only contain known built-in tools."""
        registered = set(handler._dispatch.keys())
        extra = registered - EXPECTED_BUILTIN_TOOLS
        assert extra == set(), f"Unexpected tools in dispatch dict: {extra}"

    def test_dispatch_count(self, handler: ToolHandler):
        """Dispatch dict should have exactly 22 entries."""
        assert len(handler._dispatch) == 22

    def test_all_dispatch_values_are_callable(self, handler: ToolHandler):
        """Every value in the dispatch dict must be callable."""
        for name, func in handler._dispatch.items():
            assert callable(func), f"Dispatch entry '{name}' is not callable: {func}"


# ── Dispatch routing ──────────────────────────────────────────


class TestDispatchRouting:
    """Verify dispatch dict correctly routes to handler methods."""

    def test_known_tool_dispatches_via_dict(self, handler: ToolHandler):
        """A known tool name should be routed through the dispatch dict."""
        mock_handler = MagicMock(return_value="mock result")
        handler._dispatch["search_memory"] = mock_handler

        result = handler.handle("search_memory", {"query": "test"})

        mock_handler.assert_called_once_with({"query": "test"})
        assert result == "mock result"

    def test_unknown_tool_falls_to_external(self, handler: ToolHandler):
        """An unknown tool name should fall through to external dispatch."""
        handler._external = MagicMock()
        handler._external.dispatch.return_value = "external result"

        result = handler.handle("some_unknown_tool", {"arg": "val"})

        handler._external.dispatch.assert_called_once()
        assert result == "external result"

    def test_unknown_tool_returns_unknown_when_external_returns_none(
        self, handler: ToolHandler,
    ):
        """When external dispatch returns None, should return 'Unknown tool'."""
        handler._external = MagicMock()
        handler._external.dispatch.return_value = None

        result = handler.handle("nonexistent_tool", {})

        assert "Unknown tool" in result

    def test_background_eligible_tool_submitted(self, handler: ToolHandler):
        """Background-eligible external tools should be submitted to background manager."""
        bg_manager = MagicMock()
        bg_manager.is_eligible.return_value = True
        bg_manager.submit.return_value = "task-123"
        handler._background_manager = bg_manager

        result = handler.handle("slow_external_tool", {"data": "value"})

        bg_manager.submit.assert_called_once()
        assert "task-123" in result
        assert "background" in result.lower()

    def test_dispatch_exception_caught(self, handler: ToolHandler):
        """Exceptions from dispatch handlers should be caught by top-level try."""
        handler._dispatch["search_memory"] = MagicMock(
            side_effect=RuntimeError("handler crashed"),
        )

        result = handler.handle("search_memory", {"query": "test"})

        assert "Tool execution failed" in result
        assert "search_memory" in result
        assert "handler crashed" in result


# ── Activity type map ─────────────────────────────────────────


class TestActivityTypeMap:
    """Verify _ACTIVITY_TYPE_MAP covers the expected special-case tools."""

    def test_expected_entries(self):
        """_ACTIVITY_TYPE_MAP should contain the 4 special-case tools."""
        expected = {"post_channel", "read_channel", "read_dm_history", "call_human"}
        assert set(ToolHandler._ACTIVITY_TYPE_MAP.keys()) == expected

    def test_post_channel_maps_to_channel_post(self):
        assert ToolHandler._ACTIVITY_TYPE_MAP["post_channel"] == "channel_post"

    def test_read_channel_maps_to_channel_read(self):
        assert ToolHandler._ACTIVITY_TYPE_MAP["read_channel"] == "channel_read"

    def test_read_dm_history_maps_to_channel_read(self):
        assert ToolHandler._ACTIVITY_TYPE_MAP["read_dm_history"] == "channel_read"

    def test_call_human_maps_to_human_notify(self):
        assert ToolHandler._ACTIVITY_TYPE_MAP["call_human"] == "human_notify"
