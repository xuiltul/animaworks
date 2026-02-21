"""Tests for guide.py compact summary table and background annotations."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ── _get_module_description ──────────────────────────────────


class TestGetModuleDescription:
    def test_explicit_tool_description(self):
        """TOOL_DESCRIPTION attribute takes priority."""
        from core.tooling.guide import _get_module_description

        mod = MagicMock()
        mod.TOOL_DESCRIPTION = "Custom description"
        assert _get_module_description(mod, "test") == "Custom description"

    def test_docstring_first_line(self):
        """First line of module docstring is used."""
        from core.tooling.guide import _get_module_description

        mod = MagicMock(spec=[])
        mod.__doc__ = "Short first line.\n\nMore details here."
        assert _get_module_description(mod, "test") == "Short first line"

    def test_long_docstring_truncated(self):
        """Long first line is truncated to 60 chars."""
        from core.tooling.guide import _get_module_description

        mod = MagicMock(spec=[])
        mod.__doc__ = "A" * 80 + "\n\nMore details."
        desc = _get_module_description(mod, "test")
        assert len(desc) <= 60
        assert desc.endswith("...")

    def test_no_docstring_returns_tool_name(self):
        """Falls back to tool name when no docstring."""
        from core.tooling.guide import _get_module_description

        mod = MagicMock(spec=[])
        mod.__doc__ = None
        assert _get_module_description(mod, "my_tool") == "my_tool"


# ── _extract_subcommand_names ────────────────────────────────


class TestExtractSubcommandNames:
    def test_strips_tool_prefix(self):
        """Tool name prefix is removed from subcommand names."""
        from core.tooling.guide import _extract_subcommand_names

        schemas = [
            {"name": "chatwork_send"},
            {"name": "chatwork_rooms"},
        ]
        result = _extract_subcommand_names("chatwork", schemas)
        assert result == ["send", "rooms"]

    def test_no_prefix(self):
        """Subcommand names without tool prefix are kept as-is."""
        from core.tooling.guide import _extract_subcommand_names

        schemas = [{"name": "search"}, {"name": "fetch"}]
        result = _extract_subcommand_names("web_search", schemas)
        assert result == ["search", "fetch"]

    def test_function_fallback(self):
        """Falls back to function.name if name is missing."""
        from core.tooling.guide import _extract_subcommand_names

        schemas = [{"function": {"name": "web_search_query"}}]
        result = _extract_subcommand_names("web_search", schemas)
        assert result == ["query"]


# ── _build_summary_row ───────────────────────────────────────


class TestBuildSummaryRow:
    def test_generates_table_row(self):
        """Generates a Markdown table row with name, description, subcommands."""
        from core.tooling.guide import _build_summary_row

        mod = MagicMock()
        mod.__doc__ = "Test tool for testing."
        mod.get_tool_schemas.return_value = [
            {"name": "test_action1"},
            {"name": "test_action2"},
        ]
        row = _build_summary_row("test", mod)
        assert row is not None
        assert row.startswith("| test |")
        assert "action1" in row
        assert "action2" in row

    def test_no_schemas_returns_none(self):
        """Returns None if no schemas."""
        from core.tooling.guide import _build_summary_row

        mod = MagicMock(spec=[])
        assert _build_summary_row("test", mod) is None

    def test_truncates_many_subcommands(self):
        """More than 6 subcommands are truncated with '...'."""
        from core.tooling.guide import _build_summary_row

        mod = MagicMock()
        mod.__doc__ = "Test tool."
        mod.get_tool_schemas.return_value = [
            {"name": f"test_cmd{i}"} for i in range(10)
        ]
        row = _build_summary_row("test", mod)
        assert row is not None
        assert "..." in row


# ── build_tools_guide integration ─────────────────────────


class TestBuildToolsGuideIntegration:
    """Integration tests for build_tools_guide with compact summary format."""

    def test_guide_contains_warning_for_eligible_tools(self):
        """Guide output contains background warning for tools with eligible subcommands."""
        with patch("core.tools.TOOL_MODULES", {"image_gen": "core.tools.image_gen"}), \
             patch("core.tools._base.load_execution_profiles", return_value={
                 "image_gen": {"3d": {"expected_seconds": 600, "background_eligible": True}},
             }), \
             patch("core.tooling.guide._get_tool_summary", return_value="| image_gen | Image gen | 3d, bustup |"):
            from core.tooling.guide import build_tools_guide

            guide = build_tools_guide(["image_gen"])

        assert "image_gen" in guide
        assert "submit" in guide

    def test_guide_contains_submit_instruction_when_bg_tools(self):
        """Guide contains submit instruction when background tools exist."""
        with patch("core.tools.TOOL_MODULES", {"image_gen": "core.tools.image_gen"}), \
             patch("core.tools._base.load_execution_profiles", return_value={
                 "image_gen": {"3d": {"expected_seconds": 600, "background_eligible": True}},
             }), \
             patch("core.tooling.guide._get_tool_summary", return_value="| image_gen | Image gen | 3d |"):
            from core.tooling.guide import build_tools_guide

            guide = build_tools_guide(["image_gen"])

        assert "animaworks-tool submit" in guide

    def test_guide_no_submit_instruction_without_bg_tools(self):
        """Guide does NOT contain submit instruction when no background tools exist."""
        with patch("core.tools.TOOL_MODULES", {"web_search": "core.tools.web_search"}), \
             patch("core.tools._base.load_execution_profiles", return_value={
                 "web_search": {"search": {"expected_seconds": 10, "background_eligible": False}},
             }), \
             patch("core.tooling.guide._get_tool_summary", return_value="| web_search | Web search | search |"):
            from core.tooling.guide import build_tools_guide

            guide = build_tools_guide(["web_search"])

        # The background-specific submit line should NOT be present
        assert "animaworks-tool submit" not in guide

    def test_guide_no_tools(self):
        """Empty tool registry returns empty string."""
        from core.tooling.guide import build_tools_guide

        guide = build_tools_guide([])
        assert guide == ""

    def test_guide_empty_profiles(self):
        """Tools with no execution profiles don't produce warnings."""
        with patch("core.tools.TOOL_MODULES", {"simple": "core.tools.simple"}), \
             patch("core.tools._base.load_execution_profiles", return_value={}), \
             patch("core.tooling.guide._get_tool_summary", return_value="| simple | Simple tool | action |"):
            from core.tooling.guide import build_tools_guide

            guide = build_tools_guide(["simple"])

        # Should have the guide text but no background warning
        assert "simple" in guide
        assert "animaworks-tool submit" not in guide

    def test_guide_has_table_format(self):
        """Guide output uses markdown table format."""
        with patch("core.tools.TOOL_MODULES", {"my_tool": "core.tools.my_tool"}), \
             patch("core.tools._base.load_execution_profiles", return_value={}), \
             patch("core.tooling.guide._get_tool_summary", return_value="| my_tool | My tool | cmd1, cmd2 |"):
            from core.tooling.guide import build_tools_guide

            guide = build_tools_guide(["my_tool"])

        assert "| ツール | 概要 | サブコマンド |" in guide
        assert "|--------|------|------------|" in guide
        assert "| my_tool |" in guide
