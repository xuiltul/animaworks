"""Tests for guide.py background task annotations."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest


# ── _build_background_warning ─────────────────────────────


class TestBuildBackgroundWarning:
    def test_eligible_tool_gets_warning(self):
        """Tools with background-eligible subcommands get a warning."""
        from core.tooling.guide import _build_background_warning

        profile = {
            "3d": {"expected_seconds": 600, "background_eligible": True},
            "bustup": {"expected_seconds": 120, "background_eligible": True},
        }
        warning = _build_background_warning("image_gen", profile)
        assert warning  # Not empty
        assert "image_gen" in warning
        assert "submit" in warning
        assert "3d" in warning
        assert "bustup" in warning

    def test_no_eligible_returns_empty(self):
        """Tools with no eligible subcommands return empty string."""
        from core.tooling.guide import _build_background_warning

        profile = {
            "search": {"expected_seconds": 10, "background_eligible": False},
        }
        warning = _build_background_warning("web_search", profile)
        assert warning == ""

    def test_empty_profile_returns_empty(self):
        """Empty profile returns empty string."""
        from core.tooling.guide import _build_background_warning

        assert _build_background_warning("test", {}) == ""

    def test_time_format_minutes(self):
        """>=600 seconds displayed as minutes."""
        from core.tooling.guide import _build_background_warning

        profile = {"3d": {"expected_seconds": 600, "background_eligible": True}}
        warning = _build_background_warning("image_gen", profile)
        assert "10分" in warning

    def test_time_format_seconds(self):
        """<600 seconds displayed as seconds."""
        from core.tooling.guide import _build_background_warning

        profile = {"bustup": {"expected_seconds": 120, "background_eligible": True}}
        warning = _build_background_warning("image_gen", profile)
        assert "120秒" in warning

    def test_time_format_minutes_rounding(self):
        """Large second values are divided evenly to minutes."""
        from core.tooling.guide import _build_background_warning

        profile = {"slow_op": {"expected_seconds": 1200, "background_eligible": True}}
        warning = _build_background_warning("tool", profile)
        assert "20分" in warning

    def test_max_time_across_subcommands(self):
        """Time shown is the max across all eligible subcommands."""
        from core.tooling.guide import _build_background_warning

        profile = {
            "fast": {"expected_seconds": 60, "background_eligible": True},
            "slow": {"expected_seconds": 900, "background_eligible": True},
        }
        warning = _build_background_warning("tool", profile)
        # 900 >= 600, so shown as minutes: 900 // 60 = 15
        assert "15分" in warning

    def test_only_eligible_subcommands_listed(self):
        """Non-eligible subcommands are excluded from the listing."""
        from core.tooling.guide import _build_background_warning

        profile = {
            "slow": {"expected_seconds": 300, "background_eligible": True},
            "fast": {"expected_seconds": 5, "background_eligible": False},
        }
        warning = _build_background_warning("tool", profile)
        assert "slow" in warning
        assert "fast" not in warning

    def test_warning_includes_submit_instruction(self):
        """Warning includes animaworks-tool submit command format."""
        from core.tooling.guide import _build_background_warning

        profile = {"op": {"expected_seconds": 120, "background_eligible": True}}
        warning = _build_background_warning("my_tool", profile)
        assert "animaworks-tool submit my_tool" in warning

    def test_warning_includes_heartbeat_notification(self):
        """Warning mentions heartbeat notification for results."""
        from core.tooling.guide import _build_background_warning

        profile = {"op": {"expected_seconds": 120, "background_eligible": True}}
        warning = _build_background_warning("my_tool", profile)
        assert "heartbeat" in warning

    def test_mixed_eligible_and_non_eligible(self):
        """Profile with both eligible and non-eligible: only eligible appear."""
        from core.tooling.guide import _build_background_warning

        profile = {
            "bg_op": {"expected_seconds": 300, "background_eligible": True},
            "sync_op": {"expected_seconds": 10, "background_eligible": False},
            "another_bg": {"expected_seconds": 60, "background_eligible": True},
        }
        warning = _build_background_warning("tool", profile)
        assert "bg_op" in warning
        assert "another_bg" in warning
        assert "sync_op" not in warning

    def test_all_non_eligible_returns_empty(self):
        """Profile where all subcommands are non-eligible returns empty."""
        from core.tooling.guide import _build_background_warning

        profile = {
            "a": {"expected_seconds": 10, "background_eligible": False},
            "b": {"expected_seconds": 20, "background_eligible": False},
        }
        assert _build_background_warning("tool", profile) == ""

    def test_default_expected_seconds(self):
        """Missing expected_seconds defaults to 60."""
        from core.tooling.guide import _build_background_warning

        profile = {"op": {"background_eligible": True}}
        warning = _build_background_warning("tool", profile)
        # 60 < 600, so displayed as seconds
        assert "60秒" in warning


# ── build_tools_guide integration ─────────────────────────


class TestBuildToolsGuideIntegration:
    """Integration tests for build_tools_guide with background annotations."""

    def _make_mock_module(
        self,
        tool_name: str,
        schemas: list[dict] | None = None,
        execution_profile: dict | None = None,
    ):
        """Create a mock module with optional schemas and execution profile."""
        from unittest.mock import MagicMock

        mod = MagicMock()
        if schemas is not None:
            mod.get_tool_schemas.return_value = schemas
            del mod.get_cli_guide  # Remove so auto_cli_guide path is used
        else:
            mod.get_cli_guide.return_value = f"### {tool_name}\n`animaworks-tool {tool_name} ...`"
            del mod.get_tool_schemas

        if execution_profile is not None:
            mod.EXECUTION_PROFILE = execution_profile
        else:
            del mod.EXECUTION_PROFILE

        return mod

    def test_guide_contains_warning_for_eligible_tools(self):
        """Guide output contains warning for tools with eligible subcommands."""
        from unittest.mock import patch

        with patch("core.tools.TOOL_MODULES", {"image_gen": "core.tools.image_gen"}), \
             patch("core.tools._base.load_execution_profiles", return_value={
                 "image_gen": {"3d": {"expected_seconds": 600, "background_eligible": True}},
             }), \
             patch("core.tooling.guide._guide_from_module_path", return_value="### image_gen\nUsage..."):
            from core.tooling.guide import build_tools_guide

            guide = build_tools_guide(["image_gen"])

        assert "image_gen" in guide
        assert "submit" in guide

    def test_guide_contains_submit_instruction_in_notes(self):
        """Guide contains submit instruction in the notes section when bg tools exist."""
        from unittest.mock import patch

        with patch("core.tools.TOOL_MODULES", {"image_gen": "core.tools.image_gen"}), \
             patch("core.tools._base.load_execution_profiles", return_value={
                 "image_gen": {"3d": {"expected_seconds": 600, "background_eligible": True}},
             }), \
             patch("core.tooling.guide._guide_from_module_path", return_value="### image_gen\nUsage..."):
            from core.tooling.guide import build_tools_guide

            guide = build_tools_guide(["image_gen"])

        assert "animaworks-tool submit" in guide
        assert "注意事項" in guide

    def test_guide_no_submit_instruction_without_bg_tools(self):
        """Guide does NOT contain submit instruction when no background tools exist."""
        from unittest.mock import patch

        with patch("core.tools.TOOL_MODULES", {"web_search": "core.tools.web_search"}), \
             patch("core.tools._base.load_execution_profiles", return_value={
                 "web_search": {"search": {"expected_seconds": 10, "background_eligible": False}},
             }), \
             patch("core.tooling.guide._guide_from_module_path", return_value="### web_search\nUsage..."):
            from core.tooling.guide import build_tools_guide

            guide = build_tools_guide(["web_search"])

        assert "注意事項" in guide
        # The background-specific submit line should NOT be present
        assert "animaworks-tool submit" not in guide

    def test_guide_no_tools(self):
        """Empty tool registry returns empty string."""
        from core.tooling.guide import build_tools_guide

        guide = build_tools_guide([])
        assert guide == ""

    def test_guide_empty_profiles(self):
        """Tools with no execution profiles don't produce warnings."""
        from unittest.mock import patch

        with patch("core.tools.TOOL_MODULES", {"simple": "core.tools.simple"}), \
             patch("core.tools._base.load_execution_profiles", return_value={}), \
             patch("core.tooling.guide._guide_from_module_path", return_value="### simple\nUsage..."):
            from core.tooling.guide import build_tools_guide

            guide = build_tools_guide(["simple"])

        # Should have the guide text but no background warning
        assert "simple" in guide
        assert "animaworks-tool submit" not in guide
