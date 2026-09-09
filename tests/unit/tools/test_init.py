"""Tests for core/tools/__init__.py — tool registry and CLI dispatch."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.tools import (
    TOOL_MODULES,
    discover_core_tools,
    discover_common_tools,
    discover_personal_tools,
    cli_dispatch,
)


# ── TOOL_MODULES registry ─────────────────────────────────────────


class TestToolModules:
    """Tests for the TOOL_MODULES constant."""

    def test_tool_modules_is_dict(self):
        assert isinstance(TOOL_MODULES, dict)

    def test_contains_expected_tools(self):
        expected = {
            "web_search", "x_search", "chatwork", "slack", "gmail",
            "local_llm", "transcribe", "aws_collector", "github", "image_gen",
            "call_human", "google_calendar", "google_tasks", "google_sheets",
            "notion", "discord",
        }
        assert expected == set(TOOL_MODULES.keys())

    def test_module_paths_are_strings(self):
        for name, module_path in TOOL_MODULES.items():
            assert isinstance(module_path, str), f"{name} module path is not a string"
            assert module_path.startswith("core.tools."), (
                f"{name} module path does not start with 'core.tools.'"
            )


# ── discover_personal_tools ───────────────────────────────────────


class TestDiscoverPersonalTools:
    """Tests for discover_personal_tools()."""

    def test_returns_empty_if_no_tools_dir(self, tmp_path: Path):
        result = discover_personal_tools(tmp_path)
        assert result == {}

    def test_discovers_python_files(self, tmp_path: Path):
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "my_tool.py").write_text("cli_main = lambda: None")
        (tools_dir / "another.py").write_text("cli_main = lambda: None")

        result = discover_personal_tools(tmp_path)
        assert "my_tool" in result
        assert "another" in result
        assert result["my_tool"] == str(tools_dir / "my_tool.py")

    def test_skips_underscore_prefixed_files(self, tmp_path: Path):
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "__init__.py").write_text("")
        (tools_dir / "_private.py").write_text("")
        (tools_dir / "valid.py").write_text("")

        result = discover_personal_tools(tmp_path)
        assert "__init__" not in result
        assert "_private" not in result
        assert "valid" in result

    def test_skips_shadowed_core_tools(self, tmp_path: Path):
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        # "slack" is a core tool name
        (tools_dir / "slack.py").write_text("cli_main = lambda: None")
        (tools_dir / "custom_tool.py").write_text("cli_main = lambda: None")

        result = discover_personal_tools(tmp_path)
        assert "slack" not in result
        assert "custom_tool" in result

    def test_returns_sorted_results(self, tmp_path: Path):
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "zebra.py").write_text("")
        (tools_dir / "alpha.py").write_text("")

        result = discover_personal_tools(tmp_path)
        keys = list(result.keys())
        assert keys == sorted(keys)


# ── cli_dispatch ──────────────────────────────────────────────────


class TestCliDispatch:
    """Tests for cli_dispatch()."""

    def test_shows_help_with_no_args(self, capsys: pytest.CaptureFixture):
        with patch.object(sys, "argv", ["animaworks-tool"]):
            with pytest.raises(SystemExit) as exc_info:
                cli_dispatch()
            assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "Usage:" in out

    def test_shows_help_with_help_flag(self, capsys: pytest.CaptureFixture):
        with patch.object(sys, "argv", ["animaworks-tool", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                cli_dispatch()
            assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "Available tools:" in out

    def test_dispatches_core_tool(self):
        mock_module = MagicMock()
        mock_module.cli_main = MagicMock()

        with patch.object(sys, "argv", ["animaworks-tool", "web_search", "test"]):
            with patch("importlib.import_module", return_value=mock_module):
                cli_dispatch()
        mock_module.cli_main.assert_called_once_with(["test"])

    def test_errors_on_unknown_command(self, capsys: pytest.CaptureFixture):
        with patch.object(sys, "argv", ["animaworks-tool", "nonexistent_xyz"]):
            with pytest.raises(SystemExit) as exc_info:
                cli_dispatch()
            assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "Unknown command: nonexistent_xyz" in out
        assert "Available tools:" in out
        assert "Available CLI commands:" in out

    def test_core_tool_without_cli_main(self, capsys: pytest.CaptureFixture):
        mock_module = MagicMock(spec=[])  # no cli_main attribute

        with patch.object(sys, "argv", ["animaworks-tool", "web_search"]):
            with patch("importlib.import_module", return_value=mock_module):
                with pytest.raises(SystemExit) as exc_info:
                    cli_dispatch()
                assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "has no CLI interface" in out

    def test_dispatches_personal_tool(self, tmp_path: Path):
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        tool_file = tools_dir / "my_personal.py"
        tool_file.write_text(
            "called = False\ndef cli_main(argv):\n    global called; called = True\n"
        )

        with patch.dict("os.environ", {"ANIMAWORKS_ANIMA_DIR": str(tmp_path)}):
            with patch.object(sys, "argv", ["animaworks-tool", "my_personal"]):
                cli_dispatch()

    def test_personal_tool_without_cli_main(self, tmp_path: Path, capsys: pytest.CaptureFixture):
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        tool_file = tools_dir / "no_cli.py"
        tool_file.write_text("x = 1\n")

        with patch.dict("os.environ", {"ANIMAWORKS_ANIMA_DIR": str(tmp_path)}):
            with patch.object(sys, "argv", ["animaworks-tool", "no_cli"]):
                with pytest.raises(SystemExit) as exc_info:
                    cli_dispatch()
                assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "has no CLI interface" in out


# ── cli_dispatch fallback routing ────────────────────────────────


class TestCliDispatchFallback:
    """Tests for cli_dispatch() fallback routing to main CLI."""

    def test_forwards_main_cli_command(self):
        """animaworks-tool anima list → cli_main() called with rewritten argv."""
        captured_argv = []

        def capture_cli_main():
            captured_argv.extend(sys.argv)

        with patch.object(sys, "argv", ["animaworks-tool", "anima", "list"]):
            with patch("cli.cli_main", capture_cli_main):
                cli_dispatch()
        assert captured_argv == ["animaworks", "anima", "list"]

    def test_forwards_anima_subcommand(self):
        """animaworks-tool audit sakura → argv=["animaworks", "anima", "audit", "sakura"]."""
        captured_argv = []

        def capture_cli_main():
            captured_argv.extend(sys.argv)

        with patch.object(sys, "argv", ["animaworks-tool", "audit", "sakura"]):
            with patch("cli.cli_main", capture_cli_main):
                cli_dispatch()
        assert captured_argv == ["animaworks", "anima", "audit", "sakura"]

    def test_forwards_anima_subcommand_with_flags(self):
        """animaworks-tool audit sakura --days 3 → correct argv."""
        captured_argv = []

        def capture_cli_main():
            captured_argv.extend(sys.argv)

        with patch.object(sys, "argv", ["animaworks-tool", "audit", "sakura", "--days", "3"]):
            with patch("cli.cli_main", capture_cli_main):
                cli_dispatch()
        assert captured_argv == ["animaworks", "anima", "audit", "sakura", "--days", "3"]

    def test_forwards_start_command(self):
        """animaworks-tool start → argv=["animaworks", "start"]."""
        captured_argv = []

        def capture_cli_main():
            captured_argv.extend(sys.argv)

        with patch.object(sys, "argv", ["animaworks-tool", "start"]):
            with patch("cli.cli_main", capture_cli_main):
                cli_dispatch()
        assert captured_argv == ["animaworks", "start"]

    def test_tool_takes_priority_over_fallback(self):
        """Core tools are dispatched before fallback check."""
        mock_module = MagicMock()
        mock_module.cli_main = MagicMock()
        with patch.object(sys, "argv", ["animaworks-tool", "slack", "send"]):
            with patch("importlib.import_module", return_value=mock_module):
                cli_dispatch()
        mock_module.cli_main.assert_called_once_with(["send"])

    def test_unknown_command_shows_both_lists(self, capsys: pytest.CaptureFixture):
        """Truly unknown commands show both tool and CLI command lists."""
        with patch.object(sys, "argv", ["animaworks-tool", "totally_unknown_cmd"]):
            with pytest.raises(SystemExit) as exc_info:
                cli_dispatch()
            assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "Available tools:" in out
        assert "Available CLI commands:" in out


# ── discover_core_tools ──────────────────────────────────────────


class TestDiscoverCoreTools:
    """Tests for discover_core_tools()."""

    def test_returns_dict(self):
        result = discover_core_tools()
        assert isinstance(result, dict)

    def test_contains_expected_tools(self):
        result = discover_core_tools()
        expected = {"web_search", "slack", "gmail", "chatwork", "github"}
        for name in expected:
            assert name in result, f"Expected tool '{name}' not found in discover_core_tools()"

    def test_module_paths_start_with_core_tools(self):
        result = discover_core_tools()
        for name, module_path in result.items():
            assert module_path.startswith("core.tools."), (
                f"Tool '{name}' module path '{module_path}' does not start with 'core.tools.'"
            )

    def test_skips_underscore_prefixed_files(self):
        result = discover_core_tools()
        for name in result:
            assert not name.startswith("_"), (
                f"Tool '{name}' starts with underscore and should have been skipped"
            )

    def test_matches_tool_modules(self):
        """discover_core_tools() should produce the same result as TOOL_MODULES."""
        result = discover_core_tools()
        assert result == TOOL_MODULES


# ── discover_common_tools ────────────────────────────────────────


class TestDiscoverCommonTools:
    """Tests for discover_common_tools()."""

    def test_returns_empty_when_dir_missing(self, tmp_path: Path):
        # tmp_path exists but has no common_tools/ subdirectory
        result = discover_common_tools(data_dir=tmp_path)
        assert result == {}

    def test_discovers_python_files(self, tmp_path: Path):
        tools_dir = tmp_path / "common_tools"
        tools_dir.mkdir()
        (tools_dir / "shared_util.py").write_text("cli_main = lambda: None")
        (tools_dir / "helper.py").write_text("cli_main = lambda: None")

        result = discover_common_tools(data_dir=tmp_path)
        assert "shared_util" in result
        assert "helper" in result

    def test_skips_underscore_prefixed_files(self, tmp_path: Path):
        tools_dir = tmp_path / "common_tools"
        tools_dir.mkdir()
        (tools_dir / "__init__.py").write_text("")
        (tools_dir / "_internal.py").write_text("")
        (tools_dir / "valid_tool.py").write_text("")

        result = discover_common_tools(data_dir=tmp_path)
        assert "__init__" not in result
        assert "_internal" not in result
        assert "valid_tool" in result

    def test_skips_shadowed_core_tools(self, tmp_path: Path):
        tools_dir = tmp_path / "common_tools"
        tools_dir.mkdir()
        # "slack" is a core tool name in TOOL_MODULES
        (tools_dir / "slack.py").write_text("cli_main = lambda: None")
        (tools_dir / "unique_common.py").write_text("cli_main = lambda: None")

        result = discover_common_tools(data_dir=tmp_path)
        assert "slack" not in result
        assert "unique_common" in result

    def test_tool_paths_are_absolute(self, tmp_path: Path):
        tools_dir = tmp_path / "common_tools"
        tools_dir.mkdir()
        (tools_dir / "my_tool.py").write_text("")

        result = discover_common_tools(data_dir=tmp_path)
        for name, file_path in result.items():
            assert Path(file_path).is_absolute(), (
                f"Tool '{name}' path '{file_path}' is not absolute"
            )
