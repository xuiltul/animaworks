"""Tests for core.tooling.guide — dynamic tool guide generation."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.tooling.guide import (
    _build_summary_row,
    _extract_subcommand_names,
    _get_module_description,
    _get_tool_summary,
    _get_tool_summary_from_file,
    _import_file,
    build_tools_guide,
    load_tool_schemas,
)


# ── build_tools_guide ─────────────────────────────────────────


class TestBuildToolsGuide:
    def test_empty_returns_empty_string(self):
        result = build_tools_guide([], None)
        assert result == ""

    def test_empty_registry_and_no_personal_tools(self):
        result = build_tools_guide([], {})
        assert result == ""

    @patch("core.tooling.guide._get_tool_summary")
    def test_includes_core_tools(self, mock_summary):
        mock_summary.return_value = "| web_search | Search the web | search |"
        with patch.dict("core.tools.TOOL_MODULES", {"web_search": "core.tools.web_search"}):
            result = build_tools_guide(["web_search"])

        assert "web_search" in result
        assert "Search the web" in result

    @patch("core.tooling.guide._get_tool_summary")
    def test_skips_tools_not_in_tool_modules(self, mock_summary):
        with patch.dict("core.tools.TOOL_MODULES", {}, clear=True):
            result = build_tools_guide(["nonexistent"])

        mock_summary.assert_not_called()
        assert "外部ツール" in result

    @patch("core.tooling.guide._get_tool_summary_from_file")
    def test_includes_personal_tools(self, mock_summary_file):
        mock_summary_file.return_value = "| my_tool | My custom tool | action |"
        with patch.dict("core.tools.TOOL_MODULES", {}, clear=True):
            result = build_tools_guide([], {"my_tool": "/path/to/my_tool.py"})

        assert "my_tool" in result
        assert "My custom tool" in result

    @patch("core.tooling.guide._get_tool_summary")
    def test_includes_header_and_table_format(self, mock_summary):
        mock_summary.return_value = "| test | Test tool | cmd |"
        with patch.dict("core.tools.TOOL_MODULES", {"test": "core.tools.test"}):
            result = build_tools_guide(["test"])

        assert "外部ツール" in result
        assert "| ツール | 概要 | サブコマンド |" in result
        assert "|--------|------|------------|" in result

    @patch("core.tooling.guide._get_tool_summary")
    def test_skips_none_summaries(self, mock_summary):
        mock_summary.return_value = None
        with patch.dict("core.tools.TOOL_MODULES", {"test": "core.tools.test"}):
            result = build_tools_guide(["test"])

        assert "外部ツール" in result

    @patch("core.tooling.guide._get_tool_summary")
    def test_sorts_core_tools(self, mock_summary):
        calls = []

        def track_calls(tool_name, module_path):
            calls.append(tool_name)
            return f"| {tool_name} | desc | cmd |"

        mock_summary.side_effect = track_calls
        with patch.dict(
            "core.tools.TOOL_MODULES",
            {"charlie": "core.tools.c", "alpha": "core.tools.a", "bravo": "core.tools.b"},
            clear=True,
        ):
            build_tools_guide(["charlie", "alpha", "bravo"])

        assert calls == ["alpha", "bravo", "charlie"]


# ── load_tool_schemas ─────────────────────────────────────────


class TestLoadToolSchemas:
    @patch("core.tooling.schemas.load_external_schemas")
    def test_empty_registry_no_personal(self, mock_ext):
        mock_ext.return_value = []
        result = load_tool_schemas([], None)
        assert result == []

    @patch("core.tooling.schemas.load_external_schemas")
    def test_delegates_to_load_external_schemas(self, mock_ext):
        mock_ext.return_value = [
            {"name": "web_search", "description": "d", "parameters": {}}
        ]
        result = load_tool_schemas(["web_search"])
        mock_ext.assert_called_once_with(["web_search"])
        assert len(result) == 1

    @patch("core.tooling.schemas.load_personal_tool_schemas")
    @patch("core.tooling.schemas.load_external_schemas")
    def test_includes_personal_tool_schemas(self, mock_ext, mock_personal):
        mock_ext.return_value = []
        mock_personal.return_value = [
            {
                "name": "my_tool",
                "description": "My personal tool",
                "parameters": {"type": "object", "properties": {}},
            }
        ]

        result = load_tool_schemas([], {"my_tool": "/path/to/tool.py"})
        assert len(result) == 1
        assert result[0]["name"] == "my_tool"
        assert result[0]["parameters"] == {"type": "object", "properties": {}}

    @patch("core.tooling.schemas.load_personal_tool_schemas")
    @patch("core.tooling.schemas.load_external_schemas")
    def test_personal_tool_without_get_tool_schemas(self, mock_ext, mock_personal):
        mock_ext.return_value = []
        mock_personal.return_value = []

        result = load_tool_schemas([], {"my_tool": "/path/to/tool.py"})
        assert result == []

    @patch("core.tooling.schemas.load_personal_tool_schemas")
    @patch("core.tooling.schemas.load_external_schemas")
    def test_personal_tool_import_error(self, mock_ext, mock_personal):
        mock_ext.return_value = []
        mock_personal.return_value = []

        result = load_tool_schemas([], {"my_tool": "/path/to/tool.py"})
        assert result == []

    @patch("core.tooling.schemas.load_personal_tool_schemas")
    @patch("core.tooling.schemas.load_external_schemas")
    def test_personal_tool_uses_parameters_fallback(self, mock_ext, mock_personal):
        mock_ext.return_value = []
        mock_personal.return_value = [
            {
                "name": "my_tool",
                "description": "Tool",
                "parameters": {"type": "object"},
            }
        ]

        result = load_tool_schemas([], {"my_tool": "/path/to/tool.py"})
        assert result[0]["parameters"] == {"type": "object"}


# ── _build_summary_row ───────────────────────────────────────


class TestBuildSummaryRow:
    def test_generates_table_row(self):
        mod = MagicMock()
        mod.__doc__ = "Test tool for testing."
        mod.get_tool_schemas.return_value = [
            {"name": "test_cmd1"},
            {"name": "test_cmd2"},
        ]
        row = _build_summary_row("test", mod)
        assert row is not None
        assert row.startswith("| test |")
        assert "cmd1" in row
        assert "cmd2" in row

    def test_no_get_tool_schemas_returns_none(self):
        mod = MagicMock(spec=[])
        result = _build_summary_row("test", mod)
        assert result is None

    def test_empty_schemas_returns_none(self):
        mod = MagicMock()
        mod.get_tool_schemas.return_value = []
        result = _build_summary_row("test", mod)
        assert result is None


# ── _get_tool_summary ────────────────────────────────────────


class TestGetToolSummary:
    def test_success(self):
        mock_mod = MagicMock()
        mock_mod.__doc__ = "My tool."
        mock_mod.get_tool_schemas.return_value = [{"name": "test_action"}]
        import core.tooling.guide as _guide_mod
        with patch.object(_guide_mod.importlib, "import_module", return_value=mock_mod):
            result = _get_tool_summary("test", "core.tools.test")
        assert result is not None
        assert "test" in result
        assert "action" in result

    def test_import_error_returns_none(self):
        import core.tooling.guide as _guide_mod
        with patch.object(_guide_mod.importlib, "import_module", side_effect=ImportError("fail")):
            result = _get_tool_summary("test", "core.tools.test")
        assert result is None


# ── _get_tool_summary_from_file ──────────────────────────────


class TestGetToolSummaryFromFile:
    @patch("core.tooling.guide._build_summary_row")
    @patch("core.tooling.guide._import_file")
    def test_success(self, mock_import, mock_row):
        mock_mod = MagicMock()
        mock_import.return_value = mock_mod
        mock_row.return_value = "| my_tool | Personal tool | cmd |"

        result = _get_tool_summary_from_file("my_tool", "/path/to/tool.py")
        assert result == "| my_tool | Personal tool | cmd |"

    @patch("core.tooling.guide._import_file")
    def test_import_error_returns_none(self, mock_import):
        mock_import.side_effect = ImportError("fail")
        result = _get_tool_summary_from_file("my_tool", "/path/to/tool.py")
        assert result is None


# ── _extract_subcommand_names ────────────────────────────────


class TestExtractSubcommandNames:
    def test_strips_prefix(self):
        schemas = [{"name": "chatwork_send"}, {"name": "chatwork_rooms"}]
        result = _extract_subcommand_names("chatwork", schemas)
        assert result == ["send", "rooms"]

    def test_no_prefix(self):
        schemas = [{"name": "search"}]
        result = _extract_subcommand_names("web_search", schemas)
        assert result == ["search"]


# ── _get_module_description ──────────────────────────────────


class TestGetModuleDescription:
    def test_tool_description_attr(self):
        mod = MagicMock()
        mod.TOOL_DESCRIPTION = "Custom desc"
        assert _get_module_description(mod, "t") == "Custom desc"

    def test_docstring(self):
        mod = MagicMock(spec=[])
        mod.__doc__ = "First line.\nSecond line."
        assert _get_module_description(mod, "t") == "First line"

    def test_no_doc(self):
        mod = MagicMock(spec=[])
        mod.__doc__ = None
        assert _get_module_description(mod, "my_tool") == "my_tool"


# ── _import_file ──────────────────────────────────────────────


class TestImportFile:
    def test_raises_on_none_spec(self):
        with patch("importlib.util.spec_from_file_location", return_value=None):
            try:
                _import_file("test", "/nonexistent.py")
                assert False, "Should have raised ImportError"
            except ImportError as e:
                assert "Cannot load module" in str(e)

    def test_raises_on_none_loader(self):
        mock_spec = MagicMock()
        mock_spec.loader = None
        with patch("importlib.util.spec_from_file_location", return_value=mock_spec):
            try:
                _import_file("test", "/nonexistent.py")
                assert False, "Should have raised ImportError"
            except ImportError as e:
                assert "Cannot load module" in str(e)

    def test_success(self):
        mock_spec = MagicMock()
        mock_mod = MagicMock()
        with patch("importlib.util.spec_from_file_location", return_value=mock_spec), \
             patch("importlib.util.module_from_spec", return_value=mock_mod):
            result = _import_file("test", "/path/to/tool.py")
        assert result is mock_mod
        mock_spec.loader.exec_module.assert_called_once_with(mock_mod)
