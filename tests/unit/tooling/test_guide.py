"""Tests for core.tooling.guide — dynamic tool guide generation."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.tooling.guide import (
    _extract_guide,
    _guide_from_file,
    _guide_from_module_path,
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

    @patch("core.tooling.guide._guide_from_module_path")
    def test_includes_core_tools(self, mock_guide):
        mock_guide.return_value = "### web_search\nSearch the web."
        with patch.dict("core.tools.TOOL_MODULES", {"web_search": "core.tools.web_search"}):
            result = build_tools_guide(["web_search"])

        assert "web_search" in result
        assert "Search the web." in result

    @patch("core.tooling.guide._guide_from_module_path")
    def test_skips_tools_not_in_tool_modules(self, mock_guide):
        with patch.dict("core.tools.TOOL_MODULES", {}, clear=True):
            result = build_tools_guide(["nonexistent"])

        mock_guide.assert_not_called()
        assert "外部ツール" in result

    @patch("core.tooling.guide._guide_from_file")
    def test_includes_personal_tools(self, mock_guide_file):
        mock_guide_file.return_value = "### my_tool\nMy custom tool."
        with patch.dict("core.tools.TOOL_MODULES", {}, clear=True):
            result = build_tools_guide([], {"my_tool": "/path/to/my_tool.py"})

        assert "my_tool" in result
        assert "My custom tool." in result

    @patch("core.tooling.guide._guide_from_module_path")
    def test_includes_header_and_footer(self, mock_guide):
        mock_guide.return_value = "### test\nguide"
        with patch.dict("core.tools.TOOL_MODULES", {"test": "core.tools.test"}):
            result = build_tools_guide(["test"])

        assert "外部ツール" in result
        assert "注意事項" in result

    @patch("core.tooling.guide._guide_from_module_path")
    def test_skips_none_guides(self, mock_guide):
        mock_guide.return_value = None
        with patch.dict("core.tools.TOOL_MODULES", {"test": "core.tools.test"}):
            result = build_tools_guide(["test"])

        assert "外部ツール" in result

    @patch("core.tooling.guide._guide_from_module_path")
    def test_sorts_core_tools(self, mock_guide):
        calls = []

        def track_calls(tool_name, module_path):
            calls.append(tool_name)
            return f"### {tool_name}\nguide"

        mock_guide.side_effect = track_calls
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


# ── _extract_guide ────────────────────────────────────────────


class TestExtractGuide:
    def test_uses_get_cli_guide_if_available(self):
        mod = MagicMock()
        mod.get_cli_guide.return_value = "Custom CLI guide"
        result = _extract_guide("test", mod)
        assert result == "Custom CLI guide"

    def test_falls_back_to_auto_cli_guide(self):
        mod = MagicMock(spec=["get_tool_schemas"])
        mod.get_tool_schemas.return_value = [{"name": "t", "description": "d"}]

        with patch("core.tools._base.auto_cli_guide", return_value="Auto guide"):
            result = _extract_guide("test", mod)
        assert result == "Auto guide"

    def test_returns_none_without_methods(self):
        mod = MagicMock(spec=[])
        result = _extract_guide("test", mod)
        assert result is None


# ── _guide_from_module_path ───────────────────────────────────


class TestGuideFromModulePath:
    def test_success(self):
        mock_mod = MagicMock()
        mock_extract = MagicMock(return_value="Guide text")
        import core.tooling.guide as _guide_mod
        with patch.object(_guide_mod, "_extract_guide", mock_extract), \
             patch.object(_guide_mod.importlib, "import_module", return_value=mock_mod):
            result = _guide_from_module_path("test", "core.tools.test")
        assert result == "Guide text"
        mock_extract.assert_called_once_with("test", mock_mod)

    def test_import_error_returns_none(self):
        import core.tooling.guide as _guide_mod
        with patch.object(_guide_mod.importlib, "import_module", side_effect=ImportError("fail")):
            result = _guide_from_module_path("test", "core.tools.test")
        assert result is None


# ── _guide_from_file ──────────────────────────────────────────


class TestGuideFromFile:
    @patch("core.tooling.guide._extract_guide")
    @patch("core.tooling.guide._import_file")
    def test_success(self, mock_import, mock_extract):
        mock_mod = MagicMock()
        mock_import.return_value = mock_mod
        mock_extract.return_value = "Personal guide"

        result = _guide_from_file("my_tool", "/path/to/tool.py")
        assert result == "Personal guide"

    @patch("core.tooling.guide._import_file")
    def test_import_error_returns_none(self, mock_import):
        mock_import.side_effect = ImportError("fail")
        result = _guide_from_file("my_tool", "/path/to/tool.py")
        assert result is None


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
