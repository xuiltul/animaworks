"""Tests for core.tooling.dispatch — ExternalToolDispatcher."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from core.tooling.dispatch import ExternalToolDispatcher


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def empty_dispatcher() -> ExternalToolDispatcher:
    return ExternalToolDispatcher(tool_registry=[])


@pytest.fixture
def dispatcher_with_registry() -> ExternalToolDispatcher:
    return ExternalToolDispatcher(tool_registry=["web_search"])


# ── dispatch() ────────────────────────────────────────────────


class TestDispatch:
    def test_returns_none_when_no_match(self, empty_dispatcher: ExternalToolDispatcher):
        result = empty_dispatcher.dispatch("unknown", {})
        assert result is None

    def test_delegates_to_registry_first(self):
        d = ExternalToolDispatcher(tool_registry=["web_search"])
        with patch.object(d, "_dispatch_from_registry", return_value="core result") as mock_reg:
            result = d.dispatch("web_search", {"query": "test"})
        assert result == "core result"
        mock_reg.assert_called_once_with("web_search", {"query": "test"})

    def test_falls_through_to_files(self):
        d = ExternalToolDispatcher(
            tool_registry=[],
            personal_tools={"my_tool": "/path/to/tool.py"},
        )
        with patch.object(d, "_dispatch_from_registry", return_value=None), \
             patch.object(d, "_dispatch_from_files", return_value="file result"):
            result = d.dispatch("my_fn", {})
        assert result == "file result"

    def test_returns_none_when_both_miss(self):
        d = ExternalToolDispatcher(
            tool_registry=[],
            personal_tools={"my_tool": "/path/to/tool.py"},
        )
        with patch.object(d, "_dispatch_from_registry", return_value=None), \
             patch.object(d, "_dispatch_from_files", return_value=None):
            result = d.dispatch("unknown", {})
        assert result is None

    def test_does_not_call_files_when_registry_matches(self):
        d = ExternalToolDispatcher(
            tool_registry=["web_search"],
            personal_tools={"my_tool": "/path/to/tool.py"},
        )
        with patch.object(d, "_dispatch_from_registry", return_value="core result"), \
             patch.object(d, "_dispatch_from_files") as mock_files:
            d.dispatch("web_search", {})
        mock_files.assert_not_called()


# ── _dispatch_from_registry() ─────────────────────────────────


class TestDispatchFromRegistry:
    def test_empty_registry_returns_none(self, empty_dispatcher: ExternalToolDispatcher):
        result = empty_dispatcher._dispatch_from_registry("web_search", {})
        assert result is None

    def test_tool_not_in_registry(self):
        d = ExternalToolDispatcher(tool_registry=["web_search"])
        with patch("importlib.import_module") as mock_import:
            mock_import.return_value = MagicMock()
            with patch("core.tools.TOOL_MODULES", {"slack": "core.tools.slack"}):
                result = d._dispatch_from_registry("slack_send", {})
        assert result is None

    def test_dispatches_matching_schema_via_dispatch(self):
        mock_mod = MagicMock()
        mock_mod.get_tool_schemas.return_value = [
            {"name": "web_search", "description": "Search"},
        ]
        mock_mod.dispatch.return_value = "search result"

        d = ExternalToolDispatcher(tool_registry=["web_search"])
        with patch("core.tools.TOOL_MODULES", {"web_search": "core.tools.web_search"}), \
             patch("importlib.import_module", return_value=mock_mod):
            result = d._dispatch_from_registry("web_search", {"query": "test"})

        assert result == "search result"

    def test_dispatches_matching_schema_via_function_name(self):
        mock_mod = MagicMock(spec=["get_tool_schemas", "web_search"])
        mock_mod.get_tool_schemas.return_value = [{"name": "web_search"}]
        mock_mod.web_search.return_value = "func result"

        d = ExternalToolDispatcher(tool_registry=["web_search"])
        with patch("core.tools.TOOL_MODULES", {"web_search": "core.tools.web_search"}), \
             patch("importlib.import_module", return_value=mock_mod):
            result = d._dispatch_from_registry("web_search", {"query": "test"})

        assert result == "func result"

    def test_handles_module_without_get_tool_schemas(self):
        mock_mod = MagicMock(spec=[])  # No get_tool_schemas

        d = ExternalToolDispatcher(tool_registry=["web_search"])
        with patch("core.tools.TOOL_MODULES", {"web_search": "core.tools.web_search"}), \
             patch("importlib.import_module", return_value=mock_mod):
            result = d._dispatch_from_registry("web_search", {})

        assert result is None

    def test_schema_name_not_in_module_schemas(self):
        mock_mod = MagicMock()
        mock_mod.get_tool_schemas.return_value = [{"name": "other_tool"}]

        d = ExternalToolDispatcher(tool_registry=["web_search"])
        with patch("core.tools.TOOL_MODULES", {"web_search": "core.tools.web_search"}), \
             patch("importlib.import_module", return_value=mock_mod):
            result = d._dispatch_from_registry("web_search", {})

        assert result is None

    def test_handles_execution_error(self):
        mock_mod = MagicMock()
        mock_mod.get_tool_schemas.return_value = [{"name": "web_search"}]
        mock_mod.dispatch.side_effect = RuntimeError("boom")

        d = ExternalToolDispatcher(tool_registry=["web_search"])
        with patch("core.tools.TOOL_MODULES", {"web_search": "core.tools.web_search"}), \
             patch("importlib.import_module", return_value=mock_mod):
            result = d._dispatch_from_registry("web_search", {})

        assert "Error executing" in result
        assert "boom" in result

    def test_handles_import_error(self):
        d = ExternalToolDispatcher(tool_registry=["web_search"])
        with patch("core.tools.TOOL_MODULES", {"web_search": "core.tools.web_search"}), \
             patch(
                 "importlib.import_module",
                 side_effect=ImportError("no module"),
             ):
            result = d._dispatch_from_registry("web_search", {})

        assert "Error executing" in result
        assert "no module" in result


# ── _dispatch_from_files() ────────────────────────────────────


class TestDispatchFromFiles:
    def test_empty_personal_tools(self, empty_dispatcher: ExternalToolDispatcher):
        result = empty_dispatcher._dispatch_from_files("my_fn", {})
        assert result is None

    def test_dispatches_via_module_dispatch(self):
        mock_spec = MagicMock()
        mock_mod = MagicMock()
        mock_mod.get_tool_schemas.return_value = [{"name": "my_fn"}]
        mock_mod.dispatch.return_value = "dispatched result"

        d = ExternalToolDispatcher(
            tool_registry=[],
            personal_tools={"my_tool": "/path/to/tool.py"},
        )
        with patch("importlib.util.spec_from_file_location", return_value=mock_spec), \
             patch("importlib.util.module_from_spec", return_value=mock_mod):
            result = d._dispatch_from_files("my_fn", {"arg": "val"})

        assert result == "dispatched result"

    def test_dispatches_via_function_name(self):
        mock_spec = MagicMock()
        mock_mod = MagicMock(spec=["get_tool_schemas", "my_fn"])
        mock_mod.get_tool_schemas.return_value = [{"name": "my_fn"}]
        mock_mod.my_fn.return_value = "func result"

        d = ExternalToolDispatcher(
            tool_registry=[],
            personal_tools={"my_tool": "/path/to/tool.py"},
        )
        with patch("importlib.util.spec_from_file_location", return_value=mock_spec), \
             patch("importlib.util.module_from_spec", return_value=mock_mod):
            result = d._dispatch_from_files("my_fn", {"arg": "val"})

        assert result == "func result"

    def test_spec_is_none_skips(self):
        d = ExternalToolDispatcher(
            tool_registry=[],
            personal_tools={"my_tool": "/path/to/tool.py"},
        )
        with patch("importlib.util.spec_from_file_location", return_value=None):
            result = d._dispatch_from_files("my_fn", {})

        assert result is None

    def test_loader_is_none_skips(self):
        mock_spec = MagicMock()
        mock_spec.loader = None

        d = ExternalToolDispatcher(
            tool_registry=[],
            personal_tools={"my_tool": "/path/to/tool.py"},
        )
        with patch("importlib.util.spec_from_file_location", return_value=mock_spec):
            result = d._dispatch_from_files("my_fn", {})

        assert result is None

    def test_schema_name_not_matching(self):
        mock_spec = MagicMock()
        mock_mod = MagicMock()
        mock_mod.get_tool_schemas.return_value = [{"name": "other_fn"}]

        d = ExternalToolDispatcher(
            tool_registry=[],
            personal_tools={"my_tool": "/path/to/tool.py"},
        )
        with patch("importlib.util.spec_from_file_location", return_value=mock_spec), \
             patch("importlib.util.module_from_spec", return_value=mock_mod):
            result = d._dispatch_from_files("my_fn", {})

        assert result is None

    def test_module_without_get_tool_schemas(self):
        mock_spec = MagicMock()
        mock_mod = MagicMock(spec=[])  # No get_tool_schemas

        d = ExternalToolDispatcher(
            tool_registry=[],
            personal_tools={"my_tool": "/path/to/tool.py"},
        )
        with patch("importlib.util.spec_from_file_location", return_value=mock_spec), \
             patch("importlib.util.module_from_spec", return_value=mock_mod):
            result = d._dispatch_from_files("my_fn", {})

        assert result is None

    def test_execution_error(self):
        mock_spec = MagicMock()
        mock_mod = MagicMock()
        mock_mod.get_tool_schemas.return_value = [{"name": "my_fn"}]
        mock_mod.dispatch.side_effect = RuntimeError("fail")

        d = ExternalToolDispatcher(
            tool_registry=[],
            personal_tools={"my_tool": "/path/to/tool.py"},
        )
        with patch("importlib.util.spec_from_file_location", return_value=mock_spec), \
             patch("importlib.util.module_from_spec", return_value=mock_mod):
            result = d._dispatch_from_files("my_fn", {})

        assert "Error executing" in result
        assert "fail" in result


# ── _call_module() ────────────────────────────────────────────


class TestCallModule:
    def test_calls_dispatch_if_available(self):
        mod = MagicMock()
        mod.dispatch.return_value = "dispatched"

        result = ExternalToolDispatcher._call_module(mod, "my_tool", {"a": 1})

        assert result == "dispatched"
        mod.dispatch.assert_called_once_with("my_tool", {"a": 1})

    def test_falls_to_getattr_when_no_dispatch(self):
        mod = MagicMock(spec=["my_tool"])
        mod.my_tool.return_value = "func result"

        result = ExternalToolDispatcher._call_module(mod, "my_tool", {"x": "y"})

        assert result == "func result"
        mod.my_tool.assert_called_once_with(x="y")

    def test_returns_error_when_neither_dispatch_nor_function(self):
        mod = MagicMock(spec=[])  # No dispatch, no matching function

        result = ExternalToolDispatcher._call_module(mod, "missing_fn", {})

        assert "Error" in result
        assert "no handler" in result
        assert "missing_fn" in result

    def test_serializes_dict_result_to_json(self):
        mod = MagicMock()
        mod.dispatch.return_value = {"key": "value", "count": 42}

        result = ExternalToolDispatcher._call_module(mod, "my_tool", {})

        parsed = json.loads(result)
        assert parsed == {"key": "value", "count": 42}

    def test_serializes_list_result_to_json(self):
        mod = MagicMock()
        mod.dispatch.return_value = [1, 2, 3]

        result = ExternalToolDispatcher._call_module(mod, "my_tool", {})

        parsed = json.loads(result)
        assert parsed == [1, 2, 3]

    def test_returns_no_output_for_none(self):
        mod = MagicMock()
        mod.dispatch.return_value = None

        result = ExternalToolDispatcher._call_module(mod, "my_tool", {})

        assert result == "(no output)"

    def test_converts_string_result(self):
        mod = MagicMock()
        mod.dispatch.return_value = "plain text"

        result = ExternalToolDispatcher._call_module(mod, "my_tool", {})

        assert result == "plain text"

    def test_converts_non_string_non_collection_to_str(self):
        mod = MagicMock()
        mod.dispatch.return_value = 12345

        result = ExternalToolDispatcher._call_module(mod, "my_tool", {})

        assert result == "12345"

    def test_handles_exception_in_dispatch(self):
        mod = MagicMock()
        mod.dispatch.side_effect = ValueError("bad input")

        result = ExternalToolDispatcher._call_module(mod, "my_tool", {})

        assert "Error executing" in result
        assert "bad input" in result

    def test_handles_exception_in_function_call(self):
        mod = MagicMock(spec=["my_tool"])
        mod.my_tool.side_effect = TypeError("wrong args")

        result = ExternalToolDispatcher._call_module(mod, "my_tool", {})

        assert "Error executing" in result
        assert "wrong args" in result

    def test_dispatch_preferred_over_function_name(self):
        """When both dispatch() and a matching function exist, dispatch() wins."""
        mod = MagicMock()
        mod.dispatch.return_value = "from dispatch"
        mod.my_tool = MagicMock(return_value="from function")

        result = ExternalToolDispatcher._call_module(mod, "my_tool", {})

        assert result == "from dispatch"
        mod.dispatch.assert_called_once()
        mod.my_tool.assert_not_called()


# ── update_personal_tools() ──────────────────────────────────


class TestUpdatePersonalTools:
    def test_replaces_personal_tools_mapping(self):
        d = ExternalToolDispatcher(
            tool_registry=[],
            personal_tools={"old_tool": "/old/path.py"},
        )
        new_mapping = {"new_tool": "/new/path.py", "another": "/another.py"}
        d.update_personal_tools(new_mapping)

        assert d._personal_tools == new_mapping

    def test_replaces_with_empty_mapping(self):
        d = ExternalToolDispatcher(
            tool_registry=[],
            personal_tools={"tool": "/path.py"},
        )
        d.update_personal_tools({})

        assert d._personal_tools == {}


# ── registry property ─────────────────────────────────────────


class TestRegistryProperty:
    def test_returns_registry_list(self):
        d = ExternalToolDispatcher(tool_registry=["web_search", "slack"])
        assert d.registry == ["web_search", "slack"]

    def test_empty_registry(self):
        d = ExternalToolDispatcher(tool_registry=[])
        assert d.registry == []


# ── Constructor ───────────────────────────────────────────────


class TestConstructor:
    def test_defaults_personal_tools_to_empty_dict(self):
        d = ExternalToolDispatcher(tool_registry=["web_search"])
        assert d._personal_tools == {}

    def test_accepts_personal_tools(self):
        tools = {"my_tool": "/path/to/tool.py"}
        d = ExternalToolDispatcher(tool_registry=[], personal_tools=tools)
        assert d._personal_tools == tools
