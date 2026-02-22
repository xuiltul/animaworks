"""Tests for core.tooling.schemas — canonical tool schema definitions and converters."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

import core.tools
from core.tooling.schemas import (
    DISCOVERY_TOOLS,
    FILE_TOOLS,
    MEMORY_TOOLS,
    SEARCH_TOOLS,
    TOOL_MANAGEMENT_TOOLS,
    build_tool_list,
    load_external_schemas,
    to_anthropic_format,
    to_litellm_format,
)


# ── Canonical schema structure ─────────────────────────────────


class TestMemoryTools:
    def test_memory_tools_is_list(self):
        assert isinstance(MEMORY_TOOLS, list)
        assert len(MEMORY_TOOLS) == 5

    def test_search_memory_schema(self):
        schema = next(t for t in MEMORY_TOOLS if t["name"] == "search_memory")
        assert "description" in schema
        assert schema["parameters"]["type"] == "object"
        assert "query" in schema["parameters"]["properties"]
        assert "query" in schema["parameters"]["required"]

    def test_search_memory_scope_includes_common_knowledge(self):
        schema = next(t for t in MEMORY_TOOLS if t["name"] == "search_memory")
        scope_enum = schema["parameters"]["properties"]["scope"]["enum"]
        assert "common_knowledge" in scope_enum
        # Also verify other expected values are still present
        assert "knowledge" in scope_enum
        assert "episodes" in scope_enum
        assert "procedures" in scope_enum
        assert "all" in scope_enum

    def test_read_memory_file_schema(self):
        schema = next(t for t in MEMORY_TOOLS if t["name"] == "read_memory_file")
        assert "path" in schema["parameters"]["properties"]
        assert "path" in schema["parameters"]["required"]

    def test_write_memory_file_schema(self):
        schema = next(t for t in MEMORY_TOOLS if t["name"] == "write_memory_file")
        props = schema["parameters"]["properties"]
        assert "path" in props
        assert "content" in props
        assert "mode" in props
        assert set(schema["parameters"]["required"]) == {"path", "content"}

    def test_send_message_schema(self):
        schema = next(t for t in MEMORY_TOOLS if t["name"] == "send_message")
        props = schema["parameters"]["properties"]
        assert "to" in props
        assert "content" in props
        assert set(schema["parameters"]["required"]) == {"to", "content", "intent"}


class TestSendMessageSchema:
    def test_intent_property_exists(self):
        send_msg = next(t for t in MEMORY_TOOLS if t["name"] == "send_message")
        assert "intent" in send_msg["parameters"]["properties"]

    def test_intent_required(self):
        send_msg = next(t for t in MEMORY_TOOLS if t["name"] == "send_message")
        assert "intent" in send_msg["parameters"]["required"]

    def test_intent_type_is_string(self):
        send_msg = next(t for t in MEMORY_TOOLS if t["name"] == "send_message")
        assert send_msg["parameters"]["properties"]["intent"]["type"] == "string"


class TestFileTools:
    def test_file_tools_is_list(self):
        assert isinstance(FILE_TOOLS, list)
        assert len(FILE_TOOLS) == 4

    def test_read_file_schema(self):
        schema = next(t for t in FILE_TOOLS if t["name"] == "read_file")
        assert "path" in schema["parameters"]["properties"]

    def test_write_file_schema(self):
        schema = next(t for t in FILE_TOOLS if t["name"] == "write_file")
        assert set(schema["parameters"]["required"]) == {"path", "content"}

    def test_edit_file_schema(self):
        schema = next(t for t in FILE_TOOLS if t["name"] == "edit_file")
        assert set(schema["parameters"]["required"]) == {"path", "old_string", "new_string"}

    def test_execute_command_schema(self):
        schema = next(t for t in FILE_TOOLS if t["name"] == "execute_command")
        assert "command" in schema["parameters"]["properties"]
        assert "timeout" in schema["parameters"]["properties"]


class TestSearchTools:
    def test_search_tools_is_list(self):
        assert isinstance(SEARCH_TOOLS, list)
        assert len(SEARCH_TOOLS) == 2

    def test_search_code_schema(self):
        schema = next(t for t in SEARCH_TOOLS if t["name"] == "search_code")
        assert "pattern" in schema["parameters"]["properties"]
        assert "pattern" in schema["parameters"]["required"]

    def test_list_directory_schema(self):
        schema = next(t for t in SEARCH_TOOLS if t["name"] == "list_directory")
        assert "path" in schema["parameters"]["properties"]
        assert "recursive" in schema["parameters"]["properties"]


class TestDiscoveryTools:
    def test_discovery_tools_is_list(self):
        assert isinstance(DISCOVERY_TOOLS, list)
        assert len(DISCOVERY_TOOLS) == 1

    def test_discover_tools_schema(self):
        schema = DISCOVERY_TOOLS[0]
        assert schema["name"] == "discover_tools"
        assert "category" in schema["parameters"]["properties"]


class TestToolManagementTools:
    def test_tool_management_tools_is_list(self):
        assert isinstance(TOOL_MANAGEMENT_TOOLS, list)
        assert len(TOOL_MANAGEMENT_TOOLS) == 2

    def test_contains_refresh_tools(self):
        names = [t["name"] for t in TOOL_MANAGEMENT_TOOLS]
        assert "refresh_tools" in names

    def test_contains_share_tool(self):
        names = [t["name"] for t in TOOL_MANAGEMENT_TOOLS]
        assert "share_tool" in names

    def test_refresh_tools_schema(self):
        schema = next(t for t in TOOL_MANAGEMENT_TOOLS if t["name"] == "refresh_tools")
        assert "description" in schema
        assert schema["parameters"]["type"] == "object"
        # refresh_tools has no required parameters
        assert "required" not in schema["parameters"]

    def test_share_tool_schema(self):
        schema = next(t for t in TOOL_MANAGEMENT_TOOLS if t["name"] == "share_tool")
        assert "description" in schema
        assert schema["parameters"]["type"] == "object"
        assert "tool_name" in schema["parameters"]["properties"]
        assert "tool_name" in schema["parameters"]["required"]


# ── Format converters ─────────────────────────────────────────


class TestToAnthropicFormat:
    def test_converts_single_tool(self):
        tools = [{"name": "foo", "description": "desc", "parameters": {"type": "object"}}]
        result = to_anthropic_format(tools)
        assert len(result) == 1
        assert result[0]["name"] == "foo"
        assert result[0]["description"] == "desc"
        assert result[0]["input_schema"] == {"type": "object"}
        assert "parameters" not in result[0]

    def test_converts_multiple_tools(self):
        result = to_anthropic_format(MEMORY_TOOLS)
        assert len(result) == len(MEMORY_TOOLS)
        for item in result:
            assert "name" in item
            assert "description" in item
            assert "input_schema" in item

    def test_empty_list(self):
        assert to_anthropic_format([]) == []


class TestToLitellmFormat:
    def test_converts_single_tool(self):
        tools = [{"name": "bar", "description": "desc2", "parameters": {"type": "object"}}]
        result = to_litellm_format(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "bar"
        assert result[0]["function"]["description"] == "desc2"
        assert result[0]["function"]["parameters"] == {"type": "object"}

    def test_converts_multiple_tools(self):
        result = to_litellm_format(FILE_TOOLS)
        assert len(result) == len(FILE_TOOLS)
        for item in result:
            assert item["type"] == "function"
            assert "name" in item["function"]

    def test_empty_list(self):
        assert to_litellm_format([]) == []


# ── build_tool_list ───────────────────────────────────────────


class TestBuildToolList:
    def test_default_returns_memory_tools_only(self):
        result = build_tool_list()
        names = [t["name"] for t in result]
        assert "search_memory" in names
        assert "read_memory_file" in names
        assert "write_memory_file" in names
        assert "send_message" in names
        assert "read_file" not in names

    def test_include_file_tools(self):
        result = build_tool_list(include_file_tools=True)
        names = [t["name"] for t in result]
        assert "read_file" in names
        assert "write_file" in names
        assert "edit_file" in names
        assert "execute_command" in names

    def test_include_external_schemas(self):
        ext = [{"name": "custom_tool", "description": "custom", "parameters": {}}]
        result = build_tool_list(external_schemas=ext)
        names = [t["name"] for t in result]
        assert "custom_tool" in names

    def test_combined(self):
        ext = [{"name": "ext1", "description": "e", "parameters": {}}]
        result = build_tool_list(
            include_file_tools=True,
            external_schemas=ext,
        )
        names = [t["name"] for t in result]
        assert "search_memory" in names
        assert "read_file" in names
        assert "ext1" in names

    def test_include_search_tools(self):
        result = build_tool_list(include_search_tools=True)
        names = [t["name"] for t in result]
        assert "search_code" in names
        assert "list_directory" in names
        # Should NOT include file tools unless requested
        assert "read_file" not in names

    def test_include_discovery_tools(self):
        result = build_tool_list(include_discovery_tools=True)
        names = [t["name"] for t in result]
        assert "discover_tools" in names

    def test_include_tool_management(self):
        result = build_tool_list(include_tool_management=True)
        names = [t["name"] for t in result]
        assert "refresh_tools" in names
        assert "share_tool" in names
        # Should still include memory tools
        assert "search_memory" in names
        # Should NOT include other optional tools unless requested
        assert "read_file" not in names
        assert "discover_tools" not in names

    def test_all_flags_combined(self):
        result = build_tool_list(
            include_file_tools=True,
            include_search_tools=True,
            include_discovery_tools=True,
            include_tool_management=True,
        )
        names = [t["name"] for t in result]
        # 5 memory + 3 channel + 1 report_procedure_outcome + 1 report_knowledge_outcome + 4 file + 2 search + 1 discovery + 2 tool_management = 19
        assert len(result) == 19
        assert "search_code" in names
        assert "list_directory" in names
        assert "discover_tools" in names
        assert "refresh_tools" in names
        assert "share_tool" in names
        assert "post_channel" in names
        assert "read_channel" in names
        assert "read_dm_history" in names

    def test_does_not_mutate_memory_tools(self):
        original_len = len(MEMORY_TOOLS)
        build_tool_list(include_file_tools=True)
        assert len(MEMORY_TOOLS) == original_len


# ── load_external_schemas ─────────────────────────────────────


class TestLoadExternalSchemas:
    def test_empty_registry(self):
        assert load_external_schemas([]) == []

    def test_unknown_tool_name(self):
        result = load_external_schemas(["nonexistent_tool_xyz"])
        assert result == []

    def test_loads_schemas_from_module(self):
        mock_mod = MagicMock()
        mock_mod.get_tool_schemas.return_value = [
            {
                "name": "web_search",
                "description": "Search the web",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        ]

        with patch.dict(core.tools.TOOL_MODULES, {"web_search": "core.tools.web_search"}, clear=True), \
             patch("importlib.import_module", return_value=mock_mod):
            result = load_external_schemas(["web_search"])

        assert len(result) == 1
        assert result[0]["name"] == "web_search"
        assert result[0]["parameters"] == {
            "type": "object",
            "properties": {"query": {"type": "string"}},
        }

    def test_handles_module_without_get_tool_schemas(self):
        mock_mod = MagicMock(spec=[])  # No get_tool_schemas attribute

        with patch.dict(core.tools.TOOL_MODULES, {"web_search": "core.tools.web_search"}, clear=True), \
             patch("importlib.import_module", return_value=mock_mod):
            result = load_external_schemas(["web_search"])

        assert result == []

    def test_handles_import_error(self):
        with patch.dict(core.tools.TOOL_MODULES, {"web_search": "core.tools.web_search"}, clear=True), \
             patch("importlib.import_module", side_effect=ImportError("no module")):
            result = load_external_schemas(["web_search"])

        assert result == []

    def test_skips_tool_not_in_registry(self):
        with patch.dict(core.tools.TOOL_MODULES, {"web_search": "core.tools.web_search"}, clear=True):
            result = load_external_schemas(["slack"])

        assert result == []

    def test_uses_parameters_key_as_fallback(self):
        mock_mod = MagicMock()
        mock_mod.get_tool_schemas.return_value = [
            {
                "name": "test_tool",
                "description": "Test",
                "parameters": {"type": "object", "properties": {}},
            }
        ]

        with patch.dict(core.tools.TOOL_MODULES, {"test": "core.tools.test"}, clear=True), \
             patch("importlib.import_module", return_value=mock_mod):
            result = load_external_schemas(["test"])

        assert len(result) == 1
        assert result[0]["parameters"] == {"type": "object", "properties": {}}
