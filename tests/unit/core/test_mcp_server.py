"""Unit tests for AnimaWorks MCP server."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import TextContent, Tool


# ── Expected tool names ──────────────────────────────────────────────

EXPECTED_TOOL_NAMES: frozenset[str] = frozenset({
    "send_message",
    "post_channel",
    "read_channel",
    "read_dm_history",
    "add_task",
    "update_task",
    "list_tasks",
    "call_human",
    "search_memory",
    "report_procedure_outcome",
    "report_knowledge_outcome",
    "discover_tools",
})


# ── TestMcpToolSchemas ───────────────────────────────────────────────


class TestMcpToolSchemas:
    """Tests for the static MCP_TOOLS list built at import time."""

    def test_mcp_tools_count(self) -> None:
        """MCP_TOOLS has exactly 12 tools."""
        from core.mcp.server import MCP_TOOLS

        assert len(MCP_TOOLS) == 12

    def test_all_expected_tool_names_present(self) -> None:
        """All 12 expected tool names are present in MCP_TOOLS."""
        from core.mcp.server import MCP_TOOLS

        actual_names = {t.name for t in MCP_TOOLS}
        assert actual_names == EXPECTED_TOOL_NAMES

    def test_each_tool_has_nonempty_description(self) -> None:
        """Every tool has a non-empty description string."""
        from core.mcp.server import MCP_TOOLS

        for tool in MCP_TOOLS:
            assert isinstance(tool.description, str), (
                f"Tool '{tool.name}' description is not a string"
            )
            assert len(tool.description) > 0, (
                f"Tool '{tool.name}' has an empty description"
            )

    def test_each_tool_has_object_input_schema(self) -> None:
        """Every tool has an inputSchema with type 'object'."""
        from core.mcp.server import MCP_TOOLS

        for tool in MCP_TOOLS:
            schema = tool.inputSchema
            assert isinstance(schema, dict), (
                f"Tool '{tool.name}' inputSchema is not a dict"
            )
            assert schema.get("type") == "object", (
                f"Tool '{tool.name}' inputSchema type is "
                f"'{schema.get('type')}', expected 'object'"
            )


# ── TestBuildMcpTools ────────────────────────────────────────────────


class TestBuildMcpTools:
    """Tests for the _build_mcp_tools() helper function."""

    def test_returns_tool_objects(self) -> None:
        """_build_mcp_tools() returns a list of mcp.types.Tool objects."""
        from core.mcp.server import _build_mcp_tools

        tools = _build_mcp_tools()
        assert isinstance(tools, list)
        for tool in tools:
            assert isinstance(tool, Tool)

    def test_filters_by_exposed_tool_names(self) -> None:
        """Only tools in _EXPOSED_TOOL_NAMES are returned."""
        from core.mcp.server import _EXPOSED_TOOL_NAMES, _build_mcp_tools

        tools = _build_mcp_tools()
        actual_names = {t.name for t in tools}
        # Every returned tool must be in the exposed set
        assert actual_names <= _EXPOSED_TOOL_NAMES
        # And the full set should be covered (no missing schemas)
        assert actual_names == _EXPOSED_TOOL_NAMES


# ── TestListToolsHandler ─────────────────────────────────────────────


class TestListToolsHandler:
    """Tests for the list_tools() MCP handler."""

    async def test_returns_mcp_tools_list(self) -> None:
        """list_tools() returns the MCP_TOOLS list."""
        from core.mcp.server import MCP_TOOLS, list_tools

        result = await list_tools()
        assert result is MCP_TOOLS

    def test_list_tools_is_async(self) -> None:
        """list_tools should be a coroutine function."""
        import asyncio

        from core.mcp.server import list_tools

        assert asyncio.iscoroutinefunction(list_tools)


# ── TestCallToolHandler ──────────────────────────────────────────────


class TestCallToolHandler:
    """Tests for the call_tool() MCP handler."""

    async def test_rejects_tool_not_in_exposed_set(self) -> None:
        """call_tool() rejects tool names not in _EXPOSED_TOOL_NAMES."""
        import core.mcp.server as mcp_mod

        result = await mcp_mod.call_tool("nonexistent_tool", {"arg": "val"})

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        payload = json.loads(result[0].text)
        assert payload["status"] == "error"
        assert payload["error_type"] == "ToolNotFound"
        assert "nonexistent_tool" in payload["message"]

    async def test_rejects_external_tool_name(self) -> None:
        """call_tool() rejects a tool that exists in ToolHandler but is not exposed."""
        import core.mcp.server as mcp_mod

        # web_search exists in ToolHandler dispatch but NOT in _EXPOSED_TOOL_NAMES
        result = await mcp_mod.call_tool("web_search", {"query": "test"})

        assert len(result) == 1
        payload = json.loads(result[0].text)
        assert payload["status"] == "error"
        assert payload["error_type"] == "ToolNotFound"

    async def test_returns_error_when_tool_handler_not_initialised(self) -> None:
        """When _get_tool_handler returns None, returns error TextContent."""
        import core.mcp.server as mcp_mod

        with patch.object(mcp_mod, "_get_tool_handler", return_value=None), \
             patch.object(mcp_mod, "_init_error", "Test init error"):
            result = await mcp_mod.call_tool("send_message", {"to": "x", "content": "y"})

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert result[0].type == "text"
        payload = json.loads(result[0].text)
        assert payload["status"] == "error"
        assert payload["error_type"] == "InitError"
        assert "Test init error" in payload["message"]

    async def test_returns_error_when_init_error_is_none(self) -> None:
        """When handler is None and _init_error is also None, uses fallback message."""
        import core.mcp.server as mcp_mod

        with patch.object(mcp_mod, "_get_tool_handler", return_value=None), \
             patch.object(mcp_mod, "_init_error", None):
            result = await mcp_mod.call_tool("send_message", {"to": "x", "content": "y"})

        assert len(result) == 1
        payload = json.loads(result[0].text)
        assert payload["status"] == "error"
        assert payload["error_type"] == "InitError"
        assert "not available" in payload["message"]

    async def test_returns_error_when_tool_handler_raises(self) -> None:
        """When ToolHandler.handle() raises, returns UnhandledError TextContent."""
        import core.mcp.server as mcp_mod

        mock_handler = MagicMock()
        mock_handler.handle.side_effect = RuntimeError("boom")

        with patch.object(mcp_mod, "_get_tool_handler", return_value=mock_handler):
            result = await mcp_mod.call_tool("send_message", {"to": "x", "content": "y"})

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        payload = json.loads(result[0].text)
        assert payload["status"] == "error"
        assert payload["error_type"] == "UnhandledError"
        assert "send_message" in payload["message"]
        assert "boom" in payload["message"]

    async def test_returns_result_on_success(self) -> None:
        """When ToolHandler.handle() returns a result, returns TextContent with it."""
        import core.mcp.server as mcp_mod

        mock_handler = MagicMock()
        mock_handler.handle.return_value = '{"status": "ok", "data": "hello"}'

        with patch.object(mcp_mod, "_get_tool_handler", return_value=mock_handler):
            result = await mcp_mod.call_tool("send_message", {"to": "x", "content": "y"})

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert result[0].type == "text"
        assert result[0].text == '{"status": "ok", "data": "hello"}'
        mock_handler.handle.assert_called_once_with("send_message", {"to": "x", "content": "y"})

    async def test_passes_empty_dict_when_arguments_none(self) -> None:
        """When arguments is None, passes {} to ToolHandler.handle()."""
        import core.mcp.server as mcp_mod

        mock_handler = MagicMock()
        mock_handler.handle.return_value = '{"status": "ok"}'

        with patch.object(mcp_mod, "_get_tool_handler", return_value=mock_handler):
            result = await mcp_mod.call_tool("list_tasks", None)

        mock_handler.handle.assert_called_once_with("list_tasks", {})


# ── TestGetToolHandler ───────────────────────────────────────────────


class TestGetToolHandler:
    """Tests for the _get_tool_handler() lazy initialisation."""

    def setup_method(self) -> None:
        """Reset module-level globals before each test."""
        import core.mcp.server as mcp_mod

        mcp_mod._tool_handler = None
        mcp_mod._init_error = None

    def teardown_method(self) -> None:
        """Reset module-level globals after each test to avoid leaking."""
        import core.mcp.server as mcp_mod

        mcp_mod._tool_handler = None
        mcp_mod._init_error = None

    def test_returns_none_when_env_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When ANIMAWORKS_ANIMA_DIR is not set, returns None and sets _init_error."""
        import core.mcp.server as mcp_mod

        monkeypatch.delenv("ANIMAWORKS_ANIMA_DIR", raising=False)

        result = mcp_mod._get_tool_handler()

        assert result is None
        assert mcp_mod._init_error is not None
        assert "not set" in mcp_mod._init_error

    def test_returns_none_when_dir_does_not_exist(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """When ANIMAWORKS_ANIMA_DIR points to a nonexistent dir, returns None."""
        import core.mcp.server as mcp_mod

        nonexistent = tmp_path / "does_not_exist"
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(nonexistent))

        result = mcp_mod._get_tool_handler()

        assert result is None
        assert mcp_mod._init_error is not None
        assert "does not exist" in mcp_mod._init_error

    def test_caches_error_on_subsequent_calls(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """After an error, subsequent calls return None without retrying."""
        import core.mcp.server as mcp_mod

        monkeypatch.delenv("ANIMAWORKS_ANIMA_DIR", raising=False)

        # First call sets the error
        result1 = mcp_mod._get_tool_handler()
        assert result1 is None
        first_error = mcp_mod._init_error

        # Second call returns immediately (cached error)
        result2 = mcp_mod._get_tool_handler()
        assert result2 is None
        assert mcp_mod._init_error is first_error

    def test_returns_cached_handler_if_already_initialised(self) -> None:
        """If _tool_handler is already set, returns it directly."""
        import core.mcp.server as mcp_mod

        sentinel = MagicMock()
        mcp_mod._tool_handler = sentinel

        result = mcp_mod._get_tool_handler()

        assert result is sentinel

    def test_catches_exception_during_initialisation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """If ToolHandler construction fails, the error is cached."""
        import core.mcp.server as mcp_mod

        # Create a real directory so the env-var and dir-exist checks pass
        anima_dir = tmp_path / "test-anima"
        anima_dir.mkdir()
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        # Patch MemoryManager to explode so we never reach ToolHandler
        with patch(
            "core.mcp.server.MemoryManager",
            side_effect=RuntimeError("memory init failed"),
            create=True,
        ), patch.dict("sys.modules", {}, clear=False):
            # We need to patch the import inside _get_tool_handler.
            # The function does `from core.memory import MemoryManager`,
            # so patching the name in the server module after import.
            with patch(
                "core.memory.MemoryManager",
                side_effect=RuntimeError("memory init failed"),
            ):
                result = mcp_mod._get_tool_handler()

        assert result is None
        assert mcp_mod._init_error is not None
        assert "initialisation failed" in mcp_mod._init_error
        assert "memory init failed" in mcp_mod._init_error

    def test_successful_initialisation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """When all dependencies succeed, returns a ToolHandler instance."""
        import core.mcp.server as mcp_mod

        anima_dir = tmp_path / "test-anima"
        anima_dir.mkdir()
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        mock_memory = MagicMock()
        mock_messenger = MagicMock()
        mock_shared_dir = tmp_path / "shared"
        mock_shared_dir.mkdir()
        mock_tool_handler = MagicMock()

        with patch("core.memory.MemoryManager", return_value=mock_memory), \
             patch("core.paths.get_shared_dir", return_value=mock_shared_dir), \
             patch("core.messenger.Messenger", return_value=mock_messenger), \
             patch("core.tooling.handler.ToolHandler", return_value=mock_tool_handler) as mock_th_cls, \
             patch("core.config.models.load_config") as mock_load_config, \
             patch("core.notification.notifier.HumanNotifier") as mock_hn_cls, \
             patch("core.tools.TOOL_MODULES", {"web_search": None}), \
             patch("core.tools.discover_common_tools", return_value={}), \
             patch("core.tools.discover_personal_tools", return_value={}):
            # HumanNotifier with no channels -> None
            mock_hn_inst = MagicMock()
            mock_hn_inst.channel_count = 0
            mock_hn_cls.from_config.return_value = mock_hn_inst

            result = mcp_mod._get_tool_handler()

        assert result is mock_tool_handler
        assert mcp_mod._tool_handler is mock_tool_handler
        assert mcp_mod._init_error is None
        # ToolHandler was constructed with correct anima_dir
        mock_th_cls.assert_called_once()
        call_kwargs = mock_th_cls.call_args.kwargs
        assert call_kwargs["anima_dir"] == anima_dir
        assert call_kwargs["memory"] is mock_memory
        assert call_kwargs["messenger"] is mock_messenger
        # human_notifier should be None (channel_count=0)
        assert call_kwargs["human_notifier"] is None

    def test_successful_init_with_human_notifier(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """When HumanNotifier has channels, it is passed to ToolHandler."""
        import core.mcp.server as mcp_mod

        anima_dir = tmp_path / "test-anima"
        anima_dir.mkdir()
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        mock_memory = MagicMock()
        mock_messenger = MagicMock()
        mock_shared_dir = tmp_path / "shared"
        mock_shared_dir.mkdir()
        mock_tool_handler = MagicMock()
        mock_hn_inst = MagicMock()
        mock_hn_inst.channel_count = 2  # Has channels

        with patch("core.memory.MemoryManager", return_value=mock_memory), \
             patch("core.paths.get_shared_dir", return_value=mock_shared_dir), \
             patch("core.messenger.Messenger", return_value=mock_messenger), \
             patch("core.tooling.handler.ToolHandler", return_value=mock_tool_handler) as mock_th_cls, \
             patch("core.config.models.load_config"), \
             patch("core.notification.notifier.HumanNotifier") as mock_hn_cls, \
             patch("core.tools.TOOL_MODULES", {}), \
             patch("core.tools.discover_common_tools", return_value={}), \
             patch("core.tools.discover_personal_tools", return_value={}):
            mock_hn_cls.from_config.return_value = mock_hn_inst

            result = mcp_mod._get_tool_handler()

        assert result is mock_tool_handler
        call_kwargs = mock_th_cls.call_args.kwargs
        assert call_kwargs["human_notifier"] is mock_hn_inst

    def test_human_notifier_failure_is_tolerated(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """If HumanNotifier init fails, ToolHandler still initialises."""
        import core.mcp.server as mcp_mod

        anima_dir = tmp_path / "test-anima"
        anima_dir.mkdir()
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        mock_memory = MagicMock()
        mock_messenger = MagicMock()
        mock_shared_dir = tmp_path / "shared"
        mock_shared_dir.mkdir()
        mock_tool_handler = MagicMock()

        with patch("core.memory.MemoryManager", return_value=mock_memory), \
             patch("core.paths.get_shared_dir", return_value=mock_shared_dir), \
             patch("core.messenger.Messenger", return_value=mock_messenger), \
             patch("core.tooling.handler.ToolHandler", return_value=mock_tool_handler) as mock_th_cls, \
             patch("core.config.models.load_config", side_effect=RuntimeError("no config")), \
             patch("core.tools.TOOL_MODULES", {}), \
             patch("core.tools.discover_common_tools", return_value={}), \
             patch("core.tools.discover_personal_tools", return_value={}):
            result = mcp_mod._get_tool_handler()

        assert result is mock_tool_handler
        call_kwargs = mock_th_cls.call_args.kwargs
        assert call_kwargs["human_notifier"] is None

    def test_tool_discovery_failure_is_tolerated(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """If tool discovery fails, ToolHandler still initialises with empty tools."""
        import core.mcp.server as mcp_mod

        anima_dir = tmp_path / "test-anima"
        anima_dir.mkdir()
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        mock_memory = MagicMock()
        mock_messenger = MagicMock()
        mock_shared_dir = tmp_path / "shared"
        mock_shared_dir.mkdir()
        mock_tool_handler = MagicMock()

        with patch("core.memory.MemoryManager", return_value=mock_memory), \
             patch("core.paths.get_shared_dir", return_value=mock_shared_dir), \
             patch("core.messenger.Messenger", return_value=mock_messenger), \
             patch("core.tooling.handler.ToolHandler", return_value=mock_tool_handler) as mock_th_cls, \
             patch("core.config.models.load_config"), \
             patch("core.notification.notifier.HumanNotifier") as mock_hn_cls, \
             patch("core.tools.TOOL_MODULES", side_effect=ImportError("no tools")):
            mock_hn_inst = MagicMock()
            mock_hn_inst.channel_count = 0
            mock_hn_cls.from_config.return_value = mock_hn_inst

            result = mcp_mod._get_tool_handler()

        assert result is mock_tool_handler
        call_kwargs = mock_th_cls.call_args.kwargs
        assert call_kwargs["tool_registry"] == []
        assert call_kwargs["personal_tools"] == {}
