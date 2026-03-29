"""Unit tests for AnimaWorks MCP server."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from mcp.types import TextContent, Tool


# ── Expected internal tool names (fixed set) ─────────────────────────

EXPECTED_INTERNAL_TOOL_NAMES: frozenset[str] = frozenset(
    {
        # AW-essential 10 tools (unified architecture)
        "search_memory",
        "read_memory_file",
        "write_memory_file",
        "send_message",
        "post_channel",
        "call_human",
        "delegate_task",
        "submit_tasks",
        "update_task",
        "create_skill",
    }
)


# ── TestMcpToolSchemas ───────────────────────────────────────────────


class TestMcpToolSchemas:
    """Tests for the static MCP_TOOLS list built at import time."""

    def test_mcp_tools_includes_all_internal(self) -> None:
        """MCP_TOOLS includes at least all 10 AW-essential tools."""
        from core.mcp.server import MCP_TOOLS

        actual_names = {t.name for t in MCP_TOOLS}
        assert actual_names >= EXPECTED_INTERNAL_TOOL_NAMES

    def test_all_expected_internal_tool_names_present(self) -> None:
        """All 10 AW-essential tool names are present in MCP_TOOLS."""
        from core.mcp.server import MCP_TOOLS

        actual_names = {t.name for t in MCP_TOOLS}
        assert actual_names >= EXPECTED_INTERNAL_TOOL_NAMES

    def test_discover_tools_not_exposed(self) -> None:
        """discover_tools is no longer in the MCP tool list."""
        from core.mcp.server import MCP_TOOLS

        actual_names = {t.name for t in MCP_TOOLS}
        assert "discover_tools" not in actual_names

    def test_each_tool_has_nonempty_description(self) -> None:
        """Every tool has a non-empty description string."""
        from core.mcp.server import MCP_TOOLS

        for tool in MCP_TOOLS:
            assert isinstance(tool.description, str), f"Tool '{tool.name}' description is not a string"
            assert len(tool.description) > 0, f"Tool '{tool.name}' has an empty description"

    def test_each_tool_has_object_input_schema(self) -> None:
        """Every tool has an inputSchema with type 'object'."""
        from core.mcp.server import MCP_TOOLS

        for tool in MCP_TOOLS:
            schema = tool.inputSchema
            assert isinstance(schema, dict), f"Tool '{tool.name}' inputSchema is not a dict"
            assert schema.get("type") == "object", (
                f"Tool '{tool.name}' inputSchema type is '{schema.get('type')}', expected 'object'"
            )


# ── TestBuildMcpTools ────────────────────────────────────────────────


class TestBuildMcpTools:
    """Tests for the _build_mcp_tools() helper function."""

    def test_returns_tuple(self) -> None:
        """_build_mcp_tools() returns a (tools, exposed_names) tuple."""
        from core.mcp.server import _build_mcp_tools

        result = _build_mcp_tools()
        assert isinstance(result, tuple)
        tools, exposed = result
        assert isinstance(tools, list)
        assert isinstance(exposed, frozenset)
        for tool in tools:
            assert isinstance(tool, Tool)

    def test_internal_tools_always_included(self) -> None:
        """Internal tools from _EXPOSED_TOOL_NAMES are always returned."""
        from core.mcp.server import _EXPOSED_TOOL_NAMES, _build_mcp_tools

        tools, exposed = _build_mcp_tools()
        actual_names = {t.name for t in tools}
        assert actual_names >= _EXPOSED_TOOL_NAMES
        assert exposed >= _EXPOSED_TOOL_NAMES


# ── TestListToolsHandler ─────────────────────────────────────────────


class TestListToolsHandler:
    """Tests for the list_tools() MCP handler with dynamic supervisor filtering."""

    async def test_returns_all_tools_when_supervisor(self) -> None:
        """list_tools() returns all MCP_TOOLS when Anima has subordinates."""
        import core.mcp.server as mcp_mod
        from core.mcp.server import MCP_TOOLS, list_tools

        with patch.object(mcp_mod, "_is_supervisor", True):
            result = await list_tools()
        assert result is MCP_TOOLS

    async def test_filters_supervisor_tools_when_non_supervisor(self) -> None:
        """list_tools() excludes supervisor tools when Anima has no subordinates."""
        import core.mcp.server as mcp_mod
        from core.mcp.server import MCP_TOOLS, _SUPERVISOR_TOOL_NAMES, list_tools

        with patch.object(mcp_mod, "_is_supervisor", False):
            result = await list_tools()

        result_names = {t.name for t in result}
        assert result_names & _SUPERVISOR_TOOL_NAMES == set()
        non_supervisor_names = {t.name for t in MCP_TOOLS if t.name not in _SUPERVISOR_TOOL_NAMES}
        assert result_names == non_supervisor_names

    async def test_supervisor_tool_names_from_schemas(self) -> None:
        """_SUPERVISOR_TOOL_NAMES matches SUPERVISOR_TOOLS from schemas.py."""
        from core.mcp.server import _SUPERVISOR_TOOL_NAMES
        from core.tooling.schemas import _supervisor_tools

        expected = frozenset(t["name"] for t in _supervisor_tools())
        assert expected == _SUPERVISOR_TOOL_NAMES

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

    async def test_rejects_tool_name_not_in_exposed_names(self) -> None:
        """call_tool() rejects a tool not in the dynamic _EXPOSED_NAMES set."""
        import core.mcp.server as mcp_mod

        # A completely fabricated name that won't be in any exposed set
        result = await mcp_mod.call_tool("__totally_fake_tool__", {"query": "test"})

        assert len(result) == 1
        payload = json.loads(result[0].text)
        assert payload["status"] == "error"
        assert payload["error_type"] == "ToolNotFound"

    async def test_returns_error_when_tool_handler_not_initialised(self) -> None:
        """When _get_tool_handler returns None, returns error TextContent."""
        import core.mcp.server as mcp_mod

        with (
            patch.object(mcp_mod, "_get_tool_handler", return_value=None),
            patch.object(mcp_mod, "_init_error", "Test init error"),
        ):
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

        with patch.object(mcp_mod, "_get_tool_handler", return_value=None), patch.object(mcp_mod, "_init_error", None):
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
        """When ToolHandler.handle() returns a result, returns wrapped TextContent."""
        import core.mcp.server as mcp_mod

        mock_handler = MagicMock()
        mock_handler.handle.return_value = '{"status": "ok", "data": "hello"}'

        with patch.object(mcp_mod, "_get_tool_handler", return_value=mock_handler):
            result = await mcp_mod.call_tool("send_message", {"to": "x", "content": "y"})

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert result[0].type == "text"
        assert '<tool_result tool="send_message" trust="trusted">' in result[0].text
        assert '{"status": "ok", "data": "hello"}' in result[0].text
        assert "</tool_result>" in result[0].text
        mock_handler.handle.assert_called_once_with("send_message", {"to": "x", "content": "y"})

    async def test_passes_empty_dict_when_arguments_none(self) -> None:
        """When arguments is None, passes {} to ToolHandler.handle()."""
        import core.mcp.server as mcp_mod

        mock_handler = MagicMock()
        mock_handler.handle.return_value = '{"status": "ok"}'

        with patch.object(mcp_mod, "_get_tool_handler", return_value=mock_handler):
            result = await mcp_mod.call_tool("send_message", None)

        mock_handler.handle.assert_called_once_with("send_message", {})


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
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
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
        self,
        monkeypatch: pytest.MonkeyPatch,
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
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """If ToolHandler construction fails, the error is cached."""
        import core.mcp.server as mcp_mod

        # Create a real directory so the env-var and dir-exist checks pass
        anima_dir = tmp_path / "test-anima"
        anima_dir.mkdir()
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        # Patch MemoryManager to explode so we never reach ToolHandler
        with (
            patch(
                "core.mcp.server.MemoryManager",
                side_effect=RuntimeError("memory init failed"),
                create=True,
            ),
            patch.dict("sys.modules", {}, clear=False),
        ):
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
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
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

        with (
            patch("core.memory.MemoryManager", return_value=mock_memory),
            patch("core.paths.get_shared_dir", return_value=mock_shared_dir),
            patch("core.messenger.Messenger", return_value=mock_messenger),
            patch("core.tooling.handler.ToolHandler", return_value=mock_tool_handler) as mock_th_cls,
            patch("core.config.models.load_config") as mock_load_config,
            patch("core.notification.notifier.HumanNotifier") as mock_hn_cls,
            patch("core.tools.TOOL_MODULES", {"web_search": None}),
            patch("core.tools.discover_common_tools", return_value={}),
            patch("core.tools.discover_personal_tools", return_value={}),
        ):
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
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
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

        with (
            patch("core.memory.MemoryManager", return_value=mock_memory),
            patch("core.paths.get_shared_dir", return_value=mock_shared_dir),
            patch("core.messenger.Messenger", return_value=mock_messenger),
            patch("core.tooling.handler.ToolHandler", return_value=mock_tool_handler) as mock_th_cls,
            patch("core.config.models.load_config"),
            patch("core.notification.notifier.HumanNotifier") as mock_hn_cls,
            patch("core.tools.TOOL_MODULES", {}),
            patch("core.tools.discover_common_tools", return_value={}),
            patch("core.tools.discover_personal_tools", return_value={}),
        ):
            mock_hn_cls.from_config.return_value = mock_hn_inst

            result = mcp_mod._get_tool_handler()

        assert result is mock_tool_handler
        call_kwargs = mock_th_cls.call_args.kwargs
        assert call_kwargs["human_notifier"] is mock_hn_inst

    def test_human_notifier_failure_is_tolerated(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
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

        with (
            patch("core.memory.MemoryManager", return_value=mock_memory),
            patch("core.paths.get_shared_dir", return_value=mock_shared_dir),
            patch("core.messenger.Messenger", return_value=mock_messenger),
            patch("core.tooling.handler.ToolHandler", return_value=mock_tool_handler) as mock_th_cls,
            patch("core.config.models.load_config", side_effect=RuntimeError("no config")),
            patch("core.tools.TOOL_MODULES", {}),
            patch("core.tools.discover_common_tools", return_value={}),
            patch("core.tools.discover_personal_tools", return_value={}),
        ):
            result = mcp_mod._get_tool_handler()

        assert result is mock_tool_handler
        call_kwargs = mock_th_cls.call_args.kwargs
        assert call_kwargs["human_notifier"] is None

    def test_tool_discovery_failure_is_tolerated(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
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

        with (
            patch("core.memory.MemoryManager", return_value=mock_memory),
            patch("core.paths.get_shared_dir", return_value=mock_shared_dir),
            patch("core.messenger.Messenger", return_value=mock_messenger),
            patch("core.tooling.handler.ToolHandler", return_value=mock_tool_handler) as mock_th_cls,
            patch("core.config.models.load_config"),
            patch("core.notification.notifier.HumanNotifier") as mock_hn_cls,
            patch("core.tools.TOOL_MODULES", side_effect=ImportError("no tools")),
            patch("core.tools.discover_common_tools", return_value={}),
            patch("core.tools.discover_personal_tools", return_value={}),
        ):
            mock_hn_inst = MagicMock()
            mock_hn_inst.channel_count = 0
            mock_hn_cls.from_config.return_value = mock_hn_inst

            result = mcp_mod._get_tool_handler()

        assert result is mock_tool_handler
        call_kwargs = mock_th_cls.call_args.kwargs
        assert call_kwargs["tool_registry"] == []
        assert call_kwargs["personal_tools"] == {}


# ── TestLoadPermittedCategories ──────────────────────────────────────


class TestLoadPermittedCategories:
    """Tests for _load_permitted_categories() permissions.md parser."""

    def test_no_permissions_file_returns_all(self, tmp_path: Path) -> None:
        """Without permissions.md, all tools are returned."""
        from core.mcp.server import _load_permitted_categories

        with patch("core.tools.TOOL_MODULES", {"chatwork": "core.tools.chatwork", "slack": "core.tools.slack"}):
            result = _load_permitted_categories(tmp_path)
        assert result == {"chatwork", "slack"}

    def test_no_external_tools_section_returns_all(self, tmp_path: Path) -> None:
        """When permissions.md has no 外部ツール section, returns all."""
        from core.mcp.server import _load_permitted_categories

        perms = tmp_path / "permissions.md"
        perms.write_text("## 実行できるコマンド\n- git: OK\n", encoding="utf-8")

        with patch("core.tools.TOOL_MODULES", {"chatwork": "x", "slack": "x"}):
            result = _load_permitted_categories(tmp_path)
        assert result == {"chatwork", "slack"}

    def test_whitelist_mode(self, tmp_path: Path) -> None:
        """Individual allow entries produce a whitelist."""
        from core.mcp.server import _load_permitted_categories

        perms = tmp_path / "permissions.md"
        perms.write_text(
            "## 外部ツール\n- chatwork: 全権限\n- slack: 読み取りのみ\n",
            encoding="utf-8",
        )

        with patch("core.tools.TOOL_MODULES", {"chatwork": "x", "slack": "x", "gmail": "x"}):
            result = _load_permitted_categories(tmp_path)
        assert result == {"chatwork", "slack"}
        assert "gmail" not in result

    def test_all_yes_mode(self, tmp_path: Path) -> None:
        """'all: yes' enables all tools."""
        from core.mcp.server import _load_permitted_categories

        perms = tmp_path / "permissions.md"
        perms.write_text("## 外部ツール\n- all: yes\n", encoding="utf-8")

        with patch("core.tools.TOOL_MODULES", {"chatwork": "x", "slack": "x", "gmail": "x"}):
            result = _load_permitted_categories(tmp_path)
        assert result == {"chatwork", "slack", "gmail"}

    def test_all_yes_with_deny(self, tmp_path: Path) -> None:
        """'all: yes' with individual deny entries."""
        from core.mcp.server import _load_permitted_categories

        perms = tmp_path / "permissions.md"
        perms.write_text("## 外部ツール\n- all: yes\n- gmail: no\n", encoding="utf-8")

        with patch("core.tools.TOOL_MODULES", {"chatwork": "x", "slack": "x", "gmail": "x"}):
            result = _load_permitted_categories(tmp_path)
        assert result == {"chatwork", "slack"}
        assert "gmail" not in result


# ── TestExternalToolsInMcpTools ──────────────────────────────────────


class TestExternalToolsInMcpTools:
    """Tests for external tool schema loading in _build_mcp_tools()."""

    def test_use_tool_not_exposed_in_mcp(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """MCP server does NOT expose use_tool (Mode B only)."""
        from core.mcp.server import _build_mcp_tools

        anima_dir = tmp_path / "test-anima"
        anima_dir.mkdir()
        perms = anima_dir / "permissions.md"
        perms.write_text("## 外部ツール\n- chatwork: 全権限\n", encoding="utf-8")
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        tools, exposed = _build_mcp_tools()

        tool_names = {t.name for t in tools}
        assert "use_tool" not in tool_names
        assert "use_tool" not in exposed

    def test_unpermitted_tools_excluded(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """External tools not in permissions.md are excluded."""
        from core.mcp.server import _build_mcp_tools

        anima_dir = tmp_path / "test-anima"
        anima_dir.mkdir()
        perms = anima_dir / "permissions.md"
        perms.write_text("## 外部ツール\n- chatwork: 全権限\n", encoding="utf-8")
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        with patch(
            "core.tooling.schemas.load_external_schemas_by_category",
            return_value=[],
        ):
            tools, exposed = _build_mcp_tools()

        tool_names = {t.name for t in tools}
        assert "gmail_send" not in tool_names

    def test_external_tool_dispatch_via_call_tool(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """call_tool() dispatches external tools through ToolHandler."""
        import core.mcp.server as mcp_mod

        # Temporarily add an external tool to the exposed set
        original_exposed = mcp_mod._EXPOSED_NAMES
        mcp_mod._EXPOSED_NAMES = original_exposed | frozenset({"chatwork_send"})

        mock_handler = MagicMock()
        mock_handler.handle.return_value = '{"status": "ok"}'

        try:
            import asyncio

            with patch.object(mcp_mod, "_get_tool_handler", return_value=mock_handler):
                result = asyncio.run(mcp_mod.call_tool("chatwork_send", {"room_id": "123", "body": "test"}))

            assert len(result) == 1
            assert "ok" in result[0].text
            assert "<tool_result" in result[0].text
            mock_handler.handle.assert_called_once_with(
                "chatwork_send",
                {"room_id": "123", "body": "test"},
            )
        finally:
            mcp_mod._EXPOSED_NAMES = original_exposed


# ── TestCallToolTrustWrapping ────────────────────────────────────────


class TestCallToolTrustWrapping:
    """Tests for trust boundary labeling in call_tool()."""

    async def test_untrusted_tool_gets_untrusted_tag(self) -> None:
        """web_search results are wrapped with trust='untrusted'."""
        import core.mcp.server as mcp_mod

        original_exposed = mcp_mod._EXPOSED_NAMES
        mcp_mod._EXPOSED_NAMES = original_exposed | frozenset({"web_search"})

        mock_handler = MagicMock()
        mock_handler.handle.return_value = "Search results here"

        try:
            with patch.object(mcp_mod, "_get_tool_handler", return_value=mock_handler):
                result = await mcp_mod.call_tool("web_search", {"query": "test"})

            assert len(result) == 1
            assert '<tool_result tool="web_search" trust="untrusted">' in result[0].text
            assert "Search results here" in result[0].text
            assert "</tool_result>" in result[0].text
        finally:
            mcp_mod._EXPOSED_NAMES = original_exposed

    async def test_trusted_tool_gets_trusted_tag(self) -> None:
        """search_memory results are wrapped with trust='trusted'."""
        import core.mcp.server as mcp_mod

        mock_handler = MagicMock()
        mock_handler.handle.return_value = "Memory result"

        with patch.object(mcp_mod, "_get_tool_handler", return_value=mock_handler):
            result = await mcp_mod.call_tool("search_memory", {"query": "test"})

        assert len(result) == 1
        assert '<tool_result tool="search_memory" trust="trusted">' in result[0].text
        assert "Memory result" in result[0].text

    async def test_medium_trust_tool_gets_medium_tag(self) -> None:
        """read_file results are wrapped with trust='medium'."""
        import core.mcp.server as mcp_mod

        original_exposed = mcp_mod._EXPOSED_NAMES
        mcp_mod._EXPOSED_NAMES = original_exposed | frozenset({"read_file"})

        mock_handler = MagicMock()
        mock_handler.handle.return_value = "file contents"

        try:
            with patch.object(mcp_mod, "_get_tool_handler", return_value=mock_handler):
                result = await mcp_mod.call_tool("read_file", {"path": "/tmp/x"})

            assert len(result) == 1
            assert '<tool_result tool="read_file" trust="medium">' in result[0].text
            assert "file contents" in result[0].text
        finally:
            mcp_mod._EXPOSED_NAMES = original_exposed

    async def test_empty_result_not_wrapped(self) -> None:
        """Empty tool results are returned as-is (not wrapped)."""
        import core.mcp.server as mcp_mod

        mock_handler = MagicMock()
        mock_handler.handle.return_value = ""

        with patch.object(mcp_mod, "_get_tool_handler", return_value=mock_handler):
            result = await mcp_mod.call_tool("send_message", {"to": "x", "content": "y"})

        assert len(result) == 1
        assert result[0].text == ""
        assert "<tool_result" not in result[0].text

    async def test_error_responses_not_wrapped(self) -> None:
        """Error JSON responses (ToolNotFound, InitError, UnhandledError) have no trust tag."""
        import core.mcp.server as mcp_mod

        # ToolNotFound
        result = await mcp_mod.call_tool("__nonexistent__", {})
        assert "<tool_result" not in result[0].text
        payload = json.loads(result[0].text)
        assert payload["status"] == "error"

        # InitError
        with (
            patch.object(mcp_mod, "_get_tool_handler", return_value=None),
            patch.object(mcp_mod, "_init_error", "init failed"),
        ):
            result = await mcp_mod.call_tool("send_message", {"to": "x", "content": "y"})
        assert "<tool_result" not in result[0].text
        payload = json.loads(result[0].text)
        assert payload["status"] == "error"

        # UnhandledError
        mock_handler = MagicMock()
        mock_handler.handle.side_effect = RuntimeError("boom")
        with patch.object(mcp_mod, "_get_tool_handler", return_value=mock_handler):
            result = await mcp_mod.call_tool("send_message", {"to": "x", "content": "y"})
        assert "<tool_result" not in result[0].text
        payload = json.loads(result[0].text)
        assert payload["status"] == "error"

    async def test_unknown_tool_defaults_to_untrusted(self) -> None:
        """Tools not in TOOL_TRUST_LEVELS default to trust='untrusted'."""
        import core.mcp.server as mcp_mod

        original_exposed = mcp_mod._EXPOSED_NAMES
        mcp_mod._EXPOSED_NAMES = original_exposed | frozenset({"custom_tool_xyz"})

        mock_handler = MagicMock()
        mock_handler.handle.return_value = "custom output"

        try:
            with patch.object(mcp_mod, "_get_tool_handler", return_value=mock_handler):
                result = await mcp_mod.call_tool("custom_tool_xyz", {})

            assert len(result) == 1
            assert '<tool_result tool="custom_tool_xyz" trust="untrusted">' in result[0].text
        finally:
            mcp_mod._EXPOSED_NAMES = original_exposed


# ── TestWrapResultHelper ─────────────────────────────────────────────


class TestWrapResultHelper:
    """Tests for the _wrap_result() helper function."""

    def test_wraps_result_with_trust_tag(self) -> None:
        """_wrap_result applies trust label from TOOL_TRUST_LEVELS."""
        from core.mcp.server import _wrap_result

        wrapped = _wrap_result("search_memory", "some data")
        assert '<tool_result tool="search_memory" trust="trusted">' in wrapped
        assert "some data" in wrapped
        assert "</tool_result>" in wrapped

    def test_returns_empty_unchanged(self) -> None:
        """_wrap_result returns empty strings unchanged."""
        from core.mcp.server import _wrap_result

        assert _wrap_result("search_memory", "") == ""

    def test_fallback_on_import_error(self) -> None:
        """_wrap_result returns raw result when wrap_tool_result import fails."""
        from core.mcp.server import _wrap_result

        with patch(
            "core.execution._sanitize.wrap_tool_result",
            side_effect=ImportError("no module"),
        ):
            result = _wrap_result("search_memory", "raw data")

        assert result == "raw data"


# ── TestHasSubordinatesForAnima ──────────────────────────────────────


class TestHasSubordinatesForAnima:
    """Tests for _has_subordinates_for_anima() helper."""

    def setup_method(self) -> None:
        import core.mcp.server as mcp_mod

        mcp_mod._is_supervisor = None

    def teardown_method(self) -> None:
        import core.mcp.server as mcp_mod

        mcp_mod._is_supervisor = None

    def test_returns_false_when_env_not_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Falls back to False when ANIMAWORKS_ANIMA_DIR is not set."""
        import core.mcp.server as mcp_mod

        monkeypatch.delenv("ANIMAWORKS_ANIMA_DIR", raising=False)
        assert mcp_mod._has_subordinates_for_anima() is False

    def test_returns_true_when_has_subordinates(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Returns True when config.json shows subordinates."""
        import core.mcp.server as mcp_mod

        anima_dir = tmp_path / "animas" / "boss"
        anima_dir.mkdir(parents=True)
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        config_data = {
            "animas": {
                "boss": {"supervisor": None},
                "worker1": {"supervisor": "boss"},
            }
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config_data), encoding="utf-8")

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            result = mcp_mod._has_subordinates_for_anima()

        assert result is True

    def test_returns_false_when_no_subordinates(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Returns False when config.json shows no subordinates."""
        import core.mcp.server as mcp_mod

        anima_dir = tmp_path / "animas" / "leaf"
        anima_dir.mkdir(parents=True)
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        config_data = {
            "animas": {
                "boss": {"supervisor": None},
                "leaf": {"supervisor": "boss"},
            }
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config_data), encoding="utf-8")

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            result = mcp_mod._has_subordinates_for_anima()

        assert result is False

    def test_caches_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Result is cached on second call."""
        import core.mcp.server as mcp_mod

        mcp_mod._is_supervisor = False
        assert mcp_mod._has_subordinates_for_anima() is False

    def test_config_read_failure_defaults_to_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Falls back to False if config.json cannot be read."""
        import core.mcp.server as mcp_mod

        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        with patch("core.paths.get_data_dir", side_effect=RuntimeError("no dir")):
            result = mcp_mod._has_subordinates_for_anima()

        assert result is False
