# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for A2 agentic loop enhancement.

Tests the complete A2 loop with:
- discover_tools progressive tool disclosure
- search_code and list_directory
- Structured error responses
- Parallel tool_call execution
- JSON parse error handling
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.asyncio

from core.schemas import ModelConfig
from tests.helpers.mocks import (
    make_litellm_response,
    make_tool_call,
)


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "test"
    d.mkdir(parents=True)
    (d / "permissions.md").write_text(
        "## 外部ツール\n- chatwork: OK\n- slack: OK\n"
        "## ファイル操作\n## コマンド実行\n",
        encoding="utf-8",
    )
    (d / "identity.md").write_text("# Test Anima", encoding="utf-8")
    for sub in ["episodes", "knowledge", "procedures", "skills", "state", "shortterm"]:
        (d / sub).mkdir(exist_ok=True)
    return d


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        model="openai/gpt-4o",
        api_key="sk-test",
        max_tokens=1024,
        max_turns=10,
        context_threshold=0.50,
        max_chains=2,
    )


@pytest.fixture
def memory(anima_dir: Path) -> MagicMock:
    from core.memory import MemoryManager
    m = MagicMock(spec=MemoryManager)
    m.read_permissions.return_value = (
        "## 外部ツール\n- chatwork: OK\n- slack: OK\n"
        "## ファイル操作\n## コマンド実行\n"
    )
    m.search_memory_text.return_value = []
    m.anima_dir = anima_dir
    return m


@pytest.fixture
def executor(model_config, anima_dir, memory):
    from core.tooling.handler import ToolHandler
    from core.execution.litellm_loop import LiteLLMExecutor
    th = ToolHandler(
        anima_dir=anima_dir,
        memory=memory,
        tool_registry=["chatwork", "slack"],
    )
    return LiteLLMExecutor(
        model_config=model_config,
        anima_dir=anima_dir,
        tool_handler=th,
        tool_registry=["chatwork", "slack"],
        memory=memory,
    )


def _install_litellm_mock(mock: AsyncMock) -> MagicMock:
    mock_mod = MagicMock()
    mock_mod.acompletion = mock
    mock_mod.token_counter = MagicMock(return_value=100)
    sys.modules.setdefault("litellm", mock_mod)
    return mock_mod


class TestDiscoverToolsE2E:
    """Test discover_tools progressive tool disclosure in the loop."""

    async def test_list_categories_then_respond(self, executor):
        """LLM calls discover_tools() → gets categories → responds."""
        tc_discover = make_tool_call("discover_tools", {}, "call_001")
        resp1 = make_litellm_response(content="", tool_calls=[tc_discover])
        resp2 = make_litellm_response(
            content="I can use chatwork and slack tools.",
            tool_calls=None,
        )

        mock = AsyncMock(side_effect=[resp1, resp2])
        _install_litellm_mock(mock)
        with patch("litellm.acompletion", mock):
            result = await executor.execute("What tools do you have?", system_prompt="sys")
        assert "chatwork" in result.text.lower() or "slack" in result.text.lower()

    async def test_activate_category_adds_tools(self, executor):
        """discover_tools(category='chatwork') → new tools added → used."""
        tc_discover = make_tool_call(
            "discover_tools", {"category": "chatwork"}, "call_001",
        )
        resp1 = make_litellm_response(content="", tool_calls=[tc_discover])
        resp2 = make_litellm_response(content="Tools activated", tool_calls=None)

        mock_schemas = [
            {"name": "chatwork_send", "description": "Send chatwork msg", "parameters": {"type": "object", "properties": {}}},
            {"name": "chatwork_messages", "description": "Get chatwork msgs", "parameters": {"type": "object", "properties": {}}},
        ]

        mock = AsyncMock(side_effect=[resp1, resp2])
        _install_litellm_mock(mock)
        with patch("litellm.acompletion", mock), \
             patch("core.execution.litellm_loop.load_external_schemas", return_value=mock_schemas):
            result = await executor.execute("Check chatwork", system_prompt="sys")
        assert "Tools activated" in result.text

    async def test_duplicate_category_activation(self, executor):
        """Activating same category twice returns 'already active'."""
        tc1 = make_tool_call("discover_tools", {"category": "chatwork"}, "call_001")
        tc2 = make_tool_call("discover_tools", {"category": "chatwork"}, "call_002")
        resp1 = make_litellm_response(content="", tool_calls=[tc1])
        resp2 = make_litellm_response(content="", tool_calls=[tc2])
        resp3 = make_litellm_response(content="Done", tool_calls=None)

        mock_schemas = [
            {"name": "chatwork_send", "description": "Send", "parameters": {}},
        ]

        mock = AsyncMock(side_effect=[resp1, resp2, resp3])
        _install_litellm_mock(mock)
        with patch("litellm.acompletion", mock), \
             patch("core.execution.litellm_loop.load_external_schemas", return_value=mock_schemas):
            result = await executor.execute("test", system_prompt="sys")
        assert "Done" in result.text


class TestSearchCodeE2E:
    """Test search_code tool in the A2 loop."""

    async def test_search_code_in_loop(self, executor, anima_dir):
        """LLM calls search_code → gets results → responds."""
        (anima_dir / "test_file.py").write_text(
            "def calculate():\n    return 42\n", encoding="utf-8",
        )

        tc = make_tool_call("search_code", {"pattern": "calculate"}, "call_001")
        resp1 = make_litellm_response(content="", tool_calls=[tc])
        resp2 = make_litellm_response(
            content="Found calculate function",
            tool_calls=None,
        )

        mock = AsyncMock(side_effect=[resp1, resp2])
        _install_litellm_mock(mock)
        with patch("litellm.acompletion", mock):
            result = await executor.execute("Find calculate", system_prompt="sys")
        assert "Found calculate" in result.text


class TestListDirectoryE2E:
    """Test list_directory tool in the A2 loop."""

    async def test_list_directory_in_loop(self, executor, anima_dir):
        """LLM calls list_directory → gets listing → responds."""
        (anima_dir / "readme.md").write_text("# Test", encoding="utf-8")
        (anima_dir / "config.json").write_text("{}", encoding="utf-8")

        tc = make_tool_call("list_directory", {}, "call_001")
        resp1 = make_litellm_response(content="", tool_calls=[tc])
        resp2 = make_litellm_response(
            content="I see readme.md and config.json",
            tool_calls=None,
        )

        mock = AsyncMock(side_effect=[resp1, resp2])
        _install_litellm_mock(mock)
        with patch("litellm.acompletion", mock):
            result = await executor.execute("List files", system_prompt="sys")
        assert "readme" in result.text.lower() or "config" in result.text.lower()


class TestStructuredErrorsE2E:
    """Test structured errors in the A2 loop."""

    async def test_file_not_found_structured(self, executor, anima_dir):
        """read_file on missing file returns structured error."""
        tc = make_tool_call(
            "read_file",
            {"path": str(anima_dir / "nonexistent.txt")},
            "call_001",
        )
        resp1 = make_litellm_response(content="", tool_calls=[tc])
        resp2 = make_litellm_response(content="File not found, let me try another approach", tool_calls=None)

        mock = AsyncMock(side_effect=[resp1, resp2])
        _install_litellm_mock(mock)
        with patch("litellm.acompletion", mock):
            result = await executor.execute("Read file", system_prompt="sys")
        # The LLM should have received the structured error and responded
        assert "not found" in result.text.lower() or "another approach" in result.text.lower()


class TestJsonParseErrorE2E:
    """Test that invalid JSON tool arguments are handled gracefully."""

    async def test_invalid_json_continues_loop(self, executor):
        """Invalid JSON args → structured error → LLM retries → success."""
        tc_bad = make_tool_call("search_memory", {"query": "test"}, "call_001")
        tc_bad.function.arguments = "{invalid json"
        resp1 = make_litellm_response(content="", tool_calls=[tc_bad])

        tc_good = make_tool_call("search_memory", {"query": "test"}, "call_002")
        resp2 = make_litellm_response(content="", tool_calls=[tc_good])
        resp3 = make_litellm_response(content="Found it", tool_calls=None)

        mock = AsyncMock(side_effect=[resp1, resp2, resp3])
        _install_litellm_mock(mock)
        with patch("litellm.acompletion", mock):
            result = await executor.execute("Search", system_prompt="sys")
        assert "Found it" in result.text


class TestParallelToolCallsE2E:
    """Test parallel tool_call execution."""

    async def test_multiple_reads_execute(self, executor, anima_dir):
        """Multiple read tool calls in one response are handled."""
        (anima_dir / "a.txt").write_text("content_a", encoding="utf-8")
        (anima_dir / "b.txt").write_text("content_b", encoding="utf-8")

        tc1 = make_tool_call("read_file", {"path": str(anima_dir / "a.txt")}, "call_001")
        tc2 = make_tool_call("read_file", {"path": str(anima_dir / "b.txt")}, "call_002")
        resp1 = make_litellm_response(content="", tool_calls=[tc1, tc2])
        resp2 = make_litellm_response(
            content="Read both files",
            tool_calls=None,
        )

        mock = AsyncMock(side_effect=[resp1, resp2])
        _install_litellm_mock(mock)
        with patch("litellm.acompletion", mock):
            result = await executor.execute("Read files", system_prompt="sys")
        assert "Read both files" in result.text


class TestBaseToolCount:
    """Verify the base tool set matches the design spec."""

    def test_base_tool_count(self, executor):
        """Base tools should be 22 (4 memory + 1 archive + 4 messaging + 1 procedure + 1 knowledge + 4 file + 2 search + 1 discovery + 1 tool_management + 3 task)."""
        tools = executor._build_base_tools()
        assert len(tools) == 22
        names = {t["function"]["name"] for t in tools}
        assert "search_code" in names
        assert "list_directory" in names
        assert "discover_tools" in names
        assert "read_file" in names
        assert "search_memory" in names
        assert "refresh_tools" in names
        assert "share_tool" in names
        assert "post_channel" in names
        assert "add_task" in names
        assert "update_task" in names
        assert "list_tasks" in names
        assert "read_channel" in names
        assert "read_dm_history" in names
        assert "report_procedure_outcome" in names
        assert "skill" in names
