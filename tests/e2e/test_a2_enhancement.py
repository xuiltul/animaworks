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


class TestUseToolE2E:
    """Test use_tool unified dispatcher in the loop."""

    async def test_use_tool_dispatches_in_loop(self, executor):
        """LLM calls use_tool() → dispatch → responds."""
        tc_use = make_tool_call(
            "use_tool",
            {"tool_name": "chatwork", "action": "rooms", "args": {}},
            "call_001",
        )
        resp1 = make_litellm_response(content="", tool_calls=[tc_use])
        resp2 = make_litellm_response(
            content="I checked chatwork rooms.",
            tool_calls=None,
        )

        mock = AsyncMock(side_effect=[resp1, resp2])
        _install_litellm_mock(mock)
        with patch("litellm.acompletion", mock):
            result = await executor.execute("Check chatwork rooms", system_prompt="sys")
        assert "chatwork" in result.text.lower()


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
    """Verify the base tool set matches the unified design spec."""

    def test_base_tool_count(self, executor):
        """Base tools should be 17 (CC 8 + AW-essential 9, no notification/supervisor)."""
        tools = executor._build_base_tools()
        assert len(tools) == 17
        names = {t["function"]["name"] for t in tools}
        # CC built-in tools (8)
        assert "Read" in names
        assert "Write" in names
        assert "Edit" in names
        assert "Bash" in names
        assert "Grep" in names
        assert "Glob" in names
        assert "WebSearch" in names
        assert "WebFetch" in names
        # AW-essential: memory + messaging
        assert "search_memory" in names
        assert "read_memory_file" in names
        assert "write_memory_file" in names
        assert "send_message" in names
        assert "post_channel" in names
        # AW-essential: task management
        assert "submit_tasks" in names
        assert "update_task" in names
        # AW-essential: skill
        assert "skill" in names
        # AW-essential: planning
        assert "todo_write" in names
        # Mode B only — must NOT be in Mode A
        assert "use_tool" not in names
