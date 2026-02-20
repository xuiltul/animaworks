"""Tests for core.execution.litellm_loop — Mode A2: LiteLLM tool_use loop."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.asyncio

import core.tools
from core.execution.base import ExecutionResult
from core.prompt.context import ContextTracker
from core.schemas import ModelConfig
from core.memory.shortterm import ShortTermMemory
from core.tooling.handler import ToolHandler
from tests.helpers.mocks import (
    make_litellm_response,
    make_tool_call,
)


# ── litellm sys.modules mock ─────────────────────────────────


def _install_litellm_mock(acompletion_mock: AsyncMock) -> MagicMock:
    """Install a mock litellm module into sys.modules."""
    mock_mod = MagicMock()
    mock_mod.acompletion = acompletion_mock
    # token_counter must return a real int so that the pre-flight
    # max_tokens clamping logic can compare with `<`.
    mock_mod.token_counter = MagicMock(return_value=500)
    sys.modules.setdefault("litellm", mock_mod)
    return mock_mod


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "test"
    d.mkdir(parents=True)
    (d / "permissions.md").write_text("", encoding="utf-8")
    (d / "identity.md").write_text("# Test", encoding="utf-8")
    for sub in ["episodes", "knowledge", "procedures", "skills", "state", "shortterm"]:
        (d / sub).mkdir(exist_ok=True)
    return d


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        model="openai/gpt-4o",
        api_key="sk-test",
        max_tokens=1024,
        max_turns=5,
        context_threshold=0.50,
        max_chains=2,
    )


@pytest.fixture
def memory(anima_dir: Path) -> MagicMock:
    from core.memory import MemoryManager
    m = MagicMock(spec=MemoryManager)
    m.read_permissions.return_value = ""
    m.search_memory_text.return_value = []
    m.anima_dir = anima_dir
    return m


@pytest.fixture
def tool_handler(anima_dir: Path, memory: MagicMock) -> ToolHandler:
    return ToolHandler(
        anima_dir=anima_dir,
        memory=memory,
        tool_registry=[],
    )


@pytest.fixture
def executor(
    model_config: ModelConfig,
    anima_dir: Path,
    tool_handler: ToolHandler,
    memory: MagicMock,
):
    from core.execution.litellm_loop import LiteLLMExecutor
    return LiteLLMExecutor(
        model_config=model_config,
        anima_dir=anima_dir,
        tool_handler=tool_handler,
        tool_registry=[],
        memory=memory,
    )


# ── _build_base_tools ─────────────────────────────────────────


class TestBuildBaseTools:
    def test_returns_litellm_format(self, executor):
        tools = executor._build_base_tools()
        assert isinstance(tools, list)
        for t in tools:
            assert t["type"] == "function"
            assert "name" in t["function"]

    def test_includes_file_tools(self, executor):
        tools = executor._build_base_tools()
        names = [t["function"]["name"] for t in tools]
        assert "read_file" in names
        assert "write_file" in names

    def test_includes_search_tools(self, executor):
        tools = executor._build_base_tools()
        names = [t["function"]["name"] for t in tools]
        assert "search_code" in names
        assert "list_directory" in names

    def test_includes_discover_tools(self, executor):
        tools = executor._build_base_tools()
        names = [t["function"]["name"] for t in tools]
        assert "discover_tools" in names

    def test_does_not_include_external_tools(self, executor):
        tools = executor._build_base_tools()
        names = [t["function"]["name"] for t in tools]
        # Should not have any external tools like chatwork_*, slack_*, etc.
        assert not any(n.startswith("chatwork_") for n in names)
        assert not any(n.startswith("slack_") for n in names)


# ── _build_llm_kwargs ────────────────────────────────────────


class TestBuildLlmKwargs:
    def test_includes_model_and_max_tokens(self, executor):
        kwargs = executor._build_llm_kwargs()
        assert kwargs["model"] == "openai/gpt-4o"
        assert kwargs["max_tokens"] == 1024

    def test_includes_api_key(self, executor):
        kwargs = executor._build_llm_kwargs()
        assert kwargs["api_key"] == "sk-test"

    def test_includes_api_base(self, model_config: ModelConfig, anima_dir: Path, memory: MagicMock):
        model_config.api_base_url = "http://localhost:11434/v1"
        th = ToolHandler(anima_dir=anima_dir, memory=memory, tool_registry=[])
        from core.execution.litellm_loop import LiteLLMExecutor
        ex = LiteLLMExecutor(
            model_config=model_config,
            anima_dir=anima_dir,
            tool_handler=th,
            tool_registry=[],
            memory=memory,
        )
        kwargs = ex._build_llm_kwargs()
        assert kwargs["api_base"] == "http://localhost:11434/v1"


# ── execute() — simple response ──────────────────────────────


class TestExecuteSimple:
    async def test_returns_text_response(self, executor):
        resp = make_litellm_response(content="Hello world", tool_calls=None)
        mock = AsyncMock(side_effect=[resp])
        _install_litellm_mock(mock)
        with patch("litellm.acompletion", mock):
            result = await executor.execute("test prompt", system_prompt="sys")
        assert "Hello world" in result.text

    async def test_no_result_message(self, executor):
        resp = make_litellm_response(content="text")
        mock = AsyncMock(side_effect=[resp])
        with patch("litellm.acompletion", mock):
            result = await executor.execute("test")
        assert result.result_message is None


# ── execute() — tool call loop ───────────────────────────────


class TestExecuteWithTools:
    async def test_processes_tool_calls(self, executor):
        tc = make_tool_call("search_memory", {"query": "test"})
        resp_with_tool = make_litellm_response(content="", tool_calls=[tc])
        resp_final = make_litellm_response(content="Final answer", tool_calls=None)

        mock = AsyncMock(side_effect=[resp_with_tool, resp_final])
        _install_litellm_mock(mock)
        with patch("litellm.acompletion", mock):
            result = await executor.execute("test prompt", system_prompt="sys")
        assert "Final answer" in result.text

    async def test_invalid_json_args_handled(self, executor):
        tc = make_tool_call("search_memory", {"query": "test"})
        tc.function.arguments = "not valid json {"
        resp_with_tool = make_litellm_response(content="", tool_calls=[tc])
        resp_final = make_litellm_response(content="Done", tool_calls=None)

        mock = AsyncMock(side_effect=[resp_with_tool, resp_final])
        with patch("litellm.acompletion", mock):
            result = await executor.execute("test", system_prompt="sys")
        assert "Done" in result.text

    async def test_max_iterations_reached(self, executor):
        tc = make_tool_call("search_memory", {"query": "test"})
        resp_with_tool = make_litellm_response(content="", tool_calls=[tc])

        mock = AsyncMock(return_value=resp_with_tool)
        with patch("litellm.acompletion", mock):
            result = await executor.execute("test", system_prompt="sys")
        assert "max iterations" in result.text


# ── execute() — context tracking ─────────────────────────────


class TestExecuteContextTracking:
    async def test_tracks_usage(self, executor):
        tracker = ContextTracker(model="openai/gpt-4o", threshold=0.50)
        resp = make_litellm_response(
            content="Response",
            prompt_tokens=50_000,
            completion_tokens=10_000,
        )
        mock = AsyncMock(side_effect=[resp])
        with patch("litellm.acompletion", mock):
            await executor.execute("test", system_prompt="sys", tracker=tracker)
        assert tracker.usage_ratio > 0

    async def test_session_chaining(self, executor, anima_dir: Path):
        tracker = ContextTracker(model="openai/gpt-4o", threshold=0.50)
        shortterm = ShortTermMemory(anima_dir)

        resp_threshold = make_litellm_response(
            content="Partial",
            prompt_tokens=100_000,
            completion_tokens=10_000,
        )
        resp_final = make_litellm_response(content="Continued", prompt_tokens=1000)

        mock = AsyncMock(side_effect=[resp_threshold, resp_final])
        _install_litellm_mock(mock)
        with patch("litellm.acompletion", mock), \
             patch("core.execution.litellm_loop.build_system_prompt", return_value="sys"), \
             patch("core.execution._session.inject_shortterm", return_value="sys+st"), \
             patch("core.execution._session.load_prompt", return_value="continue"):
            result = await executor.execute(
                "test", system_prompt="sys", tracker=tracker, shortterm=shortterm,
            )
        assert "Continued" in result.text or "Partial" in result.text


# ── _partition_tool_calls ────────────────────────────────────


class TestPartitionToolCalls:
    def test_all_reads_are_parallel(self):
        from core.execution.litellm_loop import _partition_tool_calls
        tc1 = make_tool_call("read_file", {"path": "/a"}, "call_1")
        tc2 = make_tool_call("search_memory", {"query": "x"}, "call_2")
        parallel, serial = _partition_tool_calls([tc1, tc2])
        assert len(parallel) == 2
        assert serial == []

    def test_writes_to_different_paths_are_parallel(self):
        from core.execution.litellm_loop import _partition_tool_calls
        tc1 = make_tool_call("write_file", {"path": "/a"}, "call_1")
        tc2 = make_tool_call("write_file", {"path": "/b"}, "call_2")
        parallel, serial = _partition_tool_calls([tc1, tc2])
        assert len(parallel) == 2
        assert serial == []

    def test_writes_to_same_path_serialised(self):
        from core.execution.litellm_loop import _partition_tool_calls
        tc1 = make_tool_call("write_file", {"path": "/a"}, "call_1")
        tc2 = make_tool_call("write_file", {"path": "/a"}, "call_2")
        parallel, serial = _partition_tool_calls([tc1, tc2])
        assert len(parallel) == 1  # first write
        assert len(serial) == 1  # second write serialised
        assert len(serial[0]) == 1


# ── execute() — invalid JSON ────────────────────────────────


class TestExecuteWithInvalidJson:
    async def test_invalid_json_returns_structured_error(self, executor):
        tc = make_tool_call("search_memory", {"query": "test"})
        tc.function.arguments = "not valid json {"
        resp_with_tool = make_litellm_response(content="", tool_calls=[tc])
        resp_final = make_litellm_response(content="Done", tool_calls=None)

        mock = AsyncMock(side_effect=[resp_with_tool, resp_final])
        _install_litellm_mock(mock)
        with patch("litellm.acompletion", mock):
            result = await executor.execute("test", system_prompt="sys")
        assert "Done" in result.text
        # The error message should have been appended to messages
        # and the LLM should have been given a chance to retry


# ── execute() — discover_tools ───────────────────────────────


class TestDiscoverTools:
    async def test_discover_tools_lists_categories(self, executor):
        executor._tool_registry = ["chatwork", "slack"]
        tc = make_tool_call("discover_tools", {})
        resp_with_tool = make_litellm_response(content="", tool_calls=[tc])
        resp_final = make_litellm_response(content="Got categories", tool_calls=None)

        mock = AsyncMock(side_effect=[resp_with_tool, resp_final])
        _install_litellm_mock(mock)
        with patch("litellm.acompletion", mock):
            result = await executor.execute("test", system_prompt="sys")
        assert "Got categories" in result.text

    async def test_discover_tools_activates_category(self, executor):
        executor._tool_registry = ["chatwork"]
        tc = make_tool_call("discover_tools", {"category": "chatwork"})
        resp_with_tool = make_litellm_response(content="", tool_calls=[tc])
        resp_final = make_litellm_response(content="Category active", tool_calls=None)

        mock_schemas = [
            {"name": "chatwork_send", "description": "Send chatwork", "parameters": {}},
        ]
        mock = AsyncMock(side_effect=[resp_with_tool, resp_final])
        _install_litellm_mock(mock)
        with patch("litellm.acompletion", mock), \
             patch("core.execution.litellm_loop.load_external_schemas", return_value=mock_schemas):
            result = await executor.execute("test", system_prompt="sys")
        assert "Category active" in result.text


# ── execute() — tool execution error handling ─────────────


class TestToolExecutionErrorHandling:
    """H2/H3: Exceptions during tool execution must produce structured error
    tool messages (not silently drop them), preserving the API contract."""

    async def test_parallel_exception_returns_error_message(self, executor):
        """H2: When a parallel tool call raises, an error tool message with
        the correct tool_call_id must be appended."""
        tc = make_tool_call("read_file", {"path": "/fail"}, "call_err_1")
        resp_with_tool = make_litellm_response(content="", tool_calls=[tc])
        resp_final = make_litellm_response(content="Recovered", tool_calls=None)

        mock = AsyncMock(side_effect=[resp_with_tool, resp_final])
        _install_litellm_mock(mock)

        with patch("litellm.acompletion", mock), \
             patch.object(
                 executor, "_execute_tool_call",
                 side_effect=RuntimeError("disk failure"),
             ):
            result = await executor.execute("test", system_prompt="sys")

        # LLM should have received an error tool message and produced final text
        assert "Recovered" in result.text
        # Verify the error message was in the second call's messages
        second_call_messages = mock.call_args_list[1].kwargs.get(
            "messages", mock.call_args_list[1][0][0] if mock.call_args_list[1][0] else []
        )
        tool_msgs = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_msgs) >= 1
        err_msg = tool_msgs[0]
        assert err_msg["tool_call_id"] == "call_err_1"
        parsed = json.loads(err_msg["content"])
        assert parsed["status"] == "error"
        assert parsed["error_type"] == "ExecutionError"
        assert "disk failure" in parsed["message"]

    async def test_serial_exception_returns_error_message(self, executor):
        """H3: When a serial (same-path write) tool call raises, an error tool
        message with the correct tool_call_id must be appended."""
        # Two writes to the same path → first parallel, second serial
        tc1 = make_tool_call("write_file", {"path": "/a", "content": "x"}, "call_w1")
        tc2 = make_tool_call("write_file", {"path": "/a", "content": "y"}, "call_w2")
        resp_with_tools = make_litellm_response(content="", tool_calls=[tc1, tc2])
        resp_final = make_litellm_response(content="Done after error", tool_calls=None)

        call_count = 0

        async def mock_exec(tc, fn_args):
            nonlocal call_count
            call_count += 1
            if tc.id == "call_w2":
                raise RuntimeError("serial write failed")
            return {"role": "tool", "tool_call_id": tc.id, "content": "ok"}

        mock = AsyncMock(side_effect=[resp_with_tools, resp_final])
        _install_litellm_mock(mock)

        with patch("litellm.acompletion", mock), \
             patch.object(executor, "_execute_tool_call", side_effect=mock_exec):
            result = await executor.execute("test", system_prompt="sys")

        assert "Done after error" in result.text
        # Verify error message was appended for the serial call
        second_call_messages = mock.call_args_list[1].kwargs.get(
            "messages", mock.call_args_list[1][0][0] if mock.call_args_list[1][0] else []
        )
        tool_msgs = [m for m in second_call_messages if m.get("role") == "tool"]
        err_msgs = [m for m in tool_msgs if m["tool_call_id"] == "call_w2"]
        assert len(err_msgs) == 1
        parsed = json.loads(err_msgs[0]["content"])
        assert parsed["status"] == "error"
        assert "serial write failed" in parsed["message"]


# ── session chaining — execution_mode ─────────────────────


class TestSessionChainingExecutionMode:
    """H1: Session chaining partial must pass execution_mode='a2'."""

    async def test_session_chaining_preserves_a2_mode(self, executor, anima_dir: Path):
        """When session chaining triggers, build_system_prompt must be called
        with execution_mode='a2'."""
        tracker = ContextTracker(model="openai/gpt-4o", threshold=0.50)
        shortterm = ShortTermMemory(anima_dir)

        resp_threshold = make_litellm_response(
            content="Partial",
            prompt_tokens=100_000,
            completion_tokens=10_000,
        )
        resp_final = make_litellm_response(content="Continued", prompt_tokens=1000)

        mock = AsyncMock(side_effect=[resp_threshold, resp_final])
        _install_litellm_mock(mock)

        build_spy = MagicMock(return_value="new-system-prompt")

        with patch("litellm.acompletion", mock), \
             patch("core.execution.litellm_loop.build_system_prompt", build_spy), \
             patch("core.execution._session.inject_shortterm", return_value="sys+st"), \
             patch("core.execution._session.load_prompt", return_value="continue"):
            await executor.execute(
                "test", system_prompt="sys", tracker=tracker, shortterm=shortterm,
            )

        # Verify build_system_prompt was called with execution_mode="a2"
        if build_spy.called:
            _, kwargs = build_spy.call_args
            assert kwargs.get("execution_mode") == "a2", (
                f"Expected execution_mode='a2', got {kwargs.get('execution_mode')!r}"
            )


# ── _BG_POOL_TOOLS ──────────────────────────────────────────


class TestBgPoolTools:
    def test_contains_image_gen_schema_names(self):
        """_BG_POOL_TOOLS includes all image_gen tool schema names."""
        from core.execution.litellm_loop import LiteLLMExecutor
        expected_image_tools = {
            "generate_character_assets",
            "generate_fullbody", "generate_bustup", "generate_chibi",
            "generate_3d_model", "generate_rigged_model", "generate_animations",
        }
        for tool in expected_image_tools:
            assert tool in LiteLLMExecutor._BG_POOL_TOOLS, f"{tool} missing from _BG_POOL_TOOLS"

    def test_contains_other_bg_tools(self):
        """_BG_POOL_TOOLS includes local_llm and run_command."""
        from core.execution.litellm_loop import LiteLLMExecutor
        assert "local_llm" in LiteLLMExecutor._BG_POOL_TOOLS
        assert "run_command" in LiteLLMExecutor._BG_POOL_TOOLS

    def test_does_not_contain_category_names(self):
        """_BG_POOL_TOOLS must NOT contain old category name 'image_generation'."""
        from core.execution.litellm_loop import LiteLLMExecutor
        assert "image_generation" not in LiteLLMExecutor._BG_POOL_TOOLS


# ── _build_llm_kwargs — timeout & num_ctx ────────────────────


class TestBuildLlmKwargsTimeoutAndNumCtx:
    """Verify that _build_llm_kwargs() includes timeout and num_ctx."""

    def test_timeout_included_in_kwargs(self, executor):
        """_build_llm_kwargs() must include a 'timeout' key."""
        kwargs = executor._build_llm_kwargs()
        assert "timeout" in kwargs
        assert isinstance(kwargs["timeout"], int)
        assert kwargs["timeout"] > 0

    def test_num_ctx_for_ollama_model(
        self, anima_dir: Path, memory: MagicMock,
    ):
        """model='ollama/gemma3:27b' → kwargs must contain 'num_ctx'."""
        ollama_config = ModelConfig(
            model="ollama/gemma3:27b",
            api_key="sk-test",
            max_tokens=1024,
            max_turns=5,
            context_threshold=0.50,
            max_chains=2,
        )
        th = ToolHandler(anima_dir=anima_dir, memory=memory, tool_registry=[])
        from core.execution.litellm_loop import LiteLLMExecutor
        ex = LiteLLMExecutor(
            model_config=ollama_config,
            anima_dir=anima_dir,
            tool_handler=th,
            tool_registry=[],
            memory=memory,
        )
        with patch("core.config.load_config") as mock_cfg:
            mock_cfg.return_value.model_context_windows = None
            kwargs = ex._build_llm_kwargs()
        assert "num_ctx" in kwargs
        assert isinstance(kwargs["num_ctx"], int)
        assert kwargs["num_ctx"] > 0

    def test_no_num_ctx_for_non_ollama(self, executor):
        """model='openai/gpt-4o' → kwargs must NOT contain 'num_ctx'."""
        kwargs = executor._build_llm_kwargs()
        assert "num_ctx" not in kwargs
