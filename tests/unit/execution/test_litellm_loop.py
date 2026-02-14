"""Tests for core.execution.litellm_loop — Mode A2: LiteLLM tool_use loop."""
from __future__ import annotations

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
    sys.modules.setdefault("litellm", mock_mod)
    return mock_mod


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def person_dir(tmp_path: Path) -> Path:
    d = tmp_path / "persons" / "test"
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
def memory(person_dir: Path) -> MagicMock:
    from core.memory import MemoryManager
    m = MagicMock(spec=MemoryManager)
    m.read_permissions.return_value = ""
    m.search_memory_text.return_value = []
    m.person_dir = person_dir
    return m


@pytest.fixture
def tool_handler(person_dir: Path, memory: MagicMock) -> ToolHandler:
    return ToolHandler(
        person_dir=person_dir,
        memory=memory,
        tool_registry=[],
    )


@pytest.fixture
def executor(
    model_config: ModelConfig,
    person_dir: Path,
    tool_handler: ToolHandler,
    memory: MagicMock,
):
    from core.execution.litellm_loop import LiteLLMExecutor
    return LiteLLMExecutor(
        model_config=model_config,
        person_dir=person_dir,
        tool_handler=tool_handler,
        tool_registry=[],
        memory=memory,
    )


# ── _build_tools ──────────────────────────────────────────────


class TestBuildTools:
    def test_returns_litellm_format(self, executor):
        with patch("core.tooling.schemas.load_external_schemas", return_value=[]):
            tools = executor._build_tools()
        assert isinstance(tools, list)
        for t in tools:
            assert t["type"] == "function"
            assert "name" in t["function"]

    def test_includes_file_tools(self, executor):
        with patch("core.tooling.schemas.load_external_schemas", return_value=[]):
            tools = executor._build_tools()
        names = [t["function"]["name"] for t in tools]
        assert "read_file" in names
        assert "write_file" in names



# ── _build_llm_kwargs ────────────────────────────────────────


class TestBuildLlmKwargs:
    def test_includes_model_and_max_tokens(self, executor):
        kwargs = executor._build_llm_kwargs()
        assert kwargs["model"] == "openai/gpt-4o"
        assert kwargs["max_tokens"] == 1024

    def test_includes_api_key(self, executor):
        kwargs = executor._build_llm_kwargs()
        assert kwargs["api_key"] == "sk-test"

    def test_includes_api_base(self, model_config: ModelConfig, person_dir: Path, memory: MagicMock):
        model_config.api_base_url = "http://localhost:11434/v1"
        th = ToolHandler(person_dir=person_dir, memory=memory, tool_registry=[])
        from core.execution.litellm_loop import LiteLLMExecutor
        ex = LiteLLMExecutor(
            model_config=model_config,
            person_dir=person_dir,
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

    async def test_session_chaining(self, executor, person_dir: Path):
        tracker = ContextTracker(model="openai/gpt-4o", threshold=0.50)
        shortterm = ShortTermMemory(person_dir)

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
