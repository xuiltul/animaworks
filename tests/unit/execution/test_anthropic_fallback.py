"""Tests for core.execution.anthropic_fallback — Anthropic SDK fallback executor."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.asyncio

from core.execution.anthropic_fallback import AnthropicFallbackExecutor
from core.execution.base import ExecutionResult
from core.memory import MemoryManager
from core.prompt.context import ContextTracker
from core.schemas import ModelConfig
from core.memory.shortterm import ShortTermMemory
from core.tooling.handler import ToolHandler


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        model="claude-sonnet-4-20250514",
        api_key="sk-test",
        max_tokens=1024,
        max_turns=5,
        context_threshold=0.50,
        max_chains=2,
    )


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "test"
    d.mkdir(parents=True)
    for sub in ["episodes", "knowledge", "procedures", "skills", "state", "shortterm"]:
        (d / sub).mkdir(exist_ok=True)
    (d / "permissions.md").write_text("", encoding="utf-8")
    (d / "identity.md").write_text("# Test", encoding="utf-8")
    return d


@pytest.fixture
def memory(anima_dir: Path) -> MagicMock:
    m = MagicMock(spec=MemoryManager)
    m.read_permissions.return_value = ""
    m.search_memory_text.return_value = []
    m.anima_dir = anima_dir
    return m


@pytest.fixture
def tool_handler(anima_dir: Path, memory: MagicMock) -> ToolHandler:
    return ToolHandler(anima_dir=anima_dir, memory=memory, tool_registry=[])


@pytest.fixture
def executor(
    model_config: ModelConfig,
    anima_dir: Path,
    tool_handler: ToolHandler,
    memory: MagicMock,
) -> AnthropicFallbackExecutor:
    return AnthropicFallbackExecutor(
        model_config=model_config,
        anima_dir=anima_dir,
        tool_handler=tool_handler,
        tool_registry=[],
        memory=memory,
    )


# ── Mock helpers ──────────────────────────────────────────────


def _make_text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(name: str, input_data: dict, tool_id: str = "tu_001") -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = input_data
    block.id = tool_id
    return block


def _make_response(
    content: list,
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> MagicMock:
    resp = MagicMock()
    resp.content = content
    resp.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return resp


# ── _build_tools ──────────────────────────────────────────────


class TestBuildTools:
    def test_returns_anthropic_format(self, executor: AnthropicFallbackExecutor):
        with patch("core.tooling.schemas.load_external_schemas", return_value=[]):
            tools = executor._build_tools()
        assert isinstance(tools, list)
        for t in tools:
            assert "name" in t
            assert "input_schema" in t

    def test_does_not_include_file_tools(self, executor: AnthropicFallbackExecutor):
        with patch("core.tooling.schemas.load_external_schemas", return_value=[]):
            tools = executor._build_tools()
        names = [t["name"] for t in tools]
        assert "read_file" not in names
        assert "write_file" not in names



# ── execute() — simple response ──────────────────────────────


class TestExecuteSimple:
    async def test_returns_text(self, executor: AnthropicFallbackExecutor):
        resp = _make_response([_make_text_block("Hello from Anthropic")])

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=resp)
            mock_cls.return_value = mock_client

            result = await executor.execute("test prompt", system_prompt="sys")

        assert isinstance(result, ExecutionResult)
        assert "Hello from Anthropic" in result.text

    async def test_passes_api_key(self, executor: AnthropicFallbackExecutor):
        resp = _make_response([_make_text_block("resp")])

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=resp)
            mock_cls.return_value = mock_client

            await executor.execute("test")

            mock_cls.assert_called_once()
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs.get("api_key") == "sk-test"

    async def test_passes_base_url(
        self, model_config: ModelConfig, anima_dir: Path,
        tool_handler: ToolHandler, memory: MagicMock,
    ):
        model_config.api_base_url = "https://custom.api"
        executor = AnthropicFallbackExecutor(
            model_config=model_config,
            anima_dir=anima_dir,
            tool_handler=tool_handler,
            tool_registry=[],
            memory=memory,
        )
        resp = _make_response([_make_text_block("resp")])

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=resp)
            mock_cls.return_value = mock_client

            await executor.execute("test")

            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs.get("base_url") == "https://custom.api"


# ── execute() — tool use loop ────────────────────────────────


class TestExecuteWithTools:
    async def test_processes_tool_calls(self, executor: AnthropicFallbackExecutor):
        tool_block = _make_tool_use_block("search_memory", {"query": "test"})
        resp_with_tool = _make_response([tool_block])
        resp_final = _make_response([_make_text_block("Final answer")])

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages = MagicMock()
            mock_client.messages.create = AsyncMock(
                side_effect=[resp_with_tool, resp_final],
            )
            mock_cls.return_value = mock_client

            result = await executor.execute("test prompt", system_prompt="sys")

        assert "Final answer" in result.text

    async def test_max_iterations_reached(self, executor: AnthropicFallbackExecutor):
        tool_block = _make_tool_use_block("search_memory", {"query": "test"})
        resp_with_tool = _make_response([tool_block])

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages = MagicMock()
            # Always returns tool calls, never reaches final
            mock_client.messages.create = AsyncMock(return_value=resp_with_tool)
            mock_cls.return_value = mock_client

            result = await executor.execute("test", system_prompt="sys")

        assert "max iterations" in result.text

    async def test_multiple_tool_calls_in_response(self, executor: AnthropicFallbackExecutor):
        tool1 = _make_tool_use_block("search_memory", {"query": "a"}, tool_id="tu_001")
        tool2 = _make_tool_use_block("search_memory", {"query": "b"}, tool_id="tu_002")
        resp_with_tools = _make_response([tool1, tool2])
        resp_final = _make_response([_make_text_block("Done")])

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages = MagicMock()
            mock_client.messages.create = AsyncMock(
                side_effect=[resp_with_tools, resp_final],
            )
            mock_cls.return_value = mock_client

            result = await executor.execute("test", system_prompt="sys")

        assert "Done" in result.text


# ── execute() — context tracking and session chaining ─────────


class TestExecuteContextTracking:
    async def test_tracks_usage(self, executor: AnthropicFallbackExecutor):
        tracker = ContextTracker(model="claude-sonnet-4-20250514", threshold=0.50)
        resp = _make_response(
            [_make_text_block("Response")],
            input_tokens=10_000,
            output_tokens=1_000,
        )

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=resp)
            mock_cls.return_value = mock_client

            await executor.execute("test", system_prompt="sys", tracker=tracker)

        assert tracker.usage_ratio > 0

    async def test_session_chaining(
        self, executor: AnthropicFallbackExecutor, anima_dir: Path,
    ):
        tracker = ContextTracker(model="claude-sonnet-4-20250514", threshold=0.50)
        shortterm = ShortTermMemory(anima_dir)

        # First response crosses threshold
        resp_threshold = _make_response(
            [_make_text_block("Partial")],
            input_tokens=150_000,  # crosses 50% of 200k
            output_tokens=10_000,
        )
        # Second response is final
        resp_final = _make_response(
            [_make_text_block("Continued")],
            input_tokens=1000,
            output_tokens=100,
        )

        with patch("anthropic.AsyncAnthropic") as mock_cls, \
             patch("core.execution.anthropic_fallback.build_system_prompt", return_value="sys"), \
             patch("core.execution._session.inject_shortterm", return_value="sys+st"), \
             patch("core.execution._session.load_prompt", return_value="continue"):
            mock_client = MagicMock()
            mock_client.messages = MagicMock()
            mock_client.messages.create = AsyncMock(
                side_effect=[resp_threshold, resp_final],
            )
            mock_cls.return_value = mock_client

            result = await executor.execute(
                "test", system_prompt="sys", tracker=tracker, shortterm=shortterm,
            )

        assert "Continued" in result.text or "Partial" in result.text

    async def test_chaining_limited_by_max_chains(
        self, executor: AnthropicFallbackExecutor, anima_dir: Path,
    ):
        executor._model_config.max_chains = 0  # No chaining allowed
        tracker = ContextTracker(model="claude-sonnet-4-20250514", threshold=0.50)
        shortterm = ShortTermMemory(anima_dir)

        # Response crosses threshold but chaining is not allowed
        resp = _make_response(
            [_make_text_block("No chain")],
            input_tokens=150_000,
            output_tokens=10_000,
        )

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=resp)
            mock_cls.return_value = mock_client

            result = await executor.execute(
                "test", system_prompt="sys", tracker=tracker, shortterm=shortterm,
            )

        assert "No chain" in result.text

    async def test_no_tracker_skips_tracking(self, executor: AnthropicFallbackExecutor):
        resp = _make_response([_make_text_block("Response")])

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=resp)
            mock_cls.return_value = mock_client

            result = await executor.execute("test", system_prompt="sys", tracker=None)

        assert result.text == "Response"
