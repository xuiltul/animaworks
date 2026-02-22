"""Tests for core.execution.agent_sdk — Mode A1: Claude Agent SDK executor."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.asyncio

from core.schemas import ModelConfig
from tests.helpers.mocks import (
    MockAssistantMessage,
    MockClaudeSDKClient,
    MockResultMessage,
    MockStreamEvent,
    MockSystemMessage,
    MockTextBlock,
    MockToolResultBlock,
    MockToolUseBlock,
    MockUserMessage,
    patch_agent_sdk,
    patch_agent_sdk_streaming,
)


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        model="claude-sonnet-4-20250514",
        api_key="sk-test",
        max_turns=5,
        context_threshold=0.50,
    )


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "test"
    d.mkdir(parents=True)
    return d


# ── AgentSDKExecutor ──────────────────────────────────────────


class TestAgentSDKExecutor:
    def _make_executor(self, model_config, anima_dir, **kwargs):
        with patch_agent_sdk():
            from core.execution.agent_sdk import AgentSDKExecutor
            return AgentSDKExecutor(
                model_config=model_config,
                anima_dir=anima_dir,
                **kwargs,
            )

    def test_resolve_agent_sdk_model_strips_prefix(self, model_config, anima_dir):
        model_config.model = "anthropic/claude-sonnet-4-20250514"
        with patch_agent_sdk():
            from core.execution.agent_sdk import AgentSDKExecutor
            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            assert executor._resolve_agent_sdk_model() == "claude-sonnet-4-20250514"

    def test_resolve_agent_sdk_model_no_prefix(self, model_config, anima_dir):
        with patch_agent_sdk():
            from core.execution.agent_sdk import AgentSDKExecutor
            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            assert executor._resolve_agent_sdk_model() == "claude-sonnet-4-20250514"

    def test_build_env(self, model_config, anima_dir):
        model_config.api_base_url = "https://custom.api"
        with patch_agent_sdk():
            from core.execution.agent_sdk import AgentSDKExecutor
            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            env = executor._build_env()
            assert env["ANIMAWORKS_ANIMA_DIR"] == str(anima_dir)
            # A1 mode blocks ANTHROPIC_API_KEY leaking (sets to empty string)
            assert env["ANTHROPIC_API_KEY"] == ""
            assert env["ANTHROPIC_BASE_URL"] == "https://custom.api"

    def test_build_env_disables_skill_improvement(self, model_config, anima_dir):
        with patch_agent_sdk():
            from core.execution.agent_sdk import AgentSDKExecutor
            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            env = executor._build_env()
            assert env.get("CLAUDE_CODE_DISABLE_SKILL_IMPROVEMENT") == "true"

    def test_build_env_no_api_key(self, anima_dir):
        config = ModelConfig(model="test", api_key=None, api_key_env="NONEXISTENT_XYZ")
        with patch_agent_sdk():
            from core.execution.agent_sdk import AgentSDKExecutor
            executor = AgentSDKExecutor(model_config=config, anima_dir=anima_dir)
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("NONEXISTENT_XYZ", None)
                env = executor._build_env()
                # A1 mode blocks ANTHROPIC_API_KEY leaking (sets to empty string)
                assert env["ANTHROPIC_API_KEY"] == ""

    async def test_execute_returns_text(self, model_config, anima_dir):
        with patch_agent_sdk(response_text="Hello from Agent SDK"):
            from core.execution.agent_sdk import AgentSDKExecutor
            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            result = await executor.execute("test prompt", system_prompt="sys")
            assert "Hello from Agent SDK" in result.text

    async def test_execute_returns_result_message(self, model_config, anima_dir):
        with patch_agent_sdk(
            response_text="Response",
            usage={"input_tokens": 500, "output_tokens": 100},
        ):
            from core.execution.agent_sdk import AgentSDKExecutor
            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            result = await executor.execute("test")
            assert result.result_message is not None
            assert result.result_message.usage["input_tokens"] == 500

    async def test_execute_empty_response(self, model_config, anima_dir):
        with patch_agent_sdk(response_text=""):
            from core.execution.agent_sdk import AgentSDKExecutor
            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            result = await executor.execute("test")
            # Empty text blocks produce "(no response)"
            # Actually empty string joined would be "", then or "(no response)"
            assert result.text == "(no response)" or result.text == ""

    async def test_execute_with_tracker(self, model_config, anima_dir):
        from core.prompt.context import ContextTracker
        tracker = ContextTracker(model="claude-sonnet-4-20250514", threshold=0.50)

        with patch_agent_sdk(
            response_text="tracked response",
            usage={"input_tokens": 1000, "output_tokens": 200},
        ):
            from core.execution.agent_sdk import AgentSDKExecutor
            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            result = await executor.execute("test", tracker=tracker)
            assert "tracked response" in result.text


# ── Streaming execution ──────────────────────────────────────


class TestAgentSDKExecutorStreaming:
    async def test_streaming_yields_text_deltas(self, model_config, anima_dir):
        from core.prompt.context import ContextTracker
        tracker = ContextTracker(model="claude-sonnet-4-20250514")

        with patch_agent_sdk_streaming(text_deltas=["Hello ", "World"]):
            from core.execution.agent_sdk import AgentSDKExecutor
            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)

            events = []
            async for event in executor.execute_streaming(
                system_prompt="sys",
                prompt="test",
                tracker=tracker,
            ):
                events.append(event)

            text_events = [e for e in events if e["type"] == "text_delta"]
            assert len(text_events) >= 2

            done_events = [e for e in events if e["type"] == "done"]
            assert len(done_events) == 1
            assert "Hello " in done_events[0]["full_text"] or "World" in done_events[0]["full_text"]

    async def test_streaming_done_has_result_message(self, model_config, anima_dir):
        from core.prompt.context import ContextTracker
        tracker = ContextTracker(model="claude-sonnet-4-20250514")

        with patch_agent_sdk_streaming(
            text_deltas=["test"],
            usage={"input_tokens": 500, "output_tokens": 100},
        ):
            from core.execution.agent_sdk import AgentSDKExecutor
            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)

            last_event = None
            async for event in executor.execute_streaming(
                system_prompt="sys",
                prompt="test",
                tracker=tracker,
            ):
                last_event = event

            assert last_event["type"] == "done"
            assert last_event["result_message"] is not None


# ── ExecutionResult.unconfirmed_sends ────────────────────


class TestExecutionResultUnconfirmedSends:
    """Verify the new field on ExecutionResult."""

    def test_default_empty(self):
        from core.execution.base import ExecutionResult

        result = ExecutionResult(text="hello")
        assert result.unconfirmed_sends == []

    def test_with_unconfirmed(self):
        from core.execution.base import ExecutionResult

        sends = [{"to": "kotoha", "command": "send kotoha msg"}]
        result = ExecutionResult(text="hello", unconfirmed_sends=sends)
        assert result.unconfirmed_sends == sends
        assert len(result.unconfirmed_sends) == 1
