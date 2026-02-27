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
        model="claude-sonnet-4-6",
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
        model_config.model = "anthropic/claude-sonnet-4-6"
        with patch_agent_sdk():
            from core.execution.agent_sdk import AgentSDKExecutor
            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            assert executor._resolve_agent_sdk_model() == "claude-sonnet-4-6"

    def test_resolve_agent_sdk_model_no_prefix(self, model_config, anima_dir):
        with patch_agent_sdk():
            from core.execution.agent_sdk import AgentSDKExecutor
            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            assert executor._resolve_agent_sdk_model() == "claude-sonnet-4-6"

    def test_build_env_api_direct(self, model_config, anima_dir):
        """mode_s_auth=api with api_key → API direct mode."""
        model_config.api_base_url = "https://custom.api"
        model_config.mode_s_auth = "api"
        with patch_agent_sdk():
            from core.execution.agent_sdk import AgentSDKExecutor
            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            env = executor._build_env()
            assert env["ANIMAWORKS_ANIMA_DIR"] == str(anima_dir)
            assert env["ANTHROPIC_API_KEY"] == "sk-test"
            assert env["ANTHROPIC_BASE_URL"] == "https://custom.api"
            assert "CLAUDE_CODE_USE_BEDROCK" not in env
            assert "CLAUDE_CODE_USE_VERTEX" not in env

    def test_build_env_disables_skill_improvement(self, model_config, anima_dir):
        with patch_agent_sdk():
            from core.execution.agent_sdk import AgentSDKExecutor
            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            env = executor._build_env()
            assert env.get("CLAUDE_CODE_DISABLE_SKILL_IMPROVEMENT") == "true"

    def test_build_env_max_plan(self, anima_dir):
        """mode_s_auth=None (default) → Max plan regardless of api_key."""
        config = ModelConfig(model="claude-sonnet-4-6", api_key="sk-test")
        with patch_agent_sdk():
            from core.execution.agent_sdk import AgentSDKExecutor
            executor = AgentSDKExecutor(model_config=config, anima_dir=anima_dir)
            env = executor._build_env()
            assert env["ANTHROPIC_API_KEY"] == ""
            assert "CLAUDE_CODE_USE_BEDROCK" not in env
            assert "CLAUDE_CODE_USE_VERTEX" not in env

    def test_build_env_max_plan_explicit(self, anima_dir):
        """mode_s_auth='max' → Max plan explicitly."""
        config = ModelConfig(model="claude-sonnet-4-6", api_key="sk-test", mode_s_auth="max")
        with patch_agent_sdk():
            from core.execution.agent_sdk import AgentSDKExecutor
            executor = AgentSDKExecutor(model_config=config, anima_dir=anima_dir)
            env = executor._build_env()
            assert env["ANTHROPIC_API_KEY"] == ""

    def test_build_env_bedrock(self, anima_dir):
        """mode_s_auth=bedrock → Bedrock mode."""
        config = ModelConfig(
            model="claude-sonnet-4-6",
            api_key=None,
            mode_s_auth="bedrock",
            extra_keys={
                "aws_access_key_id": "AKIA_TEST",
                "aws_secret_access_key": "secret_test",
                "aws_region_name": "us-east-1",
            },
        )
        with patch_agent_sdk():
            from core.execution.agent_sdk import AgentSDKExecutor
            executor = AgentSDKExecutor(model_config=config, anima_dir=anima_dir)
            env = executor._build_env()
            assert env["ANTHROPIC_API_KEY"] == ""
            assert env["CLAUDE_CODE_USE_BEDROCK"] == "1"
            assert env["AWS_ACCESS_KEY_ID"] == "AKIA_TEST"
            assert env["AWS_SECRET_ACCESS_KEY"] == "secret_test"
            assert env["AWS_REGION"] == "us-east-1"
            assert "CLAUDE_CODE_USE_VERTEX" not in env

    def test_build_env_vertex(self, anima_dir):
        """mode_s_auth=vertex → Vertex AI mode."""
        config = ModelConfig(
            model="claude-sonnet-4-6",
            api_key=None,
            mode_s_auth="vertex",
            extra_keys={
                "vertex_project": "my-gcp-project",
                "vertex_location": "us-central1",
            },
        )
        with patch_agent_sdk():
            from core.execution.agent_sdk import AgentSDKExecutor
            executor = AgentSDKExecutor(model_config=config, anima_dir=anima_dir)
            env = executor._build_env()
            assert env["ANTHROPIC_API_KEY"] == ""
            assert env["CLAUDE_CODE_USE_VERTEX"] == "1"
            assert env["CLOUD_ML_PROJECT_ID"] == "my-gcp-project"
            assert env["CLOUD_ML_REGION"] == "us-central1"
            assert "CLAUDE_CODE_USE_BEDROCK" not in env

    def test_build_env_api_with_no_key_falls_back_to_max(self, anima_dir):
        """mode_s_auth=api but no api_key → falls back to Max plan."""
        config = ModelConfig(
            model="claude-sonnet-4-6",
            api_key=None,
            api_key_env="NONEXISTENT_XYZ",
            mode_s_auth="api",
        )
        with patch_agent_sdk():
            from core.execution.agent_sdk import AgentSDKExecutor
            executor = AgentSDKExecutor(model_config=config, anima_dir=anima_dir)
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("NONEXISTENT_XYZ", None)
                env = executor._build_env()
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
        tracker = ContextTracker(model="claude-sonnet-4-6", threshold=0.50)

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
        tracker = ContextTracker(model="claude-sonnet-4-6")

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
        tracker = ContextTracker(model="claude-sonnet-4-6")

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

    async def test_streaming_skips_historical_messages_before_stream_event(
        self, model_config, anima_dir,
    ):
        """セッション再開時の再送 AssistantMessage/UserMessage は
        最初の StreamEvent が届くまでスキップされることを確認する。"""
        import sys
        from contextlib import contextmanager
        from unittest.mock import MagicMock

        from tests.helpers.mocks import (
            MockAssistantMessage,
            MockClaudeSDKClient,
            MockResultMessage,
            MockStreamEvent,
            MockTextBlock,
            MockToolResultBlock,
            MockUserMessage,
        )

        @contextmanager
        def _patch_with_historical_messages():
            # 再送シーケンス: historical AssistantMessage → StreamEvent → AssistantMessage → ResultMessage
            historical_msg = MockAssistantMessage([MockTextBlock("historical old response")])
            historical_user = MockUserMessage([MockToolResultBlock("tu_old", "old result")])
            stream_event = MockStreamEvent({
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "new response"},
                "index": 0,
            })
            current_msg = MockAssistantMessage([MockTextBlock("new response")])
            result_msg = MockResultMessage(usage={"input_tokens": 100, "output_tokens": 50})

            messages = [historical_msg, historical_user, stream_event, current_msg, result_msg]

            def _client_factory(**kwargs):
                return MockClaudeSDKClient(messages=messages)

            mock_module = MagicMock()
            mock_module.ClaudeSDKClient = _client_factory
            mock_module.AssistantMessage = MockAssistantMessage
            mock_module.ResultMessage = MockResultMessage
            mock_module.TextBlock = MockTextBlock
            mock_module.ToolUseBlock = MagicMock
            mock_module.ToolResultBlock = MockToolResultBlock
            mock_module.UserMessage = MockUserMessage
            mock_module.SystemMessage = MagicMock
            mock_module.ClaudeAgentOptions = MagicMock
            mock_module.HookMatcher = MagicMock

            mock_types = MagicMock()
            mock_types.StreamEvent = MockStreamEvent
            mock_module.types = mock_types

            saved_modules = {}
            for key in ["claude_agent_sdk", "claude_agent_sdk.types"]:
                saved_modules[key] = sys.modules.get(key)
                sys.modules[key] = mock_types if key == "claude_agent_sdk.types" else mock_module
            try:
                yield mock_module
            finally:
                for key, saved in saved_modules.items():
                    if saved is None:
                        sys.modules.pop(key, None)
                    else:
                        sys.modules[key] = saved

        from core.prompt.context import ContextTracker
        tracker = ContextTracker(model="claude-sonnet-4-6")

        with _patch_with_historical_messages():
            from core.execution.agent_sdk import AgentSDKExecutor
            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)

            events = []
            async for event in executor.execute_streaming(
                system_prompt="sys",
                prompt="test",
                tracker=tracker,
            ):
                events.append(event)

        # text_delta は新しい応答のみ届く
        text_events = [e for e in events if e["type"] == "text_delta"]
        assert len(text_events) == 1
        assert text_events[0]["text"] == "new response"

        # done の full_text に historical テキストが混入しない
        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1
        assert "historical" not in done_events[0]["full_text"]
        assert "new response" in done_events[0]["full_text"]

    async def test_streaming_include_partial_messages_true(self, model_config, anima_dir):
        """execute_streaming() が _build_sdk_options に include_partial_messages=True を
        渡すことを確認する（ストリーミング StreamEvent 発行のために必須）。"""
        captured_kwargs: list[dict] = []

        original_build = None

        def _capturing_build(self_inner, *args, **kwargs):
            captured_kwargs.append(kwargs)
            return original_build(self_inner, *args, **kwargs)

        from core.prompt.context import ContextTracker
        tracker = ContextTracker(model="claude-sonnet-4-6")

        with patch_agent_sdk_streaming(text_deltas=["hi"]):
            from core.execution.agent_sdk import AgentSDKExecutor
            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            original_build = AgentSDKExecutor._build_sdk_options

            with patch.object(AgentSDKExecutor, "_build_sdk_options", _capturing_build):
                async for _ in executor.execute_streaming(
                    system_prompt="sys",
                    prompt="test",
                    tracker=tracker,
                ):
                    pass

        assert len(captured_kwargs) >= 1, "build_sdk_options が呼ばれていない"
        for call_kwargs in captured_kwargs:
            assert call_kwargs.get("include_partial_messages") is True, (
                f"include_partial_messages=True が渡されていない: {call_kwargs}"
            )


# ── Image input (multimodal) ──────────────────────────────────


class TestAgentSDKImageInput:
    """Mode S image input via _build_sdk_query_input / _image_prompt_messages."""

    def _make_executor(self, model_config, anima_dir, **kwargs):
        with patch_agent_sdk():
            from core.execution.agent_sdk import AgentSDKExecutor
            return AgentSDKExecutor(
                model_config=model_config,
                anima_dir=anima_dir,
                **kwargs,
            )

    async def test_build_sdk_query_input_text_only(self, model_config, anima_dir):
        """Text-only prompt returns a plain string."""
        with patch_agent_sdk():
            from core.execution.agent_sdk import _build_sdk_query_input
            result = _build_sdk_query_input("hello", None)
            assert isinstance(result, str)
            assert result == "hello"

    async def test_build_sdk_query_input_with_images(self, model_config, anima_dir):
        """Images present → returns an async generator with content blocks."""
        with patch_agent_sdk():
            from core.execution.agent_sdk import _build_sdk_query_input
            images = [{"media_type": "image/jpeg", "data": "dGVzdA=="}]
            result = _build_sdk_query_input("describe this", images)
            assert not isinstance(result, str)

            # Consume the async generator
            messages = []
            async for msg in result:
                messages.append(msg)

            assert len(messages) == 1
            msg = messages[0]
            assert msg["type"] == "user"
            content = msg["message"]["content"]
            assert len(content) == 2
            assert content[0]["type"] == "image"
            assert content[0]["source"]["media_type"] == "image/jpeg"
            assert content[0]["source"]["data"] == "dGVzdA=="
            assert content[1]["type"] == "text"
            assert content[1]["text"] == "describe this"

    async def test_build_sdk_query_input_multiple_images(self, model_config, anima_dir):
        """Multiple images produce multiple image blocks before the text block."""
        with patch_agent_sdk():
            from core.execution.agent_sdk import _build_sdk_query_input
            images = [
                {"media_type": "image/png", "data": "aW1nMQ=="},
                {"media_type": "image/jpeg", "data": "aW1nMg=="},
            ]
            result = _build_sdk_query_input("compare these", images)
            messages = []
            async for msg in result:
                messages.append(msg)

            content = messages[0]["message"]["content"]
            assert len(content) == 3
            assert content[0]["type"] == "image"
            assert content[1]["type"] == "image"
            assert content[2]["type"] == "text"

    async def test_execute_with_images_passes_to_query(self, model_config, anima_dir):
        """execute() with images should build multimodal prompt (no warning log)."""
        with patch_agent_sdk(response_text="I see a cat"):
            from core.execution.agent_sdk import AgentSDKExecutor
            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            images = [{"media_type": "image/jpeg", "data": "dGVzdA=="}]
            result = await executor.execute("What is this?", system_prompt="sys", images=images)
            assert "I see a cat" in result.text

    async def test_streaming_with_images_passes_to_query(self, model_config, anima_dir):
        """execute_streaming() with images should build multimodal prompt."""
        from core.prompt.context import ContextTracker
        tracker = ContextTracker(model="claude-sonnet-4-6")

        with patch_agent_sdk_streaming(text_deltas=["I see ", "a cat"]):
            from core.execution.agent_sdk import AgentSDKExecutor
            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            images = [{"media_type": "image/jpeg", "data": "dGVzdA=="}]

            events = []
            async for event in executor.execute_streaming(
                system_prompt="sys",
                prompt="What is this?",
                tracker=tracker,
                images=images,
            ):
                events.append(event)

            text_events = [e for e in events if e["type"] == "text_delta"]
            assert len(text_events) >= 2
            done_events = [e for e in events if e["type"] == "done"]
            assert len(done_events) == 1

    async def test_build_sdk_query_input_empty_images(self, model_config, anima_dir):
        """Empty images list is treated as text-only."""
        with patch_agent_sdk():
            from core.execution.agent_sdk import _build_sdk_query_input
            result = _build_sdk_query_input("hello", [])
            assert isinstance(result, str)


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
