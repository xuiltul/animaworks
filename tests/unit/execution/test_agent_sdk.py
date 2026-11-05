"""Tests for core.execution.engines.claude.executor — Mode A1: Claude Agent SDK executor."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.asyncio

from core.schemas import ModelConfig
from tests.helpers.mocks import (
    MockAssistantMessage,
    MockClaudeSDKClient,
    MockResultMessage,
    MockStreamEvent,
    MockTextBlock,
    MockToolResultBlock,
    MockToolUseBlock,
    MockUserMessage,
    patch_agent_sdk,
    patch_agent_sdk_streaming,
)


@contextmanager
def _patch_agent_sdk_sequences(message_sequences: list[list[Any]]):
    """Patch claude_agent_sdk so each client uses the next message sequence."""
    sequence_iter = iter(message_sequences)

    def _client_factory(**kwargs: Any) -> MockClaudeSDKClient:
        return MockClaudeSDKClient(messages=next(sequence_iter), **kwargs)

    mock_module = MagicMock()
    mock_module.ClaudeSDKClient = _client_factory
    mock_module.AssistantMessage = MockAssistantMessage
    mock_module.ResultMessage = MockResultMessage
    mock_module.TextBlock = MockTextBlock
    mock_module.ToolUseBlock = MockToolUseBlock
    mock_module.ToolResultBlock = MockToolResultBlock
    mock_module.UserMessage = MockUserMessage
    mock_module.SystemMessage = MagicMock
    mock_module.ClaudeAgentOptions = MagicMock
    mock_module.HookMatcher = MagicMock
    mock_module.ClaudeSDKError = Exception
    mock_module.ProcessError = Exception

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


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        model="claude-sonnet-4-6",
        api_key="sk-test",
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
            from core.execution.engines.claude.executor import AgentSDKExecutor

            return AgentSDKExecutor(
                model_config=model_config,
                anima_dir=anima_dir,
                **kwargs,
            )

    def test_resolve_agent_sdk_model_strips_prefix(self, model_config, anima_dir):
        model_config.model = "anthropic/claude-sonnet-4-6"
        with patch_agent_sdk():
            from core.execution.engines.claude.executor import AgentSDKExecutor

            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            assert executor._resolve_agent_sdk_model() == "claude-sonnet-4-6"

    def test_resolve_agent_sdk_model_no_prefix(self, model_config, anima_dir):
        with patch_agent_sdk():
            from core.execution.engines.claude.executor import AgentSDKExecutor

            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            assert executor._resolve_agent_sdk_model() == "claude-sonnet-4-6"

    def test_build_env_api_direct(self, model_config, anima_dir):
        """mode_s_auth=api with api_key → API direct mode."""
        model_config.api_base_url = "https://custom.api"
        model_config.mode_s_auth = "api"
        with patch_agent_sdk():
            from core.execution.engines.claude.executor import AgentSDKExecutor

            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            env = executor._build_env()
            assert env["ANIMAWORKS_ANIMA_DIR"] == str(anima_dir)
            assert env["ANTHROPIC_API_KEY"] == "sk-test"
            assert env["ANTHROPIC_BASE_URL"] == "https://custom.api"
            assert "CLAUDE_CODE_USE_BEDROCK" not in env
            assert "CLAUDE_CODE_USE_VERTEX" not in env

    def test_build_env_disables_skill_improvement(self, model_config, anima_dir):
        with patch_agent_sdk():
            from core.execution.engines.claude.executor import AgentSDKExecutor

            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            env = executor._build_env()
            assert env.get("CLAUDE_CODE_DISABLE_SKILL_IMPROVEMENT") == "true"
            assert env["BASH_DEFAULT_TIMEOUT_MS"] == "1200000"
            assert env["BASH_MAX_TIMEOUT_MS"] == "1200000"

    def test_build_env_max_plan(self, anima_dir):
        """mode_s_auth=None (default) → Max plan regardless of api_key."""
        config = ModelConfig(model="claude-sonnet-4-6", api_key="sk-test")
        with patch_agent_sdk():
            from core.execution.engines.claude.executor import AgentSDKExecutor

            executor = AgentSDKExecutor(model_config=config, anima_dir=anima_dir)
            env = executor._build_env()
            assert env["ANTHROPIC_API_KEY"] == ""
            assert "CLAUDE_CODE_USE_BEDROCK" not in env
            assert "CLAUDE_CODE_USE_VERTEX" not in env

    def test_build_env_max_plan_explicit(self, anima_dir):
        """mode_s_auth='max' → Max plan explicitly."""
        config = ModelConfig(model="claude-sonnet-4-6", api_key="sk-test", mode_s_auth="max")
        with patch_agent_sdk():
            from core.execution.engines.claude.executor import AgentSDKExecutor

            executor = AgentSDKExecutor(model_config=config, anima_dir=anima_dir)
            env = executor._build_env()
            assert env["ANTHROPIC_API_KEY"] == ""

    def test_build_env_max_plan_reads_subscription_token_secret(self, anima_dir):
        """keys.oauth_token_secret → CLAUDE_CODE_OAUTH_TOKEN from the enclave secrets store."""
        config = ModelConfig(
            model="claude-opus-5-5",
            api_key=None,
            mode_s_auth="max",
            extra_keys={"oauth_token_secret": "claude-oauth-token"},
        )
        with (
            patch_agent_sdk(),
            patch("core.enclave.secrets.read_enclave_secret", return_value="tok-123") as read_secret,
        ):
            from core.execution.engines.claude.executor import AgentSDKExecutor

            executor = AgentSDKExecutor(model_config=config, anima_dir=anima_dir)
            env = executor._build_env()
            read_secret.assert_called_once_with("claude-oauth-token")
            assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "tok-123"
            assert env["ANTHROPIC_API_KEY"] == ""

    def test_build_env_max_plan_without_token_secret_uses_login(self, anima_dir):
        """No oauth_token_secret → the CLI's own login, no token env."""
        config = ModelConfig(model="claude-opus-5-5", api_key=None, mode_s_auth="max")
        with patch_agent_sdk():
            from core.execution.engines.claude.executor import AgentSDKExecutor

            executor = AgentSDKExecutor(model_config=config, anima_dir=anima_dir)
            env = executor._build_env()
            assert "CLAUDE_CODE_OAUTH_TOKEN" not in env

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
            from core.execution.engines.claude.executor import AgentSDKExecutor

            executor = AgentSDKExecutor(model_config=config, anima_dir=anima_dir)
            env = executor._build_env()
            assert env["ANTHROPIC_API_KEY"] == ""
            assert env["CLAUDE_CODE_USE_BEDROCK"] == "1"
            assert env["AWS_ACCESS_KEY_ID"] == "AKIA_TEST"
            assert env["AWS_SECRET_ACCESS_KEY"] == "secret_test"
            assert env["AWS_REGION"] == "us-east-1"
            assert "CLAUDE_CODE_USE_VERTEX" not in env
            assert env["DISABLE_TELEMETRY"] == "1"
            assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"

    def test_build_env_bedrock_reads_enclave_secret(self, anima_dir):
        """keys.aws_secret → AWS keys come from the enclave secrets store."""
        config = ModelConfig(
            model="bedrock/jp.anthropic.claude-opus-5-5",
            api_key=None,
            mode_s_auth="bedrock",
            extra_keys={"aws_secret": "aws-test", "aws_region_name": "ap-northeast-1"},
        )
        secret = '{"aws_access_key_id": "AKIA_ENC", "aws_secret_access_key": "enc_secret"}'
        with (
            patch_agent_sdk(),
            patch("core.enclave.secrets.read_enclave_secret", return_value=secret) as read_secret,
        ):
            from core.execution.engines.claude.executor import AgentSDKExecutor

            executor = AgentSDKExecutor(model_config=config, anima_dir=anima_dir)
            env = executor._build_env()
            read_secret.assert_called_once_with("aws-test")
            assert env["AWS_ACCESS_KEY_ID"] == "AKIA_ENC"
            assert env["AWS_SECRET_ACCESS_KEY"] == "enc_secret"
            assert env["AWS_REGION"] == "ap-northeast-1"
            assert executor._resolve_agent_sdk_model() == "jp.anthropic.claude-opus-5-5"
            assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "jp.anthropic.claude-opus-5-5"

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
            from core.execution.engines.claude.executor import AgentSDKExecutor

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
            from core.execution.engines.claude.executor import AgentSDKExecutor

            executor = AgentSDKExecutor(model_config=config, anima_dir=anima_dir)
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("NONEXISTENT_XYZ", None)
                env = executor._build_env()
                assert env["ANTHROPIC_API_KEY"] == ""

    async def test_execute_returns_text(self, model_config, anima_dir):
        with patch_agent_sdk(response_text="Hello from Agent SDK"):
            from core.execution.engines.claude.executor import AgentSDKExecutor

            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            result = await executor.execute("test prompt", system_prompt="sys")
            assert "Hello from Agent SDK" in result.text

    async def test_execute_returns_result_message(self, model_config, anima_dir):
        with patch_agent_sdk(
            response_text="Response",
            usage={"input_tokens": 500, "output_tokens": 100},
        ):
            from core.execution.engines.claude.executor import AgentSDKExecutor

            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            result = await executor.execute("test")
            assert result.result_message is not None
            assert result.result_message.usage["input_tokens"] == 500

    async def test_execute_aggregates_streamed_result_metadata(self, model_config, anima_dir):
        from types import SimpleNamespace

        from core.execution.base import TokenUsage, ToolCallRecord
        from core.execution.engines.claude.executor import AgentSDKExecutor

        executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
        result_message = SimpleNamespace(session_id="aggregate-session", num_turns=3)
        streamed_record = {
            "tool_name": "Read",
            "tool_id": "tu_aggregate",
            "input_summary": "path=README.md",
            "result_summary": "contents",
            "is_error": False,
        }

        async def _events(**kwargs):
            assert kwargs["_aggregate_result"] is True
            yield {"type": "text_delta", "text": "incremental text"}
            yield {
                "type": "done",
                "full_text": "assembled response",
                "result_message": result_message,
                "replied_to_from_transcript": {"alice"},
                "tool_call_records": [streamed_record],
                "force_chain": True,
                "task_compact_requested": True,
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 12,
                    "cache_read_tokens": 13,
                    "cache_write_tokens": 14,
                },
            }

        with patch.object(executor, "execute_streaming", _events):
            result = await executor.execute("prompt", system_prompt="system")

        assert result.text == "assembled response"
        assert result.result_message is result_message
        assert result.replied_to_from_transcript == {"alice"}
        assert result.tool_call_records == [ToolCallRecord(**streamed_record)]
        assert result.force_chain is True
        assert result.task_compact_requested is True
        assert result.usage == TokenUsage(11, 12, 13, 14)

    async def test_execute_retries_failed_resume_with_fresh_session(self, model_config, anima_dir):
        from core.execution.engines.claude._sdk_session import _load_session_id, _save_session_id
        from core.execution.engines.claude.executor import AgentSDKExecutor

        fresh_messages = [
            MockAssistantMessage([MockTextBlock("fresh session response")]),
            MockResultMessage(session_id="fresh-session"),
        ]
        with _patch_agent_sdk_sequences([fresh_messages]) as sdk_module:
            default_factory = sdk_module.ClaudeSDKClient
            attempts = []

            class _FailedResumeClient:
                async def __aenter__(self):
                    raise RuntimeError("stale session")

                async def __aexit__(self, *args):
                    return False

            def _client_factory(**kwargs):
                attempts.append(kwargs)
                return _FailedResumeClient() if len(attempts) == 1 else default_factory(**kwargs)

            sdk_module.ClaudeSDKClient = _client_factory
            _save_session_id(anima_dir, "stale-session", "chat")
            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            result = await executor.execute("prompt", system_prompt="system")

        assert result.text == "fresh session response"
        assert len(attempts) == 2
        assert _load_session_id(anima_dir, "chat") == "fresh-session"

    async def test_execute_timeout_keeps_received_text_and_incomplete_tool_record(self, model_config, anima_dir):
        from core.execution.base import ToolCallRecord
        from core.execution.engines.claude.executor import AgentSDKExecutor

        async def _receive_then_timeout(self):
            yield MockAssistantMessage(
                [
                    MockTextBlock("received before timeout"),
                    MockToolUseBlock("Read", {"path": "partial.txt"}, id="tu_partial"),
                ]
            )
            raise TimeoutError("read timed out")

        with (
            _patch_agent_sdk_sequences([[]]),
            patch.object(MockClaudeSDKClient, "receive_messages", _receive_then_timeout),
        ):
            result = await AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir).execute(
                "prompt", system_prompt="system"
            )

        assert result.text == "[Agent SDK Error: read timed out]\nreceived before timeout"
        assert result.error is True
        assert result.tool_call_records == [
            ToolCallRecord(
                tool_name="Read",
                tool_id="tu_partial",
                input_summary="{'path': 'partial.txt'}",
                result_summary="",
                is_error=True,
            )
        ]

    async def test_execute_idle_timeout_preserves_buffered_assistant_text(self, model_config, anima_dir):
        from core.execution.engines.claude.executor import AgentSDKExecutor

        waiting_for_next_message = asyncio.Event()

        async def _receive_then_block(self):
            yield MockAssistantMessage([MockTextBlock("received before idle timeout")])
            waiting_for_next_message.set()
            await asyncio.Event().wait()

        with (
            _patch_agent_sdk_sequences([[]]),
            patch.object(MockClaudeSDKClient, "receive_messages", _receive_then_block),
            patch("core.execution.events.DEFAULT_EVENT_IDLE_TIMEOUT_SECONDS", 0.05),
        ):
            result = await AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir).execute(
                "prompt", system_prompt="system"
            )

        assert waiting_for_next_message.is_set()
        assert result.error is True
        assert result.text.endswith("\nreceived before idle timeout")

    async def test_execute_cancellation_propagates(self, model_config, anima_dir):
        from core.execution.engines.claude.executor import AgentSDKExecutor

        waiting_for_next_message = asyncio.Event()

        async def _receive_then_block(self):
            yield MockAssistantMessage([MockTextBlock("received before cancellation")])
            waiting_for_next_message.set()
            await asyncio.Event().wait()

        with (
            _patch_agent_sdk_sequences([[]]),
            patch.object(MockClaudeSDKClient, "receive_messages", _receive_then_block),
        ):
            task = asyncio.create_task(
                AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir).execute(
                    "prompt", system_prompt="system"
                )
            )
            await asyncio.wait_for(waiting_for_next_message.wait(), timeout=1.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    async def test_execute_auth_retry_failure_keeps_legacy_result_shape(self, model_config, anima_dir):
        from core.execution.engines.claude.executor import AgentSDKExecutor

        auth_text = 'Failed to authenticate. API Error: 401 {"type":"error","error":{"type":"authentication_error","message":"Invalid authentication credentials"}}'
        first_messages = [
            MockAssistantMessage([MockTextBlock(auth_text)]),
            MockResultMessage(usage={"input_tokens": 10, "output_tokens": 5}),
        ]
        with _patch_agent_sdk_sequences([first_messages]) as sdk_module:
            default_factory = sdk_module.ClaudeSDKClient
            attempts = []

            class _FailedRetryClient:
                async def __aenter__(self):
                    raise RuntimeError("retry failed")

                async def __aexit__(self, *args):
                    return False

            def _client_factory(**kwargs):
                attempts.append(kwargs)
                return default_factory(**kwargs) if len(attempts) == 1 else _FailedRetryClient()

            sdk_module.ClaudeSDKClient = _client_factory
            result = await AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir).execute(
                "prompt", system_prompt="system"
            )

        assert result.text == "[Agent SDK Error: retry failed]"
        assert result.error is True
        assert result.tool_call_records == []
        assert result.usage.input_tokens == 0

    async def test_execute_empty_response(self, model_config, anima_dir):
        with patch_agent_sdk(response_text=""):
            from core.execution.engines.claude.executor import AgentSDKExecutor

            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            result = await executor.execute("test")
            # Empty text blocks produce "(no response)"
            # Actually empty string joined would be "", then or "(no response)"
            assert result.text == "(no response)" or result.text == ""

    async def test_execute_retries_max_auth_failure_once(self, model_config, anima_dir):
        auth_text = 'Failed to authenticate. API Error: 401 {"type":"error","error":{"type":"authentication_error","message":"Invalid authentication credentials"}}'
        first_messages = [
            MockStreamEvent(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": auth_text},
                    "index": 0,
                }
            ),
            MockAssistantMessage([MockTextBlock(auth_text)]),
            MockResultMessage(usage={"input_tokens": 10, "output_tokens": 5}),
        ]
        second_messages = [
            MockAssistantMessage([MockTextBlock("Recovered response")]),
            MockResultMessage(usage={"input_tokens": 12, "output_tokens": 6}),
        ]

        with _patch_agent_sdk_sequences([first_messages, second_messages]):
            from core.execution.engines.claude.executor import AgentSDKExecutor

            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            result = await executor.execute("test", system_prompt="sys")

        assert result.text == "Recovered response"

    async def test_execute_does_not_retry_api_auth_failure(self, anima_dir):
        auth_text = 'Failed to authenticate. API Error: 401 {"type":"error","error":{"type":"authentication_error","message":"Invalid authentication credentials"}}'
        config = ModelConfig(
            model="claude-sonnet-4-6",
            api_key="sk-test",
            mode_s_auth="api",
        )
        first_messages = [
            MockAssistantMessage([MockTextBlock(auth_text)]),
            MockResultMessage(usage={"input_tokens": 10, "output_tokens": 5}),
        ]

        with _patch_agent_sdk_sequences([first_messages]):
            from core.execution.engines.claude.executor import AgentSDKExecutor

            executor = AgentSDKExecutor(model_config=config, anima_dir=anima_dir)
            result = await executor.execute("test", system_prompt="sys")

        assert auth_text in result.text
        assert result.error is True

    async def test_execute_with_tracker(self, model_config, anima_dir):
        from core.prompt.context import ContextTracker

        tracker = ContextTracker(model="claude-sonnet-4-6", threshold=0.50)

        with patch_agent_sdk(
            response_text="tracked response",
            usage={"input_tokens": 1000, "output_tokens": 200},
        ):
            from core.execution.engines.claude.executor import AgentSDKExecutor

            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            result = await executor.execute("test", tracker=tracker)
            assert "tracked response" in result.text

    async def test_execute_keeps_blocking_context_tracker_semantics(self, model_config, anima_dir):
        from core.execution.engines.claude.executor import AgentSDKExecutor
        from core.prompt.context import ContextTracker

        tracker = ContextTracker(model=model_config.model, threshold=0.5)
        messages = [
            MockStreamEvent(
                {
                    "type": "message_start",
                    "message": {"usage": {"input_tokens": 30, "cache_read_input_tokens": 4}},
                }
            ),
            MockAssistantMessage([MockTextBlock("response")]),
            MockResultMessage(usage={"input_tokens": 100, "output_tokens": 20}),
        ]
        with _patch_agent_sdk_sequences([messages]):
            result = await AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir).execute(
                "prompt", tracker=tracker
            )

        assert result.text == "response"
        assert tracker.baseline_tokens == 0
        assert tracker._input_tokens == 100
        assert tracker._output_tokens == 20


# ── Streaming execution ──────────────────────────────────────


class TestAgentSDKExecutorStreaming:
    async def test_streaming_yields_text_deltas(self, model_config, anima_dir):
        from core.prompt.context import ContextTracker

        tracker = ContextTracker(model="claude-sonnet-4-6")

        with patch_agent_sdk_streaming(text_deltas=["Hello ", "World"]):
            from core.execution.engines.claude.executor import AgentSDKExecutor

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
            from core.execution.engines.claude.executor import AgentSDKExecutor

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
        self,
        model_config,
        anima_dir,
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
            stream_event = MockStreamEvent(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "new response"},
                    "index": 0,
                }
            )
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
            from core.execution.engines.claude.executor import AgentSDKExecutor

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
            from core.execution.engines.claude.executor import AgentSDKExecutor

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

    async def test_streaming_falls_back_to_completed_messages_when_stream_events_missing(
        self,
        model_config,
        anima_dir,
    ):
        """StreamEvent が一度も来なくても completed AssistantMessage を採用する。"""
        import sys
        from contextlib import contextmanager
        from unittest.mock import MagicMock

        from tests.helpers.mocks import (
            MockAssistantMessage,
            MockClaudeSDKClient,
            MockResultMessage,
            MockTextBlock,
        )

        @contextmanager
        def _patch_without_stream_events():
            assistant_msg = MockAssistantMessage([MockTextBlock("reply from completed message")])
            result_msg = MockResultMessage(usage={"input_tokens": 120, "output_tokens": 30})
            messages = [assistant_msg, result_msg]

            def _client_factory(**kwargs):
                return MockClaudeSDKClient(messages=messages)

            mock_module = MagicMock()
            mock_module.ClaudeSDKClient = _client_factory
            mock_module.AssistantMessage = MockAssistantMessage
            mock_module.ResultMessage = MockResultMessage
            mock_module.TextBlock = MockTextBlock
            mock_module.ToolUseBlock = MagicMock
            mock_module.ToolResultBlock = MagicMock
            mock_module.UserMessage = MagicMock
            mock_module.SystemMessage = MagicMock
            mock_module.ClaudeAgentOptions = MagicMock
            mock_module.HookMatcher = MagicMock

            mock_types = MagicMock()
            mock_types.StreamEvent = MagicMock
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

        with _patch_without_stream_events():
            from core.execution.engines.claude.executor import AgentSDKExecutor

            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            events = []
            async for event in executor.execute_streaming(
                system_prompt="sys",
                prompt="test",
                tracker=tracker,
            ):
                events.append(event)

        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1
        assert done_events[0]["full_text"] == "reply from completed message"

    async def test_streaming_retries_max_auth_failure_without_text_deltas(
        self,
        model_config,
        anima_dir,
    ):
        from core.prompt.context import ContextTracker

        auth_text = 'Failed to authenticate. API Error: 401 {"type":"error","error":{"type":"authentication_error","message":"Invalid authentication credentials"}}'
        first_messages = [
            MockAssistantMessage([MockTextBlock(auth_text)]),
            MockResultMessage(usage={"input_tokens": 10, "output_tokens": 5}),
        ]
        second_messages = [
            MockStreamEvent(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "Recovered"},
                    "index": 0,
                }
            ),
            MockAssistantMessage([MockTextBlock("Recovered")]),
            MockResultMessage(usage={"input_tokens": 12, "output_tokens": 6}),
        ]

        tracker = ContextTracker(model="claude-sonnet-4-6")

        with _patch_agent_sdk_sequences([first_messages, second_messages]):
            from core.execution.engines.claude.executor import AgentSDKExecutor

            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            events = []
            async for event in executor.execute_streaming(
                system_prompt="sys",
                prompt="test",
                tracker=tracker,
            ):
                events.append(event)

        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1
        assert done_events[0]["full_text"] == "Recovered"


# ── Image input (multimodal) ──────────────────────────────────


@pytest.mark.parametrize("signal", ["result_error", "result_subtype", "assistant_error", "cli_envelope"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_sdk_provider_failures_are_not_successful_answers(model_config, anima_dir, signal, streaming):
    from core.execution.engines.claude.executor import AgentSDKExecutor
    from core.prompt.context import ContextTracker

    text = "API Error: ConnectionRefused: Unable to connect to the API"
    assistant = MockAssistantMessage([MockTextBlock(text)])
    result = MockResultMessage(usage={"input_tokens": 17, "output_tokens": 3})
    if signal == "result_error":
        result.is_error = True
        result.errors = [text]
        assistant.content = [MockTextBlock("Partial provider diagnostic")]
    elif signal == "result_subtype":
        result.subtype = "error_during_execution"
        result.result = text
    elif signal == "assistant_error":
        assistant.error = "server_error"
        assistant.content = [MockTextBlock("Provider cannot fulfill the request")]
    with _patch_agent_sdk_sequences([[assistant, result]]):
        executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
        if streaming:
            events = [
                event
                async for event in executor.execute_streaming("sys", "test", ContextTracker(model=model_config.model))
            ]
            assert not any(event["type"] == "done" for event in events)
            failure = events[-1]
            assert failure["type"] == "error" and failure["terminal"] is True
            assert failure["usage"]["input_tokens"] == 17
            if signal != "assistant_error":
                assert failure["reason"] == "network"
        else:
            output = await executor.execute("test")
            assert output.error is True
            assert output.usage.input_tokens == 17


@pytest.mark.parametrize(
    "text",
    [
        "The logs show API Error: ConnectionRefused; the application itself is healthy.",
        "The resource is forbidden for ordinary users; that is the intended permission rule.",
        'Example: "API Error: 503" is the message to look for.',
        "No error occurred.",
        "The log says Failed to authenticate. API Error: 401 authentication_error; update the application's credential.",
    ],
)
async def test_normal_sdk_answers_mentioning_errors_stay_successful(model_config, anima_dir, text):
    from core.execution.engines.claude.executor import AgentSDKExecutor
    from core.prompt.context import ContextTracker

    sequence = [MockAssistantMessage([MockTextBlock(text)]), MockResultMessage()]
    with _patch_agent_sdk_sequences([sequence, sequence]):
        executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
        result = await executor.execute("test", trigger="heartbeat")
        assert result.error is False
        events = [
            event
            async for event in executor.execute_streaming(
                "sys", "test", ContextTracker(model=model_config.model), trigger="heartbeat"
            )
        ]
        assert events[-1]["type"] == "done"
        assert events[-1]["full_text"] == text


async def test_structured_sdk_failure_without_assistant_text_uses_error_details(model_config, anima_dir):
    from core.execution.engines.claude.executor import AgentSDKExecutor

    result = MockResultMessage()
    result.is_error = True
    result.errors = ["API Error: ConnectionRefused"]
    with _patch_agent_sdk_sequences([[result]]):
        output = await AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir).execute("test")
    assert output.error is True
    assert output.text == "API Error: ConnectionRefused"


@pytest.mark.parametrize(
    "error,reason",
    [
        ("authentication_failed", "auth"),
        ("billing_error", "billing"),
        ("rate_limit", "rate_limit"),
        ("invalid_request", "invalid_request"),
        ("server_error", "server_error"),
    ],
)
async def test_structured_assistant_error_preserves_provider_reason(model_config, anima_dir, error, reason):
    from core.execution.engines.claude.executor import AgentSDKExecutor
    from core.prompt.context import ContextTracker

    assistant = MockAssistantMessage([MockTextBlock("Provider request failed")])
    assistant.error = error
    with _patch_agent_sdk_sequences([[assistant, MockResultMessage()]]):
        executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
        events = [
            event
            async for event in executor.execute_streaming(
                "sys", "test", ContextTracker(model=model_config.model), trigger="heartbeat"
            )
        ]
    assert events[-1]["type"] == "error"
    assert events[-1]["reason"] == reason


class TestAgentSDKImageInput:
    """Mode S image input via _build_sdk_query_input / _image_prompt_messages."""

    def _make_executor(self, model_config, anima_dir, **kwargs):
        with patch_agent_sdk():
            from core.execution.engines.claude.executor import AgentSDKExecutor

            return AgentSDKExecutor(
                model_config=model_config,
                anima_dir=anima_dir,
                **kwargs,
            )

    async def test_build_sdk_query_input_text_only(self, model_config, anima_dir):
        """Text-only prompt returns a plain string."""
        with patch_agent_sdk():
            from core.execution.engines.claude.executor import _build_sdk_query_input

            result = _build_sdk_query_input("hello", None)
            assert isinstance(result, str)
            assert result == "hello"

    async def test_build_sdk_query_input_with_images(self, model_config, anima_dir):
        """Images present → returns an async generator with content blocks."""
        with patch_agent_sdk():
            from core.execution.engines.claude.executor import _build_sdk_query_input

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
            from core.execution.engines.claude.executor import _build_sdk_query_input

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
            from core.execution.engines.claude.executor import AgentSDKExecutor

            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            images = [{"media_type": "image/jpeg", "data": "dGVzdA=="}]
            result = await executor.execute("What is this?", system_prompt="sys", images=images)
            assert "I see a cat" in result.text

    async def test_streaming_with_images_passes_to_query(self, model_config, anima_dir):
        """execute_streaming() with images should build multimodal prompt."""
        from core.prompt.context import ContextTracker

        tracker = ContextTracker(model="claude-sonnet-4-6")

        with patch_agent_sdk_streaming(text_deltas=["I see ", "a cat"]):
            from core.execution.engines.claude.executor import AgentSDKExecutor

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
            from core.execution.engines.claude.executor import _build_sdk_query_input

            result = _build_sdk_query_input("hello", [])
            assert isinstance(result, str)
