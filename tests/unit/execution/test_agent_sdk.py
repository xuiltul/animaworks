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
    MockResultMessage,
    MockStreamEvent,
    MockTextBlock,
    MockToolUseBlock,
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
            # A1 mode does NOT pass ANTHROPIC_API_KEY (uses subscription auth)
            assert "ANTHROPIC_API_KEY" not in env
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
                assert "ANTHROPIC_API_KEY" not in env

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


# ── _BASH_SEND_RE regex ──────────────────────────────────


class TestBashSendRegex:
    """Verify _BASH_SEND_RE correctly detects send commands."""

    def test_bash_send_matches(self):
        from core.execution.agent_sdk import _BASH_SEND_RE

        m = _BASH_SEND_RE.match('bash send kotoha "Hello"')
        assert m is not None
        assert m.group(1) == "kotoha"

    def test_send_without_bash_prefix(self):
        from core.execution.agent_sdk import _BASH_SEND_RE

        m = _BASH_SEND_RE.match('send kotoha "Hello"')
        assert m is not None
        assert m.group(1) == "kotoha"

    def test_send_with_leading_whitespace(self):
        from core.execution.agent_sdk import _BASH_SEND_RE

        m = _BASH_SEND_RE.match('  send kotoha "Hello"')
        assert m is not None
        assert m.group(1) == "kotoha"

    def test_echo_send_does_not_match(self):
        from core.execution.agent_sdk import _BASH_SEND_RE

        m = _BASH_SEND_RE.match('echo "send kotoha hello"')
        assert m is None

    def test_cat_send_does_not_match(self):
        from core.execution.agent_sdk import _BASH_SEND_RE

        m = _BASH_SEND_RE.match("cat send_script.sh")
        assert m is None

    def test_grep_send_does_not_match(self):
        from core.execution.agent_sdk import _BASH_SEND_RE

        m = _BASH_SEND_RE.match("grep send file.txt")
        assert m is None

    def test_send_no_recipient_does_not_match(self):
        """send with no args should not match (needs recipient + space after)."""
        from core.execution.agent_sdk import _BASH_SEND_RE

        m = _BASH_SEND_RE.match("send")
        assert m is None


# ── _check_unconfirmed_sends ─────────────────────────────


class TestCheckUnconfirmedSends:
    """Verify unconfirmed send detection logic."""

    def _make_executor(self, tmp_path):
        from core.execution.agent_sdk import AgentSDKExecutor
        from core.schemas import ModelConfig
        from tests.helpers.mocks import patch_agent_sdk

        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)

        mc = ModelConfig(model="claude-sonnet-4-20250514", api_key="sk-test")
        with patch_agent_sdk():
            executor = AgentSDKExecutor(model_config=mc, anima_dir=anima_dir)
        return executor

    def test_no_pending_returns_empty(self, tmp_path):
        executor = self._make_executor(tmp_path)
        result = executor._check_unconfirmed_sends([], set())
        assert result == []

    def test_all_confirmed_returns_empty(self, tmp_path):
        executor = self._make_executor(tmp_path)
        pending = [{"to": "kotoha", "command": "send kotoha hello"}]
        confirmed = {"kotoha"}
        result = executor._check_unconfirmed_sends(pending, confirmed)
        assert result == []

    def test_unconfirmed_detected(self, tmp_path):
        executor = self._make_executor(tmp_path)
        pending = [
            {"to": "kotoha", "command": 'send kotoha "hello"'},
            {"to": "rin", "command": 'send rin "task"'},
        ]
        confirmed = {"kotoha"}  # only kotoha confirmed
        result = executor._check_unconfirmed_sends(pending, confirmed)
        assert len(result) == 1
        assert result[0]["to"] == "rin"

    def test_all_unconfirmed(self, tmp_path):
        executor = self._make_executor(tmp_path)
        pending = [{"to": "kotoha", "command": "send kotoha hello"}]
        confirmed = set()
        result = executor._check_unconfirmed_sends(pending, confirmed)
        assert len(result) == 1
        assert result[0]["to"] == "kotoha"

    def test_unconfirmed_logs_warning(self, tmp_path, caplog):
        import logging

        executor = self._make_executor(tmp_path)
        pending = [{"to": "rin", "command": 'send rin "msg"'}]

        with caplog.at_level(logging.WARNING, logger="animaworks.execution.agent_sdk"):
            executor._check_unconfirmed_sends(pending, set())

        assert any("Unconfirmed sends" in r.message for r in caplog.records)
        assert any("rin" in r.message for r in caplog.records)


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
