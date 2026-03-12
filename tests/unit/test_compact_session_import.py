# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for compact_session import path (claude_agent_sdk vs claude_code_sdk)."""

from __future__ import annotations

import builtins
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.schemas import ModelConfig

# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        model="claude-sonnet-4-6",
        api_key="sk-test",
        max_turns=5,
        context_threshold=0.50,
    )


@pytest.fixture
def anima_dir_with_session(tmp_path: Path) -> Path:
    """Anima dir with a persisted session ID for compaction."""
    d = tmp_path / "animas" / "test"
    d.mkdir(parents=True)
    (d / "state").mkdir(parents=True)
    from core.execution._sdk_session import _save_session_id

    _save_session_id(d, "sess-compact-test", "chat", "default")
    return d


@contextmanager
def _patch_sdk_unavailable():
    """Simulate claude_agent_sdk not being available (ImportError on import)."""
    fake_module = type(sys)("claude_agent_sdk")

    def __getattr__(name: str) -> None:
        raise ImportError(f"cannot import name '{name}' from 'claude_agent_sdk'")

    fake_module.__getattr__ = __getattr__
    saved = sys.modules.get("claude_agent_sdk")
    sys.modules["claude_agent_sdk"] = fake_module
    try:
        yield
    finally:
        if saved is None:
            sys.modules.pop("claude_agent_sdk", None)
        else:
            sys.modules["claude_agent_sdk"] = saved


@contextmanager
def _patch_sdk_available(messages: list[Any] | None = None):
    """Patch claude_agent_sdk with a mock that provides ClaudeSDKClient and ClaudeAgentOptions."""
    from tests.helpers.mocks import (
        MockAssistantMessage,
        MockClaudeSDKClient,
        MockResultMessage,
        MockTextBlock,
    )

    if messages is None:
        result_msg = MockResultMessage(usage={"input_tokens": 0, "output_tokens": 0})
        result_msg.session_id = "sess-after-compact"
        messages = [MockAssistantMessage([MockTextBlock("")]), result_msg]

    async def _receive_messages(self: MockClaudeSDKClient) -> Any:
        for msg in self._messages:
            yield msg

    MockClaudeSDKClient.receive_messages = _receive_messages

    client_factory_mock = MagicMock(side_effect=lambda **kwargs: MockClaudeSDKClient(messages=messages, **kwargs))

    mock_module = MagicMock()
    mock_module.ClaudeSDKClient = client_factory_mock
    mock_module.ClaudeAgentOptions = MagicMock

    saved = sys.modules.get("claude_agent_sdk")
    sys.modules["claude_agent_sdk"] = mock_module
    try:
        yield mock_module
    finally:
        if saved is None:
            sys.modules.pop("claude_agent_sdk", None)
        else:
            sys.modules["claude_agent_sdk"] = saved


# ── Tests ───────────────────────────────────────────────────────


class TestCompactSessionImportPath:
    """compact_session imports from claude_agent_sdk, not claude_code_sdk."""

    @pytest.mark.asyncio
    async def test_imports_from_claude_agent_sdk_not_claude_code_sdk(
        self, model_config: ModelConfig, anima_dir_with_session: Path
    ) -> None:
        """compact_session attempts to import from claude_agent_sdk (not claude_code_sdk)."""
        import_calls: list[str] = []

        original_import = builtins.__import__

        def _spy_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if "claude" in name and "sdk" in name:
                import_calls.append(name)
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=_spy_import), _patch_sdk_available():
            from core.execution.agent_sdk import AgentSDKExecutor

            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir_with_session)
            result = await executor.compact_session(
                anima_dir=anima_dir_with_session,
                session_type="chat",
                thread_id="default",
            )

        assert result is True
        assert "claude_agent_sdk" in import_calls
        assert "claude_code_sdk" not in import_calls

    @pytest.mark.asyncio
    async def test_returns_false_when_sdk_unavailable(
        self, model_config: ModelConfig, anima_dir_with_session: Path
    ) -> None:
        """When claude_agent_sdk is not available, compact_session returns False gracefully."""
        with _patch_sdk_unavailable():
            from core.execution.agent_sdk import AgentSDKExecutor

            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir_with_session)
            result = await executor.compact_session(
                anima_dir=anima_dir_with_session,
                session_type="chat",
                thread_id="default",
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_uses_claude_sdk_client_not_claude_code_sdk_client(
        self, model_config: ModelConfig, anima_dir_with_session: Path
    ) -> None:
        """compact_session uses ClaudeSDKClient (not ClaudeCodeSDKClient)."""
        with _patch_sdk_available() as mock_module:
            from core.execution.agent_sdk import AgentSDKExecutor

            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir_with_session)
            result = await executor.compact_session(
                anima_dir=anima_dir_with_session,
                session_type="chat",
                thread_id="default",
            )

        assert result is True
        mock_module.ClaudeSDKClient.assert_called_once()
