from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for AnthropicFallbackExecutor adaptive thinking kwargs.

Verifies that the blocking (execute) and streaming (execute_streaming)
paths pass the correct Anthropic SDK parameters for thinking/effort.
The Anthropic Python SDK uses ``output_config={"effort": ...}`` — NOT
``reasoning_effort`` (which is a LiteLLM abstraction).
"""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.schemas import ModelConfig


# ── Helpers ──────────────────────────────────────────────────


def _make_executor(
    model: str,
    thinking: bool | None = None,
    thinking_effort: str | None = None,
    anima_dir: Path | None = None,
):
    """Build an AnthropicFallbackExecutor with minimal mocks."""
    from core.execution.anthropic_fallback import AnthropicFallbackExecutor
    from core.memory import MemoryManager
    from core.tooling.handler import ToolHandler

    cfg = ModelConfig(
        model=model,
        thinking=thinking,
        thinking_effort=thinking_effort,
        api_key="test-key",
        max_tokens=8192,
    )
    _dir = anima_dir or Path("/tmp/test-anima")
    th = MagicMock(spec=ToolHandler)
    th._human_notifier = None
    mem = MagicMock(spec=MemoryManager)
    mem.read_permissions.return_value = ""
    mem.anima_dir = _dir

    return AnthropicFallbackExecutor(
        model_config=cfg,
        anima_dir=_dir,
        tool_handler=th,
        tool_registry=[],
        memory=mem,
    )


def _make_mock_response(stop_reason: str = "end_turn") -> MagicMock:
    """Create a minimal Anthropic Message response mock."""
    resp = MagicMock()
    resp.content = [MagicMock(type="text", text="OK")]
    resp.stop_reason = stop_reason
    resp.usage = MagicMock(input_tokens=100, output_tokens=50)
    resp.model = "claude-sonnet-4-6"
    return resp


# ── Blocking path tests ──────────────────────────────────────


class TestAnthropicBlockingThinkingKwargs:
    """Verify kwargs passed to client.messages.create() in execute()."""

    @pytest.mark.asyncio
    async def test_adaptive_model_uses_output_config(self):
        """Claude 4.6 + thinking → output_config, NOT reasoning_effort."""
        ex = _make_executor("claude-sonnet-4-6", thinking=True, thinking_effort="medium")

        mock_resp = _make_mock_response()
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        with patch.object(ex, "_build_client", return_value=mock_client), \
             patch.object(ex, "_build_tools", return_value=[]):
            await ex.execute(
                prompt="test",
                system_prompt="sys",
                max_turns_override=1,
            )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["thinking"] == {"type": "adaptive"}
        assert call_kwargs["output_config"] == {"effort": "medium"}
        assert "reasoning_effort" not in call_kwargs
        assert call_kwargs["temperature"] == 1

    @pytest.mark.asyncio
    async def test_old_claude_uses_manual_thinking(self):
        """Pre-4.6 Claude → thinking type=enabled with budget_tokens."""
        ex = _make_executor("claude-sonnet-4-5-20250929", thinking=True)

        mock_resp = _make_mock_response()
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        with patch.object(ex, "_build_client", return_value=mock_client), \
             patch.object(ex, "_build_tools", return_value=[]):
            await ex.execute(
                prompt="test",
                system_prompt="sys",
                max_turns_override=1,
            )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 10000}
        assert "output_config" not in call_kwargs
        assert "reasoning_effort" not in call_kwargs
        assert call_kwargs["temperature"] == 1

    @pytest.mark.asyncio
    async def test_no_thinking_no_extra_params(self):
        """thinking=None → no thinking/output_config/temperature params."""
        ex = _make_executor("claude-sonnet-4-6", thinking=None)

        mock_resp = _make_mock_response()
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        with patch.object(ex, "_build_client", return_value=mock_client), \
             patch.object(ex, "_build_tools", return_value=[]):
            await ex.execute(
                prompt="test",
                system_prompt="sys",
                max_turns_override=1,
            )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "thinking" not in call_kwargs
        assert "output_config" not in call_kwargs
        assert "reasoning_effort" not in call_kwargs
        assert "temperature" not in call_kwargs

    @pytest.mark.asyncio
    async def test_effort_default_high(self):
        """thinking_effort=None → defaults to 'high'."""
        ex = _make_executor("claude-opus-4-6", thinking=True, thinking_effort=None)

        mock_resp = _make_mock_response()
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        with patch.object(ex, "_build_client", return_value=mock_client), \
             patch.object(ex, "_build_tools", return_value=[]):
            await ex.execute(
                prompt="test",
                system_prompt="sys",
                max_turns_override=1,
            )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["output_config"] == {"effort": "high"}

    @pytest.mark.asyncio
    async def test_effort_max_clamped_on_sonnet(self):
        """effort='max' on Sonnet 4.6 → clamped to 'high'."""
        ex = _make_executor("claude-sonnet-4-6", thinking=True, thinking_effort="max")

        mock_resp = _make_mock_response()
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        with patch.object(ex, "_build_client", return_value=mock_client), \
             patch.object(ex, "_build_tools", return_value=[]):
            await ex.execute(
                prompt="test",
                system_prompt="sys",
                max_turns_override=1,
            )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["output_config"] == {"effort": "high"}

    @pytest.mark.asyncio
    async def test_effort_max_preserved_on_opus(self):
        """effort='max' on Opus 4.6 → preserved as 'max'."""
        ex = _make_executor("claude-opus-4-6", thinking=True, thinking_effort="max")

        mock_resp = _make_mock_response()
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        with patch.object(ex, "_build_client", return_value=mock_client), \
             patch.object(ex, "_build_tools", return_value=[]):
            await ex.execute(
                prompt="test",
                system_prompt="sys",
                max_turns_override=1,
            )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["output_config"] == {"effort": "max"}
