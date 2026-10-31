"""Unit tests for context-overflow recovery, proactive compaction, and
DeepSeek thinking wiring (2026-08-18)."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from core.execution.loop_guards import call_llm_with_retry
from core.llm.guard.error_classifier import FailoverReason, RecoveryHint

# ── call_llm_with_retry: on_context_overflow ─────────────────


def _overflow_classify(exc: Exception):
    return (
        FailoverReason.CONTEXT_OVERFLOW,
        RecoveryHint(retryable=False, fallback_ok=True, backoff_s=None, is_terminal=False),
    )


class TestOnContextOverflow:
    @pytest.mark.asyncio
    async def test_compact_then_retry_succeeds(self):
        calls = {"n": 0}

        async def factory():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("maximum context length exceeded")
            return "ok"

        compact = AsyncMock(return_value=True)
        result = await call_llm_with_retry(
            factory,
            classify=_overflow_classify,
            next_backoff=lambda p: 0.0,
            on_context_overflow=compact,
            sleep=AsyncMock(),
        )
        assert result == "ok"
        compact.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_overflow_callback_used_at_most_once(self):
        async def factory():
            raise RuntimeError("maximum context length exceeded")

        compact = AsyncMock(return_value=True)
        with pytest.raises(RuntimeError):
            await call_llm_with_retry(
                factory,
                classify=_overflow_classify,
                next_backoff=lambda p: 0.0,
                on_context_overflow=compact,
                sleep=AsyncMock(),
            )
        compact.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_compact_failure_raises_original(self):
        async def factory():
            raise RuntimeError("maximum context length exceeded")

        compact = AsyncMock(return_value=False)
        with pytest.raises(RuntimeError):
            await call_llm_with_retry(
                factory,
                classify=_overflow_classify,
                next_backoff=lambda p: 0.0,
                on_context_overflow=compact,
                sleep=AsyncMock(),
            )
        compact.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_overflow_error_ignores_callback(self):
        def classify(exc):
            return (
                FailoverReason.INVALID_REQUEST,
                RecoveryHint(retryable=False, fallback_ok=False, backoff_s=None, is_terminal=True),
            )

        async def factory():
            raise RuntimeError("bad request")

        compact = AsyncMock(return_value=True)
        with pytest.raises(RuntimeError):
            await call_llm_with_retry(
                factory,
                classify=classify,
                next_backoff=lambda p: 0.0,
                on_context_overflow=compact,
                sleep=AsyncMock(),
            )
        compact.assert_not_awaited()


# ── ContextMixin: keep-recent compaction + deepseek thinking ──


def _make_mixin(model: str = "openai/deepseek-v4-flash", thinking: bool | None = True, effort: str | None = "high"):
    from core.execution.engines.litellm._litellm_context import ContextMixin

    obj = ContextMixin()
    obj._model_config = SimpleNamespace(
        model=model,
        thinking=thinking,
        thinking_effort=effort,
        max_tokens=65536,
        api_base_url=None,
        frequency_penalty=None,
        presence_penalty=None,
    )
    obj._resolve_cw = lambda: 1_048_576
    return obj


def _messages(n_pairs: int) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]
    for i in range(n_pairs):
        msgs.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": f"t{i}", "arguments": "{}"}}],
            }
        )
        msgs.append({"role": "tool", "content": f"result-{i}"})
    return msgs


class TestKeepRecentCompaction:
    @pytest.mark.asyncio
    async def test_recent_tail_preserved(self):
        obj = _make_mixin()
        msgs = _messages(10)  # 22 messages total
        orig_last = msgs[-1]

        fake_resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="SUMMARY"))])
        litellm = SimpleNamespace(acompletion=AsyncMock(return_value=fake_resp))

        ok = await obj._try_compact_messages(msgs, {"model": "openai/deepseek-v4-flash"}, litellm)
        assert ok
        # system + user + summary + tail
        assert msgs[0]["role"] == "system"
        assert msgs[1]["content"] == "task"
        assert "SUMMARY" in msgs[2]["content"]
        assert msgs[-1] is orig_last
        # tail must not start with an orphaned tool message
        assert msgs[3]["role"] != "tool"
        assert len(msgs) < 22

    @pytest.mark.asyncio
    async def test_short_history_falls_back_to_full_summarization(self):
        obj = _make_mixin()
        msgs = _messages(1)  # 4 messages — too short to keep a tail
        fake_resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="SUMMARY"))])
        litellm = SimpleNamespace(acompletion=AsyncMock(return_value=fake_resp))
        ok = await obj._try_compact_messages(msgs, {"model": "m"}, litellm)
        assert ok
        assert len(msgs) == 3  # system + user + summary, no tail


class TestDeepseekThinkingKwargs:
    def _kwargs(self, thinking, effort="high"):
        obj = _make_mixin(thinking=thinking, effort=effort)
        obj._resolve_num_retries = lambda: 2
        obj._resolve_api_key = lambda: "k"
        obj._apply_provider_kwargs = lambda kw: None
        with patch(
            "core.config.model_mode._match_models_json",
            return_value={
                "mode": "A",
                "context_window": 1_048_576,
                "thinking_format": "deepseek",
                "tool_calling": {"stream": True},
            },
        ):
            return obj._build_llm_kwargs()

    def test_thinking_true_sends_deepseek_format(self):
        kw = self._kwargs(thinking=True)
        assert kw["extra_body"]["thinking"] == {"type": "enabled"}
        assert kw["extra_body"]["chat_template_kwargs"]["thinking"] is True
        assert kw["extra_body"]["reasoning_effort"] == "high"
        assert "enable_thinking" not in kw["extra_body"]

    def test_thinking_none_defaults_enabled(self):
        kw = self._kwargs(thinking=None)
        assert kw["extra_body"]["thinking"] == {"type": "enabled"}
        assert kw["extra_body"]["reasoning_effort"] == "high"

    def test_thinking_false_disabled(self):
        kw = self._kwargs(thinking=False)
        assert kw["extra_body"]["thinking"] == {"type": "disabled"}
        assert kw["extra_body"]["chat_template_kwargs"]["thinking"] is False
        assert "reasoning_effort" not in kw["extra_body"]


class TestFgTimeoutDefault:
    def test_default_is_120(self):
        from core.tooling.handler_exec import _FG_CMD_TIMEOUT_DEFAULT

        assert _FG_CMD_TIMEOUT_DEFAULT == 120
