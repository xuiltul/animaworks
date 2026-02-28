from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Context management mixin for LiteLLMExecutor.

Handles LLM kwargs construction, initial message building, and
pre-flight context window clamping.
"""

import json as _json
import logging
from typing import Any

from core.prompt.context import resolve_context_window

logger = logging.getLogger("animaworks.execution.litellm_loop")


def _extract_tool_uses_from_messages(messages: list[dict]) -> list[dict]:
    """Extract tool_use info from LiteLLM-format messages."""
    tool_uses: list[dict] = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                tool_uses.append({
                    "name": fn.get("name", ""),
                    "input": fn.get("arguments", "")[:500],
                })
        elif msg.get("role") == "tool":
            if tool_uses:
                tool_uses[-1]["result"] = str(msg.get("content", ""))[:500]
    return tool_uses[-20:]  # Keep last 20 entries


class ContextMixin:
    """Mixin providing LLM kwargs, message building, and context clamping."""

    def _build_llm_kwargs(self) -> dict[str, Any]:
        """Credential + model kwargs for ``litellm.acompletion``."""
        from core.config.models import resolve_max_tokens
        from core.execution.base import is_adaptive_model, is_anthropic_claude, resolve_thinking_effort
        _eff_max = resolve_max_tokens(
            self._model_config.model,
            self._model_config.max_tokens,
            self._model_config.thinking,
        )
        kwargs: dict[str, Any] = {
            "model": self._model_config.model,
            "max_tokens": _eff_max,
            "timeout": self._resolve_llm_timeout(),
        }
        api_key = self._resolve_api_key()
        if api_key:
            kwargs["api_key"] = api_key
        if self._model_config.api_base_url:
            kwargs["api_base"] = self._model_config.api_base_url
        self._apply_provider_kwargs(kwargs)
        # Extended thinking / reasoning control
        if self._model_config.thinking is not None:
            model = self._model_config.model
            if model.startswith("bedrock/"):
                if self._model_config.thinking:
                    kwargs["reasoning_effort"] = resolve_thinking_effort(
                        model, self._model_config.thinking_effort,
                    )
            elif is_anthropic_claude(model):
                if self._model_config.thinking:
                    if is_adaptive_model(model):
                        kwargs["thinking"] = {"type": "adaptive"}
                        kwargs["reasoning_effort"] = resolve_thinking_effort(
                            model, self._model_config.thinking_effort,
                        )
                    else:
                        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}
                    kwargs["temperature"] = 1
            else:
                kwargs["think"] = self._model_config.thinking
        elif self._model_config.model.startswith("ollama/"):
            kwargs["think"] = False
        # Ollama num_ctx: explicitly set context window to prevent silent truncation
        if self._model_config.model.startswith("ollama/"):
            from core.config import load_config
            try:
                _cw_overrides = load_config().model_context_windows
            except Exception:
                logger.debug("Failed to load model context windows", exc_info=True)
                _cw_overrides = None
            kwargs["num_ctx"] = resolve_context_window(
                self._model_config.model, _cw_overrides,
            )
        return kwargs

    def _build_initial_messages(
        self,
        system_prompt: str,
        prompt: str,
        images: list[dict[str, Any]] | None,
        prior_messages: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build the initial messages list for an LLM call.

        When *prior_messages* is provided (structured conversation history
        with tool_use/tool_result blocks), they are inserted between the
        system prompt and the current user message.  The current user
        message is still appended as the final message.
        """
        msgs: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        if prior_messages:
            msgs.extend(prior_messages)
            if images and msgs and msgs[-1].get("role") == "user":
                last = msgs[-1]
                text = last["content"] if isinstance(last["content"], str) else ""
                content_parts: list[dict[str, Any]] = []
                for img in images:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{img['media_type']};base64,{img['data']}",
                        },
                    })
                content_parts.append({"type": "text", "text": text})
                msgs[-1] = {"role": "user", "content": content_parts}
            return msgs  # prior_messages already includes the current user msg
        if images:
            content_parts: list[dict[str, Any]] = []
            for img in images:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img['media_type']};base64,{img['data']}",
                    },
                })
            content_parts.append({"type": "text", "text": prompt})
            msgs.append({"role": "user", "content": content_parts})
        else:
            msgs.append({"role": "user", "content": prompt})
        return msgs

    def _preflight_clamp(
        self,
        llm_kwargs: dict[str, Any],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        litellm: Any,
    ) -> dict[str, Any] | None:
        """Pre-flight context window check, clamping max_tokens if needed.

        Returns the (possibly adjusted) kwargs dict, or ``None`` if the
        prompt is too large to fit in the context window at all.

        As a last resort, truncates the system message to fit within the
        context window rather than failing outright.
        """
        try:
            from core.config import load_config
            _cw_overrides = load_config().model_context_windows
        except Exception:
            logger.debug("Failed to load model context windows", exc_info=True)
            _cw_overrides = None
        ctx_window = resolve_context_window(
            self._model_config.model, _cw_overrides,
        )

        def _estimate_tokens() -> int:
            try:
                return litellm.token_counter(
                    model=self._model_config.model,
                    messages=messages,
                    tools=tools,
                )
            except Exception:
                logger.debug("Token counter fallback to char estimate", exc_info=True)
                msg_chars = sum(len(str(m.get("content", ""))) for m in messages)
                tool_chars = len(_json.dumps(tools)) if tools else 0
                return (msg_chars + tool_chars) // 2

        est_input = _estimate_tokens()
        available = ctx_window - est_input
        configured_max = llm_kwargs.get("max_tokens", 8192)

        if available < configured_max:
            if available - 128 >= 256:
                clamped = available - 128
                logger.info(
                    "Clamping max_tokens %d -> %d "
                    "(est_input ~%d, window %d)",
                    configured_max, clamped, est_input, ctx_window,
                )
                return {**llm_kwargs, "max_tokens": clamped}

            # Last resort: truncate system message to fit
            if (
                messages
                and messages[0].get("role") == "system"
                and isinstance(messages[0].get("content"), str)
            ):
                sys_content = messages[0]["content"]
                excess_tokens = est_input - ctx_window + 512
                excess_chars = excess_tokens * 4
                if len(sys_content) > excess_chars + 2000:
                    messages[0]["content"] = sys_content[
                        : len(sys_content) - excess_chars
                    ]
                    logger.warning(
                        "Hard-truncated system prompt by %d chars "
                        "to fit context window %d "
                        "(est_input ~%d, excess ~%d tokens)",
                        excess_chars, ctx_window, est_input, excess_tokens,
                    )
                    est_input = _estimate_tokens()
                    available = ctx_window - est_input
                    if available - 128 >= 256:
                        return {**llm_kwargs, "max_tokens": available - 128}

            logger.error(
                "Prompt too large for context window: "
                "~%d tokens input, %d window",
                est_input, ctx_window,
            )
            return None

        return llm_kwargs
