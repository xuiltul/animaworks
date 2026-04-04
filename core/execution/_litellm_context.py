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
import types
from typing import Any

from core.schemas import ImageData

logger = logging.getLogger("animaworks.execution.litellm_loop")

_COMPACT_MSG_TRUNCATE_CHARS = 2000
_COMPACT_TOOL_ARGS_TRUNCATE_CHARS = 200
_COMPACT_MIN_CONTEXT_WINDOW = 16_000
_COMPACT_MAX_SUMMARY_TOKENS = 4096


def _extract_tool_uses_from_messages(messages: list[dict]) -> list[dict]:
    """Extract tool_use info from LiteLLM-format messages."""
    tool_uses: list[dict] = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                tool_uses.append(
                    {
                        "name": fn.get("name", ""),
                        "input": fn.get("arguments", "")[:500],
                    }
                )
        elif msg.get("role") == "tool":
            if tool_uses:
                tool_uses[-1]["result"] = str(msg.get("content", ""))[:500]
    return tool_uses[-20:]  # Keep last 20 entries


class ContextMixin:
    """Mixin providing LLM kwargs, message building, and context clamping."""

    def _build_llm_kwargs(self) -> dict[str, Any]:
        """Credential + model kwargs for ``litellm.acompletion``."""
        from core.config.models import resolve_max_tokens
        from core.execution.base import (
            is_adaptive_model,
            is_anthropic_claude,
            is_bedrock_glm,
            is_bedrock_kimi,
            is_bedrock_qwen,
            resolve_thinking_effort,
        )

        _thinking = self._model_config.thinking
        if _thinking is None and self._model_config.model.startswith("openai/"):
            _thinking = True
        _eff_max = resolve_max_tokens(
            self._model_config.model,
            self._model_config.max_tokens,
            _thinking,
        )
        _model_name = self._model_config.model
        # nanoGPT: rewrite nanogpt/ prefix to openai/ for LiteLLM routing
        # (nanoGPT is an OpenAI-compatible API aggregator)
        if _model_name.startswith("nanogpt/"):
            _model_name = "openai/" + _model_name[len("nanogpt/") :]
        kwargs: dict[str, Any] = {
            "model": _model_name,
            "max_tokens": _eff_max,
            "timeout": self._resolve_llm_timeout(),
            "num_retries": self._resolve_num_retries(),
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
            if is_bedrock_kimi(model):
                # Kimi K2.5 on Bedrock: pass reasoning_config which LiteLLM
                # forwards to additionalModelRequestFields in Converse API.
                # Supported values: "high" (enables thinking mode).
                if self._model_config.thinking:
                    kwargs["reasoning_config"] = resolve_thinking_effort(
                        model,
                        self._model_config.thinking_effort,
                    )
            elif is_bedrock_qwen(model) or is_bedrock_glm(model):
                # Qwen / GLM on Bedrock: pass enable_thinking which LiteLLM
                # forwards to additionalModelRequestFields in the Converse API.
                # Only send when True — explicitly sending False causes some
                # models (e.g. qwen3-next) to degrade to "Please continue."
                if self._model_config.thinking:
                    kwargs["enable_thinking"] = True
            elif model.startswith("bedrock/"):
                if self._model_config.thinking:
                    kwargs["reasoning_effort"] = resolve_thinking_effort(
                        model,
                        self._model_config.thinking_effort,
                    )
            elif is_anthropic_claude(model):
                if self._model_config.thinking:
                    if is_adaptive_model(model):
                        kwargs["thinking"] = {"type": "adaptive"}
                        kwargs["reasoning_effort"] = resolve_thinking_effort(
                            model,
                            self._model_config.thinking_effort,
                        )
                    else:
                        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}
                    kwargs["temperature"] = 1
            elif model.startswith("openai/"):
                kwargs.setdefault("extra_body", {})
                kwargs["extra_body"]["enable_thinking"] = self._model_config.thinking
                kwargs["extra_body"].setdefault("chat_template_kwargs", {})
                kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] = self._model_config.thinking
            else:
                kwargs["think"] = self._model_config.thinking
        elif self._model_config.model.startswith("openai/"):
            kwargs.setdefault("extra_body", {})
            kwargs["extra_body"]["enable_thinking"] = True
            kwargs["extra_body"].setdefault("chat_template_kwargs", {})
            kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] = True
        elif self._model_config.model.startswith("ollama/"):
            kwargs["think"] = False
        # Ollama num_ctx: explicitly set context window to prevent silent truncation
        # Also set keep_alive so the model does not monopolise VRAM indefinitely between
        # requests on shared hardware (multiple animas using different models).
        if self._model_config.model.startswith("ollama/"):
            kwargs["num_ctx"] = self._resolve_cw()
            try:
                from core.config import load_config as _lc

                _keep = _lc().server.ollama_keep_alive
            except Exception:
                _keep = ""
            if _keep:
                kwargs.setdefault("extra_body", {})
                kwargs["extra_body"]["keep_alive"] = _keep
        # ── Repetition penalty parameters ──
        from core.config.models import resolve_penalties

        penalties = resolve_penalties(self._model_config.model)
        if self._model_config.frequency_penalty is not None:
            penalties["frequency_penalty"] = self._model_config.frequency_penalty
        if self._model_config.presence_penalty is not None:
            penalties["presence_penalty"] = self._model_config.presence_penalty
        kwargs.update(penalties)
        return kwargs

    def _build_initial_messages(
        self,
        system_prompt: str,
        prompt: str,
        images: list[ImageData] | None,
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
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{img['media_type']};base64,{img['data']}",
                            },
                        }
                    )
                content_parts.append({"type": "text", "text": text})
                msgs[-1] = {"role": "user", "content": content_parts}
            return msgs  # prior_messages already includes the current user msg
        if images:
            content_parts: list[dict[str, Any]] = []
            for img in images:
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{img['media_type']};base64,{img['data']}",
                        },
                    }
                )
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
        litellm: types.ModuleType,
    ) -> dict[str, Any] | None:
        """Pre-flight context window check, clamping max_tokens if needed.

        Returns the (possibly adjusted) kwargs dict, or ``None`` if the
        prompt is too large to fit in the context window at all.

        As a last resort, truncates the system message to fit within the
        context window rather than failing outright.
        """
        ctx_window = self._resolve_cw()

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
                    "Clamping max_tokens %d -> %d (est_input ~%d, window %d)",
                    configured_max,
                    clamped,
                    est_input,
                    ctx_window,
                )
                return {**llm_kwargs, "max_tokens": clamped}

            # Last resort: truncate system message to fit
            if messages and messages[0].get("role") == "system" and isinstance(messages[0].get("content"), str):
                sys_content = messages[0]["content"]
                excess_tokens = est_input - ctx_window + 512
                excess_chars = excess_tokens * 4
                _min_sys_chars = 2000
                available_for_trim = len(sys_content) - _min_sys_chars
                if available_for_trim > 0:
                    trim_amount = min(excess_chars, available_for_trim)
                    messages[0]["content"] = sys_content[: len(sys_content) - trim_amount]
                    logger.warning(
                        "Hard-truncated system prompt by %d chars "
                        "to fit context window %d "
                        "(est_input ~%d, excess ~%d tokens)",
                        trim_amount,
                        ctx_window,
                        est_input,
                        excess_tokens,
                    )
                    est_input = _estimate_tokens()
                    available = ctx_window - est_input
                    if available - 128 >= 256:
                        return {**llm_kwargs, "max_tokens": available - 128}

            logger.error(
                "Prompt too large for context window: ~%d tokens input, %d window",
                est_input,
                ctx_window,
            )
            return None

        return llm_kwargs

    async def _preflight_clamp_with_compaction(
        self,
        llm_kwargs: dict[str, Any],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        litellm: types.ModuleType,
    ) -> dict[str, Any] | None:
        """Pre-flight with automatic compaction fallback.

        Tries normal _preflight_clamp first. If it returns None (too large),
        attempts LLM one-shot compaction and retries.
        """
        result = self._preflight_clamp(llm_kwargs, messages, tools, litellm)
        if result is not None:
            return result

        # Try compaction
        compacted = await self._try_compact_messages(messages, llm_kwargs, litellm)
        if not compacted:
            return None

        # Retry preflight after compaction
        return self._preflight_clamp(llm_kwargs, messages, tools, litellm)

    async def _try_compact_messages(
        self,
        messages: list[dict[str, Any]],
        llm_kwargs: dict[str, Any],
        litellm: types.ModuleType,
    ) -> bool:
        """Compact conversation by asking the same model to summarize."""
        if len(messages) <= 3:
            return False

        ctx_window = self._resolve_cw()
        if ctx_window < _COMPACT_MIN_CONTEXT_WINDOW:
            return False

        # Format conversation history for summarization
        history_parts: list[str] = []
        for msg in messages[2:]:  # Skip system + original user
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "assistant" and msg.get("tool_calls"):
                calls = msg["tool_calls"]
                call_summaries = []
                for tc in calls:
                    fn = tc.get("function", {})
                    call_summaries.append(
                        f"  Tool: {fn.get('name', '?')}({fn.get('arguments', '')[:_COMPACT_TOOL_ARGS_TRUNCATE_CHARS]})"
                    )
                history_parts.append("[Assistant tool calls]\n" + "\n".join(call_summaries))
            elif role == "tool":
                content_str = content if isinstance(content, str) else str(content)
                truncated = content_str[:_COMPACT_MSG_TRUNCATE_CHARS]
                history_parts.append(f"[Tool result] {truncated}")
            elif content:
                content_str = content if isinstance(content, str) else str(content)
                history_parts.append(f"[{role}] {content_str[:_COMPACT_MSG_TRUNCATE_CHARS]}")

        history_text = "\n\n".join(history_parts)

        from core.i18n import t

        compact_system = t("litellm_context.compact_system")

        compact_prompt = [
            {"role": "system", "content": compact_system},
            {"role": "user", "content": history_text},
        ]

        compact_kwargs = {k: v for k, v in llm_kwargs.items() if k not in ("max_tokens",)}
        compact_kwargs["max_tokens"] = min(_COMPACT_MAX_SUMMARY_TOKENS, ctx_window // 4)
        # Remove thinking-related params for compaction (avoid overhead)
        for key in ["thinking", "think", "reasoning_effort", "enable_thinking", "reasoning_config"]:
            compact_kwargs.pop(key, None)
        if "extra_body" in compact_kwargs:
            eb = dict(compact_kwargs["extra_body"])
            for key in ("enable_thinking", "chat_template_kwargs"):
                eb.pop(key, None)
            if not eb:
                del compact_kwargs["extra_body"]
            else:
                compact_kwargs["extra_body"] = eb

        try:
            response = await litellm.acompletion(messages=compact_prompt, **compact_kwargs)
            summary = response.choices[0].message.content or ""
        except Exception:
            logger.warning("Compaction LLM call failed", exc_info=True)
            return False

        if not summary.strip():
            return False

        # Log compaction
        old_count = len(messages) - 2
        logger.info(
            "Compacted %d messages into summary (%d chars)",
            old_count,
            len(summary),
        )

        # Reset messages: keep system + original user, replace rest with summary
        messages[2:] = [
            {
                "role": "user",
                "content": t("litellm_context.compact_summary_prefix") + "\n" + summary,
            }
        ]
        return True
