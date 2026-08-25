# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Voice front lane — lightweight speech-first chat path via a local LLM.

The front lane connects directly to a small, fast OpenAI-compatible endpoint
(llama.cpp) that reuses the same Anima persona, and keeps an in-session
user/assistant history so it can carry a conversation without the full
agent loop (large-model TTFT regression).  Tools are not used yet; the
``tools`` argument is accepted so PR-3 (``ask_anima``) can hook in later.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

import litellm

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 2048
_DEFAULT_TEMPERATURE = 0.7
_HEALTH_TIMEOUT = 5.0

# Maximum number of tool-return rounds before the stream stops retrying
# (initial call + 2 tool rounds).
_MAX_TOOL_ROUNDS = 2

# OpenAI tool schema exposed to the front model. The front lane itself knows
# nothing about the supervisor: executing the tool is delegated to an
# injected ``tool_executor`` callback so PR-3's async delegation can stay
# in ``core/voice/session.py``.
ASK_ANIMA_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "ask_anima",
        "description": (
            "時間のかかる作業・ツール実行・記憶の検索保存・タスク化・調査・実装などを、"
            "自分自身の本体（フルエージェント）に依頼する。会話で即答できないことはこれで依頼する。"
            "挨拶・お礼・雑談・意見や感想・自分で即答できる質問には使わない"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "本体に依頼する内容（自然言語）",
                }
            },
            "required": ["request"],
        },
    },
}


def extract_emotion(full_text: str) -> str:
    """Parse the emotion tag from a front response; default to ``neutral``."""
    from core.emotion_tag import parse_emotion_value

    return parse_emotion_value(full_text)


class VoiceFrontLane:
    """Direct conversation lane to the fixed, fast front model.

    The system prompt is fixed for the whole session (prefix-cache
    friendly); only the user/assistant message list grows each turn.
    """

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        system_prompt: str,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = _DEFAULT_TEMPERATURE,
        timeout: float = 120.0,
        num_retries: int = 1,
        api_key: str = "local",
        tool_executor: Callable[[str], str] | None = None,
    ) -> None:
        self._model = model
        self._api_base = (api_base or "").rstrip("/")
        # litellm's openai provider requires an api_key even for local
        # llama.cpp endpoints; "local" is a harmless placeholder.
        self._api_key = api_key
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout
        self._num_retries = num_retries
        self._tool_executor = tool_executor
        self._history: list[dict[str, str]] = []
        self._last_full_text = ""

    # ── session control ──────────────────────────────────────────

    def set_history(self, history: list[dict[str, str]]) -> None:
        """Seed the in-session conversation history (optional)."""
        self._history = list(history)

    def reset_turn(self) -> None:
        """Clear the previous turn's accumulator before a new stream."""
        self._last_full_text = ""

    @property
    def last_full_text(self) -> str:
        """Text of the most recent assistant turn (accumulated deltas)."""
        return self._last_full_text

    @property
    def history(self) -> list[dict[str, str]]:
        """Copy of the in-session user/assistant message list."""
        return list(self._history)

    # ── health / reachability ───────────────────────────────────

    async def check_health(self) -> bool:
        """Lightweight reachability check against the OpenAI-compatible API.

        Mirrors the shape of the existing TTS health check: cheap, caches
        nothing (call before every response), and returns ``False`` on any
        failure so the caller can fall back to the full agent loop.
        """
        if not self._api_base:
            return False
        try:
            import httpx

            async with httpx.AsyncClient(timeout=_HEALTH_TIMEOUT) as client:
                resp = await client.get(f"{self._api_base}/models")
                return resp.status_code < 500
        except Exception:
            logger.debug("voice front health check failed", exc_info=True)
            return False

    # ── streaming completion ────────────────────────────────────

    async def stream(
        self,
        user_text: str,
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str], str] | None = None,
    ) -> AsyncIterator[str]:
        """Send a user turn through the front lane and yield text deltas.

        If the model emits a ``tool_calls`` finish (e.g. ``ask_anima``), the
        injected ``tool_executor`` is invoked per tool call, the ``role:"tool"``
        result is appended, and the completion is retried (up to
        ``_MAX_TOOL_ROUNDS`` extra rounds) so the model can give its own
        spoken answer. Text deltas are still yielded as they arrive, so the
        caller's interface is unchanged. The final assistant reply is
        appended to the in-session history.

        Args:
            user_text: The user's spoken turn.
            tools: Optional OpenAI tool schemas to expose.
            tool_executor: Callable ``(request_text) -> str`` receiving the
                tool call's ``request`` argument (or the raw arguments string
                when JSON parsing fails). Falls back to the value passed to
                ``__init__``.
        """
        executor = tool_executor or getattr(self, "_tool_executor", None)
        messages: list[dict[str, Any]] = [{"role": "system", "content": self._system_prompt}]
        messages.extend(self._history)
        messages.append({"role": "user", "content": user_text})

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "timeout": self._timeout,
            "num_retries": self._num_retries,
            # The front lane is the low-latency lane: a thinking model burns
            # the whole max_tokens budget on reasoning and streams *no*
            # content (silent turn). Servers that don't know the kwarg ignore it.
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        if self._api_base:
            kwargs["api_base"] = self._api_base
            kwargs["api_key"] = self._api_key
        if tools:
            kwargs["tools"] = tools

        emitted: list[str] = []
        for _round in range(_MAX_TOOL_ROUNDS + 1):
            kwargs["messages"] = messages
            response = await litellm.acompletion(**kwargs)

            chunks: list[str] = []
            tool_calls: dict[int, dict[str, str]] = {}
            finish_reason: str | None = None
            async for chunk in response:
                if chunk is None:
                    continue
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                choice = choices[0]
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                content = getattr(delta, "content", None)
                if content:
                    chunks.append(content)
                    emitted.append(content)
                    yield content
                finish_reason = getattr(choice, "finish_reason", None)
                for tc in getattr(delta, "tool_calls", None) or []:
                    idx = int(getattr(tc, "index", 0) or 0)
                    entry = tool_calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if getattr(tc, "id", None):
                        entry["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn is None:
                        continue
                    if getattr(fn, "name", None):
                        entry["name"] += fn.name
                    if getattr(fn, "arguments", None):
                        entry["arguments"] += fn.arguments

            full_tool_text = "".join(chunks)
            if finish_reason == "tool_calls" and tool_calls and executor:
                tool_call_list: list[dict[str, Any]] = []
                for idx in sorted(tool_calls):
                    tcd = tool_calls[idx]
                    tool_call_list.append(
                        {
                            "id": tcd["id"] or f"call_{idx}",
                            "type": "function",
                            "function": {
                                "name": tcd["name"],
                                "arguments": tcd["arguments"],
                            },
                        }
                    )
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": full_tool_text or None,
                    "tool_calls": tool_call_list,
                }
                messages.append(assistant_msg)
                for tcd in tool_call_list:
                    raw_args = tcd["function"]["arguments"] or ""
                    try:
                        args = json.loads(raw_args or "{}")
                        request_text = args.get("request", "") if isinstance(args, dict) else ""
                    except json.JSONDecodeError:
                        # small models sometimes emit free text instead of JSON
                        request_text = raw_args
                    try:
                        result = executor(str(request_text))
                        if not isinstance(result, str):
                            result = str(result)
                    except Exception as exc:  # keep the conversation going
                        logger.exception("tool_executor failed for %s", tcd["function"]["name"])
                        result = f"error: {exc}"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tcd["id"],
                            "content": result,
                        }
                    )
                continue
            break

        full_text = "".join(emitted)
        self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant", "content": full_text})
        self._last_full_text = full_text
