from __future__ import annotations
# AnimaWorks - Digital Person Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of AnimaWorks core/server, licensed under AGPL-3.0.
# See LICENSES/AGPL-3.0.txt for the full license text.


"""Mode A1 executor: Claude Agent SDK.

Runs Claude as a fully autonomous agent with Read/Write/Edit/Bash/Grep/Glob
tools via the Agent SDK subprocess.  Supports both blocking and streaming
execution, plus a ``PostToolUse`` hook for context monitoring.
"""

import logging
import re
from collections.abc import AsyncGenerator
from typing import Any

from core.prompt.context import ContextTracker
from core.execution.base import BaseExecutor, ExecutionResult
from core.schemas import ModelConfig
from core.memory.shortterm import ShortTermMemory
from pathlib import Path

logger = logging.getLogger("animaworks.execution.agent_sdk")

_SEND_PATTERNS = [
    re.compile(r"Sent:\s+\S+\s+->\s+(\w+)"),           # CLI output: "Sent: sakura -> kotoha (...)"
    re.compile(r"Message sent to (\w+)"),                # ToolHandler: "Message sent to kotoha (...)"
]


class AgentSDKExecutor(BaseExecutor):
    """Execute via Claude Agent SDK (Mode A1).

    The SDK spawns a subprocess where Claude has full tool access.
    Context monitoring is handled via a PostToolUse hook that fires
    when token usage crosses the configured threshold.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        person_dir: Path,
        tool_registry: list[str] | None = None,
        personal_tools: dict[str, str] | None = None,
    ) -> None:
        super().__init__(model_config, person_dir)
        self._tool_registry = tool_registry or []
        self._personal_tools = personal_tools or {}

    @property
    def supports_streaming(self) -> bool:  # noqa: D102
        return True

    def _resolve_agent_sdk_model(self) -> str:
        """Return the model name suitable for Agent SDK (strip provider prefix)."""
        m = self._model_config.model
        if m.startswith("anthropic/"):
            return m[len("anthropic/"):]
        return m

    def _build_env(self) -> dict[str, str]:
        """Build env dict so the child process uses per-person credentials.

        Also sets ``ANIMAWORKS_PERSON_DIR`` so that ``animaworks-tool``
        can discover personal tools in the person's ``tools/`` directory.
        """
        env: dict[str, str] = {
            "ANIMAWORKS_PERSON_DIR": str(self._person_dir),
        }
        api_key = self._resolve_api_key()
        if api_key:
            env["ANTHROPIC_API_KEY"] = api_key
        if self._model_config.api_base_url:
            env["ANTHROPIC_BASE_URL"] = self._model_config.api_base_url
        return env

    # ── Transcript parsing ─────────────────────────────────────

    def _parse_replied_to(self, transcript_path: str) -> set[str]:
        """Parse Agent SDK transcript for message send patterns."""
        if not transcript_path:
            return set()
        path = Path(transcript_path)
        if not path.exists():
            return set()
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            logger.debug("Could not read transcript: %s", transcript_path)
            return set()
        names: set[str] = set()
        for pat in _SEND_PATTERNS:
            for m in pat.finditer(content):
                names.add(m.group(1))
        if names:
            logger.debug("Parsed replied_to from transcript: %s", names)
        return names

    # ── Blocking execution ───────────────────────────────────

    async def execute(
        self,
        prompt: str,
        system_prompt: str = "",
        tracker: ContextTracker | None = None,
        shortterm: ShortTermMemory | None = None,
        trigger: str = "",
    ) -> ExecutionResult:
        """Run a session via Claude Agent SDK with context monitoring hook.

        Returns ``ExecutionResult`` with the response text and the SDK
        ``ResultMessage`` (used for session chaining by AgentCore).
        """
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            HookMatcher,
            ResultMessage,
            TextBlock,
            query,
        )
        from claude_agent_sdk.types import (
            HookContext,
            HookInput,
            PostToolUseHookSpecificOutput,
            SyncHookJSONOutput,
        )

        threshold = self._model_config.context_threshold
        _hook_fired = False
        _transcript_path = ""

        async def _post_tool_hook(
            input_data: HookInput,
            tool_use_id: str | None,
            context: HookContext,
        ) -> SyncHookJSONOutput:
            nonlocal _hook_fired, _transcript_path
            if tracker is None:
                return SyncHookJSONOutput()
            transcript_path = input_data.get("transcript_path", "")
            _transcript_path = transcript_path or _transcript_path
            ratio = tracker.estimate_from_transcript(transcript_path)
            if ratio >= threshold and not _hook_fired:
                _hook_fired = True
                logger.info(
                    "PostToolUse hook: context at %.1f%%, injecting save instruction",
                    ratio * 100,
                )
                return SyncHookJSONOutput(
                    hookSpecificOutput=PostToolUseHookSpecificOutput(
                        hookEventName="PostToolUse",
                        additionalContext=(
                            f"\u30b3\u30f3\u30c6\u30ad\u30b9\u30c8\u4f7f\u7528\u7387\u304c{ratio:.0%}\u306b\u9054\u3057\u307e\u3057\u305f\u3002"
                            "shortterm/session_state.md \u306b\u73fe\u5728\u306e\u4f5c\u696d\u72b6\u614b\u3092\u66f8\u304d\u51fa\u3057\u3066\u304f\u3060\u3055\u3044\u3002"
                            "\u5185\u5bb9: \u4f55\u3092\u3057\u3066\u3044\u305f\u304b\u3001\u3069\u3053\u307e\u3067\u9032\u3093\u3060\u304b\u3001\u6b21\u306b\u4f55\u3092\u3059\u3079\u304d\u304b\u3002"
                            "\u66f8\u304d\u51fa\u3057\u5f8c\u3001\u4f5c\u696d\u3092\u4e2d\u65ad\u3057\u3066\u305d\u306e\u65e8\u3092\u5831\u544a\u3057\u3066\u304f\u3060\u3055\u3044\u3002"
                        ),
                    )
                )
            return SyncHookJSONOutput()

        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            allowed_tools=["Read", "Write", "Edit", "Bash", "Grep", "Glob"],
            permission_mode="acceptEdits",
            cwd=str(self._person_dir),
            max_turns=self._model_config.max_turns,
            model=self._resolve_agent_sdk_model(),
            env=self._build_env(),
            hooks={
                "PostToolUse": [HookMatcher(matcher=None, hooks=[_post_tool_hook])],
            },
        )

        response_text: list[str] = []
        result_message: ResultMessage | None = None
        message_count = 0

        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, ResultMessage):
                    result_message = message
                    if tracker:
                        tracker.update_from_result_message(message.usage)
                elif isinstance(message, AssistantMessage):
                    message_count += 1
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text.append(block.text)
        except Exception as e:
            logger.exception("Agent SDK execution error")
            return ExecutionResult(
                text="\n".join(response_text) or f"[Agent SDK Error: {e}]",
            )

        logger.debug(
            "Agent SDK completed, messages=%d text_blocks=%d",
            message_count, len(response_text),
        )
        replied_to = self._parse_replied_to(_transcript_path)
        return ExecutionResult(
            text="\n".join(response_text) or "(no response)",
            result_message=result_message,
            replied_to_from_transcript=replied_to,
        )

    # ── Streaming execution ──────────────────────────────────

    async def execute_streaming(
        self,
        system_prompt: str,
        prompt: str,
        tracker: ContextTracker,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream events from Claude Agent SDK.

        Yields dicts:
            ``{"type": "text_delta", "text": "..."}``
            ``{"type": "tool_start", "tool_name": "...", "tool_id": "..."}``
            ``{"type": "tool_end", "tool_id": "...", "tool_name": "..."}``
            ``{"type": "done", "full_text": "...", "result_message": ...}``
        """
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            HookMatcher,
            ResultMessage,
            TextBlock,
            ToolUseBlock,
            query,
        )
        from claude_agent_sdk.types import (
            HookContext,
            HookInput,
            PostToolUseHookSpecificOutput,
            StreamEvent,
            SyncHookJSONOutput,
        )

        threshold = self._model_config.context_threshold
        _hook_fired = False
        _transcript_path = ""

        async def _post_tool_hook(
            input_data: HookInput,
            tool_use_id: str | None,
            context: HookContext,
        ) -> SyncHookJSONOutput:
            nonlocal _hook_fired, _transcript_path
            transcript_path = input_data.get("transcript_path", "")
            _transcript_path = transcript_path or _transcript_path
            ratio = tracker.estimate_from_transcript(transcript_path)
            if ratio >= threshold and not _hook_fired:
                _hook_fired = True
                logger.info(
                    "PostToolUse hook (stream): context at %.1f%%",
                    ratio * 100,
                )
                return SyncHookJSONOutput(
                    hookSpecificOutput=PostToolUseHookSpecificOutput(
                        hookEventName="PostToolUse",
                        additionalContext=(
                            f"\u30b3\u30f3\u30c6\u30ad\u30b9\u30c8\u4f7f\u7528\u7387\u304c{ratio:.0%}\u306b\u9054\u3057\u307e\u3057\u305f\u3002"
                            "shortterm/session_state.md \u306b\u73fe\u5728\u306e\u4f5c\u696d\u72b6\u614b\u3092\u66f8\u304d\u51fa\u3057\u3066\u304f\u3060\u3055\u3044\u3002"
                            "\u5185\u5bb9: \u4f55\u3092\u3057\u3066\u3044\u305f\u304b\u3001\u3069\u3053\u307e\u3067\u9032\u3093\u3060\u304b\u3001\u6b21\u306b\u4f55\u3092\u3059\u3079\u304d\u304b\u3002"
                            "\u66f8\u304d\u51fa\u3057\u5f8c\u3001\u4f5c\u696d\u3092\u4e2d\u65ad\u3057\u3066\u305d\u306e\u65e8\u3092\u5831\u544a\u3057\u3066\u304f\u3060\u3055\u3044\u3002"
                        ),
                    )
                )
            return SyncHookJSONOutput()

        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            allowed_tools=["Read", "Write", "Edit", "Bash", "Grep", "Glob"],
            permission_mode="acceptEdits",
            cwd=str(self._person_dir),
            max_turns=self._model_config.max_turns,
            model=self._resolve_agent_sdk_model(),
            env=self._build_env(),
            include_partial_messages=True,
            hooks={
                "PostToolUse": [HookMatcher(matcher=None, hooks=[_post_tool_hook])],
            },
        )

        response_text: list[str] = []
        result_message: ResultMessage | None = None
        active_tool_ids: set[str] = set()
        message_count = 0

        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, StreamEvent):
                    event = message.event
                    event_type = event.get("type", "")

                    if event_type == "content_block_start":
                        block = event.get("content_block", {})
                        if block.get("type") == "tool_use":
                            tool_id = block.get("id", "")
                            active_tool_ids.add(tool_id)
                            yield {
                                "type": "tool_start",
                                "tool_name": block.get("name", ""),
                                "tool_id": tool_id,
                            }

                    elif event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                yield {"type": "text_delta", "text": text}

                elif isinstance(message, AssistantMessage):
                    message_count += 1
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text.append(block.text)
                        elif isinstance(block, ToolUseBlock):
                            if block.id in active_tool_ids:
                                active_tool_ids.discard(block.id)
                                yield {
                                    "type": "tool_end",
                                    "tool_id": block.id,
                                    "tool_name": block.name,
                                }

                elif isinstance(message, ResultMessage):
                    result_message = message
                    tracker.update_from_result_message(message.usage)
        except Exception as e:
            logger.exception("Agent SDK streaming error")
            yield {
                "type": "error",
                "message": f"[Agent SDK Error: {e}]",
            }
            return

        logger.debug(
            "Agent SDK streaming completed, messages=%d text_blocks=%d",
            message_count, len(response_text),
        )
        full_text = "\n".join(response_text) or "(no response)"
        replied_to = self._parse_replied_to(_transcript_path)
        yield {
            "type": "done",
            "full_text": full_text,
            "result_message": result_message,
            "replied_to_from_transcript": replied_to,
        }
