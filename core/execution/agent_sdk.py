from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Mode S executor: Claude Agent SDK.

Runs Claude as a fully autonomous agent with Read/Write/Edit/Bash/Grep/Glob
tools via the Agent SDK subprocess.  Supports both blocking and streaming
execution.  Tool results are captured from UserMessage ToolResultBlock
instead of PostToolUse hooks.

Implementation is split across submodules for readability:
  - ``_sdk_security``: Security checks and output size guards
  - ``_sdk_session``: Session persistence, SDK input helpers, cleanup
  - ``_sdk_stream``: Tool logging/sanitization, stream block processing
  - ``_sdk_hooks``: PreToolUse/PreCompact hooks, subordinate management
  - ``_sdk_options``: SDK option building (Mixin)
  - ``_sdk_interrupt``: Graceful interrupt helpers
"""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    try:
        from claude_agent_sdk import ClaudeSDKClient, ResultMessage
    except ImportError:
        pass

from pathlib import Path

from core.exceptions import ExecutionError, LLMAPIError, MemoryWriteError  # noqa: F401
from core.execution import _sdk_session
from core.execution._sdk_patch import apply_sdk_transport_patch

apply_sdk_transport_patch()

# ── Re-exports from submodules (backward compatibility) ──────
from core.execution._sdk_hooks import (  # noqa: F401
    _build_post_tool_hook,
    _build_pre_compact_hook,
    _build_pre_tool_hook,
    _cache_subordinate_paths,
    _collect_all_subordinates,
    _intercept_task_to_delegation,
    _intercept_task_to_pending,
    _read_status_json,
    _select_subordinate,
)
from core.execution._sdk_interrupt import (  # noqa: F401
    _graceful_interrupt_blocking,
    _graceful_interrupt_stream,
)
from core.execution._sdk_options import SDKOptionsMixin  # noqa: F401
from core.execution._sdk_security import (  # noqa: F401
    _BASH_HEAD_BYTES,
    _BASH_TAIL_BYTES,
    _BASH_TRUNCATE_BYTES,
    _GLOB_DEFAULT_HEAD_LIMIT,
    _GREP_DEFAULT_HEAD_LIMIT,
    _PROTECTED_FILES,
    _READ_DEFAULT_LIMIT,
    _WRITE_COMMANDS,
    _build_output_guard,
    _check_a1_bash_command,
    _check_a1_file_access,
    _guard_bash,
    _guard_glob,
    _guard_grep,
    _guard_read,
)
from core.execution._sdk_session import (  # noqa: F401
    _CONTEXT_AUTOCOMPACT_SAFETY,
    _PROMPT_FILE_THRESHOLD,
    _RESUMABLE_SESSION_TYPES,
    _SDK_MAX_BUFFER_SIZE,
    INTERRUPT_TIMEOUT_SEC,
    RESUME_TIMEOUT_SEC,
    SESSION_TYPE_CHAT,
    SESSION_TYPE_CRON,
    SESSION_TYPE_HEARTBEAT,
    SESSION_TYPE_INBOX,
    SESSION_TYPE_TASK,
    _build_sdk_query_input,
    _cleanup_prompt_files,
    _cleanup_tool_outputs,
    _clear_session_id,
    _image_prompt_messages,
    _is_debug_superuser,
    _load_session_id,
    _resolve_session_type,
    _save_session_id,
    _session_file,
    compact_sdk_session,
)
from core.execution._sdk_stream import (  # noqa: F401
    StreamingContext,
    StreamingState,
    _finalize_pending_records,
    _handle_tool_result_block,
    _handle_tool_use_block,
    _log_tool_result,
    _log_tool_use,
    _sanitise_tool_args,
    _summarise_tool_input,
    _tool_result_content_len,
    process_stream_messages,
)
from core.execution.base import BaseExecutor, ExecutionResult, StreamDisconnectedError, TokenUsage, ToolCallRecord
from core.memory.shortterm import ShortTermMemory
from core.prompt.context import CHARS_PER_TOKEN, ContextTracker
from core.schemas import ImageData, ModelConfig

logger = logging.getLogger("animaworks.execution.agent_sdk")

__all__ = ["AgentSDKExecutor", "StreamDisconnectedError"]


# ── AgentSDKExecutor ─────────────────────────────────────────


class AgentSDKExecutor(SDKOptionsMixin, BaseExecutor):
    """Execute via Claude Agent SDK (Mode S).

    The SDK spawns a subprocess where Claude has full tool access.
    Tool results are captured from UserMessage ToolResultBlock content
    via ``_handle_tool_result_block``.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        anima_dir: Path,
        tool_registry: list[str] | None = None,
        personal_tools: dict[str, str] | None = None,
        interrupt_event: asyncio.Event | None = None,
    ) -> None:
        super().__init__(model_config, anima_dir, interrupt_event=interrupt_event)
        self._tool_registry = tool_registry or []
        self._personal_tools = personal_tools or {}

    @property
    def supports_streaming(self) -> bool:  # noqa: D102
        return True

    def _init_session_stats(self, system_prompt: str, prompt: str, trigger: str) -> dict[str, Any]:
        """Build the mutable session-stats dict shared with PreToolUse hook."""
        return {
            "tool_call_count": 0,
            "total_result_bytes": 0,
            "system_prompt_tokens": len(system_prompt) // CHARS_PER_TOKEN,
            "user_prompt_tokens": len(prompt) // CHARS_PER_TOKEN,
            "force_chain": False,
            "trigger": trigger,
            "start_time": time.monotonic(),
            "hb_soft_warned": False,
            "hb_soft_timeout": self._hb_soft_timeout_s,
        }

    # ── Blocking execution ───────────────────────────────────

    async def _process_blocking_messages(
        self,
        client: ClaudeSDKClient,
        prompt: str,
        response_text: list[str],
        pending_records: dict[str, ToolCallRecord],
        session_stats: dict[str, Any],
        tracker: ContextTracker | None,
        session_type: str = "chat",
        images: list[ImageData] | None = None,
        usage_acc: TokenUsage | None = None,
        thread_id: str = "default",
    ) -> ResultMessage | None:
        """Run query + message loop for blocking (non-streaming) execution."""
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            SystemMessage,
            TextBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
        )

        result_message: ResultMessage | None = None
        await client.query(_build_sdk_query_input(prompt, images))
        async for message in client.receive_response():
            if self._check_interrupted():
                logger.info("Agent SDK execute interrupted — sending graceful interrupt")
                response_text.append("[Session interrupted by user]")
                await _graceful_interrupt_blocking(
                    client,
                    self._anima_dir,
                    session_type,
                    thread_id=thread_id,
                )
                return result_message

            if isinstance(message, ResultMessage):
                result_message = message
                if message.session_id and session_type in _RESUMABLE_SESSION_TYPES:
                    _save_session_id(self._anima_dir, message.session_id, session_type, thread_id=thread_id)
                if tracker:
                    tracker.update_from_result_message(message.usage)
                if usage_acc and message.usage:
                    u = message.usage
                    usage_acc.input_tokens = u.get("input_tokens", 0) or 0
                    usage_acc.output_tokens = u.get("output_tokens", 0) or 0
                    usage_acc.cache_read_tokens = u.get("cache_read_input_tokens", 0) or 0
                    usage_acc.cache_write_tokens = u.get("cache_creation_input_tokens", 0) or 0
            elif isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response_text.append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        _handle_tool_use_block(
                            block,
                            pending_records,
                            None,
                            self._model_config.model,
                            cw_overrides=self._resolve_cw_overrides(),
                        )
            elif isinstance(message, UserMessage):
                if isinstance(message.content, list):
                    for block in message.content:
                        if isinstance(block, ToolResultBlock):
                            session_stats["total_result_bytes"] += _tool_result_content_len(block)
                            _handle_tool_result_block(
                                block,
                                pending_records,
                                None,
                                self._model_config.model,
                                anima_dir=self._anima_dir,
                                cw_overrides=self._resolve_cw_overrides(),
                            )
            elif isinstance(message, SystemMessage):
                if message.subtype == "init" and message.data:
                    for srv in message.data.get("mcp_servers", []):
                        name = srv.get("name", "unknown")
                        status = srv.get("status", "unknown")
                        if status != "connected":
                            logger.error("MCP server '%s' failed to connect: status=%s", name, status)
                        else:
                            logger.info("MCP server '%s' connected successfully", name)

        return result_message

    async def execute(
        self,
        prompt: str,
        system_prompt: str = "",
        tracker: ContextTracker | None = None,
        shortterm: ShortTermMemory | None = None,
        trigger: str = "",
        images: list[ImageData] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        max_turns_override: int | None = None,
        thread_id: str = "default",
    ) -> ExecutionResult:
        """Run a session via Claude Agent SDK with context monitoring hook."""
        from claude_agent_sdk import ClaudeSDKClient, ClaudeSDKError, ProcessError

        _cw = self._resolve_cw()
        _max_turns = max_turns_override or self._model_config.max_turns
        session_stats = self._init_session_stats(system_prompt, prompt, trigger)
        session_type = _resolve_session_type(trigger)
        session_id_to_resume = (
            _load_session_id(self._anima_dir, session_type, thread_id=thread_id)
            if session_type in _RESUMABLE_SESSION_TYPES
            else None
        )

        options, prompt_file = self._build_sdk_options(
            system_prompt,
            _max_turns,
            _cw,
            session_stats,
            resume=session_id_to_resume,
        )
        _prompt_files: list[Path] = [prompt_file] if prompt_file else []
        response_text: list[str] = []
        pending_records: dict[str, ToolCallRecord] = {}
        result_message = None
        usage_acc = TokenUsage()
        _msg_args = dict(
            prompt=prompt,
            response_text=response_text,
            pending_records=pending_records,
            session_stats=session_stats,
            tracker=tracker,
            session_type=session_type,
            images=images,
            usage_acc=usage_acc,
            thread_id=thread_id,
        )

        try:
            logger.info("ClaudeSDKClient connecting (blocking mode, resume=%s)", session_id_to_resume)
            async with ClaudeSDKClient(options=options) as client:
                logger.info("ClaudeSDKClient connected")
                result_message = await self._process_blocking_messages(client, **_msg_args)
            logger.debug("ClaudeSDKClient disconnected")
        except (ProcessError, ClaudeSDKError) as e:
            if session_id_to_resume:
                logger.warning("SDK session resume failed (session_id=%s): %s", session_id_to_resume, e)
                _sdk_session._clear_session_id(self._anima_dir, session_type, thread_id=thread_id)
                options, pf = self._build_sdk_options(system_prompt, _max_turns, _cw, session_stats, resume=None)
                if pf:
                    _prompt_files.append(pf)
                try:
                    async with ClaudeSDKClient(options=options) as client:
                        logger.info("ClaudeSDKClient connected (fresh session retry)")
                        result_message = await self._process_blocking_messages(client, **_msg_args)
                except Exception as retry_exc:
                    logger.exception("Agent SDK execution error (fresh session retry)")
                    return ExecutionResult(
                        text="\n".join(response_text) or f"[Agent SDK Error: {retry_exc}]",
                        tool_call_records=_finalize_pending_records(pending_records),
                    )
            else:
                logger.exception("Agent SDK execution error")
                return ExecutionResult(
                    text="\n".join(response_text) or f"[Agent SDK Error: {e}]",
                    tool_call_records=_finalize_pending_records(pending_records),
                )
        except Exception as e:
            logger.exception("Agent SDK execution error")
            return ExecutionResult(
                text="\n".join(response_text) or f"[Agent SDK Error: {e}]",
                tool_call_records=_finalize_pending_records(pending_records),
            )
        finally:
            _cleanup_tool_outputs(self._anima_dir)
            _cleanup_prompt_files(_prompt_files)

        all_tool_records = _finalize_pending_records(pending_records)
        replied_to = self._read_replied_to_file()
        return ExecutionResult(
            text="\n".join(response_text) or "(no response)",
            result_message=result_message,
            replied_to_from_transcript=replied_to,
            tool_call_records=all_tool_records,
            force_chain=session_stats.get("force_chain", False),
            usage=usage_acc,
        )

    # ── Streaming execution ──────────────────────────────────

    async def execute_streaming(
        self,
        system_prompt: str,
        prompt: str,
        tracker: ContextTracker,
        images: list[ImageData] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        max_turns_override: int | None = None,
        trigger: str = "",
        thread_id: str = "default",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream events from Claude Agent SDK."""
        from claude_agent_sdk import ClaudeSDKClient, ClaudeSDKError, ProcessError

        _cw = self._resolve_cw()
        _max_turns = max_turns_override or self._model_config.max_turns
        session_stats = self._init_session_stats(system_prompt, prompt, trigger)
        session_type = _resolve_session_type(trigger)
        session_id_to_resume = (
            _load_session_id(self._anima_dir, session_type, thread_id=thread_id)
            if session_type in _RESUMABLE_SESSION_TYPES
            else None
        )

        options, prompt_file = self._build_sdk_options(
            system_prompt,
            _max_turns,
            _cw,
            session_stats,
            resume=session_id_to_resume,
            include_partial_messages=True,
        )
        _prompt_files: list[Path] = [prompt_file] if prompt_file else []
        state = StreamingState(usage_acc=TokenUsage())
        ctx = StreamingContext(
            prompt=prompt,
            images=images,
            session_stats=session_stats,
            tracker=tracker,
            session_type=session_type,
            model=self._model_config.model,
            anima_dir=self._anima_dir,
            cw_overrides=self._resolve_cw_overrides(),
            check_interrupted=self._check_interrupted,
            thread_id=thread_id,
        )

        async def _fresh_session() -> AsyncGenerator[dict[str, Any], None]:
            fresh_opts, pf = self._build_sdk_options(
                system_prompt,
                _max_turns,
                _cw,
                session_stats,
                resume=None,
                include_partial_messages=True,
            )
            if pf:
                _prompt_files.append(pf)
            try:
                async with ClaudeSDKClient(options=fresh_opts) as fc:
                    logger.info("ClaudeSDKClient connected (fresh session retry)")
                    async for ev in process_stream_messages(fc, ctx, state):
                        yield ev
            except BaseException as exc:
                if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
                    raise
                logger.exception("Agent SDK streaming error (fresh session retry)")
                raise StreamDisconnectedError(
                    f"Agent SDK stream error ({type(exc).__name__}): {exc}",
                    partial_text="\n".join(state.response_text),
                ) from exc

        try:
            logger.info("ClaudeSDKClient connecting (streaming, resume=%s)", session_id_to_resume)
            if session_id_to_resume:
                fell_back = False
                try:
                    async with ClaudeSDKClient(options=options) as client:
                        logger.info("ClaudeSDKClient connected")
                        gen = process_stream_messages(client, ctx, state)
                        try:
                            first = await asyncio.wait_for(gen.__anext__(), timeout=RESUME_TIMEOUT_SEC)
                        except TimeoutError:
                            logger.warning("Resume timed out (session_id=%s)", session_id_to_resume)
                            await gen.aclose()
                            _sdk_session._clear_session_id(self._anima_dir, session_type, thread_id=thread_id)
                            fell_back = True
                        except StopAsyncIteration:
                            logger.warning("Resume stream empty (session_id=%s)", session_id_to_resume)
                            _sdk_session._clear_session_id(self._anima_dir, session_type, thread_id=thread_id)
                            fell_back = True
                        else:
                            yield first
                            async for ev in gen:
                                yield ev
                except (ProcessError, ClaudeSDKError) as e:
                    logger.warning("SDK resume failed (session_id=%s): %s", session_id_to_resume, e)
                    _sdk_session._clear_session_id(self._anima_dir, session_type, thread_id=thread_id)
                    fell_back = True
                if fell_back:
                    async for ev in _fresh_session():
                        yield ev
            else:
                async with ClaudeSDKClient(options=options) as client:
                    logger.info("ClaudeSDKClient connected")
                    async for ev in process_stream_messages(client, ctx, state):
                        yield ev
            logger.debug("ClaudeSDKClient disconnected")
        except BaseException as e:
            if isinstance(e, (asyncio.CancelledError, GeneratorExit)):
                raise
            logger.exception("Agent SDK streaming error")
            raise StreamDisconnectedError(
                f"Agent SDK stream error ({type(e).__name__}): {e}",
                partial_text="\n".join(state.response_text),
            ) from e
        finally:
            _cleanup_tool_outputs(self._anima_dir)
            _cleanup_prompt_files(_prompt_files)

        all_tool_records = _finalize_pending_records(state.pending_records)
        full_text = "\n".join(state.response_text) or "(no response)"
        replied_to = self._read_replied_to_file()
        yield {
            "type": "done",
            "full_text": full_text,
            "result_message": state.result_message,
            "replied_to_from_transcript": replied_to,
            "tool_call_records": [asdict(r) for r in all_tool_records],
            "force_chain": session_stats.get("force_chain", False),
            "usage": state.usage_acc.to_dict(),
        }

    async def compact_session(
        self,
        anima_dir: Path,
        session_type: str = "chat",
        thread_id: str = "default",
    ) -> bool:
        """Delegate to ``compact_sdk_session`` for backward compatibility."""
        return await compact_sdk_session(anima_dir, session_type, thread_id)
