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
"""

import asyncio
import logging
import os
import sys
import tempfile
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    try:
        from claude_code_sdk import ClaudeAgentOptions, ResultMessage
        from claude_code_sdk import ClaudeCodeSDKClient as ClaudeSDKClient
    except ImportError:
        pass

from pathlib import Path

from core.exceptions import ExecutionError, LLMAPIError, MemoryWriteError  # noqa: F401
from core.execution import _sdk_session
from core.execution._sdk_patch import apply_sdk_transport_patch

apply_sdk_transport_patch()
from core.execution._sdk_hooks import (  # noqa: F401
    _build_pre_compact_hook,
    _build_pre_tool_hook,
    _cache_subordinate_paths,
    _collect_all_subordinates,
    _intercept_task_to_delegation,
    _intercept_task_to_pending,
    _read_status_json,
    _select_subordinate,
)

# ── Re-exports from submodules (backward compatibility) ──────
from core.execution._sdk_security import (  # noqa: F401
    _BASH_BLOCKED_PATTERNS,
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
)
from core.execution._sdk_stream import (  # noqa: F401
    _finalize_pending_records,
    _handle_tool_result_block,
    _handle_tool_use_block,
    _log_tool_result,
    _log_tool_use,
    _sanitise_tool_args,
    _summarise_tool_input,
    _tool_result_content_len,
)
from core.execution._tool_summary import make_tool_detail_chunk
from core.execution.base import BaseExecutor, ExecutionResult, StreamDisconnectedError, TokenUsage, ToolCallRecord
from core.memory.shortterm import ShortTermMemory
from core.prompt.context import CHARS_PER_TOKEN, ContextTracker
from core.schemas import ImageData, ModelConfig

logger = logging.getLogger("animaworks.execution.agent_sdk")

__all__ = ["AgentSDKExecutor", "StreamDisconnectedError"]


# ── AgentSDKExecutor ─────────────────────────────────────────


class AgentSDKExecutor(BaseExecutor):
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

    def _resolve_agent_sdk_model(self) -> str:
        """Return the model name suitable for Agent SDK (strip provider prefix)."""
        import re

        m = self._model_config.model
        # Strip provider prefixes:
        #   anthropic/claude-sonnet-4-6
        #   bedrock/jp.anthropic.claude-sonnet-4-6
        #   bedrock/claude-sonnet-4-6
        #   vertex_ai/claude-sonnet-4-6
        m = re.sub(
            r"^(anthropic|bedrock|vertex_ai)/"
            r"([a-z]{2}\.anthropic\.)?",
            "",
            m,
        )
        return m

    def _build_env(self) -> dict[str, str]:
        """Build env dict for the Claude Code child process.

        Authentication mode is determined by ``mode_s_auth`` (per-Anima
        setting from status.json / anima_defaults):

        * ``"api"`` — credential has ``api_key`` →
          ``ANTHROPIC_API_KEY`` is set to that key.
        * ``"bedrock"`` — credential ``keys`` contain ``aws_access_key_id``
          → ``CLAUDE_CODE_USE_BEDROCK=1`` plus AWS env vars.
        * ``"vertex"`` — credential ``keys`` contain ``vertex_project`` →
          ``CLAUDE_CODE_USE_VERTEX=1`` plus GCP env vars.
        * ``"max"`` / ``None`` (default) — subscription auth →
          ``ANTHROPIC_API_KEY=""`` (Max plan).
        """
        from core.paths import PROJECT_DIR

        env: dict[str, str] = {
            "ANIMAWORKS_ANIMA_DIR": str(self._anima_dir),
            "ANIMAWORKS_PROJECT_DIR": str(PROJECT_DIR),
            "PATH": f"{self._anima_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
            "CLAUDE_CODE_DISABLE_SKILL_IMPROVEMENT": "true",
            "CLAUDECODE": "",
        }

        auth = self._model_config.mode_s_auth
        extra = self._model_config.extra_keys
        api_key = self._resolve_api_key()

        if auth == "api":
            if api_key:
                env["ANTHROPIC_API_KEY"] = api_key
                logger.info("Mode S auth: API direct (mode_s_auth=api)")
            else:
                logger.warning("Mode S auth: mode_s_auth=api but no api_key found; falling back to Max plan")
                env["ANTHROPIC_API_KEY"] = ""
        elif auth == "bedrock":
            env["ANTHROPIC_API_KEY"] = ""
            env["CLAUDE_CODE_USE_BEDROCK"] = "1"
            for env_key, extra_key in (
                ("AWS_ACCESS_KEY_ID", "aws_access_key_id"),
                ("AWS_SECRET_ACCESS_KEY", "aws_secret_access_key"),
                ("AWS_SESSION_TOKEN", "aws_session_token"),
                ("AWS_REGION", "aws_region_name"),
                ("AWS_PROFILE", "aws_profile"),
            ):
                val = extra.get(extra_key) or os.environ.get(env_key)
                if val:
                    env[env_key] = val
            logger.info("Mode S auth: Bedrock (mode_s_auth=bedrock)")
        elif auth == "vertex":
            env["ANTHROPIC_API_KEY"] = ""
            env["CLAUDE_CODE_USE_VERTEX"] = "1"
            for env_key, extra_key in (
                ("CLOUD_ML_PROJECT_ID", "vertex_project"),
                ("CLOUD_ML_REGION", "vertex_location"),
                ("GOOGLE_APPLICATION_CREDENTIALS", "vertex_credentials"),
            ):
                val = extra.get(extra_key) or os.environ.get(env_key)
                if val:
                    env[env_key] = val
            logger.info("Mode S auth: Vertex AI (mode_s_auth=vertex)")
        else:
            # Default: Max plan (subscription auth).  Block any API key
            # that might leak from the parent process or anima_defaults.
            env["ANTHROPIC_API_KEY"] = ""
            logger.info("Mode S auth: Max plan (mode_s_auth=%s)", auth)

        if self._model_config.api_base_url:
            env["ANTHROPIC_BASE_URL"] = self._model_config.api_base_url
        return env

    def _build_mcp_env(self) -> dict[str, str]:
        """Build env dict for the MCP server subprocess.

        The MCP server needs ANIMAWORKS_ANIMA_DIR and ANIMAWORKS_PROJECT_DIR
        to initialize ToolHandler, plus PYTHONPATH so it can import core modules.
        """
        from core.paths import PROJECT_DIR

        return {
            "ANIMAWORKS_ANIMA_DIR": str(self._anima_dir),
            "ANIMAWORKS_PROJECT_DIR": str(PROJECT_DIR),
            "PYTHONPATH": str(PROJECT_DIR),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        }

    def _make_pending_executor_wake_callback(self) -> Callable[[], None] | None:
        """Create a callback that writes a .wake file for PendingTaskExecutor.

        The wake file signals the pending executor (running in the runner
        subprocess) to check for new tasks immediately rather than waiting
        for the next poll interval.
        """
        wake_path = self._anima_dir / "state" / "pending" / ".wake"

        def _wake() -> None:
            try:
                wake_path.parent.mkdir(parents=True, exist_ok=True)
                wake_path.write_text("1", encoding="utf-8")
            except Exception:
                pass

        return _wake

    # ── SDK helpers (shared by execute / execute_streaming) ──

    def _build_sdk_options(
        self,
        system_prompt: str,
        max_turns: int,
        context_window: int,
        session_stats: dict[str, Any],
        *,
        resume: str | None = None,
        include_partial_messages: bool = False,
    ) -> tuple[ClaudeAgentOptions, Path | None]:
        """Construct ``ClaudeAgentOptions`` for the Agent SDK client.

        Shared by both ``execute()`` and ``execute_streaming()`` (initial
        and retry attempts).  All SDK-specific lazy imports live here so
        callers need not repeat them.

        Returns:
            A tuple of (options, prompt_file).  *prompt_file* is ``None``
            when the prompt fits in a CLI argument; otherwise it is a
            ``Path`` to a temp file that the **caller must delete** after
            the SDK client has been closed.
        """
        from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

        _cw = context_window

        # When the system prompt exceeds MAX_ARG_STRLEN (128 KiB on Linux),
        # execve fails with E2BIG.  Fall back to --system-prompt-file, an
        # undocumented but functional CLI flag.  The SDK always emits
        # --system-prompt ""; JS treats "" as falsy so the conflict check
        # `if (f.systemPrompt)` passes and --system-prompt-file takes effect.
        prompt_file: Path | None = None
        extra_args: dict[str, str | None] = {}
        if len(system_prompt.encode("utf-8")) > _PROMPT_FILE_THRESHOLD:
            fd, tmp_path = tempfile.mkstemp(
                suffix=".txt",
                prefix="aw-sysprompt-",
            )
            try:
                os.write(fd, system_prompt.encode("utf-8"))
            finally:
                os.close(fd)
            prompt_file = Path(tmp_path)
            prompt_kwarg: str | None = None
            extra_args["system-prompt-file"] = tmp_path
            logger.info(
                "System prompt too large for CLI arg (%d bytes > %d); using --system-prompt-file %s",
                len(system_prompt.encode("utf-8")),
                _PROMPT_FILE_THRESHOLD,
                tmp_path,
            )
        else:
            prompt_kwarg = system_prompt

        # ── Resolve effective max_tokens ──────────────────────
        from core.config.models import resolve_max_tokens

        _effective_max_tokens = resolve_max_tokens(
            self._model_config.model,
            self._model_config.max_tokens,
            self._model_config.thinking,
        )

        _has_subs = self._has_subordinates()

        _allowed_tools = [
            "Read",
            "Write",
            "Edit",
            "Bash",
            "Grep",
            "Glob",
            "WebFetch",
            "WebSearch",
            "mcp__aw__*",
        ]
        _allowed_tools.extend(["Task", "Agent"])

        kwargs: dict[str, Any] = dict(
            system_prompt=prompt_kwarg,
            allowed_tools=_allowed_tools,
            permission_mode="acceptEdits",
            cwd=str(self._anima_dir),
            max_turns=max_turns,
            model=self._resolve_agent_sdk_model(),
            env=self._build_env(),
            max_buffer_size=_SDK_MAX_BUFFER_SIZE,
            resume=resume,
            setting_sources=[],  # CLI内蔵hook(settings.json)の読み込みを防止
            extra_args=extra_args,
            mcp_servers={
                "aw": {
                    "command": sys.executable,
                    "args": ["-m", "core.mcp.server"],
                    "env": self._build_mcp_env(),
                },
            },
            hooks={
                "PreToolUse": [
                    HookMatcher(
                        matcher=".*",
                        hooks=[
                            _build_pre_tool_hook(
                                self._anima_dir,
                                max_tokens=_effective_max_tokens,
                                context_window=_cw,
                                session_stats=session_stats,
                                superuser=_is_debug_superuser(self._anima_dir),
                                on_task_intercepted=self._make_pending_executor_wake_callback(),
                                has_subordinates=_has_subs,
                            )
                        ],
                    )
                ],
                "PreCompact": [
                    HookMatcher(
                        matcher=".*",
                        hooks=[_build_pre_compact_hook(self._anima_dir)],
                    )
                ],
            },
        )
        # ── Adaptive thinking ─────────────────────────────────
        if self._model_config.thinking:
            from core.execution.base import is_adaptive_model, resolve_thinking_effort

            if is_adaptive_model(self._model_config.model):
                kwargs["thinking"] = {"type": "adaptive"}
                kwargs["effort"] = resolve_thinking_effort(
                    self._model_config.model,
                    self._model_config.thinking_effort,
                )

        if include_partial_messages:
            kwargs["include_partial_messages"] = True
        return ClaudeAgentOptions(**kwargs), prompt_file

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
        """Run query + message loop for blocking (non-streaming) execution.

        Sends *prompt* via ``client.query()``, then iterates
        ``client.receive_response()`` to collect assistant text, tool
        records and the final ``ResultMessage``.  Returns the
        ``ResultMessage`` (or ``None`` if the loop ended without one).
        """
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
                logger.info("Agent SDK execute interrupted")
                response_text.append("[Session interrupted by user]")
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
                    mcp_servers = message.data.get("mcp_servers", [])
                    for srv in mcp_servers:
                        name = srv.get("name", "unknown")
                        status = srv.get("status", "unknown")
                        if status != "connected":
                            logger.error(
                                "MCP server '%s' failed to connect: status=%s",
                                name,
                                status,
                            )
                        else:
                            logger.info("MCP server '%s' connected successfully", name)

        return result_message

    # ── Blocking execution ───────────────────────────────────

    async def execute(
        self,
        prompt: str,
        system_prompt: str = "",
        tracker: ContextTracker | None = None,
        shortterm: ShortTermMemory | None = None,
        trigger: str = "",
        images: list[ImageData] | None = None,
        # S mode: prior_messages is intentionally unused. The Agent SDK manages
        # conversation history internally via session resume. AnimaWorks only
        # provides system_prompt (rebuilt each time with fresh Priming/RAG)
        # and the current user message.
        prior_messages: list[dict[str, Any]] | None = None,
        max_turns_override: int | None = None,
        thread_id: str = "default",
    ) -> ExecutionResult:
        """Run a session via Claude Agent SDK with context monitoring hook.

        Returns ``ExecutionResult`` with the response text and the SDK
        ``ResultMessage`` (used for session chaining by AgentCore).
        """
        from claude_agent_sdk import (
            ClaudeSDKClient,
            ClaudeSDKError,
            ProcessError,
        )

        # ── Session stats: shared between PreToolUse hook closure and this
        #    outer message loop.  The hook reads these values to decide
        #    whether to terminate the session for auto-compact; the loop
        #    updates total_result_bytes after each ToolResultBlock.
        #    Both run in the same async task — no concurrent access.
        _cw = self._resolve_cw()
        _max_turns = max_turns_override or self._model_config.max_turns
        session_stats: dict[str, Any] = {
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
        _prompt_files: list[Path] = []
        if prompt_file:
            _prompt_files.append(prompt_file)

        response_text: list[str] = []
        pending_records: dict[str, ToolCallRecord] = {}
        result_message = None
        message_count = 0
        usage_acc = TokenUsage()

        try:
            logger.info(
                "ClaudeSDKClient connecting (blocking mode, resume=%s)",
                session_id_to_resume,
            )
            async with ClaudeSDKClient(options=options) as client:
                logger.info("ClaudeSDKClient connected")
                result_message = await self._process_blocking_messages(
                    client,
                    prompt,
                    response_text,
                    pending_records,
                    session_stats,
                    tracker,
                    session_type,
                    images=images,
                    usage_acc=usage_acc,
                    thread_id=thread_id,
                )
            logger.debug("ClaudeSDKClient disconnected")
        except (ProcessError, ClaudeSDKError) as e:
            if session_id_to_resume:
                logger.warning(
                    "SDK session resume failed (session_id=%s): %s. Retrying with fresh session.",
                    session_id_to_resume,
                    e,
                )
                _sdk_session._clear_session_id(self._anima_dir, session_type, thread_id=thread_id)
                # Retry without resume
                options, pf = self._build_sdk_options(
                    system_prompt,
                    _max_turns,
                    _cw,
                    session_stats,
                    resume=None,
                )
                if pf:
                    _prompt_files.append(pf)
                try:
                    async with ClaudeSDKClient(options=options) as client:
                        logger.info("ClaudeSDKClient connected (fresh session retry)")
                        result_message = await self._process_blocking_messages(
                            client,
                            prompt,
                            response_text,
                            pending_records,
                            session_stats,
                            tracker,
                            session_type,
                            images=images,
                            usage_acc=usage_acc,
                            thread_id=thread_id,
                        )
                except Exception as retry_exc:
                    logger.exception("Agent SDK execution error (fresh session retry)")
                    all_tool_records = _finalize_pending_records(pending_records)
                    return ExecutionResult(
                        text="\n".join(response_text) or f"[Agent SDK Error: {retry_exc}]",
                        tool_call_records=all_tool_records,
                    )
            else:
                logger.exception("Agent SDK execution error")
                all_tool_records = _finalize_pending_records(pending_records)
                return ExecutionResult(
                    text="\n".join(response_text) or f"[Agent SDK Error: {e}]",
                    tool_call_records=all_tool_records,
                )
        except Exception as e:
            logger.exception("Agent SDK execution error")
            all_tool_records = _finalize_pending_records(pending_records)
            return ExecutionResult(
                text="\n".join(response_text) or f"[Agent SDK Error: {e}]",
                tool_call_records=all_tool_records,
            )
        finally:
            _cleanup_tool_outputs(self._anima_dir)
            _cleanup_prompt_files(_prompt_files)

        all_tool_records = _finalize_pending_records(pending_records)
        logger.debug(
            "Agent SDK completed, messages=%d text_blocks=%d tools=%d",
            message_count,
            len(response_text),
            len(all_tool_records),
        )
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
        # S mode: prior_messages is intentionally unused. The Agent SDK manages
        # conversation history internally via session resume. AnimaWorks only
        # provides system_prompt (rebuilt each time with fresh Priming/RAG)
        # and the current user message.
        prior_messages: list[dict[str, Any]] | None = None,
        max_turns_override: int | None = None,
        trigger: str = "",
        thread_id: str = "default",
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
            ClaudeSDKClient,
            ClaudeSDKError,
            ProcessError,
            ResultMessage,
            SystemMessage,
            TextBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
        )
        from claude_agent_sdk.types import StreamEvent

        # ── Session stats: shared between PreToolUse hook closure and this
        #    outer message loop (see execute() for detailed comment).
        _cw = self._resolve_cw()
        _max_turns = max_turns_override or self._model_config.max_turns
        session_stats: dict[str, Any] = {
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
        _prompt_files: list[Path] = []
        if prompt_file:
            _prompt_files.append(prompt_file)

        response_text: list[str] = []
        pending_records: dict[str, ToolCallRecord] = {}
        result_message: ResultMessage | None = None
        active_tool_ids: set[str] = set()
        message_count = 0
        usage_acc = TokenUsage()

        # --- inline helper: streaming message loop (not extractable because
        #     it yields from the generator) ---
        async def _stream_messages(
            client: ClaudeSDKClient,
        ) -> AsyncGenerator[dict[str, Any], None]:
            nonlocal result_message, message_count
            got_stream_event = False
            _in_thinking_block = False
            await client.query(_build_sdk_query_input(prompt, images))
            async for message in client.receive_messages():
                if self._check_interrupted():
                    logger.info("Agent SDK streaming interrupted")
                    yield {"type": "text_delta", "text": "[Session interrupted by user]"}
                    return

                if isinstance(message, StreamEvent):
                    got_stream_event = True
                    event = message.event
                    event_type = event.get("type", "")

                    if event_type == "message_start":
                        # Accurate per-turn context size (input + cache tokens).
                        # This is the authoritative source for threshold tracking
                        # in S mode — unlike ResultMessage.usage which is a
                        # cumulative sum across all turns.
                        usage = event.get("message", {}).get("usage", {})
                        if usage:
                            tracker.update_from_message_start(usage)
                            usage_acc.cache_read_tokens += usage.get("cache_read_input_tokens", 0)
                            usage_acc.cache_write_tokens += usage.get("cache_creation_input_tokens", 0)

                    elif event_type == "content_block_start":
                        block = event.get("content_block", {})
                        if block.get("type") == "tool_use":
                            tool_id = block.get("id", "")
                            tool_name = block.get("name", "")
                            active_tool_ids.add(tool_id)
                            yield {
                                "type": "tool_start",
                                "tool_name": tool_name,
                                "tool_id": tool_id,
                            }
                        elif block.get("type") == "thinking":
                            _in_thinking_block = True
                            yield {"type": "thinking_start"}

                    elif event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                yield {"type": "text_delta", "text": text}
                        elif delta.get("type") == "thinking_delta":
                            thinking_text = delta.get("thinking", "")
                            if thinking_text:
                                yield {"type": "thinking_delta", "text": thinking_text}

                    elif event_type == "content_block_stop":
                        if _in_thinking_block:
                            _in_thinking_block = False
                            yield {"type": "thinking_end"}

                elif isinstance(message, AssistantMessage):
                    if not got_stream_event:
                        continue
                    message_count += 1
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
                            detail_chunk = make_tool_detail_chunk(
                                block.name,
                                block.id,
                                block.input or {},
                            )
                            if detail_chunk:
                                yield detail_chunk
                            if block.id in active_tool_ids:
                                active_tool_ids.discard(block.id)
                                yield {
                                    "type": "tool_end",
                                    "tool_id": block.id,
                                    "tool_name": block.name,
                                }

                elif isinstance(message, UserMessage):
                    if not got_stream_event:
                        continue
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

                elif isinstance(message, ResultMessage):
                    result_message = message
                    if message.session_id and session_type in _RESUMABLE_SESSION_TYPES:
                        _save_session_id(self._anima_dir, message.session_id, session_type, thread_id=thread_id)
                    # Do NOT call tracker.update_from_result_message() here.
                    # ResultMessage.usage.input_tokens is a cumulative sum across
                    # all turns (not the current context size) and would
                    # produce inaccurate threshold checks.  Context tracking is
                    # handled per-turn via message_start events above.
                    if message.usage:
                        u = message.usage
                        usage_acc.input_tokens = u.get("input_tokens", 0) or 0
                        usage_acc.output_tokens = u.get("output_tokens", 0) or 0
                    break  # receive_messages() does not auto-stop on ResultMessage

                elif isinstance(message, SystemMessage):
                    if message.subtype == "init" and message.data:
                        mcp_servers = message.data.get("mcp_servers", [])
                        for srv in mcp_servers:
                            name = srv.get("name", "unknown")
                            status = srv.get("status", "unknown")
                            if status != "connected":
                                logger.error(
                                    "MCP server '%s' failed to connect: status=%s",
                                    name,
                                    status,
                                )
                            else:
                                logger.info("MCP server '%s' connected successfully", name)

        async def _run_fresh_session() -> AsyncGenerator[dict[str, Any], None]:
            """Run a fresh (no-resume) streaming session and yield events."""
            fresh_options, pf = self._build_sdk_options(
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
                async with ClaudeSDKClient(options=fresh_options) as fresh_client:
                    logger.info("ClaudeSDKClient connected (fresh session retry)")
                    async for event in _stream_messages(fresh_client):
                        yield event
            except BaseException as retry_exc:
                if isinstance(retry_exc, (asyncio.CancelledError, GeneratorExit)):
                    raise
                if not isinstance(retry_exc, Exception):
                    logger.critical(
                        "Agent SDK raised %s during streaming retry: %s",
                        type(retry_exc).__name__,
                        retry_exc,
                    )
                else:
                    logger.exception("Agent SDK streaming error (fresh session retry)")
                partial = "\n".join(response_text)
                raise StreamDisconnectedError(
                    f"Agent SDK stream error ({type(retry_exc).__name__}): {retry_exc}",
                    partial_text=partial,
                ) from retry_exc

        try:
            logger.info(
                "ClaudeSDKClient connecting (streaming mode, resume=%s)",
                session_id_to_resume,
            )
            if session_id_to_resume:
                # SDK Issue #387: an invalid/stale session ID causes the SDK to
                # hang for ~60 s before raising.  Guard the connection and first
                # event with RESUME_TIMEOUT_SEC; on timeout (or any SDK error)
                # clear the bad session ID and fall back to a fresh session.
                #
                # NOTE: The timeout guards "first yield from _stream_messages",
                # which occurs at content_block_start/delta — NOT at
                # message_start (which only updates the context tracker).
                # This means a valid resume where the model runs a long tool
                # before producing text could be falsely timed out.  In
                # practice 15 s is generous for the connection + first chunk
                # latency; long-running tools are rare on resume.
                fell_back = False
                try:
                    async with ClaudeSDKClient(options=options) as client:
                        logger.info("ClaudeSDKClient connected")
                        stream_gen = _stream_messages(client)

                        async def _get_first_event() -> dict[str, Any]:
                            return await stream_gen.__anext__()

                        try:
                            first_event = await asyncio.wait_for(_get_first_event(), timeout=RESUME_TIMEOUT_SEC)
                        except TimeoutError:
                            logger.warning(
                                "Resume timed out after %.1fs (SDK Issue #387, "
                                "session_id=%s), falling back to fresh session.",
                                RESUME_TIMEOUT_SEC,
                                session_id_to_resume,
                            )
                            await stream_gen.aclose()
                            _sdk_session._clear_session_id(self._anima_dir, session_type, thread_id=thread_id)
                            fell_back = True
                        except StopAsyncIteration:
                            logger.warning(
                                "Resume stream empty (session_id=%s), falling back to fresh session.",
                                session_id_to_resume,
                            )
                            _sdk_session._clear_session_id(self._anima_dir, session_type, thread_id=thread_id)
                            fell_back = True
                        else:
                            yield first_event
                            async for event in stream_gen:
                                yield event
                except (ProcessError, ClaudeSDKError) as e:
                    logger.warning(
                        "SDK session resume failed (session_id=%s): %s. Retrying with fresh session.",
                        session_id_to_resume,
                        e,
                    )
                    _sdk_session._clear_session_id(self._anima_dir, session_type, thread_id=thread_id)
                    fell_back = True

                if fell_back:
                    async for event in _run_fresh_session():
                        yield event
            else:
                async with ClaudeSDKClient(options=options) as client:
                    logger.info("ClaudeSDKClient connected")
                    async for event in _stream_messages(client):
                        yield event
            logger.debug("ClaudeSDKClient disconnected")
        except BaseException as e:
            if isinstance(e, (asyncio.CancelledError, GeneratorExit)):
                raise
            if not isinstance(e, Exception):
                logger.critical(
                    "Agent SDK raised %s during streaming: %s",
                    type(e).__name__,
                    e,
                )
            else:
                logger.exception("Agent SDK streaming error")
            partial = "\n".join(response_text)
            raise StreamDisconnectedError(
                f"Agent SDK stream error ({type(e).__name__}): {e}",
                partial_text=partial,
            ) from e
        finally:
            _cleanup_tool_outputs(self._anima_dir)
            _cleanup_prompt_files(_prompt_files)

        all_tool_records = _finalize_pending_records(pending_records)
        logger.debug(
            "Agent SDK streaming completed, messages=%d text_blocks=%d tools=%d",
            message_count,
            len(response_text),
            len(all_tool_records),
        )
        full_text = "\n".join(response_text) or "(no response)"
        replied_to = self._read_replied_to_file()
        yield {
            "type": "done",
            "full_text": full_text,
            "result_message": result_message,
            "replied_to_from_transcript": replied_to,
            "tool_call_records": [asdict(r) for r in all_tool_records],
            "force_chain": session_stats.get("force_chain", False),
            "usage": usage_acc.to_dict(),
        }

    async def compact_session(
        self,
        anima_dir: Path,
        session_type: str = "chat",
        thread_id: str = "default",
    ) -> bool:
        """Send /compact to an idle SDK session to trigger compaction.

        Resumes the session and sends ``/compact`` as the query.  The SDK
        compresses the transcript and the same session_id is preserved for
        future resume.  On failure the session file is **preserved** so
        that the next regular chat can still resume from the old state.

        Returns:
            True if compaction succeeded, False otherwise.
        """
        session_id = _load_session_id(anima_dir, session_type, thread_id)
        if not session_id:
            logger.info("No session to compact for %s/%s/%s", session_type, anima_dir.name, thread_id)
            return False

        logger.info(
            "Starting idle compaction (session=%s, type=%s, thread=%s)",
            session_id,
            session_type,
            thread_id,
        )

        try:
            from claude_code_sdk import ClaudeAgentOptions
            from claude_code_sdk import ClaudeCodeSDKClient as ClaudeSDKClient

            options = ClaudeAgentOptions(
                system_prompt=f"{anima_dir.name} session compaction",
                max_turns=1,
                resume=session_id,
            )

            found_session_id = False
            async with asyncio.timeout(RESUME_TIMEOUT_SEC):
                async with ClaudeSDKClient(options=options) as client:
                    await client.query("/compact")
                    async for message in client.receive_messages():
                        if hasattr(message, "session_id") and message.session_id:
                            _save_session_id(
                                anima_dir,
                                message.session_id,
                                session_type,
                                thread_id,
                            )
                            logger.info(
                                "Idle compaction completed (session=%s, type=%s, thread=%s)",
                                message.session_id,
                                session_type,
                                thread_id,
                            )
                            found_session_id = True

            if not found_session_id:
                logger.warning(
                    "Idle compaction did not receive session_id for %s/%s/%s",
                    anima_dir.name,
                    session_type,
                    thread_id,
                )
            return found_session_id
        except ImportError:
            logger.info("claude_code_sdk not available; skipping /compact")
            return False
        except TimeoutError:
            logger.warning(
                "Idle compaction timed out for %s/%s/%s; session preserved for next resume",
                anima_dir.name,
                session_type,
                thread_id,
            )
            return False
        except Exception:
            logger.warning(
                "Idle compaction failed for %s/%s/%s; session preserved for next resume",
                anima_dir.name,
                session_type,
                thread_id,
                exc_info=True,
            )
            return False
