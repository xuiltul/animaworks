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

import psutil

if TYPE_CHECKING:
    try:
        from claude_agent_sdk import ClaudeSDKClient, ResultMessage
    except ImportError:
        pass

from pathlib import Path

from core.exceptions import ExecutionError, LLMAPIError, MemoryWriteError  # noqa: F401
from core.execution import _sdk_session
from core.execution._claude_auth_lock import (
    ClaudeOAuthCircuitOpen,
    claude_execution_lock,
    trip_claude_oauth_circuit,
    trip_claude_oauth_circuit_from_result,
)
from core.execution._sdk_patch import apply_sdk_transport_patch

apply_sdk_transport_patch()

# ── Re-exports from submodules (backward compatibility) ──────
from core.execution._sdk_hooks import (  # noqa: F401
    _build_post_tool_hook,
    _build_pre_compact_hook,
    _build_pre_tool_hook,
    _cache_subordinate_paths,
    _read_status_json,
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
    RESUME_MAX_ATTEMPTS,
    RESUME_TIMEOUT_SEC,
    SESSION_TYPE_CHAT,
    SESSION_TYPE_CRON,
    SESSION_TYPE_HEARTBEAT,
    SESSION_TYPE_INBOX,
    SESSION_TYPE_TASK,
    SessionContextState,
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
    clear_session_id_for_type,
    compact_sdk_session,
    load_session_state,
    record_session_measurement,
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
from core.execution.error_classifier import (
    classify_llm_error_message,
    detect_cli_error_envelope,
    guard_key,
    provider_family_of,
)
from core.execution.rate_guard import get_rate_guard
from core.memory.shortterm import ShortTermMemory
from core.prompt.context import ContextTracker
from core.prompt.tokens import estimate_tokens
from core.schemas import ImageData, ModelConfig

logger = logging.getLogger("animaworks.execution.agent_sdk")

__all__ = ["AgentSDKExecutor", "StreamDisconnectedError"]


def _detect_sdk_auth_failure(text: str) -> str | None:
    """Return auth failure text when Claude Code surfaced a 401 auth error."""
    body = (text or "").strip()
    if not body:
        return None

    folded = body.casefold()
    auth_markers = (
        "failed to authenticate",
        "invalid authentication credentials",
        "authentication_error",
        "not authenticated",
    )
    if not any(marker in folded for marker in auth_markers):
        return None
    if not any(marker in folded for marker in ("401", "api error", "unauthorized", "auth")):
        return None
    return body


def _sdk_failure_text(result: Any, text: str, assistant_error: str | None = None) -> str | None:
    """Recognize SDK failure envelopes, not error words in normal answers."""
    result_text = getattr(result, "result", None)
    errors = getattr(result, "errors", None)
    details = [item for item in errors if isinstance(item, str)] if isinstance(errors, list) else []
    if isinstance(result_text, str) and result_text.strip():
        details.append(result_text.strip())
    subtype = getattr(result, "subtype", "")
    explicit_error = (
        getattr(result, "is_error", False) is True
        or (isinstance(subtype, str) and subtype.startswith("error_"))
        or bool(assistant_error)
    )
    if explicit_error:
        detail = "\n".join(details) or text.strip() or assistant_error or str(subtype) or "SDKError"
        error_status = {
            "authentication_failed": 401,
            "billing_error": 402,
            "rate_limit": 429,
            "invalid_request": 400,
            "server_error": 500,
        }.get(assistant_error or "")
        return f"API Error: {error_status} ({assistant_error})\n{detail}" if error_status else detail
    # Some Claude CLI transports omit is_error and print a synthetic API error
    # as assistant text. Restrict this compatibility path to their leading
    # envelope + known transport/status signature; prose mentioning an error
    # or a quoted log is still a successful model answer.
    return detect_cli_error_envelope(text)


# ── SDK subprocess PID tracking / cleanup ────────────


def _extract_sdk_pid(client: Any) -> int | None:
    """Return the Claude Agent SDK subprocess PID if available.

    Reads ``client._transport._process.pid`` defensively when the SDK
    wired a subprocess transport.

    Args:
        client: An active ``ClaudeSDKClient`` instance.

    Returns:
        Subprocess PID, or ``None`` when unavailable or invalid.
    """
    try:
        transport = getattr(client, "_transport", None)
        if transport is None:
            return None
        proc = getattr(transport, "_process", None)
        if proc is None:
            return None
        raw_pid = getattr(proc, "pid", None)
        if raw_pid is None:
            return None
        pid = int(raw_pid)
    except Exception:
        logger.debug("failed to extract SDK subprocess pid", exc_info=True)
        return None
    return pid if pid > 0 else None


def _kill_sdk_process(pid: int | None, create_time: float | None) -> None:
    """Best-effort terminate of a leaked SDK subprocess and its descendants.

    Verifies the PID still refers to the same OS process (when ``create_time``
    was recorded) and that the process name looks like Claude/node before
    sending signals. Never raises.

    Args:
        pid: Target PID from :func:`_extract_sdk_pid`, or ``None``.
        create_time: ``psutil.Process.create_time()`` captured while the client
            was connected, used to detect PID reuse; ``None`` skips this check.
    """
    if pid is None:
        return
    try:
        try:
            proc = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return

        if create_time is not None:
            try:
                actual_ct = float(proc.create_time())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return
            except Exception:
                logger.debug(
                    "SDK cleanup: cannot read create_time for pid=%s",
                    pid,
                    exc_info=True,
                )
                return
            if abs(actual_ct - create_time) > 2.0:
                logger.debug(
                    "SDK cleanup: skipping kill pid=%s (create_time mismatch, possible PID reuse)",
                    pid,
                )
                return

        try:
            name = (proc.name() or "").lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return
        except Exception:
            logger.debug("SDK cleanup: cannot read name for pid=%s", pid, exc_info=True)
            return
        if "claude" not in name and "node" not in name:
            logger.debug(
                "SDK cleanup: skipping kill pid=%s (unexpected process name %r)",
                pid,
                name,
            )
            return

        children = proc.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
            except Exception:
                logger.debug(
                    "SDK cleanup: failed to kill child pid=%s",
                    getattr(child, "pid", None),
                    exc_info=True,
                )
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
        except Exception:
            logger.debug("SDK cleanup: failed to kill pid=%s", pid, exc_info=True)
            return

        logger.info("terminated leaked Claude SDK subprocess tree (pid=%s)", pid)
    except Exception:
        logger.debug("SDK cleanup: unexpected error for pid=%s", pid, exc_info=True)


# ── Resume fallback context recovery (Fix 4b) ────────────────


def _build_resume_fallback_handoff(
    anima_dir: Path,
    session_type: str,
    thread_id: str,
    failed_session_id: str | None,
) -> str:
    """Recover conversation context after a failed SDK session resume.

    Fix 4b: mid-conversation (below the context threshold, before idle
    compaction fires) the SDK session transcript is the *only* holder of
    the conversation memory — no shortterm handoff exists.  When a resume
    fails, falling back to a plain fresh session therefore loses the whole
    conversation ("total amnesia").

    This reuses ``session_compactor._extract_recent_chat_context`` to build
    an activity_log digest of the recent conversation, persists it as a
    shortterm handoff with ``trigger="resume_fallback"`` (so the context
    also survives if the fresh session itself fails), and returns a
    rendered digest block for injection into the fresh-session prompt —
    turning total loss into graceful degradation.

    A pending higher-fidelity handoff (e.g. the context-threshold save) is
    never overwritten (same protection as Fix 1); the digest is then only
    injected into the current prompt.

    Best-effort: returns ``""`` when there is nothing to recover or on any
    error, in which case the fresh session proceeds exactly as before.
    """
    if session_type != SESSION_TYPE_CHAT:
        # The digest extractor scans chat activity_log entries only.
        return ""
    try:
        from core.memory.shortterm import SessionState, ShortTermMemory
        from core.session_compactor import _extract_recent_chat_context

        extracted = _extract_recent_chat_context(anima_dir, thread_id=thread_id)
        if not (extracted.get("accumulated_response") or extracted.get("tool_uses")):
            return ""
        digest_state = SessionState(
            accumulated_response=extracted.get("accumulated_response", ""),
            tool_uses=extracted.get("tool_uses", []),
            original_prompt=extracted.get("original_prompt", ""),
            timestamp=extracted.get("timestamp", ""),
            trigger="resume_fallback",
            notes=(f"Auto-extracted from activity_log after SDK resume failure (session_id={failed_session_id})"),
        )
        shortterm = ShortTermMemory(anima_dir, session_type=session_type, thread_id=thread_id)
        existing = shortterm.load()
        if existing is not None and existing.trigger not in ("", "idle_compaction", "resume_fallback"):
            logger.info(
                "resume_fallback: pending shortterm preserved (trigger=%s); digest injected without save (thread=%s)",
                existing.trigger,
                thread_id,
            )
        else:
            shortterm.save(digest_state)
            logger.info(
                "resume_fallback: shortterm digest saved from activity_log (session_id=%s, thread=%s)",
                failed_session_id,
                thread_id,
            )
        return shortterm._render_markdown(digest_state)
    except Exception:
        logger.warning("resume_fallback: context recovery failed", exc_info=True)
        return ""


def _inject_recovered_context(prompt: str, recovered: str) -> str:
    """Prepend the recovered conversation digest to the fresh-session prompt."""
    return (
        "<recovered_context>\n"
        "The previous session could not be resumed. The following digest was "
        "recovered from the activity log. Treat it as the immediately "
        "preceding conversation and continue seamlessly.\n\n"
        f"{recovered}\n"
        "</recovered_context>\n\n"
        f"{prompt}"
    )


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
        self._active_client: ClaudeSDKClient | None = None

    @property
    def supports_streaming(self) -> bool:  # noqa: D102
        return True

    @property
    def supports_message_injection(self) -> bool:  # noqa: D102
        return True

    async def inject_message(self, message: str) -> bool:
        """Send an additional user turn to the active bidirectional SDK client."""
        client = self._active_client
        if client is None:
            return False
        await client.query(message)
        return True

    def _init_session_stats(self, system_prompt: str, prompt: str, trigger: str) -> dict[str, Any]:
        """Build the mutable session-stats dict shared with PreToolUse hook."""
        return {
            "tool_call_count": 0,
            "total_result_bytes": 0,
            "system_prompt_tokens": estimate_tokens(system_prompt),
            "user_prompt_tokens": estimate_tokens(prompt),
            "force_chain": False,
            "trigger": trigger,
            "start_time": time.monotonic(),
            "hb_soft_warned": False,
            "hb_soft_timeout": self._hb_soft_timeout_s,
        }

    def _should_retry_sdk_auth_failure(self) -> bool:
        """Do not multiply SDK processes after a shared-OAuth auth failure."""
        return False

    def _rate_guard_preflight(self) -> None:
        """Log when this model's realm is rate-guarded (start-time suppression only).

        The SDK owns its internal retries, so a guarded realm does not defer the
        session — this is observability so a fleet-wide throttle is visible at
        session start.  Keyed on the Mode-S auth realm the SDK authenticates
        against (``max``/``api``/``bedrock``/``vertex``), which is independent of
        the API-key ``api`` realm used by LiteLLM.
        """
        family = provider_family_of(self._model_config.model)
        realm = self._model_config.mode_s_auth or "max"
        key = guard_key(family, realm)
        blocked = get_rate_guard().blocked_remaining(key)
        if blocked > 0:
            logger.info(
                "S session start: %s rate-guarded for %.0fs (continuing; SDK retries apply)",
                key,
                blocked,
            )

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
                session_id = getattr(message, "session_id", "")
                if session_id and session_type in _RESUMABLE_SESSION_TYPES:
                    _save_session_id(self._anima_dir, session_id, session_type, thread_id=thread_id)
                if tracker:
                    tracker.update_from_result_message(message.usage)
                if usage_acc and message.usage:
                    u = message.usage
                    usage_acc.input_tokens = u.get("input_tokens", 0) or 0
                    usage_acc.output_tokens = u.get("output_tokens", 0) or 0
                    usage_acc.cache_read_tokens = u.get("cache_read_input_tokens", 0) or 0
                    usage_acc.cache_write_tokens = u.get("cache_creation_input_tokens", 0) or 0
            elif isinstance(message, AssistantMessage):
                sdk_error = getattr(message, "error", None)
                if isinstance(sdk_error, str) and sdk_error:
                    session_stats["sdk_error"] = sdk_error
                    assistant_error_text = "\n".join(
                        block.text for block in message.content if isinstance(block, TextBlock)
                    )
                    if assistant_error_text:
                        session_stats.setdefault("assistant_error_text", []).append(assistant_error_text)
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
        thread_id: str = "default",
    ) -> ExecutionResult:
        """Run a session via Claude Agent SDK with context monitoring hook."""
        from claude_agent_sdk import ClaudeSDKClient, ClaudeSDKError, ProcessError

        self._rate_guard_preflight()
        _cw = self._resolve_cw()
        session_stats = self._init_session_stats(system_prompt, prompt, trigger)
        session_type = _resolve_session_type(trigger)
        if session_type in _RESUMABLE_SESSION_TYPES:
            session_id_to_resume = _load_session_id(self._anima_dir, session_type, thread_id=thread_id)
        else:
            _sdk_session.clear_session_id_for_type(self._anima_dir, session_type, thread_id=thread_id)
            session_id_to_resume = None

        options, _temp_files = self._build_sdk_options(
            system_prompt,
            _cw,
            session_stats,
            resume=session_id_to_resume,
        )
        _prompt_files: list[Path] = list(_temp_files)
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

        sdk_pid: int | None = None
        sdk_pid_create_time: float | None = None

        async def _run_blocking_client(run_options, *, log_label: str) -> ResultMessage | None:
            nonlocal sdk_pid, sdk_pid_create_time
            logger.info("ClaudeSDKClient connecting (%s, resume=%s)", log_label, getattr(run_options, "resume", None))
            async with (
                claude_execution_lock(getattr(run_options, "env", None)),
                ClaudeSDKClient(options=run_options) as client,
            ):
                logger.info("ClaudeSDKClient connected")
                sdk_pid = _extract_sdk_pid(client)
                sdk_pid_create_time = None
                if sdk_pid is not None:
                    try:
                        sdk_pid_create_time = float(psutil.Process(sdk_pid).create_time())
                    except Exception:
                        logger.debug(
                            "failed to read create_time for SDK subprocess pid=%s",
                            sdk_pid,
                            exc_info=True,
                        )
                result = await self._process_blocking_messages(client, **_msg_args)
                trip_claude_oauth_circuit_from_result(
                    getattr(run_options, "env", None),
                    result,
                    "\n".join(response_text),
                    "\n".join(session_stats.get("assistant_error_text", ())),
                )
                return result

        try:
            result_message = await _run_blocking_client(options, log_label="blocking mode")
            logger.debug("ClaudeSDKClient disconnected")
        except (ProcessError, ClaudeSDKError) as e:
            if trip_claude_oauth_circuit(getattr(options, "env", None), str(e)):
                logger.error("Claude OAuth revoked; fleet-wide circuit opened")
                return ExecutionResult(
                    text=f"[Agent SDK Error: {e}]",
                    tool_call_records=_finalize_pending_records(pending_records),
                )
            if session_id_to_resume:
                logger.warning("SDK session resume failed (session_id=%s): %s", session_id_to_resume, e)
                _sdk_session._clear_session_id(self._anima_dir, session_type, thread_id=thread_id)
                options, tfs = self._build_sdk_options(system_prompt, _cw, session_stats, resume=None)
                _prompt_files.extend(tfs)
                try:
                    result_message = await _run_blocking_client(options, log_label="blocking mode fresh session retry")
                except Exception as retry_exc:
                    logger.exception("Agent SDK execution error (fresh session retry)")
                    return ExecutionResult(
                        text=f"[Agent SDK Error: {retry_exc}]\n" + "\n".join(response_text),
                        tool_call_records=_finalize_pending_records(pending_records),
                        error=True,
                        usage=usage_acc,
                    )
            else:
                logger.exception("Agent SDK execution error")
                return ExecutionResult(
                    text=f"[Agent SDK Error: {e}]\n" + "\n".join(response_text),
                    tool_call_records=_finalize_pending_records(pending_records),
                    error=True,
                    usage=usage_acc,
                )
        except Exception as e:
            logger.exception("Agent SDK execution error")
            return ExecutionResult(
                text=f"[Agent SDK Error: {e}]\n" + "\n".join(response_text),
                tool_call_records=_finalize_pending_records(pending_records),
                error=True,
                usage=usage_acc,
            )
        finally:
            _kill_sdk_process(sdk_pid, sdk_pid_create_time)
            _cleanup_tool_outputs(self._anima_dir)
            _cleanup_prompt_files(_prompt_files)

        auth_failure_text = (
            _sdk_failure_text(result_message, "\n".join(response_text), session_stats.get("sdk_error")) or ""
        )
        auth_failure = _detect_sdk_auth_failure(auth_failure_text)
        if trip_claude_oauth_circuit_from_result(
            getattr(options, "env", None),
            result_message,
            "\n".join(response_text),
            "\n".join(session_stats.get("assistant_error_text", ())),
        ):
            logger.error("Claude OAuth revoked; fleet-wide circuit opened")
        if auth_failure and self._should_retry_sdk_auth_failure():
            logger.warning("Claude SDK returned auth failure text; retrying fresh session once")
            response_text.clear()
            pending_records.clear()
            result_message = None
            usage_acc = TokenUsage()
            session_stats.pop("sdk_error", None)
            session_stats.pop("assistant_error_text", None)
            _msg_args["usage_acc"] = usage_acc
            if session_type in _RESUMABLE_SESSION_TYPES:
                _sdk_session._clear_session_id(self._anima_dir, session_type, thread_id=thread_id)
            retry_options, retry_files = self._build_sdk_options(
                system_prompt,
                _cw,
                session_stats,
                resume=None,
            )
            try:
                result_message = await _run_blocking_client(
                    retry_options,
                    log_label="blocking mode auth failure retry",
                )
            except Exception as retry_exc:
                logger.exception("Agent SDK auth failure retry failed")
                return ExecutionResult(
                    text=f"[Agent SDK Error: {retry_exc}]",
                    tool_call_records=[],
                    error=True,
                    usage=usage_acc,
                )
            finally:
                _kill_sdk_process(sdk_pid, sdk_pid_create_time)
                _cleanup_tool_outputs(self._anima_dir)
                _cleanup_prompt_files(retry_files)

        all_tool_records = _finalize_pending_records(pending_records)
        replied_to = self._read_replied_to_file()
        failure = _sdk_failure_text(result_message, "\n".join(response_text), session_stats.get("sdk_error"))
        return ExecutionResult(
            text=failure or "\n".join(response_text) or "(no response)",
            result_message=result_message,
            replied_to_from_transcript=replied_to,
            tool_call_records=all_tool_records,
            force_chain=session_stats.get("force_chain", False),
            usage=usage_acc,
            error=failure is not None,
        )

    # ── Streaming execution ──────────────────────────────────

    async def execute_streaming(
        self,
        system_prompt: str,
        prompt: str,
        tracker: ContextTracker,
        images: list[ImageData] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        trigger: str = "",
        thread_id: str = "default",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream events from Claude Agent SDK."""
        from claude_agent_sdk import ClaudeSDKClient, ClaudeSDKError, ProcessError

        self._rate_guard_preflight()
        _cw = self._resolve_cw()
        session_stats = self._init_session_stats(system_prompt, prompt, trigger)
        session_type = _resolve_session_type(trigger)
        if session_type in _RESUMABLE_SESSION_TYPES:
            session_id_to_resume = _load_session_id(self._anima_dir, session_type, thread_id=thread_id)
        else:
            _sdk_session.clear_session_id_for_type(self._anima_dir, session_type, thread_id=thread_id)
            session_id_to_resume = None

        options, _temp_files = self._build_sdk_options(
            system_prompt,
            _cw,
            session_stats,
            resume=session_id_to_resume,
            include_partial_messages=True,
        )
        _prompt_files: list[Path] = list(_temp_files)
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
        emitted_text_delta = False

        sdk_pid: int | None = None
        sdk_pid_create_time: float | None = None

        async def _fresh_session() -> AsyncGenerator[dict[str, Any], None]:
            nonlocal sdk_pid, sdk_pid_create_time
            fresh_opts, tfs = self._build_sdk_options(
                system_prompt,
                _cw,
                session_stats,
                resume=None,
                include_partial_messages=True,
            )
            _prompt_files.extend(tfs)
            try:
                async with (
                    claude_execution_lock(getattr(fresh_opts, "env", None)),
                    ClaudeSDKClient(options=fresh_opts) as fc,
                ):
                    logger.info("ClaudeSDKClient connected (fresh session retry)")
                    self._active_client = fc
                    sdk_pid = _extract_sdk_pid(fc)
                    sdk_pid_create_time = None
                    if sdk_pid is not None:
                        try:
                            sdk_pid_create_time = float(psutil.Process(sdk_pid).create_time())
                        except Exception:
                            logger.debug(
                                "failed to read create_time for SDK subprocess pid=%s",
                                sdk_pid,
                                exc_info=True,
                            )
                    try:
                        async for ev in process_stream_messages(fc, ctx, state):
                            yield ev
                        trip_claude_oauth_circuit_from_result(
                            getattr(fresh_opts, "env", None),
                            state.result_message,
                            "\n".join(state.response_text),
                            "\n".join(state.assistant_error_text),
                        )
                    finally:
                        if self._active_client is fc:
                            self._active_client = None
            except BaseException as exc:
                if isinstance(exc, (asyncio.CancelledError, GeneratorExit, ClaudeOAuthCircuitOpen)):
                    raise
                logger.exception("Agent SDK streaming error (fresh session retry)")
                raise StreamDisconnectedError(
                    f"Agent SDK stream error ({type(exc).__name__}): {exc}",
                    partial_text="\n".join(state.response_text),
                ) from exc

        async def _run_stream_options(run_options, *, resume_guard: bool) -> AsyncGenerator[dict[str, Any], None]:
            nonlocal emitted_text_delta, sdk_pid, sdk_pid_create_time
            async with (
                claude_execution_lock(getattr(run_options, "env", None)),
                ClaudeSDKClient(options=run_options) as client,
            ):
                logger.info("ClaudeSDKClient connected")
                self._active_client = client
                sdk_pid = _extract_sdk_pid(client)
                sdk_pid_create_time = None
                if sdk_pid is not None:
                    try:
                        sdk_pid_create_time = float(psutil.Process(sdk_pid).create_time())
                    except Exception:
                        logger.debug(
                            "failed to read create_time for SDK subprocess pid=%s",
                            sdk_pid,
                            exc_info=True,
                        )
                try:
                    gen = process_stream_messages(client, ctx, state)
                    if resume_guard:
                        try:
                            first = await asyncio.wait_for(gen.__anext__(), timeout=RESUME_TIMEOUT_SEC)
                        except TimeoutError:
                            logger.warning("Resume timed out (session_id=%s)", session_id_to_resume)
                            await gen.aclose()
                            # Fix 4c: do NOT clear the session id here — the
                            # session file in ~/.claude survives a first-event
                            # timeout, so the caller retries the resume before
                            # discarding the id for good.
                            raise
                        except StopAsyncIteration:
                            logger.warning("Resume stream empty (session_id=%s)", session_id_to_resume)
                            _sdk_session._clear_session_id(self._anima_dir, session_type, thread_id=thread_id)
                            raise
                        else:
                            if first.get("type") == "text_delta":
                                emitted_text_delta = True
                            yield first
                    async for ev in gen:
                        if ev.get("type") == "text_delta":
                            emitted_text_delta = True
                        yield ev
                    trip_claude_oauth_circuit_from_result(
                        getattr(run_options, "env", None),
                        state.result_message,
                        "\n".join(state.response_text),
                        "\n".join(state.assistant_error_text),
                    )
                finally:
                    if self._active_client is client:
                        self._active_client = None

        try:
            logger.info("ClaudeSDKClient connecting (streaming, resume=%s)", session_id_to_resume)
            if session_id_to_resume:
                fell_back = False
                for resume_attempt in range(1, RESUME_MAX_ATTEMPTS + 1):
                    resume_yielded = False
                    try:
                        async for ev in _run_stream_options(options, resume_guard=True):
                            resume_yielded = True
                            yield ev
                        break
                    except TimeoutError:
                        if resume_yielded:
                            # Mid-stream timeout after the resume handshake
                            # succeeded — not a resume failure; keep the old
                            # behavior (fresh-session fallback, no retry).
                            fell_back = True
                            break
                        # Fix 4c: first-event timeout — retry the resume once
                        # before discarding the session id (the session file
                        # in ~/.claude is still intact; the timeout is usually
                        # transient host load, not a broken session).
                        if resume_attempt < RESUME_MAX_ATTEMPTS:
                            logger.warning(
                                "Resume attempt %d/%d timed out; retrying (session_id=%s)",
                                resume_attempt,
                                RESUME_MAX_ATTEMPTS,
                                session_id_to_resume,
                            )
                            continue
                        logger.warning(
                            "Resume failed after %d attempts; discarding session_id=%s",
                            RESUME_MAX_ATTEMPTS,
                            session_id_to_resume,
                        )
                        _sdk_session._clear_session_id(self._anima_dir, session_type, thread_id=thread_id)
                        fell_back = True
                        break
                    except StopAsyncIteration:
                        fell_back = True
                        break
                    except (ProcessError, ClaudeSDKError) as e:
                        if trip_claude_oauth_circuit(getattr(options, "env", None), str(e)):
                            logger.error("Claude OAuth revoked; fleet-wide circuit opened")
                            raise StreamDisconnectedError(
                                f"Agent SDK stream error ({type(e).__name__}): {e}",
                                partial_text="\n".join(state.response_text),
                            ) from e
                        logger.warning("SDK resume failed (session_id=%s): %s", session_id_to_resume, e)
                        _sdk_session._clear_session_id(self._anima_dir, session_type, thread_id=thread_id)
                        fell_back = True
                        break
                    except ClaudeOAuthCircuitOpen:
                        raise
                    except Exception as e:
                        logger.warning(
                            "SDK resume failed with unexpected error (session_id=%s): %s", session_id_to_resume, e
                        )
                        _sdk_session._clear_session_id(self._anima_dir, session_type, thread_id=thread_id)
                        fell_back = True
                        break
                if fell_back:
                    # Fix 4b: recover the recent conversation from the
                    # activity_log so the fresh session does not start with
                    # total amnesia. Best-effort — on failure the fallback
                    # behaves exactly as before.
                    recovered = _build_resume_fallback_handoff(
                        self._anima_dir,
                        session_type,
                        thread_id,
                        session_id_to_resume,
                    )
                    if recovered:
                        from dataclasses import replace as _dc_replace

                        ctx = _dc_replace(ctx, prompt=_inject_recovered_context(prompt, recovered))
                    async for ev in _fresh_session():
                        yield ev
            else:
                async for ev in _run_stream_options(options, resume_guard=False):
                    yield ev
            logger.debug("ClaudeSDKClient disconnected")
        except BaseException as e:
            if isinstance(e, (asyncio.CancelledError, GeneratorExit, ClaudeOAuthCircuitOpen)):
                raise
            logger.exception("Agent SDK streaming error")
            raise StreamDisconnectedError(
                f"Agent SDK stream error ({type(e).__name__}): {e}",
                partial_text="\n".join(state.response_text),
            ) from e
        finally:
            _kill_sdk_process(sdk_pid, sdk_pid_create_time)
            _cleanup_tool_outputs(self._anima_dir)
            _cleanup_prompt_files(_prompt_files)

        auth_failure_text = (
            _sdk_failure_text(state.result_message, "\n".join(state.response_text), state.sdk_error) or ""
        )
        auth_failure = _detect_sdk_auth_failure(auth_failure_text)
        if trip_claude_oauth_circuit_from_result(
            getattr(options, "env", None),
            state.result_message,
            "\n".join(state.response_text),
            "\n".join(state.assistant_error_text),
        ):
            logger.error("Claude OAuth revoked; fleet-wide circuit opened")
        if auth_failure and self._should_retry_sdk_auth_failure() and not emitted_text_delta:
            logger.warning("Claude SDK returned auth failure text during streaming; retrying fresh session once")
            if session_type in _RESUMABLE_SESSION_TYPES:
                _sdk_session._clear_session_id(self._anima_dir, session_type, thread_id=thread_id)
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
            emitted_text_delta = False
            async for ev in _fresh_session():
                if ev.get("type") == "text_delta":
                    emitted_text_delta = True
                yield ev

        all_tool_records = _finalize_pending_records(state.pending_records)
        full_text = "\n".join(state.response_text) or "(no response)"
        failure = _sdk_failure_text(state.result_message, "\n".join(state.response_text), state.sdk_error)
        if failure and not state.interrupted:
            reason, _hint = classify_llm_error_message(failure)
            yield {
                "type": "error",
                "terminal": True,
                "message": failure,
                "reason": reason.value,
                "usage": state.usage_acc.to_dict(),
                "tool_call_records": [asdict(r) for r in all_tool_records],
            }
            return
        replied_to = self._read_replied_to_file()
        yield {
            "type": "done",
            "full_text": full_text,
            "result_message": state.result_message,
            "stop_kind": "interrupted"
            if state.interrupted
            else "empty_response"
            if full_text == "(no response)"
            else "normal",
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
