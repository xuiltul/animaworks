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
"""

import asyncio
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from collections.abc import AsyncGenerator, Callable
from dataclasses import asdict
from datetime import datetime
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    try:
        from claude_code_sdk import ClaudeCodeSDKClient as ClaudeSDKClient, ClaudeAgentOptions, ResultMessage
    except ImportError:
        pass

from core.prompt.context import CHARS_PER_TOKEN, ContextTracker, resolve_context_window
from core.exceptions import ExecutionError, LLMAPIError, MemoryWriteError  # noqa: F401
from core.execution.base import BaseExecutor, ExecutionResult, StreamDisconnectedError, ToolCallRecord, _truncate_for_record, tool_input_save_budget, tool_result_save_budget
from core.execution.reminder import MSG_CONTEXT_THRESHOLD
from core.schemas import ModelConfig
from core.memory.shortterm import ShortTermMemory
from pathlib import Path

logger = logging.getLogger("animaworks.execution.agent_sdk")


# Re-export for backward compatibility (agent.py imports from here)
__all__ = ["AgentSDKExecutor", "StreamDisconnectedError", "clear_session_ids"]


# ── Mode S Bash blocklist ────────────────────────────────────

# Patterns that are unconditionally blocked in Bash tool calls.
# Each entry is (compiled_regex, human-readable reason).
_BASH_BLOCKED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"chatwork\s+send", re.IGNORECASE),
     "Chatwork send via Bash is blocked; use the mcp__aw__chatwork_send tool instead"),
    (re.compile(r"chatwork_cli\.py\s+send", re.IGNORECASE),
     "Chatwork CLI send via Bash is blocked; use the mcp__aw__chatwork_send tool instead"),
    (re.compile(r"curl.*api\.chatwork\.com.*/messages", re.IGNORECASE),
     "Direct Chatwork API post is blocked; use the mcp__aw__chatwork_send tool instead"),
    (re.compile(r"wget.*api\.chatwork\.com.*/messages", re.IGNORECASE),
     "Direct Chatwork API post via wget is blocked; use the mcp__aw__chatwork_send tool instead"),
]

# ── Mode S security ──────────────────────────────────────────

# Files that animas cannot modify themselves (privilege/bootstrap protection).
# identity.md and specialty_prompt.md are intentionally editable —
# Animas can evolve their personality and specialization over time.
_PROTECTED_FILES = frozenset({
    "permissions.md",
    "bootstrap.md",
})

# Commands that can write files (checked for path traversal).
_WRITE_COMMANDS = frozenset({
    "cp", "mv", "tee", "dd", "install", "rsync",
})

# Safety margin for Agent SDK JSON-RPC buffer.  The default (1 MB) is too
# small when system_prompt + conversation history grow large; 4 MB gives
# comfortable headroom while still catching genuinely broken messages.
_SDK_MAX_BUFFER_SIZE = 4 * 1024 * 1024  # 4 MB

# Linux MAX_ARG_STRLEN is 128 KiB (131072 bytes) per argument — a kernel
# compile-time constant that cannot be changed at runtime.  When the
# system prompt exceeds this limit, `execve` fails with E2BIG ([Errno 7]).
# We use a conservative threshold (100 KB) to leave headroom for encoding
# overhead and other arguments.  When exceeded, the prompt is written to a
# temp file and passed via the CLI's undocumented --system-prompt-file flag.
_PROMPT_FILE_THRESHOLD = 100_000  # 100 KB


def _is_debug_superuser(anima_dir: Path) -> bool:
    """Check if an anima has debug_superuser flag in status.json."""
    status_path = anima_dir / "status.json"
    if not status_path.is_file():
        return False
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
        return bool(data.get("debug_superuser"))
    except (json.JSONDecodeError, OSError):
        return False

# SDK Issue #387: invalid session ID causes SDK to hang for ~60s before
# raising an error.  We wrap the first-event receive in asyncio.wait_for
# so that a stale/invalid resume fails fast and falls back to a fresh session.
RESUME_TIMEOUT_SEC = 15.0


def _cleanup_prompt_files(files: list[Path]) -> None:
    """Remove temp prompt files created for --system-prompt-file."""
    for f in files:
        try:
            f.unlink(missing_ok=True)
        except OSError:
            logger.debug("Failed to remove temp prompt file: %s", f)

# When estimated context usage leaves fewer than max_tokens * this factor
# free, the PreToolUse hook triggers session termination for auto-compact.
_CONTEXT_AUTOCOMPACT_SAFETY = 2


# ── Session ID persistence ───────────────────────────────────

def _session_file(session_type: str) -> str:
    """Return the session file name for the given session type."""
    return f"current_session_{session_type}.json"


def _load_session_id(anima_dir: Path, session_type: str = "chat") -> str | None:
    """Load persisted session ID for SDK session resume."""
    path = anima_dir / "state" / _session_file(session_type)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("session_id")
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to load session ID from %s", path)
        return None


def _save_session_id(anima_dir: Path, session_id: str, session_type: str = "chat") -> None:
    """Persist session ID for future SDK session resume."""
    path = anima_dir / "state" / _session_file(session_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
    }, ensure_ascii=False), encoding="utf-8")


def _clear_session_id(anima_dir: Path, session_type: str = "chat") -> None:
    """Clear persisted session ID (e.g., after resume failure)."""
    path = anima_dir / "state" / _session_file(session_type)
    if path.exists():
        path.unlink(missing_ok=True)


def clear_session_ids(anima_dir: Path) -> None:
    """Clear all session IDs for an anima (chat and heartbeat).

    Public wrapper for use by streaming_handler.py on done=False disconnection.
    """
    for session_type in ("chat", "heartbeat"):
        _clear_session_id(anima_dir, session_type)


def _check_a1_file_access(
    file_path: str,
    anima_dir: Path,
    *,
    write: bool,
    subordinate_activity_dirs: list[Path] | None = None,
    subordinate_management_files: list[Path] | None = None,
    superuser: bool = False,
) -> str | None:
    """Check if a file path is allowed for Mode S tools.

    Returns violation reason string if blocked, None if allowed.
    """
    if superuser:
        return None
    if not file_path:
        return None

    resolved = Path(file_path).resolve()
    anima_resolved = anima_dir.resolve()
    animas_root = anima_resolved.parent

    # Block access to other animas' directories
    if resolved.is_relative_to(animas_root):
        if not resolved.is_relative_to(anima_resolved):
            # Supervisor can read subordinate's activity_log
            if not write and subordinate_activity_dirs:
                for sub_activity in subordinate_activity_dirs:
                    if resolved.is_relative_to(sub_activity):
                        return None

            # Supervisor can read/write subordinate's cron.md & heartbeat.md
            if subordinate_management_files:
                for mgmt_file in subordinate_management_files:
                    if resolved == mgmt_file:
                        return None

            return f"Access to other anima's directory is not allowed: {file_path}"

        # Block writes to protected files within own directory
        if write:
            rel = str(resolved.relative_to(anima_resolved))
            if rel in _PROTECTED_FILES:
                return f"'{rel}' is a protected file and cannot be modified"

    return None


def _check_a1_bash_command(
    command: str,
    anima_dir: Path,
    *,
    superuser: bool = False,
) -> str | None:
    """Check bash commands against blocklist patterns and file operation violations.

    Blocklist patterns are matched against the raw command string (before
    shlex parsing) to prevent bypass via pipes/subshells.  Path traversal
    checks use parsed argv for precision.

    This is a best-effort heuristic — not a complete sandbox.
    """
    if superuser:
        return None
    # Blocklist check (raw command string, before shlex parsing)
    for pattern, reason in _BASH_BLOCKED_PATTERNS:
        if pattern.search(command):
            logger.warning("Bash command blocked: %s (command: %s)", reason, command[:200])
            return reason

    import shlex

    try:
        argv = shlex.split(command)
    except ValueError:
        return None

    if not argv:
        return None

    cmd_base = Path(argv[0]).name

    # Check file-writing commands for path violations
    if cmd_base in _WRITE_COMMANDS:
        animas_root = str(anima_dir.parent.resolve())
        anima_resolved = str(anima_dir.resolve())
        for arg in argv[1:]:
            if arg.startswith("-"):
                continue
            try:
                resolved = str(Path(arg).resolve())
                # Writing to other anima's directory
                if resolved.startswith(animas_root) and not resolved.startswith(
                    anima_resolved
                ):
                    return f"Command targets other anima's directory: {arg}"
            except (ValueError, OSError):
                pass

    return None


# ── Mode S output guard ──────────────────────────────────────

_BASH_TRUNCATE_BYTES = 10_000   # 10 KB
_BASH_HEAD_BYTES = 5_000        # head display
_BASH_TAIL_BYTES = 3_000        # tail display
_READ_DEFAULT_LIMIT = 500       # lines
_GREP_DEFAULT_HEAD_LIMIT = 200  # entries
_GLOB_DEFAULT_HEAD_LIMIT = 500  # entries


def _build_output_guard(
    tool_name: str,
    tool_input: dict[str, Any],
    anima_dir: Path,
) -> dict[str, Any] | None:
    """Build updatedInput for output size control.

    Returns modified tool_input dict, or None if no modification needed.
    """
    if tool_name == "Bash":
        return _guard_bash(tool_input, anima_dir)
    if tool_name == "Read":
        return _guard_read(tool_input)
    if tool_name == "Grep":
        return _guard_grep(tool_input)
    if tool_name == "Glob":
        return _guard_glob(tool_input)
    return None


def _guard_bash(tool_input: dict[str, Any], anima_dir: Path) -> dict[str, Any]:
    """Wrap bash command to save full output to file and truncate display."""
    command = tool_input.get("command", "")
    if not command:
        return tool_input

    out_dir = anima_dir / "shortterm" / "tool_outputs"

    wrapped = (
        f'_OUTDIR="{out_dir}"\n'
        f'mkdir -p "$_OUTDIR"\n'
        f'_OUTF="$_OUTDIR/bash_$(date +%s%N).txt"\n'
        f'{{ {command} ; }} > "$_OUTF" 2>&1\n'
        f'_EC=$?\n'
        f'_SZ=$(wc -c < "$_OUTF")\n'
        f'if [ "$_SZ" -gt {_BASH_TRUNCATE_BYTES} ]; then\n'
        f'  head -c {_BASH_HEAD_BYTES} "$_OUTF"\n'
        f'  echo ""\n'
        f'  echo "... [truncated: $_SZ bytes total] ..."\n'
        f'  echo ""\n'
        f'  tail -c {_BASH_TAIL_BYTES} "$_OUTF"\n'
        f'  echo ""\n'
        f'  echo "[Full output saved: $_OUTF]"\n'
        f'  echo "[Use Read tool with file_path=$_OUTF to view full content]"\n'
        f'else\n'
        f'  cat "$_OUTF"\n'
        f'  rm -f "$_OUTF"\n'
        f'fi\n'
        f'exit $_EC'
    )
    return {**tool_input, "command": wrapped}


def _guard_read(tool_input: dict[str, Any]) -> dict[str, Any] | None:
    """Inject default limit for Read if not specified."""
    if "limit" in tool_input and tool_input["limit"] is not None:
        return None  # agent explicitly specified -> pass through
    return {**tool_input, "limit": _READ_DEFAULT_LIMIT}


def _guard_grep(tool_input: dict[str, Any]) -> dict[str, Any] | None:
    """Inject default head_limit for Grep if not specified."""
    if "head_limit" in tool_input and tool_input["head_limit"] is not None:
        return None
    return {**tool_input, "head_limit": _GREP_DEFAULT_HEAD_LIMIT}


def _guard_glob(tool_input: dict[str, Any]) -> dict[str, Any] | None:
    """Inject default head_limit for Glob if not specified."""
    if "head_limit" in tool_input and tool_input["head_limit"] is not None:
        return None
    return {**tool_input, "head_limit": _GLOB_DEFAULT_HEAD_LIMIT}


async def _image_prompt_messages(
    prompt: str,
    images: list[dict[str, Any]],
) -> AsyncGenerator[dict[str, Any], None]:
    """Yield a single SDK user message with image content blocks.

    The Agent SDK ``query()`` accepts ``str | AsyncIterable[dict]``.
    When images are present we build an Anthropic-format multimodal
    content block list and wrap it in the SDK's message envelope.
    """
    content_blocks: list[dict[str, Any]] = []
    for img in images:
        content_blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": img["media_type"],
                "data": img["data"],
            },
        })
    content_blocks.append({"type": "text", "text": prompt})
    yield {
        "type": "user",
        "message": {"role": "user", "content": content_blocks},
        "parent_tool_use_id": None,
    }


def _build_sdk_query_input(
    prompt: str,
    images: list[dict[str, Any]] | None,
) -> str | AsyncGenerator[dict[str, Any], None]:
    """Return the appropriate input for ``ClaudeSDKClient.query()``.

    Text-only prompts are passed as plain strings.  When images are
    present, an async generator of Anthropic-format message dicts is
    returned.  Each call produces a fresh generator (they are single-use).
    """
    if images:
        return _image_prompt_messages(prompt, images)
    return prompt


def _cleanup_tool_outputs(anima_dir: Path) -> None:
    """Remove temporary tool output files created during the session."""
    tool_output_dir = anima_dir / "shortterm" / "tool_outputs"
    if tool_output_dir.exists():
        shutil.rmtree(tool_output_dir, ignore_errors=True)
        logger.debug("Cleaned up tool output directory: %s", tool_output_dir)


def _summarise_tool_input(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Return a concise one-line summary of tool_input for activity log content."""
    if tool_name == "Bash":
        cmd = tool_input.get("command", "")
        return cmd[:300] if cmd else "(empty)"
    if tool_name in ("Read", "Write", "Edit"):
        return tool_input.get("file_path", "(no path)")
    if tool_name == "Grep":
        return tool_input.get("pattern", "(no pattern)")
    if tool_name == "Glob":
        return tool_input.get("pattern", "(no pattern)")
    return str(tool_input)[:300]


def _sanitise_tool_args(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Strip large payload fields from tool_input before logging."""
    if tool_name == "Write":
        # content can be an entire file — only keep the path and length.
        sanitised = {k: v for k, v in tool_input.items() if k != "content"}
        if "content" in tool_input:
            sanitised["content_length"] = len(tool_input["content"])
        return sanitised
    if tool_name == "Edit":
        # old_string / new_string can be large diffs.
        sanitised = {}
        for k, v in tool_input.items():
            if k in ("old_string", "new_string"):
                sanitised[k] = v[:200] if isinstance(v, str) else v
            else:
                sanitised[k] = v
        return sanitised
    return tool_input


def _log_tool_use(
    anima_dir: Path,
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    tool_use_id: str | None = None,
    blocked: bool = False,
    block_reason: str = "",
) -> None:
    """Record a tool call to the activity log (best-effort, never raises)."""
    try:
        from core.memory.activity import ActivityLogger

        activity = ActivityLogger(anima_dir)
        meta: dict[str, Any] = {"args": _sanitise_tool_args(tool_name, tool_input)}
        if tool_use_id:
            meta["tool_use_id"] = tool_use_id
        if blocked:
            meta["blocked"] = True
            meta["reason"] = block_reason
        activity.log(
            "tool_use",
            tool=tool_name,
            content=_summarise_tool_input(tool_name, tool_input),
            meta=meta,
        )
    except Exception:
        # Never let logging failures disrupt tool execution.
        logger.debug("Failed to log tool_use for %s", tool_name, exc_info=True)


def _log_tool_result(
    anima_dir: Path,
    tool_name: str,
    tool_use_id: str,
    result_content: str,
    *,
    is_error: bool = False,
) -> None:
    """Record a tool result to the activity log (best-effort, never raises)."""
    try:
        from core.memory.activity import ActivityLogger

        activity = ActivityLogger(anima_dir)
        activity.log(
            "tool_result",
            tool=tool_name,
            content=result_content,
            meta={"tool_use_id": tool_use_id, "is_error": is_error},
        )
    except Exception:
        logger.debug("Failed to log tool_result for %s", tool_name, exc_info=True)


def _collect_all_subordinates(
    anima_name: str,
    animas_cfg: dict[str, Any],
) -> set[str]:
    """Recursively collect all subordinates (direct + transitive) of *anima_name*."""
    result: set[str] = set()
    queue = [anima_name]
    while queue:
        current = queue.pop()
        for sub_name, sub_cfg in animas_cfg.items():
            if sub_cfg.supervisor == current and sub_name not in result:
                result.add(sub_name)
                queue.append(sub_name)
    return result


def _cache_subordinate_paths(
    anima_dir: Path,
) -> tuple[list[Path], list[Path]]:
    """Cache subordinate paths for permission checks at hook build time.

    Collects paths for **all** hierarchical subordinates (not just direct
    reports) so that a top-level supervisor can access cron.md, heartbeat.md,
    and activity_log of any anima beneath them in the org tree.
    """
    sub_activity_dirs: list[Path] = []
    sub_mgmt_files: list[Path] = []
    try:
        from core.config.models import load_config
        from core.paths import get_animas_dir

        cfg = load_config()
        animas_dir = get_animas_dir()
        anima_name = anima_dir.name
        all_subs = _collect_all_subordinates(anima_name, cfg.animas)
        for sub_name in all_subs:
            sub_dir = (animas_dir / sub_name).resolve()
            sub_activity_dirs.append(sub_dir / "activity_log")
            sub_mgmt_files.append(sub_dir / "cron.md")
            sub_mgmt_files.append(sub_dir / "heartbeat.md")
    except Exception:
        logger.debug("Failed to cache subordinate paths for Mode S hook", exc_info=True)
    return sub_activity_dirs, sub_mgmt_files


def _intercept_task_to_pending(
    anima_dir: Path,
    tool_input: dict[str, Any],
    tool_use_id: str | None,
) -> str:
    """Convert a Task tool call into a pending LLM task JSON.

    Writes a task descriptor to ``state/pending/`` so that
    ``PendingTaskExecutor`` picks it up and runs it as an independent
    minimal-context LLM session.  Returns the generated task_id.
    """
    import uuid as _uuid
    from datetime import timezone as _tz

    task_id = _uuid.uuid4().hex[:12]
    description = tool_input.get("description", "Background task")
    prompt = tool_input.get("prompt", description)

    task_desc = {
        "task_type": "llm",
        "task_id": task_id,
        "title": description,
        "description": prompt,
        "context": "",
        "acceptance_criteria": [],
        "constraints": [],
        "file_paths": [],
        "submitted_by": "self_task_intercept",
        "submitted_at": datetime.now(_tz.utc).isoformat(),
    }

    pending_dir = anima_dir / "state" / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    task_path = pending_dir / f"{task_id}.json"
    task_path.write_text(
        json.dumps(task_desc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    _log_tool_use(
        anima_dir, "Task", tool_input, tool_use_id=tool_use_id,
        blocked=False,
    )

    logger.info(
        "Task tool intercepted → pending LLM task: id=%s title=%s",
        task_id, description,
    )
    return task_id


def _build_pre_tool_hook(
    anima_dir: Path,
    *,
    max_tokens: int = 8192,
    context_window: int = 200_000,
    session_stats: dict[str, Any] | None = None,
    superuser: bool = False,
    on_task_intercepted: Callable[[], None] | None = None,
) -> Callable:
    """Build a PreToolUse hook with security checks, output guards, and tool logging.

    When *session_stats* is provided the hook also performs mid-session
    context budget observation.  If the estimated token usage leaves fewer
    than ``max_tokens * _CONTEXT_AUTOCOMPACT_SAFETY`` tokens free, the
    hook logs the observation for monitoring but does NOT return
    ``continue_=False`` — the SDK's built-in auto-compact handles
    context management.
    """
    from claude_agent_sdk.types import (
        HookContext,
        HookInput,
        PreToolUseHookSpecificOutput,
        SyncHookJSONOutput,
    )

    # Cache subordinate paths once at hook build time
    _sub_activity_dirs, _sub_mgmt_files = _cache_subordinate_paths(anima_dir)
    intercepted_task_ids: set[str] = set()

    async def _pre_tool_hook(
        input_data: HookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> SyncHookJSONOutput:
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})

        # ── Context budget observation (SDK auto-compact handles limits) ──
        if session_stats is not None:
            session_stats["tool_call_count"] += 1
            estimated_tokens = (
                session_stats["system_prompt_tokens"]
                + session_stats["user_prompt_tokens"]
                + session_stats["total_result_bytes"] // CHARS_PER_TOKEN
            )
            remaining = context_window - estimated_tokens
            budget = max_tokens * _CONTEXT_AUTOCOMPACT_SAFETY
            if remaining < budget:
                logger.info(
                    "Context approaching limit: estimated=%d remaining=%d "
                    "context_window=%d — SDK auto-compact will handle",
                    estimated_tokens, remaining, context_window,
                )
                # Do NOT return continue_=False. Let SDK handle auto-compact.
                # Still log for observability.
                _log_tool_use(
                    anima_dir, tool_name, tool_input,
                    tool_use_id=tool_use_id,
                    blocked=False,
                    block_reason=(
                        f"context_observation: estimated {estimated_tokens} tokens, "
                        f"remaining {remaining} — SDK managing"
                    ),
                )

        # Task tool intercept → pending LLM task
        if tool_name == "Task":
            task_id = _intercept_task_to_pending(
                anima_dir, tool_input, tool_use_id,
            )
            intercepted_task_ids.add(task_id)
            if on_task_intercepted is not None:
                try:
                    on_task_intercepted()
                except Exception:
                    logger.debug("on_task_intercepted callback failed", exc_info=True)
            return SyncHookJSONOutput(
                hookSpecificOutput=PreToolUseHookSpecificOutput(
                    hookEventName="PreToolUse",
                    permissionDecision="deny",
                    permissionDecisionReason=(
                        f"INTERCEPT_OK: Task accepted (task_id: {task_id}). "
                        f"This task was redirected to state/pending and will run in "
                        f"your background task executor shortly. "
                        f"Do not call TaskOutput for this task_id in this session. "
                        f"Continue the conversation now."
                    ),
                )
            )

        # TaskOutput for intercepted Task is not backed by SDK task IDs.
        if tool_name == "TaskOutput":
            task_id = str(tool_input.get("task_id", "")).strip()
            if task_id and task_id in intercepted_task_ids:
                _log_tool_use(
                    anima_dir,
                    "TaskOutput",
                    tool_input,
                    tool_use_id=tool_use_id,
                    blocked=False,
                )
                return SyncHookJSONOutput(
                    hookSpecificOutput=PreToolUseHookSpecificOutput(
                        hookEventName="PreToolUse",
                        permissionDecision="deny",
                        permissionDecisionReason=(
                            f"INTERCEPT_OK: task_id {task_id} is managed by "
                            f"PendingTaskExecutor (not SDK TaskOutput). "
                            f"Treat this as expected and continue."
                        ),
                    )
                )

        # Write / Edit: check file path
        if tool_name in ("Write", "Edit"):
            file_path = tool_input.get("file_path", "")
            violation = _check_a1_file_access(
                file_path, anima_dir, write=True,
                subordinate_activity_dirs=_sub_activity_dirs,
                subordinate_management_files=_sub_mgmt_files,
                superuser=superuser,
            )
            if violation:
                _log_tool_use(anima_dir, tool_name, tool_input, tool_use_id=tool_use_id, blocked=True, block_reason=violation)
                return SyncHookJSONOutput(
                    hookSpecificOutput=PreToolUseHookSpecificOutput(
                        hookEventName="PreToolUse",
                        permissionDecision="deny",
                        permissionDecisionReason=violation,
                    )
                )

        # Read: check for path traversal to other animas
        if tool_name == "Read":
            file_path = tool_input.get("file_path", "")
            violation = _check_a1_file_access(
                file_path, anima_dir, write=False,
                subordinate_activity_dirs=_sub_activity_dirs,
                subordinate_management_files=_sub_mgmt_files,
                superuser=superuser,
            )
            if violation:
                _log_tool_use(anima_dir, tool_name, tool_input, tool_use_id=tool_use_id, blocked=True, block_reason=violation)
                return SyncHookJSONOutput(
                    hookSpecificOutput=PreToolUseHookSpecificOutput(
                        hookEventName="PreToolUse",
                        permissionDecision="deny",
                        permissionDecisionReason=violation,
                    )
                )

        # Bash: inspect command
        if tool_name == "Bash":
            command = tool_input.get("command", "")
            violation = _check_a1_bash_command(command, anima_dir, superuser=superuser)
            if violation:
                _log_tool_use(anima_dir, tool_name, tool_input, tool_use_id=tool_use_id, blocked=True, block_reason=violation)
                return SyncHookJSONOutput(
                    hookSpecificOutput=PreToolUseHookSpecificOutput(
                        hookEventName="PreToolUse",
                        permissionDecision="deny",
                        permissionDecisionReason=violation,
                    )
                )

        # Log the tool call (allowed)
        _log_tool_use(anima_dir, tool_name, tool_input, tool_use_id=tool_use_id)

        # Output guard
        updated = _build_output_guard(tool_name, tool_input, anima_dir)
        if updated is not None:
            return SyncHookJSONOutput(
                hookSpecificOutput=PreToolUseHookSpecificOutput(
                    hookEventName="PreToolUse",
                    permissionDecision="allow",
                    updatedInput=updated,
                )
            )

        return SyncHookJSONOutput()

    return _pre_tool_hook


def _build_pre_compact_hook(anima_dir: Path) -> "Callable":
    """Build a PreCompact hook that logs whenever SDK auto-compact fires.

    This hook is called by the Claude Code subprocess before it summarises and
    compresses the conversation history.  We record the event to the activity
    log so that compaction history is visible in the Anima's timeline.
    """
    from claude_agent_sdk.types import HookContext, SyncHookJSONOutput

    async def _pre_compact_hook(
        input_data: dict,
        tool_use_id: str | None,
        context: HookContext,
    ) -> SyncHookJSONOutput:
        trigger = input_data.get("trigger", "unknown")
        logger.info(
            "SDK auto-compact triggered: trigger=%s anima=%s",
            trigger,
            anima_dir.name,
        )
        # Record to activity log for observability
        try:
            from core.memory.activity import ActivityLogger
            activity = ActivityLogger(anima_dir)
            activity.log(
                event_type="tool_use",
                content=f"SDK context compaction ({trigger})",
                summary=f"auto-compact:{trigger}",
                meta={"trigger": trigger},
            )
        except Exception:
            logger.debug("Failed to write compaction activity log", exc_info=True)
        return SyncHookJSONOutput()

    return _pre_compact_hook


# ── Common tool record helpers ───────────────────────────────


def _handle_tool_use_block(
    block: Any,
    pending_records: dict[str, ToolCallRecord],
    journal: Any | None,
    model: str,
) -> ToolCallRecord:
    """Process a ToolUseBlock from AssistantMessage.

    Registers the block in ``pending_records`` and writes a WAL entry
    via the streaming journal (if provided).
    """
    context_window = resolve_context_window(model)
    record = ToolCallRecord(
        tool_name=block.name,
        tool_id=block.id,
        input_summary=_truncate_for_record(
            str(getattr(block, "input", "")),
            tool_input_save_budget(context_window),
        ),
        result_summary="",
        is_error=False,
    )
    pending_records[block.id] = record
    if journal:
        journal.write_tool_start(block.name, record.input_summary, tool_id=block.id)
    logger.debug("ToolUseBlock registered: tool=%s id=%s", block.name, block.id)
    return record


def _tool_result_content_len(block: Any) -> int:
    """Return the character length of a ToolResultBlock's textual content.

    Used by session_stats tracking to estimate context consumption
    without duplicating the full content extraction in the outer loop.
    """
    content = block.content
    if isinstance(content, list):
        return sum(
            len(str(c.get("text", "")))
            for c in content
            if isinstance(c, dict)
        )
    return len(str(content)) if content else 0


def _handle_tool_result_block(
    block: Any,
    pending_records: dict[str, ToolCallRecord],
    journal: Any | None,
    model: str,
    *,
    anima_dir: Path | None = None,
) -> None:
    """Process a ToolResultBlock from UserMessage.

    Updates the matching entry in ``pending_records`` and writes a WAL
    entry via the streaming journal (if provided).  When *anima_dir* is
    given, the full result is also recorded in the activity log.
    """
    content = block.content
    if isinstance(content, list):
        content = " ".join(
            str(c.get("text", "")) for c in content if isinstance(c, dict)
        )
    content_str = str(content) if content else ""
    is_error = block.is_error if block.is_error is not None else False

    record = pending_records.get(block.tool_use_id)
    if record:
        context_window = resolve_context_window(model)
        record.result_summary = _truncate_for_record(
            content_str,
            tool_result_save_budget(record.tool_name, context_window),
        )
        record.is_error = is_error
        logger.info(
            "ToolResult captured: tool=%s id=%s result_len=%d is_error=%s",
            record.tool_name, block.tool_use_id, len(content_str), is_error,
        )
    else:
        logger.warning("ToolResultBlock for unknown tool_use_id=%s", block.tool_use_id)

    if journal:
        tool_name = record.tool_name if record else "unknown"
        journal.write_tool_end(tool_name, content_str[:500], tool_id=block.tool_use_id)

    # Record full tool result in activity log
    if anima_dir is not None:
        tool_name = record.tool_name if record else "unknown"
        _log_tool_result(
            anima_dir, tool_name, block.tool_use_id,
            content_str, is_error=is_error,
        )


def _finalize_pending_records(
    pending_records: dict[str, ToolCallRecord],
) -> list[ToolCallRecord]:
    """Collect all tool records after the message loop ends.

    Marks any record that never received a result as an error.
    """
    records: list[ToolCallRecord] = []
    for tool_id, record in pending_records.items():
        if not record.result_summary:
            record.is_error = True
            logger.warning(
                "ToolCallRecord without result: tool=%s id=%s",
                record.tool_name, tool_id,
            )
        records.append(record)
    return records


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
        m = self._model_config.model
        if m.startswith("anthropic/"):
            return m[len("anthropic/"):]
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
                logger.warning(
                    "Mode S auth: mode_s_auth=api but no api_key found; "
                    "falling back to Max plan"
                )
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
    ) -> tuple["ClaudeAgentOptions", Path | None]:
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
                suffix=".txt", prefix="aw-sysprompt-",
            )
            try:
                os.write(fd, system_prompt.encode("utf-8"))
            finally:
                os.close(fd)
            prompt_file = Path(tmp_path)
            prompt_kwarg: str | None = None
            extra_args["system-prompt-file"] = tmp_path
            logger.info(
                "System prompt too large for CLI arg (%d bytes > %d); "
                "using --system-prompt-file %s",
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

        kwargs: dict[str, Any] = dict(
            system_prompt=prompt_kwarg,
            allowed_tools=[
                "Read", "Write", "Edit", "Bash", "Grep", "Glob",
                "WebFetch", "WebSearch",
                "mcp__aw__*",
            ],
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
                "PreToolUse": [HookMatcher(
                    matcher=".*",
                    hooks=[_build_pre_tool_hook(
                        self._anima_dir,
                        max_tokens=_effective_max_tokens,
                        context_window=_cw,
                        session_stats=session_stats,
                        superuser=_is_debug_superuser(self._anima_dir),
                    )],
                )],
                "PreCompact": [HookMatcher(
                    matcher=".*",
                    hooks=[_build_pre_compact_hook(self._anima_dir)],
                )],
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
        client: "ClaudeSDKClient",
        prompt: str,
        response_text: list[str],
        pending_records: dict[str, ToolCallRecord],
        session_stats: dict[str, Any],
        tracker: ContextTracker | None,
        session_type: str = "chat",
        images: list[dict[str, Any]] | None = None,
    ) -> "ResultMessage | None":
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
                if message.session_id:
                    _save_session_id(self._anima_dir, message.session_id, session_type)
                if tracker:
                    tracker.update_from_result_message(message.usage)
            elif isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response_text.append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        _handle_tool_use_block(
                            block, pending_records, None,
                            self._model_config.model,
                        )
            elif isinstance(message, UserMessage):
                if isinstance(message.content, list):
                    for block in message.content:
                        if isinstance(block, ToolResultBlock):
                            session_stats["total_result_bytes"] += (
                                _tool_result_content_len(block)
                            )
                            _handle_tool_result_block(
                                block, pending_records, None,
                                self._model_config.model,
                                anima_dir=self._anima_dir,
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
                                name, status,
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
        images: list[dict[str, Any]] | None = None,
        # S mode: prior_messages is intentionally unused. The Agent SDK manages
        # conversation history internally via session resume. AnimaWorks only
        # provides system_prompt (rebuilt each time with fresh Priming/RAG)
        # and the current user message.
        prior_messages: list[dict[str, Any]] | None = None,
        max_turns_override: int | None = None,
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
        _cw = resolve_context_window(self._model_config.model)
        _max_turns = max_turns_override or self._model_config.max_turns
        session_stats: dict[str, Any] = {
            "tool_call_count": 0,
            "total_result_bytes": 0,
            "system_prompt_tokens": len(system_prompt) // CHARS_PER_TOKEN,
            "user_prompt_tokens": len(prompt) // CHARS_PER_TOKEN,
            "force_chain": False,
        }

        session_type = "heartbeat" if trigger in ("heartbeat",) or (trigger and trigger.startswith("cron:")) else "chat"
        session_id_to_resume = _load_session_id(self._anima_dir, session_type)

        options, prompt_file = self._build_sdk_options(
            system_prompt, _max_turns, _cw, session_stats,
            resume=session_id_to_resume,
        )
        _prompt_files: list[Path] = []
        if prompt_file:
            _prompt_files.append(prompt_file)

        response_text: list[str] = []
        pending_records: dict[str, ToolCallRecord] = {}
        result_message = None
        message_count = 0

        try:
            logger.info(
                "ClaudeSDKClient connecting (blocking mode, resume=%s)",
                session_id_to_resume,
            )
            async with ClaudeSDKClient(options=options) as client:
                logger.info("ClaudeSDKClient connected")
                result_message = await self._process_blocking_messages(
                    client, prompt, response_text, pending_records,
                    session_stats, tracker, session_type,
                    images=images,
                )
            logger.debug("ClaudeSDKClient disconnected")
        except (ProcessError, ClaudeSDKError) as e:
            if session_id_to_resume:
                logger.warning(
                    "SDK session resume failed (session_id=%s): %s. "
                    "Retrying with fresh session.",
                    session_id_to_resume, e,
                )
                _clear_session_id(self._anima_dir, session_type)
                # Retry without resume
                options, pf = self._build_sdk_options(
                    system_prompt, _max_turns, _cw, session_stats,
                    resume=None,
                )
                if pf:
                    _prompt_files.append(pf)
                try:
                    async with ClaudeSDKClient(options=options) as client:
                        logger.info("ClaudeSDKClient connected (fresh session retry)")
                        result_message = await self._process_blocking_messages(
                            client, prompt, response_text, pending_records,
                            session_stats, tracker, session_type,
                            images=images,
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
            message_count, len(response_text), len(all_tool_records),
        )
        replied_to = self._read_replied_to_file()
        return ExecutionResult(
            text="\n".join(response_text) or "(no response)",
            result_message=result_message,
            replied_to_from_transcript=replied_to,
            tool_call_records=all_tool_records,
            force_chain=session_stats.get("force_chain", False),
        )

    # ── Streaming execution ──────────────────────────────────

    async def execute_streaming(
        self,
        system_prompt: str,
        prompt: str,
        tracker: ContextTracker,
        images: list[dict[str, Any]] | None = None,
        # S mode: prior_messages is intentionally unused. The Agent SDK manages
        # conversation history internally via session resume. AnimaWorks only
        # provides system_prompt (rebuilt each time with fresh Priming/RAG)
        # and the current user message.
        prior_messages: list[dict[str, Any]] | None = None,
        max_turns_override: int | None = None,
        trigger: str = "",
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
        _cw = resolve_context_window(self._model_config.model)
        _max_turns = max_turns_override or self._model_config.max_turns
        session_stats: dict[str, Any] = {
            "tool_call_count": 0,
            "total_result_bytes": 0,
            "system_prompt_tokens": len(system_prompt) // CHARS_PER_TOKEN,
            "user_prompt_tokens": len(prompt) // CHARS_PER_TOKEN,
            "force_chain": False,
        }

        # Derive session_type from trigger so heartbeat/cron resume their own
        # SDK session (current_session_heartbeat.json) rather than the chat
        # session (current_session_chat.json).  Mirrors the logic in execute().
        session_type = "heartbeat" if trigger in ("heartbeat",) or (trigger and trigger.startswith("cron:")) else "chat"
        session_id_to_resume = _load_session_id(self._anima_dir, session_type)

        options, prompt_file = self._build_sdk_options(
            system_prompt, _max_turns, _cw, session_stats,
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
                        # Resume時に先頭へ混ざる履歴メッセージは
                        # 今回ターンの出力ではないため無視する。
                        continue
                    message_count += 1
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text.append(block.text)
                        elif isinstance(block, ToolUseBlock):
                            _handle_tool_use_block(
                                block, pending_records, None,
                                self._model_config.model,
                            )
                            if block.id in active_tool_ids:
                                active_tool_ids.discard(block.id)
                                yield {
                                    "type": "tool_end",
                                    "tool_id": block.id,
                                    "tool_name": block.name,
                                }

                elif isinstance(message, UserMessage):
                    if not got_stream_event:
                        # 履歴側の ToolResult は現在ターンの集計対象外。
                        continue
                    if isinstance(message.content, list):
                        for block in message.content:
                            if isinstance(block, ToolResultBlock):
                                session_stats["total_result_bytes"] += (
                                    _tool_result_content_len(block)
                                )
                                _handle_tool_result_block(
                                    block, pending_records, None,
                                    self._model_config.model,
                                    anima_dir=self._anima_dir,
                                )

                elif isinstance(message, ResultMessage):
                    result_message = message
                    if message.session_id:
                        _save_session_id(self._anima_dir, message.session_id, session_type)
                    # Do NOT call tracker.update_from_result_message() here.
                    # ResultMessage.usage.input_tokens is a cumulative sum across
                    # all turns (not the current context size) and would
                    # produce inaccurate threshold checks.  Context tracking is
                    # handled per-turn via message_start events above.
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
                                    name, status,
                                )
                            else:
                                logger.info("MCP server '%s' connected successfully", name)

        async def _run_fresh_session() -> AsyncGenerator[dict[str, Any], None]:
            """Run a fresh (no-resume) streaming session and yield events."""
            fresh_options, pf = self._build_sdk_options(
                system_prompt, _max_turns, _cw, session_stats,
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
                if isinstance(retry_exc, asyncio.CancelledError):
                    raise
                if not isinstance(retry_exc, Exception):
                    logger.critical(
                        "Agent SDK raised %s during streaming retry: %s",
                        type(retry_exc).__name__, retry_exc,
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
                            first_event = await asyncio.wait_for(
                                _get_first_event(), timeout=RESUME_TIMEOUT_SEC
                            )
                        except asyncio.TimeoutError:
                            logger.warning(
                                "Resume timed out after %.1fs (SDK Issue #387, "
                                "session_id=%s), falling back to fresh session.",
                                RESUME_TIMEOUT_SEC, session_id_to_resume,
                            )
                            await stream_gen.aclose()
                            _clear_session_id(self._anima_dir, session_type)
                            fell_back = True
                        except StopAsyncIteration:
                            logger.warning(
                                "Resume stream empty (session_id=%s), "
                                "falling back to fresh session.",
                                session_id_to_resume,
                            )
                            _clear_session_id(self._anima_dir, session_type)
                            fell_back = True
                        else:
                            yield first_event
                            async for event in stream_gen:
                                yield event
                except (ProcessError, ClaudeSDKError) as e:
                    logger.warning(
                        "SDK session resume failed (session_id=%s): %s. "
                        "Retrying with fresh session.",
                        session_id_to_resume, e,
                    )
                    _clear_session_id(self._anima_dir, session_type)
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
            # CancelledError は正常な asyncio ライフサイクル（SIGTERM等）。
            # 捕捉せずそのまま伝播させる。
            if isinstance(e, asyncio.CancelledError):
                raise
            # BaseException を捕捉して StreamDisconnectedError に変換。
            # Agent SDK hook callback の "Stream closed" が SystemExit を
            # 発生させ、except Exception をすり抜ける問題への対策。
            if not isinstance(e, Exception):
                logger.critical(
                    "Agent SDK raised %s during streaming: %s",
                    type(e).__name__, e,
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
            message_count, len(response_text), len(all_tool_records),
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
        }
