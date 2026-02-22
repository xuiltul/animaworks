from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Mode A1 executor: Claude Agent SDK.

Runs Claude as a fully autonomous agent with Read/Write/Edit/Bash/Grep/Glob
tools via the Agent SDK subprocess.  Supports both blocking and streaming
execution.  Tool results are captured from UserMessage ToolResultBlock
instead of PostToolUse hooks.
"""

import asyncio
import logging
import os
import re
import shutil
import sys
from collections.abc import AsyncGenerator, Callable
from dataclasses import asdict
from typing import Any

from core.prompt.context import CHARS_PER_TOKEN, ContextTracker, resolve_context_window
from core.exceptions import ExecutionError, LLMAPIError, MemoryWriteError  # noqa: F401
from core.execution.base import BaseExecutor, ExecutionResult, StreamDisconnectedError, ToolCallRecord, _truncate_for_record, tool_input_save_budget, tool_result_save_budget
from core.execution.reminder import MSG_CONTEXT_THRESHOLD
from core.schemas import ModelConfig
from core.memory.shortterm import ShortTermMemory
from pathlib import Path

logger = logging.getLogger("animaworks.execution.agent_sdk")


# Re-export for backward compatibility (agent.py imports from here)
__all__ = ["AgentSDKExecutor", "StreamDisconnectedError"]


# ── A1 Bash blocklist ────────────────────────────────────────

# Patterns that are unconditionally blocked in Bash tool calls.
# Each entry is (compiled_regex, human-readable reason).
# NOTE: Patterns are intentionally broad (security over convenience).
# False positives (e.g. "echo chatwork send") are acceptable — the LLM
# can retry with a different phrasing, while a missed send is unrecoverable.
_BASH_BLOCKED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"chatwork.*\bsend\b", re.IGNORECASE),
     "Chatwork send is blocked"),
    (re.compile(r"chatwork_cli.*\bsend\b", re.IGNORECASE),
     "Chatwork CLI send is blocked"),
    (re.compile(r"curl.*api\.chatwork\.com.*/messages", re.IGNORECASE),
     "Direct Chatwork API post is blocked"),
    (re.compile(r"wget.*api\.chatwork\.com.*/messages", re.IGNORECASE),
     "Direct Chatwork API post via wget is blocked"),
]

# ── A1 mode security ──────────────────────────────────────────

# Files that animas cannot modify themselves (identity/privilege protection).
_PROTECTED_FILES = frozenset({
    "permissions.md",
    "identity.md",
    "bootstrap.md",
    "specialty_prompt.md",
})

# Commands that can write files (checked for path traversal).
_WRITE_COMMANDS = frozenset({
    "cp", "mv", "tee", "dd", "install", "rsync",
})

# Safety margin for Agent SDK JSON-RPC buffer.  The default (1 MB) is too
# small when system_prompt + conversation history grow large; 4 MB gives
# comfortable headroom while still catching genuinely broken messages.
_SDK_MAX_BUFFER_SIZE = 4 * 1024 * 1024  # 4 MB

# When estimated context usage leaves fewer than max_tokens * this factor
# free, the PreToolUse hook triggers session termination for auto-compact.
_CONTEXT_AUTOCOMPACT_SAFETY = 2


def _check_a1_file_access(
    file_path: str,
    anima_dir: Path,
    *,
    write: bool,
    subordinate_activity_dirs: list[Path] | None = None,
    subordinate_management_files: list[Path] | None = None,
) -> str | None:
    """Check if a file path is allowed for A1 mode tools.

    Returns violation reason string if blocked, None if allowed.
    """
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


def _check_a1_bash_command(command: str, anima_dir: Path) -> str | None:
    """Check bash commands against blocklist patterns and file operation violations.

    Blocklist patterns are matched against the raw command string (before
    shlex parsing) to prevent bypass via pipes/subshells.  Path traversal
    checks use parsed argv for precision.

    This is a best-effort heuristic — not a complete sandbox.
    """
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


# ── A1 output guard ──────────────────────────────────────────

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
    blocked: bool = False,
    block_reason: str = "",
) -> None:
    """Record a tool call to the activity log (best-effort, never raises)."""
    try:
        from core.memory.activity import ActivityLogger

        activity = ActivityLogger(anima_dir)
        meta: dict[str, Any] = {"args": _sanitise_tool_args(tool_name, tool_input)}
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


def _cache_subordinate_paths(
    anima_dir: Path,
) -> tuple[list[Path], list[Path]]:
    """Cache subordinate paths for permission checks at hook build time."""
    sub_activity_dirs: list[Path] = []
    sub_mgmt_files: list[Path] = []
    try:
        from core.config.models import load_config
        from core.paths import get_animas_dir

        cfg = load_config()
        animas_dir = get_animas_dir()
        anima_name = anima_dir.name
        for sub_name, sub_cfg in cfg.animas.items():
            if sub_cfg.supervisor == anima_name:
                sub_dir = (animas_dir / sub_name).resolve()
                sub_activity_dirs.append(sub_dir / "activity_log")
                sub_mgmt_files.append(sub_dir / "cron.md")
                sub_mgmt_files.append(sub_dir / "heartbeat.md")
    except Exception:
        logger.debug("Failed to cache subordinate paths for A1 hook", exc_info=True)
    return sub_activity_dirs, sub_mgmt_files


def _build_pre_tool_hook(
    anima_dir: Path,
    *,
    max_tokens: int = 4096,
    context_window: int = 200_000,
    session_stats: dict[str, Any] | None = None,
) -> Callable:
    """Build a PreToolUse hook with security checks, output guards, and tool logging.

    When *session_stats* is provided the hook also performs mid-session
    context budget estimation.  If the estimated token usage leaves fewer
    than ``max_tokens * _CONTEXT_AUTOCOMPACT_SAFETY`` tokens free, the
    hook returns ``continue_=False`` to trigger session termination so
    that AgentCore can chain into a fresh session.
    """
    from claude_agent_sdk.types import (
        HookContext,
        HookInput,
        PreToolUseHookSpecificOutput,
        SyncHookJSONOutput,
    )

    # Cache subordinate paths once at hook build time
    _sub_activity_dirs, _sub_mgmt_files = _cache_subordinate_paths(anima_dir)

    async def _pre_tool_hook(
        input_data: HookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> SyncHookJSONOutput:
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})

        # ── Context budget check ──────────────────────────
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
                session_stats["force_chain"] = True
                _log_tool_use(
                    anima_dir, tool_name, tool_input,
                    blocked=True,
                    block_reason=(
                        f"context_autocompact: estimated {estimated_tokens} tokens, "
                        f"remaining {remaining} < max_tokens*{_CONTEXT_AUTOCOMPACT_SAFETY}"
                    ),
                )
                logger.warning(
                    "Context auto-compact triggered: estimated=%d remaining=%d "
                    "budget=%d (max_tokens=%d * %d) context_window=%d",
                    estimated_tokens, remaining, budget,
                    max_tokens, _CONTEXT_AUTOCOMPACT_SAFETY, context_window,
                )
                return SyncHookJSONOutput(
                    continue_=False,
                    stopReason=(
                        f"Context auto-compact: approaching context window limit "
                        f"(estimated {estimated_tokens}/{context_window} tokens). "
                        f"Session will be chained."
                    ),
                )

        # Write / Edit: check file path
        if tool_name in ("Write", "Edit"):
            file_path = tool_input.get("file_path", "")
            violation = _check_a1_file_access(
                file_path, anima_dir, write=True,
                subordinate_activity_dirs=_sub_activity_dirs,
                subordinate_management_files=_sub_mgmt_files,
            )
            if violation:
                _log_tool_use(anima_dir, tool_name, tool_input, blocked=True, block_reason=violation)
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
            )
            if violation:
                _log_tool_use(anima_dir, tool_name, tool_input, blocked=True, block_reason=violation)
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
            violation = _check_a1_bash_command(command, anima_dir)
            if violation:
                _log_tool_use(anima_dir, tool_name, tool_input, blocked=True, block_reason=violation)
                return SyncHookJSONOutput(
                    hookSpecificOutput=PreToolUseHookSpecificOutput(
                        hookEventName="PreToolUse",
                        permissionDecision="deny",
                        permissionDecisionReason=violation,
                    )
                )

        # Log the tool call (allowed)
        _log_tool_use(anima_dir, tool_name, tool_input)

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
    """Execute via Claude Agent SDK (Mode A1).

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
    ) -> None:
        super().__init__(model_config, anima_dir)
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

        A1 mode does NOT pass ``ANTHROPIC_API_KEY`` so that the Claude Code
        subprocess uses its own subscription authentication (Max plan etc.)
        instead of consuming API credits.

        Sets ``ANIMAWORKS_ANIMA_DIR`` so that ``animaworks-tool`` can
        discover personal tools in the anima's ``tools/`` directory.
        ``ANIMAWORKS_PROJECT_DIR`` is propagated so tools can locate
        ``main.py``.
        """
        from core.paths import PROJECT_DIR

        env: dict[str, str] = {
            "ANIMAWORKS_ANIMA_DIR": str(self._anima_dir),
            "ANIMAWORKS_PROJECT_DIR": str(PROJECT_DIR),
            "PATH": f"{self._anima_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
            "CLAUDE_CODE_DISABLE_SKILL_IMPROVEMENT": "true",
            # Block API key leaking from parent (load_dotenv) — force Max plan auth.
            "ANTHROPIC_API_KEY": "",
        }
        # Only pass ANTHROPIC_BASE_URL if a custom endpoint is configured.
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

    # ── Blocking execution ───────────────────────────────────

    async def execute(
        self,
        prompt: str,
        system_prompt: str = "",
        tracker: ContextTracker | None = None,
        shortterm: ShortTermMemory | None = None,
        trigger: str = "",
        images: list[dict[str, Any]] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        max_turns_override: int | None = None,
    ) -> ExecutionResult:
        """Run a session via Claude Agent SDK with context monitoring hook.

        Returns ``ExecutionResult`` with the response text and the SDK
        ``ResultMessage`` (used for session chaining by AgentCore).
        """
        if images:
            logger.warning(
                "Agent SDK (Mode A1) does not support multimodal image input; "
                "images will be ignored"
            )
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ClaudeSDKClient,
            HookMatcher,
            ResultMessage,
            SystemMessage,
            TextBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
        )

        # ── Session stats: shared between PreToolUse hook closure and this
        #    outer message loop.  The hook reads these values to decide
        #    whether to terminate the session for auto-compact; the loop
        #    updates total_result_bytes after each ToolResultBlock.
        #    Both run in the same async task — no concurrent access.
        _cw = resolve_context_window(self._model_config.model)
        session_stats: dict[str, Any] = {
            "tool_call_count": 0,
            "total_result_bytes": 0,
            "system_prompt_tokens": len(system_prompt) // CHARS_PER_TOKEN,
            "user_prompt_tokens": len(prompt) // CHARS_PER_TOKEN,
            "force_chain": False,
        }

        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            allowed_tools=["Read", "Write", "Edit", "Bash", "Grep", "Glob",
                           "mcp__aw__*"],
            permission_mode="acceptEdits",
            cwd=str(self._anima_dir),
            max_turns=max_turns_override or self._model_config.max_turns,
            model=self._resolve_agent_sdk_model(),
            env=self._build_env(),
            max_buffer_size=_SDK_MAX_BUFFER_SIZE,
            setting_sources=[],  # CLI内蔵hook(settings.json)の読み込みを防止
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
                        max_tokens=self._model_config.max_tokens or 4096,
                        context_window=_cw,
                        session_stats=session_stats,
                    )],
                )],
            },
        )

        response_text: list[str] = []
        pending_records: dict[str, ToolCallRecord] = {}
        result_message: ResultMessage | None = None
        message_count = 0

        try:
            logger.info("ClaudeSDKClient connecting (blocking mode)")
            async with ClaudeSDKClient(options=options) as client:
                logger.info("ClaudeSDKClient connected")
                await client.query(prompt)
                async for message in client.receive_response():
                    if isinstance(message, ResultMessage):
                        result_message = message
                        if tracker:
                            tracker.update_from_result_message(message.usage)
                    elif isinstance(message, AssistantMessage):
                        message_count += 1
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
            logger.debug("ClaudeSDKClient disconnected")
        except Exception as e:
            logger.exception("Agent SDK execution error")
            all_tool_records = _finalize_pending_records(pending_records)
            return ExecutionResult(
                text="\n".join(response_text) or f"[Agent SDK Error: {e}]",
                tool_call_records=all_tool_records,
            )
        finally:
            _cleanup_tool_outputs(self._anima_dir)

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
        prior_messages: list[dict[str, Any]] | None = None,
        max_turns_override: int | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream events from Claude Agent SDK.

        Yields dicts:
            ``{"type": "text_delta", "text": "..."}``
            ``{"type": "tool_start", "tool_name": "...", "tool_id": "..."}``
            ``{"type": "tool_end", "tool_id": "...", "tool_name": "..."}``
            ``{"type": "done", "full_text": "...", "result_message": ...}``
        """
        if images:
            logger.warning(
                "Agent SDK (Mode A1) streaming does not support multimodal "
                "image input; images will be ignored"
            )
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ClaudeSDKClient,
            HookMatcher,
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
        session_stats: dict[str, Any] = {
            "tool_call_count": 0,
            "total_result_bytes": 0,
            "system_prompt_tokens": len(system_prompt) // CHARS_PER_TOKEN,
            "user_prompt_tokens": len(prompt) // CHARS_PER_TOKEN,
            "force_chain": False,
        }

        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            allowed_tools=["Read", "Write", "Edit", "Bash", "Grep", "Glob",
                           "mcp__aw__*"],
            permission_mode="acceptEdits",
            cwd=str(self._anima_dir),
            max_turns=max_turns_override or self._model_config.max_turns,
            model=self._resolve_agent_sdk_model(),
            env=self._build_env(),
            max_buffer_size=_SDK_MAX_BUFFER_SIZE,
            include_partial_messages=True,
            setting_sources=[],  # CLI内蔵hook(settings.json)の読み込みを防止
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
                        max_tokens=self._model_config.max_tokens or 4096,
                        context_window=_cw,
                        session_stats=session_stats,
                    )],
                )],
            },
        )

        response_text: list[str] = []
        pending_records: dict[str, ToolCallRecord] = {}
        result_message: ResultMessage | None = None
        active_tool_ids: set[str] = set()
        message_count = 0

        try:
            logger.info("ClaudeSDKClient connecting (streaming mode)")
            async with ClaudeSDKClient(options=options) as client:
                logger.info("ClaudeSDKClient connected")
                await client.query(prompt)
                async for message in client.receive_messages():
                    if isinstance(message, StreamEvent):
                        event = message.event
                        event_type = event.get("type", "")

                        if event_type == "content_block_start":
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
                        tracker.update_from_result_message(message.usage)
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
