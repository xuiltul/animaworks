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
execution, plus a ``PostToolUse`` hook for context monitoring.
"""

import logging
import os
import re
import shutil
import sys
from collections.abc import AsyncGenerator, Callable
from dataclasses import asdict
from typing import Any

from core.prompt.context import ContextTracker
from core.exceptions import ExecutionError, LLMAPIError, MemoryWriteError  # noqa: F401
from core.execution.base import BaseExecutor, ExecutionResult, StreamDisconnectedError, ToolCallRecord, _truncate_for_record
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


def _check_a1_file_access(
    file_path: str, anima_dir: Path, *, write: bool,
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


def _build_pre_tool_hook(
    anima_dir: Path,
) -> Callable:
    """Build a PreToolUse hook with security checks, output guards, and tool logging."""
    from claude_agent_sdk.types import (
        HookContext,
        HookInput,
        PreToolUseHookSpecificOutput,
        SyncHookJSONOutput,
    )

    async def _pre_tool_hook(
        input_data: HookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> SyncHookJSONOutput:
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})

        # Write / Edit: check file path
        if tool_name in ("Write", "Edit"):
            file_path = tool_input.get("file_path", "")
            violation = _check_a1_file_access(
                file_path, anima_dir, write=True,
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


class AgentSDKExecutor(BaseExecutor):
    """Execute via Claude Agent SDK (Mode A1).

    The SDK spawns a subprocess where Claude has full tool access.
    Context monitoring is handled via a PostToolUse hook that fires
    when token usage crosses the configured threshold.
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
            HookMatcher,
            ResultMessage,
            SystemMessage,
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
            allowed_tools=["Read", "Write", "Edit", "Bash", "Grep", "Glob",
                           "mcp__aw__*"],
            permission_mode="acceptEdits",
            cwd=str(self._anima_dir),
            max_turns=self._model_config.max_turns,
            model=self._resolve_agent_sdk_model(),
            env=self._build_env(),
            max_buffer_size=_SDK_MAX_BUFFER_SIZE,
            mcp_servers={
                "aw": {
                    "command": sys.executable,
                    "args": ["-m", "core.mcp.server"],
                    "env": self._build_mcp_env(),
                },
            },
            hooks={
                "PreToolUse": [HookMatcher(
                    matcher="Write|Edit|Bash|Read|Grep|Glob",
                    hooks=[_build_pre_tool_hook(self._anima_dir)],
                )],
                "PostToolUse": [HookMatcher(matcher=None, hooks=[_post_tool_hook])],
            },
        )

        response_text: list[str] = []
        all_tool_records: list[ToolCallRecord] = []
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
                        elif hasattr(block, 'name') and hasattr(block, 'id'):
                            # ToolUseBlock from Agent SDK
                            all_tool_records.append(ToolCallRecord(
                                tool_name=block.name,
                                tool_id=block.id,
                                input_summary=_truncate_for_record(
                                    str(getattr(block, 'input', '')), 200,
                                ),
                                result_summary="",  # Not available - managed by Agent SDK internally
                            ))
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
        except Exception as e:
            logger.exception("Agent SDK execution error")
            return ExecutionResult(
                text="\n".join(response_text) or f"[Agent SDK Error: {e}]",
                tool_call_records=all_tool_records,
            )
        finally:
            _cleanup_tool_outputs(self._anima_dir)

        logger.debug(
            "Agent SDK completed, messages=%d text_blocks=%d",
            message_count, len(response_text),
        )
        replied_to = self._read_replied_to_file()
        return ExecutionResult(
            text="\n".join(response_text) or "(no response)",
            result_message=result_message,
            replied_to_from_transcript=replied_to,
            tool_call_records=all_tool_records,
        )

    # ── Streaming execution ──────────────────────────────────

    async def execute_streaming(
        self,
        system_prompt: str,
        prompt: str,
        tracker: ContextTracker,
        images: list[dict[str, Any]] | None = None,
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
            HookMatcher,
            ResultMessage,
            SystemMessage,
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
            allowed_tools=["Read", "Write", "Edit", "Bash", "Grep", "Glob",
                           "mcp__aw__*"],
            permission_mode="acceptEdits",
            cwd=str(self._anima_dir),
            max_turns=self._model_config.max_turns,
            model=self._resolve_agent_sdk_model(),
            env=self._build_env(),
            max_buffer_size=_SDK_MAX_BUFFER_SIZE,
            include_partial_messages=True,
            mcp_servers={
                "aw": {
                    "command": sys.executable,
                    "args": ["-m", "core.mcp.server"],
                    "env": self._build_mcp_env(),
                },
            },
            hooks={
                "PreToolUse": [HookMatcher(
                    matcher="Write|Edit|Bash|Read|Grep|Glob",
                    hooks=[_build_pre_tool_hook(self._anima_dir)],
                )],
                "PostToolUse": [HookMatcher(matcher=None, hooks=[_post_tool_hook])],
            },
        )

        response_text: list[str] = []
        all_tool_records: list[ToolCallRecord] = []
        result_message: ResultMessage | None = None
        active_tool_ids: set[str] = set()
        # Track tool names from content_block_start for streaming records
        active_tool_names: dict[str, str] = {}
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
                            tool_name = block.get("name", "")
                            active_tool_ids.add(tool_id)
                            active_tool_names[tool_id] = tool_name
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
                            all_tool_records.append(ToolCallRecord(
                                tool_name=block.name,
                                tool_id=block.id,
                                input_summary=_truncate_for_record(
                                    str(getattr(block, 'input', '')), 200,
                                ),
                                result_summary="",  # Not available - managed by Agent SDK internally
                            ))
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
        except Exception as e:
            logger.exception("Agent SDK streaming error")
            partial = "\n".join(response_text)
            raise StreamDisconnectedError(
                f"Agent SDK stream error: {e}",
                partial_text=partial,
            ) from e
        finally:
            _cleanup_tool_outputs(self._anima_dir)

        logger.debug(
            "Agent SDK streaming completed, messages=%d text_blocks=%d",
            message_count, len(response_text),
        )
        full_text = "\n".join(response_text) or "(no response)"
        replied_to = self._read_replied_to_file()
        yield {
            "type": "done",
            "full_text": full_text,
            "result_message": result_message,
            "replied_to_from_transcript": replied_to,
            "tool_call_records": [asdict(r) for r in all_tool_records],
        }
