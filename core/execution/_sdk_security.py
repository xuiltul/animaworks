from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Mode S security checks and output size guards.

Pure functions with no framework state — leaf module in the dependency graph.
"""

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger("animaworks.execution.agent_sdk")


# ── Mode S Bash blocklist ────────────────────────────────────

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

_PROTECTED_FILES = frozenset({
    "permissions.md",
    "bootstrap.md",
})

_WRITE_COMMANDS = frozenset({
    "cp", "mv", "tee", "dd", "install", "rsync",
})


# ── Mode S output guard ──────────────────────────────────────

_BASH_TRUNCATE_BYTES = 10_000   # 10 KB
_BASH_HEAD_BYTES = 5_000        # head display
_BASH_TAIL_BYTES = 3_000        # tail display
_READ_DEFAULT_LIMIT = 500       # lines
_GREP_DEFAULT_HEAD_LIMIT = 200  # entries
_GLOB_DEFAULT_HEAD_LIMIT = 500  # entries


# ── Security check functions ─────────────────────────────────

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


# ── Output guard functions ───────────────────────────────────

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
