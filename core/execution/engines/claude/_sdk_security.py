from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Mode S security checks, injection audit logging, and output size guards.

Helpers have no executor state; this remains a leaf module in the dependency graph.
"""

import logging
import re
import tempfile
from pathlib import Path
from typing import Any

from core.anima.workspace_files import (
    AGENTS_FILE,
    IDENTITY_FILE,
    LEGACY_IDENTITY,
    LEGACY_RENAMES,
    SOUL_FILE,
)
from core.config.file_access_policy import (
    PROTECTED_FILE_PATHS,
    FileAccessContext,
    FileAccessDecision,
    effective_write_roots,
    evaluate_file_access,
    resolve_effective_denied_roots,
    shared_tool_cache_write_root,
)

logger = logging.getLogger("animaworks.execution.agent_sdk")
injection_logger = logging.getLogger("animaworks.security.sdk_bash_injection")
_PROTECTED_FILES = PROTECTED_FILE_PATHS


# ── Mode S security ──────────────────────────────────────────

_WRITE_COMMANDS = frozenset(
    {
        "cp",
        "mv",
        "tee",
        "dd",
        "install",
        "rsync",
    }
)


# ── Mode S output guard ──────────────────────────────────────

_BASH_TRUNCATE_BYTES = 10_000  # 10 KB
_BASH_HEAD_BYTES = 5_000  # head display
_BASH_TAIL_BYTES = 3_000  # tail display
_READ_DEFAULT_LIMIT = 2_000  # lines (Claude Code compatible)
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
    descendant_read_files: list[Path] | None = None,
    descendant_read_dirs: list[Path] | None = None,
    peer_activity_dirs: list[Path] | None = None,
    superuser: bool = False,
    task_cwd: Path | None = None,
) -> str | None:
    """Check if a file path is allowed for Mode S tools.

    Returns violation reason string if blocked, None if allowed.
    """
    if superuser or not file_path:
        context = FileAccessContext(
            anima_dir=anima_dir,
            data_dir=anima_dir.resolve().parent.parent,
            superuser=superuser,
        )
        return _mode_s_file_access_error(
            file_path,
            evaluate_file_access(file_path, context, write=write),
            write=write,
        )

    from core.config.schemas import load_permissions
    from core.paths import get_data_dir

    try:
        permissions = load_permissions(anima_dir)
        denied_roots = resolve_effective_denied_roots(anima_dir, permissions.file_roots_denied)
        data_dir = get_data_dir().resolve()
    except Exception:
        logger.exception("Failed to load file permission policy for anima_dir=%s", anima_dir)
        return "File permission check failed"

    try:
        if write:
            from core.paths import get_common_knowledge_dir, get_common_skills_dir

            # Mode S keeps its historical workspace-write grants in addition
            # to the configured roots used by ToolHandler.
            write_roots = (
                *effective_write_roots(anima_dir, permissions.file_roots, task_cwd),
                Path(tempfile.gettempdir()).resolve(),
                get_common_knowledge_dir().resolve(),
                get_common_skills_dir().resolve(),
                *((cache_root,) if (cache_root := shared_tool_cache_write_root(anima_dir)) is not None else ()),
            )
        else:
            write_roots = ()
    except Exception:
        logger.exception("Failed to evaluate file write permission for anima_dir=%s", anima_dir)
        return "File permission check failed"

    context = FileAccessContext(
        anima_dir=anima_dir.resolve(),
        data_dir=data_dir,
        denied_roots=denied_roots,
        file_roots=tuple(permissions.file_roots),
        file_roots_readonly=tuple(permissions.file_roots_readonly),
        write_roots=tuple(write_roots),
        restrict_reads_to_roots=False,
        subordinate_activity_dirs=tuple(Path(path).resolve() for path in (subordinate_activity_dirs or ())),
        peer_activity_dirs=tuple(Path(path).resolve() for path in (peer_activity_dirs or ())),
        subordinate_management_files=tuple(Path(path).resolve() for path in (subordinate_management_files or ())),
        subordinate_root_dirs=tuple(Path(path).resolve().parent for path in (subordinate_activity_dirs or ())),
        descendant_read_files=tuple(Path(path).resolve() for path in (descendant_read_files or ())),
        descendant_read_dirs=tuple(Path(path).resolve() for path in (descendant_read_dirs or ())),
    )
    decision = evaluate_file_access(file_path, context, write=write)
    return _mode_s_file_access_error(file_path, decision, write=write)


def _mode_s_file_access_error(
    file_path: str,
    decision: FileAccessDecision,
    *,
    write: bool,
) -> str | None:
    """Convert a shared decision to Mode S's plain-string error format."""
    if decision.allowed:
        return None
    if decision.reason == "internal_cache":
        return f"Direct access to internal runtime cache is not allowed: '{file_path}'"
    if decision.reason == "denied_root":
        return f"Access to denied directory is not allowed: {file_path}"
    if decision.reason == "global_permissions":
        return "permissions.global.json cannot be modified via Mode S tools"
    if decision.reason == "protected_file":
        prompt_files = tuple(
            name.casefold() for name in (IDENTITY_FILE, SOUL_FILE, AGENTS_FILE, LEGACY_IDENTITY, *LEGACY_RENAMES)
        )
        if str(decision.protected_path).casefold().endswith(prompt_files):
            return (
                f"'{decision.protected_path}' is a protected file (root-owned setting); "
                "use write_memory_file with this path to update it"
            )
        return f"'{decision.protected_path}' is a protected file and cannot be modified"
    if decision.reason == "protected_directory":
        return f"'{decision.protected_path}/' is a protected directory and cannot be modified"
    if decision.reason == "other_anima":
        return f"Access to other anima's directory is not allowed: {file_path}"
    if decision.reason == "readonly_dir":
        return f"Write access to read-only directory is not allowed: {file_path}"
    if decision.reason == "outside_allowed_dirs":
        if write:
            return f"Write access outside configured file roots is not allowed: {file_path}"
        return f"Read access outside configured file roots is not allowed: {file_path}"
    return "File permission check failed"


def _check_a1_bash_command(
    command: str,
    anima_dir: Path,
    *,
    superuser: bool = False,
    trigger: str = "unknown",
) -> str | None:
    """Check a bash command against the shared command policy.

    Thin wrapper over ``core.tooling.policy.command_policy.evaluate_command`` so Mode S
    applies exactly the same layers (injection, global deny, recursive-search
    guard, per-anima deny, allowlist, traversal, other-anima write) as the
    ToolHandler and the Codex hook.  Returns a denial reason or None.
    """
    if superuser or not command or not command.strip():
        return None

    from core.tooling.policy.command_policy import (
        evaluate_command,
        load_command_policy_context,
        record_injection_hit,
    )

    try:
        ctx = load_command_policy_context(anima_dir, cwd=anima_dir, superuser=superuser)
    except Exception:
        # fail-closed: a broken config must block the command.
        logger.exception("command_policy load failed for anima_dir=%s", anima_dir)
        return "Command policy check failed"

    decision = evaluate_command(command, ctx)
    if decision.injection_hit:
        record_injection_hit(command, ctx, pattern_name=decision.injection_hit, trigger=trigger)
    if decision.allowed:
        return None
    return decision.reason


def _matching_injection_pattern(command: str, config: Any) -> str:
    """Return the first configured injection pattern matching *command*."""
    if config is not None:
        for item in config.injection_patterns:
            try:
                if re.search(item.pattern, command):
                    return item.name or item.pattern
            except re.error:
                continue
    return "unknown"


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
        # stdin is the SDK's socket: a command that reads stdin (jq with an
        # empty file list, cat, read) would wait on it forever.  The newline
        # before "}" lets a command end with a heredoc terminator.
        f'{{ {command}\n}} < /dev/null > "$_OUTF" 2>&1\n'
        f"_EC=$?\n"
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
        f"else\n"
        f'  cat "$_OUTF"\n'
        f'  rm -f "$_OUTF"\n'
        f"fi\n"
        f"exit $_EC"
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
