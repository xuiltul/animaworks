from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Mode S PreToolUse / PreCompact hook factories, subordinate path management,
and task-to-pending interception.

Depends on ``_sdk_security``, ``_sdk_stream``, and ``_sdk_session`` within
this package.  ``claude_agent_sdk.types`` is imported at function scope
because the SDK is an optional dependency.
"""

import json
import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from core.prompt.context import CHARS_PER_TOKEN

from core.execution._sdk_security import (
    _build_output_guard,
    _check_a1_bash_command,
    _check_a1_file_access,
)
from core.execution._sdk_session import _CONTEXT_AUTOCOMPACT_SAFETY
from core.execution._sdk_stream import _log_tool_use

logger = logging.getLogger("animaworks.execution.agent_sdk")


# ── Subordinate management ───────────────────────────────────

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


# ── Task interception ────────────────────────────────────────

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


# ── Hook factories ───────────────────────────────────────────

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


def _build_pre_compact_hook(anima_dir: Path) -> Callable:
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
