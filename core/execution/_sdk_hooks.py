from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Mode S PreToolUse / PreCompact / Stop hook factories, subordinate path
management, and task-to-pending interception.

Depends on ``_sdk_security``, ``_sdk_stream``, and ``_sdk_session`` within
this package.  ``claude_agent_sdk.types`` is imported at function scope
because the SDK is an optional dependency.
"""

import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.execution._sanitize import TOOL_TRUST_LEVELS
from core.execution._sdk_security import (
    _build_output_guard,
    _check_a1_bash_command,
    _check_a1_file_access,
)
from core.execution._sdk_session import _CONTEXT_AUTOCOMPACT_SAFETY
from core.execution._sdk_stream import _log_tool_use
from core.prompt.context import CHARS_PER_TOKEN

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
) -> tuple[list[Path], list[Path], list[Path], list[Path], list[Path]]:
    """Cache subordinate and peer paths for permission checks at hook build time.

    Collects paths for **all** hierarchical subordinates (not just direct
    reports) so that a top-level supervisor can access management files
    and activity_log of any anima beneath them in the org tree.

    Returns:
        (sub_activity_dirs, sub_mgmt_files, peer_activity_dirs,
         descendant_read_files, descendant_read_dirs)

    - sub_mgmt_files: all descendants' cron.md, heartbeat.md, status.json,
      injection.md (read/write)
    - descendant_read_files: all descendants' identity.md, injection.md,
      status.json, state files (read-only)
    - descendant_read_dirs: all descendants' state/pending/ (read-only dir)
    """
    sub_activity_dirs: list[Path] = []
    sub_mgmt_files: list[Path] = []
    peer_activity_dirs: list[Path] = []
    descendant_read_files: list[Path] = []
    descendant_read_dirs: list[Path] = []
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

            for fname in ("cron.md", "heartbeat.md", "status.json", "injection.md"):
                sub_mgmt_files.append(sub_dir / fname)

            descendant_read_files.append(sub_dir / "identity.md")
            descendant_read_files.append(sub_dir / "injection.md")
            descendant_read_files.append(sub_dir / "status.json")
            descendant_read_files.append(sub_dir / "state" / "current_state.md")
            descendant_read_files.append(sub_dir / "state" / "task_queue.jsonl")
            descendant_read_dirs.append(sub_dir / "state" / "pending")

        # Collect peer activity_log dirs (same supervisor, excluding self)
        my_supervisor = None
        if anima_name in cfg.animas:
            my_supervisor = cfg.animas[anima_name].supervisor
        for peer_name, peer_cfg in cfg.animas.items():
            if peer_name != anima_name and peer_cfg.supervisor == my_supervisor:
                peer_dir = (animas_dir / peer_name).resolve()
                peer_activity_dirs.append(peer_dir / "activity_log")
    except Exception:
        logger.debug("Failed to cache subordinate paths for Mode S hook", exc_info=True)
    return sub_activity_dirs, sub_mgmt_files, peer_activity_dirs, descendant_read_files, descendant_read_dirs


# ── Task interception ────────────────────────────────────────


def _intercept_task_to_pending(
    anima_dir: Path,
    tool_input: dict[str, Any],
    tool_use_id: str | None,
    *,
    actual_tool_name: str = "Task",
) -> str:
    """Convert a Task/Agent tool call into a pending LLM task JSON.

    Writes a task descriptor to ``state/pending/`` so that
    ``PendingTaskExecutor`` picks it up and runs it as an independent
    minimal-context LLM session.  Returns the generated task_id.
    """
    import uuid as _uuid

    task_id = _uuid.uuid4().hex[:12]
    description = tool_input.get("description", "Background task")
    prompt = tool_input.get("prompt", description)

    context_parts: list[str] = []
    ctx_path = anima_dir / "state" / "current_state.md"
    if ctx_path.exists():
        try:
            content = ctx_path.read_text(encoding="utf-8").strip()
            if content and content != "status: idle":
                context_parts.append(f"[current_state.md]\n{content}")
        except Exception:
            pass

    task_desc = {
        "task_type": "llm",
        "task_id": task_id,
        "title": description,
        "description": prompt,
        "context": "\n\n".join(context_parts),
        "acceptance_criteria": [],
        "constraints": [],
        "file_paths": [],
        "submitted_by": "self_task_intercept",
        "submitted_at": datetime.now(UTC).isoformat(),
        "reply_to": anima_dir.name,
        "working_directory": "",
    }

    pending_dir = anima_dir / "state" / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    task_path = pending_dir / f"{task_id}.json"
    task_path.write_text(
        json.dumps(task_desc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Layer 2: Register in task_queue.jsonl for tracking
    try:
        from core.memory.task_queue import TaskQueueManager

        manager = TaskQueueManager(anima_dir)
        manager.add_task(
            source="anima",
            original_instruction=prompt[:5000],
            assignee=anima_dir.name,
            summary=description[:200],
            task_id=task_id,
            status="in_progress",
            meta={
                "executor": "taskexec",
                "task_desc": {
                    "title": description,
                    "description": prompt,
                    "context": "\n\n".join(context_parts),
                    "acceptance_criteria": [],
                    "constraints": [],
                    "file_paths": [],
                    "reply_to": anima_dir.name,
                    "working_directory": "",
                },
            },
        )
    except Exception:
        logger.warning("Failed to register intercepted task in task_queue: %s", task_id, exc_info=True)

    _log_tool_use(
        anima_dir,
        actual_tool_name,
        tool_input,
        tool_use_id=tool_use_id,
        blocked=False,
    )

    logger.info(
        "%s tool intercepted → pending LLM task: id=%s title=%s",
        actual_tool_name,
        task_id,
        description,
    )
    return task_id


def _do_pending_intercept(
    anima_dir: Path,
    tool_input: dict[str, Any],
    tool_use_id: str | None,
    tool_name: str,
    intercepted_task_ids: set[str],
    on_task_intercepted: Callable[[], None] | None,
) -> Any:
    """Intercept a Task/Agent call to state/pending/ and return deny response."""
    from claude_agent_sdk.types import (
        PreToolUseHookSpecificOutput,
        SyncHookJSONOutput,
    )

    task_id = _intercept_task_to_pending(
        anima_dir,
        tool_input,
        tool_use_id,
        actual_tool_name=tool_name,
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
                f"Written to state/pending/ for background execution. "
                f"The executor has your identity, injection, behavior rules, "
                f"memory guide, and org context. "
                f"Do NOT call Task or TaskOutput for this task_id again. "
                f"Proceed with your current conversation."
            ),
        )
    )


# ── Delegation helpers ────────────────────────────────────────


def _read_status_json(anima_dir: Path) -> dict[str, Any]:
    """Read and parse status.json for an anima. Returns empty dict on failure."""
    status_path = anima_dir / "status.json"
    if not status_path.exists():
        return {}
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _select_subordinate(
    anima_dir: Path,
    description: str,
) -> str | None:
    """Select a subordinate for a task only when explicitly named.

    Only delegates when the task description explicitly mentions a
    subordinate by name.  Auto-selection (role matching + load
    balancing) has been removed — the LLM decides who executes.
    Returns None if no subordinate is named (caller falls back to
    self-pending).
    """
    from core.config.models import load_config
    from core.paths import get_animas_dir

    try:
        cfg = load_config()
    except Exception:
        return None

    my_name = anima_dir.name
    animas_dir = get_animas_dir()
    direct_subs: list[str] = [name for name, acfg in cfg.animas.items() if acfg.supervisor == my_name]
    if not direct_subs:
        return None

    desc_lower = description.lower()
    for sub_name in direct_subs:
        if sub_name.lower() in desc_lower:
            sub_status = _read_status_json(animas_dir / sub_name)
            if sub_status.get("enabled", True):
                return sub_name

    return None


def _intercept_task_to_delegation(
    anima_dir: Path,
    tool_input: dict[str, Any],
    tool_use_id: str | None,
) -> dict[str, Any] | None:
    """Delegate a Task tool call to a subordinate.

    Returns a dict with ``task_id`` and ``reason`` keys on success,
    or ``None`` if no suitable subordinate is available (caller should
    fall back to pending-task path).
    """
    from core.paths import get_animas_dir, get_data_dir, get_shared_dir

    description = tool_input.get("description", "Background task")
    prompt = tool_input.get("prompt", description)

    target_name = _select_subordinate(anima_dir, description)
    if target_name is None:
        return None

    my_name = anima_dir.name
    animas_dir = get_animas_dir()
    target_dir = animas_dir / target_name

    # Add task to subordinate's queue
    from core.memory.task_queue import TaskQueueManager

    sub_tqm = TaskQueueManager(target_dir)
    try:
        sub_entry = sub_tqm.add_task(
            source="anima",
            original_instruction=prompt,
            assignee=target_name,
            summary=description[:100],
            deadline="2h",
            relay_chain=[my_name],
        )
    except Exception as e:
        logger.error("Task persistence failed in delegate_task (subordinate queue): %s", e)
        return {
            "task_id": "persist_failed",
            "reason": f"DELEGATION_FAILED: Failed to persist task to subordinate queue: {e}. Please retry.",
        }

    # Write pending task JSON so PendingTaskExecutor picks it up for immediate execution
    task_desc = {
        "task_type": "llm",
        "task_id": sub_entry.task_id,
        "title": description[:100],
        "description": prompt,
        "context": "",
        "acceptance_criteria": [],
        "constraints": [],
        "file_paths": [],
        "submitted_by": my_name,
        "submitted_at": datetime.now(UTC).isoformat(),
        "reply_to": my_name,
        "source": "delegation",
        "working_directory": "",
    }
    pending_dir = target_dir / "state" / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    (pending_dir / f"{sub_entry.task_id}.json").write_text(
        json.dumps(task_desc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Send DM via Messenger
    dm_result = ""
    try:
        from core.i18n import t
        from core.messenger import Messenger

        messenger = Messenger(get_shared_dir(), my_name)
        messenger.send(
            to=target_name,
            content=t(
                "handler.delegation_dm_content",
                instruction=prompt[:500],
                deadline="2h",
                task_id=sub_entry.task_id,
            ),
            intent="delegation",
            meta={"task_id": sub_entry.task_id},
        )
        dm_result = "DM sent"
    except Exception as e:
        dm_result = f"DM failed: {e}"
        logger.warning("Delegation DM failed: %s -> %s: %s", my_name, target_name, e)

    # Add tracking entry to own queue
    own_tqm = TaskQueueManager(anima_dir)
    try:
        own_entry = own_tqm.add_delegated_task(
            original_instruction=prompt,
            assignee=target_name,
            summary=f"[delegated→{target_name}] {description[:80]}",
            deadline="2h",
            relay_chain=[my_name, target_name],
            meta={
                "delegated_to": target_name,
                "delegated_task_id": sub_entry.task_id,
            },
        )
    except Exception as e:
        logger.warning("Failed to persist tracking entry for delegate_task (DM already sent): %s", e)
        return {
            "task_id": "persist_failed",
            "reason": f"DELEGATION_PARTIAL: Task sent to {target_name} but tracking entry failed: {e}. "
            "DM was delivered; check subordinate queue manually.",
        }

    # Write wake file so inbox_wake_dispatcher triggers process_inbox
    try:
        wake_dir = get_data_dir() / "run" / "inbox_wake"
        wake_dir.mkdir(parents=True, exist_ok=True)
        wake_file = wake_dir / target_name
        wake_file.write_text(target_name, encoding="utf-8")
    except Exception:
        logger.debug("Failed to write inbox wake file for %s", target_name, exc_info=True)

    _log_tool_use(
        anima_dir,
        "Task",
        tool_input,
        tool_use_id=tool_use_id,
        blocked=False,
    )

    logger.info(
        "Task tool intercepted → delegated to %s: sub_task=%s own_task=%s",
        target_name,
        sub_entry.task_id,
        own_entry.task_id,
    )

    return {
        "task_id": own_entry.task_id,
        "reason": (
            f"DELEGATION_OK: Task delegated to {target_name} "
            f"(sub_task_id: {sub_entry.task_id}, own_tracking_id: {own_entry.task_id}). "
            f"{dm_result}. "
            f"Do NOT call Task or TaskOutput for this task again. "
            f"Proceed with your current conversation."
        ),
    }


# ── Hook factories ───────────────────────────────────────────


def _build_pre_tool_hook(
    anima_dir: Path,
    *,
    max_tokens: int = 8192,
    context_window: int = 200_000,
    session_stats: dict[str, Any] | None = None,
    superuser: bool = False,
    on_task_intercepted: Callable[[], None] | None = None,
    has_subordinates: bool = False,
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

    # Cache subordinate and peer paths once at hook build time
    _sub_activity_dirs, _sub_mgmt_files, _peer_activity_dirs, _desc_read_files, _desc_read_dirs = (
        _cache_subordinate_paths(anima_dir)
    )
    intercepted_task_ids: set[str] = set()
    _trust_order = {"trusted": 2, "medium": 1, "untrusted": 0}

    # SDK tools → TOOL_TRUST_LEVELS mapping (SDK uses PascalCase names)
    _SDK_TOOL_TRUST: dict[str, str] = {
        "Read": "medium",
        "Write": "medium",
        "Edit": "medium",
        "Bash": "medium",
        "Grep": "medium",
        "Glob": "medium",
        "WebFetch": "untrusted",
        "WebSearch": "untrusted",
    }

    async def _pre_tool_hook(
        input_data: HookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> SyncHookJSONOutput:
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})

        # ── Heartbeat soft timeout check ──
        if session_stats is not None and session_stats.get("trigger") == "heartbeat":
            elapsed = time.monotonic() - session_stats["start_time"]
            soft_timeout = session_stats.get("hb_soft_timeout", 300)
            if elapsed > soft_timeout and not session_stats.get("hb_soft_warned"):
                session_stats["hb_soft_warned"] = True
                from core.i18n import t as _t

                logger.info(
                    "Heartbeat soft timeout reached (%.0fs > %ds) for %s — injecting wrap-up reminder",
                    elapsed,
                    soft_timeout,
                    anima_dir.name,
                )
                return SyncHookJSONOutput(
                    hookSpecificOutput=PreToolUseHookSpecificOutput(
                        hookEventName="PreToolUse",
                        permissionDecision="allow",
                        additionalContext=_t("reminder.hb_time_limit"),
                    )
                )

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
                    estimated_tokens,
                    remaining,
                    context_window,
                )
                _log_tool_use(
                    anima_dir,
                    tool_name,
                    tool_input,
                    tool_use_id=tool_use_id,
                    blocked=False,
                    block_reason=(
                        f"context_observation: estimated {estimated_tokens} tokens, "
                        f"remaining {remaining} — SDK managing"
                    ),
                )

        # ── Trust tracking: update min_trust_seen in session_stats ──
        if session_stats is not None:
            # Resolve trust for SDK-native tools or MCP tools (mcp__aw__X)
            effective_name = tool_name
            if tool_name.startswith("mcp__aw__"):
                effective_name = tool_name[len("mcp__aw__") :]
            trust_str = _SDK_TOOL_TRUST.get(tool_name) or TOOL_TRUST_LEVELS.get(effective_name, "untrusted")
            rank = _trust_order.get(trust_str, 0)
            current_min = session_stats.get("min_trust_seen", 2)
            session_stats["min_trust_seen"] = min(current_min, rank)

            # Persist to file so MCP server subprocess can read
            _trust_file = anima_dir / "run" / "min_trust_seen"
            try:
                _trust_file.parent.mkdir(parents=True, exist_ok=True)
                _trust_file.write_text(
                    str(session_stats["min_trust_seen"]),
                    encoding="utf-8",
                )
            except Exception:
                logger.debug("Failed to persist min_trust_seen", exc_info=True)

        # Task / Agent tool intercept (SDK uses "Agent" as the subagent tool name)
        if tool_name in ("Agent", "Task"):
            if has_subordinates:
                # Supervisor path: delegate to subordinate, fallback to pending
                try:
                    delegation_result = _intercept_task_to_delegation(
                        anima_dir,
                        tool_input,
                        tool_use_id,
                    )
                    if delegation_result is not None:
                        intercepted_task_ids.add(delegation_result["task_id"])
                        if on_task_intercepted is not None:
                            try:
                                on_task_intercepted()
                            except Exception:
                                logger.debug("on_task_intercepted callback failed", exc_info=True)
                        return SyncHookJSONOutput(
                            hookSpecificOutput=PreToolUseHookSpecificOutput(
                                hookEventName="PreToolUse",
                                permissionDecision="deny",
                                permissionDecisionReason=delegation_result["reason"],
                            )
                        )
                except Exception:
                    logger.warning("Delegation failed, falling back to pending", exc_info=True)

            # Non-supervisor or supervisor fallback → state/pending/
            return _do_pending_intercept(
                anima_dir,
                tool_input,
                tool_use_id,
                tool_name,
                intercepted_task_ids,
                on_task_intercepted,
            )

        # submit_tasks intercept → DAG batch to pending
        if tool_name in ("submit_tasks", "mcp__aw__submit_tasks"):
            from core.tooling.handler_base import _error_result
            from core.tooling.handler_skills import SkillsToolsMixin

            class _SubmitTasksProxy(SkillsToolsMixin):
                _anima_dir = anima_dir
                _anima_name = anima_dir.name

            proxy = _SubmitTasksProxy()
            proxy._pending_executor_wake = on_task_intercepted
            try:
                result_str = proxy._handle_submit_tasks(tool_input)
            except Exception as exc:
                result_str = _error_result("SubmitTasksError", str(exc))

            _log_tool_use(
                anima_dir,
                "submit_tasks",
                tool_input,
                tool_use_id=tool_use_id,
                blocked=False,
            )

            # Build deny reason: distinguish success from error
            is_error = False
            task_ids_str = ""
            try:
                parsed = json.loads(result_str)
                if isinstance(parsed, dict):
                    if "error" in parsed or parsed.get("status") == "error":
                        is_error = True
                    else:
                        task_ids_str = ", ".join(parsed.get("task_ids", []))
            except (json.JSONDecodeError, TypeError):
                pass

            if is_error:
                deny_reason = f"INTERCEPT_OK: submit_tasks error: {result_str}"
            else:
                tasks_label = f" ({task_ids_str})" if task_ids_str else ""
                deny_reason = (
                    f"SUCCESS — submit_tasks completed successfully (this is NOT an error). "
                    f"Result: {result_str}\n"
                    f"All tasks{tasks_label} are now queued in YOUR OWN TaskExecutor "
                    f"and YOU will execute them yourself in background. "
                    f"These tasks are NOT sent to any subordinate. "
                    f"Do NOT re-submit the same tasks via any method — "
                    f"doing so WILL cause DUPLICATE execution. "
                    f"Proceed with your current conversation."
                )

            return SyncHookJSONOutput(
                hookSpecificOutput=PreToolUseHookSpecificOutput(
                    hookEventName="PreToolUse",
                    permissionDecision="deny",
                    permissionDecisionReason=deny_reason,
                )
            )

        # TaskOutput/AgentOutput for intercepted tasks — not backed by SDK task IDs.
        if tool_name in ("TaskOutput", "AgentOutput"):
            task_id = str(tool_input.get("task_id", "")).strip()
            if task_id and task_id in intercepted_task_ids:
                _log_tool_use(
                    anima_dir,
                    tool_name,
                    tool_input,
                    tool_use_id=tool_use_id,
                    blocked=False,
                )
                return SyncHookJSONOutput(
                    hookSpecificOutput=PreToolUseHookSpecificOutput(
                        hookEventName="PreToolUse",
                        permissionDecision="deny",
                        permissionDecisionReason=(
                            f"INTERCEPT_OK: task_id {task_id} is already running in "
                            f"PendingTaskExecutor with full context (identity, injection, "
                            f"memory guide, org info). Do NOT retry. "
                            f"Proceed with your current conversation."
                        ),
                    )
                )

        # Write / Edit: check file path
        if tool_name in ("Write", "Edit"):
            file_path = tool_input.get("file_path", "")
            violation = _check_a1_file_access(
                file_path,
                anima_dir,
                write=True,
                subordinate_activity_dirs=_sub_activity_dirs,
                subordinate_management_files=_sub_mgmt_files,
                descendant_read_files=_desc_read_files,
                descendant_read_dirs=_desc_read_dirs,
                peer_activity_dirs=_peer_activity_dirs,
                superuser=superuser,
            )
            if violation:
                _log_tool_use(
                    anima_dir, tool_name, tool_input, tool_use_id=tool_use_id, blocked=True, block_reason=violation
                )
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
                file_path,
                anima_dir,
                write=False,
                subordinate_activity_dirs=_sub_activity_dirs,
                subordinate_management_files=_sub_mgmt_files,
                descendant_read_files=_desc_read_files,
                descendant_read_dirs=_desc_read_dirs,
                peer_activity_dirs=_peer_activity_dirs,
                superuser=superuser,
            )
            if violation:
                _log_tool_use(
                    anima_dir, tool_name, tool_input, tool_use_id=tool_use_id, blocked=True, block_reason=violation
                )
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
                _log_tool_use(
                    anima_dir, tool_name, tool_input, tool_use_id=tool_use_id, blocked=True, block_reason=violation
                )
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


# ── PostToolUse: knowledge frontmatter ──────────────────


def _build_post_tool_hook(anima_dir: Path) -> Callable:
    """Build a PostToolUse hook that updates knowledge frontmatter after Write/Edit."""

    knowledge_dir_str = str(anima_dir / "knowledge")

    async def _post_tool_hook(
        input_data: dict,
        tool_use_id: str | None,
        context: Any,
    ) -> dict:
        tool_name = input_data.get("tool_name", "")
        if tool_name not in ("Write", "Edit"):
            return {}

        file_path = input_data.get("tool_input", {}).get("file_path", "")
        if not file_path.startswith(knowledge_dir_str + "/") or not file_path.endswith(".md"):
            return {}

        import asyncio

        asyncio.create_task(_update_knowledge_frontmatter(Path(file_path)))
        return {"async_": True}

    return _post_tool_hook


# ── Stop hook: completion gate (Mode S) ────────────────────────

from core.execution._completion_gate import (
    cleanup_gate_marker as _cleanup_gate_marker,
)
from core.execution._completion_gate import (
    completion_gate_applies_to_trigger as _completion_gate_applies_to_trigger,
)


def _build_stop_hook(
    anima_dir: Path,
    *,
    session_stats: dict[str, Any] | None = None,
) -> Callable:
    """Build a Stop hook that injects the pre-completion checklist directly.

    On the first stop attempt for a gated trigger (chat, task, cron), the hook
    returns ``decision="block"`` with the checklist as ``reason``.  The SDK
    injects the reason into the model's context and forces it to continue.
    On the second attempt (``stop_hook_active=True``), the hook allows the stop.

    No ``completion_gate`` tool call is required — the hook itself is the gate.
    """
    from claude_agent_sdk.types import HookContext, SyncHookJSONOutput

    async def _stop_hook(
        input_data: dict[str, Any],
        tool_use_id: str | None,
        context: HookContext,
    ) -> SyncHookJSONOutput | dict[str, Any]:
        stop_hook_active = bool(input_data.get("stop_hook_active", False))
        if stop_hook_active:
            _cleanup_gate_marker(anima_dir)
            return {}

        trigger = (session_stats or {}).get("trigger")

        if not _completion_gate_applies_to_trigger(trigger):
            return {}

        from core.i18n import t

        return SyncHookJSONOutput(
            decision="block",
            reason=t("completion_gate.checklist"),
        )

    return _stop_hook


async def _update_knowledge_frontmatter(path: Path) -> None:
    """Update ``updated_at`` and increment ``version`` in knowledge YAML frontmatter.

    Runs as a fire-and-forget async task from the PostToolUse hook.
    Silently ignores files without frontmatter or any errors.
    """
    try:
        from core.memory.frontmatter import parse_frontmatter
        from core.time_utils import now_iso

        text = path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            return

        meta, body = parse_frontmatter(text)
        if not meta:
            return

        meta["updated_at"] = now_iso()
        meta["version"] = meta.get("version", 0) + 1

        import yaml

        from core.memory._io import atomic_write_text

        fm = yaml.dump(meta, default_flow_style=False, allow_unicode=True)
        atomic_write_text(path, f"---\n{fm}---\n\n{body.lstrip()}")
        logger.debug("Updated knowledge frontmatter: %s (version=%d)", path.name, meta["version"])
    except Exception:
        logger.debug("Failed to update frontmatter for %s", path, exc_info=True)
