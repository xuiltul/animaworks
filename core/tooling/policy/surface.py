from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Single source of truth for runtime tool visibility.

The returned names are AnimaWorks tools only. Native provider tools (for
example Claude's Read/Bash tools) are supplied by their execution engine and
are included here only for the Mode A/B unified-schema builder.
"""

from dataclasses import dataclass

from core.skills.trust_gate import trust_skill_enabled_for_trigger

# Full MCP allowlist shared by every MCP-backed execution mode. Keep these
# names here rather than maintaining a second list in the MCP server.
MCP_TOOL_NAMES: tuple[str, ...] = (
    "search_memory",
    "read_memory_file",
    "write_memory_file",
    "archive_memory_file",
    "report_procedure_outcome",
    "report_knowledge_outcome",
    "send_message",
    "attach_file",
    "call_human",
    "delegate_task",
    "submit_tasks",
    "update_task",
    "list_tasks",
    "grant_workspace_access",
    "create_skill",
    "promote_procedure_to_skill",
    "curate_skills",
    "archive_skill",
    "restore_skill",
    "block_skill",
    "unblock_skill",
    "delete_skill",
    "set_skill_lifecycle",
    "create_anima",
)

SKILL_MANAGEMENT_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "curate_skills",
        "archive_skill",
        "restore_skill",
        "block_skill",
        "unblock_skill",
        "delete_skill",
        "set_skill_lifecycle",
        "promote_procedure_to_skill",
    }
)

# Tools unavailable during memory consolidation. grant_workspace_access and
# supervisor operations are included because the Mode A builder already hid
# those categories; the MCP surface now follows the same stricter policy.
CONSOLIDATION_BLOCKED_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "delegate_task",
        "ping_subordinate",
        "submit_tasks",
        "send_message",
        "grant_workspace_access",
    }
)

COMPACT_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "Read",
        "Write",
        "Edit",
        "Bash",
        "Grep",
        "Glob",
        "search_memory",
        "read_memory_file",
        "write_memory_file",
        "send_message",
    }
)

# Mode A/B candidates in the same order as build_unified_tool_list. Keeping
# candidate names here lets the resolver own both allowlisting and filtering;
# the builder only supplies schemas and applies the returned list.
_BUILDER_TOOL_NAMES: tuple[str, ...] = (
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Grep",
    "Glob",
    "WebSearch",
    "WebFetch",
    "search_memory",
    "read_memory_file",
    "write_memory_file",
    "send_message",
    "report_procedure_outcome",
    "report_knowledge_outcome",
    "grant_workspace_access",
    "attach_file",
    "call_human",
    "ping_subordinate",
    "delegate_task",
    "submit_tasks",
    "update_task",
    "list_tasks",
    "todo_write",
    "create_skill",
    "trust_skill",
    "curate_skills",
    "archive_skill",
    "restore_skill",
    "block_skill",
    "unblock_skill",
    "delete_skill",
    "set_skill_lifecycle",
)

_BUILDER_SKILL_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "create_skill",
        "trust_skill",
        "curate_skills",
        "archive_skill",
        "restore_skill",
        "block_skill",
        "unblock_skill",
        "delete_skill",
        "set_skill_lifecycle",
    }
)

_MCP_MODES: frozenset[str] = frozenset({"S", "C", "D", "G", "X", "Y"})
_BUILDER_MODES: frozenset[str] = frozenset({"A", "B"})
_SUBMIT_TASKS_ALLOWED_TRIGGERS: frozenset[str] = frozenset({"background", "submit_tasks", "heartbeat"})
_SUBMIT_TASKS_ALLOWED_PREFIXES: tuple[str, ...] = ("background:", "submit_tasks:")


@dataclass(frozen=True, slots=True)
class ToolSurfaceContext:
    """Only per-Anima/runtime attributes that affect tool visibility."""

    has_subordinates: bool = False
    has_newstaff_skill: bool = False
    include_notification_tools: bool = False
    include_create_skill: bool = True
    compact: bool = False
    trigger_scoped_tools: bool = True
    consolidation_mode: bool = False
    force_submit_tasks: bool = False


def is_full_tool_trigger(trigger: str | None) -> bool:
    """Return whether a trigger receives the complete skill-management set."""
    normalized = (trigger or "").strip()
    if not normalized:
        return True
    return (
        normalized == "heartbeat"
        or normalized.startswith("heartbeat:")
        or normalized == "consolidation"
        or normalized.startswith("consolidation:")
    )


def is_consolidation_trigger(trigger: str | None) -> bool:
    """Return whether *trigger* identifies a consolidation run."""
    normalized = (trigger or "").strip()
    return normalized == "consolidation" or normalized.startswith("consolidation:")


def submit_tasks_enabled_for_trigger(trigger: str | None) -> bool:
    """Return whether the trigger is allowed to author or resubmit tasks."""
    normalized = (trigger or "").strip()
    return normalized in _SUBMIT_TASKS_ALLOWED_TRIGGERS or normalized.startswith(_SUBMIT_TASKS_ALLOWED_PREFIXES)


def chat_attachments_enabled_for_trigger(trigger: str | None) -> bool:
    """Return whether the run is a reply to a human in a web chat."""
    return (trigger or "").strip().startswith("message:")


def resolve_tool_surface(ctx: ToolSurfaceContext, trigger: str, mode: str) -> list[str]:
    """Return the ordered visible tool names for an execution mode.

    Modes S/C/D/G/X consume the MCP profile. Modes A and legacy B use the
    unified provider-neutral builder profile. B is accepted as an alias for A
    because the model-mode resolver normalizes it to the LiteLLM path.
    """
    normalized_mode = mode.strip().upper()
    normalized_trigger = (trigger or "").strip()
    is_consolidation = ctx.consolidation_mode or is_consolidation_trigger(normalized_trigger)

    if normalized_mode in _MCP_MODES:
        names = set(MCP_TOOL_NAMES)
        if not ctx.include_notification_tools:
            names.discard("call_human")
        if not ctx.has_subordinates:
            names.discard("delegate_task")
        if not ctx.has_newstaff_skill:
            names.discard("create_anima")
        if ctx.trigger_scoped_tools and not is_full_tool_trigger(normalized_trigger):
            names.difference_update(SKILL_MANAGEMENT_TOOL_NAMES)
        if not (ctx.force_submit_tasks or submit_tasks_enabled_for_trigger(normalized_trigger)):
            names.discard("submit_tasks")
        if not chat_attachments_enabled_for_trigger(normalized_trigger):
            names.discard("attach_file")
        if is_consolidation:
            names.difference_update(CONSOLIDATION_BLOCKED_TOOL_NAMES)
        return [name for name in MCP_TOOL_NAMES if name in names]

    if normalized_mode in _BUILDER_MODES:
        names = set(_BUILDER_TOOL_NAMES)
        if not ctx.include_notification_tools:
            names.discard("call_human")
        if not ctx.has_subordinates:
            names.difference_update({"delegate_task", "ping_subordinate"})
        if not ctx.include_create_skill:
            names.difference_update(_BUILDER_SKILL_TOOL_NAMES)
        elif not trust_skill_enabled_for_trigger(normalized_trigger):
            names.discard("trust_skill")
        if ctx.trigger_scoped_tools and not is_full_tool_trigger(normalized_trigger):
            names.difference_update(SKILL_MANAGEMENT_TOOL_NAMES)
        if not submit_tasks_enabled_for_trigger(normalized_trigger):
            names.discard("submit_tasks")
        if not chat_attachments_enabled_for_trigger(normalized_trigger):
            names.discard("attach_file")
        if is_consolidation:
            names.difference_update(CONSOLIDATION_BLOCKED_TOOL_NAMES)
        if ctx.compact:
            names.intersection_update(COMPACT_TOOL_NAMES)
        return [name for name in _BUILDER_TOOL_NAMES if name in names]

    raise ValueError(f"Unsupported execution mode for tool surface: {mode!r}")


__all__ = [
    "COMPACT_TOOL_NAMES",
    "CONSOLIDATION_BLOCKED_TOOL_NAMES",
    "MCP_TOOL_NAMES",
    "SKILL_MANAGEMENT_TOOL_NAMES",
    "ToolSurfaceContext",
    "chat_attachments_enabled_for_trigger",
    "is_consolidation_trigger",
    "is_full_tool_trigger",
    "resolve_tool_surface",
    "submit_tasks_enabled_for_trigger",
]
