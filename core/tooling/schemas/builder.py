from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Tool list builder helpers."""

from typing import Any

from core.skills.trust_gate import trust_skill_enabled_for_trigger
from core.tooling.schemas.admin import _AW_CORE_NAMES, ADMIN_TOOLS, CC_TOOLS
from core.tooling.schemas.channel import _channel_tools
from core.tooling.schemas.converters import apply_db_descriptions
from core.tooling.schemas.goal import _goal_tools
from core.tooling.schemas.memory import (
    FILE_TOOLS,
    KNOWLEDGE_TOOLS,
    MEMORY_TOOLS,
    PROCEDURE_TOOLS,
    SEARCH_TOOLS,
)
from core.tooling.schemas.notification import _notification_tools
from core.tooling.schemas.session_todo import _session_todo_tools
from core.tooling.schemas.skill import (
    DISCOVERY_TOOLS,
    TOOL_MANAGEMENT_TOOLS,
    USE_TOOL,
    _create_skill_schemas,
    _curator_skill_schemas,
)
from core.tooling.schemas.supervisor import (
    _background_task_tools,
    _check_permissions_tools,
    _supervisor_tools,
    _vault_tools,
)
from core.tooling.schemas.task import _submit_tasks_tools, _task_tools
from core.tooling.schemas.workspace import WORKSPACE_TOOLS

_CONSOLIDATION_BLOCKED_TOOLS: frozenset[str] = frozenset(
    {"delegate_task", "submit_tasks", "goal", "send_message", "post_channel"}
)

_COMPACT_COMM_TOOLS: frozenset[str] = frozenset(
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
        "post_channel",
        "completion_gate",
    }
)

_SUBMIT_TASKS_ALLOWED_TRIGGERS: frozenset[str] = frozenset({"background", "submit_tasks"})
_SUBMIT_TASKS_ALLOWED_PREFIXES: tuple[str, ...] = ("background:", "submit_tasks:")


def submit_tasks_enabled_for_trigger(trigger: str | None) -> bool:
    """Return True only for explicit background task-authoring sessions."""
    normalized = (trigger or "").strip()
    return normalized in _SUBMIT_TASKS_ALLOWED_TRIGGERS or normalized.startswith(_SUBMIT_TASKS_ALLOWED_PREFIXES)


def _skill_authoring_schemas_for_trigger(trigger: str) -> list[dict[str, Any]]:
    allow_trust = trust_skill_enabled_for_trigger(trigger)
    return [t for t in _create_skill_schemas() if allow_trust or t["name"] != "trust_skill"]


def build_tool_list(
    *,
    include_file_tools: bool = False,
    include_search_tools: bool = False,
    include_discovery_tools: bool = False,
    include_use_tool: bool = False,
    include_notification_tools: bool = False,
    include_admin_tools: bool = False,
    include_supervisor_tools: bool = False,
    include_tool_management: bool = False,
    include_task_tools: bool = False,
    include_submit_tasks: bool = False,
    include_goal_tools: bool = False,
    include_background_task_tools: bool = False,
    include_vault_tools: bool = False,
    include_create_skill: bool = False,
    external_schemas: list[dict[str, Any]] | None = None,
    trigger: str = "",
) -> list[dict[str, Any]]:
    """Assemble a tool list from canonical definitions.

    Args:
        include_file_tools: Include file/command operation tools (for Mode A).
        include_search_tools: Include search_code/list_directory tools.
        include_discovery_tools: Include discover_tools tool (deprecated, now empty).
        include_use_tool: Include use_tool unified external tool dispatcher.
        include_notification_tools: Include call_human tool (for top-level Animas).
        include_admin_tools: Include admin tools (create_anima etc.).
        include_supervisor_tools: Include supervisor tools (disable/enable subordinate).
        include_tool_management: Include refresh_tools/share_tool tools.
        include_task_tools: Include task queue tools (backlog_task, update_task, list_tasks).
        include_submit_tasks: Include submit_tasks DAG batch submission tool
            only for explicit background task-authoring triggers.
        include_goal_tools: Include persistent goal loop tool.
        include_background_task_tools: Include background task check/list tools.
        include_vault_tools: Include credential vault tools (get/store/list).
        include_create_skill: Include create_skill tool.
        external_schemas: Additional tool schemas in canonical format.
        trigger: Execution trigger (e.g. ``"consolidation:daily"``).

    Returns:
        Combined list in canonical format.
    """
    is_consolidation = trigger.startswith("consolidation:")

    tools: list[dict[str, Any]] = list(MEMORY_TOOLS)
    tools.extend(_channel_tools())
    tools.extend(PROCEDURE_TOOLS)
    tools.extend(KNOWLEDGE_TOOLS)
    if not is_consolidation:
        tools.extend(WORKSPACE_TOOLS)
    tools.extend(_check_permissions_tools())
    if include_file_tools:
        tools.extend(FILE_TOOLS)
    if include_search_tools:
        tools.extend(SEARCH_TOOLS)
    if include_discovery_tools:
        tools.extend(DISCOVERY_TOOLS)
    if include_use_tool:
        tools.extend(USE_TOOL)
    if include_notification_tools:
        tools.extend(_notification_tools())
    if include_admin_tools:
        tools.extend(ADMIN_TOOLS)
    if include_supervisor_tools and not is_consolidation:
        tools.extend(_supervisor_tools())
    if include_tool_management:
        tools.extend(TOOL_MANAGEMENT_TOOLS)
    if include_task_tools:
        tools.extend(_task_tools())
    if include_submit_tasks and submit_tasks_enabled_for_trigger(trigger) and not is_consolidation:
        tools.extend(_submit_tasks_tools())
    if (include_goal_tools or include_task_tools) and not is_consolidation:
        tools.extend(_goal_tools())
    if include_background_task_tools:
        tools.extend(_background_task_tools())
    if include_vault_tools:
        tools.extend(_vault_tools())
    if external_schemas:
        tools.extend(external_schemas)

    if is_consolidation:
        tools = [t for t in tools if t["name"] not in _CONSOLIDATION_BLOCKED_TOOLS]

    tools = apply_db_descriptions(tools)

    if include_create_skill:
        tools.extend(_skill_authoring_schemas_for_trigger(trigger))
        tools.extend(_curator_skill_schemas())
    return tools


def build_unified_tool_list(
    *,
    include_notification_tools: bool = False,
    include_supervisor_tools: bool = False,
    include_create_skill: bool = True,
    trigger: str = "",
    compact: bool = False,
) -> list[dict[str, Any]]:
    """Build the unified runtime tool list.

    Used by Mode A (LiteLLM) and Mode B (Assisted) executors.
    Mode S/C get the CC tools from the Agent SDK built-ins and the AW tools
    from MCP, so they do NOT call this function.

    Args:
        include_notification_tools: Include call_human (when HumanNotifier is configured).
        include_supervisor_tools: Include delegate_task (when Anima has subordinates).
        include_create_skill: Include create_skill tool (default True).
        trigger: Execution trigger (e.g. ``"consolidation:daily"``).
            When the trigger starts with ``consolidation:``, delegation and
            messaging tools are excluded.
        compact: When True, filter to the ``_COMPACT_COMM_TOOLS`` subset
            (12 essential tools) for small context windows (<8K).

    Returns:
        Combined list in canonical format (up to 18 tools).
    """
    is_consolidation = trigger.startswith("consolidation:")

    tools: list[dict[str, Any]] = list(CC_TOOLS)

    # AW-essential: memory + messaging (always present, but filtered during consolidation)
    for t in MEMORY_TOOLS:
        if t["name"] in _AW_CORE_NAMES:
            if is_consolidation and t["name"] in _CONSOLIDATION_BLOCKED_TOOLS:
                continue
            tools.append(t)
    tools.extend(PROCEDURE_TOOLS)
    tools.extend(KNOWLEDGE_TOOLS)

    if not is_consolidation:
        tools.extend(WORKSPACE_TOOLS)
        for t in _channel_tools():
            if t["name"] == "post_channel":
                tools.append(t)
                break

    # AW-essential: notification (conditional)
    if include_notification_tools:
        tools.extend(_notification_tools())

    # AW-essential: supervisor delegation + status check (conditional, blocked during consolidation)
    if include_supervisor_tools and not is_consolidation:
        _sup_include = {"delegate_task", "ping_subordinate"}
        for t in _supervisor_tools():
            if t["name"] in _sup_include:
                tools.append(t)

    # AW-essential: task management. submit_tasks is withheld from normal
    # chat/heartbeat/cron/task sessions to avoid duplicate self-execution.
    if not is_consolidation:
        if submit_tasks_enabled_for_trigger(trigger):
            tools.extend(_submit_tasks_tools())
        tools.extend(_goal_tools())
    for t in _task_tools():
        if t["name"] == "update_task":
            tools.append(t)
            break

    # AW-essential: session todo (planning aid for Mode A)
    tools.extend(_session_todo_tools())

    # completion_gate: pre-completion verification (applicable triggers only)
    if not is_consolidation:
        from core.tooling.schemas.completion_gate import _completion_gate_tools

        tools.extend(_completion_gate_tools())

    tools = apply_db_descriptions(tools)

    if include_create_skill:
        tools.extend(
            t for t in _skill_authoring_schemas_for_trigger(trigger) if t["name"] in {"create_skill", "trust_skill"}
        )
        tools.extend(_curator_skill_schemas())

    if compact:
        tools = [t for t in tools if t["name"] in _COMPACT_COMM_TOOLS]

    return tools
