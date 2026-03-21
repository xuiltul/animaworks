from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Tool list builder helpers."""

from typing import Any

from core.tooling.schemas.admin import _AW_CORE_NAMES, ADMIN_TOOLS, CC_TOOLS
from core.tooling.schemas.channel import _channel_tools
from core.tooling.schemas.converters import apply_db_descriptions
from core.tooling.schemas.memory import (
    FILE_TOOLS,
    KNOWLEDGE_TOOLS,
    MEMORY_TOOLS,
    PROCEDURE_TOOLS,
    SEARCH_TOOLS,
)
from core.tooling.schemas.notification import _notification_tools
from core.tooling.schemas.session_todo import _session_todo_tools
from core.tooling.schemas.skill import DISCOVERY_TOOLS, TOOL_MANAGEMENT_TOOLS, USE_TOOL, _skill_tools
from core.tooling.schemas.supervisor import (
    _background_task_tools,
    _check_permissions_tools,
    _supervisor_tools,
    _vault_tools,
)
from core.tooling.schemas.task import _submit_tasks_tools, _task_tools

_CONSOLIDATION_BLOCKED_TOOLS: frozenset[str] = frozenset(
    {"delegate_task", "submit_tasks", "send_message", "post_channel"}
)


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
    include_background_task_tools: bool = False,
    include_vault_tools: bool = False,
    include_skill_tools: bool = False,
    skill_metas: list[Any] | None = None,
    common_skill_metas: list[Any] | None = None,
    procedure_metas: list[Any] | None = None,
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
        include_submit_tasks: Include submit_tasks DAG batch submission tool.
        include_background_task_tools: Include background task check/list tools.
        include_vault_tools: Include credential vault tools (get/store/list).
        include_skill_tools: Include skill on-demand loading tool.
        skill_metas: Personal skill metadata for dynamic description generation.
        common_skill_metas: Common skill metadata for dynamic description generation.
        procedure_metas: Procedure metadata for dynamic description generation.
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
    if include_submit_tasks and not is_consolidation:
        tools.extend(_submit_tasks_tools())
    if include_background_task_tools:
        tools.extend(_background_task_tools())
    if include_vault_tools:
        tools.extend(_vault_tools())
    if external_schemas:
        tools.extend(external_schemas)

    if is_consolidation:
        tools = [t for t in tools if t["name"] not in _CONSOLIDATION_BLOCKED_TOOLS]

    tools = apply_db_descriptions(tools)

    # Skill tool description is dynamically generated — append AFTER
    # apply_db_descriptions to prevent DB overwrite of <available_skills>.
    if include_skill_tools:
        from core.tooling.skill_tool import build_skill_tool_description

        desc = build_skill_tool_description(
            skill_metas or [],
            common_skill_metas or [],
            procedure_metas or [],
        )
        skill_schemas = _skill_tools()
        skill_tool_schema = {**skill_schemas[0], "description": desc}
        tools.append(skill_tool_schema)
        for st in skill_schemas[1:]:
            tools.append(st)
    return tools


def build_unified_tool_list(
    *,
    include_notification_tools: bool = False,
    include_supervisor_tools: bool = False,
    include_skill_tools: bool = True,
    skill_metas: list[Any] | None = None,
    common_skill_metas: list[Any] | None = None,
    procedure_metas: list[Any] | None = None,
    trigger: str = "",
) -> list[dict[str, Any]]:
    """Build the unified 18-tool list (Claude Code-compatible 8 + AW-essential 10).

    Used by Mode A (LiteLLM) and Mode B (Assisted) executors.
    Mode S/C get the CC tools from the Agent SDK built-ins and the AW tools
    from MCP, so they do NOT call this function.

    Args:
        include_notification_tools: Include call_human (when HumanNotifier is configured).
        include_supervisor_tools: Include delegate_task (when Anima has subordinates).
        include_skill_tools: Include skill tool (default True).
        skill_metas: Personal skill metadata for dynamic description.
        common_skill_metas: Common skill metadata for dynamic description.
        procedure_metas: Procedure metadata for dynamic description.
        trigger: Execution trigger (e.g. ``"consolidation:daily"``).
            When the trigger starts with ``consolidation:``, delegation and
            messaging tools are excluded.

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

    if not is_consolidation:
        for t in _channel_tools():
            if t["name"] == "post_channel":
                tools.append(t)
                break

    # AW-essential: notification (conditional)
    if include_notification_tools:
        tools.extend(_notification_tools())

    # AW-essential: supervisor delegation (conditional, blocked during consolidation)
    if include_supervisor_tools and not is_consolidation:
        for t in _supervisor_tools():
            if t["name"] == "delegate_task":
                tools.append(t)
                break

    # AW-essential: task management (always present, but submit_tasks blocked during consolidation)
    if not is_consolidation:
        tools.extend(_submit_tasks_tools())
    for t in _task_tools():
        if t["name"] == "update_task":
            tools.append(t)
            break

    # AW-essential: session todo (planning aid for Mode A)
    tools.extend(_session_todo_tools())

    tools = apply_db_descriptions(tools)

    # AW-essential: skill (always present, appended AFTER apply_db_descriptions)
    if include_skill_tools:
        from core.tooling.skill_tool import build_skill_tool_description

        desc = build_skill_tool_description(
            skill_metas or [],
            common_skill_metas or [],
            procedure_metas or [],
        )
        skill_schemas = _skill_tools()
        tools.append({**skill_schemas[0], "description": desc})

    return tools
