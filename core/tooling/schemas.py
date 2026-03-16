from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Canonical tool schema definitions and format converters.

All tool schemas are defined once in a provider-neutral format and converted
to Anthropic or LiteLLM/OpenAI formats on demand.  This eliminates the
duplicate definitions that previously lived in ``_build_a2_tools()`` and
``_build_anthropic_tools()``.
"""

import logging
from typing import Any

from core.exceptions import ToolConfigError  # noqa: F401
from core.i18n import t as _t

logger = logging.getLogger("animaworks.tool_schemas")


# ── DB description overlay ──────────────────────────────────


def apply_db_descriptions(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Override tool descriptions from DB if available.

    Uses a single ``list_descriptions()`` call to avoid N+1 queries.
    """
    from core.tooling.prompt_db import get_prompt_store

    store = get_prompt_store()
    if store is None:
        return tools
    all_descs = store.list_descriptions()
    if not all_descs:
        return tools
    desc_map = {d["name"]: d["description"] for d in all_descs}
    result = []
    for t in tools:
        db_desc = desc_map.get(t["name"])
        if db_desc is not None:
            t = {**t, "description": db_desc}
        result.append(t)
    return result


# ── Canonical definitions ────────────────────────────────────
#
# Format: {"name", "description", "parameters"} where ``parameters`` is a
# standard JSON Schema object.  This is convertible to both Anthropic
# (``input_schema``) and OpenAI/LiteLLM (``function.parameters``) formats.

MEMORY_TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_memory",
        "description": ("Search the anima's long-term memory (knowledge, episodes, procedures) by keyword."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search keyword"},
                "scope": {
                    "type": "string",
                    "enum": ["knowledge", "episodes", "procedures", "common_knowledge", "all"],
                    "description": "Memory category to search",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_memory_file",
        "description": "Read a file from the anima's memory directory by relative path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path within anima dir",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_memory_file",
        "description": "Write or append to a file in the anima's memory directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "mode": {"type": "string", "enum": ["overwrite", "append"]},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "archive_memory_file",
        "description": (
            "Archive a memory file (knowledge, procedures) that is no longer needed. "
            "The file is moved to archive/ directory, not permanently deleted. "
            "Use this to clean up stale, outdated, or redundant memory files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path within anima dir (e.g. 'knowledge/old-info.md')",
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for archiving (e.g. 'superseded by new-info.md')",
                },
            },
            "required": ["path", "reason"],
        },
    },
    {
        "name": "send_message",
        "description": (
            "Send a direct message to another anima or a human user. "
            "DM is limited to max 2 recipients per run, 1 message each, "
            "with intent 'report' or 'question' only. "
            "For task delegation, use delegate_task instead. "
            "For acknowledgments, FYI, or messages to 3+ people, "
            "use post_channel (Board) instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": (
                        "Recipient name. Can be an anima name (e.g. 'sakura') "
                        "or a human alias (e.g. 'user', 'taka'). "
                        "Messages to human aliases are automatically delivered "
                        "via the configured external channel."
                    ),
                },
                "content": {"type": "string", "description": "Message content"},
                "reply_to": {"type": "string", "description": "Message ID to reply to"},
                "thread_id": {"type": "string", "description": "Thread ID"},
                "intent": {
                    "type": "string",
                    "description": (
                        "Message intent (REQUIRED for DM). "
                        "Permitted values: 'report', 'question' only. "
                        "'report' = status/result to supervisor, "
                        "'question' = ask a specific question requiring a response. "
                        "For task assignment use delegate_task. "
                        "Acknowledgments, thanks, and FYI must use "
                        "post_channel (Board) instead of DM."
                    ),
                },
            },
            "required": ["to", "content", "intent"],
        },
    },
]


def _channel_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "post_channel",
            "description": _t("schema.post_channel.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": _t("schema.post_channel.channel"),
                    },
                    "text": {
                        "type": "string",
                        "description": _t("schema.post_channel.text"),
                    },
                },
                "required": ["channel", "text"],
            },
        },
        {
            "name": "read_channel",
            "description": _t("schema.read_channel.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": _t("schema.read_channel.channel"),
                    },
                    "limit": {
                        "type": "integer",
                        "description": _t("schema.read_channel.limit"),
                    },
                    "human_only": {
                        "type": "boolean",
                        "description": _t("schema.read_channel.human_only"),
                    },
                },
                "required": ["channel"],
            },
        },
        {
            "name": "read_dm_history",
            "description": _t("schema.read_dm_history.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "peer": {
                        "type": "string",
                        "description": _t("schema.read_dm_history.peer"),
                    },
                    "limit": {
                        "type": "integer",
                        "description": _t("schema.read_dm_history.limit"),
                    },
                },
                "required": ["peer"],
            },
        },
        {
            "name": "manage_channel",
            "description": _t("schema.manage_channel.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "add_member", "remove_member", "info"],
                        "description": _t("schema.manage_channel.action"),
                    },
                    "channel": {
                        "type": "string",
                        "description": _t("schema.manage_channel.channel"),
                    },
                    "members": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": _t("schema.manage_channel.members"),
                    },
                    "description": {
                        "type": "string",
                        "description": _t("schema.manage_channel.description"),
                    },
                },
                "required": ["action", "channel"],
            },
        },
    ]


FILE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": (
            "Read a file with line numbers. "
            "For large files, use offset and limit to read specific sections. "
            "Output lines are numbered in 'N|content' format."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute file path"},
                "offset": {
                    "type": "integer",
                    "description": "Starting line number (1-based, default: 1)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file (subject to permissions).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute file path"},
                "content": {"type": "string", "description": "File content"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace a specific string in a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute file path"},
                "old_string": {"type": "string", "description": "Text to find"},
                "new_string": {"type": "string", "description": "Replacement text"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "execute_command",
        "description": (
            "Execute a shell command (subject to permissions allow-list). "
            "Set background=true for long-running commands — returns immediately "
            "with a cmd_id and output file path. Read the output file to check progress."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "timeout": {
                    "type": "integer",
                    "description": ("Timeout in seconds. Default: 30 (foreground), 1800 (background)."),
                },
                "background": {
                    "type": "boolean",
                    "description": "Run in background. Returns cmd_id + output file path immediately.",
                    "default": False,
                },
            },
            "required": ["command"],
        },
    },
]

SEARCH_TOOLS: list[dict[str, Any]] = [
    {
        "name": "web_fetch",
        "description": (
            "Fetch content from a URL and return it as markdown. "
            "Use this to read web pages, documentation, articles. "
            "Content is from external sources (untrusted). "
            "Results may be truncated for large pages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch (must be fully-formed, HTTPS preferred)",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "search_code",
        "description": (
            "Search for a text pattern in files using regex. "
            "Returns matching lines with file paths and line numbers. "
            "Use this instead of execute_command with grep."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file path to search in (default: anima_dir)",
                },
                "glob": {
                    "type": "string",
                    "description": "File glob filter (e.g. '*.py', '*.md')",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "list_directory",
        "description": (
            "List files and directories at a given path. "
            "Supports glob patterns for filtering. "
            "Use this instead of execute_command with ls or find."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path (default: anima_dir)",
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern filter (e.g. '**/*.py')",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Include subdirectories (default: false)",
                },
            },
        },
    },
]


# ── Claude Code-compatible tools ──────────────────────────────
#
# These 8 tools mirror Claude Code's built-in tools.  In Mode S/C they are
# provided by the Agent SDK; in Mode A/B they are implemented by
# ``handler_files.py`` and exposed as native function-calling tools.

CC_TOOLS: list[dict[str, Any]] = [
    {
        "name": "Read",
        "description": (
            "Read a file with line numbers. "
            "For large files, use offset and limit to read specific sections. "
            "Output lines are numbered in 'N|content' format."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute file path"},
                "offset": {
                    "type": "integer",
                    "description": "Starting line number (1-based)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "Write",
        "description": "Write content to a file, creating parent directories as needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute file path"},
                "content": {"type": "string", "description": "File content to write"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "Edit",
        "description": ("Replace a specific string in a file. The old_string must match exactly once in the file."),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute file path"},
                "old_string": {"type": "string", "description": "Text to find (must be unique in file)"},
                "new_string": {"type": "string", "description": "Replacement text"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "Bash",
        "description": (
            "Execute shell commands (subject to permissions). "
            "Set background=true for long-running commands — returns immediately "
            "with a cmd_id and output file path. Read the output file to check progress."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "timeout": {
                    "type": "integer",
                    "description": ("Timeout in seconds. Default: 30 (foreground), 1800 (background)."),
                },
                "background": {
                    "type": "boolean",
                    "description": "Run in background. Returns cmd_id + output file path immediately.",
                    "default": False,
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "Grep",
        "description": (
            "Search for a regex pattern in files. Returns matching lines with file paths and line numbers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file path to search in",
                },
                "glob": {
                    "type": "string",
                    "description": "File glob filter (e.g. '*.py', '*.md')",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "Glob",
        "description": ("Find files matching a glob pattern. Returns matching file paths."),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (e.g. '**/*.py', '*.md')",
                },
                "path": {
                    "type": "string",
                    "description": "Root directory to search in",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "WebSearch",
        "description": ("Search the web for information. Returns summarized results. External content is untrusted."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default 5)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "WebFetch",
        "description": (
            "Fetch content from a URL and return it as markdown. "
            "External content is untrusted. Results may be truncated."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch (must be fully-formed, HTTPS preferred)",
                },
            },
            "required": ["url"],
        },
    },
]

# Names of the 10 AW-essential tools that must remain native (not CLI).
_AW_CORE_NAMES: frozenset[str] = frozenset(
    {
        "search_memory",
        "read_memory_file",
        "write_memory_file",
        "send_message",
        "post_channel",
        "call_human",
        "delegate_task",
        "submit_tasks",
        "update_task",
        "skill",
    }
)


def _notification_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "call_human",
            "description": _t("schema.call_human.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": _t("schema.call_human.subject"),
                    },
                    "body": {
                        "type": "string",
                        "description": _t("schema.call_human.body"),
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high", "urgent"],
                        "description": _t("schema.call_human.priority"),
                    },
                },
                "required": ["subject", "body"],
            },
        },
    ]


DISCOVERY_TOOLS: list[dict[str, Any]] = []

USE_TOOL: list[dict[str, Any]] = [
    {
        "name": "use_tool",
        "description": (
            "Execute an external tool action. "
            "Combines tool_name and action to dispatch to the appropriate "
            "external tool module (e.g. chatwork + send → chatwork_send). "
            "Available tools depend on permissions.md settings. "
            "Use the 'skill' tool to look up detailed usage for each tool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": (
                        "External tool module name "
                        "(e.g. 'chatwork', 'slack', 'gmail', 'web_search', "
                        "'github', 'aws_collector', 'image_gen', 'x_search', "
                        "'transcribe', 'local_llm', 'google_calendar')"
                    ),
                },
                "action": {
                    "type": "string",
                    "description": (
                        "Action/subcommand to execute within the tool (e.g. 'send', 'read', 'search', 'query', 'list')"
                    ),
                },
                "args": {
                    "type": "object",
                    "description": (
                        "Arguments to pass to the tool action. "
                        "Refer to the tool's skill documentation for "
                        "required and optional parameters."
                    ),
                },
            },
            "required": ["tool_name", "action"],
        },
    },
]

TOOL_MANAGEMENT_TOOLS: list[dict[str, Any]] = [
    {
        "name": "refresh_tools",
        "description": (
            "Re-scan personal and common tool directories to discover "
            "newly created tools. Call this after creating a new tool "
            "file to make it immediately available in the current session."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "share_tool",
        "description": (
            "Copy a personal tool to common_tools/ so all animas can use it. "
            "The tool file is copied from your tools/ directory to the shared "
            "common_tools/ directory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Tool file name without .py extension",
                },
            },
            "required": ["tool_name"],
        },
    },
]

ADMIN_TOOLS: list[dict[str, Any]] = [
    {
        "name": "create_anima",
        "description": (
            "Create a new Digital Anima from a character sheet. "
            "Pass the character sheet content directly via character_sheet_content, "
            "or specify a path via character_sheet_path. "
            "The factory creates the directory structure atomically, "
            "and the new anima self-configures via bootstrap on first startup."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "character_sheet_content": {
                    "type": "string",
                    "description": (
                        "Character sheet markdown content as a string. "
                        "Preferred over character_sheet_path. "
                        "Must include required sections: 基本情報, 人格, 役割・行動方針."
                    ),
                },
                "character_sheet_path": {
                    "type": "string",
                    "description": (
                        "Path to the character_sheet.md file "
                        "(absolute or relative to anima_dir). "
                        "Ignored if character_sheet_content is provided."
                    ),
                },
                "name": {
                    "type": "string",
                    "description": "Anima name (lowercase alphanumeric). If omitted, extracted from sheet.",
                },
                "supervisor": {
                    "type": "string",
                    "description": (
                        "Supervisor person name (lowercase). "
                        "Overrides the 上司 field in the character sheet. "
                        "If omitted, falls back to the sheet value, "
                        "then to the calling person."
                    ),
                },
                "role": {
                    "type": "string",
                    "enum": ["engineer", "researcher", "manager", "writer", "ops", "general"],
                    "description": (
                        "Role template to apply. Determines specialty prompt, "
                        "default model, and execution parameters. "
                        "Default: general."
                    ),
                },
            },
            "required": [],
        },
    },
]


def _supervisor_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "disable_subordinate",
            "description": _t("schema.disable_subordinate.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": _t("schema.disable_subordinate.name"),
                    },
                    "reason": {
                        "type": "string",
                        "description": _t("schema.disable_subordinate.reason"),
                    },
                },
                "required": ["name"],
            },
        },
        {
            "name": "enable_subordinate",
            "description": _t("schema.enable_subordinate.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": _t("schema.enable_subordinate.name"),
                    },
                },
                "required": ["name"],
            },
        },
        {
            "name": "set_subordinate_model",
            "description": _t("schema.set_subordinate_model.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": _t("schema.set_subordinate_model.name"),
                    },
                    "model": {
                        "type": "string",
                        "description": _t("schema.set_subordinate_model.model"),
                    },
                    "reason": {
                        "type": "string",
                        "description": _t("schema.set_subordinate_model.reason"),
                    },
                },
                "required": ["name", "model"],
            },
        },
        {
            "name": "set_subordinate_background_model",
            "description": _t("schema.set_subordinate_background_model.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": _t("schema.set_subordinate_background_model.name"),
                    },
                    "model": {
                        "type": "string",
                        "description": _t("schema.set_subordinate_background_model.model"),
                    },
                    "credential": {
                        "type": "string",
                        "description": _t("schema.set_subordinate_background_model.credential"),
                    },
                    "reason": {
                        "type": "string",
                        "description": _t("schema.set_subordinate_background_model.reason"),
                    },
                },
                "required": ["name", "model"],
            },
        },
        {
            "name": "restart_subordinate",
            "description": _t("schema.restart_subordinate.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": _t("schema.restart_subordinate.name"),
                    },
                    "reason": {
                        "type": "string",
                        "description": _t("schema.restart_subordinate.reason"),
                    },
                },
                "required": ["name"],
            },
        },
        {
            "name": "org_dashboard",
            "description": _t("schema.org_dashboard.desc"),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "ping_subordinate",
            "description": _t("schema.ping_subordinate.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": _t("schema.ping_subordinate.name"),
                    },
                },
            },
        },
        {
            "name": "read_subordinate_state",
            "description": _t("schema.read_subordinate_state.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": _t("schema.read_subordinate_state.name"),
                    },
                },
                "required": ["name"],
            },
        },
        {
            "name": "delegate_task",
            "description": _t("schema.delegate_task.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": _t("schema.delegate_task.name"),
                    },
                    "instruction": {
                        "type": "string",
                        "description": _t("schema.delegate_task.instruction"),
                    },
                    "summary": {
                        "type": "string",
                        "description": _t("schema.delegate_task.summary"),
                    },
                    "deadline": {
                        "type": "string",
                        "description": _t("schema.delegate_task.deadline"),
                    },
                    "workspace": {
                        "type": "string",
                        "description": _t("schema.delegate_task.workspace"),
                    },
                },
                "required": ["name", "instruction", "deadline"],
            },
        },
        {
            "name": "task_tracker",
            "description": _t("schema.task_tracker.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["all", "active", "completed"],
                        "description": _t("schema.task_tracker.status"),
                    },
                },
            },
        },
        {
            "name": "audit_subordinate",
            "description": _t("schema.audit_subordinate.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": _t("schema.audit_subordinate.name"),
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["summary", "report"],
                        "description": _t("schema.audit_subordinate.mode"),
                    },
                    "hours": {
                        "type": "integer",
                        "description": _t("schema.audit_subordinate.hours"),
                    },
                    "direct_only": {
                        "type": "boolean",
                        "description": _t("schema.audit_subordinate.direct_only"),
                    },
                    "since": {
                        "type": "string",
                        "description": _t("schema.audit_subordinate.since"),
                    },
                },
            },
        },
    ]


def _check_permissions_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "check_permissions",
            "description": _t("schema.check_permissions.desc"),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    ]


PROCEDURE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "report_procedure_outcome",
        "description": (
            "Report the outcome of following a procedure or skill. "
            "Updates success/failure counts and confidence. "
            "Call this after completing a procedure to track its reliability."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Relative path to the procedure or skill file "
                        "(e.g. 'procedures/deploy.md' or 'skills/git-flow.md')"
                    ),
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether the procedure succeeded",
                },
                "notes": {
                    "type": "string",
                    "description": "Optional notes on what worked or failed",
                },
            },
            "required": ["path", "success"],
        },
    },
]

KNOWLEDGE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "report_knowledge_outcome",
        "description": (
            "Report the usefulness of a knowledge file. "
            "Updates success/failure counts and confidence. "
            "Call this after using knowledge that was helpful (success=true) "
            "or found to be inaccurate/irrelevant (success=false)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": ("Relative path to the knowledge file (e.g. 'knowledge/deployment-notes.md')"),
                },
                "success": {
                    "type": "boolean",
                    "description": (
                        "Whether the knowledge was useful/accurate (true) or inaccurate/irrelevant (false)"
                    ),
                },
                "notes": {
                    "type": "string",
                    "description": "Optional notes on what was useful or inaccurate",
                },
            },
            "required": ["path", "success"],
        },
    },
]


def _vault_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "vault_get",
            "description": _t("schema.vault_get.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "description": _t("schema.vault_get.section"),
                    },
                    "key": {
                        "type": "string",
                        "description": _t("schema.vault_get.key"),
                    },
                },
                "required": ["section", "key"],
            },
        },
        {
            "name": "vault_store",
            "description": _t("schema.vault_store.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "description": _t("schema.vault_store.section"),
                    },
                    "key": {
                        "type": "string",
                        "description": _t("schema.vault_store.key"),
                    },
                    "value": {
                        "type": "string",
                        "description": _t("schema.vault_store.value"),
                    },
                },
                "required": ["section", "key", "value"],
            },
        },
        {
            "name": "vault_list",
            "description": _t("schema.vault_list.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "description": _t("schema.vault_list.section"),
                    },
                },
            },
        },
    ]


def _skill_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "skill",
            "description": _t("schema.skill.desc"),  # Enriched at runtime via build_skill_tool_description()
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": _t("schema.skill.skill_name"),
                    },
                    "context": {
                        "type": "string",
                        "description": _t("schema.skill.context"),
                    },
                },
                "required": ["skill_name"],
            },
        },
        {
            "name": "create_skill",
            "description": _t("schema.create_skill.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": _t("schema.create_skill.skill_name"),
                    },
                    "description": {
                        "type": "string",
                        "description": _t("schema.create_skill.description"),
                    },
                    "body": {
                        "type": "string",
                        "description": _t("schema.create_skill.body"),
                    },
                    "location": {
                        "type": "string",
                        "enum": ["personal", "common"],
                        "description": _t("schema.create_skill.location"),
                    },
                    "references": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "filename": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["filename", "content"],
                        },
                        "description": _t("schema.create_skill.references"),
                    },
                    "templates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "filename": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["filename", "content"],
                        },
                        "description": _t("schema.create_skill.templates"),
                    },
                    "allowed_tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": _t("schema.create_skill.allowed_tools"),
                    },
                },
                "required": ["skill_name", "description", "body"],
            },
        },
    ]


def _background_task_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "check_background_task",
            "description": _t("schema.check_background_task.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": _t("schema.check_background_task.task_id"),
                    },
                },
                "required": ["task_id"],
            },
        },
        {
            "name": "list_background_tasks",
            "description": _t("schema.list_background_tasks.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["running", "completed", "failed", "pending"],
                        "description": _t("schema.list_background_tasks.status"),
                    },
                },
            },
        },
    ]


SUBMIT_TASKS_TOOLS: list[dict[str, Any]] = [
    {
        "name": "submit_tasks",
        "description": (
            "Submit multiple tasks as a DAG for parallel/serial execution. "
            "Independent tasks with parallel=true run concurrently. "
            "Tasks with depends_on wait for all listed tasks to complete. "
            "Results from completed dependencies are automatically injected "
            "into dependent task context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "batch_id": {
                    "type": "string",
                    "description": "Unique identifier for this batch of tasks",
                },
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string"},
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "parallel": {"type": "boolean", "default": False},
                            "depends_on": {
                                "type": "array",
                                "items": {"type": "string"},
                                "default": [],
                            },
                            "acceptance_criteria": {
                                "type": "array",
                                "items": {"type": "string"},
                                "default": [],
                            },
                            "constraints": {
                                "type": "array",
                                "items": {"type": "string"},
                                "default": [],
                            },
                            "file_paths": {
                                "type": "array",
                                "items": {"type": "string"},
                                "default": [],
                            },
                            "context": {
                                "type": "string",
                                "description": "Background context for the task executor",
                                "default": "",
                            },
                            "reply_to": {
                                "type": "string",
                                "description": "Name of the Anima to notify on completion (default: submitter)",
                            },
                            "workspace": {
                                "type": "string",
                                "description": "Workspace alias or alias#hash for the task's working directory",
                                "default": "",
                            },
                        },
                        "required": ["task_id", "title", "description"],
                    },
                    "minItems": 1,
                },
            },
            "required": ["batch_id", "tasks"],
        },
    },
]


def _task_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "backlog_task",
            "description": _t("schema.backlog_task.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["human", "anima"],
                        "description": _t("schema.backlog_task.source"),
                    },
                    "original_instruction": {
                        "type": "string",
                        "description": _t("schema.backlog_task.original_instruction"),
                    },
                    "assignee": {
                        "type": "string",
                        "description": _t("schema.backlog_task.assignee"),
                    },
                    "summary": {
                        "type": "string",
                        "description": _t("schema.backlog_task.summary"),
                    },
                    "deadline": {
                        "type": "string",
                        "description": _t("schema.backlog_task.deadline"),
                    },
                    "relay_chain": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": _t("schema.backlog_task.relay_chain"),
                    },
                },
                "required": ["source", "original_instruction", "assignee", "summary", "deadline"],
            },
        },
        {
            "name": "update_task",
            "description": _t("schema.update_task.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": _t("schema.update_task.task_id"),
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "done", "cancelled", "blocked", "failed"],
                        "description": _t("schema.update_task.status"),
                    },
                    "summary": {
                        "type": "string",
                        "description": _t("schema.update_task.summary"),
                    },
                },
                "required": ["task_id", "status"],
            },
        },
        {
            "name": "list_tasks",
            "description": _t("schema.list_tasks.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "done", "cancelled", "blocked", "failed", "delegated"],
                        "description": _t("schema.list_tasks.status"),
                    },
                    "detail": {
                        "type": "boolean",
                        "description": _t("schema.list_tasks.detail"),
                    },
                },
            },
        },
    ]


# ── Format converters ────────────────────────────────────────


def to_anthropic_format(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert canonical schemas to Anthropic API format (``input_schema``)."""
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],
        }
        for t in tools
    ]


def to_litellm_format(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert canonical schemas to LiteLLM/OpenAI function calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in tools
    ]


def to_text_format(
    schemas: list[dict[str, Any]],
    *,
    locale: str | None = None,
) -> str:
    """Convert canonical tool schemas to text specification for Mode B.

    Generates a markdown-formatted tool guide that instructs the LLM to
    output tool calls as JSON code blocks.  Used by ``AssistedExecutor``
    to inject tool specifications into the system prompt.

    Includes imperative instructions, few-shot examples, and
    anti-hallucination rules to maximise tool-call compliance from
    weaker models.
    """
    header = _t("schema.text_format.header")
    instruction = _t("schema.text_format.instruction")
    example = _t("schema.text_format.example")
    rules = [
        _t("schema.text_format.rule_wait"),
        _t("schema.text_format.rule_plain_text"),
        _t("schema.text_format.rule_one_call"),
        _t("schema.text_format.rule_no_fabricate"),
        _t("schema.text_format.rule_no_empty_promise"),
    ]
    fewshot_header = _t("schema.text_format.fewshot_header")
    fewshot_items = [
        (
            _t("schema.text_format.fewshot1_prompt"),
            '```json\n{"tool": "Bash", "arguments": {"command": "docker ps"}}\n```',
        ),
        (
            _t("schema.text_format.fewshot2_prompt"),
            '```json\n{"tool": "Bash", "arguments": {"command": "free -h"}}\n```',
        ),
    ]
    args_label = _t("schema.text_format.args_label")
    required_label = _t("schema.text_format.required_label")
    tools_header = _t("schema.text_format.tools_header")

    lines = [
        header,
        "",
        instruction,
        "",
        "```json",
        example,
        "```",
        "",
    ]
    for rule in rules:
        lines.append(f"- {rule}")
    lines.append("")

    # Few-shot examples
    lines.append(fewshot_header)
    lines.append("")
    for prompt_ex, call_ex in fewshot_items:
        lines.append(prompt_ex)
        lines.append("")
        lines.append(call_ex)
        lines.append("")

    # Tool list
    lines.append(tools_header)
    lines.append("")
    for schema in schemas:
        name = schema["name"]
        desc = schema.get("description", "")
        params = schema.get("parameters", {}).get("properties", {})
        required = set(schema.get("parameters", {}).get("required", []))
        args_parts = []
        for k, v in params.items():
            type_str = v.get("type", "?")
            req_str = f" {required_label}" if k in required else ""
            args_parts.append(f"{k}: {type_str}{req_str}")
        args_desc = ", ".join(args_parts)
        lines.append(f"- **{name}**: {desc}")
        if args_desc:
            lines.append(f"  - {args_label}: {args_desc}")
    return "\n".join(lines)


# ── Builder helpers ──────────────────────────────────────────


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

    Returns:
        Combined list in canonical format.
    """
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
    if include_supervisor_tools:
        tools.extend(_supervisor_tools())
    if include_tool_management:
        tools.extend(TOOL_MANAGEMENT_TOOLS)
    if include_task_tools:
        tools.extend(_task_tools())
    if include_submit_tasks:
        tools.extend(SUBMIT_TASKS_TOOLS)
    if include_background_task_tools:
        tools.extend(_background_task_tools())
    if include_vault_tools:
        tools.extend(_vault_tools())
    if external_schemas:
        tools.extend(external_schemas)
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

    Returns:
        Combined list in canonical format (up to 18 tools).
    """
    tools: list[dict[str, Any]] = list(CC_TOOLS)

    # AW-essential: memory + messaging (always present)
    for t in MEMORY_TOOLS:
        if t["name"] in _AW_CORE_NAMES:
            tools.append(t)

    for t in _channel_tools():
        if t["name"] == "post_channel":
            tools.append(t)
            break

    # AW-essential: notification (conditional)
    if include_notification_tools:
        tools.extend(_notification_tools())

    # AW-essential: supervisor delegation (conditional)
    if include_supervisor_tools:
        for t in _supervisor_tools():
            if t["name"] == "delegate_task":
                tools.append(t)
                break

    # AW-essential: task management (always present)
    tools.extend(SUBMIT_TASKS_TOOLS)
    for t in _task_tools():
        if t["name"] == "update_task":
            tools.append(t)
            break

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


# ── Schema loading ───────────────────────────────────────────


def _normalise_schema(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalise a single tool schema to canonical format."""
    return {
        "name": raw["name"],
        "description": raw.get("description", ""),
        "parameters": raw.get("input_schema", raw.get("parameters", {})),
    }


def load_external_schemas(tool_registry: list[str]) -> list[dict[str, Any]]:
    """Load schemas from external tool modules, normalised to canonical format."""
    if not tool_registry:
        return []

    import importlib

    from core.tools import TOOL_MODULES

    schemas: list[dict[str, Any]] = []
    for tool_name in tool_registry:
        if tool_name not in TOOL_MODULES:
            continue
        try:
            mod = importlib.import_module(TOOL_MODULES[tool_name])
            if not hasattr(mod, "get_tool_schemas"):
                continue
            for s in mod.get_tool_schemas():
                schemas.append(_normalise_schema(s))
        except Exception:
            logger.debug("Failed to load schemas for %s", tool_name, exc_info=True)
    return schemas


def load_personal_tool_schemas(
    personal_tools: dict[str, str],
) -> list[dict[str, Any]]:
    """Load schemas from personal tool modules, normalised to canonical format."""
    import importlib.util

    schemas: list[dict[str, Any]] = []
    for tool_name, file_path in personal_tools.items():
        try:
            spec = importlib.util.spec_from_file_location(
                f"animaworks_personal_tool_{tool_name}",
                file_path,
            )
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            if not hasattr(mod, "get_tool_schemas"):
                continue
            for s in mod.get_tool_schemas():
                schemas.append(_normalise_schema(s))
        except Exception:
            logger.debug(
                "Failed to load personal tool schemas: %s",
                tool_name,
                exc_info=True,
            )
    return schemas


def load_external_schemas_by_category(
    categories: set[str],
) -> list[dict[str, Any]]:
    """Load external tool schemas filtered by permitted categories.

    *categories* is a set of tool module names (e.g. ``{"chatwork", "slack"}``).
    Only schemas belonging to those modules are returned.
    """
    from core.tools import TOOL_MODULES

    filtered_registry = [name for name in TOOL_MODULES if name in categories]
    return load_external_schemas(filtered_registry)


def load_all_tool_schemas(
    tool_registry: list[str] | None = None,
    personal_tools: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Load and normalise tool schemas from all enabled modules."""
    schemas = load_external_schemas(tool_registry or [])
    if personal_tools:
        schemas.extend(load_personal_tool_schemas(personal_tools))
    return schemas
