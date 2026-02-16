from __future__ import annotations
# AnimaWorks - Digital Person Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of AnimaWorks core/server, licensed under AGPL-3.0.
# See LICENSES/AGPL-3.0.txt for the full license text.


"""Canonical tool schema definitions and format converters.

All tool schemas are defined once in a provider-neutral format and converted
to Anthropic or LiteLLM/OpenAI formats on demand.  This eliminates the
duplicate definitions that previously lived in ``_build_a2_tools()`` and
``_build_anthropic_tools()``.
"""

import logging
from typing import Any

logger = logging.getLogger("animaworks.tool_schemas")

# ── Canonical definitions ────────────────────────────────────
#
# Format: {"name", "description", "parameters"} where ``parameters`` is a
# standard JSON Schema object.  This is convertible to both Anthropic
# (``input_schema``) and OpenAI/LiteLLM (``function.parameters``) formats.

MEMORY_TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_memory",
        "description": (
            "Search the person's long-term memory "
            "(knowledge, episodes, procedures) by keyword."
        ),
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
        "description": "Read a file from the person's memory directory by relative path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path within person dir",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_memory_file",
        "description": "Write or append to a file in the person's memory directory.",
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
        "name": "send_message",
        "description": "Send a message to another person.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient person name"},
                "content": {"type": "string", "description": "Message content"},
                "reply_to": {"type": "string", "description": "Message ID to reply to"},
                "thread_id": {"type": "string", "description": "Thread ID"},
            },
            "required": ["to", "content"],
        },
    },
]

FILE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read an arbitrary file (subject to permissions).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute file path"},
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
        "description": "Execute a shell command (subject to permissions allow-list).",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 30)",
                },
            },
            "required": ["command"],
        },
    },
]

SEARCH_TOOLS: list[dict[str, Any]] = [
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
                    "description": "Directory or file path to search in (default: person_dir)",
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
                    "description": "Directory path (default: person_dir)",
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

NOTIFICATION_TOOLS: list[dict[str, Any]] = [
    {
        "name": "notify_human",
        "description": (
            "人間の管理者に通知を送信します。"
            "重要な報告、問題のエスカレーション、判断が必要な事項がある場合に使用してください。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "通知の件名（簡潔に）",
                },
                "body": {
                    "type": "string",
                    "description": "通知の本文（詳細な報告内容）",
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "normal", "high", "urgent"],
                    "description": "通知の優先度（デフォルト: normal）",
                },
            },
            "required": ["subject", "body"],
        },
    },
]

DISCOVERY_TOOLS: list[dict[str, Any]] = [
    {
        "name": "discover_tools",
        "description": (
            "Discover available external tools. "
            "Call without arguments to list available categories. "
            "Call with a category name to activate that category's tools."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": (
                        "Tool category to activate "
                        "(e.g. 'chatwork', 'slack', 'gmail'). "
                        "Omit to list all available categories."
                    ),
                },
            },
        },
    },
]

ADMIN_TOOLS: list[dict[str, Any]] = [
    {
        "name": "create_person",
        "description": (
            "Create a new Digital Person from a character sheet. "
            "Write the character_sheet.md first, then call this tool. "
            "The factory creates the directory structure atomically, "
            "and the new person self-configures via bootstrap on first startup."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "character_sheet_path": {
                    "type": "string",
                    "description": "Path to the character_sheet.md file (absolute or relative to person_dir)",
                },
                "name": {
                    "type": "string",
                    "description": "Person name (lowercase alphanumeric). If omitted, extracted from sheet.",
                },
            },
            "required": ["character_sheet_path"],
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


# ── Builder helpers ──────────────────────────────────────────


def build_tool_list(
    *,
    include_file_tools: bool = False,
    include_search_tools: bool = False,
    include_discovery_tools: bool = False,
    include_notification_tools: bool = False,
    include_admin_tools: bool = False,
    external_schemas: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Assemble a tool list from canonical definitions.

    Args:
        include_file_tools: Include file/command operation tools (for A2 mode).
        include_search_tools: Include search_code/list_directory tools.
        include_discovery_tools: Include discover_tools tool.
        include_notification_tools: Include notify_human tool (for top-level Persons).
        include_admin_tools: Include admin tools (create_person etc.).
        external_schemas: Additional tool schemas in canonical format.

    Returns:
        Combined list in canonical format.
    """
    tools: list[dict[str, Any]] = list(MEMORY_TOOLS)
    if include_file_tools:
        tools.extend(FILE_TOOLS)
    if include_search_tools:
        tools.extend(SEARCH_TOOLS)
    if include_discovery_tools:
        tools.extend(DISCOVERY_TOOLS)
    if include_notification_tools:
        tools.extend(NOTIFICATION_TOOLS)
    if include_admin_tools:
        tools.extend(ADMIN_TOOLS)
    if external_schemas:
        tools.extend(external_schemas)
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
                f"animaworks_personal_tool_{tool_name}", file_path,
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
                tool_name, exc_info=True,
            )
    return schemas


def load_all_tool_schemas(
    tool_registry: list[str] | None = None,
    personal_tools: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Load and normalise tool schemas from all enabled modules."""
    schemas = load_external_schemas(tool_registry or [])
    if personal_tools:
        schemas.extend(load_personal_tool_schemas(personal_tools))
    return schemas
