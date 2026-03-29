from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Admin and Claude Code-compatible tool schemas."""

from typing import Any

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
        "create_skill",
    }
)

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
