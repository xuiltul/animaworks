from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Memory, file, search, procedure, and knowledge tool schemas."""

from typing import Any

MEMORY_TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_memory",
        "description": (
            "Search the anima's long-term memory by semantic similarity. "
            "Returns ranked results with scores and full content. "
            "Use offset for pagination (10 results per page)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (natural language)"},
                "scope": {
                    "type": "string",
                    "enum": [
                        "knowledge",
                        "episodes",
                        "procedures",
                        "common_knowledge",
                        "skills",
                        "activity_log",
                        "all",
                    ],
                    "description": (
                        "Memory category to search. 'activity_log' searches recent tool results "
                        "and messages (last 3 days via BM25)."
                    ),
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset (0=first page, 10=second page, max 50)",
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
                "path": {"type": "string", "description": "Relative path within anima dir"},
                "content": {"type": "string"},
                "mode": {"type": "string", "enum": ["overwrite", "append"]},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "archive_memory_file",
        "description": (
            "Archive a memory file (knowledge, procedures, or state/overflow_inbox) "
            "that is no longer needed. "
            "The file is moved to archive/ directory, not permanently deleted. "
            "Use this to clean up stale, outdated, or redundant memory files, "
            "or to mark overflow inbox messages as processed."
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
