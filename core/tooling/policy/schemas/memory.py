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
            "Search YOUR OWN long-term memory (knowledge, episodes, procedures, facts) by semantic similarity. "
            "Returns ranked results with scores and full content. "
            "Use offset for pagination (10 results per page). "
            "NOT for locating subordinate Animas — use ping_subordinate for that."
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
                        "facts",
                        "common_knowledge",
                        "skills",
                        "activity_log",
                        "code",
                        "all",
                    ],
                    "description": (
                        "Memory category to search. 'facts' searches atomic extracted facts. 'activity_log' performs a full-history "
                        "keyword search across active and rotated logs, newest first, with optional time_range filtering. "
                        "Other RAG/vector activity retrieval remains limited to the recent 3 days. 'code' searches tracked files in a registered project."
                    ),
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset (0=first page, 10=second page, max 50)",
                },
                "project": {
                    "type": "string",
                    "description": "Restrict results to the given project archive (projects/<name>/ subtree)",
                },
                "time_range": {
                    "type": "object",
                    "properties": {
                        "after": {
                            "type": "string",
                            "description": "Start date (ISO format, e.g. '2026-04-01'). Results after this date.",
                        },
                        "before": {
                            "type": "string",
                            "description": "End date (ISO format, e.g. '2026-04-30'). Results before this date.",
                        },
                    },
                    "description": "Optional temporal filter. Restricts results to a date range.",
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
        "description": (
            "Write or append to a file in the anima's memory directory. Keep confirmed reusable knowledge in knowledge/; "
            "put temporary unresolved failures and PR/SHA/date-specific case records in episodes/. Do not record "
            "unverified negative claims as knowledge or skills. Case-record hints do not block writes."
        ),
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
            "use `animaworks-tool board post <channel> <text>` (Board) instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": (
                        "Recipient name: an anima name (e.g. 'sakura') or a human "
                        "alias (e.g. 'user', 'owner'). Human aliases are delivered "
                        "via the configured external channel."
                    ),
                },
                "content": {"type": "string", "description": "Message content"},
                "reply_to": {"type": "string", "description": "Message ID to reply to"},
                "thread_id": {"type": "string", "description": "Thread ID"},
                "intent": {
                    "type": "string",
                    "description": (
                        "Message intent (REQUIRED for DM), permitted values: "
                        "'report' (status/result to supervisor) or 'question' "
                        "(ask a specific question). For task assignment use "
                        "delegate_task; acknowledgments and FYI use `animaworks-tool board post <channel> <text>`."
                    ),
                },
            },
            "required": ["to", "content", "intent"],
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
                        "(e.g. 'procedures/deploy.md' or 'skills/git-flow/SKILL.md')"
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
