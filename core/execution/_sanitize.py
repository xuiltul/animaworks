from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Prompt injection defense boundary labeling.

Provides trust-level tagging for tool results and priming content
to help the model distinguish framework-controlled data from
externally-sourced or user-controllable data.
"""

# ── Tool trust levels ─────────────────────────────────────────

TOOL_TRUST_LEVELS: dict[str, str] = {
    "search_memory": "trusted",
    "read_memory_file": "trusted",
    "write_memory_file": "trusted",
    "archive_memory_file": "trusted",
    "skill": "trusted",
    "list_directory": "trusted",
    "report_procedure_outcome": "trusted",
    "report_knowledge_outcome": "trusted",
    "discover_tools": "trusted",
    "refresh_tools": "trusted",
    "share_tool": "trusted",
    "add_task": "trusted",
    "update_task": "trusted",
    "list_tasks": "trusted",
    "post_channel": "trusted",
    "send_message": "trusted",
    "create_anima": "trusted",
    "disable_subordinate": "trusted",
    "enable_subordinate": "trusted",
    "set_subordinate_model": "trusted",
    "restart_subordinate": "trusted",
    "call_human": "trusted",
    "read_file": "medium",
    "search_code": "medium",
    "write_file": "medium",
    "edit_file": "medium",
    "execute_command": "medium",
    "web_fetch": "untrusted",
    "read_channel": "untrusted",
    "read_dm_history": "untrusted",
    "web_search": "untrusted",
    "x_search": "untrusted",
    "x_user_tweets": "untrusted",
    "slack_messages": "untrusted",
    "slack_search": "untrusted",
    "slack_unreplied": "untrusted",
    "slack_channels": "untrusted",
    "chatwork_messages": "untrusted",
    "chatwork_search": "untrusted",
    "chatwork_unreplied": "untrusted",
    "chatwork_mentions": "untrusted",
    "chatwork_rooms": "untrusted",
    "gmail_unread": "untrusted",
    "gmail_read_body": "untrusted",
    "local_llm": "untrusted",
}


# ── Boundary wrappers ──────────────────────────────────────────

def wrap_tool_result(tool_name: str, result: str) -> str:
    """Wrap a tool result with trust-level boundary tags.

    Args:
        tool_name: Name of the tool that produced the result.
        result: Raw tool output string.

    Returns:
        Result unchanged if empty/falsy; otherwise wrapped in
        <tool_result tool="..." trust="...">...</tool_result>.
    """
    if not result:
        return result
    trust = TOOL_TRUST_LEVELS.get(tool_name, "untrusted")
    return f'<tool_result tool="{tool_name}" trust="{trust}">\n{result}\n</tool_result>'


def wrap_priming(source: str, content: str, trust: str = "mixed") -> str:
    """Wrap priming content with source and trust boundary tags.

    Args:
        source: Identifier for the priming source (e.g. channel name).
        content: Priming text to inject.
        trust: Trust level for the content (default "mixed").

    Returns:
        Content unchanged if empty/falsy; otherwise wrapped in
        <priming source="..." trust="...">...</priming>.
    """
    if not content:
        return content
    return f'<priming source="{source}" trust="{trust}">\n{content}\n</priming>'
