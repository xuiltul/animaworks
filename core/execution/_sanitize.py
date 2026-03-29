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

Also defines origin categories and ``resolve_trust()`` for
provenance-aware trust resolution (Phase 1 foundation).
"""

# ── Origin categories ─────────────────────────────────────────

ORIGIN_SYSTEM: str = "system"
ORIGIN_HUMAN: str = "human"
ORIGIN_ANIMA: str = "anima"
ORIGIN_EXTERNAL_PLATFORM: str = "external_platform"
ORIGIN_EXTERNAL_WEB: str = "external_web"
ORIGIN_CONSOLIDATION: str = "consolidation"
ORIGIN_UNKNOWN: str = "unknown"

ORIGIN_TRUST_MAP: dict[str, str] = {
    ORIGIN_SYSTEM: "trusted",
    ORIGIN_HUMAN: "medium",
    ORIGIN_ANIMA: "trusted",
    ORIGIN_EXTERNAL_PLATFORM: "untrusted",
    ORIGIN_EXTERNAL_WEB: "untrusted",
    ORIGIN_CONSOLIDATION: "medium",
    ORIGIN_UNKNOWN: "untrusted",
}

MAX_ORIGIN_CHAIN_LENGTH: int = 10

_TRUST_RANK: dict[str, int] = {"trusted": 2, "medium": 1, "untrusted": 0}
_RANK_TRUST: dict[int, str] = {v: k for k, v in _TRUST_RANK.items()}


def resolve_trust(
    origin: str | None = None,
    origin_chain: list[str] | None = None,
) -> str:
    """Resolve trust level from origin and optional origin_chain.

    When *origin_chain* is present the function returns the **minimum**
    trust across all nodes in the chain plus the *origin* itself
    (conservative / anti-laundering default).

    Trust hierarchy: trusted > medium > untrusted
    """
    if origin is None and origin_chain is None:
        return "untrusted"

    base_trust = ORIGIN_TRUST_MAP.get(origin or ORIGIN_UNKNOWN, "untrusted")

    if not origin_chain:
        return base_trust

    chain = origin_chain[:MAX_ORIGIN_CHAIN_LENGTH]
    all_origins = chain + [origin or ORIGIN_UNKNOWN]
    trusts = [ORIGIN_TRUST_MAP.get(o, "untrusted") for o in all_origins]
    min_rank = min(_TRUST_RANK.get(t, 0) for t in trusts)
    return _RANK_TRUST[min_rank]


# ── Tool trust levels ─────────────────────────────────────────

TOOL_TRUST_LEVELS: dict[str, str] = {
    "search_memory": "trusted",
    "read_memory_file": "trusted",
    "write_memory_file": "trusted",
    "archive_memory_file": "trusted",
    "create_skill": "trusted",
    "list_directory": "trusted",
    "report_procedure_outcome": "trusted",
    "report_knowledge_outcome": "trusted",
    # discover_tools: deprecated (DISCOVERY_TOOLS is empty)
    "refresh_tools": "trusted",
    "share_tool": "trusted",
    "backlog_task": "trusted",
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
    "slack_channel_post": "untrusted",
    "slack_channel_update": "untrusted",
    "chatwork_messages": "untrusted",
    "chatwork_search": "untrusted",
    "chatwork_unreplied": "untrusted",
    "chatwork_mentions": "untrusted",
    "chatwork_rooms": "untrusted",
    "gmail_unread": "untrusted",
    "gmail_read_body": "untrusted",
    "google_tasks_list_tasklists": "untrusted",
    "google_tasks_list_tasks": "untrusted",
    "google_tasks_insert_task": "untrusted",
    "google_tasks_insert_tasklist": "untrusted",
    "google_tasks_update_task": "untrusted",
    "google_tasks_update_tasklist": "untrusted",
    "local_llm": "untrusted",
    "completion_gate": "trusted",
}


# ── Boundary wrappers ──────────────────────────────────────────


def wrap_tool_result(
    tool_name: str,
    result: str,
    origin: str | None = None,
    origin_chain: list[str] | None = None,
) -> str:
    """Wrap a tool result with trust-level boundary tags.

    Args:
        tool_name: Name of the tool that produced the result.
        result: Raw tool output string.
        origin: Origin category of the data (optional, provenance Phase 1).
            When provided, trust is resolved via ``resolve_trust()``
            instead of the static ``TOOL_TRUST_LEVELS`` lookup.
        origin_chain: Intermediate origins the data traversed.

    Returns:
        Result unchanged if empty/falsy; otherwise wrapped in
        ``<tool_result tool="..." trust="..." ...>...</tool_result>``.
    """
    if not result:
        return result

    if origin is not None:
        trust = resolve_trust(origin, origin_chain)
    else:
        trust = TOOL_TRUST_LEVELS.get(tool_name, "untrusted")

    attrs = f'tool="{tool_name}" trust="{trust}"'
    if origin:
        attrs += f' origin="{origin}"'
    if origin_chain:
        attrs += f' origin_chain="{",".join(origin_chain[:MAX_ORIGIN_CHAIN_LENGTH])}"'

    return f"<tool_result {attrs}>\n{result}\n</tool_result>"


def wrap_priming(
    source: str,
    content: str,
    trust: str = "mixed",
    origin: str | None = None,
    origin_chain: list[str] | None = None,
) -> str:
    """Wrap priming content with source and trust boundary tags.

    Args:
        source: Identifier for the priming source (e.g. channel name).
        content: Priming text to inject.
        trust: Trust level for the content (default "mixed").
            Overridden by ``resolve_trust()`` when *origin* is provided.
        origin: Origin category of the data (optional, provenance Phase 1).
        origin_chain: Intermediate origins the data traversed.

    Returns:
        Content unchanged if empty/falsy; otherwise wrapped in
        ``<priming source="..." trust="..." ...>...</priming>``.
    """
    if not content:
        return content

    effective_trust = trust
    if origin is not None:
        effective_trust = resolve_trust(origin, origin_chain)

    attrs = f'source="{source}" trust="{effective_trust}"'
    if origin:
        attrs += f' origin="{origin}"'
    if origin_chain:
        attrs += f' origin_chain="{",".join(origin_chain[:MAX_ORIGIN_CHAIN_LENGTH])}"'

    return f"<priming {attrs}>\n{content}\n</priming>"
