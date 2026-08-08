from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Prompt injection defense boundary labeling.

Provides trust-level tagging for tool results, priming content, and
inbox external messages to help the model distinguish framework-controlled
data from externally-sourced or user-controllable data.

Also defines origin categories and ``resolve_trust()`` for
provenance-aware trust resolution (Phase 1 foundation).
"""

import logging
import re

logger = logging.getLogger("animaworks.execution.sanitize")

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

# Boundary tag names used by wrap_* helpers. Only these tag-like strings
# are neutralized in content (leading "<" → fullwidth "＜").
_BOUNDARY_TAG_NAMES = ("external_message", "tool_result", "priming")
_BOUNDARY_TAG_RE = re.compile(
    r"</?(?:" + "|".join(_BOUNDARY_TAG_NAMES) + r")\b",
    re.IGNORECASE,
)


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


def escape_boundary_tags(content: str) -> str:
    """Neutralize trust-boundary tag names inside untrusted content.

    Only strings that look like our boundary tags
    (``<tool_result``, ``</tool_result>``, ``<priming``, ``</priming>``,
    ``<external_message``, ``</external_message>``) have their leading
    ``<`` replaced with the fullwidth ``＜`` (U+FF1C). Ordinary HTML/XML
    tags and code fragments are left untouched so multi-language
    readability is preserved.
    """
    if not content:
        return content
    return _BOUNDARY_TAG_RE.sub(lambda m: "＜" + m.group(0)[1:], content)


def is_registered_human_sender(source: str, sender_id: str | None) -> bool:
    """Return True when *sender_id* matches a registered human platform ID.

    Uses existing config registries only (no new ID ledger):

    - ``external_messaging.user_aliases`` platform IDs
      (``slack_user_id`` / ``discord_user_id``)
    - ``interaction.default_approver_ids`` (Slack user IDs)

    Matching is **exact platform user ID** equality (not display name /
    alias key). Unconfigured or unloadable config → False (safe default).
    """
    if not sender_id:
        return False
    try:
        from core.config.models import load_config

        cfg = load_config()
    except Exception:
        logger.debug("is_registered_human_sender: config load failed", exc_info=True)
        return False

    sid = str(sender_id).strip()
    if not sid:
        return False

    # Slack approver IDs are known human operators.
    if source == "slack":
        for approver in cfg.interaction.default_approver_ids or []:
            if str(approver).strip() == sid:
                return True

    aliases = cfg.external_messaging.user_aliases or {}
    for alias_cfg in aliases.values():
        if source == "slack" and (alias_cfg.slack_user_id or "").strip() == sid:
            return True
        if source == "discord" and (alias_cfg.discord_user_id or "").strip() == sid:
            return True
        # chatwork: UserAliasConfig has room_id only (not account ID) — no
        # user-ID elevation path. zoom: no alias field yet.
    return False


# ── Tool trust levels ─────────────────────────────────────────

TOOL_TRUST_LEVELS: dict[str, str] = {
    "search_memory": "trusted",
    "read_memory_file": "trusted",
    "write_memory_file": "trusted",
    "archive_memory_file": "trusted",
    "create_skill": "trusted",
    "promote_procedure_to_skill": "trusted",
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
        Content has boundary tag names escaped so embedded
        ``</tool_result>`` cannot break out of the wrapper.
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

    escaped = escape_boundary_tags(result)
    return f"<tool_result {attrs}>\n{escaped}\n</tool_result>"


def wrap_priming(
    source: str,
    content: str,
    trust: str = "mixed",
    origin: str | None = None,
    origin_chain: list[str] | None = None,
    render_mode: str | None = None,
) -> str:
    """Wrap priming content with source and trust boundary tags.

    Args:
        source: Identifier for the priming source (e.g. channel name).
        content: Priming text to inject.
        trust: Trust level for the content (default "mixed").
            Overridden by ``resolve_trust()`` when *origin* is provided.
        origin: Origin category of the data (optional, provenance Phase 1).
        origin_chain: Intermediate origins the data traversed.
        render_mode: Optional priming gate render mode (e.g. ``pointer``,
            ``evidence``, ``guardrail``).  Omitted by default for backwards
            compatible prompt text.

    Returns:
        Content unchanged if empty/falsy; otherwise wrapped in
        ``<priming source="..." trust="..." ...>...</priming>``.
        Content has boundary tag names escaped so embedded
        ``</priming>`` cannot break out of the wrapper.
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
    if render_mode:
        attrs += f' render_mode="{render_mode}"'

    escaped = escape_boundary_tags(content)
    return f"<priming {attrs}>\n{escaped}\n</priming>"


def wrap_inbox_message(
    content: str,
    source: str,
    origin: str,
    sender: str | None = None,
) -> str:
    """Wrap an inbox message body with external-message boundary tags.

    Args:
        content: Message body (already truncated if needed).
        source: Message ``source`` field (slack/chatwork/discord/zoom/...).
        origin: Origin category from ``_SOURCE_TO_ORIGIN`` (or equivalent).
            Elevated to ``ORIGIN_HUMAN`` when *sender* matches a registered
            human platform user ID.
        sender: Platform user ID (``external_user_id``). Not a display name.

    Returns:
        Content unchanged if empty/falsy; otherwise wrapped in
        ``<external_message source="..." trust="..." sender="...">...</external_message>``.
        Boundary tag names inside the body are escaped.
    """
    if not content:
        return content

    effective_origin = origin
    if is_registered_human_sender(source, sender):
        effective_origin = ORIGIN_HUMAN

    trust = resolve_trust(effective_origin)
    safe_sender = (sender or "").replace('"', "")
    attrs = f'source="{source}" trust="{trust}"'
    if safe_sender:
        attrs += f' sender="{safe_sender}"'
    attrs += f' origin="{effective_origin}"'

    escaped = escape_boundary_tags(content)
    return f"<external_message {attrs}>\n{escaped}\n</external_message>"
