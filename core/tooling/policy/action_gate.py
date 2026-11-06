from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Action rule lookup for side-effect tools.

Dithered-down version of the former action memory gate.  Instead of
blocking and asking the model to read memory then retry, we simply look
up ``[ACTION-RULE]`` entries relevant to a side-effect tool call and
*attach* their body to the tool result so the model can act on them.
No blocking, no state files, no fail modes, no notifications.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("animaworks.action_memory_gate")

_HANDLER_ACTION_TOOLS: frozenset[str] = frozenset(
    {
        "send_message",
        "post_channel",
        "call_human",
        "write_memory_file",
        "create_skill",
        "gmail_draft",
        "gmail_draft_update",
        "gmail_send",
        "chatwork_send",
        "slack_send",
        "discord_send",
    }
)

_CLI_ACTION_MAP: dict[tuple[str, str], str] = {
    ("gmail", "draft"): "gmail_draft",
    ("gmail", "draft-update"): "gmail_draft_update",
    ("gmail", "send"): "gmail_send",
    ("chatwork", "send"): "chatwork_send",
    ("chatwork", "upload"): "chatwork_send",
    ("slack", "send"): "slack_send",
    ("discord", "send"): "discord_send",
    ("board", "post"): "post_channel",
}


ACTION_TOOL_NAMES: frozenset[str] = _HANDLER_ACTION_TOOLS

# Score threshold for treating a retrieved ACTION-RULE as matching.
_MATCH_SCORE_THRESHOLD = 0.80

# Hard cap on rule body length shown to the model.
_MAX_RULE_BODY_CHARS = 2000

# Maximum number of matching rules attached per call.
_MAX_RULES = 3


@dataclass
class ActionRule:
    """A single matching ACTION-RULE to show to the model."""

    rule_id: str  # 検索結果の doc_id（= read_memory_file 用の相対パスに相当）
    content: str  # ルール本文。先頭 2000 文字で切る
    score: float


def action_tool_name_for_handler(name: str) -> str | None:
    """Return the action-rule tool name for a ToolHandler schema name."""
    return name if name in _HANDLER_ACTION_TOOLS else None


def action_tool_name_for_sdk(name: str) -> str | None:
    """Return the canonical action-rule name for SDK/MCP PreToolUse names."""
    if name.startswith("mcp__aw__"):
        name = name[len("mcp__aw__") :]
    return action_tool_name_for_handler(name)


def action_tool_name_from_cli_argv(argv: list[str]) -> str | None:
    """Map ``animaworks-tool`` argv to an action-rule tool name."""
    if not argv:
        return None
    tool_name = argv[0]
    if tool_name == "submit":
        return None
    if tool_name == "call_human":
        return "call_human"

    subcommand = ""
    for arg in argv[1:]:
        if not arg.startswith("-"):
            subcommand = arg
            break
    if not subcommand:
        return None
    return _CLI_ACTION_MAP.get((tool_name, subcommand))


def _get_retriever(anima_dir: Path) -> Any | None:
    knowledge_dir = anima_dir / "knowledge"
    if not knowledge_dir.is_dir():
        return None
    try:
        from core.memory.rag import MemoryRetriever
        from core.memory.rag.indexer import MemoryIndexer
        from core.memory.rag.vector_registry import get_vector_store

        vector_store = get_vector_store(anima_dir.name)
        if vector_store is None:
            return None
        indexer = MemoryIndexer(vector_store, anima_dir.name, anima_dir)
        return MemoryRetriever(vector_store, indexer, knowledge_dir)
    except Exception:
        logger.debug("Action rule retriever init failed", exc_info=True)
        return None


def _search_action_rules(anima_dir: Path, tool_name: str, query: str) -> list[Any]:
    retriever = _get_retriever(anima_dir)
    if retriever is None:
        return []
    return retriever.search_action_rules(tool_name, query, anima_dir.name)


def _json_query(tool_name: str, args: dict[str, Any] | None) -> str:
    try:
        args_text = json.dumps(args or {}, ensure_ascii=False, default=str)
    except TypeError:
        args_text = str(args or {})
    return f"{tool_name} {args_text[:500]}"


def find_action_rules(anima_dir: Path, tool_name: str, args: dict | None) -> list[ActionRule]:
    """Search for ``[ACTION-RULE]`` entries relevant to a side-effect tool call.

    Only rules with ``score >= 0.80`` are returned, in score descending
    order, up to 3 entries.  Search infrastructure failures are logged at
    debug level and produce an empty list (the call proceeds anyway).
    Each matched rule logs one audit line.
    """
    if not tool_name:
        return []
    try:
        results = _search_action_rules(anima_dir, tool_name, _json_query(tool_name, args))
    except Exception:
        logger.debug("Action rule search failed", exc_info=True)
        return []

    rules: list[ActionRule] = []
    for result in results or []:
        score = float(getattr(result, "score", 0.0) or 0.0)
        if score < _MATCH_SCORE_THRESHOLD:
            continue
        rule_id = str(getattr(result, "doc_id", "") or "")
        content = str(getattr(result, "content", "") or "")[:_MAX_RULE_BODY_CHARS]
        rules.append(ActionRule(rule_id=rule_id, content=content, score=score))

    rules.sort(key=lambda rule: rule.score, reverse=True)
    rules = rules[:_MAX_RULES]

    for rule in rules:
        logger.info(
            "action_rule matched anima=%s tool=%s rule=%s score=%.2f",
            anima_dir.name,
            tool_name,
            rule.rule_id,
            rule.score,
        )
    return rules


def format_action_rules(rules: list[ActionRule]) -> str:
    """Render rules for the model to read.  Empty list returns ``""``."""
    if not rules:
        return ""
    from core.i18n import t

    lines = [t("action_rule.attached")]
    for rule in rules:
        lines.append(f'<action-rule path="{rule.rule_id}" score="{rule.score:.2f}">')
        lines.append(rule.content)
        lines.append("</action-rule>")
    return "\n".join(lines)
