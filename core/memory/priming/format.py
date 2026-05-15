from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Format priming result as Markdown section for system prompt injection."""

import re
from typing import Any

from core.i18n import t
from core.memory.priming.result import PrimingResult

_POINTER_RE = re.compile(r'read_memory_file\(path="([^"]+)"\)')


def _decision_for(result: PrimingResult, channel: str) -> Any | None:
    plan = getattr(result, "gate_plan", None)
    decisions = getattr(plan, "channel_decisions", None)
    if not decisions:
        return None
    return decisions.get(channel)


def _render_mode_for(result: PrimingResult, channel: str) -> str | None:
    decision = _decision_for(result, channel)
    mode = getattr(decision, "render_mode", None)
    if mode is None:
        return None
    return str(getattr(mode, "value", mode))


def _content_for_render_mode(content: str, render_mode: str | None) -> str:
    if not content or render_mode != "pointer":
        return content
    return _collapse_pointer_content(content)


def _collapse_pointer_content(content: str) -> str:
    """Collapse pointer-mode memories to cue + read_memory_file lines."""
    collapsed: list[str] = []
    seen: set[str] = set()
    previous_label = ""

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = _POINTER_RE.search(line)
        if match:
            path = match.group(1)
            before_pointer = line[: match.start()]
            label = _clean_pointer_label(before_pointer) or previous_label
            if label:
                rendered = f'- {label} -> read_memory_file(path="{path}")'
            else:
                rendered = f'- read_memory_file(path="{path}")'
            if rendered not in seen:
                collapsed.append(rendered)
                seen.add(rendered)
            continue

        if line.startswith("<") or line.startswith("--- Result"):
            continue
        previous_label = _clean_pointer_label(line)

    return "\n".join(collapsed) if collapsed else content


def _clean_pointer_label(text: str) -> str:
    text = re.sub(r"^[^\w\[]+", "", text.strip(), flags=re.UNICODE)
    text = text.strip(" \t-:>")
    text = re.sub(r"\s+", " ", text)
    if len(text) > 160:
        text = text[:157] + "..."
    return text


def _wrap_priming_for_mode(
    result: PrimingResult,
    channel: str,
    source: str,
    content: str,
    *,
    trust: str = "mixed",
    origin: str | None = None,
    origin_chain: list[str] | None = None,
) -> str:
    from core.execution._sanitize import wrap_priming

    render_mode = _render_mode_for(result, channel)
    rendered_content = _content_for_render_mode(content, render_mode)
    return wrap_priming(
        source,
        rendered_content,
        trust=trust,
        origin=origin,
        origin_chain=origin_chain,
        render_mode=render_mode,
    )


def format_priming_section(result: PrimingResult, sender_name: str = "human") -> str:
    """Format priming result as a Markdown section for system prompt injection.

    Args:
        result: The priming result to format
        sender_name: Name of the message sender

    Returns:
        Formatted markdown section, or empty string if no memories primed
    """
    if result.is_empty():
        return ""

    parts: list[str] = []
    parts.append(t("priming.section_title"))
    parts.append("")
    parts.append(t("priming.section_intro"))
    parts.append("")

    if result.sender_profile:
        parts.append(t("priming.about_sender", sender_name=sender_name))
        parts.append("")
        parts.append(
            _wrap_priming_for_mode(
                result,
                "sender_profile",
                "sender_profile",
                result.sender_profile,
                trust="medium",
            )
        )
        parts.append("")

    if result.recent_activity:
        parts.append(t("priming.recent_activity_header"))
        parts.append("")
        parts.append(
            _wrap_priming_for_mode(
                result,
                "recent_activity",
                "recent_activity",
                result.recent_activity,
                trust="untrusted",
            )
        )
        parts.append("")

    if result.related_knowledge or result.related_knowledge_untrusted:
        from core.execution._sanitize import ORIGIN_CONSOLIDATION, ORIGIN_EXTERNAL_PLATFORM

        parts.append(t("priming.related_knowledge_header"))
        parts.append("")
        if result.related_knowledge:
            if result.related_knowledge_untrusted:
                parts.append(
                    _wrap_priming_for_mode(
                        result,
                        "related_knowledge",
                        "related_knowledge",
                        result.related_knowledge,
                        trust="medium",
                        origin=ORIGIN_CONSOLIDATION,
                    )
                )
            else:
                parts.append(
                    _wrap_priming_for_mode(
                        result,
                        "related_knowledge",
                        "related_knowledge",
                        result.related_knowledge,
                        trust="medium",
                    )
                )
            parts.append("")
        if result.related_knowledge_untrusted:
            parts.append(
                _wrap_priming_for_mode(
                    result,
                    "related_knowledge_untrusted",
                    "related_knowledge_external",
                    result.related_knowledge_untrusted,
                    trust="untrusted",
                    origin=ORIGIN_EXTERNAL_PLATFORM,
                )
            )
            parts.append("")

    if result.episodes:
        parts.append(t("priming.episodes_header"))
        parts.append("")
        parts.append(_wrap_priming_for_mode(result, "episodes", "episodes", result.episodes, trust="medium"))
        parts.append("")

    if result.pending_tasks:
        parts.append(t("priming.pending_tasks_header"))
        parts.append("")
        parts.append(
            _wrap_priming_for_mode(
                result,
                "pending_tasks",
                "pending_tasks",
                result.pending_tasks,
                trust="medium",
            )
        )
        parts.append("")

    if result.recent_outbound:
        parts.append(
            _wrap_priming_for_mode(
                result,
                "recent_outbound",
                "recent_outbound",
                result.recent_outbound,
                trust="trusted",
            )
        )
        parts.append("")

    if result.graph_context:
        parts.append(
            _wrap_priming_for_mode(
                result,
                "graph_context",
                "graph_context",
                result.graph_context,
                trust="medium",
            )
        )
        parts.append("")

    return "\n".join(parts)
