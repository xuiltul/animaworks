from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Format priming result as Markdown section for system prompt injection."""

from core.i18n import t
from core.memory.priming.engine import PrimingResult


def format_priming_section(result: PrimingResult, sender_name: str = "human") -> str:
    """Format priming result as a Markdown section for system prompt injection.

    Args:
        result: The priming result to format
        sender_name: Name of the message sender

    Returns:
        Formatted markdown section, or empty string if no memories primed
    """
    from core.execution._sanitize import wrap_priming

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
        parts.append(wrap_priming("sender_profile", result.sender_profile, trust="medium"))
        parts.append("")

    if result.recent_activity:
        parts.append(t("priming.recent_activity_header"))
        parts.append("")
        parts.append(wrap_priming("recent_activity", result.recent_activity, trust="untrusted"))
        parts.append("")

    if result.related_knowledge or result.related_knowledge_untrusted:
        from core.execution._sanitize import ORIGIN_CONSOLIDATION, ORIGIN_EXTERNAL_PLATFORM

        parts.append(t("priming.related_knowledge_header"))
        parts.append("")
        if result.related_knowledge:
            if result.related_knowledge_untrusted:
                parts.append(
                    wrap_priming(
                        "related_knowledge",
                        result.related_knowledge,
                        trust="medium",
                        origin=ORIGIN_CONSOLIDATION,
                    )
                )
            else:
                parts.append(
                    wrap_priming(
                        "related_knowledge",
                        result.related_knowledge,
                        trust="medium",
                    )
                )
            parts.append("")
        if result.related_knowledge_untrusted:
            parts.append(
                wrap_priming(
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
        parts.append(wrap_priming("episodes", result.episodes, trust="medium"))
        parts.append("")

    if result.pending_tasks:
        parts.append(t("priming.pending_tasks_header"))
        parts.append("")
        parts.append(wrap_priming("pending_tasks", result.pending_tasks, trust="medium"))
        parts.append("")

    if result.recent_outbound:
        parts.append(wrap_priming("recent_outbound", result.recent_outbound, trust="trusted"))
        parts.append("")

    return "\n".join(parts)
