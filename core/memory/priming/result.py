from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared priming result container."""

from dataclasses import dataclass, field
from typing import Any

from core.memory.priming.items import MemoryItem
from core.prompt.tokens import estimate_tokens


@dataclass
class PrimingResult:
    """Result of priming memory retrieval."""

    sender_profile: str = ""
    recent_activity: str = ""
    related_knowledge: str = ""
    related_knowledge_untrusted: str = ""
    pending_tasks: str = ""
    recent_outbound: str = ""
    episodes: str = ""
    pending_human_notifications: str = ""
    graph_context: str = ""
    items: dict[str, tuple[MemoryItem, ...]] = field(default_factory=dict)
    gate_plan: Any | None = None
    resident_knowledge: str = ""

    def is_empty(self) -> bool:
        """Return True if no memories were primed."""
        return (
            not self.sender_profile
            and not self.resident_knowledge
            and not self.recent_activity
            and not self.related_knowledge
            and not self.related_knowledge_untrusted
            and not self.pending_tasks
            and not self.recent_outbound
            and not self.episodes
            and not self.pending_human_notifications
            and not self.graph_context
        )

    def total_chars(self) -> int:
        """Estimate total character count."""
        return (
            len(self.sender_profile)
            + len(self.resident_knowledge)
            + len(self.recent_activity)
            + len(self.related_knowledge)
            + len(self.related_knowledge_untrusted)
            + len(self.pending_tasks)
            + len(self.recent_outbound)
            + len(self.episodes)
            + len(self.pending_human_notifications)
            + len(self.graph_context)
        )

    def estimated_tokens(self) -> int:
        """Estimate token count."""
        return estimate_tokens(
            "".join(
                (
                    self.sender_profile,
                    self.resident_knowledge,
                    self.recent_activity,
                    self.related_knowledge,
                    self.related_knowledge_untrusted,
                    self.pending_tasks,
                    self.recent_outbound,
                    self.episodes,
                    self.pending_human_notifications,
                    self.graph_context,
                )
            )
        )
