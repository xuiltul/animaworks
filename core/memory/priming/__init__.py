# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Priming layer - automatic memory retrieval (自動想起).

Implements brain-science-inspired automatic memory activation before agent
execution, reducing the need for explicit search_memory tool calls.

Based on: docs/design/priming-layer-design.md Phase 1
"""

from __future__ import annotations

# Load submodules first so engine can import them without circular import
from core.memory.priming import (
    budget,
    channel_a,
    channel_b,
    channel_c,
    channel_e,
    channel_f,
    outbound,
)
from core.memory.priming.constants import (
    _BUDGET_GREETING,
    _BUDGET_HEARTBEAT,
    _BUDGET_IMPORTANT_KNOWLEDGE,
    _BUDGET_PENDING_TASKS,
    _BUDGET_QUESTION,
    _BUDGET_RECENT_ACTIVITY,
    _BUDGET_RELATED_EPISODES,
    _BUDGET_RELATED_KNOWLEDGE,
    _BUDGET_REQUEST,
    _BUDGET_SENDER_PROFILE,
    _BUDGET_SKILL_MATCH,
    _CHARS_PER_TOKEN,
    _DEFAULT_MAX_PRIMING_TOKENS,
    _MAX_KEYWORD_INPUT_LEN,
    _MINIMAL_STOPWORDS,
    _RE_UNICODE_WORDS,
)
from core.memory.priming.engine import PrimingEngine, PrimingResult
from core.memory.priming.format import format_priming_section

__all__ = [
    "PrimingEngine",
    "PrimingResult",
    "format_priming_section",
    "_BUDGET_GREETING",
    "_BUDGET_HEARTBEAT",
    "_BUDGET_IMPORTANT_KNOWLEDGE",
    "_BUDGET_PENDING_TASKS",
    "_BUDGET_QUESTION",
    "_BUDGET_RECENT_ACTIVITY",
    "_BUDGET_REQUEST",
    "_BUDGET_RELATED_EPISODES",
    "_BUDGET_RELATED_KNOWLEDGE",
    "_BUDGET_SENDER_PROFILE",
    "_BUDGET_SKILL_MATCH",
    "_CHARS_PER_TOKEN",
    "_DEFAULT_MAX_PRIMING_TOKENS",
    "_MAX_KEYWORD_INPUT_LEN",
    "_MINIMAL_STOPWORDS",
    "_RE_UNICODE_WORDS",
]
