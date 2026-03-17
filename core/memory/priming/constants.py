from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Priming layer constants."""

import re

# Default maximum tokens for entire priming injection
_DEFAULT_MAX_PRIMING_TOKENS = 2000

# Message type budgets (Phase 3: dynamic budget adjustment)
_BUDGET_GREETING = 500
_BUDGET_QUESTION = 1500
_BUDGET_REQUEST = 3000
_BUDGET_HEARTBEAT = 200

# Channel-specific token budgets (default distribution)
_BUDGET_SENDER_PROFILE = 500
_BUDGET_RECENT_ACTIVITY = 1300  # Unified: old B(600) + E(700)
_BUDGET_RELATED_KNOWLEDGE = 1000
_BUDGET_IMPORTANT_KNOWLEDGE = 500
_BUDGET_SKILL_MATCH = 200
_BUDGET_PENDING_TASKS = 500
_BUDGET_RELATED_EPISODES = 500

# Rough characters-per-token for Japanese/English mixed text
_CHARS_PER_TOKEN = 4

# Pre-compiled regex pattern for language-agnostic keyword extraction
_RE_UNICODE_WORDS = re.compile(r"[\w]+", re.UNICODE)
# Maximum message length to process for keyword extraction
_MAX_KEYWORD_INPUT_LEN = 5000

# Minimal stopwords: only clear function words that pass the length filter.
# Intentionally small — dual-query strategy reduces dependence on keyword quality.
_MINIMAL_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "can",
        "her",
        "was",
        "one",
        "our",
        "out",
        "has",
        "had",
        "with",
        "from",
        "this",
        "that",
        "they",
        "been",
        "have",
        "will",
        "would",
        "could",
        "should",
        "about",
        "which",
        "their",
        "there",
        "these",
        "those",
        "being",
        "through",
        "during",
        "before",
        "after",
        "into",
        "の",
        "に",
        "は",
        "を",
        "が",
        "で",
        "と",
        "も",
        "や",
        "へ",
        "より",
        "から",
        "まで",
        "など",
        "について",
        "している",
        "できる",
        "ために",
        "として",
        "における",
        "これ",
        "それ",
        "あれ",
    }
)
