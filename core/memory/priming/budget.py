from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Budget classification and adjustment for priming."""

import logging

from core.memory.priming.constants import (
    _BUDGET_GREETING,
    _BUDGET_HEARTBEAT,
    _BUDGET_QUESTION,
    _BUDGET_REQUEST,
    _DEFAULT_MAX_PRIMING_TOKENS,
)

logger = logging.getLogger("animaworks.priming")


def classify_message_type(message: str, channel: str, *, intent: str = "") -> str:
    """Classify message type for budget adjustment.

    Args:
        message: Message text
        channel: Message channel
        intent: Sender-declared intent (preferred when provided)

    Returns:
        Message type: "greeting", "question", "request", "heartbeat"
    """
    if channel == "heartbeat":
        return "heartbeat"

    intent_map = {
        "delegation": "request",
        "report": "question",
        "question": "question",
    }
    intent_norm = str(intent or "").strip().lower()
    mapped = intent_map.get(intent_norm)
    if mapped:
        return mapped

    message_lower = message.lower()

    greeting_patterns = [
        "こんにちは",
        "おはよう",
        "こんばんは",
        "よろしく",
        "hello",
        "hi",
        "hey",
        "good morning",
        "good evening",
    ]
    if any(p in message_lower for p in greeting_patterns) and len(message) < 50:
        return "greeting"

    question_patterns = [
        "?",
        "？",
        "教えて",
        "どう",
        "なぜ",
        "いつ",
        "どこ",
        "誰",
        "what",
        "why",
        "when",
        "where",
        "who",
        "how",
        "can you",
    ]
    if any(p in message_lower for p in question_patterns):
        return "question"

    if len(message) > 100:
        return "request"

    return "question"


def load_config_budgets() -> tuple[int, int, int, int, float]:
    """Load budget values from config.json (lazy, once per call).

    Returns:
        (budget_greeting, budget_question, budget_request, budget_heartbeat, heartbeat_context_pct)
    """
    try:
        from core.config.models import load_config

        config = load_config()
        p = config.priming
        return (
            p.budget_greeting,
            p.budget_question,
            p.budget_request,
            p.budget_heartbeat,
            p.heartbeat_context_pct,
        )
    except Exception:
        logger.debug("Failed to load priming config; using defaults")
        return (
            _BUDGET_GREETING,
            _BUDGET_QUESTION,
            _BUDGET_REQUEST,
            _BUDGET_HEARTBEAT,
            0.05,
        )


def adjust_token_budget(
    message: str,
    channel: str,
    context_window: int,
    *,
    intent: str = "",
    budget_greeting: int = _BUDGET_GREETING,
    budget_question: int = _BUDGET_QUESTION,
    budget_request: int = _BUDGET_REQUEST,
    budget_heartbeat: int = _BUDGET_HEARTBEAT,
    heartbeat_context_pct: float = 0.05,
) -> int:
    """Adjust token budget based on message type.

    For heartbeat, the budget is ``max(config.budget_heartbeat,
    int(context_window * config.heartbeat_context_pct))`` so that
    models with large context windows get proportionally more priming.

    Args:
        message: Message text
        channel: Message channel
        context_window: Model context window size
        intent: Sender-declared intent (preferred when provided)
        budget_*: Budget values (default from constants; engine passes config-loaded)

    Returns:
        Adjusted token budget
    """
    msg_type = classify_message_type(message, channel, intent=intent)

    if msg_type == "heartbeat":
        base = budget_heartbeat
        if context_window > 0:
            pct_budget = int(context_window * heartbeat_context_pct)
            budget = max(base, pct_budget)
        else:
            budget = base
    else:
        budget_map = {
            "greeting": budget_greeting,
            "question": budget_question,
            "request": budget_request,
        }
        budget = budget_map.get(msg_type, _DEFAULT_MAX_PRIMING_TOKENS)

    logger.debug("Message type: %s -> budget: %d", msg_type, budget)
    return budget
