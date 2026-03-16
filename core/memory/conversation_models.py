# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Data classes and constants for conversation memory."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.time_utils import now_iso

_RE_INVALID_TOOL_ID = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitize_tool_id(tool_id: str) -> str:
    """Sanitize tool_use_id for Bedrock Converse API compatibility.

    Bedrock requires tool_use_id to match ``[a-zA-Z0-9_-]+``.
    Other providers (OpenAI/Kimi) may produce IDs containing dots,
    colons, etc.  Replace any invalid character with ``_``.
    """
    return _RE_INVALID_TOOL_ID.sub("_", tool_id) if tool_id else tool_id


# ── Error detection patterns for auto-tracking ───────────────

_ERROR_PATTERN = re.compile(
    r"\b(error|failed)\b|エラー[がはをの]|失敗し[たてま]",
    re.IGNORECASE,
)
_RESOLVED_PATTERN = re.compile(
    r"(fixed|resolved|解決|修正済み|成功)",
    re.IGNORECASE,
)

# Truncate assistant responses to this length in the history display.
_MAX_RESPONSE_CHARS_IN_HISTORY = 2500

# Truncate human messages to this length in the history display.
_MAX_HUMAN_CHARS_IN_HISTORY = 800

# Hard cap on content stored in conversation.json per turn.
_MAX_STORED_CONTENT_CHARS = 5000

# Rough characters-per-token for estimation (conservative for Japanese).
_CHARS_PER_TOKEN = 4

# Maximum number of turns to include in the chat prompt.
_MAX_DISPLAY_TURNS = 15

# Trigger compression when stored turns exceed this count.
_MAX_TURNS_BEFORE_COMPRESS = 50

# ── Tool record limits ─────────────────────────────────────
_MAX_TOOL_INPUT_SUMMARY = 500
_MAX_TOOL_RESULT_SUMMARY = 2000
_MAX_TOOL_RECORDS_PER_TURN = 10
_MAX_RENDERED_TOOL_RECORDS = 50

SESSION_GAP_MINUTES = 10


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ToolRecord:
    """A single tool call + result pair stored in conversation memory."""

    tool_name: str
    tool_id: str = ""
    input_summary: str = ""  # max _MAX_TOOL_INPUT_SUMMARY chars
    result_summary: str = ""  # max _MAX_TOOL_RESULT_SUMMARY chars
    is_error: bool = False

    def __post_init__(self) -> None:
        if len(self.input_summary) > _MAX_TOOL_INPUT_SUMMARY:
            self.input_summary = self.input_summary[:_MAX_TOOL_INPUT_SUMMARY] + "..."
        if len(self.result_summary) > _MAX_TOOL_RESULT_SUMMARY:
            self.result_summary = self.result_summary[:_MAX_TOOL_RESULT_SUMMARY] + "..."

    @classmethod
    def from_dict(cls, d: dict) -> ToolRecord:
        """Create a ToolRecord from a dict (e.g., from CycleResult)."""
        return cls(
            tool_name=d.get("tool_name", ""),
            tool_id=d.get("tool_id", ""),
            input_summary=d.get("input_summary", ""),
            result_summary=d.get("result_summary", ""),
            is_error=d.get("is_error", False),
        )


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    role: str  # "human" or "assistant"
    content: str
    timestamp: str = ""
    token_estimate: int = 0
    attachments: list[str] = field(default_factory=list)
    tool_records: list[ToolRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = now_iso()
        if not self.token_estimate:
            self.token_estimate = len(self.content) // _CHARS_PER_TOKEN


@dataclass
class ConversationState:
    """Full conversation state including compressed summary."""

    anima_name: str = ""
    turns: list[ConversationTurn] = field(default_factory=list)
    compressed_summary: str = ""
    compressed_turn_count: int = 0
    last_finalized_turn_index: int = 0

    @property
    def total_token_estimate(self) -> int:
        summary_tokens = len(self.compressed_summary) // _CHARS_PER_TOKEN
        turn_tokens = sum(t.token_estimate for t in self.turns)
        return summary_tokens + turn_tokens

    @property
    def total_turn_count(self) -> int:
        return len(self.turns) + self.compressed_turn_count


@dataclass
class ParsedSessionSummary:
    """Parsed result of LLM session summary with state changes."""

    title: str
    episode_body: str
    resolved_items: list[str]
    new_tasks: list[str]
    current_status: str
    has_state_changes: bool
