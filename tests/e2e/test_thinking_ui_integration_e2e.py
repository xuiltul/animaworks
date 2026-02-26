# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for thinking UI integration — thinking text persistence and retrieval."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.memory.activity import ActivityLogger
from core.schemas import CycleResult


# ── Fixtures ─────────────────────────────────────────────────────

JST = timezone(timedelta(hours=9))


def _ts(minutes_ago: int = 0) -> str:
    """Generate ISO timestamp relative to now."""
    t = datetime.now(JST) - timedelta(minutes=minutes_ago)
    return t.isoformat()


def _write_entry(anima_dir: Path, entry: dict) -> None:
    """Write a raw JSONL entry to the activity log."""
    log_dir = anima_dir / "activity_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    date_str = entry["ts"][:10]
    path = log_dir / f"{date_str}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Test: thinking_text persisted in activity log ─────────────────


def test_thinking_text_persisted_in_activity_log(data_dir, make_anima) -> None:
    """ActivityLogger.log() with meta.thinking_text persists and is read back."""
    anima_dir = make_anima("thinking-test")

    al = ActivityLogger(anima_dir)
    al.log(
        "response_sent",
        content="Hi, how can I help?",
        to_person="user",
        channel="chat",
        meta={"thinking_text": "I thought about the best way to respond..."},
    )

    entries = al.recent(days=1)
    assert len(entries) >= 1
    response_entries = [e for e in entries if e.type == "response_sent"]
    assert len(response_entries) >= 1
    assert response_entries[-1].meta.get("thinking_text") == (
        "I thought about the best way to respond..."
    )


# ── Test: thinking_text in conversation messages ──────────────────


def test_thinking_text_in_conversation_messages(data_dir, make_anima) -> None:
    """Manually written response_sent with meta.thinking_text surfaces in get_conversation_view."""
    anima_dir = make_anima("conv-thinking-test")

    _write_entry(anima_dir, {
        "ts": _ts(5),
        "type": "message_received",
        "content": "こんにちは",
        "from": "user",
        "channel": "chat",
    })
    _write_entry(anima_dir, {
        "ts": _ts(4),
        "type": "response_sent",
        "content": "こんにちは！元気ですか？",
        "to": "user",
        "channel": "chat",
        "meta": {"thinking_text": "ユーザーが挨拶してきたので、親しみやすく返事しよう。"},
    })

    al = ActivityLogger(anima_dir)
    result = al.get_conversation_view(limit=50)

    assert "sessions" in result
    assert len(result["sessions"]) >= 1
    session = result["sessions"][0]
    assert len(session["messages"]) >= 2

    assistant_msg = next(m for m in session["messages"] if m["role"] == "assistant")
    assert assistant_msg.get("thinking_text") == (
        "ユーザーが挨拶してきたので、親しみやすく返事しよう。"
    )


# ── Test: backward compat without thinking_text ───────────────────


def test_backward_compat_no_thinking(data_dir, make_anima) -> None:
    """Entries without thinking_text do not expose thinking_text key in messages."""
    anima_dir = make_anima("no-thinking-test")

    _write_entry(anima_dir, {
        "ts": _ts(5),
        "type": "message_received",
        "content": "Hello",
        "from": "user",
        "channel": "chat",
    })
    _write_entry(anima_dir, {
        "ts": _ts(4),
        "type": "response_sent",
        "content": "Hello there!",
        "to": "user",
        "channel": "chat",
        # No meta.thinking_text — legacy entries
    })

    al = ActivityLogger(anima_dir)
    result = al.get_conversation_view(limit=50)

    assistant_msg = next(m for m in result["sessions"][0]["messages"] if m["role"] == "assistant")
    assert "thinking_text" not in assistant_msg


# ── Test: CycleResult with long thinking_text ─────────────────────


def test_cycle_result_thinking_text_truncation() -> None:
    """CycleResult schema accepts long thinking_text (>10000 chars).

    Truncation to 10000 chars happens in agent.py; this verifies
    the schema and model_dump chain work with long strings.
    """
    long_text = "x" * 15000
    result = CycleResult(
        trigger="chat",
        action="responded",
        summary="Done",
        thinking_text=long_text,
    )

    assert len(result.thinking_text) == 15000

    # model_dump must serialize correctly (no truncation at schema level)
    dumped = result.model_dump(mode="json")
    assert "thinking_text" in dumped
    assert len(dumped["thinking_text"]) == 15000
