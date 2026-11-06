"""E2E tests for messaging data model unification.

Tests the full pipeline: messenger.send() → activity log recording →
alias-based retrieval → cascade limiter depth counting.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.activity.logger import ActivityLogger
from core.messaging.messenger import Messenger


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Create a full data directory structure."""
    animas = tmp_path / "animas"
    shared = tmp_path / "shared"
    for name in ("alice", "bob"):
        (animas / name / "activity_log").mkdir(parents=True)
    (shared / "inbox" / "alice").mkdir(parents=True)
    (shared / "inbox" / "bob").mkdir(parents=True)
    return tmp_path


class TestSendToActivityLogPipeline:
    """End-to-end: send → activity_log → alias retrieval."""

    def test_send_and_retrieve_as_message_sent(self, data_dir: Path) -> None:
        shared = data_dir / "shared"
        m = Messenger(shared, "alice")
        m.send("bob", "Hello Bob!")

        activity = ActivityLogger(data_dir / "animas" / "alice")
        entries = activity.recent(days=1, types=["message_sent"])
        assert len(entries) == 1
        e = entries[0]
        assert e.type == "message_sent"
        assert e.to_person == "bob"
        assert e.meta.get("from_type") == "anima"

    def test_read_dm_history_uses_new_types(self, data_dir: Path) -> None:
        shared = data_dir / "shared"
        m = Messenger(shared, "alice")
        m.send("bob", "Hello Bob!")

        history = m.read_dm_history("bob", limit=10)
        assert len(history) >= 1
        assert any("Hello Bob!" in h.get("text", "") for h in history)


class TestLegacyLogCompatibility:
    """E2E: legacy dm_sent/dm_received logs are still retrievable."""

    def test_legacy_entries_found_via_new_type_filter(self, data_dir: Path) -> None:
        from core.time_utils import now_iso, now_jst

        anima_dir = data_dir / "animas" / "alice"
        today = now_jst().date().isoformat()
        log_path = anima_dir / "activity_log" / f"{today}.jsonl"

        legacy_entries = [
            {"ts": now_iso(), "type": "dm_sent", "content": "legacy out", "to": "bob"},
            {"ts": now_iso(), "type": "dm_received", "content": "legacy in", "from": "bob"},
        ]
        with log_path.open("a", encoding="utf-8") as f:
            for entry in legacy_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        activity = ActivityLogger(anima_dir)

        sent = activity.recent(days=1, types=["message_sent"])
        assert len(sent) == 1
        assert sent[0].type == "dm_sent"
        assert sent[0].content == "legacy out"

        received = activity.recent(days=1, types=["message_received"])
        assert len(received) == 1
        assert received[0].type == "dm_received"
        assert received[0].content == "legacy in"


class TestFormatForPrimingIntegration:
    """E2E: format_for_priming renders unified labels for mixed logs."""

    def test_mixed_entries_render_with_unified_labels(self, data_dir: Path) -> None:
        from core.time_utils import now_iso, now_jst

        anima_dir = data_dir / "animas" / "alice"
        today = now_jst().date().isoformat()
        log_path = anima_dir / "activity_log" / f"{today}.jsonl"

        entries_raw = [
            {"ts": now_iso(), "type": "dm_sent", "content": "old msg", "to": "bob"},
            {
                "ts": now_iso(),
                "type": "message_sent",
                "content": "new msg",
                "to": "bob",
                "meta": {"from_type": "anima"},
            },
            {"ts": now_iso(), "type": "response_sent", "content": "reply to human", "to": "human"},
        ]
        with log_path.open("a", encoding="utf-8") as f:
            for entry in entries_raw:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        activity = ActivityLogger(anima_dir)
        all_entries = activity.recent(days=1, limit=100)
        formatted = activity.format_for_priming(all_entries, budget_tokens=5000)

        assert "MSG>" in formatted
        assert "RESP>" in formatted
