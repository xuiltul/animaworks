from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def shared_dir(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    return shared


@pytest.fixture
def anima_dir(tmp_path):
    d = tmp_path / "animas" / "testanima"
    d.mkdir(parents=True)
    (d / "activity_log").mkdir()
    return d


@pytest.fixture
def messenger(tmp_path, shared_dir, anima_dir):
    from core.messaging.messenger import Messenger

    m = Messenger.__new__(Messenger)
    m.anima_name = "testanima"
    m.shared_dir = shared_dir
    return m


def _write_activity(anima_dir: Path, entries: list[dict]) -> None:
    """Write activity log entries for today."""
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    log_file = anima_dir / "activity_log" / f"{today}.jsonl"
    with log_file.open("w") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def _make_entry(
    type_: str,
    to: str = "peer",
    content: str = "hello",
    hours_ago: float = 0,
    from_person: str | None = None,
) -> dict:
    ts = (datetime.now(tz=UTC) - timedelta(hours=hours_ago)).isoformat()
    entry: dict = {
        "ts": ts,
        "type": type_,
        "content": content,
        "summary": f"→ {to}: {content[:40]}",
        "to": to,
        "from": from_person or "testanima",
        "meta": {"from_type": "anima"},
    }
    return entry


# ── Tests: read_dm_history direction filter ────────────────


class TestReadDmHistoryDirection:
    def test_direction_sent_only(self, messenger, anima_dir):
        entries = [
            _make_entry("message_sent", to="peer", content="outgoing"),
            _make_entry("message_received", to="peer", content="incoming", from_person="peer"),
        ]
        _write_activity(anima_dir, entries)

        result = messenger.read_dm_history("peer", direction="sent")

        assert len(result) == 1
        assert result[0]["text"] == "outgoing"

    def test_direction_received_only(self, messenger, anima_dir):
        entries = [
            _make_entry("message_sent", to="peer", content="outgoing"),
            _make_entry("message_received", to="peer", content="incoming", from_person="peer"),
        ]
        _write_activity(anima_dir, entries)

        result = messenger.read_dm_history("peer", direction="received")

        assert len(result) == 1
        assert result[0]["text"] == "incoming"

    def test_direction_both(self, messenger, anima_dir):
        entries = [
            _make_entry("message_sent", to="peer", content="outgoing"),
            _make_entry("message_received", to="peer", content="incoming", from_person="peer"),
        ]
        _write_activity(anima_dir, entries)

        result = messenger.read_dm_history("peer", direction="both")

        assert len(result) == 2

    def test_direction_invalid_falls_back_to_both(self, messenger, anima_dir):
        entries = [
            _make_entry("message_sent", to="peer", content="outgoing"),
            _make_entry("message_received", to="peer", content="incoming", from_person="peer"),
        ]
        _write_activity(anima_dir, entries)

        result = messenger.read_dm_history("peer", direction="invalid_value")

        assert len(result) == 2


# ── Tests: read_dm_history hours filter ────────────────────


class TestReadDmHistoryHours:
    def test_hours_filter(self, messenger, anima_dir):
        entries = [
            _make_entry("message_sent", to="peer", content="recent", hours_ago=1),
            _make_entry("message_sent", to="peer", content="old", hours_ago=48),
        ]
        _write_activity(anima_dir, entries)

        result = messenger.read_dm_history("peer", hours=24)

        assert len(result) == 1
        assert result[0]["text"] == "recent"

    def test_hours_none_returns_all(self, messenger, anima_dir):
        entries = [
            _make_entry("message_sent", to="peer", content="recent", hours_ago=1),
            _make_entry("message_sent", to="peer", content="old", hours_ago=2),
        ]
        _write_activity(anima_dir, entries)

        result = messenger.read_dm_history("peer", hours=None)

        assert len(result) == 2


# ── Tests: read_dm_history keyword filter ──────────────────


class TestReadDmHistoryKeyword:
    def test_keyword_filter(self, messenger, anima_dir):
        entries = [
            _make_entry("message_sent", to="peer", content="PR #469 承認申請"),
            _make_entry("message_sent", to="peer", content="Movacal障害報告"),
        ]
        _write_activity(anima_dir, entries)

        result = messenger.read_dm_history("peer", keyword="承認申請")

        assert len(result) == 1
        assert "承認申請" in result[0]["text"]

    def test_keyword_none_returns_all(self, messenger, anima_dir):
        entries = [
            _make_entry("message_sent", to="peer", content="PR #469 承認申請"),
            _make_entry("message_sent", to="peer", content="Movacal障害報告"),
        ]
        _write_activity(anima_dir, entries)

        result = messenger.read_dm_history("peer", keyword=None)

        assert len(result) == 2

    def test_keyword_no_match_returns_empty(self, messenger, anima_dir):
        entries = [
            _make_entry("message_sent", to="peer", content="hello world"),
        ]
        _write_activity(anima_dir, entries)

        result = messenger.read_dm_history("peer", keyword="nonexistent")

        assert len(result) == 0


# ── Tests: combined filters ────────────────────────────────


class TestReadDmHistoryCombined:
    def test_direction_and_keyword(self, messenger, anima_dir):
        entries = [
            _make_entry("message_sent", to="peer", content="PR #469 承認申請"),
            _make_entry("message_sent", to="peer", content="Movacal障害報告"),
            _make_entry("message_received", to="peer", content="PR #469 了解", from_person="peer"),
        ]
        _write_activity(anima_dir, entries)

        result = messenger.read_dm_history("peer", direction="sent", keyword="PR #469")

        assert len(result) == 1
        assert result[0]["text"] == "PR #469 承認申請"


# ── Tests: send_message feedback ───────────────────────────


class TestSendFeedback:
    def test_build_send_feedback_with_history(self, anima_dir):
        from core.tooling.handler_comms import CommsToolsMixin

        entries = [
            _make_entry("message_sent", to="sakura", content="report 1", hours_ago=2),
            _make_entry("message_sent", to="sakura", content="report 2", hours_ago=1),
            _make_entry("message_sent", to="sakura", content="report 3", hours_ago=0.5),
        ]
        _write_activity(anima_dir, entries)

        mixin = CommsToolsMixin.__new__(CommsToolsMixin)
        mixin._anima_dir = anima_dir

        result = mixin._build_send_feedback("sakura")

        assert "3" in result
        assert "sakura" in result

    def test_build_send_feedback_no_history(self, anima_dir):
        from core.tooling.handler_comms import CommsToolsMixin

        _write_activity(anima_dir, [])

        mixin = CommsToolsMixin.__new__(CommsToolsMixin)
        mixin._anima_dir = anima_dir

        result = mixin._build_send_feedback("sakura")

        assert result == ""

    def test_build_send_feedback_exception_returns_empty(self):
        from core.tooling.handler_comms import CommsToolsMixin

        mixin = CommsToolsMixin.__new__(CommsToolsMixin)
        mixin._anima_dir = Path("/nonexistent/path")

        result = mixin._build_send_feedback("sakura")

        assert result == ""
