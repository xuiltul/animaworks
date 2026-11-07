"""Tests for the Board/DM call-flow builder."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from core.messaging.flows import build_flows, clear_flows_cache
from core.time_utils import now_local


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_flows_cache()
    yield
    clear_flows_cache()


def _make_anima(root: Path, name: str) -> Path:
    anima_dir = root / "animas" / name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "status.json").write_text(json.dumps({"enabled": True}), encoding="utf-8")
    return anima_dir


def _write_board(shared: Path, channel: str, rows: list[dict]) -> None:
    channels_dir = shared / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    (channels_dir / f"{channel}.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_delivery(shared: Path, channel: str, record: dict) -> None:
    deliveries = shared / "board_deliveries"
    deliveries.mkdir(parents=True, exist_ok=True)
    path = deliveries / f"{channel}.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_activity(root: Path, name: str, entries: list[dict]) -> None:
    log_dir = root / "animas" / name / "activity_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        day = datetime.fromisoformat(entry["ts"]).date().isoformat()
        with (log_dir / f"{day}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _ts(offset: timedelta) -> str:
    return (now_local() + offset).isoformat()


def _call_by_text(result: dict, text: str) -> dict:
    matches = [call for call in result["calls"] if call["text"] == text]
    assert len(matches) == 1, result["calls"]
    return matches[0]


def test_board_delivery_read_and_recorded_reply(tmp_path: Path):
    shared = tmp_path / "shared"
    for name in ("alice", "bob", "carol"):
        _make_anima(tmp_path, name)

    start = _ts(-timedelta(hours=3))
    received_at = _ts(-timedelta(hours=2, minutes=55))
    reply_at = _ts(-timedelta(hours=2, minutes=50))
    _write_board(
        shared,
        "general",
        [
            {"ts": start, "from": "alice", "text": "@bob hello @carol", "source": "anima", "id": "p1", "kind": "anima"},
            {
                "ts": reply_at,
                "from": "bob",
                "text": "pong",
                "source": "anima",
                "id": "p2",
                "kind": "anima",
                "reply_to": "p1",
            },
        ],
    )
    _write_delivery(
        shared, "general", {"post_id": "p1", "ts": start, "notified": ["bob"], "not_notified": {"carol": "disabled"}}
    )
    _write_activity(
        tmp_path,
        "bob",
        [
            {
                "ts": received_at,
                "type": "message_received",
                "from": "alice",
                "to": "bob",
                "content": "hello",
                "meta": {"message_id": "inbox-1", "board_channel": "general", "board_post_id": "p1"},
            }
        ],
    )

    result = build_flows(shared, days=7, via="all")
    to_bob = next(call for call in result["calls"] if call["to"] == "bob")
    to_carol = next(call for call in result["calls"] if call["to"] == "carol")

    assert to_bob["via"] == "board"
    assert to_bob["place"] == "general"
    assert to_bob["from"] == "alice"
    assert to_bob["delivered"] is True
    assert to_bob["reason"] == ""
    assert to_bob["read"] == received_at
    assert to_bob["reply"] == reply_at
    assert to_bob["reply_id"] == "p2"
    assert to_bob["reply_text"] == "pong"
    assert to_bob["inferred"] is False
    assert to_bob["text"] == "@bob hello @carol"

    assert to_carol["delivered"] is False
    assert to_carol["reason"] == "disabled"
    assert to_carol["read"] is None
    assert to_carol["reply"] is None


def test_legacy_board_call_and_inferred_reply_include_human_posts(tmp_path: Path):
    shared = tmp_path / "shared"
    for name in ("alice", "bob", "carol"):
        _make_anima(tmp_path, name)

    start = _ts(-timedelta(hours=4))
    next_at = _ts(-timedelta(hours=3))
    _write_board(
        shared,
        "general",
        [
            {"ts": start, "from": "alice", "text": "@bob legacy", "source": "anima"},
            {"ts": next_at, "from": "bob", "text": "follow-up", "source": "anima", "id": "p11", "kind": "anima"},
            {"ts": start, "from": "owner", "text": "@carol please", "source": "human", "id": "p20", "kind": "human"},
        ],
    )
    _write_delivery(shared, "general", {"post_id": "p20", "ts": start, "notified": ["carol"], "not_notified": {}})

    result = build_flows(shared, days=7, via="all")
    legacy = next(call for call in result["calls"] if call["text"] == "@bob legacy")
    human = next(call for call in result["calls"] if call["text"] == "@carol please")

    assert legacy["from"] == "alice"
    assert legacy["to"] == "bob"
    assert legacy["delivered"] is True
    assert legacy["inferred"] is True
    assert legacy["reply"] == next_at
    assert legacy["reply_id"] == "p11"
    assert legacy["reply_text"] == "follow-up"

    assert human["from"] == "owner"
    assert human["to"] == "carol"
    assert human["delivered"] is True
    assert human["inferred"] is False


def test_board_reply_via_dm_trigger_id(tmp_path: Path):
    shared = tmp_path / "shared"
    for name in ("alice", "bob"):
        _make_anima(tmp_path, name)

    start = _ts(-timedelta(hours=3))
    received_at = _ts(-timedelta(hours=2, minutes=55))
    reply_at = _ts(-timedelta(hours=2, minutes=50))
    _write_board(
        shared,
        "general",
        [{"ts": start, "from": "alice", "text": "@bob call", "source": "anima", "id": "p1", "kind": "anima"}],
    )
    _write_delivery(shared, "general", {"post_id": "p1", "ts": start, "notified": ["bob"], "not_notified": {}})
    _write_activity(
        tmp_path,
        "bob",
        [
            {
                "ts": received_at,
                "type": "message_received",
                "from": "alice",
                "to": "bob",
                "content": "call",
                "meta": {"message_id": "inbox-1", "board_channel": "general", "board_post_id": "p1"},
            },
            {
                "ts": reply_at,
                "type": "message_sent",
                "from": "bob",
                "to": "alice",
                "content": "dm answer",
                "meta": {"message_id": "m10", "trigger_id": "inbox-1"},
            },
        ],
    )

    result = build_flows(shared, days=7, via="all")
    board_call = next(call for call in result["calls"] if call["via"] == "board")
    dm_call = next(call for call in result["calls"] if call["via"] == "dm")

    assert board_call["reply"] == reply_at
    assert board_call["reply_id"] == "m10"
    assert board_call["reply_text"] == "dm answer"
    assert board_call["inferred"] is False
    assert dm_call["parent"] == board_call["id"]


def test_parent_via_message_trigger_of_board_mention(tmp_path: Path):
    shared = tmp_path / "shared"
    for name in ("alice", "bob"):
        _make_anima(tmp_path, name)

    start = _ts(-timedelta(hours=3))
    _write_board(
        shared,
        "general",
        [
            {"ts": start, "from": "alice", "text": "@bob call", "source": "anima", "id": "p1", "kind": "anima"},
            {
                "ts": _ts(-timedelta(hours=2)),
                "from": "bob",
                "text": "@alice answer",
                "source": "anima",
                "id": "p2",
                "kind": "anima",
                "trigger": {"type": "message", "id": "inbox-1"},
            },
        ],
    )
    _write_delivery(shared, "general", {"post_id": "p1", "ts": start, "notified": ["bob"], "not_notified": {}})
    _write_activity(
        tmp_path,
        "bob",
        [
            {
                "ts": _ts(-timedelta(hours=2, minutes=55)),
                "type": "message_received",
                "from": "alice",
                "to": "bob",
                "content": "call",
                "meta": {"message_id": "inbox-1", "board_channel": "general", "board_post_id": "p1"},
            }
        ],
    )

    result = build_flows(shared, days=7, via="board")
    root = next(call for call in result["calls"] if call["id"] == "board:p1:bob")
    answer = next(call for call in result["calls"] if call["id"] == "board:p2:alice")
    assert answer["parent"] == root["id"]


def test_dm_read_and_reply_join_on_message_ids(tmp_path: Path):
    shared = tmp_path / "shared"
    for name in ("alice", "bob"):
        _make_anima(tmp_path, name)

    first = _ts(-timedelta(hours=5))
    read_at = _ts(-timedelta(hours=4, minutes=55))
    reply_at = _ts(-timedelta(hours=4, minutes=40))
    second = _ts(-timedelta(hours=4))
    inferred_at = _ts(-timedelta(hours=3))
    _write_board(shared, "general", [])
    _write_activity(
        tmp_path,
        "alice",
        [
            {
                "ts": first,
                "type": "message_sent",
                "from": "alice",
                "to": "bob",
                "content": "hello",
                "meta": {"message_id": "m1"},
            },
            {
                "ts": second,
                "type": "message_sent",
                "from": "alice",
                "to": "bob",
                "content": "second",
                "meta": {"message_id": "m3"},
            },
        ],
    )
    _write_activity(
        tmp_path,
        "bob",
        [
            {
                "ts": read_at,
                "type": "message_received",
                "from": "alice",
                "to": "bob",
                "content": "hello",
                "meta": {"message_id": "m1"},
            },
            {
                "ts": reply_at,
                "type": "message_sent",
                "from": "bob",
                "to": "alice",
                "content": "hi there",
                "meta": {"message_id": "m2", "reply_to": "m1"},
            },
            {
                "ts": inferred_at,
                "type": "message_sent",
                "from": "bob",
                "to": "alice",
                "content": "inferred",
                "meta": {"message_id": "m4"},
            },
        ],
    )

    result = build_flows(shared, days=7, via="dm")
    recorded = _call_by_text(result, "hello")
    inferred = _call_by_text(result, "second")

    assert recorded["read"] == read_at
    assert recorded["reply"] == reply_at
    assert recorded["reply_id"] == "m2"
    assert recorded["reply_text"] == "hi there"
    assert recorded["inferred"] is False

    assert inferred["reply"] == inferred_at
    assert inferred["reply_id"] == "m4"
    assert inferred["inferred"] is True


def test_parents_join_board_and_dm_triggers(tmp_path: Path):
    shared = tmp_path / "shared"
    for name in ("alice", "bob"):
        _make_anima(tmp_path, name)

    start = _ts(-timedelta(hours=5))
    received_at = _ts(-timedelta(hours=4, minutes=55))
    _write_board(
        shared,
        "general",
        [
            {"ts": start, "from": "alice", "text": "@bob call", "source": "anima", "id": "p1", "kind": "anima"},
            {
                "ts": _ts(-timedelta(hours=4, minutes=40)),
                "from": "bob",
                "text": "@alice answer",
                "source": "anima",
                "id": "p2",
                "kind": "anima",
                "reply_to": "p1",
                "trigger": {"type": "board_mention", "id": "p1"},
            },
            {
                "ts": _ts(-timedelta(hours=4, minutes=20)),
                "from": "alice",
                "text": "@bob from dm",
                "source": "anima",
                "id": "p3",
                "kind": "anima",
                "trigger": {"type": "message", "id": "m20"},
            },
        ],
    )
    _write_delivery(shared, "general", {"post_id": "p1", "ts": start, "notified": ["bob"], "not_notified": {}})
    _write_activity(
        tmp_path,
        "bob",
        [
            {
                "ts": received_at,
                "type": "message_received",
                "from": "alice",
                "to": "bob",
                "content": "call",
                "meta": {"message_id": "inbox-1", "board_channel": "general", "board_post_id": "p1"},
            },
            {
                "ts": _ts(-timedelta(hours=4, minutes=30)),
                "type": "message_sent",
                "from": "bob",
                "to": "alice",
                "content": "dm reply",
                "meta": {"message_id": "m10", "trigger_id": "inbox-1"},
            },
        ],
    )
    _write_activity(
        tmp_path,
        "alice",
        [
            {
                "ts": _ts(-timedelta(hours=4, minutes=30)),
                "type": "message_sent",
                "from": "alice",
                "to": "bob",
                "content": "orig",
                "meta": {"message_id": "m20"},
            }
        ],
    )

    result = build_flows(shared, days=7, via="all")
    root = next(call for call in result["calls"] if call["id"] == "board:p1:bob")
    board_reply = next(call for call in result["calls"] if call["id"] == "board:p2:alice")
    dm_reply = next(call for call in result["calls"] if call["id"] == "dm:m10")
    dm_root = next(call for call in result["calls"] if call["id"] == "dm:m20")
    board_from_dm = next(call for call in result["calls"] if call["id"] == "board:p3:bob")

    assert board_reply["parent"] == root["id"]
    assert dm_reply["parent"] == root["id"]
    assert dm_root["parent"] is None
    assert board_from_dm["parent"] == dm_root["id"]


def test_period_and_via_filters(tmp_path: Path):
    shared = tmp_path / "shared"
    for name in ("alice", "bob"):
        _make_anima(tmp_path, name)

    old = _ts(-timedelta(days=10))
    _write_board(
        shared,
        "general",
        [{"ts": old, "from": "alice", "text": "@bob old", "source": "anima", "id": "p-old", "kind": "anima"}],
    )
    _write_activity(
        tmp_path,
        "alice",
        [
            {
                "ts": old,
                "type": "message_sent",
                "from": "alice",
                "to": "bob",
                "content": "old dm",
                "meta": {"message_id": "m-old"},
            }
        ],
    )

    assert build_flows(shared, days=7, via="all")["calls"] == []

    all_calls = build_flows(shared, days=30, via="all")["calls"]
    board_calls = build_flows(shared, days=30, via="board")["calls"]
    dm_calls = build_flows(shared, days=30, via="dm")["calls"]
    assert {call["via"] for call in all_calls} == {"board", "dm"}
    assert {call["via"] for call in board_calls} == {"board"}
    assert {call["via"] for call in dm_calls} == {"dm"}


def test_unreadable_board_is_excluded(tmp_path: Path):
    shared = tmp_path / "shared"
    for name in ("alice", "bob"):
        _make_anima(tmp_path, name)

    start = _ts(-timedelta(hours=1))
    _write_board(
        shared,
        "general",
        [{"ts": start, "from": "alice", "text": "@bob hi", "source": "anima", "id": "p1", "kind": "anima"}],
    )
    _write_board(
        shared,
        "hidden",
        [{"ts": start, "from": "alice", "text": "@bob secret", "source": "anima", "id": "p2", "kind": "anima"}],
    )
    (shared / "channels" / "hidden.meta.json").write_text("not json", encoding="utf-8")
    _write_delivery(shared, "general", {"post_id": "p1", "ts": start, "notified": ["bob"], "not_notified": {}})
    _write_delivery(shared, "hidden", {"post_id": "p2", "ts": start, "notified": ["bob"], "not_notified": {}})

    result = build_flows(shared, days=7, via="all")
    assert [call["place"] for call in result["calls"]] == ["general"]


def test_reply_without_reply_to_is_inferred(tmp_path: Path):
    shared = tmp_path / "shared"
    for name in ("alice", "bob"):
        _make_anima(tmp_path, name)

    start = _ts(-timedelta(hours=3))
    reply_at = _ts(-timedelta(hours=2))
    _write_board(
        shared,
        "general",
        [
            {"ts": start, "from": "alice", "text": "@bob call", "source": "anima", "id": "p1", "kind": "anima"},
            {"ts": reply_at, "from": "bob", "text": "answer", "source": "anima", "id": "p2", "kind": "anima"},
        ],
    )
    _write_delivery(shared, "general", {"post_id": "p1", "ts": start, "notified": ["bob"], "not_notified": {}})

    result = build_flows(shared, days=7, via="board")
    call = next(call for call in result["calls"] if call["to"] == "bob")
    assert call["reply"] == reply_at
    assert call["reply_id"] == "p2"
    assert call["inferred"] is True


def test_cache_is_clearable(tmp_path: Path):
    shared = tmp_path / "shared"
    for name in ("alice", "bob"):
        _make_anima(tmp_path, name)

    start = _ts(-timedelta(hours=2))
    _write_board(
        shared,
        "general",
        [{"ts": start, "from": "alice", "text": "@bob first", "source": "anima", "id": "p1", "kind": "anima"}],
    )
    _write_delivery(shared, "general", {"post_id": "p1", "ts": start, "notified": ["bob"], "not_notified": {}})

    assert len(build_flows(shared, days=7, via="all")["calls"]) == 1

    second_at = _ts(-timedelta(hours=1))
    _write_board(
        shared,
        "general",
        [
            {"ts": start, "from": "alice", "text": "@bob first", "source": "anima", "id": "p1", "kind": "anima"},
            {"ts": second_at, "from": "alice", "text": "@bob second", "source": "anima", "id": "p2", "kind": "anima"},
        ],
    )
    _write_delivery(shared, "general", {"post_id": "p2", "ts": second_at, "notified": ["bob"], "not_notified": {}})

    assert len(build_flows(shared, days=7, via="all")["calls"]) == 1
    clear_flows_cache()
    assert len(build_flows(shared, days=7, via="all")["calls"]) == 2


def test_summary_counts_and_median(tmp_path: Path):
    shared = tmp_path / "shared"
    for name in ("alice", "bob", "carol"):
        _make_anima(tmp_path, name)

    _write_board(
        shared,
        "general",
        [
            {
                "ts": _ts(-timedelta(hours=1)),
                "from": "alice",
                "text": "@carol miss",
                "source": "anima",
                "id": "p1",
                "kind": "anima",
            }
        ],
    )
    _write_delivery(
        shared,
        "general",
        {"post_id": "p1", "ts": _ts(-timedelta(hours=1)), "notified": [], "not_notified": {"carol": "disabled"}},
    )

    first = now_local() - timedelta(hours=5)
    second = now_local() - timedelta(hours=4)
    waiting = now_local() - timedelta(hours=3)
    _write_activity(
        tmp_path,
        "alice",
        [
            {
                "ts": first.isoformat(),
                "type": "message_sent",
                "from": "alice",
                "to": "bob",
                "content": "one",
                "meta": {"message_id": "m1"},
            },
            {
                "ts": second.isoformat(),
                "type": "message_sent",
                "from": "alice",
                "to": "bob",
                "content": "two",
                "meta": {"message_id": "m2"},
            },
            {
                "ts": waiting.isoformat(),
                "type": "message_sent",
                "from": "alice",
                "to": "bob",
                "content": "three",
                "meta": {"message_id": "m3"},
            },
        ],
    )
    _write_activity(
        tmp_path,
        "bob",
        [
            {
                "ts": (first + timedelta(seconds=60)).isoformat(),
                "type": "message_received",
                "from": "alice",
                "to": "bob",
                "content": "one",
                "meta": {"message_id": "m1"},
            },
            {
                "ts": (first + timedelta(seconds=60)).isoformat(),
                "type": "message_sent",
                "from": "bob",
                "to": "alice",
                "content": "reply one",
                "meta": {"message_id": "r1", "reply_to": "m1"},
            },
            {
                "ts": (second + timedelta(seconds=120)).isoformat(),
                "type": "message_sent",
                "from": "bob",
                "to": "alice",
                "content": "reply two",
                "meta": {"message_id": "r2", "reply_to": "m2"},
            },
        ],
    )

    result = build_flows(shared, days=7, via="all")
    summary = result["summary"]
    assert summary["calls"] == 6
    assert summary["delivered"] == 5
    assert summary["read"] == 1
    assert summary["replied"] == 4
    assert summary["median_reply_seconds"] == 1800
    assert summary["waiting"] == 1


def test_legacy_dm_read_is_inferred_from_first_receipt(tmp_path: Path):
    shared = tmp_path / "shared"
    for name in ("alice", "bob"):
        _make_anima(tmp_path, name)

    sent_at = _ts(-timedelta(hours=5))
    taken_in = _ts(-timedelta(hours=4))
    _write_activity(
        tmp_path,
        "alice",
        [{"ts": sent_at, "type": "message_sent", "to": "bob", "content": "legacy ask", "meta": {}}],
    )
    _write_activity(
        tmp_path,
        "bob",
        [{"ts": taken_in, "type": "message_received", "from": "alice", "content": "legacy ask", "meta": {}}],
    )

    result = build_flows(shared, days=7, via="dm")
    call = _call_by_text(result, "legacy ask")
    assert call["read"] == taken_in
    assert call["inferred"] is True
