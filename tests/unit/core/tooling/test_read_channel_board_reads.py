"""Read-position behavior of the ``read_channel`` tool handler."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from core.messaging.board_reads import load_reads
from core.messaging.messenger import Messenger
from core.tooling.handler import ToolHandler


def _make_handler(tmp_path: Path, name: str = "alice") -> tuple[ToolHandler, Path]:
    data_dir = tmp_path
    anima_dir = data_dir / "animas" / name
    anima_dir.mkdir(parents=True)
    (anima_dir / "status.json").write_text("{}", encoding="utf-8")
    shared_dir = data_dir / "shared"
    (shared_dir / "channels").mkdir(parents=True)
    handler = ToolHandler(
        anima_dir=anima_dir,
        memory=MagicMock(),
        messenger=Messenger(shared_dir, name),
        tool_registry=[],
    )
    return handler, shared_dir


def _write_channel(shared_dir: Path, entries: list[dict]) -> None:
    (shared_dir / "channels" / "general.jsonl").write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries),
        encoding="utf-8",
    )


def _payload(output: str) -> list[dict]:
    return json.loads(output.split("\n", 1)[1])


def _entry(hour: int, sender: str, text: str, post_id: str) -> dict:
    return {
        "ts": f"2026-01-01T{hour:02d}:00:00+09:00",
        "from": sender,
        "text": text,
        "id": post_id,
    }


def test_first_read_returns_latest_and_advances(tmp_path: Path) -> None:
    handler, shared_dir = _make_handler(tmp_path)
    entries = [_entry(hour, "bob", f"message {hour}", f"id{hour}") for hour in range(1, 6)]
    _write_channel(shared_dir, entries)

    output = handler._handle_read_channel({"channel": "general", "limit": 3})

    payload = _payload(output)
    assert [entry["id"] for entry in payload] == ["id3", "id4", "id5"]
    assert "最新 3 件" in output.split("\n", 1)[0]
    record = load_reads(shared_dir, "general")["readers"]["anima:alice"]
    assert record["upto_id"] == "id5"
    assert record["upto_ts"] == entries[-1]["ts"]


def test_continuation_reads_oldest_unread_first_and_advances(tmp_path: Path) -> None:
    handler, shared_dir = _make_handler(tmp_path)
    _write_channel(shared_dir, [_entry(hour, "bob", f"old {hour}", f"old{hour}") for hour in range(1, 6)])
    handler._handle_read_channel({"channel": "general", "limit": 2})

    _write_channel(
        shared_dir,
        [
            *_entry_list(1, 5, "old"),
            _entry(6, "bob", "new 6", "new6"),
            _entry(7, "bob", "new 7", "new7"),
            _entry(8, "bob", "new 8", "new8"),
        ],
    )

    output = handler._handle_read_channel({"channel": "general", "limit": 2})

    payload = _payload(output)
    assert [entry["id"] for entry in payload] == ["new6", "new7"]
    headline = output.split("\n", 1)[0]
    assert "古い順" in headline
    assert "残り 1 件" in headline
    assert "--latest" in headline
    assert load_reads(shared_dir, "general")["readers"]["anima:alice"]["upto_id"] == "new7"


def test_latest_does_not_advance_and_marks_picked(tmp_path: Path) -> None:
    handler, shared_dir = _make_handler(tmp_path)
    _write_channel(shared_dir, [_entry(hour, "bob", f"message {hour}", f"id{hour}") for hour in range(1, 6)])

    output = handler._handle_read_channel({"channel": "general", "limit": 2, "latest": True})

    assert [entry["id"] for entry in json.loads(output)] == ["id4", "id5"]
    assert load_reads(shared_dir, "general")["readers"] == {}
    assert load_reads(shared_dir, "general")["picked"]["anima:alice"] == ["id4", "id5"]


def test_human_only_marks_picked_without_advancing(tmp_path: Path) -> None:
    handler, shared_dir = _make_handler(tmp_path)
    _write_channel(
        shared_dir,
        [
            _entry(1, "bob", "anima post", "a1"),
            {"ts": "2026-01-01T02:00:00+09:00", "from": "owner", "text": "human post", "id": "h1", "kind": "human"},
            _entry(3, "bob", "another anima post", "a2"),
        ],
    )

    output = handler._handle_read_channel({"channel": "general", "limit": 5, "human_only": True})

    assert [entry["id"] for entry in json.loads(output)] == ["h1"]
    assert load_reads(shared_dir, "general")["readers"] == {}
    assert load_reads(shared_dir, "general")["picked"]["anima:alice"] == ["h1"]


def test_no_new_posts_returns_headline_without_repeating_old_posts(tmp_path: Path) -> None:
    handler, shared_dir = _make_handler(tmp_path)
    _write_channel(shared_dir, [_entry(1, "bob", "message", "id1")])
    handler._handle_read_channel({"channel": "general", "limit": 5})

    output = handler._handle_read_channel({"channel": "general", "limit": 5})

    headline = output.split("\n", 1)[0]
    assert "#general に新しい投稿はありません" in headline
    assert _payload(output) == []


def test_own_posts_do_not_make_the_summary_claim_unread(tmp_path: Path) -> None:
    handler, shared_dir = _make_handler(tmp_path)
    _write_channel(
        shared_dir,
        [
            _entry(1, "bob", "first", "b1"),
            _entry(2, "alice", "my own", "a1"),
        ],
    )

    output = handler._handle_read_channel({"channel": "general", "limit": 5})

    # The own post is still returned by the unfiltered continuation, but it is
    # not counted as unread in the headline.
    assert "未読 1 件" in output.split("\n", 1)[0]


def test_missing_channel_uses_existing_empty_message(tmp_path: Path) -> None:
    handler, shared_dir = _make_handler(tmp_path)
    _write_channel(shared_dir, [])

    assert handler._handle_read_channel({"channel": "general"}) == "#general にはまだ投稿がありません。"


def _entry_list(first_hour: int, last_hour: int, prefix: str) -> list[dict]:
    return [_entry(hour, "bob", f"{prefix} {hour}", f"{prefix}{hour}") for hour in range(first_hour, last_hour + 1)]
