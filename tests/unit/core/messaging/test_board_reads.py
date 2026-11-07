from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core.messaging.board_reads import (
    add_picked,
    advance_upto,
    clear_read_hooks,
    load_reads,
    reads_path,
    register_read_hook,
    unread_summary,
    unregister_read_hook,
)


def _write_channel(shared_dir: Path, channel: str, entries: list[dict]) -> None:
    channels_dir = shared_dir / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    (channels_dir / f"{channel}.jsonl").write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries),
        encoding="utf-8",
    )


def test_advance_upto_never_moves_backwards(tmp_path: Path) -> None:
    shared_dir = tmp_path / "shared"

    assert advance_upto(shared_dir, "general", "anima:alice", post_id="a1", post_ts="2026-01-01T10:00:00+09:00")
    assert not advance_upto(shared_dir, "general", "anima:alice", post_id="a1", post_ts="2026-01-01T10:00:00+09:00")
    assert not advance_upto(shared_dir, "general", "anima:alice", post_id="a0", post_ts="2026-01-01T09:00:00+09:00")
    assert advance_upto(shared_dir, "general", "anima:alice", post_id="a2", post_ts="2026-01-01T11:00:00+09:00")

    record = load_reads(shared_dir, "general")["readers"]["anima:alice"]
    assert record["upto_id"] == "a2"
    assert record["upto_ts"] == "2026-01-01T11:00:00+09:00"


def test_advance_upto_handles_idless_legacy_rows(tmp_path: Path) -> None:
    shared_dir = tmp_path / "shared"

    assert advance_upto(shared_dir, "general", "anima:alice", post_id=None, post_ts="2026-01-01T10:00:00+09:00")

    record = load_reads(shared_dir, "general")["readers"]["anima:alice"]
    assert record["upto_id"] == ""
    assert record["upto_ts"] == "2026-01-01T10:00:00+09:00"


def test_add_picked_keeps_only_the_newest_500(tmp_path: Path) -> None:
    shared_dir = tmp_path / "shared"

    for index in range(505):
        assert add_picked(shared_dir, "general", "anima:alice", f"id-{index}")

    picked = load_reads(shared_dir, "general")["picked"]["anima:alice"]
    assert len(picked) == 500
    assert picked[0] == "id-5"
    assert picked[-1] == "id-504"
    assert not add_picked(shared_dir, "general", "anima:alice", "id-504")


def test_unread_summary_excludes_self_and_picked_and_counts_mentions(tmp_path: Path) -> None:
    shared_dir = tmp_path / "shared"
    _write_channel(
        shared_dir,
        "general",
        [
            {"ts": "2026-01-01T09:00:00+09:00", "from": "alice", "text": "my own post"},
            {"ts": "2026-01-01T10:00:00+09:00", "from": "bob", "text": "hello @alice", "id": "b1"},
            {"ts": "2026-01-01T11:00:00+09:00", "from": "bob", "text": "plain", "id": "b2"},
            {"ts": "2026-01-01T12:00:00+09:00", "from": "owner", "text": "@alice please check", "id": "h1"},
        ],
    )

    first = unread_summary(shared_dir, "general", "anima:alice", self_name="alice")
    assert first == {"unread": 3, "to_me": 2}

    advance_upto(shared_dir, "general", "anima:alice", post_id="b1", post_ts="2026-01-01T10:00:00+09:00")
    add_picked(shared_dir, "general", "anima:alice", "h1")

    assert unread_summary(shared_dir, "general", "anima:alice", self_name="alice") == {
        "unread": 1,
        "to_me": 0,
    }


def test_unread_summary_counts_legacy_idless_row(tmp_path: Path) -> None:
    shared_dir = tmp_path / "shared"
    _write_channel(
        shared_dir,
        "general",
        [
            {"ts": "2026-01-01T10:00:00+09:00", "from": "bob", "text": "with id", "id": "b1"},
            {"ts": "2026-01-01T11:00:00+09:00", "from": "bob", "text": "legacy @alice"},
        ],
    )
    advance_upto(shared_dir, "general", "anima:alice", post_id="b1", post_ts="2026-01-01T10:00:00+09:00")

    assert unread_summary(shared_dir, "general", "anima:alice", self_name="alice") == {
        "unread": 1,
        "to_me": 1,
    }


def test_concurrent_updates_are_locked_and_atomically_replaced(tmp_path: Path) -> None:
    shared_dir = tmp_path / "shared"

    def update(reader: str) -> bool:
        return advance_upto(
            shared_dir,
            "general",
            reader,
            post_id=reader,
            post_ts="2026-01-01T10:00:00+09:00",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(pool.map(update, [f"anima:r{index}" for index in range(16)]))

    payload = json.loads(reads_path(shared_dir, "general").read_text(encoding="utf-8"))
    assert set(payload["readers"]) == {f"anima:r{index}" for index in range(16)}
    assert payload["stream"] == "board/general"
    assert sorted(path.name for path in reads_path(shared_dir, "general").parent.iterdir()) == [
        "general.json",
        "general.json.lock",
    ]


def test_read_hook_receives_update_and_cannot_break_write(tmp_path: Path) -> None:
    shared_dir = tmp_path / "shared"
    calls: list[dict] = []
    clear_read_hooks()
    try:
        register_read_hook("collect", calls.append)
        register_read_hook("explode", lambda _payload: (_ for _ in ()).throw(RuntimeError("boom")))

        assert advance_upto(
            shared_dir,
            "general",
            "anima:alice",
            post_id="a1",
            post_ts="2026-01-01T10:00:00+09:00",
        )
        assert add_picked(shared_dir, "general", "anima:alice", "a1")

        assert calls[0] == {
            "channel": "general",
            "reader": "anima:alice",
            "upto_id": "a1",
            "upto_ts": "2026-01-01T10:00:00+09:00",
            "picked_id": "",
        }
        assert calls[1]["picked_id"] == "a1"
        assert load_reads(shared_dir, "general")["picked"]["anima:alice"] == ["a1"]
    finally:
        unregister_read_hook("collect")
        unregister_read_hook("explode")
        clear_read_hooks()
