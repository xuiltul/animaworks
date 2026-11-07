"""Tests for Board delivery records."""

from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

from core.messaging.board import anima_principal, post_to_board
from core.messaging.board_deliveries import (
    append_delivery,
    attach_deliveries,
    deliveries_path,
    load_deliveries,
)


def _shared_dir(tmp_path: Path) -> Path:
    shared = tmp_path / "shared"
    (shared / "channels").mkdir(parents=True)
    (shared / "channels" / "general.jsonl").touch()
    for name, enabled in (("alice", True), ("bob", True), ("carol", False)):
        anima_dir = tmp_path / "animas" / name
        anima_dir.mkdir(parents=True)
        (anima_dir / "status.json").write_text(json.dumps({"enabled": enabled}), encoding="utf-8")
    return shared


def test_append_and_load_delivery(tmp_path: Path) -> None:
    shared = _shared_dir(tmp_path)

    append_delivery(
        shared,
        "general",
        post_id="post-1",
        ts="2026-01-01T00:00:00+00:00",
        notified=["bob"],
        not_notified={"carol": "disabled"},
    )

    deliveries = load_deliveries(shared, "general")
    assert deliveries["post-1"] == {
        "post_id": "post-1",
        "ts": "2026-01-01T00:00:00+00:00",
        "notified": ["bob"],
        "not_notified": {"carol": "disabled"},
    }


def test_load_deliveries_sees_appended_record(tmp_path: Path) -> None:
    shared = _shared_dir(tmp_path)
    append_delivery(shared, "general", post_id="post-1", ts="t1", notified=["bob"], not_notified={})
    assert set(load_deliveries(shared, "general")) == {"post-1"}

    append_delivery(shared, "general", post_id="post-2", ts="t2", notified=[], not_notified={"carol": "disabled"})

    assert set(load_deliveries(shared, "general")) == {"post-1", "post-2"}


def test_load_deliveries_ignores_malformed_lines(tmp_path: Path) -> None:
    shared = _shared_dir(tmp_path)
    path = deliveries_path(shared, "general")
    path.parent.mkdir()
    path.write_text('not json\n{"post_id": "post-1", "notified": []}\n', encoding="utf-8")

    assert set(load_deliveries(shared, "general")) == {"post-1"}


def test_attach_deliveries_copies_messages(tmp_path: Path) -> None:
    shared = _shared_dir(tmp_path)
    append_delivery(
        shared,
        "general",
        post_id="post-1",
        ts="t1",
        notified=["bob"],
        not_notified={"carol": "disabled"},
    )
    messages = [{"id": "post-1", "text": "hello"}, {"id": "post-2", "text": "other"}]

    attached = attach_deliveries(shared, "general", messages)

    assert attached[0]["delivery"] == {"notified": ["bob"], "not_notified": {"carol": "disabled"}}
    assert "delivery" not in attached[1]
    assert "delivery" not in messages[0]


def test_post_to_board_records_delivery_outside_channels(tmp_path: Path) -> None:
    shared = _shared_dir(tmp_path)
    channels_before = sorted(path.name for path in (shared / "channels").iterdir())

    result = post_to_board(
        "general",
        "Hello @bob and @carol",
        anima_principal("alice", animas_dir=tmp_path / "animas"),
        origin="internal",
        shared_dir=shared,
    )

    assert result.committed is True
    assert result.notified == ["bob"]
    assert result.not_notified == {"carol": "disabled"}
    delivery_file = shared / "board_deliveries" / "general.jsonl"
    assert delivery_file.exists()
    record = json.loads(delivery_file.read_text(encoding="utf-8").strip())
    assert record["post_id"] == result.id
    assert record["notified"] == ["bob"]
    assert record["not_notified"] == {"carol": "disabled"}
    assert sorted(path.name for path in (shared / "channels").glob("*.jsonl")) == channels_before


def test_post_without_mentions_writes_no_delivery(tmp_path: Path) -> None:
    shared = _shared_dir(tmp_path)

    post_to_board(
        "general",
        "Hello everyone",
        anima_principal("alice", animas_dir=tmp_path / "animas"),
        origin="internal",
        shared_dir=shared,
    )

    assert not deliveries_path(shared, "general").exists()
