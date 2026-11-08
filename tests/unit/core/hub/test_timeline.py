# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from core.hub import timeline
from core.hub.timeline import (
    advance_timeline_read,
    append_item,
    decorate_items,
    is_read_by_human,
    is_read_for_reader,
    list_items,
    load_timeline_reads,
)
from core.platform.atomic_io import atomic_write_json
from core.time_utils import now_local


def _append(tmp_path: Path, **overrides):
    payload = {
        "from_name": "alice",
        "company": "a",
        "kind": "done",
        "text": "Finished the report",
        "source": "anima",
    }
    payload.update(overrides)
    return append_item(tmp_path, **payload)


def test_append_item_shape_and_storage(tmp_path: Path) -> None:
    item = _append(tmp_path, text="  Finished the report  ")

    assert re.fullmatch(r"tl_[0-9a-f]{16}", item["id"])
    assert item["text"] == "Finished the report"
    assert item["about"] is None
    assert item["urgent"] is False
    assert item["ref"] is None
    assert item["priority"] is None
    assert item["delivered"] is None
    assert set(item) == {
        "id",
        "ts",
        "from",
        "company",
        "kind",
        "text",
        "about",
        "urgent",
        "ref",
        "source",
        "priority",
        "delivered",
    }

    path = tmp_path / "timeline" / f"{item['ts'][:7]}.jsonl"
    assert path.is_file()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [item]


def test_append_item_validates_and_truncates(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="kind"):
        _append(tmp_path, kind="posted")
    with pytest.raises(ValueError, match="source"):
        _append(tmp_path, source="human")
    with pytest.raises(ValueError, match="priority"):
        _append(tmp_path, priority="later")
    with pytest.raises(ValueError, match="empty"):
        _append(tmp_path, text="   ")
    with pytest.raises(ValueError, match="company"):
        _append(tmp_path, company=" ")

    long_item = _append(tmp_path, text="x" * 500)
    assert len(long_item["text"]) == 400


def test_append_item_about_validation(tmp_path: Path) -> None:
    original = _append(tmp_path)

    with pytest.raises(ValueError, match="Invalid"):
        _append(tmp_path, about="tl_short")
    with pytest.raises(ValueError, match="not found"):
        _append(tmp_path, about="tl_0123456789abcdef")

    update = _append(tmp_path, kind="decided", about=original["id"], ref="https://example.com/x")
    assert update["about"] == original["id"]
    assert update["ref"] == "https://example.com/x"


def test_list_items_filters_company_since_and_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    month = now_local().strftime("%Y-%m")
    monkeypatch.setattr(timeline, "now_iso", lambda: f"{month}-06T10:00:00+09:00")
    first = _append(tmp_path, text="first")
    monkeypatch.setattr(timeline, "now_iso", lambda: f"{month}-06T11:00:00+09:00")
    second = _append(tmp_path, text="second")
    _append(tmp_path, from_name="bob", company="b", text="other company")

    items = list_items(tmp_path, company="a")
    assert [item["id"] for item in items] == [second["id"], first["id"]]

    since = f"{month}-06T10:30:00+09:00"
    assert [item["id"] for item in list_items(tmp_path, company="a", since=since)] == [second["id"]]
    assert [item["id"] for item in list_items(tmp_path, company="a", limit=1)] == [second["id"]]


def test_list_items_skips_corrupt_lines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    month = now_local().strftime("%Y-%m")
    monkeypatch.setattr(timeline, "now_iso", lambda: f"{month}-06T10:00:00+09:00")
    item = _append(tmp_path)
    path = tmp_path / "timeline" / f"{month}.jsonl"
    path.write_text("not json\n" + json.dumps(item) + "\n", encoding="utf-8")

    assert [entry["id"] for entry in list_items(tmp_path, company="a")] == [item["id"]]


def test_read_positions_use_timeline_document(tmp_path: Path) -> None:
    item = _append(tmp_path)
    reads = load_timeline_reads(tmp_path)
    assert reads["stream"] == "timeline"
    assert is_read_by_human(item, reads) is False

    assert advance_timeline_read(tmp_path, "human:alice", item["id"]) is True
    assert advance_timeline_read(tmp_path, "human:alice", item["id"]) is False
    assert advance_timeline_read(tmp_path, "human:alice", "tl_0123456789abcdef") is False

    reads = load_timeline_reads(tmp_path)
    assert reads["readers"]["human:alice"]["upto_id"] == item["id"]
    assert reads["readers"]["human:alice"]["upto_ts"] == item["ts"]
    assert is_read_by_human(item, reads) is True
    assert is_read_for_reader(item, reads, "human:alice") is True
    assert is_read_for_reader(item, reads, "human:bob") is False


def test_picked_item_counts_as_read(tmp_path: Path) -> None:
    item = _append(tmp_path)
    reads_doc = {
        "stream": "timeline",
        "readers": {},
        "picked": {"human:alice": [item["id"]]},
    }
    atomic_write_json(tmp_path / "reads" / "timeline.json", reads_doc)

    assert is_read_by_human(item, load_timeline_reads(tmp_path)) is True


def test_decorate_items_marks_updates_and_changes(tmp_path: Path) -> None:
    original = _append(tmp_path, kind="notified", text="Started")
    advance_timeline_read(tmp_path, "human:alice", original["id"])
    update = _append(tmp_path, kind="decided", text="Decided", about=original["id"])

    reads = load_timeline_reads(tmp_path)
    originals = decorate_items([original], reads=reads, reader="human:alice", reference_items=[original, update])
    updates = decorate_items([update], reads=reads, reader="human:alice", reference_items=[original, update])

    assert originals[0]["updates"] == [update["id"]]
    assert originals[0]["read"] is True
    assert updates[0]["changed"] is True
    assert updates[0]["read"] is False
