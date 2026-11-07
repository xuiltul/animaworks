"""Unread Board counts in the recent-activity priming section."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.memory.priming.channel_b import channel_b_recent_activity
from core.messaging.board_reads import advance_upto
from core.time_utils import now_local


def _write_channel(shared_dir: Path, name: str, entries: list[dict]) -> None:
    channels_dir = shared_dir / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    (channels_dir / f"{name}.jsonl").write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries),
        encoding="utf-8",
    )


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path]:
    anima_dir = tmp_path / "animas" / "alice"
    anima_dir.mkdir(parents=True)
    (anima_dir / "status.json").write_text("{}", encoding="utf-8")
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()
    return anima_dir, shared_dir


async def test_unread_line_reports_total_and_mentions(workspace: tuple[Path, Path]) -> None:
    anima_dir, shared_dir = workspace
    now = now_local().isoformat()
    _write_channel(
        shared_dir,
        "general",
        [
            {"ts": now, "from": "bob", "text": "hello @alice", "id": "p1", "source": "anima"},
            {"ts": now, "from": "bob", "text": "plain update", "id": "p2", "source": "anima"},
            {"ts": now, "from": "alice", "text": "my own post", "id": "p3", "source": "anima"},
        ],
    )

    result = await channel_b_recent_activity(anima_dir, shared_dir, "alice", [])

    unread_items = [item for item in result.items if item.source == "board_unread"]
    assert len(unread_items) == 1
    assert unread_items[0].text == "#general 未読 2 件（うちあなた宛て 1 件）"
    assert unread_items[0].text in result


async def test_unread_line_follows_the_stored_position(workspace: tuple[Path, Path]) -> None:
    anima_dir, shared_dir = workspace
    now = now_local().isoformat()
    _write_channel(
        shared_dir,
        "general",
        [
            {"ts": "2026-01-01T10:00:00+09:00", "from": "bob", "text": "first @alice", "id": "p1", "source": "anima"},
            {"ts": now, "from": "bob", "text": "second", "id": "p2", "source": "anima"},
        ],
    )
    advance_upto(shared_dir, "general", "anima:alice", post_id="p1", post_ts="2026-01-01T10:00:00+09:00")

    result = await channel_b_recent_activity(anima_dir, shared_dir, "alice", [])

    unread_items = [item for item in result.items if item.source == "board_unread"]
    assert len(unread_items) == 1
    assert unread_items[0].text == "#general 未読 1 件（うちあなた宛て 0 件）"


async def test_only_readable_boards_are_listed(workspace: tuple[Path, Path]) -> None:
    anima_dir, shared_dir = workspace
    now = now_local().isoformat()
    _write_channel(
        shared_dir,
        "general",
        [{"ts": now, "from": "bob", "text": "hello", "id": "p1", "source": "anima"}],
    )
    _write_channel(
        shared_dir,
        "restricted",
        [{"ts": now, "from": "bob", "text": "private", "id": "p2", "source": "anima"}],
    )
    (shared_dir / "channels" / "restricted.meta.json").write_text("{not json", encoding="utf-8")

    result = await channel_b_recent_activity(anima_dir, shared_dir, "alice", [])

    unread_texts = [item.text for item in result.items if item.source == "board_unread"]
    assert unread_texts == ["#general 未読 1 件（うちあなた宛て 0 件）"]
