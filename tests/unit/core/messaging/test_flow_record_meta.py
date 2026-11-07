"""Tests that message records carry the identifiers flow joins need."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from core.activity.logger import ActivityLogger
from core.anima.inbox import InboxMixin
from core.messaging.messenger import Messenger


def _make_anima(root: Path, name: str) -> Path:
    anima_dir = root / "animas" / name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "status.json").write_text(json.dumps({"enabled": True}), encoding="utf-8")
    return anima_dir


def test_message_sent_records_message_id(tmp_path: Path):
    _make_anima(tmp_path, "alice")
    _make_anima(tmp_path, "bob")
    shared = tmp_path / "shared"

    sent = Messenger(shared, "alice").send(to="bob", content="hello")

    entries = ActivityLogger(tmp_path / "animas" / "alice").recent(days=1, types=["message_sent"])
    assert entries[-1].meta["message_id"] == sent.id


async def test_board_mention_received_records_message_and_board_ids(tmp_path: Path):
    anima_dir = _make_anima(tmp_path, "alice")
    _make_anima(tmp_path, "bob")
    shared = tmp_path / "shared"
    sent = Messenger(shared, "bob").send(
        to="alice",
        content="reply",
        msg_type="board_mention",
        meta={"board_channel": "general", "board_post_id": "post-1"},
    )

    host = MagicMock()
    host.name = "alice"
    host.anima_dir = anima_dir
    host.messenger = Messenger(shared, "alice")
    host.memory = MagicMock()
    host._activity = ActivityLogger(anima_dir)

    await InboxMixin._process_inbox_messages(host)

    entries = ActivityLogger(anima_dir).recent(days=1, types=["message_received"])
    assert entries[-1].meta["message_id"] == sent.id
    assert entries[-1].meta["board_channel"] == "general"
    assert entries[-1].meta["board_post_id"] == "post-1"
