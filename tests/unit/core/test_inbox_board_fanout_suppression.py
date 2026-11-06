"""Inbox board-mention suppression is scoped to the triggering channels."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.anima.inbox import _board_mention_suppression_channels
from core.messaging.messenger import InboxItem
from core.schemas import Message
from core.tooling.handler import ToolHandler, suppress_board_fanout_channels


def _item(content: str, *, meta: dict[str, object] | None = None, msg_type: str = "board_mention") -> InboxItem:
    message = Message(
        from_person="bob",
        to_person="alice",
        type=msg_type,
        content=content,
        meta=meta or {},
    )
    return InboxItem(msg=message, path=Path("/tmp/unused-inbox-message.json"))


def test_suppression_channels_come_from_board_mention_meta() -> None:
    items = [
        _item("reply", meta={"board_channel": "team-x"}),
        _item("other", meta={"board_channel": "ops"}),
        _item("normal message", meta={"board_channel": "ignored"}, msg_type="message"),
    ]

    assert _board_mention_suppression_channels(items) == frozenset({"team-x", "ops"})


def test_suppression_channels_fall_back_to_legacy_reply_prefix() -> None:
    items = [
        _item("[board_reply:channel=team-x,from=bob]\nlegacy body"),
        _item("[board_reply:channel=ops,from=carol]\nlegacy body"),
    ]

    assert _board_mention_suppression_channels(items) == frozenset({"team-x", "ops"})


def test_suppression_channels_ignore_missing_channel_data() -> None:
    assert _board_mention_suppression_channels([_item("no channel")]) == frozenset()


@pytest.fixture
def handler(tmp_path: Path) -> ToolHandler:
    anima_dir = tmp_path / "animas" / "alice"
    anima_dir.mkdir(parents=True)
    (anima_dir / "status.json").write_text("{}", encoding="utf-8")
    (anima_dir / "permissions.md").write_text("", encoding="utf-8")
    messenger = MagicMock()
    messenger.anima_name = "alice"
    messenger.shared_dir = tmp_path / "shared"
    (messenger.shared_dir / "channels").mkdir(parents=True)
    memory = MagicMock()
    memory.read_permissions.return_value = ""
    memory.search_memory_text.return_value = []
    return ToolHandler(anima_dir=anima_dir, memory=memory, messenger=messenger, tool_registry=[])


@pytest.fixture
def post_api(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    api = MagicMock(
        return_value={
            "post": {
                "committed": True,
                "duplicate": False,
                "id": "post-1",
                "notified": [],
                "not_notified": {},
            }
        }
    )
    monkeypatch.setattr("core.messaging.board_client.post_via_server", api)
    return api


def test_only_triggering_channel_suppresses_mentions(
    handler: ToolHandler,
    post_api: MagicMock,
) -> None:
    token = suppress_board_fanout_channels.set(frozenset({"team-x"}))
    try:
        handler._handle_post_channel({"channel": "team-x", "text": "@bob reply"})
        team_suppression = post_api.call_args.kwargs["suppress_mentions"]

        handler._handle_post_channel({"channel": "general", "text": "@bob update"})
        general_suppression = post_api.call_args.kwargs["suppress_mentions"]
    finally:
        suppress_board_fanout_channels.reset(token)

    assert team_suppression is True
    assert general_suppression is False
