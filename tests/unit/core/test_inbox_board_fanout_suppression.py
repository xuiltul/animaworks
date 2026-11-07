"""Inbox board-mention suppression is scoped to the triggering channels."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.anima.inbox import _board_mention_suppression_channels, _record_board_mention_picks
from core.messaging.messenger import InboxItem
from core.schemas import Message
from core.tooling.handler import ToolHandler, suppress_board_fanout_channels
from core.tooling.handler_base import BOARD_REPLY_CHANNELS_ENV


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


def test_board_mentions_are_recorded_as_picked(tmp_path: Path) -> None:
    from core.messaging.board_reads import load_reads

    shared_dir = tmp_path / "shared"
    items = [
        _item("reply", meta={"board_channel": "team-x", "board_post_id": "post-1"}),
        _item("reply again", meta={"board_channel": "team-x", "board_post_id": "post-2"}),
        _item("legacy", meta={"board_channel": "team-x"}),
        _item("not a board mention", meta={"board_channel": "ops", "board_post_id": "post-3"}, msg_type="message"),
    ]

    _record_board_mention_picks(shared_dir, "alice", items)

    assert load_reads(shared_dir, "team-x")["picked"]["anima:alice"] == ["post-1", "post-2"]
    assert load_reads(shared_dir, "ops")["picked"] == {}


def test_board_mention_pick_failure_only_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args, **_kwargs) -> bool:
        raise RuntimeError("disk full")

    monkeypatch.setattr("core.messaging.board_reads.add_picked", fail)

    _record_board_mention_picks(
        tmp_path / "shared",
        "alice",
        [_item("reply", meta={"board_channel": "team-x", "board_post_id": "post-1"})],
    )


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


def test_subprocess_suppression_comes_from_environment(
    handler: ToolHandler,
    post_api: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Board CLI runs in a Bash subprocess and only sees the exported set."""
    monkeypatch.setenv(BOARD_REPLY_CHANNELS_ENV, "team-x,ops")

    handler._handle_post_channel({"channel": "team-x", "text": "@bob reply"})
    team_suppression = post_api.call_args.kwargs["suppress_mentions"]
    handler._handle_post_channel({"channel": "general", "text": "@bob update"})
    general_suppression = post_api.call_args.kwargs["suppress_mentions"]

    assert team_suppression is True
    assert general_suppression is False


def test_engine_environment_exports_the_suppressed_channels(tmp_path: Path) -> None:
    from core.execution.session.session_context import RuntimeSessionContext
    from core.execution.workspace_env import build_workspace_env

    anima_dir = tmp_path / "animas" / "alice"
    anima_dir.mkdir(parents=True)
    ctx = RuntimeSessionContext.create(session_type="inbox", thread_id="default", trigger="inbox")

    assert build_workspace_env(anima_dir)[BOARD_REPLY_CHANNELS_ENV] == ""
    token = suppress_board_fanout_channels.set(frozenset({"team-x", "ops"}))
    try:
        workspace_value = build_workspace_env(anima_dir)[BOARD_REPLY_CHANNELS_ENV]
        session_value = ctx.to_env()[BOARD_REPLY_CHANNELS_ENV]
    finally:
        suppress_board_fanout_channels.reset(token)

    assert workspace_value == "ops,team-x"
    assert session_value == "ops,team-x"


def test_run_trigger_entries_describe_each_inbox_message() -> None:
    from core.anima.inbox import _run_trigger_entries

    mention = Message(
        from_person="bob",
        to_person="alice",
        type="board_mention",
        content="reply",
        thread_id="thread-1",
        meta={"board_channel": "team-x", "board_post_id": "post-9"},
    )
    direct = Message(
        from_person="carol",
        to_person="alice",
        type="message",
        content="hello",
        thread_id="thread-2",
    )
    items = [
        InboxItem(msg=mention, path=Path("/tmp/unused-inbox-message.json")),
        InboxItem(msg=direct, path=Path("/tmp/unused-inbox-message.json")),
    ]

    assert _run_trigger_entries(items) == (
        {
            "id": mention.id,
            "from": "bob",
            "type": "board_mention",
            "thread_id": "thread-1",
            "board_channel": "team-x",
            "board_post_id": "post-9",
        },
        {
            "id": direct.id,
            "from": "carol",
            "type": "message",
            "thread_id": "thread-2",
        },
    )


def test_run_trigger_entries_omit_missing_optional_values() -> None:
    from core.anima.inbox import _run_trigger_entries

    message = Message(from_person="bob", to_person="alice", type="message", content="hello")
    items = [InboxItem(msg=message, path=Path("/tmp/unused-inbox-message.json"))]

    assert _run_trigger_entries(items) == ({"id": message.id, "from": "bob", "type": "message"},)
