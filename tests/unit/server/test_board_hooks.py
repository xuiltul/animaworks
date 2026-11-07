from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.messaging.board import (
    anima_principal,
    clear_post_hooks,
    external_principal,
    human_principal,
    post_to_board,
)
from core.messaging.board_reads import add_picked, advance_upto, clear_read_hooks
from core.messaging.messenger import ChannelMeta, save_channel_meta
from server.board_hooks import register_board_post_hooks, unregister_board_post_hooks


class _WebSocketManager:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def broadcast(self, event: dict) -> None:
        self.events.append(event)


def _app(shared_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(
            shared_dir=shared_dir,
            ws_manager=_WebSocketManager(),
            disable_external_sync=False,
        )
    )


@pytest.fixture
def shared_dir(tmp_path: Path) -> Path:
    shared = tmp_path / "shared"
    (shared / "channels").mkdir(parents=True)
    for name in ("alice", "bob"):
        anima_dir = tmp_path / "animas" / name
        anima_dir.mkdir(parents=True)
        (anima_dir / "status.json").write_text("{}", encoding="utf-8")
    return shared


def _config(*, whitelist: list[str], all_boards: bool = False):
    return SimpleNamespace(
        external_messaging=SimpleNamespace(
            slack=SimpleNamespace(
                enabled=True,
                board_outbound_sync=whitelist,
                board_outbound_sync_all=all_boards,
                board_mapping={"C1": "general", "CTEAM": "team-x"},
            ),
            discord=SimpleNamespace(board_mapping={}),
        )
    )


@pytest.fixture(autouse=True)
def _clean_hooks():
    clear_post_hooks()
    clear_read_hooks()
    yield
    unregister_board_post_hooks()
    clear_post_hooks()
    clear_read_hooks()


@pytest.mark.asyncio
async def test_ws_hook_emits_post_id_and_kind(shared_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (shared_dir / "channels" / "general.jsonl").touch()
    app = _app(shared_dir)
    app.state.ws_manager = _WebSocketManager()
    monkeypatch.setattr(
        "core.config.models.load_config",
        lambda: SimpleNamespace(external_messaging=SimpleNamespace(slack=SimpleNamespace(enabled=False))),
    )
    register_board_post_hooks(app)

    result = await asyncio.to_thread(
        post_to_board,
        "general",
        "hello",
        human_principal("owner"),
        origin="http",
        shared_dir=shared_dir,
    )

    assert result.hooks == {"ws_event": "sent"}
    event = app.state.ws_manager.events[0]
    assert event["type"] == "board.post"
    assert set(event["data"]) == {"channel", "from", "text", "source", "ts", "id", "kind"}
    assert event["data"]["id"] == result.id
    assert event["data"]["kind"] == "human"


@pytest.mark.asyncio
async def test_ws_hook_includes_reply_target_and_mention_delivery(
    shared_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (shared_dir / "channels" / "general.jsonl").touch()
    bob_dir = shared_dir.parent / "animas" / "bob"
    (bob_dir / "status.json").write_text('{"enabled": true}', encoding="utf-8")
    app = _app(shared_dir)
    monkeypatch.setattr(
        "core.config.models.load_config",
        lambda: SimpleNamespace(external_messaging=SimpleNamespace(slack=SimpleNamespace(enabled=False))),
    )
    register_board_post_hooks(app)

    result = await asyncio.to_thread(
        post_to_board,
        "general",
        "reply @bob",
        anima_principal("alice", animas_dir=shared_dir.parent / "animas"),
        origin="internal",
        shared_dir=shared_dir,
        reply_to="post-1",
    )

    event = app.state.ws_manager.events[0]
    assert result.hooks == {"ws_event": "sent"}
    assert event["data"]["reply_to"] == "post-1"
    assert event["data"]["notified"] == ["bob"]
    assert event["data"]["not_notified"] == {}


@pytest.mark.asyncio
async def test_slack_hook_requires_explicit_members_whitelist_and_skips_echo(
    shared_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (shared_dir / "channels" / "general.jsonl").touch()
    (shared_dir / "channels" / "team-x.jsonl").touch()
    save_channel_meta(shared_dir, "team-x", ChannelMeta(members=["alice", "bob"]))
    monkeypatch.setattr("core.config.models.load_config", lambda: _config(whitelist=[], all_boards=True))
    monkeypatch.delenv("ANIMAWORKS_DISABLE_EXTERNAL_SYNC", raising=False)
    app = _app(shared_dir)
    register_board_post_hooks(app)

    calls: list[str] = []

    async def fake_sync(self, board_name: str, text: str, from_person: str, *, source: str = "") -> str:
        calls.append(board_name)
        return "sent"

    monkeypatch.setattr("core.messaging.outbound_auto.BoardSlackSync.sync_board_post", fake_sync)
    animas_dir = shared_dir.parent / "animas"
    principal = anima_principal("alice", animas_dir=animas_dir)

    blocked_members = await asyncio.to_thread(
        post_to_board,
        "team-x",
        "private post",
        principal,
        origin="internal",
        shared_dir=shared_dir,
    )
    allowed_open = await asyncio.to_thread(
        post_to_board,
        "general",
        "public post",
        principal,
        origin="internal",
        shared_dir=shared_dir,
    )
    slack_echo = await asyncio.to_thread(
        post_to_board,
        "general",
        "external post",
        external_principal("slack-user"),
        origin="slack",
        shared_dir=shared_dir,
    )
    mirror = await asyncio.to_thread(
        post_to_board,
        "general",
        "mirrored message",
        human_principal("guest"),
        origin="mirror",
        shared_dir=shared_dir,
    )

    assert blocked_members.hooks["slack_outbound"] == "skipped"
    assert allowed_open.hooks["slack_outbound"] == "sent"
    assert slack_echo.hooks["slack_outbound"] == "skipped"
    assert mirror.hooks["slack_outbound"] == "skipped"
    assert calls == ["general"]

    save_channel_meta(shared_dir, "general", ChannelMeta(members=[], closed=True))
    closed = await asyncio.to_thread(
        post_to_board,
        "general",
        "human post",
        human_principal("owner"),
        origin="http",
        shared_dir=shared_dir,
    )
    assert closed.error == "denied:closed"
    assert calls == ["general"]


@pytest.mark.asyncio
async def test_read_hook_emits_board_read(shared_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(shared_dir)
    monkeypatch.setattr(
        "core.config.models.load_config",
        lambda: SimpleNamespace(external_messaging=SimpleNamespace(slack=SimpleNamespace(enabled=False))),
    )
    register_board_post_hooks(app)

    await asyncio.to_thread(
        advance_upto,
        shared_dir,
        "general",
        "anima:alice",
        post_id="p1",
        post_ts="2026-01-01T10:00:00+09:00",
    )
    await asyncio.to_thread(add_picked, shared_dir, "general", "anima:alice", "p1")

    events = app.state.ws_manager.events
    assert [event["type"] for event in events] == ["board.read", "board.read"]
    assert set(events[0]["data"]) == {"channel", "reader", "upto_id", "upto_ts", "picked_id"}
    assert events[0]["data"] == {
        "channel": "general",
        "reader": "anima:alice",
        "upto_id": "p1",
        "upto_ts": "2026-01-01T10:00:00+09:00",
        "picked_id": "",
    }
    assert events[1]["data"]["picked_id"] == "p1"
    assert events[1]["data"]["upto_id"] == "p1"
