"""Read-side Board access tests for ``Messenger.read_channel``."""

from __future__ import annotations

import json
from pathlib import Path

from core.messaging.messenger import ChannelMeta, Messenger, save_channel_meta


def _make_anima(data_dir: Path, name: str, status: dict[str, object] | str | None = None) -> Path:
    anima_dir = data_dir / "animas" / name
    anima_dir.mkdir(parents=True)
    if isinstance(status, str):
        (anima_dir / "status.json").write_text(status, encoding="utf-8")
    elif status is not None:
        (anima_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
    return anima_dir


def _write_channel(shared_dir: Path, channel: str, text: str = "hello") -> None:
    channels_dir = shared_dir / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    (channels_dir / f"{channel}.jsonl").write_text(
        json.dumps({"ts": "2026-07-20T00:00:00+09:00", "from": "bob", "text": text}) + "\n",
        encoding="utf-8",
    )


def test_anima_cannot_read_cross_company_member_channel(tmp_path: Path) -> None:
    _make_anima(tmp_path, "alice", {"company": "alpha"})
    _make_anima(tmp_path, "bob", {"company": "beta"})
    shared_dir = tmp_path / "shared"
    _write_channel(shared_dir, "team", "cross-company secret")
    save_channel_meta(shared_dir, "team", ChannelMeta(members=["alice", "bob"]))

    assert Messenger(shared_dir, "alice").read_channel("team") == []


def test_anima_with_unreadable_company_cannot_read(tmp_path: Path) -> None:
    _make_anima(tmp_path, "alice", "{not json")
    shared_dir = tmp_path / "shared"
    _write_channel(shared_dir, "general", "scoped secret")
    save_channel_meta(shared_dir, "general", ChannelMeta(members=[], company="alpha"))

    assert Messenger(shared_dir, "alice").read_channel("general") == []


def test_human_and_operator_can_read_cross_company_channel(tmp_path: Path) -> None:
    _make_anima(tmp_path, "alice", {"company": "alpha"})
    _make_anima(tmp_path, "bob", {"company": "beta"})
    shared_dir = tmp_path / "shared"
    _write_channel(shared_dir, "team", "readable content")
    save_channel_meta(shared_dir, "team", ChannelMeta(members=["alice", "bob"]))

    human_messages = Messenger(shared_dir, "human").read_channel("team", source="human")
    operator_messages = Messenger(shared_dir, "cli").read_channel("team")

    assert [message["text"] for message in human_messages] == ["readable content"]
    assert [message["text"] for message in operator_messages] == ["readable content"]


def test_human_and_operator_can_read_closed_channel(tmp_path: Path) -> None:
    _make_anima(tmp_path, "alice", {})
    shared_dir = tmp_path / "shared"
    _write_channel(shared_dir, "old-board", "archived content")
    save_channel_meta(shared_dir, "old-board", ChannelMeta(members=[], closed=True))

    assert Messenger(shared_dir, "alice").read_channel("old-board") == []
    assert len(Messenger(shared_dir, "human").read_channel("old-board", source="human")) == 1
    assert len(Messenger(shared_dir, "cli").read_channel("old-board")) == 1


def test_human_only_uses_kind_and_legacy_source(tmp_path: Path) -> None:
    _make_anima(tmp_path, "alice", {})
    shared_dir = tmp_path / "shared"
    channels_dir = shared_dir / "channels"
    channels_dir.mkdir(parents=True)
    rows = [
        {"ts": "2026-07-20T00:00:01+09:00", "from": "h", "text": "human", "kind": "human"},
        {"ts": "2026-07-20T00:00:02+09:00", "from": "u", "text": "user", "kind": "external_user"},
        {"ts": "2026-07-20T00:00:03+09:00", "from": "b", "text": "bot", "kind": "external_bot"},
        {"ts": "2026-07-20T00:00:04+09:00", "from": "a", "text": "anima", "kind": "anima"},
        {"ts": "2026-07-20T00:00:05+09:00", "from": "o", "text": "operator", "kind": "operator"},
        {"ts": "2026-07-20T00:00:06+09:00", "from": "l", "text": "legacy", "source": "human"},
        {"ts": "2026-07-20T00:00:07+09:00", "from": "n", "text": "no-kind", "source": "anima"},
    ]
    (channels_dir / "general.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = Messenger(shared_dir, "alice").read_channel("general", human_only=True)

    assert [message["text"] for message in result] == ["human", "user", "legacy"]
