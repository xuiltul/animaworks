"""Unit tests for Board channel company boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.messenger import (
    ChannelMeta,
    Messenger,
    is_channel_member,
    load_channel_meta,
    save_channel_meta,
)
from core.tooling.handler import ToolHandler


def _make_anima(data_dir: Path, name: str, company: str | None) -> Path:
    anima_dir = data_dir / "animas" / name
    anima_dir.mkdir(parents=True)
    status: dict[str, object] = {"enabled": True}
    if company is not None:
        status["company"] = company
    (anima_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
    return anima_dir


def _write_company(data_dir: Path, name: str, display_name: str) -> None:
    company_dir = data_dir / "companies" / name
    company_dir.mkdir(parents=True)
    (company_dir / "company.json").write_text(
        json.dumps({"name": name, "display_name": display_name}),
        encoding="utf-8",
    )


def _make_handler(data_dir: Path, anima_name: str = "alice") -> ToolHandler:
    memory = MagicMock()
    memory.read_permissions.return_value = ""
    memory.search_memory_text.return_value = []
    messenger = Messenger(data_dir / "shared", anima_name)
    handler = ToolHandler(
        anima_dir=data_dir / "animas" / anima_name,
        memory=memory,
        messenger=messenger,
        tool_registry=[],
    )
    handler._fire_board_slack_sync = MagicMock()
    return handler


def _make_channel(data_dir: Path, channel: str, members: list[str]) -> Path:
    channels_dir = data_dir / "shared" / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    channel_file = channels_dir / f"{channel}.jsonl"
    channel_file.write_text(
        json.dumps(
            {"ts": "2026-07-20T00:00:00+09:00", "from": members[-1], "text": "secret"}
        )
        + "\n",
        encoding="utf-8",
    )
    save_channel_meta(data_dir / "shared", channel, ChannelMeta(members=members))
    return channel_file


def test_post_rejects_mixed_company_channel_then_observes_status_change(tmp_path: Path) -> None:
    _make_anima(tmp_path, "alice", "alpha")
    bob_dir = _make_anima(tmp_path, "bob", "beta")
    _write_company(tmp_path, "beta", "Beta Corporation")
    channel_file = _make_channel(tmp_path, "team", ["alice", "bob"])
    handler = _make_handler(tmp_path)

    denied = handler._handle_post_channel({"channel": "team", "text": "hello"})

    assert "Beta Corporation" in denied
    assert len(channel_file.read_text(encoding="utf-8").splitlines()) == 1

    # Membership is read from status.json for every operation, without a cache.
    (bob_dir / "status.json").write_text(
        json.dumps({"enabled": True, "company": "alpha"}),
        encoding="utf-8",
    )
    limiter = MagicMock()
    limiter.check_global_outbound.return_value = True
    with patch("core.cascade_limiter.get_depth_limiter", return_value=limiter):
        allowed = handler._handle_post_channel({"channel": "team", "text": "hello"})

    assert allowed == "Posted to #team"
    assert len(channel_file.read_text(encoding="utf-8").splitlines()) == 2


def test_read_channel_rejects_mixed_company_members(tmp_path: Path) -> None:
    _make_anima(tmp_path, "alice", "alpha")
    _make_anima(tmp_path, "bob", "beta")
    _write_company(tmp_path, "beta", "Beta Corporation")
    _make_channel(tmp_path, "team", ["alice", "bob"])
    handler = _make_handler(tmp_path)

    result = handler._handle_read_channel({"channel": "team"})

    assert "Beta Corporation" in result
    assert "secret" not in result


def test_open_channel_treats_all_animas_as_implicit_members(tmp_path: Path) -> None:
    _make_anima(tmp_path, "alice", "alpha")
    _make_anima(tmp_path, "bob", "beta")
    _write_company(tmp_path, "beta", "Beta Corporation")
    channels_dir = tmp_path / "shared" / "channels"
    channels_dir.mkdir(parents=True)
    channel_file = channels_dir / "general.jsonl"
    channel_file.write_text("", encoding="utf-8")
    handler = _make_handler(tmp_path)

    result = handler._handle_post_channel({"channel": "general", "text": "hello"})

    assert "Beta Corporation" in result
    assert channel_file.read_text(encoding="utf-8") == ""


def test_read_dm_history_rejects_cross_company_peer(tmp_path: Path) -> None:
    _make_anima(tmp_path, "alice", "alpha")
    _make_anima(tmp_path, "bob", "beta")
    _write_company(tmp_path, "beta", "Beta Corporation")
    handler = _make_handler(tmp_path)
    handler._messenger.read_dm_history = MagicMock(return_value=[])

    result = handler._handle_read_dm_history({"peer": "bob"})

    assert "Beta Corporation" in result
    handler._messenger.read_dm_history.assert_not_called()


def test_create_channel_rejects_cross_company_member(tmp_path: Path) -> None:
    _make_anima(tmp_path, "alice", "alpha")
    _make_anima(tmp_path, "bob", "beta")
    _write_company(tmp_path, "beta", "Beta Corporation")
    handler = _make_handler(tmp_path)

    result = handler._handle_manage_channel(
        {"action": "create", "channel": "mixed", "members": ["bob"]}
    )

    assert "Beta Corporation" in result
    assert not (tmp_path / "shared" / "channels" / "mixed.jsonl").exists()
    assert load_channel_meta(tmp_path / "shared", "mixed") is None


def test_add_member_rejects_cross_company_member_without_mutation(tmp_path: Path) -> None:
    _make_anima(tmp_path, "alice", "alpha")
    _make_anima(tmp_path, "bob", "beta")
    _write_company(tmp_path, "beta", "Beta Corporation")
    _make_channel(tmp_path, "team", ["alice"])
    handler = _make_handler(tmp_path)

    result = handler._handle_manage_channel(
        {"action": "add_member", "channel": "team", "members": ["bob"]}
    )

    assert "Beta Corporation" in result
    meta = load_channel_meta(tmp_path / "shared", "team")
    assert meta is not None
    assert meta.members == ["alice"]


def test_unassigned_anima_keeps_legacy_cross_company_access(tmp_path: Path) -> None:
    _make_anima(tmp_path, "legacy", None)
    _make_anima(tmp_path, "bob", "beta")
    handler = _make_handler(tmp_path, "legacy")

    result = handler._handle_manage_channel(
        {"action": "create", "channel": "legacy-team", "members": ["bob"]}
    )

    assert "legacy-team" in result
    meta = load_channel_meta(tmp_path / "shared", "legacy-team")
    assert meta is not None
    assert meta.members == ["legacy", "bob"]


def test_mention_fanout_only_notifies_same_company_targets(tmp_path: Path) -> None:
    _make_anima(tmp_path, "alice", "alpha")
    _make_anima(tmp_path, "same", "alpha")
    _make_anima(tmp_path, "other", "beta")
    _make_channel(tmp_path, "team", ["alice", "same", "other"])
    sockets_dir = tmp_path / "run" / "sockets"
    sockets_dir.mkdir(parents=True)
    (sockets_dir / "same.sock").touch()
    (sockets_dir / "other.sock").touch()
    handler = _make_handler(tmp_path)
    handler._messenger.send = MagicMock()

    with patch("core.paths.get_data_dir", return_value=tmp_path):
        handler._fanout_board_mentions("team", "@same @other please review")

    assert [call.kwargs["to"] for call in handler._messenger.send.call_args_list] == ["same"]


def test_explicitly_closed_empty_channel_denies_animas(tmp_path: Path) -> None:
    shared_dir = tmp_path / "shared"
    save_channel_meta(shared_dir, "old-board", ChannelMeta(members=[], closed=True))

    assert not is_channel_member(shared_dir, "old-board", "alice")
    assert is_channel_member(shared_dir, "old-board", "alice", source="human")
    assert is_channel_member(shared_dir, "legacy-open", "alice")
