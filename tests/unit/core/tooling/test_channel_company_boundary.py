"""Unit tests for Board channel company boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.i18n import t
from core.messaging.messenger import (
    ChannelMeta,
    Messenger,
    is_channel_member,
    load_channel_meta,
    save_channel_meta,
)
from core.tooling.handler import ToolHandler


@pytest.fixture(autouse=True)
def _direct_board_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from core.config import invalidate_cache
    from core.messaging.board import anima_principal, post_to_board
    from core.paths import get_animas_dir, get_shared_dir

    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
    invalidate_cache()

    def post_direct(
        channel: str,
        text: str,
        *,
        from_anima: str,
        source: str = "anima",
        from_name: str | None = None,
        request_id: str | None = None,
        suppress_mentions: bool = False,
    ) -> dict:
        result = post_to_board(
            channel,
            text,
            anima_principal(from_anima, animas_dir=get_animas_dir()),
            request_id=request_id,
            origin="internal",
            notify=not suppress_mentions,
            shared_dir=get_shared_dir(),
        )
        return {"post": result.to_dict()}

    monkeypatch.setattr("core.messaging.board_client.post_via_server", post_direct)
    yield
    invalidate_cache()


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
    return handler


def _make_channel(data_dir: Path, channel: str, members: list[str]) -> Path:
    channels_dir = data_dir / "shared" / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    channel_file = channels_dir / f"{channel}.jsonl"
    channel_file.write_text(
        json.dumps({"ts": "2026-07-20T00:00:00+09:00", "from": members[-1], "text": "secret"}) + "\n",
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
    allowed = handler._handle_post_channel({"channel": "team", "text": "hello"})

    assert allowed == t("handler.channel_post_success", channel="team")
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


def test_open_channel_without_company_rejects_assigned_anima(tmp_path: Path) -> None:
    """Legacy open channels no longer expand to all animas; require company attr."""
    _make_anima(tmp_path, "alice", "alpha")
    _make_anima(tmp_path, "bob", "beta")
    _write_company(tmp_path, "beta", "Beta Corporation")
    channels_dir = tmp_path / "shared" / "channels"
    channels_dir.mkdir(parents=True)
    channel_file = channels_dir / "general.jsonl"
    channel_file.write_text("", encoding="utf-8")
    handler = _make_handler(tmp_path)

    result = handler._handle_post_channel({"channel": "general", "text": "hello"})

    assert "会社帰属が未設定" in result
    assert "Beta Corporation" not in result  # must not be the all-animas expansion error
    assert channel_file.read_text(encoding="utf-8") == ""


def test_company_scoped_open_channel_same_company_allows_post(tmp_path: Path) -> None:
    _make_anima(tmp_path, "alice", "alpha")
    _make_anima(tmp_path, "bob", "beta")
    _write_company(tmp_path, "alpha", "Alpha Co")
    channels_dir = tmp_path / "shared" / "channels"
    channels_dir.mkdir(parents=True)
    channel_file = channels_dir / "general.jsonl"
    channel_file.write_text("", encoding="utf-8")
    save_channel_meta(tmp_path / "shared", "general", ChannelMeta(members=[], company="alpha"))
    handler = _make_handler(tmp_path)

    result = handler._handle_post_channel({"channel": "general", "text": "hello"})

    assert result == t("handler.channel_post_success", channel="general")
    assert len(channel_file.read_text(encoding="utf-8").splitlines()) == 1


def test_company_scoped_open_channel_other_company_rejects(tmp_path: Path) -> None:
    _make_anima(tmp_path, "alice", "alpha")
    _write_company(tmp_path, "beta", "Beta Corporation")
    channels_dir = tmp_path / "shared" / "channels"
    channels_dir.mkdir(parents=True)
    channel_file = channels_dir / "ops.jsonl"
    channel_file.write_text("", encoding="utf-8")
    save_channel_meta(tmp_path / "shared", "ops", ChannelMeta(members=[], company="beta"))
    handler = _make_handler(tmp_path)

    result = handler._handle_post_channel({"channel": "ops", "text": "hello"})

    assert "Beta Corporation" in result
    assert channel_file.read_text(encoding="utf-8") == ""


def test_company_scoped_open_channel_unassigned_legacy_allows(tmp_path: Path) -> None:
    _make_anima(tmp_path, "legacy", None)
    _write_company(tmp_path, "alpha", "Alpha Co")
    channels_dir = tmp_path / "shared" / "channels"
    channels_dir.mkdir(parents=True)
    channel_file = channels_dir / "ops.jsonl"
    channel_file.write_text("", encoding="utf-8")
    save_channel_meta(tmp_path / "shared", "ops", ChannelMeta(members=[], company="alpha"))
    handler = _make_handler(tmp_path, "legacy")

    result = handler._handle_post_channel({"channel": "ops", "text": "hello"})

    assert result == t("handler.channel_post_success", channel="ops")
    assert len(channel_file.read_text(encoding="utf-8").splitlines()) == 1


def test_unassigned_open_channel_allows_unassigned_anima(tmp_path: Path) -> None:
    _make_anima(tmp_path, "legacy", None)
    channels_dir = tmp_path / "shared" / "channels"
    channels_dir.mkdir(parents=True)
    channel_file = channels_dir / "general.jsonl"
    channel_file.write_text("", encoding="utf-8")
    handler = _make_handler(tmp_path, "legacy")

    result = handler._handle_post_channel({"channel": "general", "text": "hello"})

    assert result == t("handler.channel_post_success", channel="general")
    assert len(channel_file.read_text(encoding="utf-8").splitlines()) == 1


def test_company_scoped_open_channel_read_same_company_ok(tmp_path: Path) -> None:
    _make_anima(tmp_path, "alice", "alpha")
    channels_dir = tmp_path / "shared" / "channels"
    channels_dir.mkdir(parents=True)
    channel_file = channels_dir / "general.jsonl"
    channel_file.write_text(
        json.dumps({"ts": "2026-07-20T00:00:00+09:00", "from": "alice", "text": "secret"}) + "\n",
        encoding="utf-8",
    )
    save_channel_meta(tmp_path / "shared", "general", ChannelMeta(members=[], company="alpha"))
    handler = _make_handler(tmp_path)

    result = handler._handle_read_channel({"channel": "general"})

    assert "secret" in result


def test_company_scoped_open_channel_read_other_company_rejects(tmp_path: Path) -> None:
    _make_anima(tmp_path, "alice", "alpha")
    _write_company(tmp_path, "beta", "Beta Corporation")
    channels_dir = tmp_path / "shared" / "channels"
    channels_dir.mkdir(parents=True)
    channel_file = channels_dir / "general.jsonl"
    channel_file.write_text(
        json.dumps({"ts": "2026-07-20T00:00:00+09:00", "from": "other", "text": "secret"}) + "\n",
        encoding="utf-8",
    )
    save_channel_meta(tmp_path / "shared", "general", ChannelMeta(members=[], company="beta"))
    handler = _make_handler(tmp_path)

    result = handler._handle_read_channel({"channel": "general"})

    assert "Beta Corporation" in result
    assert "secret" not in result


def test_read_channel_rejects_unreadable_company(tmp_path: Path) -> None:
    _make_anima(tmp_path, "alice", "alpha")
    (tmp_path / "animas" / "alice" / "status.json").write_text("{broken", encoding="utf-8")
    channels_dir = tmp_path / "shared" / "channels"
    channels_dir.mkdir(parents=True)
    (channels_dir / "general.jsonl").write_text(
        json.dumps({"ts": "2026-07-20T00:00:00+09:00", "from": "other", "text": "secret"}) + "\n",
        encoding="utf-8",
    )
    handler = _make_handler(tmp_path)

    result = handler._handle_read_channel({"channel": "general"})

    assert result == t("handler.company_boundary_unverifiable")
    assert "secret" not in result


def test_read_channel_rejects_closed_channel_for_anima(tmp_path: Path) -> None:
    _make_anima(tmp_path, "alice", "alpha")
    channels_dir = tmp_path / "shared" / "channels"
    channels_dir.mkdir(parents=True)
    (channels_dir / "old-board.jsonl").write_text(
        json.dumps({"ts": "2026-07-20T00:00:00+09:00", "from": "other", "text": "secret"}) + "\n",
        encoding="utf-8",
    )
    save_channel_meta(tmp_path / "shared", "old-board", ChannelMeta(members=["alice"], closed=True))
    handler = _make_handler(tmp_path)

    result = handler._handle_read_channel({"channel": "old-board"})

    assert result == t("handler.channel_acl_denied", channel="old-board")
    assert "secret" not in result


def test_unassigned_anima_can_read_scoped_and_unattributed_open_channels(
    tmp_path: Path,
) -> None:
    _make_anima(tmp_path, "legacy", None)
    channels_dir = tmp_path / "shared" / "channels"
    channels_dir.mkdir(parents=True)
    for channel in ("scoped", "unattributed"):
        (channels_dir / f"{channel}.jsonl").write_text(
            json.dumps(
                {
                    "ts": "2026-07-20T00:00:00+09:00",
                    "from": "other",
                    "text": f"{channel}-content",
                }
            )
            + "\n",
            encoding="utf-8",
        )
    save_channel_meta(
        tmp_path / "shared",
        "scoped",
        ChannelMeta(members=[], company="alpha"),
    )
    handler = _make_handler(tmp_path, "legacy")

    assert "scoped-content" in handler._handle_read_channel({"channel": "scoped"})
    assert "unattributed-content" in handler._handle_read_channel({"channel": "unattributed"})


def test_restricted_channel_members_take_precedence_over_company(tmp_path: Path) -> None:
    _make_anima(tmp_path, "alice", "alpha")
    channels_dir = tmp_path / "shared" / "channels"
    channels_dir.mkdir(parents=True)
    (channels_dir / "team.jsonl").write_text(
        json.dumps(
            {
                "ts": "2026-07-20T00:00:00+09:00",
                "from": "alice",
                "text": "member-content",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    save_channel_meta(
        tmp_path / "shared",
        "team",
        ChannelMeta(members=["alice"], company="beta"),
    )
    handler = _make_handler(tmp_path)

    result = handler._handle_read_channel({"channel": "team"})

    assert "member-content" in result


def test_archive_channel_through_handler_moves_history_and_writes_closed_tombstone(tmp_path: Path) -> None:
    _make_anima(tmp_path, "alice", "alpha")
    channel_file = _make_channel(tmp_path, "team", ["alice"])
    handler = _make_handler(tmp_path)

    result = json.loads(handler.handle("manage_channel", {"action": "archive", "channel": "team"}))

    archived_file = tmp_path / "shared" / "channels" / "archive" / "team.jsonl"
    meta = load_channel_meta(tmp_path / "shared", "team")
    assert result["archived"] is True
    assert result["closed"] is True
    assert result["to"] == "channels/archive/team.jsonl"
    assert not channel_file.exists()
    assert archived_file.read_text(encoding="utf-8").find("secret") >= 0
    assert meta is not None
    assert meta.closed is True
    assert meta.members == ["alice"]
    assert not is_channel_member(tmp_path / "shared", "team", "alice")


def test_archive_channel_rejects_non_member_without_mutation(tmp_path: Path) -> None:
    _make_anima(tmp_path, "alice", "alpha")
    _make_anima(tmp_path, "bob", "alpha")
    channel_file = _make_channel(tmp_path, "team", ["bob"])
    handler = _make_handler(tmp_path)

    result = handler.handle("manage_channel", {"action": "archive", "channel": "team"})

    meta = load_channel_meta(tmp_path / "shared", "team")
    assert "not a member" in result.lower() or "メンバーではない" in result
    assert channel_file.exists()
    assert meta is not None and not meta.closed


def test_archive_channel_rejects_cross_company_members_without_mutation(tmp_path: Path) -> None:
    _make_anima(tmp_path, "alice", "alpha")
    _make_anima(tmp_path, "bob", "beta")
    _write_company(tmp_path, "beta", "Beta Corporation")
    channel_file = _make_channel(tmp_path, "team", ["alice", "bob"])
    handler = _make_handler(tmp_path)

    result = handler.handle("manage_channel", {"action": "archive", "channel": "team"})

    meta = load_channel_meta(tmp_path / "shared", "team")
    assert "Beta Corporation" in result
    assert channel_file.exists()
    assert meta is not None and not meta.closed


def test_create_channel_auto_assigns_creator_company(tmp_path: Path) -> None:
    _make_anima(tmp_path, "alice", "alpha")
    _make_anima(tmp_path, "same", "alpha")
    handler = _make_handler(tmp_path)

    result = handler._handle_manage_channel({"action": "create", "channel": "alpha-room", "members": ["same"]})

    assert "alpha-room" in result
    meta = load_channel_meta(tmp_path / "shared", "alpha-room")
    assert meta is not None
    assert meta.company == "alpha"
    assert "alice" in meta.members
    assert "same" in meta.members


def test_channel_meta_company_roundtrip(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    save_channel_meta(shared, "ops", ChannelMeta(members=[], company="alpha"))
    loaded = load_channel_meta(shared, "ops")
    assert loaded is not None
    assert loaded.company == "alpha"
    # Missing company field in older files defaults to empty string
    meta_path = shared / "channels" / "legacy.meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({"members": []}), encoding="utf-8")
    legacy = load_channel_meta(shared, "legacy")
    assert legacy is not None
    assert legacy.company == ""


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

    result = handler._handle_manage_channel({"action": "create", "channel": "mixed", "members": ["bob"]})

    assert "Beta Corporation" in result
    assert not (tmp_path / "shared" / "channels" / "mixed.jsonl").exists()
    assert load_channel_meta(tmp_path / "shared", "mixed") is None


def test_add_member_rejects_cross_company_member_without_mutation(tmp_path: Path) -> None:
    _make_anima(tmp_path, "alice", "alpha")
    _make_anima(tmp_path, "bob", "beta")
    _write_company(tmp_path, "beta", "Beta Corporation")
    _make_channel(tmp_path, "team", ["alice"])
    handler = _make_handler(tmp_path)

    result = handler._handle_manage_channel({"action": "add_member", "channel": "team", "members": ["bob"]})

    assert "Beta Corporation" in result
    meta = load_channel_meta(tmp_path / "shared", "team")
    assert meta is not None
    assert meta.members == ["alice"]


def test_unassigned_anima_keeps_legacy_cross_company_access(tmp_path: Path) -> None:
    _make_anima(tmp_path, "legacy", None)
    _make_anima(tmp_path, "bob", "beta")
    handler = _make_handler(tmp_path, "legacy")

    result = handler._handle_manage_channel({"action": "create", "channel": "legacy-team", "members": ["bob"]})

    assert "legacy-team" in result
    meta = load_channel_meta(tmp_path / "shared", "legacy-team")
    assert meta is not None
    assert meta.members == ["legacy", "bob"]


def test_mention_fanout_only_notifies_same_company_targets(tmp_path: Path) -> None:
    _make_anima(tmp_path, "alice", "alpha")
    _make_anima(tmp_path, "same", "alpha")
    _make_anima(tmp_path, "other", "beta")
    channels_dir = tmp_path / "shared" / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    (channels_dir / "team.jsonl").write_text("", encoding="utf-8")
    save_channel_meta(tmp_path / "shared", "team", ChannelMeta(members=[], company="alpha"))
    handler = _make_handler(tmp_path)

    handler._handle_post_channel({"channel": "team", "text": "@same @other please review"})

    inbox = list((tmp_path / "shared" / "inbox" / "same").glob("*.json"))
    assert len(inbox) == 1
    message = json.loads(inbox[0].read_text(encoding="utf-8"))
    assert message["type"] == "board_mention"
    assert message["meta"]["board_channel"] == "team"
    assert not (tmp_path / "shared" / "inbox" / "other").exists()


def test_explicitly_closed_empty_channel_denies_animas(tmp_path: Path) -> None:
    shared_dir = tmp_path / "shared"
    save_channel_meta(shared_dir, "old-board", ChannelMeta(members=[], closed=True))

    assert not is_channel_member(shared_dir, "old-board", "alice")
    assert is_channel_member(shared_dir, "old-board", "alice", source="human")
    assert is_channel_member(shared_dir, "legacy-open", "alice")
