"""Tests for the Board Web UI foundation fields and filters."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

from httpx import ASGITransport, AsyncClient

JST = ZoneInfo("Asia/Tokyo")


def _make_test_app(shared_dir: Path):
    from fastapi import FastAPI

    from server.routes.channels import create_channels_router

    app = FastAPI()
    app.state.shared_dir = shared_dir
    router = create_channels_router()
    app.include_router(router, prefix="/api")
    return app


def _write_channel(shared_dir: Path, name: str, entries: list[dict]) -> Path:
    channels_dir = shared_dir / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    filepath = channels_dir / f"{name}.jsonl"
    lines = [json.dumps(entry, ensure_ascii=False) for entry in entries]
    filepath.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return filepath


def _write_meta(
    shared_dir: Path,
    name: str,
    *,
    members: list[str] | None = None,
    closed: bool = False,
    company: str = "",
    description: str = "",
) -> None:
    from core.messaging.messenger import ChannelMeta, save_channel_meta

    save_channel_meta(
        shared_dir,
        name,
        ChannelMeta(
            members=members or [],
            closed=closed,
            company=company,
            description=description,
        ),
    )


def _slack_config(
    *,
    enabled: bool = True,
    mapping: dict[str, str] | None = None,
    whitelist: list[str] | None = None,
    sync_all: bool = False,
) -> SimpleNamespace:
    slack = SimpleNamespace(
        enabled=enabled,
        board_mapping=mapping or {},
        board_outbound_sync=whitelist or [],
        board_outbound_sync_all=sync_all,
    )
    return SimpleNamespace(external_messaging=SimpleNamespace(slack=slack))


def _ts(days_ago: int = 0, hour: int = 12, minute: int = 0) -> str:
    day = datetime.now(JST).date() - timedelta(days=days_ago)
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=JST).isoformat()


# ── GET /api/channels: new fields ────────────────────────


class TestListChannelsFoundationFields:
    async def test_open_channel_always_returns_new_fields(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        _write_channel(
            shared_dir,
            "general",
            [
                {"ts": "2026-02-17T10:00:00", "from": "sakura", "text": "Hello", "source": "anima"},
                {"ts": "2026-02-17T11:00:00", "from": "owner", "text": "Hi", "source": "human"},
            ],
        )

        app = _make_test_app(shared_dir)
        with patch("core.config.models.load_config", return_value=_slack_config(enabled=False)):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get("/api/channels")

        assert resp.status_code == 200
        channel = resp.json()[0]
        assert channel["name"] == "general"
        assert channel["message_count"] == 2
        assert channel["last_post_ts"] == "2026-02-17T11:00:00"
        assert channel["last_from"] == "owner"
        assert channel["access"] == "open"
        assert channel["writable"] is True
        assert channel["company"] == ""
        assert channel["members"] == []
        assert channel["description"] == ""
        assert channel["slack"] == {"inbound": False, "outbound": False}
        assert channel["activity_14d"] == [0] * 14

    async def test_access_kinds_and_closed_board_is_not_writable(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        for name in ("general", "team-a", "archive"):
            _write_channel(shared_dir, name, [{"ts": "2026-02-17T10:00:00", "from": "sakura", "source": "anima"}])
        _write_meta(shared_dir, "team-a", members=["sakura", "mio"])
        _write_meta(shared_dir, "archive", members=["sakura"], closed=True)

        app = _make_test_app(shared_dir)
        with patch("core.config.models.load_config", return_value=_slack_config(enabled=False)):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get("/api/channels")

        channels = {channel["name"]: channel for channel in resp.json()}
        assert channels["general"]["access"] == "open"
        assert channels["team-a"]["access"] == "members"
        assert channels["archive"]["access"] == "closed"
        assert channels["general"]["writable"] is True
        assert channels["team-a"]["writable"] is True
        assert channels["archive"]["writable"] is False
        assert channels["archive"]["members"] == ["sakura"]

    async def test_company_and_description_are_always_strings(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        _write_channel(shared_dir, "ops", [])
        _write_meta(shared_dir, "ops", company="example", description="Operations board")

        app = _make_test_app(shared_dir)
        with patch("core.config.models.load_config", return_value=_slack_config(enabled=False)):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get("/api/channels")

        channel = resp.json()[0]
        assert channel["company"] == "example"
        assert channel["description"] == "Operations board"
        assert channel["members"] == []

    async def test_slack_inbound_and_outbound_flags(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        for name in ("general", "ops"):
            _write_channel(shared_dir, name, [])

        config = _slack_config(
            mapping={"C1": "general", "C2": "ops"},
            whitelist=["general"],
        )
        app = _make_test_app(shared_dir)
        with patch("core.config.models.load_config", return_value=config):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get("/api/channels")

        channels = {channel["name"]: channel for channel in resp.json()}
        assert channels["general"]["slack"] == {"inbound": True, "outbound": True}
        assert channels["ops"]["slack"] == {"inbound": True, "outbound": False}

    async def test_slack_sync_all_enables_outbound(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        _write_channel(shared_dir, "ops", [])

        config = _slack_config(mapping={"C1": "ops"}, whitelist=[], sync_all=True)
        app = _make_test_app(shared_dir)
        with patch("core.config.models.load_config", return_value=config):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get("/api/channels")

        assert resp.json()[0]["slack"] == {"inbound": True, "outbound": True}

    async def test_slack_flags_are_false_when_config_is_unreadable(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        _write_channel(shared_dir, "general", [])

        app = _make_test_app(shared_dir)
        with patch("core.config.models.load_config", side_effect=RuntimeError("no config")):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get("/api/channels")

        assert resp.json()[0]["slack"] == {"inbound": False, "outbound": False}

    async def test_activity_14d_counts_jst_days(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        # 00:30 JST today expressed as the previous UTC calendar day.
        today_boundary = datetime.now(JST).replace(hour=0, minute=30, second=0, microsecond=0)
        entries = [
            {"ts": _ts(days_ago=14), "from": "sakura", "source": "anima"},
            {"ts": _ts(days_ago=13), "from": "sakura", "source": "anima"},
            {"ts": _ts(days_ago=13, hour=13), "from": "sakura", "source": "anima"},
            {"ts": _ts(days_ago=13, hour=14), "from": "sakura", "source": "anima"},
            {"ts": _ts(days_ago=1), "from": "sakura", "source": "anima"},
            {"ts": today_boundary.astimezone(ZoneInfo("UTC")).isoformat(), "from": "sakura", "source": "anima"},
            {"ts": _ts(days_ago=0, hour=12), "from": "owner", "source": "human"},
        ]
        _write_channel(shared_dir, "general", entries)

        app = _make_test_app(shared_dir)
        with patch("core.config.models.load_config", return_value=_slack_config(enabled=False)):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get("/api/channels")

        activity = resp.json()[0]["activity_14d"]
        assert len(activity) == 14
        # Entries 14 days ago fall outside the window; the UTC timestamp is
        # counted on today's JST date.
        assert activity == [3] + [0] * 11 + [1, 2]


# ── GET /api/channels/{name}: kinds and filters ──────────


class TestGetChannelFoundation:
    async def test_infers_kind_from_source_without_rewriting_file(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        entries = [
            {"ts": "2026-02-17T10:00:00", "from": "sakura", "text": "a", "source": "anima"},
            {"ts": "2026-02-17T10:01:00", "from": "owner", "text": "b", "source": "human"},
            {"ts": "2026-02-17T10:02:00", "from": "slack-user", "text": "c", "source": "slack"},
            {"ts": "2026-02-17T10:03:00", "from": "discord-user", "text": "d", "source": "discord"},
            {"ts": "2026-02-17T10:04:00", "from": "system", "text": "e", "source": "system"},
            {"ts": "2026-02-17T10:05:00", "from": "bot", "text": "f", "source": "slack", "kind": "external_bot"},
        ]
        filepath = _write_channel(shared_dir, "general", entries)
        original = filepath.read_text(encoding="utf-8")

        app = _make_test_app(shared_dir)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/channels/general?limit=50")

        messages = resp.json()["messages"]
        assert [message["kind"] for message in messages] == [
            "anima",
            "human",
            "external_user",
            "external_user",
            "operator",
            "external_bot",
        ]
        assert [message.get("kind_inferred") for message in messages] == [True, True, True, True, True, None]
        assert filepath.read_text(encoding="utf-8") == original

    async def test_q_matches_text_and_from_case_insensitively(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        entries = [
            {"ts": "2026-02-17T10:00:00", "from": "alice", "text": "Hello World", "source": "human"},
            {"ts": "2026-02-17T10:01:00", "from": "bob", "text": "hello there", "source": "human"},
            {"ts": "2026-02-17T10:02:00", "from": "carol", "text": "Other", "source": "human"},
        ]
        _write_channel(shared_dir, "general", entries)

        app = _make_test_app(shared_dir)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/channels/general?q=HELLO")
            by_sender = await client.get("/api/channels/general?q=ALICE")
            paged = await client.get("/api/channels/general?q=hello&limit=1&offset=0")

        data = resp.json()
        assert data["total"] == 2
        assert [message["text"] for message in data["messages"]] == ["Hello World", "hello there"]
        assert by_sender.json()["total"] == 1
        assert by_sender.json()["messages"][0]["from"] == "alice"
        assert paged.json()["total"] == 2
        assert [message["text"] for message in paged.json()["messages"]] == ["hello there"]
        assert paged.json()["has_more"] is True

    async def test_kind_filter_covers_external_both_kinds(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        entries = [
            {"ts": "2026-02-17T10:00:00", "from": "sakura", "text": "a", "kind": "anima"},
            {"ts": "2026-02-17T10:01:00", "from": "owner", "text": "b", "kind": "human"},
            {"ts": "2026-02-17T10:02:00", "from": "slack-user", "text": "c", "kind": "external_user"},
            {"ts": "2026-02-17T10:03:00", "from": "bot", "text": "d", "kind": "external_bot"},
            {"ts": "2026-02-17T10:04:00", "from": "system", "text": "e", "kind": "operator"},
            {"ts": "2026-02-17T10:05:00", "from": "legacy-slack", "text": "f", "source": "slack"},
        ]
        _write_channel(shared_dir, "general", entries)

        app = _make_test_app(shared_dir)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            external = await client.get("/api/channels/general?kind=external")
            anima = await client.get("/api/channels/general?kind=anima")
            combined = await client.get("/api/channels/general?kind=external&q=slack")
            invalid = await client.get("/api/channels/general?kind=unknown")

        assert [message["text"] for message in external.json()["messages"]] == ["c", "d", "f"]
        assert external.json()["total"] == 3
        assert [message["text"] for message in anima.json()["messages"]] == ["a"]
        assert [message["text"] for message in combined.json()["messages"]] == ["c", "f"]
        assert invalid.status_code == 400

    async def test_around_returns_centered_page_with_direction_flags(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        entries = [
            {"ts": f"2026-02-17T10:{index:02d}:00", "from": "sakura", "text": f"m{index}", "id": f"m{index}"}
            for index in range(10)
        ]
        _write_channel(shared_dir, "general", entries)

        app = _make_test_app(shared_dir)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            middle = await client.get("/api/channels/general?around=m5&limit=5")
            first = await client.get("/api/channels/general?around=m0&limit=3")
            last = await client.get("/api/channels/general?around=m9&limit=5")
            missing = await client.get("/api/channels/general?around=missing&limit=5")

        middle_data = middle.json()
        assert [message["id"] for message in middle_data["messages"]] == ["m3", "m4", "m5", "m6", "m7"]
        assert middle_data["total"] == 10
        assert middle_data["has_more"] is True
        assert middle_data["has_newer"] is True

        first_data = first.json()
        assert [message["id"] for message in first_data["messages"]] == ["m0", "m1", "m2"]
        assert first_data["has_more"] is False
        assert first_data["has_newer"] is True

        last_data = last.json()
        assert [message["id"] for message in last_data["messages"]] == ["m5", "m6", "m7", "m8", "m9"]
        assert last_data["has_more"] is True
        assert last_data["has_newer"] is False
        assert missing.status_code == 404

    async def test_since_still_returns_only_newer_messages(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        entries = [
            {"ts": "2026-02-17T10:00:00", "from": "sakura", "text": "m0", "source": "anima"},
            {"ts": "2026-02-17T10:01:00", "from": "sakura", "text": "m1", "source": "anima"},
            {"ts": "2026-02-17T10:02:00", "from": "owner", "text": "m2", "source": "human"},
        ]
        _write_channel(shared_dir, "general", entries)

        app = _make_test_app(shared_dir)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/channels/general?since=2026-02-17T10:01:00")

        data = resp.json()
        assert data["incremental"] is True
        assert [message["text"] for message in data["messages"]] == ["m2"]
        assert data["messages"][0]["kind"] == "human"
        assert data["messages"][0]["kind_inferred"] is True


# ── i18n parity for the new board strings ────────────────


def test_board_foundation_i18n_keys_exist_in_all_locales():
    i18n_dir = Path(__file__).resolve().parents[4] / "server" / "static" / "i18n"
    keys = (
        "board.channels",
        "board.dms",
        "board.closed_notice",
        "board.kind_human",
        "board.kind_external_user",
        "board.kind_external_bot",
        "board.kind_operator",
    )
    for locale in ("ja", "en", "ko"):
        data = json.loads((i18n_dir / f"{locale}.json").read_text(encoding="utf-8"))
        for key in keys:
            assert data.get(key), f"Missing or empty i18n key {key!r} in {locale}.json"


async def test_list_channels_reports_human_unread_count(tmp_path: Path) -> None:
    shared_dir = tmp_path / "shared"
    _write_channel(
        shared_dir,
        "general",
        [
            {"ts": _ts(0, 9), "from": "alice", "text": "one", "id": "a1", "kind": "anima"},
            {"ts": _ts(0, 10), "from": "bob", "text": "two", "id": "b2", "kind": "anima"},
            {"ts": _ts(0, 11), "from": "alice", "text": "three", "id": "c3", "kind": "anima"},
        ],
    )
    app = _make_test_app(shared_dir)
    with patch("core.config.models.load_config", side_effect=RuntimeError("no config")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            before = (await client.get("/api/channels")).json()
            assert before[0]["unread"] == 3
            resp = await client.post("/api/channels/general/read", json={"upto_id": "b2"})
            assert resp.status_code == 200
            after = (await client.get("/api/channels")).json()
    assert after[0]["unread"] == 1
