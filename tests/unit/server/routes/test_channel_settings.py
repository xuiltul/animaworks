"""Unit tests for ``PUT /api/channels/{name}/settings``."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from core.auth.models import AuthUser
from core.i18n import t


def _make_test_app(shared_dir: Path, user: AuthUser | None = None):
    from fastapi import FastAPI

    from server.routes.channels import create_channels_router

    app = FastAPI()
    app.state.shared_dir = shared_dir
    app.state.ws_manager = MagicMock()
    app.state.ws_manager.broadcast = AsyncMock()

    if user is not None:

        @app.middleware("http")
        async def _set_user(request, call_next):  # type: ignore[no-untyped-def]
            request.state.user = user
            return await call_next(request)

    app.include_router(create_channels_router(), prefix="/api")
    return app


def _write_channel(shared_dir: Path, name: str, entries: list[dict] | None = None) -> Path:
    channels_dir = shared_dir / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    filepath = channels_dir / f"{name}.jsonl"
    lines = [json.dumps(entry, ensure_ascii=False) for entry in (entries or [])]
    filepath.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return filepath


def _write_meta(shared_dir: Path, name: str, **kwargs: object):
    from core.messaging.messenger import ChannelMeta, save_channel_meta

    meta = ChannelMeta(**kwargs)  # type: ignore[arg-type]
    save_channel_meta(shared_dir, name, meta)
    return meta


def _read_posts(shared_dir: Path, name: str) -> list[dict]:
    path = shared_dir / "channels" / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_meta(shared_dir: Path, name: str):
    from core.messaging.messenger import load_channel_meta

    return load_channel_meta(shared_dir, name)


async def _put(app, name: str, payload: dict):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.put(f"/api/channels/{name}/settings", json=payload)


async def _get(app, path: str):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path)


@pytest.fixture
def shared_dir(data_dir: Path) -> Path:
    shared = data_dir / "shared"
    (shared / "channels").mkdir(parents=True, exist_ok=True)
    for anima in ("sakura", "mio"):
        (data_dir / "animas" / anima).mkdir(parents=True, exist_ok=True)
    return shared


OWNER = AuthUser(username="owner", role="owner")
ADMIN = AuthUser(username="admin", role="admin")
USER = AuthUser(username="alice", role="user")


class TestSettingsPermissions:
    async def test_owner_can_change_settings(self, shared_dir: Path):
        _write_channel(shared_dir, "general")
        app = _make_test_app(shared_dir, OWNER)

        resp = await _put(app, "general", {"description": "hello"})

        assert resp.status_code == 200
        assert resp.json()["description"] == "hello"
        assert _read_meta(shared_dir, "general").description == "hello"

    async def test_admin_can_change_settings(self, shared_dir: Path):
        _write_channel(shared_dir, "general")
        app = _make_test_app(shared_dir, ADMIN)

        resp = await _put(app, "general", {"description": "hello"})

        assert resp.status_code == 200
        assert _read_meta(shared_dir, "general").description == "hello"

    async def test_user_is_forbidden(self, shared_dir: Path):
        _write_channel(shared_dir, "general")
        app = _make_test_app(shared_dir, USER)

        resp = await _put(app, "general", {"description": "hello"})

        assert resp.status_code == 403
        assert resp.json()["detail"] == t("server.channel_settings_forbidden")
        assert _read_meta(shared_dir, "general") is None
        assert _read_posts(shared_dir, "general") == []

    async def test_unauthenticated_is_forbidden(self, shared_dir: Path):
        _write_channel(shared_dir, "general")
        app = _make_test_app(shared_dir, None)

        resp = await _put(app, "general", {"description": "hello"})

        assert resp.status_code == 403
        assert _read_meta(shared_dir, "general") is None

    async def test_unknown_board_is_404(self, shared_dir: Path):
        app = _make_test_app(shared_dir, OWNER)

        resp = await _put(app, "missing", {"description": "hello"})

        assert resp.status_code == 404

    async def test_invalid_name_is_400(self, shared_dir: Path):
        app = _make_test_app(shared_dir, OWNER)

        resp = await _put(app, "Bad Name", {"description": "hello"})

        assert resp.status_code == 400


class TestAccessChanges:
    async def test_open_to_members_records_operator_post(self, shared_dir: Path):
        _write_channel(shared_dir, "general")
        app = _make_test_app(shared_dir, OWNER)

        resp = await _put(app, "general", {"access": "members", "members": ["sakura"]})

        assert resp.status_code == 200
        data = resp.json()
        assert data["access"] == "members"
        assert data["members"] == ["sakura"]
        assert data["writable"] is True

        meta = _read_meta(shared_dir, "general")
        assert meta is not None
        assert meta.members == ["sakura"]
        assert meta.closed is False

        posts = _read_posts(shared_dir, "general")
        assert len(posts) == 1
        record = posts[0]
        assert record["kind"] == "operator"
        assert record["source"] == "human"
        assert record["from"] == "owner"
        assert "@" not in record["text"]
        expected = "\n".join(
            [
                t("server.channel_settings_members_added", names="sakura"),
                t("server.channel_settings_access_members"),
            ]
        )
        assert record["text"] == expected

    async def test_members_to_open_clears_members(self, shared_dir: Path):
        _write_channel(shared_dir, "general")
        _write_meta(shared_dir, "general", members=["sakura", "mio"])
        app = _make_test_app(shared_dir, OWNER)

        resp = await _put(app, "general", {"access": "open"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["access"] == "open"
        assert data["members"] == []

        meta = _read_meta(shared_dir, "general")
        assert meta is not None
        assert meta.members == []
        assert meta.closed is False

        posts = _read_posts(shared_dir, "general")
        assert len(posts) == 1
        assert "@" not in posts[0]["text"]
        expected = "\n".join(
            [
                t("server.channel_settings_members_removed", names="sakura, mio"),
                t("server.channel_settings_access_open"),
            ]
        )
        assert posts[0]["text"] == expected

    async def test_close_keeps_members_and_records_before_closing(self, shared_dir: Path):
        from core.messaging.board import register_post_hook, unregister_post_hook

        _write_channel(shared_dir, "general")
        _write_meta(shared_dir, "general", members=["sakura"])
        app = _make_test_app(shared_dir, OWNER)
        seen_closed: list[bool] = []

        def hook(post, result):  # type: ignore[no-untyped-def]
            meta = _read_meta(shared_dir, post.channel)
            seen_closed.append(bool(meta.closed) if meta is not None else False)
            return "skipped"

        register_post_hook("test-settings-close-order", hook)
        try:
            resp = await _put(app, "general", {"access": "closed"})
        finally:
            unregister_post_hook("test-settings-close-order")

        assert resp.status_code == 200
        data = resp.json()
        assert data["access"] == "closed"
        assert data["members"] == ["sakura"]
        assert data["writable"] is False

        meta = _read_meta(shared_dir, "general")
        assert meta is not None
        assert meta.closed is True
        assert meta.members == ["sakura"]
        assert seen_closed == [False]
        posts = _read_posts(shared_dir, "general")
        assert len(posts) == 1
        assert t("server.channel_settings_access_closed") in posts[0]["text"]

    async def test_open_records_after_opening(self, shared_dir: Path):
        from core.messaging.board import register_post_hook, unregister_post_hook

        _write_channel(shared_dir, "general")
        _write_meta(shared_dir, "general", members=["sakura"], closed=True)
        app = _make_test_app(shared_dir, OWNER)
        seen_closed: list[bool] = []

        def hook(post, result):  # type: ignore[no-untyped-def]
            meta = _read_meta(shared_dir, post.channel)
            seen_closed.append(bool(meta.closed) if meta is not None else False)
            return "skipped"

        register_post_hook("test-settings-open-order", hook)
        try:
            resp = await _put(app, "general", {"access": "open"})
        finally:
            unregister_post_hook("test-settings-open-order")

        assert resp.status_code == 200
        assert resp.json()["access"] == "open"
        assert _read_meta(shared_dir, "general").closed is False
        assert seen_closed == [False]
        assert len(_read_posts(shared_dir, "general")) == 1


class TestMemberValidation:
    async def test_members_access_requires_members(self, shared_dir: Path):
        _write_channel(shared_dir, "general")
        app = _make_test_app(shared_dir, OWNER)

        resp = await _put(app, "general", {"access": "members"})

        assert resp.status_code == 400
        assert resp.json()["detail"] == t("server.channel_settings_members_required")
        assert _read_meta(shared_dir, "general") is None
        assert _read_posts(shared_dir, "general") == []

    async def test_members_access_rejects_empty_list(self, shared_dir: Path):
        _write_channel(shared_dir, "general")
        app = _make_test_app(shared_dir, OWNER)

        resp = await _put(app, "general", {"access": "members", "members": []})

        assert resp.status_code == 400
        assert resp.json()["detail"] == t("server.channel_settings_members_required")

    async def test_unknown_anima_is_rejected(self, shared_dir: Path):
        _write_channel(shared_dir, "general")
        app = _make_test_app(shared_dir, OWNER)

        resp = await _put(app, "general", {"access": "members", "members": ["ghost"]})

        assert resp.status_code == 400
        assert resp.json()["detail"] == t("server.channel_settings_member_not_found", name="ghost")
        assert _read_meta(shared_dir, "general") is None

    async def test_path_like_member_is_rejected(self, shared_dir: Path):
        _write_channel(shared_dir, "general")
        app = _make_test_app(shared_dir, OWNER)

        resp = await _put(app, "general", {"access": "members", "members": ["../sandbox"]})

        assert resp.status_code == 400


class TestDescriptionValidation:
    async def test_500_chars_are_accepted(self, shared_dir: Path):
        _write_channel(shared_dir, "general")
        app = _make_test_app(shared_dir, OWNER)
        description = "a" * 500

        resp = await _put(app, "general", {"description": description})

        assert resp.status_code == 200
        assert resp.json()["description"] == description

    async def test_501_chars_are_rejected(self, shared_dir: Path):
        _write_channel(shared_dir, "general")
        app = _make_test_app(shared_dir, OWNER)

        resp = await _put(app, "general", {"description": "a" * 501})

        assert resp.status_code == 400
        assert resp.json()["detail"] == t("server.channel_settings_description_too_long")
        assert _read_meta(shared_dir, "general") is None


class TestMetadataPreservation:
    async def test_other_meta_fields_are_preserved(self, shared_dir: Path):
        _write_channel(shared_dir, "general")
        _write_meta(
            shared_dir,
            "general",
            members=["sakura"],
            created_by="alice",
            created_at="2026-01-01T00:00:00",
            description="old",
            closed=False,
            slack_sync_disabled=True,
            slack_deleted_at="2026-02-01T00:00:00",
            company="company-a",
        )
        app = _make_test_app(shared_dir, OWNER)

        resp = await _put(app, "general", {"description": "new", "access": "members"})

        assert resp.status_code == 200
        meta = _read_meta(shared_dir, "general")
        assert meta is not None
        assert meta.description == "new"
        assert meta.members == ["sakura"]
        assert meta.closed is False
        assert meta.created_by == "alice"
        assert meta.created_at == "2026-01-01T00:00:00"
        assert meta.slack_sync_disabled is True
        assert meta.slack_deleted_at == "2026-02-01T00:00:00"
        assert meta.company == "company-a"

    async def test_missing_meta_is_created(self, shared_dir: Path):
        _write_channel(shared_dir, "general")
        app = _make_test_app(shared_dir, OWNER)

        resp = await _put(app, "general", {"access": "members", "members": ["mio"]})

        assert resp.status_code == 200
        meta = _read_meta(shared_dir, "general")
        assert meta is not None
        assert meta.members == ["mio"]
        assert meta.closed is False
        assert meta.created_by == "owner"
        assert meta.created_at

    async def test_no_change_does_not_record(self, shared_dir: Path):
        _write_channel(shared_dir, "general", [{"ts": "2026-02-17T10:00:00", "from": "sakura", "text": "hi"}])
        _write_meta(shared_dir, "general", members=["sakura"], description="old")
        app = _make_test_app(shared_dir, OWNER)

        resp = await _put(
            app,
            "general",
            {"access": "members", "members": ["sakura"], "description": "old"},
        )

        assert resp.status_code == 200
        assert len(_read_posts(shared_dir, "general")) == 1
        assert _read_posts(shared_dir, "general")[0]["text"] == "hi"


class TestResponseShape:
    async def test_response_matches_list_entry(self, shared_dir: Path):
        _write_channel(shared_dir, "general")
        app = _make_test_app(shared_dir, OWNER)

        put_resp = await _put(app, "general", {"access": "members", "members": ["sakura"]})
        list_resp = await _get(app, "/api/channels")

        assert put_resp.status_code == 200
        assert list_resp.status_code == 200
        entry = next(channel for channel in list_resp.json() if channel["name"] == "general")
        assert put_resp.json() == entry
