from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.hub.avatar import save_avatar
from core.hub.members import HubMember, member_path, save_member, token_digest, touch_last_seen
from server.routes.channels import create_channels_router
from server.routes.hub import create_hub_router


def _app(shared: Path) -> FastAPI:
    app = FastAPI()
    app.state.shared_dir = shared
    app.include_router(create_hub_router(), prefix="/api")
    return app


def _config(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool = True,
    board: str = "hub",
    company: str = "a",
    timeline_enabled: bool = True,
) -> None:
    config = SimpleNamespace(
        hub=SimpleNamespace(
            enabled=enabled,
            board=board,
            company=company,
            timeline_enabled=timeline_enabled,
        )
    )
    monkeypatch.setattr("server.routes.hub.load_config", lambda: config)


def _app_with_channels(shared: Path) -> FastAPI:
    app = FastAPI()
    app.state.shared_dir = shared
    app.include_router(create_hub_router(), prefix="/api")
    app.include_router(create_channels_router(), prefix="/api")
    return app


def _write_channel(shared: Path, name: str, posts: list[dict[str, str]]) -> None:
    path = shared / "channels" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(post) for post in posts) + "\n", encoding="utf-8")


def _member(member_id: str = "muse", **overrides: object) -> HubMember:
    data: dict[str, object] = {
        "id": member_id,
        "display_name": "Muse",
        "company": "a",
        "token_sha256": token_digest("secret-digest"),
        "message_route": {"type": "poll", "url_ref": "hub.url"},
    }
    data.update(overrides)
    return HubMember(**data)  # type: ignore[arg-type]


def test_members_endpoint_returns_avatar_url_without_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch)
    shared = tmp_path / "shared"
    save_member(shared, _member())
    save_avatar(shared, "muse", b"png")

    response = TestClient(_app(shared)).get("/api/hub/members")

    assert response.status_code == 200
    assert len(response.json()) == 1
    body = response.json()[0]
    assert body["id"] == "muse"
    assert body["display_name"] == "Muse"
    assert body["enabled"] is True
    assert body["avatar_url"].startswith("/api/hub/members/muse/avatar?v=")
    assert "secret-digest" not in response.text
    assert "hub.url" not in response.text


def test_members_endpoint_is_empty_when_hub_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch, enabled=False)
    shared = tmp_path / "shared"
    save_member(shared, _member())

    assert TestClient(_app(shared)).get("/api/hub/members").json() == []


def test_avatar_endpoint_serves_png_and_404s(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch)
    shared = tmp_path / "shared"
    save_member(shared, _member())
    save_avatar(shared, "muse", b"png")
    client = TestClient(_app(shared))

    response = client.get("/api/hub/members/muse/avatar")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == b"png"

    assert client.get("/api/hub/members/missing/avatar").status_code == 404
    assert client.get("/api/hub/members/../avatar").status_code == 404


def test_avatar_endpoint_404_when_hub_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch, enabled=False)
    shared = tmp_path / "shared"
    save_avatar(shared, "muse", b"png")

    assert TestClient(_app(shared)).get("/api/hub/members/muse/avatar").status_code == 404


def test_members_endpoint_returns_null_for_missing_avatar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch)
    shared = tmp_path / "shared"
    save_member(shared, _member())

    body = TestClient(_app(shared)).get("/api/hub/members").json()[0]
    assert body["avatar_url"] is None
    assert set(body) == {
        "id",
        "display_name",
        "kind",
        "enabled",
        "avatar_url",
        "route_type",
        "expect_every",
        "last_seen",
        "overdue",
    }


def test_info_is_disabled_when_hub_is_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch, enabled=False)

    response = TestClient(_app(tmp_path / "shared")).get("/api/hub/info")

    assert response.status_code == 200
    assert response.json() == {"enabled": False}


def test_info_is_disabled_when_config_cannot_be_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "server.routes.hub.load_config",
        lambda: (_ for _ in ()).throw(RuntimeError("broken config")),
    )

    assert TestClient(_app(tmp_path / "shared")).get("/api/hub/info").json() == {"enabled": False}


def test_info_returns_hub_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch, board="company", company="acme", timeline_enabled=True)

    body = TestClient(_app(tmp_path / "shared")).get("/api/hub/info").json()

    assert body == {
        "enabled": True,
        "board": "company",
        "company": "acme",
        "timeline_enabled": True,
        "unread": 0,
    }


def test_info_unread_matches_channels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch, board="hub")
    shared = tmp_path / "shared"
    _write_channel(
        shared,
        "hub",
        [
            {"id": "p1", "ts": "2026-01-01T00:00:00+00:00", "from": "muse", "text": "hello"},
            {"id": "p2", "ts": "2026-01-01T00:01:00+00:00", "from": "grok", "text": "hi"},
        ],
    )
    client = TestClient(_app_with_channels(shared))

    channels = client.get("/api/channels").json()
    info = client.get("/api/hub/info").json()

    assert len(channels) == 1
    assert channels[0]["name"] == "hub"
    assert info["unread"] == channels[0]["unread"] == 2


def test_members_endpoint_exposes_schedule_metadata_without_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config(monkeypatch)
    shared = tmp_path / "shared"
    save_member(
        shared,
        _member(
            expect_every="1h",
            message_route={"type": "webhook", "url_ref": "hub.url", "key_ref": "hub.key"},
        ),
    )
    touch_last_seen(shared, "muse")

    response = TestClient(_app(shared)).get("/api/hub/members")

    body = response.json()[0]
    assert body["kind"] == "vendor_bot"
    assert body["route_type"] == "webhook"
    assert body["expect_every"] == "1h"
    assert isinstance(body["last_seen"], str)
    assert body["overdue"] is False
    assert "secret-digest" not in response.text
    assert "hub.url" not in response.text
    assert "hub.key" not in response.text


def test_members_endpoint_marks_overdue_members(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch)
    shared = tmp_path / "shared"
    save_member(shared, _member("late", expect_every="1m"))
    save_member(shared, _member("fresh", expect_every="1h"))
    old = 1_700_000_000
    os.utime(member_path(shared, "late"), (old, old))

    bodies = {body["id"]: body for body in TestClient(_app(shared)).get("/api/hub/members").json()}

    assert bodies["late"]["overdue"] is True
    assert bodies["late"]["last_seen"] is None
    assert bodies["fresh"]["overdue"] is False


def test_members_endpoint_disabled_member_is_never_overdue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch)
    shared = tmp_path / "shared"
    save_member(shared, _member("sleeping", enabled=False, expect_every="1m"))
    old = 1_700_000_000
    os.utime(member_path(shared, "sleeping"), (old, old))

    body = TestClient(_app(shared)).get("/api/hub/members").json()[0]

    assert body["overdue"] is False
