from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.hub.avatar import save_avatar
from core.hub.members import HubMember, save_member, token_digest
from server.routes.hub import create_hub_router


def _app(shared: Path) -> FastAPI:
    app = FastAPI()
    app.state.shared_dir = shared
    app.include_router(create_hub_router(), prefix="/api")
    return app


def _config(monkeypatch: pytest.MonkeyPatch, *, enabled: bool = True) -> None:
    config = SimpleNamespace(hub=SimpleNamespace(enabled=enabled))
    monkeypatch.setattr("server.routes.hub.load_config", lambda: config)


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
    assert set(body) == {"id", "display_name", "enabled", "avatar_url"}
