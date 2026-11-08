# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from core.config.schemas import HubConfig
from core.hub import timeline
from core.hub.timeline import append_item
from core.time_utils import now_local
from server.internal_auth import InternalCaller


def _app(monkeypatch, tmp_path: Path, caller: InternalCaller | None) -> tuple[TestClient, Path]:
    def test_require_internal_caller(_request: Request):
        return caller

    monkeypatch.setattr("server.internal_auth.require_internal_caller", test_require_internal_caller)
    shared_dir = tmp_path / "shared"
    animas_dir = tmp_path / "animas"
    shared_dir.mkdir(parents=True)
    animas_dir.mkdir(parents=True)

    from server.routes.timeline import create_timeline_router

    app = FastAPI()
    app.include_router(create_timeline_router(), prefix="/api")
    app.state.shared_dir = shared_dir
    app.state.animas_dir = animas_dir

    @app.middleware("http")
    async def set_test_state(request, call_next):
        request.state.internal_caller = caller
        request.state.user = SimpleNamespace(username="alice")
        return await call_next(request)

    return TestClient(app), shared_dir


def _set_hub(monkeypatch, **kwargs) -> None:
    monkeypatch.setattr(
        "core.config.models.load_config",
        lambda: SimpleNamespace(hub=HubConfig(**kwargs)),
    )


def _set_company(animas_dir: Path, name: str, company: str) -> None:
    path = animas_dir / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "status.json").write_text(json.dumps({"company": company}), encoding="utf-8")


def test_public_timeline_disabled(monkeypatch, tmp_path: Path) -> None:
    client, _shared_dir = _app(monkeypatch, tmp_path, None)
    _set_hub(monkeypatch)

    response = client.get("/api/timeline")

    assert response.status_code == 200
    assert response.json() == {"enabled": False, "items": [], "unread": 0}


def test_public_timeline_and_read_marker(monkeypatch, tmp_path: Path) -> None:
    client, shared_dir = _app(monkeypatch, tmp_path, None)
    _set_hub(monkeypatch, timeline_enabled=True, company="a")
    month = now_local().strftime("%Y-%m")
    monkeypatch.setattr(timeline, "now_iso", lambda: f"{month}-06T10:00:00+09:00")
    first = append_item(
        shared_dir,
        from_name="alice",
        company="a",
        kind="notified",
        text="Started",
        source="anima",
        priority="normal",
        delivered=True,
    )
    monkeypatch.setattr(timeline, "now_iso", lambda: f"{month}-06T11:00:00+09:00")
    second = append_item(
        shared_dir,
        from_name="alice",
        company="a",
        kind="done",
        text="Finished",
        source="anima",
    )

    response = client.get("/api/timeline")
    data = response.json()
    assert response.status_code == 200
    assert data["enabled"] is True
    assert data["company"] == "a"
    assert data["reader"] == {"upto_id": "", "upto_ts": ""}
    assert data["unread"] == 2
    assert [item["id"] for item in data["items"]] == [second["id"], first["id"]]
    assert data["items"][0]["read"] is False

    read_response = client.post("/api/timeline/read", json={"upto_id": first["id"]})
    assert read_response.status_code == 200
    assert read_response.json()["changed"] is True
    assert read_response.json()["upto_id"] == first["id"]

    after = client.get("/api/timeline").json()
    assert after["unread"] == 1
    assert after["reader"]["upto_id"] == first["id"]
    assert after["items"][1]["read"] is True
    assert after["items"][0]["read"] is False

    missing = client.post("/api/timeline/read", json={"upto_id": "tl_0123456789abcdef"})
    assert missing.status_code == 404


def test_public_timeline_marks_updates_and_changes(monkeypatch, tmp_path: Path) -> None:
    client, shared_dir = _app(monkeypatch, tmp_path, None)
    _set_hub(monkeypatch, timeline_enabled=True, company="a")
    month = now_local().strftime("%Y-%m")
    monkeypatch.setattr(timeline, "now_iso", lambda: f"{month}-06T10:00:00+09:00")
    original = append_item(
        shared_dir,
        from_name="alice",
        company="a",
        kind="notified",
        text="Started",
        source="anima",
    )
    monkeypatch.setattr(timeline, "now_iso", lambda: f"{month}-06T11:00:00+09:00")
    update = append_item(
        shared_dir,
        from_name="alice",
        company="a",
        kind="decided",
        text="Decided",
        source="anima",
        about=original["id"],
    )
    client.post("/api/timeline/read", json={"upto_id": original["id"]})

    data = client.get("/api/timeline").json()
    by_id = {item["id"]: item for item in data["items"]}
    assert by_id[original["id"]]["updates"] == [update["id"]]
    assert by_id[update["id"]]["changed"] is True
    assert by_id[update["id"]]["read"] is False


def test_internal_add_uses_caller_identity(monkeypatch, tmp_path: Path) -> None:
    client, _shared_dir = _app(monkeypatch, tmp_path, InternalCaller(kind="anima", name="alice"))
    _set_hub(monkeypatch, timeline_enabled=True, company="a")
    _set_company(tmp_path / "animas", "alice", "a")

    response = client.post(
        "/api/internal/timeline/add",
        json={"kind": "done", "text": "Finished"},
    )

    assert response.status_code == 200
    item = response.json()["item"]
    assert item["from"] == "alice"
    assert item["source"] == "anima"
    assert item["company"] == "a"


def test_internal_add_rejects_impersonation_and_other_company(monkeypatch, tmp_path: Path) -> None:
    client, _shared_dir = _app(monkeypatch, tmp_path, InternalCaller(kind="anima", name="alice"))
    _set_hub(monkeypatch, timeline_enabled=True, company="a")
    _set_company(tmp_path / "animas", "alice", "b")

    impersonation = client.post(
        "/api/internal/timeline/add",
        json={"from_anima": "bob", "kind": "done", "text": "Finished"},
    )
    assert impersonation.status_code == 403

    other_company = client.post(
        "/api/internal/timeline/add",
        json={"from_anima": "alice", "kind": "done", "text": "Finished"},
    )
    assert other_company.status_code == 403


def test_internal_add_disabled(monkeypatch, tmp_path: Path) -> None:
    client, _shared_dir = _app(monkeypatch, tmp_path, InternalCaller(kind="operator", name="operator"))
    _set_hub(monkeypatch)

    response = client.post(
        "/api/internal/timeline/add",
        json={"kind": "done", "text": "Finished"},
    )

    assert response.status_code == 409


def test_internal_get_exposes_human_read_marker(monkeypatch, tmp_path: Path) -> None:
    client, shared_dir = _app(monkeypatch, tmp_path, InternalCaller(kind="anima", name="alice"))
    _set_hub(monkeypatch, timeline_enabled=True, company="a")
    _set_company(tmp_path / "animas", "alice", "a")
    item = append_item(
        shared_dir,
        from_name="alice",
        company="a",
        kind="done",
        text="Finished",
        source="anima",
    )

    before = client.get("/api/internal/timeline")
    assert before.status_code == 200
    assert before.json()["items"][0]["read_by_human"] is False

    client.post("/api/timeline/read", json={"upto_id": item["id"]})
    after = client.get("/api/internal/timeline")

    assert after.status_code == 200
    assert after.json()["items"][0]["read_by_human"] is True
    assert after.json()["items"][0]["read"] is True
    assert after.json()["items"][0]["id"] == item["id"]
