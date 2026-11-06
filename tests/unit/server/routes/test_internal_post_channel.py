from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from server.internal_auth import InternalCaller


def _app(monkeypatch, caller: InternalCaller | None) -> FastAPI:
    def test_require_internal_caller(_request: Request) -> InternalCaller | None:
        return caller

    monkeypatch.setattr("server.internal_auth.require_internal_caller", test_require_internal_caller)

    from server.routes.internal import create_internal_router

    app = FastAPI()
    app.include_router(create_internal_router(), prefix="/api")

    @app.middleware("http")
    async def set_test_caller(request, call_next):
        request.state.internal_caller = caller
        return await call_next(request)

    return app


def _setup_post(monkeypatch, tmp_path: Path, caller: InternalCaller | None) -> tuple[TestClient, Path]:
    shared_dir = tmp_path / "shared"
    channels_dir = shared_dir / "channels"
    channels_dir.mkdir(parents=True)
    (channels_dir / "general.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr("core.paths.get_shared_dir", lambda: shared_dir)
    monkeypatch.setattr(
        "core.config.models.load_config",
        lambda: SimpleNamespace(animas={}),
    )
    return TestClient(_app(monkeypatch, caller)), channels_dir / "general.jsonl"


def _body(source: str, *, from_name: str | None = None) -> dict[str, str | None]:
    return {
        "from_anima": "alice",
        "channel": "general",
        "text": "hello",
        "source": source,
        "from_name": from_name,
    }


def test_unverified_caller_may_only_post_with_anima_source(monkeypatch, tmp_path: Path) -> None:
    client, channel_file = _setup_post(monkeypatch, tmp_path, None)

    anima_response = client.post("/api/internal/post-channel", json=_body("anima"))
    human_response = client.post("/api/internal/post-channel", json=_body("human"))
    slack_response = client.post("/api/internal/post-channel", json=_body("slack"))

    assert anima_response.status_code == 200
    assert human_response.status_code == 403
    assert slack_response.status_code == 403
    entries = [json.loads(line) for line in channel_file.read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 1
    assert entries[0]["from"] == "alice"
    assert entries[0]["source"] == "anima"


def test_anima_caller_is_recorded_as_anima_despite_body_source(monkeypatch, tmp_path: Path) -> None:
    client, channel_file = _setup_post(monkeypatch, tmp_path, InternalCaller(kind="anima", name="alice"))

    response = client.post(
        "/api/internal/post-channel",
        json=_body("human", from_name="spoofed-name"),
    )

    assert response.status_code == 200
    entry = json.loads(channel_file.read_text(encoding="utf-8").strip())
    assert entry["from"] == "alice"
    assert entry["source"] == "anima"


def test_operator_caller_can_use_body_source_and_sender(monkeypatch, tmp_path: Path) -> None:
    client, channel_file = _setup_post(monkeypatch, tmp_path, InternalCaller(kind="operator", name="operator"))

    response = client.post(
        "/api/internal/post-channel",
        json=_body("slack", from_name="operator-sender"),
    )

    assert response.status_code == 200
    entry = json.loads(channel_file.read_text(encoding="utf-8").strip())
    assert entry["from"] == "operator-sender"
    assert entry["source"] == "slack"
