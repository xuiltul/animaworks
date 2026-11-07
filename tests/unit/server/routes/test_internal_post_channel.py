from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
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
    animas_dir = tmp_path / "animas"
    for name in ("alice", "bob"):
        (animas_dir / name).mkdir(parents=True)
        (animas_dir / name / "status.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("core.paths.get_shared_dir", lambda: shared_dir)
    monkeypatch.setattr(
        "core.config.models.load_config",
        lambda: SimpleNamespace(animas={}),
    )
    app = _app(monkeypatch, caller)
    app.state.shared_dir = shared_dir
    app.state.animas_dir = animas_dir
    return TestClient(app), channels_dir / "general.jsonl"


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
        json=_body("operator", from_name="operator-sender"),
    )

    assert response.status_code == 200
    entry = json.loads(channel_file.read_text(encoding="utf-8").strip())
    assert entry["from"] == "operator-sender"
    assert entry["source"] == "human"
    assert entry["kind"] == "operator"


def test_suppress_mentions_keeps_post_but_skips_inbox_delivery(monkeypatch, tmp_path: Path) -> None:
    client, channel_file = _setup_post(monkeypatch, tmp_path, InternalCaller(kind="anima", name="alice"))

    response = client.post(
        "/api/internal/post-channel",
        json={**_body("anima"), "text": "Please review @bob", "suppress_mentions": True},
    )

    assert response.status_code == 200
    result = response.json()["post"]
    assert result["committed"] is True
    assert result["notified"] == []
    assert list((channel_file.parent.parent / "inbox").glob("bob/*.json")) == []
    entry = json.loads(channel_file.read_text(encoding="utf-8").strip())
    assert entry["text"] == "Please review @bob"


@pytest.mark.parametrize(
    ("source", "from_name", "caller", "expected_origin"),
    [
        ("anima", None, InternalCaller(kind="anima", name="alice"), "internal"),
        ("human", "external-user", InternalCaller(kind="operator", name="operator"), "mirror"),
    ],
)
def test_suppression_preserves_origin_for_outbound_hook(
    monkeypatch,
    tmp_path: Path,
    source: str,
    from_name: str | None,
    caller: InternalCaller,
    expected_origin: str,
) -> None:
    from core.messaging.board import register_post_hook, unregister_post_hook

    client, _channel_file = _setup_post(monkeypatch, tmp_path, caller)
    origins: list[str] = []
    register_post_hook("origin_capture", lambda post, result: origins.append(post.origin) or "sent")
    try:
        response = client.post(
            "/api/internal/post-channel",
            json={**_body(source, from_name=from_name), "suppress_mentions": True},
        )
    finally:
        unregister_post_hook("origin_capture")

    assert response.status_code == 200
    assert response.json()["post"]["notified"] == []
    assert origins == [expected_origin]


def test_anima_post_records_board_reply_and_trigger(monkeypatch, tmp_path: Path) -> None:
    client, channel_file = _setup_post(monkeypatch, tmp_path, InternalCaller(kind="anima", name="alice"))
    triggers = [
        {"id": "msg-1", "from": "bob", "type": "message", "thread_id": "thread-1"},
        {
            "id": "mention-1",
            "from": "carol",
            "type": "board_mention",
            "board_channel": "general",
            "board_post_id": "post-9",
        },
    ]

    response = client.post("/api/internal/post-channel", json={**_body("anima"), "triggers": triggers})

    assert response.status_code == 200
    entry = json.loads(channel_file.read_text(encoding="utf-8").strip())
    assert entry["reply_to"] == "post-9"
    assert entry["trigger"] == {"type": "board_mention", "id": "post-9"}


def test_anima_post_has_no_reply_for_other_channel(monkeypatch, tmp_path: Path) -> None:
    client, channel_file = _setup_post(monkeypatch, tmp_path, InternalCaller(kind="anima", name="alice"))
    triggers = [
        {
            "id": "mention-1",
            "from": "carol",
            "type": "board_mention",
            "board_channel": "ops",
            "board_post_id": "post-9",
        }
    ]

    response = client.post("/api/internal/post-channel", json={**_body("anima"), "triggers": triggers})

    assert response.status_code == 200
    entry = json.loads(channel_file.read_text(encoding="utf-8").strip())
    assert "reply_to" not in entry
    assert entry["trigger"] == {"type": "board_mention", "id": "post-9"}


def test_anima_post_without_triggers_has_no_reply_metadata(monkeypatch, tmp_path: Path) -> None:
    client, channel_file = _setup_post(monkeypatch, tmp_path, InternalCaller(kind="anima", name="alice"))

    response = client.post("/api/internal/post-channel", json=_body("anima"))

    assert response.status_code == 200
    entry = json.loads(channel_file.read_text(encoding="utf-8").strip())
    assert "reply_to" not in entry
    assert "trigger" not in entry


def test_message_trigger_records_message_id_without_reply_to(monkeypatch, tmp_path: Path) -> None:
    client, channel_file = _setup_post(monkeypatch, tmp_path, InternalCaller(kind="anima", name="alice"))
    triggers = [{"id": "msg-7", "from": "bob", "type": "message"}]

    response = client.post("/api/internal/post-channel", json={**_body("anima"), "triggers": triggers})

    assert response.status_code == 200
    entry = json.loads(channel_file.read_text(encoding="utf-8").strip())
    assert "reply_to" not in entry
    assert entry["trigger"] == {"type": "message", "id": "msg-7"}
