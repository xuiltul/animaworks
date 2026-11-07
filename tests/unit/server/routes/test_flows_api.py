"""API tests for the flow endpoint and the channel information panel."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from core.messaging.flows import clear_flows_cache
from core.messaging.messenger import ChannelMeta, save_channel_meta
from core.time_utils import now_local


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_flows_cache()
    yield
    clear_flows_cache()


def _make_test_app(shared: Path):
    from fastapi import FastAPI

    from server.routes.channels import create_channels_router

    app = FastAPI()
    app.state.shared_dir = shared
    app.include_router(create_channels_router(), prefix="/api")
    return app


def _make_anima(root: Path, name: str) -> None:
    anima_dir = root / "animas" / name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "status.json").write_text(json.dumps({"enabled": True}), encoding="utf-8")


def _write_board(shared: Path, channel: str, rows: list[dict]) -> None:
    channels_dir = shared / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    (channels_dir / f"{channel}.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _ts(offset: timedelta) -> str:
    return (now_local() + offset).isoformat()


async def test_flows_endpoint_and_validation(tmp_path: Path):
    shared = tmp_path / "shared"
    _make_anima(tmp_path, "alice")
    _make_anima(tmp_path, "bob")
    start = _ts(-timedelta(hours=1))
    _write_board(
        shared,
        "general",
        [{"ts": start, "from": "alice", "text": "@bob hi", "source": "anima", "id": "p1", "kind": "anima"}],
    )
    deliveries = shared / "board_deliveries"
    deliveries.mkdir(parents=True)
    (deliveries / "general.jsonl").write_text(
        json.dumps({"post_id": "p1", "ts": start, "notified": ["bob"], "not_notified": {}}) + "\n",
        encoding="utf-8",
    )
    app = _make_test_app(shared)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/flows", params={"days": 7, "via": "all"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["calls"][0]["to"] == "bob"
        assert payload["summary"]["calls"] == 1

        for params in ({"days": 14}, {"days": "many"}, {"via": "other"}):
            bad = await client.get("/api/flows", params=params)
            assert bad.status_code == 400


async def test_channel_info_panel(tmp_path: Path):
    shared = tmp_path / "shared"
    for name in ("alice", "bob", "carol"):
        _make_anima(tmp_path, name)
    start = _ts(-timedelta(hours=5))
    _write_board(
        shared,
        "general",
        [
            {"ts": start, "from": "alice", "text": "@bob @bob @carol", "source": "anima", "kind": "anima"},
            {"ts": _ts(-timedelta(hours=4)), "from": "owner", "text": "hello @bob", "source": "human"},
            {"ts": _ts(-timedelta(hours=3)), "from": "guest", "text": "external", "source": "slack"},
            {"ts": _ts(-timedelta(hours=2)), "from": "system", "text": "system", "source": "system"},
            {"ts": _ts(-timedelta(hours=1)), "from": "bot", "text": "bot", "source": "anima", "kind": "external_bot"},
        ],
    )
    save_channel_meta(shared, "general", ChannelMeta(members=["alice"]))
    reads_dir = shared / "reads" / "board"
    reads_dir.mkdir(parents=True)
    (reads_dir / "general.json").write_text(
        json.dumps(
            {
                "stream": "board/general",
                "readers": {
                    "anima:alice": {"upto_id": "p1", "upto_ts": start, "updated_at": start},
                },
                "picked": {},
            }
        ),
        encoding="utf-8",
    )
    app = _make_test_app(shared)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/channels/general/info")
        assert response.status_code == 200
        payload = response.json()

    assert payload["members"] == ["alice"]
    assert payload["mention_targets"] == ["alice"]
    assert payload["readers"] == [{"reader": "anima:alice", "unread": 4, "updated_at": start}]
    assert payload["mentions_30d"] == [["bob", 2], ["carol", 1]]
    assert payload["kinds"] == {"anima": 1, "human": 1, "external": 2, "operator": 1}


async def test_channel_info_rejects_unreadable_board_and_missing(tmp_path: Path):
    shared = tmp_path / "shared"
    _make_anima(tmp_path, "alice")
    _write_board(shared, "hidden", [{"ts": _ts(-timedelta(hours=1)), "from": "alice", "text": "x", "source": "anima"}])
    (shared / "channels" / "hidden.meta.json").write_text("not json", encoding="utf-8")
    app = _make_test_app(shared)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        hidden = await client.get("/api/channels/hidden/info")
        missing = await client.get("/api/channels/missing/info")

    assert hidden.status_code == 403
    assert missing.status_code == 404
