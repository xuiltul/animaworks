"""Tests for GET /api/activity/group."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from core.time_utils import now_local
from server.routes.system import create_system_router


def _app(tmp_path: Path, names: list[str]) -> FastAPI:
    animas_dir = tmp_path / "animas"
    animas_dir.mkdir(parents=True)
    app = FastAPI()
    app.state.animas_dir = animas_dir
    app.state.shared_dir = tmp_path / "shared"
    app.state.anima_names = names
    app.state.supervisor = MagicMock()
    app.state.stream_registry = MagicMock()
    app.state.ws_manager = MagicMock()
    app.include_router(create_system_router(), prefix="/api")
    return app


def _write_activity(anima_dir: Path, entries: list[dict]) -> None:
    log_dir = anima_dir / "activity_log"
    log_dir.mkdir(parents=True)
    for entry in entries:
        path = log_dir / f"{entry['ts'][:10]}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")


async def test_group_id_from_list_returns_same_group_with_all_events(tmp_path) -> None:
    app = _app(tmp_path, ["alice"])
    anima_dir = app.state.animas_dir / "alice"
    anima_dir.mkdir()
    start = now_local().replace(microsecond=0) - timedelta(minutes=2)
    entries = [
        {
            "ts": start.isoformat(),
            "type": "message_received",
            "content": "Start",
            "meta": {"from_type": "human"},
        },
        *[
            {
                "ts": (start + timedelta(seconds=index + 1)).isoformat(),
                "type": "channel_read",
                "summary": f"Event {index}",
            }
            for index in range(35)
        ],
        {
            "ts": (start + timedelta(seconds=40)).isoformat(),
            "type": "response_sent",
            "content": "Finished",
        },
    ]
    _write_activity(anima_dir, entries)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        listed = await client.get(
            "/api/activity/recent?anima=alice&hours=1&grouped=true"
        )
        listed_group = listed.json()["groups"][0]
        detail = await client.get(
            "/api/activity/group",
            params={"anima": "alice", "id": listed_group["id"]},
        )

    assert listed.status_code == 200
    assert detail.status_code == 200
    detail_group = detail.json()["group"]
    assert detail_group == listed_group
    assert detail_group["event_count"] == 37
    assert len(detail_group["events"]) == 37


async def test_group_endpoint_returns_404_for_unknown_anima(tmp_path) -> None:
    app = _app(tmp_path, ["alice"])

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/api/activity/group",
            params={
                "anima": "bob",
                "id": "grp-bob:2026-07-21T10:00:00+09:00:chat",
            },
        )

    assert response.status_code == 404
    assert response.json() == {"error": "Anima not found: bob"}


async def test_group_endpoint_finds_group_outside_recent_paging_window(tmp_path) -> None:
    app = _app(tmp_path, ["alice"])
    anima_dir = app.state.animas_dir / "alice"
    anima_dir.mkdir()
    start = now_local().replace(microsecond=0) - timedelta(days=3)
    _write_activity(
        anima_dir,
        [
            {
                "ts": start.isoformat(),
                "type": "message_received",
                "content": "Archived session",
                "meta": {"from_type": "human"},
            },
            {
                "ts": (start + timedelta(seconds=1)).isoformat(),
                "type": "response_sent",
                "content": "Still addressable",
            },
        ],
    )
    group_id = f"grp-alice:{start.isoformat()}:chat"

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        recent = await client.get("/api/activity/recent?anima=alice&hours=1&grouped=true")
        detail = await client.get(
            "/api/activity/group",
            params={"anima": "alice", "id": group_id},
        )

    assert recent.json()["groups"] == []
    assert detail.status_code == 200
    assert detail.json()["group"]["id"] == group_id
    assert detail.json()["group"]["event_count"] == 2


async def test_group_endpoint_returns_404_for_invalid_or_missing_group(tmp_path) -> None:
    app = _app(tmp_path, ["alice"])
    (app.state.animas_dir / "alice").mkdir()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        malformed = await client.get(
            "/api/activity/group",
            params={"anima": "alice", "id": "bad-id"},
        )
        missing = await client.get(
            "/api/activity/group",
            params={
                "anima": "alice",
                "id": "grp-alice:2026-07-21T10:00:00+09:00:chat",
            },
        )

    assert malformed.status_code == 404
    assert malformed.json() == {"error": "Activity group not found"}
    assert missing.status_code == 404
    assert missing.json() == {"error": "Activity group not found"}
