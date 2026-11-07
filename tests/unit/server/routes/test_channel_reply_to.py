"""Tests for human reply targets and delivery metadata on channel reads."""

from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient


def _make_test_app(shared_dir: Path):
    from fastapi import FastAPI

    from server.routes.channels import create_channels_router

    app = FastAPI()
    app.state.shared_dir = shared_dir
    app.state.ws_manager = MagicMock()
    app.state.ws_manager.broadcast = AsyncMock()
    app.include_router(create_channels_router(), prefix="/api")
    return app


def _write_channel(shared_dir: Path, name: str, entries: list[dict]) -> Path:
    channels_dir = shared_dir / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    filepath = channels_dir / f"{name}.jsonl"
    filepath.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n", encoding="utf-8")
    return filepath


def _entry(post_id: str, text: str = "hello") -> dict:
    return {"ts": "2026-01-01T00:00:00+00:00", "from": "bob", "text": text, "source": "anima", "id": post_id}


async def test_human_post_records_explicit_reply_to(tmp_path: Path) -> None:
    shared_dir = tmp_path / "shared"
    channel_file = _write_channel(shared_dir, "general", [_entry("post-1")])
    app = _make_test_app(shared_dir)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/channels/general", json={"text": "reply", "reply_to": "post-1"})

    assert response.status_code == 200
    entries = [json.loads(line) for line in channel_file.read_text(encoding="utf-8").splitlines()]
    assert entries[-1]["reply_to"] == "post-1"


async def test_human_post_rejects_unknown_reply_to(tmp_path: Path) -> None:
    shared_dir = tmp_path / "shared"
    channel_file = _write_channel(shared_dir, "general", [_entry("post-1")])
    app = _make_test_app(shared_dir)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/channels/general", json={"text": "reply", "reply_to": "missing"})

    assert response.status_code == 400
    entries = [json.loads(line) for line in channel_file.read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 1
    assert "reply_to" not in entries[0]


async def test_channel_messages_include_delivery_metadata(tmp_path: Path) -> None:
    shared_dir = tmp_path / "shared"
    _write_channel(shared_dir, "general", [_entry("post-1"), _entry("post-2")])
    deliveries_dir = shared_dir / "board_deliveries"
    deliveries_dir.mkdir()
    (deliveries_dir / "general.jsonl").write_text(
        json.dumps(
            {
                "post_id": "post-1",
                "ts": "2026-01-01T00:00:00+00:00",
                "notified": ["bob"],
                "not_notified": {"carol": "disabled"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    app = _make_test_app(shared_dir)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/channels/general")

    assert response.status_code == 200
    messages = response.json()["messages"]
    assert messages[0]["delivery"] == {"notified": ["bob"], "not_notified": {"carol": "disabled"}}
    assert "delivery" not in messages[1]


@pytest.mark.parametrize("incremental", [False, True])
async def test_channel_messages_attach_deliveries_on_both_read_modes(tmp_path: Path, incremental: bool) -> None:
    shared_dir = tmp_path / "shared"
    _write_channel(shared_dir, "general", [_entry("post-1")])
    deliveries_dir = shared_dir / "board_deliveries"
    deliveries_dir.mkdir()
    (deliveries_dir / "general.jsonl").write_text(
        json.dumps({"post_id": "post-1", "ts": "2026-01-01T00:00:00+00:00", "notified": [], "not_notified": {}}) + "\n",
        encoding="utf-8",
    )
    app = _make_test_app(shared_dir)
    params = {"since": "2025-12-31T00:00:00+00:00"} if incremental else {}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/channels/general", params=params)

    assert response.status_code == 200
    assert response.json()["messages"][0]["delivery"] == {"notified": [], "not_notified": {}}
