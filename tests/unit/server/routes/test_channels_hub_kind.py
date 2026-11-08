from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

from httpx import ASGITransport, AsyncClient


def _make_test_app(shared_dir: Path):
    from fastapi import FastAPI

    from server.routes.channels import create_channels_router

    app = FastAPI()
    app.state.shared_dir = shared_dir
    app.include_router(create_channels_router(), prefix="/api")
    return app


def _write_channel(shared_dir: Path, name: str, entries: list[dict]) -> Path:
    channels_dir = shared_dir / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    path = channels_dir / f"{name}.jsonl"
    path.write_text("\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n", encoding="utf-8")
    return path


async def test_hub_source_is_inferred_as_external_agent(tmp_path: Path) -> None:
    shared_dir = tmp_path / "shared"
    _write_channel(
        shared_dir,
        "hub",
        [
            {"ts": "2026-02-17T10:00:00", "from": "alice", "text": "inside", "kind": "anima"},
            {"ts": "2026-02-17T10:01:00", "from": "grok", "text": "outside", "source": "hub"},
            {
                "ts": "2026-02-17T10:02:00",
                "from": "claude",
                "text": "explicit",
                "source": "hub",
                "kind": "external_agent",
            },
        ],
    )

    app = _make_test_app(shared_dir)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/channels/hub?limit=50")
        external = await client.get("/api/channels/hub?kind=external")

    messages = response.json()["messages"]
    assert [message["kind"] for message in messages] == ["anima", "external_agent", "external_agent"]
    assert messages[1]["kind_inferred"] is True
    assert messages[2].get("kind_inferred") is None
    assert [message["text"] for message in external.json()["messages"]] == ["outside", "explicit"]
