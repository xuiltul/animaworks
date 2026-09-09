from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from core.messenger import Messenger
from core.schemas import Message
from server.routes.internal import create_internal_router


@pytest.mark.asyncio
async def test_host_delivery_does_not_replay_archived_message(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    monkeypatch.setattr("core.paths.get_shared_dir", lambda: shared)
    app = FastAPI()
    app.include_router(create_internal_router(), prefix="/api")
    receiver = Messenger(shared, "worker")
    message = Message(id="task-wakeup-fixed", from_person="manager", to_person="worker", content="Check task")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        first = await client.post("/api/internal/send-message", json={"message": message.model_dump(mode="json")})
        assert first.status_code == 200
        original = shared / "inbox" / "worker" / f"{message.id}.json"
        processed = original.parent / "processed"
        processed.mkdir()
        original.rename(processed / original.name)
        repeated = await client.post("/api/internal/send-message", json={"message": message.model_dump(mode="json")})
        assert repeated.status_code == 200
        assert not original.exists()
        assert receiver.receive() == []


@pytest.mark.asyncio
async def test_host_delivery_rejects_path_like_message_id(tmp_path, monkeypatch):
    monkeypatch.setattr("core.paths.get_shared_dir", lambda: tmp_path / "shared")
    app = FastAPI()
    app.include_router(create_internal_router(), prefix="/api")
    message = Message(id="../escape", from_person="manager", to_person="worker", content="Invalid")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/internal/send-message", json={"message": message.model_dump(mode="json")})
    assert response.status_code == 400
    assert not (tmp_path / "shared").exists()
