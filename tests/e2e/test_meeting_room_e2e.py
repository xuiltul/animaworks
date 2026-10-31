# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for Meeting Room feature API routes.

Validates Room CRUD API and meeting chat SSE streaming through the full
FastAPI app stack, with mocked ProcessSupervisor to avoid real Anima processes.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from core.runtime.ipc import IPCResponse

# ── Helpers ──────────────────────────────────────────────────


def _create_app(
    tmp_path: Path,
    anima_names: list[str] | None = None,
    ws_connections: int = 0,
):
    """Build a real FastAPI app via create_app with mocked externals.

    Returns an app whose setup_complete flag is True so the setup-guard
    middleware lets API requests through. room_manager is created from
    shared_dir/meetings.
    """
    animas_dir = tmp_path / "animas"
    animas_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

    for name in anima_names or []:
        d = animas_dir / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "IDENTITY.md").write_text(f"# {name}", encoding="utf-8")

    with (
        patch("server.app.ProcessSupervisor") as mock_sup_cls,
        patch("server.app.load_config") as mock_cfg,
        patch("server.app.WebSocketManager") as mock_ws_cls,
        patch("server.app.load_auth") as mock_auth,
    ):
        cfg = MagicMock()
        cfg.setup_complete = True
        mock_cfg.return_value = cfg

        auth_cfg = MagicMock()
        auth_cfg.auth_mode = "local_trust"
        mock_auth.return_value = auth_cfg

        supervisor = MagicMock()
        supervisor.get_all_status.return_value = {}
        supervisor.get_process_status.return_value = {
            "status": "stopped",
            "pid": None,
        }
        supervisor.is_scheduler_running.return_value = False
        supervisor.scheduler = None
        mock_sup_cls.return_value = supervisor

        ws_manager = MagicMock()
        ws_manager.active_connections = [MagicMock() for _ in range(ws_connections)]
        mock_ws_cls.return_value = ws_manager

        from server.app import create_app

        app = create_app(animas_dir, shared_dir)

    import server.app as _sa

    _auth = MagicMock()
    _auth.auth_mode = "local_trust"
    _sa.load_auth = lambda: _auth

    if anima_names is not None:
        app.state.anima_names = anima_names

    return app


def _parse_sse_events(response_text: str) -> list[tuple[str, dict]]:
    """Parse SSE response into list of (event, data) tuples."""
    events = []
    for block in response_text.strip().split("\n\n"):
        if not block or block.startswith(":"):
            continue
        event = None
        data = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    data = {}
        if event:
            events.append((event, data or {}))
    return events


# ── Room CRUD API Tests ───────────────────────────────────────


class TestRoomCRUD:
    """Room CRUD API tests."""

    @pytest.mark.asyncio
    async def test_create_room(self, tmp_path: Path) -> None:
        """POST /api/rooms creates room and returns room_id."""
        app = _create_app(tmp_path, anima_names=["sakura", "rin"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/rooms",
                json={
                    "participants": ["sakura", "rin"],
                    "chair": "sakura",
                    "title": "Test Meeting",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "room_id" in data
        assert len(data["room_id"]) == 12
        assert data["participants"] == ["sakura", "rin"]
        assert data["chair"] == "sakura"
        assert data["title"] == "Test Meeting"
        assert data["closed"] is False
        assert "created_at" in data

    @pytest.mark.asyncio
    async def test_create_room_invalid_too_many_participants(self, tmp_path: Path) -> None:
        """422 for > 5 participants."""
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/rooms",
                json={
                    "participants": ["a", "b", "c", "d", "e", "f"],
                    "chair": "a",
                    "title": "",
                },
            )

        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_room_chair_not_in_participants(self, tmp_path: Path) -> None:
        """400 validation error when chair not in participants."""
        app = _create_app(tmp_path, anima_names=["sakura", "rin"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/rooms",
                json={
                    "participants": ["sakura", "rin"],
                    "chair": "unknown",
                    "title": "",
                },
            )

        assert resp.status_code == 400
        detail = resp.json().get("detail", "")
        assert detail  # Validation error message (may be localized)

    @pytest.mark.asyncio
    async def test_list_rooms(self, tmp_path: Path) -> None:
        """GET /api/rooms returns list."""
        app = _create_app(tmp_path, anima_names=["sakura"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Create a room first
            await client.post(
                "/api/rooms",
                json={"participants": ["sakura"], "chair": "sakura", "title": ""},
            )
            resp = await client.get("/api/rooms")

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        room = data[0]
        assert "room_id" in room
        assert "participants" in room
        assert "chair" in room
        assert "closed" in room

    @pytest.mark.asyncio
    async def test_get_room(self, tmp_path: Path) -> None:
        """GET /api/rooms/{room_id} returns room details."""
        app = _create_app(tmp_path, anima_names=["sakura", "rin"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            create_resp = await client.post(
                "/api/rooms",
                json={
                    "participants": ["sakura", "rin"],
                    "chair": "sakura",
                    "title": "Detail Test",
                },
            )
            room_id = create_resp.json()["room_id"]
            resp = await client.get(f"/api/rooms/{room_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["room_id"] == room_id
        assert data["participants"] == ["sakura", "rin"]
        assert data["chair"] == "sakura"
        assert data["title"] == "Detail Test"
        assert "conversation" in data
        assert data["conversation"] == []

    @pytest.mark.asyncio
    async def test_get_room_not_found(self, tmp_path: Path) -> None:
        """404 for unknown room_id."""
        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/rooms/nonexistent123")

        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_add_participant(self, tmp_path: Path) -> None:
        """POST /api/rooms/{room_id}/participants adds participant."""
        app = _create_app(tmp_path, anima_names=["sakura", "rin"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            create_resp = await client.post(
                "/api/rooms",
                json={"participants": ["sakura"], "chair": "sakura", "title": ""},
            )
            room_id = create_resp.json()["room_id"]
            resp = await client.post(
                f"/api/rooms/{room_id}/participants",
                json={"name": "rin"},
            )

            assert resp.status_code == 200
            assert resp.json()["ok"] is True

            get_resp = await client.get(f"/api/rooms/{room_id}")
            assert "rin" in get_resp.json()["participants"]

    @pytest.mark.asyncio
    async def test_remove_participant(self, tmp_path: Path) -> None:
        """DELETE /api/rooms/{room_id}/participants/{name} removes participant."""
        app = _create_app(tmp_path, anima_names=["sakura", "rin"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            create_resp = await client.post(
                "/api/rooms",
                json={
                    "participants": ["sakura", "rin"],
                    "chair": "sakura",
                    "title": "",
                },
            )
            room_id = create_resp.json()["room_id"]
            resp = await client.delete(
                f"/api/rooms/{room_id}/participants/rin",
            )

            assert resp.status_code == 200
            assert resp.json()["ok"] is True

            get_resp = await client.get(f"/api/rooms/{room_id}")
            assert "rin" not in get_resp.json()["participants"]

    @pytest.mark.asyncio
    async def test_close_room(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """POST /api/rooms/{room_id}/close closes room and generates minutes."""
        ck_dir = tmp_path / "common_knowledge"
        ck_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(
            "server.routes.room.get_common_knowledge_dir",
            lambda: ck_dir,
        )

        app = _create_app(tmp_path, anima_names=["sakura"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            create_resp = await client.post(
                "/api/rooms",
                json={"participants": ["sakura"], "chair": "sakura", "title": "Closed"},
            )
            room_id = create_resp.json()["room_id"]
            resp = await client.post(f"/api/rooms/{room_id}/close")

            assert resp.status_code == 200
            data = resp.json()
            assert data["ok"] is True
            assert "minutes_path" in data

            get_resp = await client.get(f"/api/rooms/{room_id}")
            assert get_resp.json()["closed"] is True


# ── Meeting SSE Streaming Tests ──────────────────────────────


class TestMeetingChatStream:
    """Meeting chat SSE streaming tests with mocked supervisor."""

    @pytest.mark.asyncio
    async def test_meeting_chat_stream_basic(self, tmp_path: Path) -> None:
        """POST /api/rooms/{room_id}/chat/stream returns SSE events.

        Mocks ProcessSupervisor.send_request_stream to yield fake IPC responses.
        Verifies speaker_start, text_delta, speaker_end, and done events.
        """
        app = _create_app(tmp_path, anima_names=["sakura"])
        transport = ASGITransport(app=app)

        # Supervisor must have sakura in processes for the stream to proceed
        app.state.supervisor.processes = {"sakura": MagicMock()}

        async def mock_send_request_stream(anima_name, method, params, timeout=60.0):
            yield IPCResponse(
                id="req-1",
                stream=True,
                chunk=json.dumps({"type": "text_delta", "text": "Hello "}),
                done=False,
            )
            yield IPCResponse(
                id="req-1",
                stream=True,
                chunk=json.dumps({"type": "text_delta", "text": "from meeting."}),
                done=False,
            )
            yield IPCResponse(
                id="req-1",
                stream=True,
                done=True,
                result={
                    "response": "Hello from meeting.",
                    "cycle_result": {"summary": "Hello from meeting."},
                },
            )

        # Assign async generator function directly (not AsyncMock) so
        # "async for" receives an async iterator
        app.state.supervisor.send_request_stream = mock_send_request_stream

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            create_resp = await client.post(
                "/api/rooms",
                json={"participants": ["sakura"], "chair": "sakura", "title": ""},
            )
            room_id = create_resp.json()["room_id"]

            resp = await client.post(
                f"/api/rooms/{room_id}/chat/stream",
                json={"message": "Test message", "from_person": "human"},
            )

        assert resp.status_code == 200
        assert resp.headers.get("content-type", "").startswith("text/event-stream")

        events = _parse_sse_events(resp.text)

        event_names = [e[0] for e in events]
        assert "speaker_start" in event_names
        assert "text_delta" in event_names
        assert "speaker_end" in event_names
        assert "done" in event_names

        # Verify speaker_start has correct speaker
        speaker_starts = [e for e in events if e[0] == "speaker_start"]
        assert len(speaker_starts) >= 1
        assert speaker_starts[0][1].get("speaker") == "sakura"
        assert speaker_starts[0][1].get("role") == "chair"

        # Verify text_delta content
        text_deltas = [e for e in events if e[0] == "text_delta"]
        combined_text = "".join(d.get("text", "") for _, d in text_deltas)
        assert "Hello" in combined_text
        assert "meeting" in combined_text

        # Verify done event
        done_events = [e for e in events if e[0] == "done"]
        assert len(done_events) >= 1

    @pytest.mark.asyncio
    async def test_meeting_redirect_and_mention_process_participant_once(self, tmp_path: Path) -> None:
        app = _create_app(tmp_path, anima_names=["sakura", "rin"])
        transport = ASGITransport(app=app)
        app.state.supervisor.processes = {"sakura": MagicMock(), "rin": MagicMock()}
        calls: list[tuple[str, dict]] = []

        async def mock_send_request_stream(anima_name, method, params, timeout=60.0):
            calls.append((anima_name, dict(params)))
            if anima_name == "sakura":
                yield IPCResponse(
                    id="req-chair",
                    stream=True,
                    chunk=json.dumps(
                        {
                            "type": "meeting_redirect",
                            "from": "sakura",
                            "to": "rin",
                            "content": "Please share status",
                            "intent": "question",
                        }
                    ),
                    done=False,
                )
                yield IPCResponse(
                    id="req-chair",
                    stream=True,
                    chunk=json.dumps({"type": "text_delta", "text": "@rin Please share status"}),
                    done=False,
                )
                yield IPCResponse(
                    id="req-chair",
                    stream=True,
                    done=True,
                    result={
                        "response": "@rin Please share status",
                        "cycle_result": {"summary": "@rin Please share status"},
                    },
                )
                return

            yield IPCResponse(
                id="req-rin",
                stream=True,
                done=True,
                result={
                    "response": "Status is green.",
                    "cycle_result": {"summary": "Status is green."},
                },
            )

        app.state.supervisor.send_request_stream = mock_send_request_stream

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            create_resp = await client.post(
                "/api/rooms",
                json={"participants": ["sakura", "rin"], "chair": "sakura", "title": "Redirect"},
            )
            room_id = create_resp.json()["room_id"]

            resp = await client.post(
                f"/api/rooms/{room_id}/chat/stream",
                json={"message": "Please coordinate", "from_person": "human"},
            )
            room_resp = await client.get(f"/api/rooms/{room_id}")

        assert resp.status_code == 200
        assert [name for name, _params in calls] == ["sakura", "rin"]
        assert sum(1 for name, _params in calls if name == "rin") == 1
        assert all(call_params["meeting_room_id"] == room_id for _name, call_params in calls)
        assert all(call_params["meeting_participants"] == ["sakura", "rin"] for _name, call_params in calls)

        events = _parse_sse_events(resp.text)
        assert sum(1 for name, _payload in events if name == "meeting_redirect") == 1

        conversation = room_resp.json()["conversation"]
        redirected = [entry for entry in conversation if entry.get("meta", {}).get("type") == "meeting_redirect"]
        assert len(redirected) == 1
        assert redirected[0]["speaker"] == "sakura"
        assert redirected[0]["text"] == "@rin Please share status"
