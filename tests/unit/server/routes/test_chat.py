"""Unit tests for server/routes/chat.py — Chat API endpoints."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import ASGITransport, AsyncClient

from server.stream_registry import StreamRegistry


def _make_test_app(animas: dict | None = None, supervisor: MagicMock | None = None):
    from fastapi import FastAPI

    from server.routes.chat import create_chat_router

    app = FastAPI()
    animas = animas or {}
    app.state.animas = animas
    app.state.ws_manager = MagicMock()
    app.state.ws_manager.broadcast = AsyncMock()
    app.state.stream_registry = StreamRegistry()
    if supervisor is not None:
        supervisor.is_bootstrapping = MagicMock(return_value=False)
        app.state.supervisor = supervisor
    else:
        # Build a supervisor mock that delegates to anima mocks
        sup = MagicMock()
        sup.processes = set(animas.keys())
        sup.is_bootstrapping = MagicMock(return_value=False)

        async def _send_request(anima_name, method, params, timeout=60.0):
            if anima_name not in animas:
                raise KeyError(anima_name)
            p = animas[anima_name]
            if method == "process_message":
                result = await p.process_message(
                    params.get("message", ""),
                    from_person=params.get("from_person", "human"),
                    intent=params.get("intent", ""),
                )
                return {"response": result, "replied_to": []}
            if method == "greet":
                return await p.process_greet(
                    mode=params.get("mode", "visit"),
                    user_name=params.get("user_name", ""),
                )
            raise ValueError(f"Unknown method: {method}")

        async def _send_request_stream(anima_name, method, params, timeout=120.0):
            if anima_name not in animas:
                raise KeyError(anima_name)
            p = animas[anima_name]
            import json as _json

            from core.runtime.ipc import IPCResponse

            async for chunk in p.process_message_stream(
                params.get("message", ""),
                from_person=params.get("from_person", "human"),
                intent=params.get("intent", ""),
            ):
                event_type = chunk.get("type", "unknown")
                if event_type == "cycle_done":
                    cycle_result = chunk.get("cycle_result", {})
                    yield IPCResponse(
                        id="test",
                        stream=True,
                        done=True,
                        result={
                            "response": cycle_result.get("summary", ""),
                            "replied_to": [],
                            "cycle_result": cycle_result,
                        },
                    )
                    return
                yield IPCResponse(
                    id="test",
                    stream=True,
                    chunk=_json.dumps(chunk, ensure_ascii=False),
                )

        sup.send_request = _send_request
        sup.send_request_stream = _send_request_stream
        app.state.supervisor = sup
    router = create_chat_router()
    app.include_router(router, prefix="/api")
    return app


def _make_mock_anima(name: str = "alice"):
    dp = MagicMock()
    dp.name = name
    dp.process_message = AsyncMock(return_value="Hello from Alice")
    return dp


# ── POST /animas/{name}/chat ────────────────────────────


class TestChat:
    async def test_chat_success(self):
        alice = _make_mock_anima("alice")
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat",
                json={"message": "Hi"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "Hello from Alice"
        assert data["anima"] == "alice"
        assert data["images"] == []

    async def test_chat_includes_images_from_cycle_result(self):
        supervisor = MagicMock()
        supervisor.send_request = AsyncMock(
            return_value={
                "response": "ok",
                "replied_to": [],
                "cycle_result": {
                    "summary": "ok",
                    "images": [{"type": "image", "source": "generated", "path": "assets/a.png"}],
                },
            }
        )
        app = _make_test_app(supervisor=supervisor)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat",
                json={"message": "Hi"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["images"]) == 1
        assert data["images"][0]["path"] == "assets/a.png"

    async def test_chat_includes_files_from_cycle_result(self):
        supervisor = MagicMock()
        supervisor.send_request = AsyncMock(
            return_value={
                "response": "ok",
                "replied_to": [],
                "cycle_result": {
                    "summary": "ok",
                    "files": [
                        {
                            "name": "summary.md",
                            "path": "attachments/20261029_101500_a1_summary.md",
                            "size": 128,
                            "media_type": "text/markdown",
                        }
                    ],
                },
            }
        )
        app = _make_test_app(supervisor=supervisor)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat",
                json={"message": "Hi"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["files"] == [
            {
                "name": "summary.md",
                "path": "attachments/20261029_101500_a1_summary.md",
                "size": 128,
                "media_type": "text/markdown",
            }
        ]

    async def test_chat_file_upload_passes_saved_paths_and_prompt_list_to_ipc(self):
        saved_files = [
            {
                "name": "estimate.pdf",
                "path": "attachments/20261029_101500_0_estimate.pdf",
                "size": 3,
                "media_type": "application/pdf",
            }
        ]
        supervisor = MagicMock()
        supervisor.send_request = AsyncMock(return_value={"response": "ok", "replied_to": []})
        app = _make_test_app(supervisor=supervisor)
        transport = ASGITransport(app=app)
        with patch("server.routes.chat.save_files", return_value=saved_files):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/animas/alice/chat",
                    json={
                        "message": "Please review",
                        "files": [
                            {
                                "name": "estimate.pdf",
                                "media_type": "application/pdf",
                                "data": base64.b64encode(b"PDF").decode("ascii"),
                            }
                        ],
                    },
                )

        assert resp.status_code == 200
        params = supervisor.send_request.await_args.kwargs["params"]
        assert params["display_message"] == "Please review"
        assert params["files"] == saved_files
        assert params["attachment_paths"] == [saved_files[0]["path"]]
        assert "[添付ファイル]" in params["message"] or "[Attached files]" in params["message"]
        assert saved_files[0]["path"] in params["message"]

    async def test_file_validation_uses_error_object_for_400_and_413(self, monkeypatch):
        alice = _make_mock_anima("alice")
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            malformed = await client.post(
                "/api/animas/alice/chat",
                json={"message": "Hi", "files": [{"name": "bad.bin", "media_type": "", "data": "%%%"}]},
            )
            monkeypatch.setattr("server.routes.chat_files.MAX_CHAT_FILE_SIZE", 2)
            oversized = await client.post(
                "/api/animas/alice/chat",
                json={"message": "Hi", "files": [{"name": "large.bin", "media_type": "", "data": "YWJj"}]},
            )

        assert malformed.status_code == 400
        assert "error" in malformed.json()
        assert oversized.status_code == 413
        assert "error" in oversized.json()

    async def test_stream_file_upload_passes_files_and_emits_paths_for_local_bubble(self):
        from core.runtime.ipc import IPCResponse

        saved_files = [
            {
                "name": "estimate.pdf",
                "path": "attachments/20261029_101500_0_estimate.pdf",
                "size": 3,
                "media_type": "application/pdf",
            }
        ]
        captured = {}
        supervisor = MagicMock()
        supervisor.processes = {"alice"}

        async def send_request_stream(*, anima_name, method, params, timeout):
            captured.update(params)
            yield IPCResponse(
                id="test",
                stream=True,
                done=True,
                result={"response": "ok", "replied_to": [], "cycle_result": {"summary": "ok"}},
            )

        supervisor.send_request_stream = send_request_stream
        app = _make_test_app(supervisor=supervisor)
        transport = ASGITransport(app=app)
        with patch("server.routes.chat.save_files", return_value=saved_files):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/animas/alice/chat/stream",
                    json={
                        "message": "Please review",
                        "files": [
                            {
                                "name": "estimate.pdf",
                                "media_type": "application/pdf",
                                "data": base64.b64encode(b"PDF").decode("ascii"),
                            }
                        ],
                    },
                )

        assert resp.status_code == 200
        assert captured["display_message"] == "Please review"
        assert captured["files"] == saved_files
        assert captured["attachment_paths"] == [saved_files[0]["path"]]
        assert saved_files[0]["path"] in captured["message"]
        assert '"files":' in resp.text
        assert saved_files[0]["path"] in resp.text

    async def test_chat_with_from_anima(self):
        alice = _make_mock_anima("alice")
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat",
                json={"message": "Hi", "from_person": "bob"},
            )
        assert resp.status_code == 200
        alice.process_message.assert_awaited_once_with("Hi", from_person="bob", intent="")

    async def test_chat_default_from_anima(self):
        alice = _make_mock_anima("alice")
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/animas/alice/chat",
                json={"message": "Hi"},
            )
        alice.process_message.assert_awaited_once_with("Hi", from_person="human", intent="")

    async def test_chat_forwards_intent_to_supervisor(self):
        supervisor = MagicMock()
        supervisor.send_request = AsyncMock(return_value={"response": "ok", "replied_to": []})
        app = _make_test_app(supervisor=supervisor)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat",
                json={"message": "Hi", "intent": "delegation"},
            )

        assert resp.status_code == 200
        supervisor.send_request.assert_awaited_once()
        _, kwargs = supervisor.send_request.await_args
        assert kwargs["params"]["intent"] == "delegation"

    async def test_chat_anima_not_found(self):
        app = _make_test_app({})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/nobody/chat",
                json={"message": "Hi"},
            )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Anima not found: nobody"

    async def test_chat_broadcasts_status(self):
        alice = _make_mock_anima("alice")
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/animas/alice/chat",
                json={"message": "Hi"},
            )

        ws = app.state.ws_manager
        # 2 broadcasts: thinking status + idle status
        assert ws.broadcast.await_count >= 2
        broadcast_types = [call[0][0]["type"] for call in ws.broadcast.call_args_list]
        assert "anima.status" in broadcast_types
        assert "chat.response" not in broadcast_types

    async def test_chat_missing_message_field(self):
        app = _make_test_app({"alice": _make_mock_anima()})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat",
                json={},
            )
        # Pydantic validation error
        assert resp.status_code == 422


# ── POST /animas/{name}/chat/stream ─────────────────────


class TestChatStream:
    async def test_stream_anima_not_found(self):
        app = _make_test_app({})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/nobody/chat/stream",
                json={"message": "Hi"},
            )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Anima not found: nobody"

    async def test_stream_success(self):
        alice = _make_mock_anima("alice")

        async def mock_stream(msg, from_person="human", intent=""):
            yield {"type": "text_delta", "text": "Hello"}
            yield {"type": "cycle_done", "cycle_result": {"summary": "Hello"}}

        alice.process_message_stream = mock_stream

        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "Hi"},
            )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        body = resp.text
        assert "event: text_delta" in body
        assert "event: done" in body

    async def test_stream_done_contains_images(self):
        alice = _make_mock_anima("alice")

        async def mock_stream(msg, from_person="human", intent=""):
            yield {
                "type": "cycle_done",
                "cycle_result": {
                    "summary": "Hello",
                    "images": [{"type": "image", "source": "generated", "path": "assets/avatar.png"}],
                },
            }

        alice.process_message_stream = mock_stream
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "Hi"},
            )
        assert resp.status_code == 200
        assert '"images": [{"type": "image", "source": "generated", "path": "assets/avatar.png"}]' in resp.text

    async def test_stream_done_contains_files(self):
        alice = _make_mock_anima("alice")
        files = [
            {
                "name": "summary.md",
                "path": "attachments/20261029_101500_a1_summary.md",
                "size": 128,
                "media_type": "text/markdown",
            }
        ]

        async def mock_stream(msg, from_person="human", intent=""):
            yield {
                "type": "cycle_done",
                "cycle_result": {"summary": "Hello", "files": files},
            }

        alice.process_message_stream = mock_stream
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "Hi"},
            )
        assert resp.status_code == 200
        assert '"files": [' in resp.text
        assert '"summary.md"' in resp.text

    async def test_stream_forwards_intent_to_supervisor(self):
        supervisor = MagicMock()
        supervisor.processes = {"alice"}
        captured_kwargs = {}

        async def _stream(*args, **kwargs):
            from core.runtime.ipc import IPCResponse

            captured_kwargs.update(kwargs)
            yield IPCResponse(
                id="test",
                stream=True,
                done=True,
                result={"response": "ok", "replied_to": [], "cycle_result": {"summary": "ok"}},
            )

        supervisor.send_request_stream = _stream
        app = _make_test_app(supervisor=supervisor)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "Hi", "intent": "report"},
            )

        assert resp.status_code == 200
        assert captured_kwargs["params"]["intent"] == "report"

    async def test_stream_tool_events(self):
        alice = _make_mock_anima("alice")

        async def mock_stream(msg, from_person="human", intent=""):
            yield {"type": "tool_start", "tool_name": "web_search", "tool_id": "t1"}
            yield {"type": "tool_end", "tool_id": "t1", "tool_name": "web_search"}
            yield {"type": "chain_start", "chain": 2}
            yield {"type": "cycle_done", "cycle_result": {"summary": "Done"}}

        alice.process_message_stream = mock_stream

        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "Hi"},
            )
        body = resp.text
        assert "event: tool_start" in body
        assert "event: tool_end" in body
        assert "event: chain_start" in body

    async def test_stream_error_event(self):
        alice = _make_mock_anima("alice")

        async def mock_stream(msg, from_person="human", intent=""):
            yield {"type": "error", "message": "Something went wrong"}

        alice.process_message_stream = mock_stream

        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "Hi"},
            )
        body = resp.text
        assert "event: error" in body

    async def test_stream_exception_handling(self):
        alice = _make_mock_anima("alice")

        async def mock_stream(msg, from_person="human", intent=""):
            raise RuntimeError("unexpected error")
            yield  # pragma: no cover  # make this an async generator

        alice.process_message_stream = mock_stream

        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/chat/stream",
                json={"message": "Hi"},
            )
        body = resp.text
        assert "event: error" in body
        assert "STREAM_ERROR" in body


# ── POST /animas/{name}/greet ──────────────────────────


class TestGreet:
    async def test_greet_success(self):
        supervisor = MagicMock()
        supervisor.send_request = AsyncMock(
            return_value={
                "response": "こんにちは！待機中です。",
                "emotion": "smile",
                "cached": False,
            }
        )
        app = _make_test_app(supervisor=supervisor)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/greet")

        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "こんにちは！待機中です。"
        assert data["emotion"] == "smile"
        assert data["cached"] is False
        assert data["anima"] == "alice"

    async def test_greet_field_cached_always_false(self):
        """The greet cache was removed; the API keeps ``cached`` as ``false``."""
        supervisor = MagicMock()
        supervisor.send_request = AsyncMock(
            return_value={
                "response": "Hi!",
                "emotion": "neutral",
                "cached": True,
            }
        )
        app = _make_test_app(supervisor=supervisor)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/greet")

        assert resp.status_code == 200
        data = resp.json()
        assert data["cached"] is False

    async def test_greet_anima_not_found(self):
        supervisor = MagicMock()
        supervisor.send_request = AsyncMock(side_effect=KeyError("bob"))
        app = _make_test_app(supervisor=supervisor)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/bob/greet")

        assert resp.status_code == 404

    async def test_greet_ipc_sends_correct_method(self):
        supervisor = MagicMock()
        supervisor.send_request = AsyncMock(
            return_value={
                "response": "Hi",
                "emotion": "neutral",
                "cached": False,
            }
        )
        app = _make_test_app(supervisor=supervisor)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post("/api/animas/alice/greet")

        supervisor.send_request.assert_awaited_once_with(
            anima_name="alice",
            method="greet",
            params={"mode": "visit", "user_name": "", "user_id": ""},
            timeout=120.0,
        )

    async def test_greet_first_meeting_passes_mode(self):
        supervisor = MagicMock()
        supervisor.send_request = AsyncMock(
            return_value={
                "response": "はじめまして！",
                "emotion": "smile",
                "cached": False,
            }
        )
        app = _make_test_app(supervisor=supervisor)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/greet",
                json={"mode": "first_meeting"},
            )

        assert resp.status_code == 200
        assert supervisor.send_request.await_args.kwargs["params"]["mode"] == "first_meeting"

    async def test_greet_rejects_unknown_mode(self):
        supervisor = MagicMock()
        supervisor.send_request = AsyncMock()
        app = _make_test_app(supervisor=supervisor)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/animas/alice/greet",
                json={"mode": "bogus"},
            )

        assert resp.status_code == 422
        supervisor.send_request.assert_not_awaited()


# ── POST /animas/{name}/chat/compact ────────────────────


class TestChatCompact:
    async def test_compact_anima_not_found(self):
        supervisor = MagicMock()
        supervisor.processes = set()
        app = _make_test_app(supervisor=supervisor)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/nobody/chat/compact", json={"thread_id": "default"})
        assert resp.status_code == 404

    async def test_compact_conflict_when_stream_active(self):
        supervisor = MagicMock()
        supervisor.processes = {"alice"}
        app = _make_test_app(supervisor=supervisor)
        app.state.stream_registry.register("alice", thread_id="default")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/chat/compact", json={"thread_id": "default"})
        assert resp.status_code == 409
        assert resp.json()["detail"] == "chat in progress"
        supervisor.send_request.assert_not_called()

    async def test_compact_success(self):
        supervisor = MagicMock()
        supervisor.processes = {"alice"}
        supervisor.send_request = AsyncMock(return_value={"status": "ok", "thread_id": "default", "mode": "s"})
        app = _make_test_app(supervisor=supervisor)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/chat/compact", json={"thread_id": "default"})
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "thread_id": "default", "mode": "s"}
        supervisor.send_request.assert_awaited_once_with(
            anima_name="alice",
            method="compact_session",
            params={"thread_id": "default"},
            timeout=90.0,
        )

    async def test_compact_timeout(self):
        supervisor = MagicMock()
        supervisor.processes = {"alice"}
        supervisor.send_request = AsyncMock(side_effect=TimeoutError)
        app = _make_test_app(supervisor=supervisor)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/chat/compact", json={"thread_id": "default"})
        assert resp.status_code == 504


class TestChatTimelineHuman:
    async def test_chat_records_human_and_answered(self):
        alice = _make_mock_anima("alice")
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        with (
            patch("server.routes.chat._human_for_chat", return_value=("alice", "web")),
            patch("server.routes.chat.record_human", return_value="tl_1234567890abcdef") as record_human,
            patch("server.routes.chat.record_answered") as record_answered,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/animas/alice/chat",
                    json={"message": "こんにちは"},
                )
        assert resp.status_code == 200
        record_human.assert_called_once_with(person="alice", anima="alice", text="こんにちは", via="web")
        record_answered.assert_called_once_with(
            anima="alice",
            person="alice",
            text="Hello from Alice",
            via="web",
            about="tl_1234567890abcdef",
        )

    async def test_chat_failure_records_human_but_not_answered(self):
        alice = _make_mock_anima("alice")
        alice.process_message = AsyncMock(side_effect=RuntimeError("boom"))
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        with (
            patch("server.routes.chat._human_for_chat", return_value=("alice", "web")),
            patch("server.routes.chat.record_human", return_value="tl_1234567890abcdef") as record_human,
            patch("server.routes.chat.record_answered") as record_answered,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/animas/alice/chat",
                    json={"message": "失敗"},
                )
        assert resp.status_code == 500
        record_human.assert_called_once_with(person="alice", anima="alice", text="失敗", via="web")
        record_answered.assert_not_called()

    async def test_chat_does_not_record_without_human(self):
        alice = _make_mock_anima("alice")
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        with (
            patch("server.routes.chat._human_for_chat", return_value=(None, "web")),
            patch("server.routes.chat.record_human") as record_human,
            patch("server.routes.chat.record_answered") as record_answered,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/animas/alice/chat",
                    json={"message": "ローカル"},
                )
        assert resp.status_code == 200
        record_human.assert_not_called()
        record_answered.assert_not_called()


class TestChatStreamTimelineHuman:
    async def test_stream_records_human_and_answered_on_done(self):
        alice = _make_mock_anima("alice")

        async def mock_stream(msg, from_person="human", intent=""):
            yield {"type": "text_delta", "text": "Hello"}
            yield {"type": "cycle_done", "cycle_result": {"summary": "Hello"}}

        alice.process_message_stream = mock_stream
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        with (
            patch("server.routes.chat._human_for_chat", return_value=("alice", "tui")),
            patch("server.routes.chat.record_human", return_value="tl_1234567890abcdef") as record_human,
            patch("core.hub.timeline_human.record_answered") as record_answered,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/animas/alice/chat/stream",
                    json={"message": "Hi"},
                )
        assert resp.status_code == 200
        assert "event: done" in resp.text
        record_human.assert_called_once_with(person="alice", anima="alice", text="Hi", via="tui")
        record_answered.assert_called_once_with(
            anima="alice",
            person="alice",
            text="Hello",
            via="tui",
            about="tl_1234567890abcdef",
        )

    async def test_stream_error_does_not_record_answered(self):
        alice = _make_mock_anima("alice")

        async def mock_stream(msg, from_person="human", intent=""):
            yield {"type": "error", "message": "boom"}

        alice.process_message_stream = mock_stream
        app = _make_test_app({"alice": alice})
        transport = ASGITransport(app=app)
        with (
            patch("server.routes.chat._human_for_chat", return_value=("alice", "web")),
            patch("server.routes.chat.record_human", return_value="tl_1234567890abcdef"),
            patch("core.hub.timeline_human.record_answered") as record_answered,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/animas/alice/chat/stream",
                    json={"message": "Hi"},
                )
        assert resp.status_code == 200
        record_answered.assert_not_called()
