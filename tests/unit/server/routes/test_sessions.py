"""Unit tests for server/routes/sessions.py — Session management endpoints."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import ASGITransport, AsyncClient


def _make_test_app(animas_dir: Path | None = None, anima_names: list[str] | None = None):
    from fastapi import FastAPI

    from server.routes.sessions import create_sessions_router

    app = FastAPI()
    app.state.animas_dir = animas_dir or Path("/tmp/fake/animas")
    app.state.anima_names = anima_names or []
    router = create_sessions_router()
    app.include_router(router, prefix="/api")
    return app


def _write_tool_call_history(anima_dir: Path, *, thread_id: str = "default") -> str:
    log_dir = anima_dir / "activity_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    start = datetime.now().astimezone().replace(microsecond=0)
    tool_ts = start.isoformat()
    result_ts = (start + timedelta(seconds=1)).isoformat()
    message_ts = (start + timedelta(seconds=2)).isoformat()
    events = [
        {
            "ts": tool_ts,
            "type": "tool_use",
            "tool": "Read",
            "ctx": "chat",
            "meta": {
                "args": {"file_path": "/var/work/" + "nested/" * 20 + "report.md"},
                "tool_use_id": "tool-1",
                "thread_id": thread_id,
            },
        },
        {
            "ts": result_ts,
            "type": "tool_result",
            "content": "結果を確認しました。" * 1000,
            "ctx": "chat",
            "meta": {"tool_use_id": "tool-1", "thread_id": thread_id},
        },
        {
            "ts": message_ts,
            "type": "response_sent",
            "content": "Finished",
            "ctx": "chat",
            "meta": {"thread_id": thread_id},
        },
    ]
    path = log_dir / f"{start.date().isoformat()}.jsonl"
    path.write_text("".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events), encoding="utf-8")
    return message_ts


# ── GET /animas/{name}/sessions ─────────────────────────


class TestListSessions:
    async def test_anima_not_found(self, tmp_path):
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/nobody/sessions")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Anima not found: nobody"

    @patch("core.config.models.load_model_config")
    @patch("server.routes.sessions.ConversationMemory")
    @patch("server.routes.sessions.ShortTermMemory")
    async def test_list_sessions_empty(
        self, mock_stm_cls, mock_conv_cls, mock_lmc, tmp_path,
    ):
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        (anima_dir / "episodes").mkdir()

        # ConversationMemory mock
        mock_state = MagicMock()
        mock_state.turns = []
        mock_state.compressed_summary = ""
        mock_conv = MagicMock()
        mock_conv.load.return_value = mock_state
        mock_conv_cls.return_value = mock_conv

        # ShortTermMemory mock
        mock_stm = MagicMock()
        mock_stm._archive_dir = anima_dir / "shortterm" / "archive"
        mock_stm_cls.return_value = mock_stm

        app = _make_test_app(animas_dir=tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/alice/sessions")

        data = resp.json()
        assert data["anima"] == "alice"
        assert data["active_conversation"] is None
        assert data["archived_sessions"] == []
        assert data["episodes"] == []
        assert data["transcripts"] == []


    @patch("core.config.models.load_model_config")
    @patch("server.routes.sessions.ConversationMemory")
    @patch("server.routes.sessions.ShortTermMemory")
    async def test_list_sessions_with_active_conversation(
        self, mock_stm_cls, mock_conv_cls, mock_lmc, tmp_path,
    ):
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        (anima_dir / "episodes").mkdir()

        mock_turn = MagicMock()
        mock_turn.timestamp = "2026-01-01T12:00:00"

        mock_state = MagicMock()
        mock_state.turns = [mock_turn]
        mock_state.compressed_summary = ""
        mock_state.total_turn_count = 1

        mock_conv = MagicMock()
        mock_conv.load.return_value = mock_state
        mock_conv_cls.return_value = mock_conv

        mock_stm = MagicMock()
        mock_stm._archive_dir = anima_dir / "shortterm" / "archive"
        mock_stm_cls.return_value = mock_stm

        app = _make_test_app(animas_dir=tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/alice/sessions")

        data = resp.json()
        assert data["active_conversation"] is not None
        assert data["active_conversation"]["exists"] is True
        assert data["active_conversation"]["turn_count"] == 1


class TestRecordVoiceCall:
    async def test_call_record_is_saved_and_returned_in_conversation_history(self, tmp_path):
        anima_dir = tmp_path / "alice"
        (anima_dir / "activity_log").mkdir(parents=True)
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        payload = {
            "started_at": "2026-10-28T12:00:00+09:00",
            "duration_seconds": 84,
            "thread_id": "default",
            "summary": "証明書の設定を確認しました。",
            "transcript": [
                {"speaker": "user", "text": "証明書の状態はどう？"},
                {"speaker": "assistant", "text": "設定を確認しました。"},
            ],
        }

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            saved = await client.post("/api/animas/alice/conversation/calls", json=payload)
            history = await client.get("/api/animas/alice/conversation/history")

        assert saved.status_code == 200
        assert saved.json()["ok"] is True
        assert history.status_code == 200
        messages = [message for session in history.json()["sessions"] for message in session["messages"]]
        call = next(message for message in messages if message.get("type") == "voice_call")
        assert call["content"] == payload["summary"]
        assert call["started_at"] == payload["started_at"]
        assert call["duration_seconds"] == 84
        assert call["transcript"] == payload["transcript"]

    async def test_call_record_survives_named_thread_file_fallback(self, tmp_path):
        anima_dir = tmp_path / "alice"
        thread_dir = anima_dir / "state" / "conversations"
        thread_dir.mkdir(parents=True)
        (anima_dir / "activity_log").mkdir()
        (thread_dir / "research.json").write_text(json.dumps({
            "turns": [
                {"role": "human", "timestamp": "2026-10-28T10:00:00+09:00", "content": "最初の質問"},
                {"role": "assistant", "timestamp": "2026-10-28T10:01:00+09:00", "content": "最初の回答"},
            ],
        }), encoding="utf-8")
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post("/api/animas/alice/conversation/calls", json={
                "started_at": "2026-10-28T12:00:00+09:00",
                "duration_seconds": 20,
                "thread_id": "research",
                "summary": "調査結果を伝えました。",
                "transcript": [],
            })
            history = await client.get("/api/animas/alice/conversation/history?thread_id=research")

        assert history.status_code == 200
        messages = [message for session in history.json()["sessions"] for message in session["messages"]]
        assert any(message.get("type") == "voice_call" for message in messages)


class TestLazyToolCallHistory:
    async def test_summary_history_is_small_and_full_is_the_default(self, tmp_path):
        anima_dir = tmp_path / "alice"
        message_ts = _write_tool_call_history(anima_dir)
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            full_response = await client.get("/api/animas/alice/conversation/history")
            summary_response = await client.get(
                "/api/animas/alice/conversation/history?tool_detail=summary"
            )

        assert full_response.status_code == 200
        assert summary_response.status_code == 200
        full = next(
            message
            for session in full_response.json()["sessions"]
            for message in session["messages"]
            if message.get("ts") == message_ts
        )
        summary = next(
            message
            for session in summary_response.json()["sessions"]
            for message in session["messages"]
            if message.get("ts") == message_ts
        )
        assert "input" in full["tool_calls"][0]
        assert "result" in full["tool_calls"][0]
        assert "tool_calls_detail" not in full
        assert set(summary["tool_calls"][0]) == {"tool_name", "tool_use_id", "is_error", "target"}
        assert "input" not in summary["tool_calls"][0]
        assert "result" not in summary["tool_calls"][0]
        assert summary["tool_calls"][0]["target"] == (
            "/var/work/" + "nested/" * 20 + "report.md"
        )[:80]
        assert summary["tool_calls_detail"] == "lazy"
        assert len(summary_response.content) < len(full_response.content) / 5

    async def test_named_thread_fallback_tool_calls_can_be_loaded_lazily(self, tmp_path):
        thread_dir = tmp_path / "alice" / "state" / "conversations"
        thread_dir.mkdir(parents=True)
        message_ts = "2026-10-28T10:01:00+09:00"
        (thread_dir / "research.json").write_text(
            json.dumps(
                {
                    "turns": [
                        {"role": "human", "timestamp": "2026-10-28T10:00:00+09:00", "content": "Read this"},
                        {
                            "role": "assistant",
                            "timestamp": message_ts,
                            "content": "Done",
                            "tool_records": [
                                {
                                    "tool_id": "file-tool-1",
                                    "tool_name": "Read",
                                    "input_summary": "/archive/notes.md",
                                    "result_summary": "file content",
                                    "is_error": False,
                                }
                            ],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            history = await client.get(
                "/api/animas/alice/conversation/history?thread_id=research&tool_detail=summary"
            )
            details = await client.get(
                "/api/animas/alice/conversation/tool-calls",
                params={"message_ts": message_ts, "thread_id": "research"},
            )

        assistant = next(
            message
            for session in history.json()["sessions"]
            for message in session["messages"]
            if message.get("role") == "assistant"
        )
        assert history.status_code == 200
        assert assistant["tool_calls_detail"] == "lazy"
        assert assistant["tool_calls"][0]["target"] == "/archive/notes.md"
        assert "input" not in assistant["tool_calls"][0]
        assert details.status_code == 200
        assert details.json()["tool_calls"][0]["result"] == "file content"

    async def test_tool_call_details_return_for_message_and_404_when_missing(self, tmp_path):
        anima_dir = tmp_path / "alice"
        message_ts = _write_tool_call_history(anima_dir)
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            full_history = await client.get("/api/animas/alice/conversation/history")
            details = await client.get(
                "/api/animas/alice/conversation/tool-calls",
                params={"message_ts": message_ts, "thread_id": "default"},
            )
            missing = await client.get(
                "/api/animas/alice/conversation/tool-calls",
                params={"message_ts": "2020-01-01T00:00:00+00:00"},
            )

        full = next(
            message
            for session in full_history.json()["sessions"]
            for message in session["messages"]
            if message.get("ts") == message_ts
        )
        assert details.status_code == 200
        assert details.json()["tool_calls"] == full["tool_calls"]
        assert missing.status_code == 404

    async def test_tool_calls_made_after_a_message_are_returned_with_it(self, tmp_path):
        anima_dir = tmp_path / "alice"
        log_dir = anima_dir / "activity_log"
        log_dir.mkdir(parents=True)
        start = datetime.now().astimezone().replace(microsecond=0)
        message_ts = start.isoformat()
        events = [
            {"ts": message_ts, "type": "response_sent", "content": "Calling done", "ctx": "chat", "meta": {}},
            {
                "ts": (start + timedelta(seconds=1)).isoformat(),
                "type": "tool_use",
                "tool": "Bash",
                "ctx": "chat",
                "meta": {"args": {"command": "echo hi"}, "tool_use_id": "after-1"},
            },
            {
                "ts": (start + timedelta(seconds=2)).isoformat(),
                "type": "tool_result",
                "content": "hi",
                "ctx": "chat",
                "meta": {"tool_use_id": "after-1"},
            },
        ]
        (log_dir / f"{start.date().isoformat()}.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8"
        )
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            details = await client.get(
                "/api/animas/alice/conversation/tool-calls", params={"message_ts": message_ts}
            )
        assert details.status_code == 200
        assert [c["tool_use_id"] for c in details.json()["tool_calls"]] == ["after-1"]

    async def test_tool_call_details_reject_invalid_anima_name(self, tmp_path):
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/api/animas/.private/conversation/tool-calls?message_ts=2026-10-28T00%3A00%3A00"
            )
        assert response.status_code == 400


# ── GET /animas/{name}/sessions/{session_id} ────────────


class TestGetSessionDetail:
    async def test_anima_not_found(self, tmp_path):
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/nobody/sessions/123")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Anima not found: nobody"

    @patch("server.routes.sessions.ShortTermMemory")
    async def test_session_not_found(self, mock_stm_cls, tmp_path):
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        archive_dir = anima_dir / "shortterm" / "archive"
        archive_dir.mkdir(parents=True)

        mock_stm = MagicMock()
        mock_stm._archive_dir = archive_dir
        mock_stm_cls.return_value = mock_stm

        app = _make_test_app(animas_dir=tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/alice/sessions/nonexistent")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Session not found"

    @patch("server.routes.sessions.ShortTermMemory")
    async def test_session_detail_success(self, mock_stm_cls, tmp_path):
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        archive_dir = anima_dir / "shortterm" / "archive"
        archive_dir.mkdir(parents=True)

        session_data = {"trigger": "heartbeat", "turn_count": 3}
        (archive_dir / "20260101.json").write_text(
            json.dumps(session_data), encoding="utf-8"
        )
        (archive_dir / "20260101.md").write_text("# Session log", encoding="utf-8")

        mock_stm = MagicMock()
        mock_stm._archive_dir = archive_dir
        mock_stm_cls.return_value = mock_stm

        app = _make_test_app(animas_dir=tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/alice/sessions/20260101")

        data = resp.json()
        assert data["anima"] == "alice"
        assert data["session_id"] == "20260101"
        assert data["data"]["trigger"] == "heartbeat"
        assert data["markdown"] == "# Session log"

    @patch("server.routes.sessions.ShortTermMemory")
    async def test_session_corrupted_json(self, mock_stm_cls, tmp_path):
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        archive_dir = anima_dir / "shortterm" / "archive"
        archive_dir.mkdir(parents=True)

        (archive_dir / "bad.json").write_text("not json", encoding="utf-8")

        mock_stm = MagicMock()
        mock_stm._archive_dir = archive_dir
        mock_stm_cls.return_value = mock_stm

        app = _make_test_app(animas_dir=tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/alice/sessions/bad")
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Session data corrupted"


class TestNotificationReply:
    async def test_reply_routes_with_context_and_marks_notification_replied(self, tmp_path, monkeypatch):
        from core.activity.logger import ActivityLogger

        anima_dir = tmp_path / "alice"
        activity = ActivityLogger(anima_dir)
        notify = activity.log(
            "human_notify",
            content="Deploy after the certificate check.",
            meta={"subject": "Deployment", "priority": "normal"},
            ctx="chat",
        )
        monkeypatch.setattr("core.paths.get_shared_dir", lambda: tmp_path / "shared")
        monkeypatch.setattr("core.paths.get_animas_dir", lambda: tmp_path)

        messenger = MagicMock()
        with patch("core.messaging.messenger.Messenger", return_value=messenger):
            app = _make_test_app(animas_dir=tmp_path)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/animas/alice/notifications/reply",
                    json={"notification_id": notify.ts, "text": "Proceed with the certificate first."},
                )

        assert response.status_code == 200
        assert response.json()["replied"] is True
        delivery = messenger.receive_external.call_args.kwargs
        assert delivery["source"] == "web"
        assert "Deployment" in delivery["content"]
        assert "Deploy after the certificate check." in delivery["content"]
        assert "Proceed with the certificate first." in delivery["content"]
        assert delivery["external_thread_ts"] == notify.ts

        messages = [
            message
            for session in activity.get_conversation_view(limit=20)["sessions"]
            for message in session["messages"]
        ]
        notify_message = next(message for message in messages if message.get("type") == "human_notify")
        assert notify_message["replied"] is True
        assert notify_message["notification_id"] == notify.ts

    async def test_reply_finds_notification_outside_the_latest_page(self, tmp_path, monkeypatch):
        from core.activity.logger import ActivityLogger

        anima_dir = tmp_path / "alice"
        activity = ActivityLogger(anima_dir)
        notify = activity.log("human_notify", content="Pick A or B", meta={"subject": "Choice"}, ctx="chat")
        for i in range(260):
            activity.log("response_sent", content=f"filler {i}", ctx="chat")
        monkeypatch.setattr("core.paths.get_shared_dir", lambda: tmp_path / "shared")
        monkeypatch.setattr("core.paths.get_animas_dir", lambda: tmp_path)
        messenger = MagicMock()
        with patch("core.messaging.messenger.Messenger", return_value=messenger):
            app = _make_test_app(animas_dir=tmp_path)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                first = await client.post(
                    "/api/animas/alice/notifications/reply", json={"notification_id": notify.ts, "text": "A"}
                )
                again = await client.post(
                    "/api/animas/alice/notifications/reply", json={"notification_id": notify.ts, "text": "B after all"}
                )
        assert first.status_code == 200
        assert again.status_code == 200
        replies = [
            message
            for session in activity.get_conversation_view(limit=20)["sessions"]
            for message in session["messages"]
            if message.get("type") == "human_reply"
        ]
        assert replies and replies[-1]["reply_quote"] == "Choice"

    async def test_unknown_notification_returns_404(self, tmp_path):
        (tmp_path / "alice").mkdir()
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/animas/alice/notifications/reply",
                json={"notification_id": "missing", "text": "reply"},
            )
        assert response.status_code == 404


class TestAwaitingNotifications:
    async def test_counts_unreplied_notifications_per_anima(self, tmp_path):
        from core.activity.logger import ActivityLogger

        ActivityLogger(tmp_path / "alice").log("human_notify", content="Pick one", meta={"subject": "Q"}, ctx="chat")
        (tmp_path / "bob").mkdir()
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/notifications/awaiting")
        assert response.status_code == 200
        body = response.json()
        assert body["animas"] == {"alice": 1}
        assert body["items"]["alice"][0]["subject"] == "Q"
        assert body["items"]["alice"][0]["preview"] == "Pick one"

    async def test_cache_skips_unchanged_logs_and_refreshes_only_changed_anima(self, tmp_path, monkeypatch):
        from core.activity.logger import ActivityLogger

        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        bob_dir = animas_dir / "bob"
        animas_dir.mkdir()
        alice_dir.mkdir()
        bob_dir.mkdir()
        ActivityLogger(alice_dir).log("human_notify", content="First", ctx="chat")

        calls = []
        original_recent = ActivityLogger.recent

        def recent_spy(self, *args, **kwargs):
            calls.append(self.anima_dir.name)
            return original_recent(self, *args, **kwargs)

        monkeypatch.setattr(ActivityLogger, "recent", recent_spy)
        app = _make_test_app(animas_dir=animas_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            first = await client.get("/api/notifications/awaiting")
            second = await client.get("/api/notifications/awaiting")
            ActivityLogger(alice_dir).log("human_notify", content="Second", ctx="chat")
            third = await client.get("/api/notifications/awaiting")

        assert first.status_code == second.status_code == third.status_code == 200
        assert first.json()["animas"] == {"alice": 1}
        assert second.json() == first.json()
        assert third.json()["animas"] == {"alice": 2}
        assert calls == ["alice", "bob", "alice"]


class TestOlderPageNotifications:
    def test_reply_on_newer_page_marks_older_notification(self, tmp_path):
        from core.activity.logger import ActivityLogger
        from server.routes.sessions import _settle_older_notifications

        activity = ActivityLogger(tmp_path / "alice")
        notify = activity.log("human_notify", content="Pick one", meta={"subject": "Q"}, ctx="chat")
        activity.log("human_reply", content="A", from_person="taka", via="slack", ctx="chat")
        older_page = {"sessions": [{"messages": [{"type": "human_notify", "ts": notify.ts, "replied": False}]}]}
        _settle_older_notifications(activity, older_page, "default")
        assert older_page["sessions"][0]["messages"][0]["replied"] is True


# ── GET /animas/{name}/transcripts/{date} ───────────────


class TestGetTranscript:
    async def test_anima_not_found(self, tmp_path):
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/nobody/transcripts/2026-01-01")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Anima not found: nobody"

    @patch("core.config.models.load_model_config")
    @patch("server.routes.sessions.ConversationMemory")
    async def test_get_transcript_success(self, mock_conv_cls, mock_lmc, tmp_path):
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()

        mock_conv = MagicMock()
        mock_conv.load_transcript.return_value = [
            {"role": "user", "content": "Hello", "timestamp": "2026-01-01T12:00:00"}
        ]
        mock_conv_cls.return_value = mock_conv

        app = _make_test_app(animas_dir=tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/alice/transcripts/2026-01-01")

        data = resp.json()
        assert data["anima"] == "alice"
        assert data["date"] == "2026-01-01"
        assert len(data["turns"]) == 1


class TestNotificationReplyTimeline:
    async def test_reply_passes_notification_key_to_delivery(self, tmp_path, monkeypatch):
        from core.activity.logger import ActivityLogger

        anima_dir = tmp_path / "alice"
        activity = ActivityLogger(anima_dir)
        notify = activity.log(
            "human_notify",
            content="Deploy?",
            meta={"subject": "Deployment", "notification_key": "key-1"},
            ctx="chat",
        )
        monkeypatch.setattr("core.paths.get_shared_dir", lambda: tmp_path / "shared")
        monkeypatch.setattr("core.paths.get_animas_dir", lambda: tmp_path)

        with (
            patch("core.notification.reply_routing._resolve_callback_text_reply", return_value=False),
            patch(
                "core.notification.reply_routing.deliver_notification_reply",
                new_callable=AsyncMock,
                return_value=True,
            ) as deliver,
        ):
            app = _make_test_app(animas_dir=tmp_path)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/animas/alice/notifications/reply",
                    json={"notification_id": notify.ts, "text": "Yes"},
                )

        assert response.status_code == 200
        assert deliver.await_args.kwargs["notification_key"] == "key-1"
