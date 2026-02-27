# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests: activity_log-based conversation history API.

Verifies that:
1. get_conversation_view() correctly builds conversation structure
2. Tool calls are paired by tool_use_id and nested into assistant messages
3. Sessions are separated by 10-minute gaps
4. Cursor-based pagination works for infinite scroll
5. Heartbeat/cron events appear as system messages
6. The API endpoint returns the correct format
7. /conversation/full is removed
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from core.memory.activity import ActivityLogger


# ── Fixtures ──────────────────────────────────────────────────

JST = timezone(timedelta(hours=9))


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    """Create a minimal anima directory with activity_log."""
    d = tmp_path / "animas" / "test-anima"
    for subdir in ("activity_log", "state", "episodes"):
        (d / subdir).mkdir(parents=True)
    return d


def _ts(minutes_ago: int = 0) -> str:
    """Generate ISO timestamp relative to now."""
    t = datetime.now(JST) - timedelta(minutes=minutes_ago)
    return t.isoformat()


def _write_entry(anima_dir: Path, entry: dict) -> None:
    """Write a raw JSONL entry to today's activity log."""
    log_dir = anima_dir / "activity_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    date_str = entry["ts"][:10]
    path = log_dir / f"{date_str}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Test: Basic Conversation View ────────────────────────────


def test_basic_conversation_view(anima_dir: Path) -> None:
    """A simple human→assistant conversation is returned correctly."""
    al = ActivityLogger(anima_dir)
    al.log("message_received", content="こんにちは", from_person="admin", channel="chat")
    al.log("response_sent", content="こんにちは！元気ですか？", to_person="admin", channel="chat")

    result = al.get_conversation_view(limit=50)

    assert "sessions" in result
    assert len(result["sessions"]) == 1

    session = result["sessions"][0]
    assert session["trigger"] == "chat"
    assert len(session["messages"]) == 2

    msg0 = session["messages"][0]
    assert msg0["role"] == "human"
    assert msg0["content"] == "こんにちは"
    assert msg0["from_person"] == "admin"

    msg1 = session["messages"][1]
    assert msg1["role"] == "assistant"
    assert "元気ですか" in msg1["content"]
    assert msg1["tool_calls"] == []


# ── Test: Tool Call Pairing by tool_use_id ───────────────────


def test_tool_call_pairing_by_id(anima_dir: Path) -> None:
    """Tool calls are paired with results via tool_use_id."""
    al = ActivityLogger(anima_dir)

    al.log("message_received", content="天気を調べて", from_person="user", channel="chat")
    al.log("tool_use", tool="web_search", content="query: 東京 天気",
           meta={"tool_use_id": "tu_001", "args": {"query": "東京 天気"}})
    al.log("tool_result", tool="web_search", content="東京は晴れ、気温20度",
           meta={"tool_use_id": "tu_001", "is_error": False})
    al.log("response_sent", content="東京は晴れで気温20度です。", to_person="user", channel="chat")

    result = al.get_conversation_view(limit=50)
    session = result["sessions"][0]

    # Should have 2 messages: human + assistant
    assert len(session["messages"]) == 2

    assistant_msg = session["messages"][1]
    assert assistant_msg["role"] == "assistant"
    assert len(assistant_msg["tool_calls"]) == 1

    tc = assistant_msg["tool_calls"][0]
    assert tc["tool_use_id"] == "tu_001"
    assert tc["tool_name"] == "web_search"
    assert tc["result"] == "東京は晴れ、気温20度"
    assert tc["is_error"] is False


def test_assistant_images_restored_from_response_meta(anima_dir: Path) -> None:
    """response_sent.meta.images should be restored into assistant message payload."""
    _write_entry(anima_dir, {
        "ts": _ts(2),
        "type": "message_received",
        "content": "画像を見せて",
        "from": "user",
        "channel": "chat",
    })
    _write_entry(anima_dir, {
        "ts": _ts(1),
        "type": "response_sent",
        "content": "こちらです",
        "to": "user",
        "channel": "chat",
        "meta": {
            "images": [
                {
                    "type": "image",
                    "source": "generated",
                    "path": "assets/avatar_fullbody.png",
                }
            ]
        },
    })

    al = ActivityLogger(anima_dir)
    result = al.get_conversation_view(limit=50)
    assistant = result["sessions"][0]["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant["images"][0]["path"] == "assets/avatar_fullbody.png"


def test_tool_call_pairing_fallback(anima_dir: Path) -> None:
    """Tool calls without tool_use_id fall back to timestamp+name matching."""
    al = ActivityLogger(anima_dir)

    al.log("message_received", content="検索して", from_person="user", channel="chat")
    # No tool_use_id in tool_use entry (legacy data)
    al.log("tool_use", tool="web_search", content="query: test")
    al.log("tool_result", tool="web_search", content="result data")
    al.log("response_sent", content="結果です", to_person="user", channel="chat")

    result = al.get_conversation_view(limit=50)
    session = result["sessions"][0]

    assistant_msg = session["messages"][1]
    assert len(assistant_msg["tool_calls"]) == 1
    # Fallback should still pair
    assert assistant_msg["tool_calls"][0]["result"] == "result data"


def test_blocked_tool_call(anima_dir: Path) -> None:
    """Blocked tool calls show error state."""
    al = ActivityLogger(anima_dir)

    al.log("message_received", content="rm -rf /", from_person="user", channel="chat")
    al.log("tool_use", tool="Bash", content="rm -rf /",
           meta={"tool_use_id": "tu_blocked", "args": {"command": "rm -rf /"},
                  "blocked": True, "reason": "dangerous command"})
    al.log("response_sent", content="実行できません", to_person="user", channel="chat")

    result = al.get_conversation_view(limit=50)
    tc = result["sessions"][0]["messages"][1]["tool_calls"][0]
    assert tc["is_error"] is True
    assert "ブロック" in tc["result"]


# ── Test: Session Boundaries ─────────────────────────────────


def test_session_boundary_by_gap(anima_dir: Path) -> None:
    """Messages separated by >10 minutes create separate sessions."""
    # Session 1: 20 minutes ago
    _write_entry(anima_dir, {
        "ts": _ts(20), "type": "message_received",
        "content": "Session 1 msg", "from": "user", "channel": "chat",
    })
    _write_entry(anima_dir, {
        "ts": _ts(19), "type": "response_sent",
        "content": "Session 1 reply", "to": "user", "channel": "chat",
    })
    # Session 2: 5 minutes ago (>10 min gap from session 1)
    _write_entry(anima_dir, {
        "ts": _ts(5), "type": "message_received",
        "content": "Session 2 msg", "from": "user", "channel": "chat",
    })
    _write_entry(anima_dir, {
        "ts": _ts(4), "type": "response_sent",
        "content": "Session 2 reply", "to": "user", "channel": "chat",
    })

    al = ActivityLogger(anima_dir)
    result = al.get_conversation_view(limit=50)

    assert len(result["sessions"]) == 2
    assert result["sessions"][0]["messages"][0]["content"] == "Session 1 msg"
    assert result["sessions"][1]["messages"][0]["content"] == "Session 2 msg"


def test_no_session_boundary_within_gap(anima_dir: Path) -> None:
    """Messages within 10 minutes stay in the same session."""
    _write_entry(anima_dir, {
        "ts": _ts(12), "type": "message_received",
        "content": "msg1", "from": "user", "channel": "chat",
    })
    _write_entry(anima_dir, {
        "ts": _ts(10), "type": "response_sent",
        "content": "reply1", "to": "user", "channel": "chat",
    })
    _write_entry(anima_dir, {
        "ts": _ts(5), "type": "message_received",
        "content": "msg2", "from": "user", "channel": "chat",
    })

    al = ActivityLogger(anima_dir)
    result = al.get_conversation_view(limit=50)

    assert len(result["sessions"]) == 1
    assert len(result["sessions"][0]["messages"]) == 3


# ── Test: Heartbeat and Cron ─────────────────────────────────


def test_heartbeat_in_timeline(anima_dir: Path) -> None:
    """Heartbeat events appear as system messages with correct trigger."""
    al = ActivityLogger(anima_dir)

    al.log("heartbeat_start", summary="定期巡回開始")
    al.log("heartbeat_end", summary="特に異常なし")

    result = al.get_conversation_view(limit=50)

    assert len(result["sessions"]) == 1
    session = result["sessions"][0]
    assert session["trigger"] == "heartbeat"
    assert len(session["messages"]) == 2
    assert session["messages"][0]["role"] == "system"
    assert session["messages"][1]["role"] == "system"


def test_cron_in_timeline(anima_dir: Path) -> None:
    """Cron events appear as system messages with correct trigger."""
    al = ActivityLogger(anima_dir)

    al.log("cron_executed", summary="日次レポート生成完了",
           meta={"task_name": "daily_report", "exit_code": 0})

    result = al.get_conversation_view(limit=50)

    session = result["sessions"][0]
    assert session["trigger"] == "cron"
    assert session["messages"][0]["role"] == "system"
    assert "日次レポート" in session["messages"][0]["content"]


# ── Test: Cursor Pagination ──────────────────────────────────


def test_cursor_pagination(anima_dir: Path) -> None:
    """Cursor-based pagination returns older messages and correct has_more."""
    # Create 10 messages (5 turns)
    for i in range(10, 0, -1):
        etype = "message_received" if i % 2 == 0 else "response_sent"
        _write_entry(anima_dir, {
            "ts": _ts(i), "type": etype,
            "content": f"Message {i}",
            "from": "user" if etype == "message_received" else "",
            "to": "user" if etype == "response_sent" else "",
            "channel": "chat",
        })

    al = ActivityLogger(anima_dir)

    # Page 1: latest 3 messages
    result1 = al.get_conversation_view(limit=3)
    all_msgs1 = [m for s in result1["sessions"] for m in s["messages"]]
    assert len(all_msgs1) == 3
    assert result1["has_more"] is True
    assert result1["next_before"] is not None

    # Page 2: use cursor
    result2 = al.get_conversation_view(limit=3, before=result1["next_before"])
    all_msgs2 = [m for s in result2["sessions"] for m in s["messages"]]
    assert len(all_msgs2) == 3

    # No overlap between pages
    ts_set1 = {m["ts"] for m in all_msgs1}
    ts_set2 = {m["ts"] for m in all_msgs2}
    assert ts_set1.isdisjoint(ts_set2), "Pages should not overlap"


def test_no_more_pages(anima_dir: Path) -> None:
    """When all messages fit in one page, has_more is False."""
    al = ActivityLogger(anima_dir)
    al.log("message_received", content="only message", from_person="user", channel="chat")

    result = al.get_conversation_view(limit=50)
    assert result["has_more"] is False
    assert result["next_before"] is None


# ── Test: Empty Activity Log ─────────────────────────────────


def test_empty_activity_log(anima_dir: Path) -> None:
    """Empty activity log returns empty sessions."""
    al = ActivityLogger(anima_dir)
    result = al.get_conversation_view(limit=50)

    assert result["sessions"] == []
    assert result["has_more"] is False
    assert result["next_before"] is None


# ── Test: Error Events ───────────────────────────────────────


def test_error_in_timeline(anima_dir: Path) -> None:
    """Error events appear as system messages."""
    al = ActivityLogger(anima_dir)

    al.log("message_received", content="何かして", from_person="user", channel="chat")
    al.log("error", content="RuntimeError: something went wrong", summary="something went wrong")

    result = al.get_conversation_view(limit=50)

    messages = result["sessions"][0]["messages"]
    error_msg = [m for m in messages if "エラー" in m["content"]]
    assert len(error_msg) >= 1


# ── Test: Multiple Tool Calls in One Turn ────────────────────


def test_multiple_tool_calls_in_one_turn(anima_dir: Path) -> None:
    """Multiple tool calls are all nested into the assistant message."""
    al = ActivityLogger(anima_dir)

    al.log("message_received", content="2つ調べて", from_person="user", channel="chat")
    al.log("tool_use", tool="web_search", content="query1",
           meta={"tool_use_id": "tu_a", "args": {"query": "topic1"}})
    al.log("tool_result", tool="web_search", content="Result for topic1",
           meta={"tool_use_id": "tu_a"})
    al.log("tool_use", tool="web_search", content="query2",
           meta={"tool_use_id": "tu_b", "args": {"query": "topic2"}})
    al.log("tool_result", tool="web_search", content="Result for topic2",
           meta={"tool_use_id": "tu_b"})
    al.log("response_sent", content="2つの結果をまとめました", to_person="user", channel="chat")

    result = al.get_conversation_view(limit=50)

    assistant_msg = result["sessions"][0]["messages"][1]
    assert len(assistant_msg["tool_calls"]) == 2
    assert assistant_msg["tool_calls"][0]["tool_use_id"] == "tu_a"
    assert assistant_msg["tool_calls"][1]["tool_use_id"] == "tu_b"


# ── Test: API Endpoint ───────────────────────────────────────


def _create_test_app(tmp_path: Path, anima_names: list[str] | None = None):
    """Build a real FastAPI app with mocked externals for testing."""
    from unittest.mock import MagicMock

    animas_dir = tmp_path / "animas"
    animas_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

    for name in (anima_names or []):
        d = animas_dir / name
        for subdir in ("activity_log", "state", "episodes"):
            (d / subdir).mkdir(parents=True, exist_ok=True)

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
        supervisor.get_process_status.return_value = {"status": "stopped", "pid": None}
        supervisor.is_scheduler_running.return_value = False
        supervisor.scheduler = None
        mock_sup_cls.return_value = supervisor

        ws_manager = MagicMock()
        ws_manager.active_connections = []
        mock_ws_cls.return_value = ws_manager

        from server.app import create_app
        app = create_app(animas_dir, shared_dir)

    # Persist auth mock so middleware returns local_trust
    import server.app as _sa
    _auth = MagicMock()
    _auth.auth_mode = "local_trust"
    _sa.load_auth = lambda: _auth

    return app, animas_dir, shared_dir


@pytest.mark.asyncio
async def test_api_endpoint_conversation_history(tmp_path: Path) -> None:
    """The /conversation/history endpoint returns correct structure."""
    from httpx import ASGITransport, AsyncClient

    app, animas_dir, _ = _create_test_app(tmp_path, anima_names=["api-test-anima"])

    # Record some activity
    al = ActivityLogger(animas_dir / "api-test-anima")
    al.log("message_received", content="API test message", from_person="tester", channel="chat")
    al.log("response_sent", content="API test response", to_person="tester", channel="chat")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/animas/api-test-anima/conversation/history?limit=50")
        assert resp.status_code == 200

        data = resp.json()
        assert "sessions" in data
        assert "has_more" in data
        assert "next_before" in data

        if data["sessions"]:
            session = data["sessions"][0]
            assert "session_start" in session
            assert "session_end" in session
            assert "trigger" in session
            assert "messages" in session


@pytest.mark.asyncio
async def test_api_endpoint_conversation_full_removed(tmp_path: Path) -> None:
    """The /conversation/full endpoint is removed and returns 404 or 405."""
    from httpx import ASGITransport, AsyncClient

    app, _, _ = _create_test_app(tmp_path, anima_names=["removed-test-anima"])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/animas/removed-test-anima/conversation/full")
        # Should be 404 (route removed) or 405
        assert resp.status_code in (404, 405, 422)
