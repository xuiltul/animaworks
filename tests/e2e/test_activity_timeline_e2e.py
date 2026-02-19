"""E2E tests for activity timeline: pagination, message logs, and type filtering."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


# ── Helpers ──────────────────────────────────────────────


def _create_app(tmp_path: Path, anima_names: list[str] | None = None):
    """Build a real FastAPI app via create_app with mocked externals."""
    animas_dir = tmp_path / "animas"
    animas_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

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
    # Persist auth mock beyond the with-block for request-time middleware
    import server.app as _sa
    _auth = MagicMock()
    _auth.auth_mode = "local_trust"
    _sa.load_auth = lambda: _auth
    if anima_names is not None:
        app.state.anima_names = anima_names
    return app


def _setup_anima(animas_dir: Path, name: str) -> Path:
    """Create a minimal anima directory."""
    anima_dir = animas_dir / name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "identity.md").write_text(f"# {name}", encoding="utf-8")
    return anima_dir


def _write_activity(animas_dir: Path, name: str, entries: list[dict]) -> None:
    """Write test activity entries to {animas_dir}/{name}/activity_log/{date}.jsonl."""
    log_dir = animas_dir / name / "activity_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    by_date: dict[str, list[dict]] = {}
    for entry in entries:
        date_str = entry["ts"][:10]
        by_date.setdefault(date_str, []).append(entry)
    for date_str, date_entries in by_date.items():
        path = log_dir / f"{date_str}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            for e in date_entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")


# ── Test 1: Pagination ───────────────────────────────────


class TestActivityPagination:
    """Test pagination parameters (offset, limit, total, has_more)."""

    async def test_default_response_structure(self, tmp_path: Path) -> None:
        app = _create_app(tmp_path, anima_names=[])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent")
        assert resp.status_code == 200
        data = resp.json()
        assert "events" in data
        assert "total" in data
        assert "offset" in data
        assert "limit" in data
        assert "has_more" in data
        assert data["offset"] == 0
        assert data["limit"] == 200
        assert data["has_more"] is False

    async def test_limit_parameter(self, tmp_path: Path) -> None:
        animas_dir = tmp_path / "animas"
        _setup_anima(animas_dir, "alice")
        now = datetime.now(timezone.utc)
        entries = [
            {"ts": (now - timedelta(seconds=10 - i)).isoformat(), "type": "heartbeat_start", "summary": f"HB {i}", "content": ""}
            for i in range(10)
        ]
        _write_activity(animas_dir, "alice", entries)
        app = _create_app(tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?limit=3")
        data = resp.json()
        assert len(data["events"]) == 3
        assert data["total"] == 10
        assert data["has_more"] is True
        assert data["limit"] == 3

    async def test_offset_parameter(self, tmp_path: Path) -> None:
        animas_dir = tmp_path / "animas"
        _setup_anima(animas_dir, "alice")
        now = datetime.now(timezone.utc)
        entries = [
            {"ts": (now - timedelta(seconds=5 - i)).isoformat(), "type": "heartbeat_start", "summary": f"HB {i}", "content": ""}
            for i in range(5)
        ]
        _write_activity(animas_dir, "alice", entries)
        app = _create_app(tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?offset=3&limit=10")
        data = resp.json()
        assert len(data["events"]) == 2
        assert data["total"] == 5
        assert data["offset"] == 3
        assert data["has_more"] is False

    async def test_limit_clamped_to_500(self, tmp_path: Path) -> None:
        app = _create_app(tmp_path, anima_names=[])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?limit=9999")
        data = resp.json()
        assert data["limit"] == 500


# ── Test 2: DM activity log entries ──────────────────────


class TestActivityMessageLog:
    """Test that DM activity_log entries (dm_sent/dm_received) appear in activity events."""

    async def test_dm_events_returned(self, tmp_path: Path) -> None:
        animas_dir = tmp_path / "animas"
        _setup_anima(animas_dir, "alice")
        now = datetime.now(timezone.utc)
        _write_activity(animas_dir, "alice", [
            {"ts": now.isoformat(), "type": "dm_sent", "summary": "Hello Bob!", "content": "Hello Bob!", "from": "alice", "to": "bob", "channel": "", "tool": "", "via": "", "meta": {}},
        ])
        app = _create_app(tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent")
        data = resp.json()
        dm_events = [e for e in data["events"] if e["type"] == "dm_sent"]
        assert len(dm_events) == 1
        evt = dm_events[0]
        assert evt["anima"] == "alice"
        assert evt["from_person"] == "alice"
        assert evt["to_person"] == "bob"
        assert "Hello Bob!" in evt["summary"]

    async def test_dm_events_with_anima_filter(self, tmp_path: Path) -> None:
        animas_dir = tmp_path / "animas"
        _setup_anima(animas_dir, "alice")
        _setup_anima(animas_dir, "charlie")
        now = datetime.now(timezone.utc)
        _write_activity(animas_dir, "alice", [
            {"ts": now.isoformat(), "type": "dm_sent", "summary": "A to B", "content": "", "from": "alice", "to": "bob"},
        ])
        _write_activity(animas_dir, "charlie", [
            {"ts": now.isoformat(), "type": "dm_sent", "summary": "C to D", "content": "", "from": "charlie", "to": "dave"},
        ])
        app = _create_app(tmp_path, anima_names=["alice", "charlie"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?anima=alice")
        data = resp.json()
        dm_events = [e for e in data["events"] if e["type"] == "dm_sent"]
        assert len(dm_events) == 1
        assert dm_events[0]["anima"] == "alice"
        assert "A to B" in dm_events[0]["summary"]

    async def test_no_activity_log_dir(self, tmp_path: Path) -> None:
        app = _create_app(tmp_path, anima_names=[])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0


# ── Test 3: Type filter ──────────────────────────────────


class TestActivityTypeFilter:
    """Test event_type filter parameter."""

    async def test_filter_by_single_type(self, tmp_path: Path) -> None:
        animas_dir = tmp_path / "animas"
        _setup_anima(animas_dir, "alice")
        now = datetime.now(timezone.utc)
        _write_activity(animas_dir, "alice", [
            {"ts": now.isoformat(), "type": "heartbeat_start", "summary": "HB 1", "content": ""},
            {"ts": now.isoformat(), "type": "cron_executed", "summary": "Cron 1", "content": ""},
            {"ts": now.isoformat(), "type": "dm_sent", "summary": "msg", "content": "", "from": "alice", "to": "bob"},
        ])
        app = _create_app(tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?event_type=heartbeat_start")
        data = resp.json()
        assert all(e["type"] == "heartbeat_start" for e in data["events"])
        assert data["total"] == 1

    async def test_filter_by_multiple_types(self, tmp_path: Path) -> None:
        animas_dir = tmp_path / "animas"
        _setup_anima(animas_dir, "alice")
        now = datetime.now(timezone.utc)
        _write_activity(animas_dir, "alice", [
            {"ts": now.isoformat(), "type": "heartbeat_start", "summary": "HB", "content": ""},
            {"ts": now.isoformat(), "type": "cron_executed", "summary": "Cron", "content": ""},
            {"ts": now.isoformat(), "type": "dm_sent", "summary": "msg", "content": "", "from": "alice", "to": "bob"},
        ])
        app = _create_app(tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?event_type=heartbeat_start,dm_sent")
        data = resp.json()
        types = {e["type"] for e in data["events"]}
        assert types == {"heartbeat_start", "dm_sent"}
        assert data["total"] == 2


# ── Test 4: Mixed events ─────────────────────────────────


class TestActivityMixedEvents:
    """Test that all event types are properly aggregated."""

    async def test_all_event_types_aggregated(self, tmp_path: Path) -> None:
        animas_dir = tmp_path / "animas"
        _setup_anima(animas_dir, "alice")
        now = datetime.now(timezone.utc)
        _write_activity(animas_dir, "alice", [
            {"ts": (now - timedelta(seconds=3)).isoformat(), "type": "heartbeat_start", "summary": "HB", "content": ""},
            {"ts": (now - timedelta(seconds=2)).isoformat(), "type": "cron_executed", "summary": "Cron", "content": ""},
            {"ts": (now - timedelta(seconds=1)).isoformat(), "type": "dm_sent", "summary": "msg", "content": "", "from": "alice", "to": "bob"},
            {"ts": now.isoformat(), "type": "tool_use", "summary": "web_search", "content": "", "tool": "web_search"},
        ])
        app = _create_app(tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent")
        data = resp.json()
        types = {e["type"] for e in data["events"]}
        assert "heartbeat_start" in types
        assert "cron_executed" in types
        assert "dm_sent" in types
        assert "tool_use" in types
        # All events should have the new format fields
        for evt in data["events"]:
            assert "id" in evt
            assert "ts" in evt
            assert "anima" in evt
            assert evt["anima"] == "alice"
