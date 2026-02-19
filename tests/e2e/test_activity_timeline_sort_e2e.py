"""E2E tests for activity timeline sort fix — mixed timestamp format handling.

Verifies that the activity API returns events sorted correctly even when
timestamps use different formats (naive local vs UTC with Z suffix).
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


# ── Helpers ──────────────────────────────────────────────


@contextmanager
def _create_app(tmp_path: Path, anima_names: list[str] | None = None):
    """Build a real FastAPI app via create_app with mocked externals.

    Must be used as a context manager so that patches (especially load_auth)
    remain active when HTTP requests hit the auth_guard middleware.
    """
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
        # Auth: local_trust mode bypasses authentication
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
        if anima_names is not None:
            app.state.anima_names = anima_names
        yield app


def _setup_anima(animas_dir: Path, name: str) -> Path:
    anima_dir = animas_dir / name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "identity.md").write_text(f"# {name}", encoding="utf-8")
    return anima_dir


def _write_activity(animas_dir: Path, name: str, entries: list[dict]) -> None:
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


# ── Test 1: API returns naive timestamps ──────────────────


class TestApiTimestampFormat:
    """Verify the API returns naive ISO timestamps without Z suffix."""

    async def test_activity_ts_format_is_naive(self, tmp_path: Path) -> None:
        animas_dir = tmp_path / "animas"
        _setup_anima(animas_dir, "alice")
        now = datetime.now()
        _write_activity(animas_dir, "alice", [
            {"ts": now.isoformat(), "type": "heartbeat_start", "summary": "HB"},
        ])
        with _create_app(tmp_path, anima_names=["alice"]) as app:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/activity/recent?hours=1")
            data = resp.json()
            assert len(data["events"]) == 1
            ts = data["events"][0]["ts"]
            # Must NOT end with Z (naive local format)
            assert not ts.endswith("Z"), f"Expected naive timestamp, got: {ts}"
            assert "+" not in ts, f"Expected no timezone offset, got: {ts}"


# ── Test 2: Cross-anima events sorted by timestamp ────────


class TestCrossAnimaSort:
    """Events from multiple animas should be sorted by timestamp descending."""

    async def test_cross_anima_sort_order(self, tmp_path: Path) -> None:
        animas_dir = tmp_path / "animas"
        _setup_anima(animas_dir, "alice")
        _setup_anima(animas_dir, "bob")
        now = datetime.now()

        _write_activity(animas_dir, "alice", [
            {"ts": (now - timedelta(minutes=5)).isoformat(), "type": "heartbeat_start", "summary": "alice older"},
            {"ts": (now - timedelta(minutes=1)).isoformat(), "type": "heartbeat_end", "summary": "alice newer"},
        ])
        _write_activity(animas_dir, "bob", [
            {"ts": (now - timedelta(minutes=3)).isoformat(), "type": "dm_sent", "summary": "bob middle", "from": "bob", "to": "alice"},
        ])

        with _create_app(tmp_path, anima_names=["alice", "bob"]) as app:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/activity/recent?hours=1")
            data = resp.json()
            events = data["events"]
            assert len(events) == 3
            # Newest first
            assert events[0]["summary"] == "alice newer"
            assert events[1]["summary"] == "bob middle"
            assert events[2]["summary"] == "alice older"


# ── Test 3: All event types that should appear in timeline ──


class TestEventTypesInApi:
    """Verify message_received, response_sent, dm_sent, human_notify
    all appear in the activity API response (confirming backend records them)."""

    @pytest.mark.parametrize("event_type,extra_fields", [
        ("message_received", {"from": "taka", "content": "hello"}),
        ("response_sent", {"to": "taka", "content": "hi there"}),
        ("dm_sent", {"from": "alice", "to": "bob", "summary": "DM test"}),
        ("dm_received", {"from": "bob", "to": "alice", "summary": "DM reply"}),
        ("human_notify", {"summary": "notification test", "via": "slack"}),
    ])
    async def test_event_type_appears(
        self,
        tmp_path: Path,
        event_type: str,
        extra_fields: dict,
    ) -> None:
        animas_dir = tmp_path / "animas"
        _setup_anima(animas_dir, "alice")
        now = datetime.now()
        entry = {"ts": now.isoformat(), "type": event_type, **extra_fields}
        _write_activity(animas_dir, "alice", [entry])

        with _create_app(tmp_path, anima_names=["alice"]) as app:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/activity/recent?hours=1")
            data = resp.json()
            assert len(data["events"]) == 1
            assert data["events"][0]["type"] == event_type


# ── Test 4: Event type filtering ───────────────────────────


class TestEventTypeFilter:
    """Verify event_type query parameter filters correctly."""

    async def test_filter_message_types(self, tmp_path: Path) -> None:
        animas_dir = tmp_path / "animas"
        _setup_anima(animas_dir, "alice")
        now = datetime.now()
        _write_activity(animas_dir, "alice", [
            {"ts": (now - timedelta(seconds=3)).isoformat(), "type": "heartbeat_start", "summary": "HB"},
            {"ts": (now - timedelta(seconds=2)).isoformat(), "type": "message_received", "from": "taka", "content": "hi"},
            {"ts": (now - timedelta(seconds=1)).isoformat(), "type": "response_sent", "to": "taka", "content": "hello"},
        ])

        with _create_app(tmp_path, anima_names=["alice"]) as app:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/activity/recent?hours=1&event_type=message_received,response_sent"
                )
            data = resp.json()
            assert len(data["events"]) == 2
            types = {e["type"] for e in data["events"]}
            assert types == {"message_received", "response_sent"}


# ── Test 5: to_api_dict includes all required fields ────────


class TestApiDictFields:
    """Verify activity API response includes all fields the frontend needs."""

    async def test_required_fields_present(self, tmp_path: Path) -> None:
        animas_dir = tmp_path / "animas"
        _setup_anima(animas_dir, "alice")
        now = datetime.now()
        _write_activity(animas_dir, "alice", [
            {
                "ts": now.isoformat(),
                "type": "dm_sent",
                "summary": "test message",
                "content": "full content here",
                "from": "alice",
                "to": "bob",
            },
        ])
        with _create_app(tmp_path, anima_names=["alice"]) as app:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/activity/recent?hours=1")
            evt = resp.json()["events"][0]
            # Required fields for frontend rendering
            assert "id" in evt
            assert "ts" in evt
            assert "type" in evt
            assert "anima" in evt
            assert "summary" in evt
            assert "content" in evt
            assert "from_person" in evt
            assert "to_person" in evt
            assert evt["anima"] == "alice"
            assert evt["from_person"] == "alice"
            assert evt["to_person"] == "bob"
