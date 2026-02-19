"""Unit tests for server/routes/system.py — System endpoints."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from server.routes.system import _parse_cron_jobs


def _make_test_app(
    animas: dict | None = None,
    animas_dir: Path | None = None,
    shared_dir: Path | None = None,
    anima_names: list[str] | None = None,
):
    from fastapi import FastAPI
    from server.routes.system import create_system_router

    app = FastAPI()
    app.state.animas_dir = animas_dir or Path("/tmp/fake/animas")
    app.state.shared_dir = shared_dir or Path("/tmp/fake/shared")
    app.state.anima_names = (
        anima_names if anima_names is not None
        else list((animas or {}).keys())
    )

    # Mock supervisor
    supervisor = MagicMock()
    supervisor.get_all_status.return_value = {}
    supervisor.get_process_status.return_value = {"status": "running", "pid": 1234}
    supervisor.is_scheduler_running.return_value = False
    supervisor.scheduler = None
    supervisor.start_anima = AsyncMock()
    supervisor.stop_anima = AsyncMock()
    supervisor.restart_anima = AsyncMock()
    app.state.supervisor = supervisor

    # Mock ws_manager
    ws_manager = MagicMock()
    ws_manager.active_connections = []
    app.state.ws_manager = ws_manager

    router = create_system_router()
    app.include_router(router, prefix="/api")
    return app


# ── GET /shared/users ────────────────────────────────────


class TestListSharedUsers:
    async def test_no_users_dir(self, tmp_path):
        shared_dir = tmp_path / "shared"
        # Don't create users dir
        app = _make_test_app(shared_dir=shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/shared/users")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_with_users(self, tmp_path):
        shared_dir = tmp_path / "shared"
        users_dir = shared_dir / "users"
        users_dir.mkdir(parents=True)
        (users_dir / "alice").mkdir()
        (users_dir / "bob").mkdir()
        # Also create a file (should be ignored)
        (users_dir / "readme.txt").write_text("ignore", encoding="utf-8")

        app = _make_test_app(shared_dir=shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/shared/users")
        data = resp.json()
        assert "alice" in data
        assert "bob" in data
        assert "readme.txt" not in data


# ── GET /system/status ───────────────────────────────────


class TestSystemStatus:
    async def test_status(self, tmp_path):
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        app = _make_test_app(animas_dir=animas_dir, anima_names=["alice"])
        app.state.supervisor.get_all_status.return_value = {
            "alice": {"status": "running", "pid": 1234},
        }
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/status")
        data = resp.json()
        assert data["animas"] == 1
        assert "processes" in data
        assert data["scheduler_running"] is False

    async def test_status_empty(self, tmp_path):
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        app = _make_test_app(animas_dir=animas_dir, anima_names=[])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/status")
        data = resp.json()
        assert data["animas"] == 0
        assert data["scheduler_running"] is False

    async def test_status_scheduler_running_with_cron(self, tmp_path):
        """scheduler_running should be True when cron.md has active jobs."""
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "cron.md").write_text(
            "# Cron: alice\n\n## Morning Report (毎朝9時)\ntype: llm\nDo report\n",
            encoding="utf-8",
        )

        app = _make_test_app(animas_dir=animas_dir, anima_names=["alice"])
        app.state.supervisor.is_scheduler_running.return_value = True
        app.state.supervisor.get_all_status.return_value = {
            "alice": {"status": "running", "pid": 1234},
        }
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/status")
        data = resp.json()
        assert data["scheduler_running"] is True


# ── POST /system/reload ─────────────────────────────────


class TestReloadAnimas:
    async def test_reload_adds_new_animas(self, tmp_path):
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        shared_dir = tmp_path / "shared"

        # Create a new anima on disk
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir()
        (alice_dir / "identity.md").write_text("# Alice", encoding="utf-8")

        app = _make_test_app(
            animas={},
            animas_dir=animas_dir,
            shared_dir=shared_dir,
            anima_names=[],
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/system/reload")

        data = resp.json()
        assert "alice" in data["added"]
        assert data["total"] == 1

    async def test_reload_removes_deleted_animas(self, tmp_path):
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        shared_dir = tmp_path / "shared"

        app = _make_test_app(
            animas={},
            animas_dir=animas_dir,
            shared_dir=shared_dir,
            anima_names=["deleted"],
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/system/reload")

        data = resp.json()
        assert "deleted" in data["removed"]
        assert data["total"] == 0

    async def test_reload_refreshes_existing(self, tmp_path):
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        shared_dir = tmp_path / "shared"

        alice_dir = animas_dir / "alice"
        alice_dir.mkdir()
        (alice_dir / "identity.md").write_text("# Alice", encoding="utf-8")

        app = _make_test_app(
            animas={},
            animas_dir=animas_dir,
            shared_dir=shared_dir,
            anima_names=["alice"],
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/system/reload")

        data = resp.json()
        assert "alice" in data["refreshed"]

    async def test_reload_no_animas_dir(self, tmp_path):
        animas_dir = tmp_path / "nonexistent"
        shared_dir = tmp_path / "shared"

        app = _make_test_app(
            animas={},
            animas_dir=animas_dir,
            shared_dir=shared_dir,
            anima_names=[],
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/system/reload")
        data = resp.json()
        assert data["total"] == 0

    async def test_reload_skips_disabled_anima(self, tmp_path):
        """An anima with status.json {enabled: false} is NOT added or started on reload."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        shared_dir = tmp_path / "shared"

        # Create anima on disk with identity.md but disabled via status.json
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir()
        (alice_dir / "identity.md").write_text("# Alice", encoding="utf-8")
        (alice_dir / "status.json").write_text(
            json.dumps({"enabled": False}), encoding="utf-8"
        )

        app = _make_test_app(
            animas={},
            animas_dir=animas_dir,
            shared_dir=shared_dir,
            anima_names=[],
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/system/reload")

        data = resp.json()
        # Disabled anima should NOT be added
        assert "alice" not in data["added"]
        assert "alice" not in data["refreshed"]
        assert data["total"] == 0

        # start_anima should NOT have been called
        app.state.supervisor.start_anima.assert_not_awaited()


# ── GET /activity/recent ─────────────────────────────────


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


class TestRecentActivity:
    async def test_activity_no_animas(self):
        app = _make_test_app(anima_names=[])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent")
        data = resp.json()
        assert data["events"] == []
        assert data["total"] == 0
        assert data["offset"] == 0
        assert data["limit"] == 200
        assert data["has_more"] is False

    async def test_activity_with_hours_param(self):
        app = _make_test_app(anima_names=[])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?hours=1")
        assert resp.status_code == 200

    async def test_activity_with_anima_filter(self, tmp_path):
        from datetime import datetime, timezone

        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir()

        now = datetime.now(timezone.utc)
        _write_activity(animas_dir, "alice", [
            {"ts": now.isoformat(), "type": "heartbeat_start", "summary": "巡回", "content": "", "from": "", "to": "", "channel": "", "tool": "", "via": "", "meta": {}},
        ])

        app = _make_test_app(
            animas_dir=animas_dir,
            anima_names=["alice"],
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?anima=alice")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        assert data["events"][0]["anima"] == "alice"
        assert data["events"][0]["type"] == "heartbeat_start"


# ── Activity endpoint: ActivityLogger JSONL integration ──


class TestActivityEndpoint:
    """Tests for activity_log JSONL reading in /activity/recent."""

    async def test_basic_activity_retrieval(self, tmp_path):
        """Activity entries from JSONL files appear in response."""
        from datetime import datetime, timezone

        animas_dir = tmp_path / "animas"
        (animas_dir / "alice").mkdir(parents=True)
        now = datetime.now(timezone.utc)
        _write_activity(animas_dir, "alice", [
            {"ts": now.isoformat(), "type": "heartbeat_start", "summary": "All clear", "content": "", "from": "", "to": "", "channel": "", "tool": "", "via": "", "meta": {}},
        ])

        app = _make_test_app(
            animas_dir=animas_dir,
            anima_names=["alice"],
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?hours=48")
        data = resp.json()

        hb_events = [e for e in data["events"] if e["type"] == "heartbeat_start"]
        assert len(hb_events) == 1
        assert hb_events[0]["anima"] == "alice"
        assert hb_events[0]["summary"] == "All clear"
        # New format includes 'id' field
        assert hb_events[0]["id"].startswith("alice:")

    async def test_filter_by_event_type(self, tmp_path):
        """event_type filter returns only matching types."""
        from datetime import datetime, timezone

        animas_dir = tmp_path / "animas"
        (animas_dir / "alice").mkdir(parents=True)
        now = datetime.now(timezone.utc)
        _write_activity(animas_dir, "alice", [
            {"ts": now.isoformat(), "type": "heartbeat_start", "summary": "HB", "content": ""},
            {"ts": now.isoformat(), "type": "cron_executed", "summary": "Cron", "content": ""},
            {"ts": now.isoformat(), "type": "message_received", "summary": "Msg", "content": "Hello"},
        ])

        app = _make_test_app(
            animas_dir=animas_dir,
            anima_names=["alice"],
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?hours=48&event_type=heartbeat_start")
        data = resp.json()

        assert all(e["type"] == "heartbeat_start" for e in data["events"])
        assert data["total"] == 1

    async def test_filter_by_anima(self, tmp_path):
        """anima filter returns only events from the specified anima."""
        from datetime import datetime, timezone

        animas_dir = tmp_path / "animas"
        (animas_dir / "alice").mkdir(parents=True)
        (animas_dir / "bob").mkdir(parents=True)
        now = datetime.now(timezone.utc)
        _write_activity(animas_dir, "alice", [
            {"ts": now.isoformat(), "type": "heartbeat_start", "summary": "Alice HB", "content": ""},
        ])
        _write_activity(animas_dir, "bob", [
            {"ts": now.isoformat(), "type": "heartbeat_start", "summary": "Bob HB", "content": ""},
        ])

        app = _make_test_app(
            animas_dir=animas_dir,
            anima_names=["alice", "bob"],
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?hours=48&anima=alice")
        data = resp.json()

        assert all(e["anima"] == "alice" for e in data["events"])
        assert data["total"] == 1

    async def test_pagination_offset_limit(self, tmp_path):
        """Offset and limit control pagination correctly."""
        from datetime import datetime, timezone, timedelta

        animas_dir = tmp_path / "animas"
        (animas_dir / "alice").mkdir(parents=True)
        now = datetime.now(timezone.utc)
        entries = []
        for i in range(10):
            ts = (now - timedelta(minutes=10 - i)).isoformat()
            entries.append({"ts": ts, "type": "tool_use", "summary": f"Entry {i}", "content": ""})
        _write_activity(animas_dir, "alice", entries)

        app = _make_test_app(
            animas_dir=animas_dir,
            anima_names=["alice"],
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?hours=48&offset=3&limit=4")
        data = resp.json()

        assert len(data["events"]) == 4
        assert data["total"] == 10
        assert data["offset"] == 3
        assert data["limit"] == 4
        assert data["has_more"] is True

    async def test_empty_activity_log_no_errors(self, tmp_path):
        """No crash when activity_log directory does not exist."""
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        # No activity_log/ directory at all

        app = _make_test_app(
            animas_dir=animas_dir,
            anima_names=["alice"],
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?hours=48")
        assert resp.status_code == 200
        data = resp.json()
        assert data["events"] == []
        assert data["total"] == 0

    async def test_events_sorted_descending(self, tmp_path):
        """Events are sorted by timestamp descending (newest first)."""
        from datetime import datetime, timezone, timedelta

        animas_dir = tmp_path / "animas"
        (animas_dir / "alice").mkdir(parents=True)
        now = datetime.now(timezone.utc)
        _write_activity(animas_dir, "alice", [
            {"ts": (now - timedelta(hours=2)).isoformat(), "type": "heartbeat_start", "summary": "Earlier heartbeat", "content": ""},
            {"ts": now.isoformat(), "type": "cron_executed", "summary": "Later cron", "content": ""},
        ])

        app = _make_test_app(
            animas_dir=animas_dir,
            anima_names=["alice"],
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?hours=48")
        data = resp.json()

        assert len(data["events"]) == 2
        # First should be the later (cron), second the earlier (heartbeat)
        assert data["events"][0]["type"] == "cron_executed"
        assert data["events"][0]["summary"] == "Later cron"
        assert data["events"][1]["type"] == "heartbeat_start"
        assert data["events"][1]["summary"] == "Earlier heartbeat"

    async def test_events_capped_at_default_limit_200(self, tmp_path):
        """No more than 200 events are returned by default."""
        from datetime import datetime, timezone, timedelta

        animas_dir = tmp_path / "animas"
        (animas_dir / "alice").mkdir(parents=True)
        now = datetime.now(timezone.utc)
        entries = []
        for i in range(250):
            ts = (now - timedelta(seconds=250 - i)).isoformat()
            entries.append({"ts": ts, "type": "heartbeat_start", "summary": f"Entry {i}", "content": ""})
        _write_activity(animas_dir, "alice", entries)

        app = _make_test_app(
            animas_dir=animas_dir,
            anima_names=["alice"],
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?hours=48")
        data = resp.json()
        assert len(data["events"]) <= 200
        assert data["total"] == 250
        assert data["has_more"] is True

    async def test_multiple_animas_merged(self, tmp_path):
        """Events from multiple animas are merged and sorted."""
        from datetime import datetime, timezone, timedelta

        animas_dir = tmp_path / "animas"
        (animas_dir / "alice").mkdir(parents=True)
        (animas_dir / "bob").mkdir(parents=True)
        now = datetime.now(timezone.utc)
        _write_activity(animas_dir, "alice", [
            {"ts": (now - timedelta(minutes=2)).isoformat(), "type": "heartbeat_start", "summary": "Alice HB", "content": ""},
        ])
        _write_activity(animas_dir, "bob", [
            {"ts": now.isoformat(), "type": "cron_executed", "summary": "Bob Cron", "content": ""},
        ])

        app = _make_test_app(
            animas_dir=animas_dir,
            anima_names=["alice", "bob"],
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?hours=48")
        data = resp.json()

        assert data["total"] == 2
        # Newest first: Bob's cron, then Alice's heartbeat
        assert data["events"][0]["anima"] == "bob"
        assert data["events"][1]["anima"] == "alice"

    async def test_malformed_jsonl_lines_skipped(self, tmp_path):
        """Malformed JSONL lines are skipped without crashing."""
        from datetime import datetime, timezone

        animas_dir = tmp_path / "animas"
        (animas_dir / "alice").mkdir(parents=True)
        now = datetime.now(timezone.utc)

        good_entry = json.dumps({
            "ts": now.isoformat(),
            "type": "heartbeat_start",
            "summary": "Good entry",
            "content": "",
        }, ensure_ascii=False)
        log_dir = animas_dir / "alice" / "activity_log"
        log_dir.mkdir(parents=True, exist_ok=True)
        date_str = now.isoformat()[:10]
        content = "this is not json\n" + good_entry + "\n" + "{bad json\n"
        (log_dir / f"{date_str}.jsonl").write_text(content, encoding="utf-8")

        app = _make_test_app(
            animas_dir=animas_dir,
            anima_names=["alice"],
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?hours=48")
        data = resp.json()

        hb_events = [e for e in data["events"] if e["type"] == "heartbeat_start"]
        assert len(hb_events) == 1
        assert hb_events[0]["summary"] == "Good entry"


# ── GET /system/connections ──────────────────────────────


class TestSystemConnections:
    async def test_connections_with_active_clients(self):
        app = _make_test_app(anima_names=["alice", "bob"])
        # Simulate 3 active websocket connections
        app.state.ws_manager.active_connections = [
            MagicMock(), MagicMock(), MagicMock(),
        ]

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/connections")
        data = resp.json()
        assert data["websocket"]["connected_clients"] == 3
        assert "alice" in data["processes"]
        assert "bob" in data["processes"]

    async def test_connections_without_active_connections_attr(self):
        app = _make_test_app(anima_names=["alice"])
        # Remove active_connections attribute
        del app.state.ws_manager.active_connections

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/connections")
        data = resp.json()
        assert data["websocket"]["connected_clients"] == 0

    async def test_connections_empty(self):
        app = _make_test_app(anima_names=[])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/connections")
        data = resp.json()
        assert data["websocket"]["connected_clients"] == 0
        assert data["processes"] == {}


# ── GET /system/scheduler ────────────────────────────────


class TestSystemScheduler:
    async def test_no_cron_files(self, tmp_path):
        """No cron.md files -> running=False, empty jobs."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        app = _make_test_app(animas_dir=animas_dir, anima_names=["alice"])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/scheduler")
        data = resp.json()
        assert data["running"] is False
        assert data["anima_jobs"] == []

    async def test_no_animas(self, tmp_path):
        """No registered animas -> running=False, empty jobs."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        app = _make_test_app(animas_dir=animas_dir, anima_names=[])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/scheduler")
        data = resp.json()
        assert data["running"] is False
        assert data["anima_jobs"] == []

    async def test_scheduler_with_cron_jobs(self, tmp_path):
        """cron.md with active jobs -> running=True, jobs populated."""
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "cron.md").write_text(
            "# Cron: alice\n\n"
            "## Morning Report (毎朝9時)\n"
            "type: llm\n"
            "朝の報告をまとめる\n",
            encoding="utf-8",
        )

        app = _make_test_app(animas_dir=animas_dir, anima_names=["alice"])
        app.state.supervisor.is_scheduler_running.return_value = True
        mock_scheduler = MagicMock()
        mock_scheduler.get_jobs.return_value = []
        app.state.supervisor.scheduler = mock_scheduler

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/scheduler")
        data = resp.json()
        assert data["running"] is True
        assert len(data["anima_jobs"]) == 1
        job = data["anima_jobs"][0]
        assert job["anima"] == "alice"
        assert job["type"] == "llm"
        assert "Morning Report" in job["name"]
        assert job["schedule"] == "毎朝9時"
        assert job["next_run"] is None

    async def test_scheduler_multiple_animas(self, tmp_path):
        """Jobs from multiple animas are aggregated."""
        animas_dir = tmp_path / "animas"
        for name in ("alice", "bob"):
            d = animas_dir / name
            d.mkdir(parents=True)
            (d / "cron.md").write_text(
                f"# Cron: {name}\n\n## Task ({name} schedule)\ntype: llm\nDo work\n",
                encoding="utf-8",
            )

        app = _make_test_app(
            animas_dir=animas_dir, anima_names=["alice", "bob"],
        )
        app.state.supervisor.is_scheduler_running.return_value = True
        mock_scheduler = MagicMock()
        mock_scheduler.get_jobs.return_value = []
        app.state.supervisor.scheduler = mock_scheduler

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/scheduler")
        data = resp.json()
        assert data["running"] is True
        assert len(data["anima_jobs"]) == 2
        animas_in_jobs = {j["anima"] for j in data["anima_jobs"]}
        assert animas_in_jobs == {"alice", "bob"}

    async def test_scheduler_commented_sections_ignored(self, tmp_path):
        """Sections inside HTML comments should not produce jobs."""
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "cron.md").write_text(
            "# Cron: alice\n\n"
            "<!--\n"
            "## Disabled Task (noon)\n"
            "type: llm\n"
            "Should be ignored\n"
            "-->\n",
            encoding="utf-8",
        )

        app = _make_test_app(animas_dir=animas_dir, anima_names=["alice"])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/scheduler")
        data = resp.json()
        assert data["running"] is False
        assert data["anima_jobs"] == []

    async def test_scheduler_mixed_active_and_commented(self, tmp_path):
        """Active jobs are returned; commented-out ones are skipped."""
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "cron.md").write_text(
            "# Cron: alice\n\n"
            "## Active Task (every 5 min)\n"
            "type: llm\n"
            "Do active work\n\n"
            "<!--\n"
            "## Disabled Task (noon)\n"
            "type: llm\n"
            "Should be ignored\n"
            "-->\n",
            encoding="utf-8",
        )

        app = _make_test_app(animas_dir=animas_dir, anima_names=["alice"])
        app.state.supervisor.is_scheduler_running.return_value = True
        mock_scheduler = MagicMock()
        mock_scheduler.get_jobs.return_value = []
        app.state.supervisor.scheduler = mock_scheduler

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/scheduler")
        data = resp.json()
        assert data["running"] is True
        assert len(data["anima_jobs"]) == 1
        assert "Active Task" in data["anima_jobs"][0]["name"]


# ── _parse_cron_jobs (unit) ────────────────────────────────


class TestParseCronJobs:
    def test_empty_animas_list(self, tmp_path):
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        assert _parse_cron_jobs(animas_dir, []) == []

    def test_no_cron_file(self, tmp_path):
        animas_dir = tmp_path / "animas"
        (animas_dir / "alice").mkdir(parents=True)
        assert _parse_cron_jobs(animas_dir, ["alice"]) == []

    def test_single_job(self, tmp_path):
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "cron.md").write_text(
            "# Cron: alice\n\n"
            "## Daily Summary (毎日18時)\n"
            "type: llm\n"
            "まとめを作成する\n",
            encoding="utf-8",
        )
        jobs = _parse_cron_jobs(animas_dir, ["alice"])
        assert len(jobs) == 1
        assert jobs[0]["anima"] == "alice"
        assert jobs[0]["type"] == "llm"
        assert jobs[0]["schedule"] == "毎日18時"
        assert "Daily Summary" in jobs[0]["name"]
        assert jobs[0]["next_run"] is None
        assert jobs[0]["id"].startswith("cron-alice-")

    def test_multiple_jobs_same_anima(self, tmp_path):
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "cron.md").write_text(
            "# Cron: alice\n\n"
            "## Morning Report (毎朝9時)\n"
            "type: llm\n"
            "朝の報告\n\n"
            "## Evening Summary (毎夕18時)\n"
            "type: llm\n"
            "夕方のまとめ\n",
            encoding="utf-8",
        )
        jobs = _parse_cron_jobs(animas_dir, ["alice"])
        assert len(jobs) == 2
        assert jobs[0]["name"] == "Morning Report (毎朝9時)"
        assert jobs[1]["name"] == "Evening Summary (毎夕18時)"

    def test_commented_section_skipped(self, tmp_path):
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "cron.md").write_text(
            "# Cron: alice\n\n"
            "<!--\n"
            "## Disabled (noon)\n"
            "type: llm\n"
            "This should be ignored\n"
            "-->\n",
            encoding="utf-8",
        )
        jobs = _parse_cron_jobs(animas_dir, ["alice"])
        assert jobs == []

    def test_mixed_active_and_commented(self, tmp_path):
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "cron.md").write_text(
            "# Cron: alice\n\n"
            "## Active (every hour)\n"
            "type: llm\n"
            "Do work\n\n"
            "<!--\n"
            "## Disabled (noon)\n"
            "type: llm\n"
            "Skip\n"
            "-->\n\n"
            "## Also Active (daily)\n"
            "type: llm\n"
            "More work\n",
            encoding="utf-8",
        )
        jobs = _parse_cron_jobs(animas_dir, ["alice"])
        assert len(jobs) == 2
        names = [j["name"] for j in jobs]
        assert "Active (every hour)" in names
        assert "Also Active (daily)" in names

    def test_multiple_animas(self, tmp_path):
        animas_dir = tmp_path / "animas"
        for name in ("alice", "bob"):
            d = animas_dir / name
            d.mkdir(parents=True)
            (d / "cron.md").write_text(
                f"# Cron: {name}\n\n## Task ({name})\ntype: llm\nWork\n",
                encoding="utf-8",
            )
        jobs = _parse_cron_jobs(animas_dir, ["alice", "bob"])
        assert len(jobs) == 2
        animas = {j["anima"] for j in jobs}
        assert animas == {"alice", "bob"}

    def test_schedule_extracted_from_parentheses(self, tmp_path):
        """Schedule info inside parentheses (both half-width and full-width)."""
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "cron.md").write_text(
            "# Cron: alice\n\n"
            "## Task（全角括弧）\n"
            "type: llm\n"
            "Work\n",
            encoding="utf-8",
        )
        jobs = _parse_cron_jobs(animas_dir, ["alice"])
        assert len(jobs) == 1
        assert jobs[0]["schedule"] == "全角括弧"

    def test_no_schedule_in_title(self, tmp_path):
        """Job title without parentheses -> empty schedule."""
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "cron.md").write_text(
            "# Cron: alice\n\n"
            "## Simple Task\n"
            "type: llm\n"
            "Work\n",
            encoding="utf-8",
        )
        jobs = _parse_cron_jobs(animas_dir, ["alice"])
        assert len(jobs) == 1
        assert jobs[0]["schedule"] == ""

    def test_anima_without_cron_skipped(self, tmp_path):
        """Anima directory exists but has no cron.md."""
        animas_dir = tmp_path / "animas"
        (animas_dir / "alice").mkdir(parents=True)
        bob_dir = animas_dir / "bob"
        bob_dir.mkdir(parents=True)
        (bob_dir / "cron.md").write_text(
            "# Cron: bob\n\n## Task (hourly)\ntype: llm\nWork\n",
            encoding="utf-8",
        )
        jobs = _parse_cron_jobs(animas_dir, ["alice", "bob"])
        assert len(jobs) == 1
        assert jobs[0]["anima"] == "bob"
