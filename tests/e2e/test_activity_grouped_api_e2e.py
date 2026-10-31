"""E2E tests for grouped activity API endpoint."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from httpx import ASGITransport, AsyncClient

# ── Helpers ──────────────────────────────────────────────


def _create_app(tmp_path: Path, anima_names: list[str] | None = None):
    """Build a real FastAPI app with mocked externals."""
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
    import server.app as _sa

    _auth = MagicMock()
    _auth.auth_mode = "local_trust"
    _sa.load_auth = lambda: _auth
    if anima_names is not None:
        app.state.anima_names = anima_names
    return app


def _setup_anima(animas_dir: Path, name: str) -> Path:
    anima_dir = animas_dir / name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "IDENTITY.md").write_text(f"# {name}", encoding="utf-8")
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


# ── Tests: grouped=false (backward compat) ───────────────


class TestGroupedFalseBackwardCompat:
    """Verify grouped=false (default) returns flat events as before."""

    async def test_default_returns_flat_events(self, tmp_path: Path) -> None:
        animas_dir = tmp_path / "animas"
        _setup_anima(animas_dir, "alice")
        now = datetime.now(UTC)
        _write_activity(
            animas_dir,
            "alice",
            [
                {"ts": now.isoformat(), "type": "heartbeat_start", "summary": "HB", "content": ""},
            ],
        )
        app = _create_app(tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent")
        assert resp.status_code == 200
        data = resp.json()
        assert "events" in data
        assert "groups" not in data
        assert data["total"] == 1

    async def test_explicit_grouped_false(self, tmp_path: Path) -> None:
        app = _create_app(tmp_path, anima_names=[])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?grouped=false")
        data = resp.json()
        assert "events" in data
        assert "groups" not in data


# ── Tests: grouped=true response structure ───────────────


class TestGroupedResponseStructure:
    """Verify grouped=true returns correct structure."""

    async def test_grouped_response_fields(self, tmp_path: Path) -> None:
        animas_dir = tmp_path / "animas"
        _setup_anima(animas_dir, "alice")
        now = datetime.now(UTC)
        _write_activity(
            animas_dir,
            "alice",
            [
                {"ts": (now - timedelta(seconds=3)).isoformat(), "type": "heartbeat_start", "summary": "HB start"},
                {
                    "ts": (now - timedelta(seconds=2)).isoformat(),
                    "type": "channel_read",
                    "summary": "read general",
                    "channel": "general",
                },
                {"ts": (now - timedelta(seconds=1)).isoformat(), "type": "heartbeat_end", "summary": "ok"},
            ],
        )
        app = _create_app(tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?grouped=true")
        assert resp.status_code == 200
        data = resp.json()
        assert "groups" in data
        assert "total_groups" in data
        assert "total_events" in data
        assert "group_offset" in data
        assert "group_limit" in data
        assert "has_more" in data
        assert data["total_groups"] == 1
        assert data["total_events"] == 3

    async def test_group_contains_expected_fields(self, tmp_path: Path) -> None:
        animas_dir = tmp_path / "animas"
        _setup_anima(animas_dir, "alice")
        now = datetime.now(UTC)
        _write_activity(
            animas_dir,
            "alice",
            [
                {"ts": now.isoformat(), "type": "heartbeat_start", "summary": "HB"},
            ],
        )
        app = _create_app(tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?grouped=true")
        data = resp.json()
        grp = data["groups"][0]
        for field in ("id", "type", "anima", "start_ts", "end_ts", "summary", "event_count", "is_open", "events"):
            assert field in grp, f"Missing field: {field}"

    async def test_empty_returns_zero_groups(self, tmp_path: Path) -> None:
        app = _create_app(tmp_path, anima_names=[])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?grouped=true")
        data = resp.json()
        assert data["groups"] == []
        assert data["total_groups"] == 0


# ── Tests: group types ───────────────────────────────────


class TestGroupTypes:
    """Verify correct group type assignment."""

    async def test_heartbeat_group_type(self, tmp_path: Path) -> None:
        animas_dir = tmp_path / "animas"
        _setup_anima(animas_dir, "alice")
        now = datetime.now(UTC)
        _write_activity(
            animas_dir,
            "alice",
            [
                {"ts": (now - timedelta(seconds=2)).isoformat(), "type": "heartbeat_start", "summary": "HB start"},
                {"ts": (now - timedelta(seconds=1)).isoformat(), "type": "heartbeat_end", "summary": "ok"},
            ],
        )
        app = _create_app(tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?grouped=true")
        grp = resp.json()["groups"][0]
        assert grp["type"] == "heartbeat"
        assert grp["is_open"] is False

    async def test_chat_group_type(self, tmp_path: Path) -> None:
        animas_dir = tmp_path / "animas"
        _setup_anima(animas_dir, "alice")
        now = datetime.now(UTC)
        _write_activity(
            animas_dir,
            "alice",
            [
                {
                    "ts": (now - timedelta(seconds=2)).isoformat(),
                    "type": "message_received",
                    "summary": "hi",
                    "content": "hi",
                    "from": "admin",
                    "meta": {"from_type": "human"},
                },
                {
                    "ts": (now - timedelta(seconds=1)).isoformat(),
                    "type": "response_sent",
                    "summary": "hello",
                    "content": "hello",
                },
            ],
        )
        app = _create_app(tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?grouped=true")
        grp = resp.json()["groups"][0]
        assert grp["type"] == "chat"

    async def test_cron_group_type(self, tmp_path: Path) -> None:
        animas_dir = tmp_path / "animas"
        _setup_anima(animas_dir, "alice")
        now = datetime.now(UTC)
        _write_activity(
            animas_dir,
            "alice",
            [
                {
                    "ts": now.isoformat(),
                    "type": "cron_executed",
                    "summary": "check mail",
                    "meta": {"task_name": "check_mail"},
                },
            ],
        )
        app = _create_app(tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?grouped=true")
        grp = resp.json()["groups"][0]
        assert grp["type"] == "cron"


# ── Tests: group pagination ──────────────────────────────


class TestGroupPagination:
    """Verify group_limit and group_offset work correctly."""

    async def test_group_limit(self, tmp_path: Path) -> None:
        animas_dir = tmp_path / "animas"
        _setup_anima(animas_dir, "alice")
        now = datetime.now(UTC)
        entries = []
        for i in range(10):
            entries.append(
                {
                    "ts": (now - timedelta(minutes=10 - i)).isoformat(),
                    "type": "heartbeat_start",
                    "summary": f"HB {i}",
                }
            )
            entries.append(
                {
                    "ts": (now - timedelta(minutes=10 - i, seconds=-30)).isoformat(),
                    "type": "heartbeat_end",
                    "summary": f"ok {i}",
                }
            )
        _write_activity(animas_dir, "alice", entries)
        app = _create_app(tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?grouped=true&group_limit=3")
        data = resp.json()
        assert len(data["groups"]) == 3
        assert data["total_groups"] == 10
        assert data["has_more"] is True

    async def test_group_offset(self, tmp_path: Path) -> None:
        animas_dir = tmp_path / "animas"
        _setup_anima(animas_dir, "alice")
        now = datetime.now(UTC)
        entries = []
        for i in range(5):
            entries.append(
                {
                    "ts": (now - timedelta(minutes=5 - i)).isoformat(),
                    "type": "heartbeat_start",
                    "summary": f"HB {i}",
                }
            )
            entries.append(
                {
                    "ts": (now - timedelta(minutes=5 - i, seconds=-10)).isoformat(),
                    "type": "heartbeat_end",
                    "summary": f"ok {i}",
                }
            )
        _write_activity(animas_dir, "alice", entries)
        app = _create_app(tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?grouped=true&group_offset=3&group_limit=10")
        data = resp.json()
        assert len(data["groups"]) == 2
        assert data["has_more"] is False


# ── Tests: tool pairing in API ───────────────────────────


class TestToolPairingApi:
    """Verify tool_use/tool_result pairing in grouped API response."""

    async def test_tool_result_attached(self, tmp_path: Path) -> None:
        animas_dir = tmp_path / "animas"
        _setup_anima(animas_dir, "alice")
        now = datetime.now(UTC)
        _write_activity(
            animas_dir,
            "alice",
            [
                {"ts": (now - timedelta(seconds=3)).isoformat(), "type": "heartbeat_start", "summary": "HB"},
                {
                    "ts": (now - timedelta(seconds=2)).isoformat(),
                    "type": "tool_use",
                    "tool": "web_search",
                    "summary": "search",
                    "meta": {"tool_use_id": "tu_1"},
                },
                {
                    "ts": (now - timedelta(seconds=1, milliseconds=500)).isoformat(),
                    "type": "tool_result",
                    "tool": "web_search",
                    "content": "3 results found",
                    "meta": {"tool_use_id": "tu_1"},
                },
                {"ts": (now - timedelta(seconds=1)).isoformat(), "type": "heartbeat_end", "summary": "done"},
            ],
        )
        app = _create_app(tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?grouped=true")
        data = resp.json()
        assert data["total_groups"] == 1
        grp = data["groups"][0]
        tool_events = [e for e in grp["events"] if e["type"] == "tool_use"]
        assert len(tool_events) == 1
        assert "tool_result" in tool_events[0]
        assert tool_events[0]["tool_result"]["content"] == "3 results found"
        # tool_result should not appear as standalone
        result_events = [e for e in grp["events"] if e["type"] == "tool_result"]
        assert len(result_events) == 0


# ── Tests: anima filter with grouping ────────────────────


class TestGroupedAnimaFilter:
    """Verify anima filter works with grouped=true."""

    async def test_filter_by_anima(self, tmp_path: Path) -> None:
        animas_dir = tmp_path / "animas"
        _setup_anima(animas_dir, "alice")
        _setup_anima(animas_dir, "bob")
        now = datetime.now(UTC)
        _write_activity(
            animas_dir,
            "alice",
            [
                {"ts": (now - timedelta(seconds=2)).isoformat(), "type": "heartbeat_start", "summary": "alice HB"},
                {"ts": (now - timedelta(seconds=1)).isoformat(), "type": "heartbeat_end", "summary": "alice ok"},
            ],
        )
        _write_activity(
            animas_dir,
            "bob",
            [
                {
                    "ts": now.isoformat(),
                    "type": "cron_executed",
                    "summary": "bob cron",
                    "meta": {"task_name": "backup"},
                },
            ],
        )
        app = _create_app(tmp_path, anima_names=["alice", "bob"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?grouped=true&anima=alice")
        data = resp.json()
        for grp in data["groups"]:
            assert grp["anima"] == "alice"


# ── Test group_type filter ─────────────────────────────────


class TestGroupTypeFilter:
    """Verify group_type filter (trigger-based) works with grouped=true."""

    async def test_filter_by_single_group_type(self, tmp_path: Path) -> None:
        animas_dir = tmp_path / "animas"
        _setup_anima(animas_dir, "alice")
        now = datetime.now(UTC)
        _write_activity(
            animas_dir,
            "alice",
            [
                {"ts": (now - timedelta(seconds=3)).isoformat(), "type": "heartbeat_start", "summary": "HB"},
                {"ts": (now - timedelta(seconds=2)).isoformat(), "type": "heartbeat_end", "summary": "ok"},
                {
                    "ts": (now - timedelta(seconds=1)).isoformat(),
                    "type": "cron_executed",
                    "summary": "Cron",
                    "meta": {"task_name": "backup"},
                },
            ],
        )
        app = _create_app(tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?grouped=true&group_type=cron")
        data = resp.json()
        assert len(data["groups"]) == 1
        assert data["groups"][0]["type"] == "cron"

    async def test_filter_by_multiple_group_types(self, tmp_path: Path) -> None:
        animas_dir = tmp_path / "animas"
        _setup_anima(animas_dir, "alice")
        now = datetime.now(UTC)
        _write_activity(
            animas_dir,
            "alice",
            [
                {"ts": (now - timedelta(seconds=4)).isoformat(), "type": "heartbeat_start", "summary": "HB"},
                {"ts": (now - timedelta(seconds=3)).isoformat(), "type": "heartbeat_end", "summary": "ok"},
                {
                    "ts": (now - timedelta(seconds=2)).isoformat(),
                    "type": "cron_executed",
                    "summary": "Cron",
                    "meta": {"task_name": "backup"},
                },
                {
                    "ts": (now - timedelta(seconds=1)).isoformat(),
                    "type": "channel_post",
                    "summary": "post",
                    "channel": "general",
                    "content": "hello",
                },
            ],
        )
        app = _create_app(tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?grouped=true&group_type=heartbeat,cron")
        data = resp.json()
        types = {g["type"] for g in data["groups"]}
        assert types == {"heartbeat", "cron"}
        assert len(data["groups"]) == 2
