# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for dashboard API fixes.

Validates the following changes through the full FastAPI app stack:

1. system.py — system_status and system_scheduler now parse cron.md files
   from animas directories instead of relying on a scheduler attribute.
2. config_routes.py — init_status now returns a ``checks`` array alongside
   the existing backward-compatible fields.
3. system.py — connections endpoint returns websocket and process info.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


# ── Helpers ──────────────────────────────────────────────


def _create_app(
    tmp_path: Path,
    anima_names: list[str] | None = None,
    ws_connections: int = 0,
):
    """Build a real FastAPI app via create_app with mocked externals.

    Returns an app whose setup_complete flag is True so the setup-guard
    middleware lets API requests through.
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

    # Persist auth mock beyond the with-block so auth_guard middleware
    # returns local_trust on every request (the with-block patch is
    # restored on exit, but the middleware calls load_auth at request time).
    import server.app as _sa
    _auth = MagicMock()
    _auth.auth_mode = "local_trust"
    _sa.load_auth = lambda: _auth

    # Override anima_names if specified
    if anima_names is not None:
        app.state.anima_names = anima_names

    return app


def _write_cron_md(animas_dir: Path, name: str, content: str) -> None:
    """Write a cron.md file for an anima."""
    anima_dir = animas_dir / name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "identity.md").write_text(f"# {name}", encoding="utf-8")
    (anima_dir / "cron.md").write_text(content, encoding="utf-8")


CRON_SAKURA = """\
# Cron: sakura

## Morning Planning (Daily 9:00 JST)
type: llm
Plan daily tasks.

## Weekly Review (Friday 17:00 JST)
type: llm
Review weekly episodes.

<!--
## Commented Out (Daily 2:00)
type: command
This should be ignored.
-->
"""

CRON_TARO = """\
# Cron: taro

## Report Generation (Monday 10:00 JST)
type: command
Generate weekly report.
"""


# ── Test 1: Scheduler status WITH cron.md ────────────────


class TestSchedulerWithCronMd:
    """Verify system_status and system_scheduler parse cron.md correctly."""

    async def test_system_status_scheduler_running_true_with_cron(
        self, tmp_path: Path,
    ) -> None:
        """scheduler_running should be True when animas have cron.md files."""
        animas_dir = tmp_path / "animas"
        _write_cron_md(animas_dir, "sakura", CRON_SAKURA)

        app = _create_app(tmp_path, anima_names=["sakura"])
        app.state.supervisor.is_scheduler_running.return_value = True
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["scheduler_running"] is True

    async def test_system_scheduler_returns_parsed_jobs(
        self, tmp_path: Path,
    ) -> None:
        """Scheduler endpoint should return jobs parsed from cron.md."""
        animas_dir = tmp_path / "animas"
        _write_cron_md(animas_dir, "sakura", CRON_SAKURA)

        app = _create_app(tmp_path, anima_names=["sakura"])
        app.state.supervisor.is_scheduler_running.return_value = True
        app.state.supervisor.scheduler = MagicMock()
        app.state.supervisor.scheduler.get_jobs.return_value = []
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/scheduler")

        assert resp.status_code == 200
        data = resp.json()
        assert data["running"] is True
        assert len(data["anima_jobs"]) == 2

        # Verify job fields
        job_names = [j["name"] for j in data["anima_jobs"]]
        assert "Morning Planning (Daily 9:00 JST)" in job_names
        assert "Weekly Review (Friday 17:00 JST)" in job_names

        for job in data["anima_jobs"]:
            assert "id" in job
            assert "name" in job
            assert "anima" in job
            assert "type" in job
            assert "schedule" in job
            assert job["anima"] == "sakura"
            assert job["type"] == "llm"
            assert job["id"].startswith("cron-sakura-")

    async def test_scheduler_extracts_schedule_from_parentheses(
        self, tmp_path: Path,
    ) -> None:
        """Schedule info should be extracted from parentheses in the title."""
        animas_dir = tmp_path / "animas"
        _write_cron_md(animas_dir, "sakura", CRON_SAKURA)

        app = _create_app(tmp_path, anima_names=["sakura"])
        app.state.supervisor.is_scheduler_running.return_value = True
        app.state.supervisor.scheduler = MagicMock()
        app.state.supervisor.scheduler.get_jobs.return_value = []
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/scheduler")

        data = resp.json()
        schedules = {j["name"]: j["schedule"] for j in data["anima_jobs"]}
        assert schedules["Morning Planning (Daily 9:00 JST)"] == "Daily 9:00 JST"
        assert schedules["Weekly Review (Friday 17:00 JST)"] == "Friday 17:00 JST"

    async def test_scheduler_skips_commented_sections(
        self, tmp_path: Path,
    ) -> None:
        """Jobs inside HTML comment blocks should not appear."""
        animas_dir = tmp_path / "animas"
        _write_cron_md(animas_dir, "sakura", CRON_SAKURA)

        app = _create_app(tmp_path, anima_names=["sakura"])
        app.state.supervisor.is_scheduler_running.return_value = True
        app.state.supervisor.scheduler = MagicMock()
        app.state.supervisor.scheduler.get_jobs.return_value = []
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/scheduler")

        data = resp.json()
        job_names = [j["name"] for j in data["anima_jobs"]]
        # "Commented Out" should NOT appear — it is inside <!-- -->
        assert all("Commented Out" not in name for name in job_names)

    async def test_scheduler_multiple_animas(
        self, tmp_path: Path,
    ) -> None:
        """Jobs from multiple animas should all be returned."""
        animas_dir = tmp_path / "animas"
        _write_cron_md(animas_dir, "sakura", CRON_SAKURA)
        _write_cron_md(animas_dir, "taro", CRON_TARO)

        app = _create_app(tmp_path, anima_names=["sakura", "taro"])
        app.state.supervisor.is_scheduler_running.return_value = True
        app.state.supervisor.scheduler = MagicMock()
        app.state.supervisor.scheduler.get_jobs.return_value = []
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/scheduler")

        data = resp.json()
        assert data["running"] is True
        # sakura has 2 jobs, taro has 1 = 3 total
        assert len(data["anima_jobs"]) == 3

        animas_in_jobs = {j["anima"] for j in data["anima_jobs"]}
        assert "sakura" in animas_in_jobs
        assert "taro" in animas_in_jobs

        # Taro's job should have type "command"
        taro_jobs = [j for j in data["anima_jobs"] if j["anima"] == "taro"]
        assert len(taro_jobs) == 1
        assert taro_jobs[0]["type"] == "command"
        assert taro_jobs[0]["schedule"] == "Monday 10:00 JST"


# ── Test 2: Scheduler status WITHOUT cron.md ─────────────


class TestSchedulerWithoutCronMd:
    """Verify scheduler reports correctly when no cron.md exists."""

    async def test_system_status_scheduler_running_false_no_cron(
        self, tmp_path: Path,
    ) -> None:
        """scheduler_running should be False when no cron.md files exist."""
        animas_dir = tmp_path / "animas"
        # Create an anima without cron.md
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "identity.md").write_text("# Alice", encoding="utf-8")

        app = _create_app(tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["scheduler_running"] is False

    async def test_system_scheduler_empty_when_no_cron(
        self, tmp_path: Path,
    ) -> None:
        """Scheduler endpoint should return empty jobs when no cron.md."""
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "identity.md").write_text("# Alice", encoding="utf-8")

        app = _create_app(tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/scheduler")

        assert resp.status_code == 200
        data = resp.json()
        assert data["running"] is False
        assert data["anima_jobs"] == []

    async def test_system_scheduler_empty_when_no_animas(
        self, tmp_path: Path,
    ) -> None:
        """Scheduler endpoint with zero animas should return empty."""
        app = _create_app(tmp_path, anima_names=[])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/scheduler")

        assert resp.status_code == 200
        data = resp.json()
        assert data["running"] is False
        assert data["anima_jobs"] == []

    async def test_system_status_no_animas_scheduler_false(
        self, tmp_path: Path,
    ) -> None:
        """system_status with zero animas should show scheduler_running=False."""
        app = _create_app(tmp_path, anima_names=[])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["animas"] == 0
        assert data["scheduler_running"] is False


# ── Test 3: Init-status with checks array ────────────────


class TestInitStatusChecksArray:
    """Verify init_status returns a checks array with backward-compatible fields."""

    async def test_checks_array_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """init-status response should contain a checks array."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/init-status")

        assert resp.status_code == 200
        data = resp.json()
        assert "checks" in data
        assert isinstance(data["checks"], list)
        assert len(data["checks"]) > 0

    async def test_checks_items_have_label_and_ok(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Each check item should have at least 'label' and 'ok' fields."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/init-status")

        data = resp.json()
        for check in data["checks"]:
            assert "label" in check, f"Missing 'label' in check: {check}"
            assert "ok" in check, f"Missing 'ok' in check: {check}"
            assert isinstance(check["ok"], bool)

    async def test_checks_reflect_actual_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Check values should reflect actual filesystem and env state."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        # Set up config and one anima
        base_dir = tmp_path / ".animaworks"
        base_dir.mkdir(parents=True, exist_ok=True)
        (base_dir / "config.json").write_text("{}", encoding="utf-8")

        animas_dir = base_dir / "animas"
        animas_dir.mkdir()
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir()
        (alice_dir / "identity.md").write_text("# Alice", encoding="utf-8")

        shared_dir = base_dir / "shared"
        shared_dir.mkdir()

        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/init-status")

        data = resp.json()
        checks_by_label = {c["label"]: c for c in data["checks"]}

        # Config file exists
        config_check = checks_by_label.get("設定ファイル")
        assert config_check is not None
        assert config_check["ok"] is True

        # Anima registered
        anima_check = checks_by_label.get("Anima登録")
        assert anima_check is not None
        assert anima_check["ok"] is True
        assert "detail" in anima_check  # Should include count detail

        # Shared dir exists
        shared_check = checks_by_label.get("共有ディレクトリ")
        assert shared_check is not None
        assert shared_check["ok"] is True

        # API key checks
        anthropic_check = checks_by_label.get("Anthropic APIキー")
        assert anthropic_check is not None
        assert anthropic_check["ok"] is True

        openai_check = checks_by_label.get("OpenAI APIキー")
        assert openai_check is not None
        assert openai_check["ok"] is False

    async def test_backward_compatible_fields_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Existing fields (config_exists, animas_count, etc.) should still be present."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/init-status")

        data = resp.json()
        # All backward-compatible fields must still exist
        assert "config_exists" in data
        assert "animas_count" in data
        assert "api_keys" in data
        assert "shared_dir_exists" in data
        assert "initialized" in data

        # api_keys should be a dict with provider keys
        assert isinstance(data["api_keys"], dict)
        assert "anthropic" in data["api_keys"]
        assert "openai" in data["api_keys"]
        assert "google" in data["api_keys"]

    async def test_initialized_true_with_config_and_animas(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """initialized should be True when config and at least one anima exist."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        base_dir = tmp_path / ".animaworks"
        base_dir.mkdir(parents=True)
        (base_dir / "config.json").write_text("{}", encoding="utf-8")

        animas_dir = base_dir / "animas"
        animas_dir.mkdir()
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir()
        (alice_dir / "identity.md").write_text("# Alice", encoding="utf-8")

        app = _create_app(tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/init-status")

        data = resp.json()
        assert data["initialized"] is True
        assert data["config_exists"] is True
        assert data["animas_count"] == 1

        # checks array should also have 初期化完了=True
        checks_by_label = {c["label"]: c for c in data["checks"]}
        init_check = checks_by_label.get("初期化完了")
        assert init_check is not None
        assert init_check["ok"] is True


# ── Test 4: Connections endpoint ─────────────────────────


class TestConnectionsEndpoint:
    """Verify /api/system/connections returns websocket and process info."""

    async def test_connections_with_active_clients(
        self, tmp_path: Path,
    ) -> None:
        """Connections should report correct websocket client count."""
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "identity.md").write_text("# Alice", encoding="utf-8")

        app = _create_app(tmp_path, anima_names=["alice"], ws_connections=3)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/connections")

        assert resp.status_code == 200
        data = resp.json()
        assert "websocket" in data
        assert data["websocket"]["connected_clients"] == 3
        assert "processes" in data
        assert "alice" in data["processes"]

    async def test_connections_zero_clients(
        self, tmp_path: Path,
    ) -> None:
        """With no websocket connections, connected_clients should be 0."""
        app = _create_app(tmp_path, anima_names=["bob"], ws_connections=0)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/connections")

        assert resp.status_code == 200
        data = resp.json()
        assert data["websocket"]["connected_clients"] == 0
        assert "bob" in data["processes"]

    async def test_connections_no_animas(
        self, tmp_path: Path,
    ) -> None:
        """With no animas, processes should be empty."""
        app = _create_app(tmp_path, anima_names=[])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/connections")

        assert resp.status_code == 200
        data = resp.json()
        assert data["websocket"]["connected_clients"] == 0
        assert data["processes"] == {}

    async def test_connections_without_active_connections_attr(
        self, tmp_path: Path,
    ) -> None:
        """When ws_manager lacks active_connections, connected_clients should be 0."""
        app = _create_app(tmp_path, anima_names=["alice"])
        # Remove active_connections attribute to test hasattr fallback
        del app.state.ws_manager.active_connections

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/connections")

        assert resp.status_code == 200
        data = resp.json()
        assert data["websocket"]["connected_clients"] == 0
