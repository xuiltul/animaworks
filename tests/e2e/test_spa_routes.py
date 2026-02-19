"""E2E tests for SPA migration — Route registration and API responses."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


# ── Test App Factory ─────────────────────────────────────────────


def _create_test_app(tmp_path: Path):
    """Create a minimal app with all routes registered via create_router().

    This exercises the exact same registration path that create_app() uses,
    but without lifespan/ProcessSupervisor/static-mount side effects.
    """
    from fastapi import FastAPI

    from server.routes import create_router

    app = FastAPI()

    # Set up required state
    animas_dir = tmp_path / "animas"
    animas_dir.mkdir(exist_ok=True)
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(exist_ok=True)

    # Mock supervisor
    supervisor = MagicMock()
    supervisor.get_all_status.return_value = {}
    supervisor.get_process_status.return_value = {"status": "idle", "pid": None}
    supervisor.processes = {}
    supervisor.is_scheduler_running.return_value = False
    supervisor.scheduler = None

    # Mock ws_manager
    ws_manager = MagicMock()
    ws_manager.active_connections = []

    app.state.supervisor = supervisor
    app.state.anima_names = []
    app.state.ws_manager = ws_manager
    app.state.animas_dir = animas_dir
    app.state.shared_dir = shared_dir

    app.include_router(create_router())
    return app


def _client(app):
    """Return an httpx AsyncClient wired to the given ASGI app."""
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


# ── 1. All New Routes Are Registered ────────────────────────────


class TestAllNewRoutesRegistered:
    """Verify new SPA endpoints are reachable (not 404/405)."""

    async def test_get_system_config_route_exists(self, tmp_path, monkeypatch):
        """GET /api/system/config — 404 is OK (no config file), but NOT 405."""
        # Point Path.home() at tmp_path so config_routes looks there
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        app = _create_test_app(tmp_path)
        async with _client(app) as c:
            resp = await c.get("/api/system/config")
        # 404 = route exists but file missing; 200 = file exists
        assert resp.status_code in (200, 404)

    async def test_get_init_status_returns_200(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        app = _create_test_app(tmp_path)
        async with _client(app) as c:
            resp = await c.get("/api/system/init-status")
        assert resp.status_code == 200

    async def test_get_connections_returns_200(self, tmp_path):
        app = _create_test_app(tmp_path)
        async with _client(app) as c:
            resp = await c.get("/api/system/connections")
        assert resp.status_code == 200

    async def test_get_scheduler_returns_200(self, tmp_path):
        app = _create_test_app(tmp_path)
        async with _client(app) as c:
            resp = await c.get("/api/system/scheduler")
        assert resp.status_code == 200

    async def test_get_logs_returns_200(self, tmp_path, monkeypatch):
        import server.routes.logs_routes as logs_mod

        monkeypatch.setattr(logs_mod, "_LOG_SEARCH_DIRS", [tmp_path / "logs"])
        app = _create_test_app(tmp_path)
        async with _client(app) as c:
            resp = await c.get("/api/system/logs")
        assert resp.status_code == 200


# ── 2. Init-Status Flow ─────────────────────────────────────────


class TestInitStatus:
    """Test /api/system/init-status with varying filesystem state."""

    async def test_empty_directory(self, tmp_path, monkeypatch):
        """No config.json, no animas => initialized=false."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        app = _create_test_app(tmp_path)
        async with _client(app) as c:
            resp = await c.get("/api/system/init-status")
        data = resp.json()
        assert data["initialized"] is False
        assert data["config_exists"] is False
        assert data["animas_count"] == 0

    async def test_with_config_and_animas(self, tmp_path, monkeypatch):
        """config.json + 1 anima => initialized=true."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        aw_dir = tmp_path / ".animaworks"
        aw_dir.mkdir()
        (aw_dir / "config.json").write_text("{}", encoding="utf-8")

        animas_dir = aw_dir / "animas"
        animas_dir.mkdir()
        alice = animas_dir / "alice"
        alice.mkdir()
        (alice / "identity.md").write_text("# Alice", encoding="utf-8")

        app = _create_test_app(tmp_path)
        async with _client(app) as c:
            resp = await c.get("/api/system/init-status")
        data = resp.json()
        assert data["initialized"] is True
        assert data["config_exists"] is True
        assert data["animas_count"] == 1

    async def test_api_key_detection(self, tmp_path, monkeypatch):
        """API key presence should be reflected."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        app = _create_test_app(tmp_path)
        async with _client(app) as c:
            resp = await c.get("/api/system/init-status")
        data = resp.json()
        assert data["api_keys"]["anthropic"] is True
        assert data["api_keys"]["openai"] is False
        assert data["api_keys"]["google"] is False


# ── 3. System Connections ────────────────────────────────────────


class TestSystemConnections:
    """Test /api/system/connections response shape."""

    async def test_returns_websocket_section(self, tmp_path):
        app = _create_test_app(tmp_path)
        async with _client(app) as c:
            resp = await c.get("/api/system/connections")
        data = resp.json()
        assert "websocket" in data
        assert "connected_clients" in data["websocket"]
        assert data["websocket"]["connected_clients"] == 0

    async def test_returns_processes_section(self, tmp_path):
        app = _create_test_app(tmp_path)
        app.state.anima_names = ["alice"]
        app.state.supervisor.get_process_status.return_value = {
            "status": "running",
            "pid": 12345,
        }
        async with _client(app) as c:
            resp = await c.get("/api/system/connections")
        data = resp.json()
        assert "processes" in data
        assert "alice" in data["processes"]
        assert data["processes"]["alice"]["status"] == "running"
        assert data["processes"]["alice"]["pid"] == 12345


# ── 4. Scheduler ─────────────────────────────────────────────────


class TestScheduler:
    """Test /api/system/scheduler response."""

    async def test_no_scheduler(self, tmp_path):
        """When no cron.md files exist, running=False."""
        app = _create_test_app(tmp_path)
        async with _client(app) as c:
            resp = await c.get("/api/system/scheduler")
        data = resp.json()
        assert data["running"] is False
        assert data["anima_jobs"] == []

    async def test_with_scheduler_and_jobs(self, tmp_path):
        """When cron.md files exist, report parsed jobs."""
        app = _create_test_app(tmp_path)

        # Create an anima with cron.md
        animas_dir = app.state.animas_dir
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "cron.md").write_text(
            "# Cron: alice\n\n"
            "## Morning Planning (Daily 9:00 JST)\n"
            "type: llm\n"
            "Plan daily tasks.\n",
            encoding="utf-8",
        )
        app.state.anima_names = ["alice"]
        app.state.supervisor.is_scheduler_running.return_value = True
        app.state.supervisor.scheduler = MagicMock()
        app.state.supervisor.scheduler.get_jobs.return_value = []

        async with _client(app) as c:
            resp = await c.get("/api/system/scheduler")
        data = resp.json()
        assert data["running"] is True
        assert len(data["anima_jobs"]) == 1
        assert data["anima_jobs"][0]["anima"] == "alice"
        assert "Morning Planning" in data["anima_jobs"][0]["name"]


# ── 5. Logs Integration ─────────────────────────────────────────


class TestLogsIntegration:
    """Test /api/system/logs, /api/system/logs/{filename}."""

    async def test_list_empty_logs(self, tmp_path, monkeypatch):
        """No log dir => empty file list."""
        import server.routes.logs_routes as logs_mod

        monkeypatch.setattr(logs_mod, "_LOG_SEARCH_DIRS", [tmp_path / "logs"])
        app = _create_test_app(tmp_path)
        async with _client(app) as c:
            resp = await c.get("/api/system/logs")
        data = resp.json()
        assert data["files"] == []

    async def test_list_with_log_files(self, tmp_path, monkeypatch):
        """Log files are discovered and listed."""
        import server.routes.logs_routes as logs_mod

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        (logs_dir / "animaworks.log").write_text(
            "2026-02-15 INFO Started\n2026-02-15 DEBUG tick\n", encoding="utf-8"
        )
        (logs_dir / "error.log").write_text("ERR something\n", encoding="utf-8")

        monkeypatch.setattr(logs_mod, "_LOG_SEARCH_DIRS", [logs_dir])
        app = _create_test_app(tmp_path)
        async with _client(app) as c:
            resp = await c.get("/api/system/logs")
        data = resp.json()
        names = [f["name"] for f in data["files"]]
        assert "animaworks.log" in names
        assert "error.log" in names

    async def test_read_log_with_pagination(self, tmp_path, monkeypatch):
        """Read a specific log file with offset/limit."""
        import server.routes.logs_routes as logs_mod

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        lines = [f"line {i}" for i in range(50)]
        (logs_dir / "test.log").write_text("\n".join(lines), encoding="utf-8")

        monkeypatch.setattr(logs_mod, "_LOG_SEARCH_DIRS", [logs_dir])
        app = _create_test_app(tmp_path)
        async with _client(app) as c:
            resp = await c.get("/api/system/logs/test.log?offset=10&limit=5")
        data = resp.json()
        assert data["filename"] == "test.log"
        assert data["total_lines"] == 50
        assert data["offset"] == 10
        assert data["limit"] == 5
        assert len(data["lines"]) == 5
        assert data["lines"][0] == "line 10"

    async def test_read_nonexistent_log_returns_404(self, tmp_path, monkeypatch):
        import server.routes.logs_routes as logs_mod

        monkeypatch.setattr(logs_mod, "_LOG_SEARCH_DIRS", [tmp_path / "logs"])
        app = _create_test_app(tmp_path)
        async with _client(app) as c:
            resp = await c.get("/api/system/logs/nonexistent.log")
        assert resp.status_code == 404

    async def test_path_traversal_rejected(self, tmp_path, monkeypatch):
        """Filenames with '..' should be rejected with 400."""
        import server.routes.logs_routes as logs_mod

        monkeypatch.setattr(logs_mod, "_LOG_SEARCH_DIRS", [tmp_path / "logs"])
        app = _create_test_app(tmp_path)
        async with _client(app) as c:
            # Filename containing '..' is caught by _validate_filename
            resp = await c.get("/api/system/logs/..secret.log")
        assert resp.status_code == 400

    async def test_dotfile_rejected(self, tmp_path, monkeypatch):
        """Filenames starting with '.' should be rejected with 400."""
        import server.routes.logs_routes as logs_mod

        monkeypatch.setattr(logs_mod, "_LOG_SEARCH_DIRS", [tmp_path / "logs"])
        app = _create_test_app(tmp_path)
        async with _client(app) as c:
            resp = await c.get("/api/system/logs/.hidden")
        assert resp.status_code == 400


# ── 6. Memory Stats ─────────────────────────────────────────────


class TestMemoryStats:
    """Test /api/animas/{name}/memory/stats."""

    async def test_stats_for_anima_with_files(self, tmp_path):
        app = _create_test_app(tmp_path)
        animas_dir = app.state.animas_dir

        # Create an anima with memory files
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir()
        (alice_dir / "identity.md").write_text("# Alice", encoding="utf-8")

        episodes = alice_dir / "episodes"
        episodes.mkdir()
        (episodes / "2026-02-14.md").write_text(
            "Worked on project", encoding="utf-8"
        )
        (episodes / "2026-02-15.md").write_text(
            "Fixed bugs", encoding="utf-8"
        )

        knowledge = alice_dir / "knowledge"
        knowledge.mkdir()
        (knowledge / "python.md").write_text(
            "Python tips", encoding="utf-8"
        )

        procedures = alice_dir / "procedures"
        procedures.mkdir()

        async with _client(app) as c:
            resp = await c.get("/api/animas/alice/memory/stats")
        data = resp.json()
        assert data["anima"] == "alice"
        assert data["episodes"]["count"] == 2
        assert data["episodes"]["total_bytes"] > 0
        assert data["knowledge"]["count"] == 1
        assert data["procedures"]["count"] == 0

    async def test_stats_for_nonexistent_anima_returns_404(self, tmp_path):
        app = _create_test_app(tmp_path)
        async with _client(app) as c:
            resp = await c.get("/api/animas/nobody/memory/stats")
        assert resp.status_code == 404


# ── 7. Anima Config ─────────────────────────────────────────────


class TestAnimaConfig:
    """Test /api/animas/{name}/config."""

    async def test_config_for_existing_anima(self, tmp_path):
        app = _create_test_app(tmp_path)
        animas_dir = app.state.animas_dir

        alice_dir = animas_dir / "alice"
        alice_dir.mkdir()
        (alice_dir / "identity.md").write_text("# Alice", encoding="utf-8")

        mock_defaults = MagicMock()
        mock_defaults.model = "anthropic/claude-sonnet-4-20250514"
        mock_defaults.execution_mode = "a1"
        mock_defaults.model_dump.return_value = {
            "model": "anthropic/claude-sonnet-4-20250514",
            "execution_mode": "a1",
        }
        mock_credential = MagicMock()

        with patch(
            "core.config.models.load_config"
        ) as mock_load, patch(
            "core.config.models.resolve_anima_config"
        ) as mock_resolve:
            mock_load.return_value = MagicMock()
            mock_resolve.return_value = (mock_defaults, mock_credential)
            async with _client(app) as c:
                resp = await c.get("/api/animas/alice/config")

        data = resp.json()
        assert data["anima"] == "alice"
        assert data["model"] == "anthropic/claude-sonnet-4-20250514"
        assert data["execution_mode"] == "a1"

    async def test_config_for_nonexistent_anima_returns_404(self, tmp_path):
        app = _create_test_app(tmp_path)
        async with _client(app) as c:
            resp = await c.get("/api/animas/nobody/config")
        assert resp.status_code == 404


# ── 8. Static File Serving ───────────────────────────────────────


class TestStaticFileServing:
    """Verify the SPA entry point and new CSS files are served."""

    async def test_index_html_served(self, tmp_path):
        """GET / should return index.html from static mount."""
        from fastapi import FastAPI
        from fastapi.staticfiles import StaticFiles

        from server.routes import create_router

        app = FastAPI()
        # Minimal state
        supervisor = MagicMock()
        supervisor.get_all_status.return_value = {}
        supervisor.get_process_status.return_value = {"status": "idle", "pid": None}
        supervisor.processes = {}
        ws_manager = MagicMock()
        ws_manager.active_connections = []

        app.state.supervisor = supervisor
        app.state.anima_names = []
        app.state.ws_manager = ws_manager
        app.state.animas_dir = tmp_path / "animas"
        app.state.animas_dir.mkdir()
        app.state.shared_dir = tmp_path / "shared"
        app.state.shared_dir.mkdir()


        app.include_router(create_router())

        static_dir = Path(__file__).resolve().parent.parent.parent / "server" / "static"
        if static_dir.exists():
            app.mount(
                "/", StaticFiles(directory=str(static_dir), html=True), name="static"
            )

        async with _client(app) as c:
            resp = await c.get("/")
        if not static_dir.exists():
            pytest.skip("server/static not present in this checkout")
        assert resp.status_code == 200
        assert "AnimaWorks" in resp.text

    async def test_index_html_contains_sidebar_nav(self, tmp_path):
        """The SPA shell includes sidebar navigation links."""
        from fastapi import FastAPI
        from fastapi.staticfiles import StaticFiles

        from server.routes import create_router

        app = FastAPI()
        supervisor = MagicMock()
        supervisor.get_all_status.return_value = {}
        supervisor.get_process_status.return_value = {"status": "idle", "pid": None}
        supervisor.processes = {}
        ws_manager = MagicMock()
        ws_manager.active_connections = []

        app.state.supervisor = supervisor
        app.state.anima_names = []
        app.state.ws_manager = ws_manager
        app.state.animas_dir = tmp_path / "animas"
        app.state.animas_dir.mkdir()
        app.state.shared_dir = tmp_path / "shared"
        app.state.shared_dir.mkdir()


        app.include_router(create_router())

        static_dir = Path(__file__).resolve().parent.parent.parent / "server" / "static"
        if static_dir.exists():
            app.mount(
                "/", StaticFiles(directory=str(static_dir), html=True), name="static"
            )

        async with _client(app) as c:
            resp = await c.get("/")
        if not static_dir.exists():
            pytest.skip("server/static not present in this checkout")
        body = resp.text
        assert "sidebar-nav" in body
        assert 'data-route="/chat"' in body
        assert 'data-route="/logs"' in body
        assert 'data-route="/memory"' in body

    async def test_new_css_files_exist(self, tmp_path):
        """All new SPA CSS files should be servable."""
        from fastapi import FastAPI
        from fastapi.staticfiles import StaticFiles

        from server.routes import create_router

        app = FastAPI()
        supervisor = MagicMock()
        supervisor.get_all_status.return_value = {}
        supervisor.get_process_status.return_value = {"status": "idle", "pid": None}
        supervisor.processes = {}
        ws_manager = MagicMock()
        ws_manager.active_connections = []

        app.state.supervisor = supervisor
        app.state.anima_names = []
        app.state.ws_manager = ws_manager
        app.state.animas_dir = tmp_path / "animas"
        app.state.animas_dir.mkdir()
        app.state.shared_dir = tmp_path / "shared"
        app.state.shared_dir.mkdir()


        app.include_router(create_router())

        static_dir = Path(__file__).resolve().parent.parent.parent / "server" / "static"
        if not static_dir.exists():
            pytest.skip("server/static not present in this checkout")
        app.mount(
            "/", StaticFiles(directory=str(static_dir), html=True), name="static"
        )

        expected_css = [
            "sidebar-nav.css",
            "layout.css",
            "dashboard.css",
            "activity.css",
            "memory.css",
            "history.css",
        ]
        async with _client(app) as c:
            for css_file in expected_css:
                resp = await c.get(f"/styles/{css_file}")
                assert resp.status_code == 200, f"{css_file} not served (got {resp.status_code})"


# ── 9. Cross-Route Integration ───────────────────────────────────


class TestCrossRouteIntegration:
    """Verify multiple route modules work together in a single app."""

    async def test_system_and_anima_routes_coexist(self, tmp_path, monkeypatch):
        """System routes and anima routes should both be reachable."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        app = _create_test_app(tmp_path)

        # Create an anima on the filesystem
        alice_dir = app.state.animas_dir / "alice"
        alice_dir.mkdir()
        (alice_dir / "identity.md").write_text("# Alice", encoding="utf-8")
        # Register alice in anima_names so the list endpoint includes her
        app.state.anima_names = ["alice"]

        async with _client(app) as c:
            # System endpoint
            resp_init = await c.get("/api/system/init-status")
            assert resp_init.status_code == 200

            # Anima list endpoint
            resp_animas = await c.get("/api/animas")
            assert resp_animas.status_code == 200
            names = [p["name"] for p in resp_animas.json()]
            assert "alice" in names

            # Connections
            resp_conn = await c.get("/api/system/connections")
            assert resp_conn.status_code == 200
            assert "alice" in resp_conn.json()["processes"]

    async def test_logs_and_config_routes_coexist(self, tmp_path, monkeypatch):
        """Config and logs routes should be independent and both reachable."""
        import server.routes.logs_routes as logs_mod

        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(logs_mod, "_LOG_SEARCH_DIRS", [tmp_path / "logs"])

        app = _create_test_app(tmp_path)
        async with _client(app) as c:
            resp_config = await c.get("/api/system/config")
            # 404 because no config file, but route exists
            assert resp_config.status_code == 404

            resp_logs = await c.get("/api/system/logs")
            assert resp_logs.status_code == 200

    async def test_memory_stats_and_episodes_for_same_anima(self, tmp_path):
        """Both /memory/stats and /episodes should work for the same anima."""
        app = _create_test_app(tmp_path)
        animas_dir = app.state.animas_dir

        bob_dir = animas_dir / "bob"
        bob_dir.mkdir()
        (bob_dir / "identity.md").write_text("# Bob", encoding="utf-8")

        episodes = bob_dir / "episodes"
        episodes.mkdir()
        (episodes / "2026-02-15.md").write_text("Daily log", encoding="utf-8")

        # Also create knowledge/procedures dirs so MemoryManager doesn't fail
        (bob_dir / "knowledge").mkdir()
        (bob_dir / "procedures").mkdir()

        async with _client(app) as c:
            resp_stats = await c.get("/api/animas/bob/memory/stats")
            assert resp_stats.status_code == 200
            assert resp_stats.json()["episodes"]["count"] == 1

            resp_eps = await c.get("/api/animas/bob/episodes")
            assert resp_eps.status_code == 200
            files = resp_eps.json()["files"]
            assert any("2026-02-15" in f for f in files)

    async def test_shared_users_route(self, tmp_path):
        """GET /api/shared/users returns user directories."""
        app = _create_test_app(tmp_path)
        users_dir = app.state.shared_dir / "users"
        users_dir.mkdir()
        (users_dir / "taro").mkdir()

        async with _client(app) as c:
            resp = await c.get("/api/shared/users")
        assert resp.status_code == 200
        assert "taro" in resp.json()
