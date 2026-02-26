# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for business theme UI — tokens, theme toggle, Lucide icons.

Validates the full theme pipeline: config API, static file serving,
CSS variables, and HTML structure for dashboard and workspace.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient


# ── Test App Factory ─────────────────────────────────────────────


def _create_test_app(tmp_path: Path, include_static: bool = True):
    """Create a test FastAPI app with create_router and optional static files.

    Mirrors the pattern from test_spa_routes: minimal app with create_router.
    When include_static=True, mounts StaticFiles at / for theme asset tests.
    """
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles

    from server.routes import create_router

    app = FastAPI()

    # Set up required state (same as test_spa_routes)
    animas_dir = tmp_path / "animas"
    animas_dir.mkdir(exist_ok=True)
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(exist_ok=True)

    supervisor = MagicMock()
    supervisor.get_all_status.return_value = {}
    supervisor.get_process_status.return_value = {"status": "idle", "pid": None}
    supervisor.processes = {}
    supervisor.is_scheduler_running.return_value = False
    supervisor.scheduler = None

    ws_manager = MagicMock()
    ws_manager.active_connections = []
    ws_manager.broadcast = AsyncMock()

    app.state.supervisor = supervisor
    app.state.anima_names = []
    app.state.ws_manager = ws_manager
    app.state.animas_dir = animas_dir
    app.state.shared_dir = shared_dir

    app.include_router(create_router())

    if include_static:
        project_root = Path(__file__).resolve().parents[2]
        static_dir = project_root / "server" / "static"
        if static_dir.exists():
            # Mount workspace first (nested path before root)
            workspace_dir = static_dir / "workspace"
            if workspace_dir.exists():
                app.mount(
                    "/workspace",
                    StaticFiles(directory=str(workspace_dir), html=True),
                    name="workspace",
                )
            app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


def _client(app):
    """Return an httpx AsyncClient wired to the given ASGI app."""
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


# ── Static File Serving ────────────────────────────────────────


class TestStaticFileServing:
    """Verify tokens.css and lucide.min.js are served correctly."""

    async def test_tokens_css_served(self, tmp_path: Path) -> None:
        """GET /styles/tokens.css returns 200 and contains --aw-color-accent."""
        app = _create_test_app(tmp_path)
        async with _client(app) as c:
            resp = await c.get("/styles/tokens.css")
        assert resp.status_code == 200
        assert "--aw-color-accent" in resp.text

    async def test_lucide_js_served(self, tmp_path: Path) -> None:
        """GET /vendor/lucide/lucide.min.js returns 200."""
        app = _create_test_app(tmp_path)
        async with _client(app) as c:
            resp = await c.get("/vendor/lucide/lucide.min.js")
        assert resp.status_code == 200
        # Lucide library exports lucide object
        assert "lucide" in resp.text or "createIcons" in resp.text


class TestOrgDashboardJS:
    """Verify workspace org-dashboard.js is served."""

    async def test_org_dashboard_js_served(self, tmp_path: Path) -> None:
        """GET /workspace/modules/org-dashboard.js returns 200."""
        app = _create_test_app(tmp_path)
        async with _client(app) as c:
            resp = await c.get("/workspace/modules/org-dashboard.js")
        assert resp.status_code == 200


# ── CSS Variable Completeness ──────────────────────────────────


class TestCSSVariableCompleteness:
    """Verify tokens.css contains all required variables."""

    async def test_css_variable_completeness(self, tmp_path: Path) -> None:
        """tokens.css contains all required --aw-color-* variables."""
        app = _create_test_app(tmp_path)
        async with _client(app) as c:
            resp = await c.get("/styles/tokens.css")
        assert resp.status_code == 200
        css = resp.text

        required_vars = [
            "--aw-color-accent",
            "--aw-color-bg-primary",
            "--aw-color-text-primary",
            "--aw-color-sidebar-bg",
            "--aw-emoji-display",
            "--aw-icon-display",
        ]
        for var in required_vars:
            assert var in css, f"tokens.css missing required variable: {var}"


class TestCSSFilesReferenceTokens:
    """Verify multiple CSS files use var(--aw-color-) references."""

    async def test_css_files_reference_tokens(self, tmp_path: Path) -> None:
        """Multiple CSS files contain var(--aw-color-) references."""
        app = _create_test_app(tmp_path)
        css_files = ["/styles/base.css", "/styles/sidebar-nav.css", "/styles/chat.css"]
        for path in css_files:
            async with _client(app) as c:
                resp = await c.get(path)
            assert resp.status_code == 200, f"Failed to fetch {path}"
            assert "var(--aw-color-" in resp.text, f"{path} should reference design tokens"


# ── HTML Structure ─────────────────────────────────────────────


class TestIndexHTMLThemeStructure:
    """Verify index.html has theme toggle and dual nav icons."""

    async def test_index_html_theme_structure(self, tmp_path: Path) -> None:
        """GET / contains theme toggle, nav-emoji, nav-lucide."""
        app = _create_test_app(tmp_path)
        async with _client(app) as c:
            resp = await c.get("/")
        assert resp.status_code == 200
        html = resp.text
        assert "themeToggle" in html or "theme-toggle" in html
        assert "nav-emoji" in html
        assert "nav-lucide" in html


class TestWorkspaceHTMLViewToggle:
    """Verify workspace index.html has view toggle and org panel."""

    async def test_workspace_html_view_toggle(self, tmp_path: Path) -> None:
        """GET /workspace/ contains wsViewToggle, wsOrgPanel."""
        app = _create_test_app(tmp_path)
        async with _client(app) as c:
            resp = await c.get("/workspace/")
        assert resp.status_code == 200
        html = resp.text
        assert "wsViewToggle" in html
        assert "wsOrgPanel" in html


# ── Config API ─────────────────────────────────────────────────


class TestConfigAPIIncludesUITheme:
    """Verify system config endpoint includes ui.theme."""

    async def test_config_api_includes_ui_theme(self, tmp_path: Path, monkeypatch) -> None:
        """GET /api/system/config includes ui.theme when config exists."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        config_dir = tmp_path / ".animaworks"
        config_dir.mkdir()
        config = {
            "setup_complete": True,
            "ui": {"theme": "business"},
        }
        (config_dir / "config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )

        app = _create_test_app(tmp_path)
        async with _client(app) as c:
            resp = await c.get("/api/system/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "ui" in data
        assert "theme" in data["ui"]
        assert data["ui"]["theme"] in ("default", "business")
