# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for supervisor field in GET /api/animas response.

Validates that the list_animas endpoint includes the ``supervisor`` field
read from config.json, which is required by the Workspace 3D office
tree layout algorithm.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ──────────────────────────────────────────────


def _create_app(
    tmp_path: Path,
    anima_names: list[str] | None = None,
    config_animas: dict | None = None,
):
    """Build a real FastAPI app with mocked externals.

    Parameters
    ----------
    config_animas:
        Dict of anima name → AnimaModelConfig-like dict to inject
        into the config.  Keys not listed default to empty config.
    """
    animas_dir = tmp_path / "animas"
    animas_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

    # Mock auth to use local_trust mode (skip authentication)
    mock_auth = MagicMock()
    mock_auth.auth_mode = "local_trust"

    with (
        patch("server.app.ProcessSupervisor") as mock_sup_cls,
        patch("server.app.load_config") as mock_cfg,
        patch("server.app.WebSocketManager") as mock_ws_cls,
        patch("server.app.load_auth", return_value=mock_auth),
    ):
        cfg = MagicMock()
        cfg.setup_complete = True
        mock_cfg.return_value = cfg

        supervisor = MagicMock()
        supervisor.get_all_status.return_value = {}
        supervisor.get_process_status.return_value = {
            "status": "idle",
            "pid": 12345,
            "bootstrapping": False,
            "uptime_sec": 100,
        }
        supervisor.is_scheduler_running.return_value = False
        supervisor.scheduler = None
        mock_sup_cls.return_value = supervisor

        ws_manager = MagicMock()
        ws_manager.active_connections = []
        mock_ws_cls.return_value = ws_manager

        from server.app import create_app

        app = create_app(animas_dir, shared_dir)

    # Permanently patch load_auth so the auth_guard middleware
    # uses local_trust mode when handling requests.
    import server.app as _server_app_mod
    app._test_original_load_auth = _server_app_mod.load_auth
    _server_app_mod.load_auth = lambda *a, **kw: mock_auth

    if anima_names is not None:
        app.state.anima_names = anima_names

    # Patch load_config used inside the animas route
    if config_animas is not None:
        _patch_route_config(app, config_animas)

    return app


def _patch_route_config(app, config_animas: dict):
    """Patch ``core.config.models.load_config`` to return a config with
    the given anima overrides.

    This patches the module-level import used inside the route handler.
    """
    from core.config.models import AnimaWorksConfig, AnimaModelConfig

    animas = {}
    for name, overrides in config_animas.items():
        animas[name] = AnimaModelConfig(**overrides)

    config = AnimaWorksConfig(animas=animas)

    # Patch load_config in the module where it was imported by animas.py
    import server.routes.animas as animas_module
    app._test_original_load_config = animas_module.load_config
    animas_module.load_config = lambda *a, **kw: config


# ── Tests ──────────────────────────────────────────────


class TestListAnimasSupervisorField:
    """Verify that GET /api/animas includes the supervisor field."""

    @pytest.fixture(autouse=True)
    def _cleanup_patch(self):
        """Restore original load_config and load_auth after each test."""
        yield
        import server.routes.animas as animas_module
        if hasattr(animas_module, "_test_original_load_config"):
            animas_module.load_config = animas_module._test_original_load_config
        import server.app as _server_app_mod
        if hasattr(_server_app_mod, "_test_original_load_auth"):
            _server_app_mod.load_auth = _server_app_mod._test_original_load_auth

    async def test_supervisor_included_in_response(
        self, tmp_path: Path,
    ) -> None:
        """Each anima in the response should have a 'supervisor' key."""
        app = _create_app(
            tmp_path,
            anima_names=["sakura", "kotoha"],
            config_animas={
                "sakura": {},
                "kotoha": {"supervisor": "sakura"},
            },
        )

        from httpx import ASGITransport, AsyncClient
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

        for anima_item in data:
            assert "supervisor" in anima_item, (
                f"Anima '{anima_item['name']}' missing 'supervisor' field"
            )

    async def test_supervisor_value_from_config(
        self, tmp_path: Path,
    ) -> None:
        """supervisor field should reflect the config.json value."""
        app = _create_app(
            tmp_path,
            anima_names=["sakura", "kotoha"],
            config_animas={
                "sakura": {},
                "kotoha": {"supervisor": "sakura"},
            },
        )

        from httpx import ASGITransport, AsyncClient
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas")

        data = resp.json()
        by_name = {p["name"]: p for p in data}

        assert by_name["sakura"]["supervisor"] is None
        assert by_name["kotoha"]["supervisor"] == "sakura"

    async def test_supervisor_none_when_not_in_config(
        self, tmp_path: Path,
    ) -> None:
        """Animas not listed in config.json should have supervisor=null."""
        app = _create_app(
            tmp_path,
            anima_names=["unknown_anima"],
            config_animas={},  # empty config
        )

        from httpx import ASGITransport, AsyncClient
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas")

        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "unknown_anima"
        assert data[0]["supervisor"] is None

    async def test_multiple_hierarchy_levels(
        self, tmp_path: Path,
    ) -> None:
        """Three-level hierarchy should be correctly represented."""
        app = _create_app(
            tmp_path,
            anima_names=["boss", "manager", "worker"],
            config_animas={
                "boss": {},
                "manager": {"supervisor": "boss"},
                "worker": {"supervisor": "manager"},
            },
        )

        from httpx import ASGITransport, AsyncClient
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas")

        data = resp.json()
        by_name = {p["name"]: p for p in data}

        assert by_name["boss"]["supervisor"] is None
        assert by_name["manager"]["supervisor"] == "boss"
        assert by_name["worker"]["supervisor"] == "manager"

    async def test_existing_fields_preserved(
        self, tmp_path: Path,
    ) -> None:
        """Adding supervisor should not break existing response fields."""
        app = _create_app(
            tmp_path,
            anima_names=["sakura"],
            config_animas={"sakura": {}},
        )

        from httpx import ASGITransport, AsyncClient
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas")

        data = resp.json()
        anima_item = data[0]

        # All pre-existing fields must still be present
        expected_fields = {"name", "status", "bootstrapping", "pid", "uptime_sec", "appearance", "supervisor", "model"}
        assert set(anima_item.keys()) == expected_fields
