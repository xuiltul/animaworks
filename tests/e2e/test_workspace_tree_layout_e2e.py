# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for Workspace 3D office tree layout fix.

Validates that the GET /api/animas endpoint returns the ``supervisor``
field required by the office3d.js tree layout algorithm, and that the
data structure correctly represents organizational hierarchy.

Tests go through the full FastAPI application stack with mocked externals.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


# ── Helpers ──────────────────────────────────────────────


def _create_app_with_config(
    tmp_path: Path,
    anima_names: list[str],
    config_data: dict,
):
    """Build a real FastAPI app with a concrete config.json on disk.

    This exercises the actual config loading path (load_config reads
    from the file) rather than mocking it, for true E2E coverage.
    """
    animas_dir = tmp_path / "animas"
    animas_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

    # Write config.json to disk so load_config() can read it
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(config_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (
        patch("server.app.ProcessSupervisor") as mock_sup_cls,
        patch("server.app.load_config") as mock_app_cfg,
        patch("server.app.WebSocketManager") as mock_ws_cls,
        patch("server.app.load_auth") as mock_auth,
    ):
        cfg = MagicMock()
        cfg.setup_complete = True
        mock_app_cfg.return_value = cfg

        auth_cfg = MagicMock()
        auth_cfg.auth_mode = "local_trust"
        mock_auth.return_value = auth_cfg

        supervisor = MagicMock()
        supervisor.get_all_status.return_value = {}
        supervisor.get_process_status.return_value = {
            "status": "idle",
            "pid": 9999,
            "bootstrapping": False,
            "uptime_sec": 60,
        }
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

    app.state.anima_names = anima_names

    # Patch load_config in the animas route module to use our config file
    from core.config.models import AnimaWorksConfig, load_config, invalidate_cache

    invalidate_cache()
    real_config = load_config(config_path)

    import server.routes.animas as animas_module
    animas_module.load_config = lambda *a, **kw: real_config

    return app


# ── E2E Test: Full tree layout data flow ──────────────────


class TestWorkspaceTreeLayoutE2E:
    """E2E: Verify /api/animas provides correct data for 3D office tree layout."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        """Invalidate config cache and restore patched imports after each test."""
        yield
        from core.config.models import invalidate_cache
        invalidate_cache()
        # Restore original load_config
        import server.routes.animas as animas_module
        from core.config.models import load_config
        animas_module.load_config = load_config

    async def test_sakura_based_tree_structure(
        self, tmp_path: Path,
    ) -> None:
        """Simulate the real AnimaWorks hierarchy with sakura as root."""
        config_data = {
            "version": 1,
            "setup_complete": True,
            "animas": {
                "sakura": {},
                "kotoha": {"supervisor": "sakura"},
                "chatwork_checker": {},
            },
        }

        app = _create_app_with_config(
            tmp_path,
            anima_names=["sakura", "kotoha", "chatwork_checker"],
            config_data=config_data,
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3

        by_name = {p["name"]: p for p in data}

        # sakura is a root (no supervisor)
        assert by_name["sakura"]["supervisor"] is None

        # kotoha is under sakura
        assert by_name["kotoha"]["supervisor"] == "sakura"

        # chatwork_checker is an independent root
        assert by_name["chatwork_checker"]["supervisor"] is None

    async def test_tree_data_enables_correct_layout_simulation(
        self, tmp_path: Path,
    ) -> None:
        """Verify the API data can correctly build a tree (simulating buildOrgTree logic)."""
        config_data = {
            "version": 1,
            "setup_complete": True,
            "animas": {
                "sakura": {},
                "kotoha": {"supervisor": "sakura"},
                "aoi": {"supervisor": "sakura"},
            },
        }

        app = _create_app_with_config(
            tmp_path,
            anima_names=["sakura", "kotoha", "aoi"],
            config_data=config_data,
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas")

        animas = resp.json()

        # Simulate buildOrgTree from office3d.js
        node_map = {}
        for p in animas:
            node_map[p["name"]] = {
                "name": p["name"],
                "supervisor": p.get("supervisor"),
                "children": [],
            }

        roots = []
        for node in node_map.values():
            if not node["supervisor"] or node["supervisor"] not in node_map:
                roots.append(node)
            else:
                parent = node_map[node["supervisor"]]
                parent["children"].append(node)

        # Should have exactly 1 root (sakura)
        assert len(roots) == 1
        assert roots[0]["name"] == "sakura"

        # sakura should have 2 children
        children_names = sorted(c["name"] for c in roots[0]["children"])
        assert children_names == ["aoi", "kotoha"]

    async def test_anima_not_in_config_defaults_to_no_supervisor(
        self, tmp_path: Path,
    ) -> None:
        """Animas present in anima_names but absent from config.json
        should have supervisor=null (treated as root in tree layout)."""
        config_data = {
            "version": 1,
            "setup_complete": True,
            "animas": {
                "sakura": {},
            },
        }

        app = _create_app_with_config(
            tmp_path,
            anima_names=["sakura", "new_anima"],
            config_data=config_data,
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas")

        data = resp.json()
        by_name = {p["name"]: p for p in data}

        assert by_name["new_anima"]["supervisor"] is None

    async def test_connector_data_available(
        self, tmp_path: Path,
    ) -> None:
        """Verify that anima data includes enough info for buildConnectors
        (which iterates animas and checks p.supervisor to draw lines)."""
        config_data = {
            "version": 1,
            "setup_complete": True,
            "animas": {
                "sakura": {},
                "kotoha": {"supervisor": "sakura"},
            },
        }

        app = _create_app_with_config(
            tmp_path,
            anima_names=["sakura", "kotoha"],
            config_data=config_data,
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas")

        animas = resp.json()

        # buildConnectors logic: for each anima with supervisor, draw a line
        connectors = []
        anima_names = {p["name"] for p in animas}
        for p in animas:
            if p.get("supervisor") and p["supervisor"] in anima_names:
                connectors.append((p["supervisor"], p["name"]))

        assert len(connectors) == 1
        assert connectors[0] == ("sakura", "kotoha")

    async def test_empty_animas_list(
        self, tmp_path: Path,
    ) -> None:
        """API should return empty list when no animas are registered."""
        config_data = {"version": 1, "setup_complete": True, "animas": {}}

        app = _create_app_with_config(
            tmp_path,
            anima_names=[],
            config_data=config_data,
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas")

        assert resp.status_code == 200
        assert resp.json() == []
