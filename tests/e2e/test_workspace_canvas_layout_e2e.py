"""E2E tests for Workspace canvas node graph layout.

Validates that the GET /api/animas endpoint returns data compatible
with the canvas-based org-dashboard layout, and that the tree layout
algorithm (simulated in Python) produces correct positions for the
draggable card graph.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


# ── Helpers ──────────────────────────────────────────────

CARD_W = 280
CARD_H = 80
GAP_X = 60
GAP_Y = 50


def _create_app_with_config(
    tmp_path: Path,
    anima_names: list[str],
    config_data: dict,
):
    """Build a real FastAPI app with a concrete config on disk."""
    animas_dir = tmp_path / "animas"
    animas_dir.mkdir(parents=True, exist_ok=True)
    for anima_name in anima_names:
        anima_dir = animas_dir / anima_name
        anima_dir.mkdir(parents=True, exist_ok=True)
        (anima_dir / 'identity.md').write_text(
            f'# {anima_name}\n\nTest identity for {anima_name}.\n',
            encoding='utf-8',
        )
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

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

    import server.app as _sa
    _auth = MagicMock()
    _auth.auth_mode = "local_trust"
    _sa.load_auth = lambda: _auth

    app.state.anima_names = anima_names

    from core.config.models import load_config, invalidate_cache

    invalidate_cache()
    real_config = load_config(config_path)

    import server.routes.animas as animas_module
    animas_module.load_config = lambda *a, **kw: real_config

    return app


def build_org_tree(animas):
    """Python port of org-dashboard.js buildOrgTree()."""
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
    return roots if roots else list(node_map.values()), node_map


def compute_tree_layout(roots, viewport_width):
    """Python port of org-dashboard.js _computeTreeLayout()."""
    positions = {}

    def measure(node):
        if not node["children"]:
            return {"w": CARD_W, "h": CARD_H, "node": node}
        child_measures = [measure(c) for c in node["children"]]
        total_child_w = sum(m["w"] for m in child_measures) + GAP_X * (len(child_measures) - 1)
        return {
            "w": max(CARD_W, total_child_w),
            "h": CARD_H + GAP_Y + max(m["h"] for m in child_measures),
            "node": node,
            "children": child_measures,
        }

    def layout(measured, x, y):
        cx = x + measured["w"] / 2 - CARD_W / 2
        positions[measured["node"]["name"]] = {"x": cx, "y": y}
        if "children" not in measured:
            return
        child_x = x
        for child in measured["children"]:
            layout(child, child_x, y + CARD_H + GAP_Y)
            child_x += child["w"] + GAP_X

    measured = [measure(r) for r in roots]
    total_w = sum(m["w"] for m in measured) + GAP_X * (len(measured) - 1)
    start_x = max(40, (viewport_width - total_w) / 2)
    for m in measured:
        layout(m, start_x, 40)
        start_x += m["w"] + GAP_X

    return positions


# ── E2E Tests ──────────────────────────────────────────────


class TestCanvasLayoutE2E:
    """E2E: Verify /api/animas provides data for canvas org-dashboard."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        from core.config.models import invalidate_cache
        invalidate_cache()
        import server.routes.animas as animas_module
        from core.config.models import load_config
        animas_module.load_config = load_config

    async def test_api_provides_supervisor_for_tree_layout(
        self, tmp_path: Path,
    ) -> None:
        """API data must include supervisor field used by canvas tree layout."""
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

        assert resp.status_code == 200
        animas = resp.json()
        by_name = {p["name"]: p for p in animas}

        assert by_name["sakura"]["supervisor"] is None
        assert by_name["kotoha"]["supervisor"] == "sakura"
        assert by_name["aoi"]["supervisor"] == "sakura"

    async def test_tree_layout_positions_3_node_hierarchy(
        self, tmp_path: Path,
    ) -> None:
        """Verify tree layout produces correct positions for a simple hierarchy."""
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
        roots, _ = build_org_tree(animas)
        positions = compute_tree_layout(roots, 1200)

        assert "sakura" in positions
        assert "kotoha" in positions
        assert "aoi" in positions

        sakura_pos = positions["sakura"]
        kotoha_pos = positions["kotoha"]
        aoi_pos = positions["aoi"]

        assert sakura_pos["y"] < kotoha_pos["y"], "Parent should be above children"
        assert kotoha_pos["y"] == aoi_pos["y"], "Siblings should be at same y level"
        assert kotoha_pos["y"] == sakura_pos["y"] + CARD_H + GAP_Y

    async def test_tree_layout_no_overlap(
        self, tmp_path: Path,
    ) -> None:
        """Cards should not overlap in the computed layout."""
        config_data = {
            "version": 1,
            "setup_complete": True,
            "animas": {
                "boss": {},
                "a": {"supervisor": "boss"},
                "b": {"supervisor": "boss"},
                "c": {"supervisor": "boss"},
                "d": {"supervisor": "a"},
                "e": {"supervisor": "a"},
            },
        }

        app = _create_app_with_config(
            tmp_path,
            anima_names=["boss", "a", "b", "c", "d", "e"],
            config_data=config_data,
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas")

        animas = resp.json()
        roots, _ = build_org_tree(animas)
        positions = compute_tree_layout(roots, 1600)

        cards = list(positions.items())
        for i, (name_a, pos_a) in enumerate(cards):
            for name_b, pos_b in cards[i + 1:]:
                x_overlap = (
                    pos_a["x"] < pos_b["x"] + CARD_W
                    and pos_b["x"] < pos_a["x"] + CARD_W
                )
                y_overlap = (
                    pos_a["y"] < pos_b["y"] + CARD_H
                    and pos_b["y"] < pos_a["y"] + CARD_H
                )
                assert not (x_overlap and y_overlap), (
                    f"Cards {name_a} and {name_b} overlap: "
                    f"{pos_a} vs {pos_b}"
                )

    async def test_svg_connections_can_be_computed(
        self, tmp_path: Path,
    ) -> None:
        """Verify connections (supervisor→child) can be derived from API data."""
        config_data = {
            "version": 1,
            "setup_complete": True,
            "animas": {
                "sakura": {},
                "kotoha": {"supervisor": "sakura"},
                "aoi": {"supervisor": "sakura"},
                "miku": {"supervisor": "kotoha"},
            },
        }

        app = _create_app_with_config(
            tmp_path,
            anima_names=["sakura", "kotoha", "aoi", "miku"],
            config_data=config_data,
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas")

        animas = resp.json()
        roots, node_map = build_org_tree(animas)
        positions = compute_tree_layout(roots, 1200)

        connections = []
        for name, node in node_map.items():
            if node["supervisor"] and node["supervisor"] in positions and name in positions:
                parent_pos = positions[node["supervisor"]]
                child_pos = positions[name]
                x1 = parent_pos["x"] + CARD_W / 2
                y1 = parent_pos["y"] + CARD_H
                x2 = child_pos["x"] + CARD_W / 2
                y2 = child_pos["y"]
                connections.append({
                    "from": node["supervisor"],
                    "to": name,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })

        assert len(connections) == 3
        conn_pairs = {(c["from"], c["to"]) for c in connections}
        assert ("sakura", "kotoha") in conn_pairs
        assert ("sakura", "aoi") in conn_pairs
        assert ("kotoha", "miku") in conn_pairs

        for c in connections:
            assert c["y1"] < c["y2"], (
                f"Connection from {c['from']} to {c['to']}: "
                f"parent bottom ({c['y1']}) should be above child top ({c['y2']})"
            )

    async def test_single_anima_layout(
        self, tmp_path: Path,
    ) -> None:
        """Single anima should be centered at top."""
        config_data = {
            "version": 1,
            "setup_complete": True,
            "animas": {"solo": {}},
        }

        app = _create_app_with_config(
            tmp_path,
            anima_names=["solo"],
            config_data=config_data,
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas")

        animas = resp.json()
        roots, _ = build_org_tree(animas)
        positions = compute_tree_layout(roots, 1200)

        assert "solo" in positions
        pos = positions["solo"]
        assert pos["y"] == 40
        center = pos["x"] + CARD_W / 2
        assert abs(center - 600) < 1, "Single card should be horizontally centered"

    async def test_multiple_roots_layout(
        self, tmp_path: Path,
    ) -> None:
        """Multiple root nodes (no supervisor) should be placed side by side."""
        config_data = {
            "version": 1,
            "setup_complete": True,
            "animas": {
                "alpha": {},
                "beta": {},
                "gamma": {},
            },
        }

        app = _create_app_with_config(
            tmp_path,
            anima_names=["alpha", "beta", "gamma"],
            config_data=config_data,
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas")

        animas = resp.json()
        roots, _ = build_org_tree(animas)
        positions = compute_tree_layout(roots, 1200)

        y_values = {pos["y"] for pos in positions.values()}
        assert len(y_values) == 1, "All roots should be at the same y level"

        x_values = sorted(pos["x"] for pos in positions.values())
        for i in range(len(x_values) - 1):
            gap = x_values[i + 1] - x_values[i]
            assert gap >= CARD_W, "Root cards should not overlap horizontally"

    async def test_empty_animas_layout(
        self, tmp_path: Path,
    ) -> None:
        """Empty anima list should produce empty positions."""
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
        animas = resp.json()
        roots, _ = build_org_tree(animas)
        positions = compute_tree_layout(roots, 1200)
        assert positions == {}
