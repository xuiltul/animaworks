# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for workspace character scale and hierarchy bug fixes.

Bug 1: Validates character.js bone-based bounding box via static file serving.
Bug 1b: Validates animation-first scaling (idle animation frame 0 applied
        before bounding box computation) and sanity cap (maxHeight=0.8).
Bug 2: Validates /api/animas returns correct 3-level hierarchy with the
       expected supervisor relationships:
       sakura (root) -> kotoha -> chatwork_checker
       sakura (root) -> rin -> aoi
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── Helpers ──────────────────────────────────────────────────


def _create_app_with_config(
    tmp_path: Path,
    anima_names: list[str],
    config_data: dict,
):
    """Build a real FastAPI app with a concrete config.json on disk."""
    animas_dir = tmp_path / "animas"
    animas_dir.mkdir(parents=True, exist_ok=True)
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

    # Persist auth mock beyond the with-block for request-time middleware
    import server.app as _sa
    _auth = MagicMock()
    _auth.auth_mode = "local_trust"
    _sa.load_auth = lambda: _auth

    app.state.anima_names = anima_names

    from core.config.models import invalidate_cache, load_config as real_load_config

    invalidate_cache()
    real_config = real_load_config(config_path)

    import server.routes.animas as animas_module
    animas_module.load_config = lambda *a, **kw: real_config

    return app


# ── Bug 1: Character scale (static file analysis) ────────────


@pytest.mark.e2e
class TestCharacterScaleFix:
    """E2E: Verify character.js uses bone-based bounding box for GLB models."""

    def test_character_js_bone_based_bbox(self) -> None:
        """Fetch character.js and verify it uses bone-based bounding box.

        The old code used Box3.setFromObject(model) which misses the
        100x armature scale on VRoid/Blender GLB models where the
        Armature node is a sibling of SkinnedMesh, not a parent.
        """
        from fastapi import FastAPI
        from fastapi.staticfiles import StaticFiles
        from fastapi.testclient import TestClient

        app = FastAPI()
        static_dir = _PROJECT_ROOT / "server" / "static"
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

        client = TestClient(app)
        resp = client.get("/workspace/modules/character.js")
        assert resp.status_code == 200

        content = resp.text

        # Must use bone-based approach
        assert "getWorldPosition" in content, (
            "character.js must use bone.getWorldPosition() for bounding box"
        )
        assert "isSkinnedMesh" in content, (
            "character.js must check isSkinnedMesh before bone traversal"
        )
        assert "expandByPoint" in content, (
            "character.js must use box.expandByPoint() with bone positions"
        )

        # Must NOT have the old bare Box3.setFromObject(model) pattern
        # for initial measurement (only as fallback for non-skinned models)
        assert "new THREE.Box3().setFromObject(model)" not in content, (
            "character.js must not use Box3().setFromObject(model) directly"
        )


@pytest.mark.e2e
class TestAnimationFirstScaleFix:
    """E2E: Verify character.js loads idle animation before computing bounds.

    natsume's bind-pose Hips bone is rotated ~50° giving half the Y-extent
    of other characters.  The fix loads animations and applies frame 0
    before computing the bounding box, plus a sanity cap at 0.8 units.
    """

    def test_animation_before_bbox_via_static_serving(self) -> None:
        """Fetch character.js and verify animation-first scaling pattern."""
        from fastapi import FastAPI
        from fastapi.staticfiles import StaticFiles
        from fastapi.testclient import TestClient

        app = FastAPI()
        static_dir = _PROJECT_ROOT / "server" / "static"
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

        client = TestClient(app)
        resp = client.get("/workspace/modules/character.js")
        assert resp.status_code == 200

        content = resp.text

        # AnimationMixer must appear before Box3
        mixer_idx = content.find("new THREE.AnimationMixer")
        box_idx = content.find("new THREE.Box3()")
        assert mixer_idx != -1, "character.js must create AnimationMixer"
        assert box_idx != -1, "character.js must create Box3"
        assert mixer_idx < box_idx, (
            "AnimationMixer must be created before Box3 for animation-first scaling"
        )

        # mixer.setTime(0) must appear before Box3
        set_time_idx = content.find("mixer.setTime(0)")
        assert set_time_idx != -1, (
            "character.js must call mixer.setTime(0) to apply idle frame"
        )
        assert set_time_idx < box_idx, (
            "mixer.setTime(0) must be called before Box3 computation"
        )

        # Sanity cap
        assert "maxHeight" in content, (
            "character.js must define maxHeight sanity cap"
        )
        assert "maxHeight = 0.8" in content, (
            "maxHeight must be 0.8 to cap oversized characters"
        )


# ── Bug 2: Organization hierarchy ────────────────────────────


@pytest.mark.e2e
class TestOrganizationHierarchyFix:
    """E2E: Verify /api/animas returns correct 3-level hierarchy."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        from core.config.models import invalidate_cache
        invalidate_cache()
        import server.routes.animas as animas_module
        from core.config.models import load_config
        animas_module.load_config = load_config

    async def test_full_animaworks_hierarchy(self, tmp_path: Path) -> None:
        """Verify the complete AnimaWorks hierarchy with all 5 animas.

        Expected tree:
        sakura (root)
        ├── kotoha
        │   └── chatwork_checker
        └── rin
            └── aoi
        """
        config_data = {
            "version": 1,
            "setup_complete": True,
            "animas": {
                "sakura": {},
                "kotoha": {"supervisor": "sakura"},
                "chatwork_checker": {"supervisor": "kotoha"},
                "rin": {"supervisor": "sakura"},
                "aoi": {"supervisor": "rin"},
            },
        }

        app = _create_app_with_config(
            tmp_path,
            anima_names=["sakura", "kotoha", "chatwork_checker", "rin", "aoi"],
            config_data=config_data,
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 5

        by_name = {p["name"]: p for p in data}

        # Verify supervisor relationships
        assert by_name["sakura"]["supervisor"] is None
        assert by_name["kotoha"]["supervisor"] == "sakura"
        assert by_name["chatwork_checker"]["supervisor"] == "kotoha"
        assert by_name["rin"]["supervisor"] == "sakura"
        assert by_name["aoi"]["supervisor"] == "rin"

    async def test_build_org_tree_simulation(self, tmp_path: Path) -> None:
        """Simulate buildOrgTree from office3d.js and verify tree structure.

        Ensures exactly 1 root (sakura) with correct children at each level.
        """
        config_data = {
            "version": 1,
            "setup_complete": True,
            "animas": {
                "sakura": {},
                "kotoha": {"supervisor": "sakura"},
                "chatwork_checker": {"supervisor": "kotoha"},
                "rin": {"supervisor": "sakura"},
                "aoi": {"supervisor": "rin"},
            },
        }

        app = _create_app_with_config(
            tmp_path,
            anima_names=["sakura", "kotoha", "chatwork_checker", "rin", "aoi"],
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

        # Exactly 1 root: sakura
        assert len(roots) == 1, (
            f"Expected 1 root, got {len(roots)}: "
            f"{[r['name'] for r in roots]}"
        )
        assert roots[0]["name"] == "sakura"

        # sakura has 2 direct children: kotoha and rin
        sakura_children = sorted(c["name"] for c in roots[0]["children"])
        assert sakura_children == ["kotoha", "rin"], (
            f"sakura children should be [kotoha, rin], got {sakura_children}"
        )

        # kotoha has 1 child: chatwork_checker
        kotoha_node = node_map["kotoha"]
        assert len(kotoha_node["children"]) == 1
        assert kotoha_node["children"][0]["name"] == "chatwork_checker"

        # rin has 1 child: aoi
        rin_node = node_map["rin"]
        assert len(rin_node["children"]) == 1
        assert rin_node["children"][0]["name"] == "aoi"

        # chatwork_checker and aoi are leaf nodes
        assert len(node_map["chatwork_checker"]["children"]) == 0
        assert len(node_map["aoi"]["children"]) == 0

    async def test_connector_lines_for_full_hierarchy(
        self, tmp_path: Path,
    ) -> None:
        """Simulate buildConnectors from office3d.js.

        Verify that 4 connector lines are drawn for the 5-anima hierarchy.
        """
        config_data = {
            "version": 1,
            "setup_complete": True,
            "animas": {
                "sakura": {},
                "kotoha": {"supervisor": "sakura"},
                "chatwork_checker": {"supervisor": "kotoha"},
                "rin": {"supervisor": "sakura"},
                "aoi": {"supervisor": "rin"},
            },
        }

        app = _create_app_with_config(
            tmp_path,
            anima_names=["sakura", "kotoha", "chatwork_checker", "rin", "aoi"],
            config_data=config_data,
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas")

        animas = resp.json()

        # Simulate buildConnectors: draw line for each anima with supervisor
        anima_names = {p["name"] for p in animas}
        connectors = []
        for p in animas:
            if p.get("supervisor") and p["supervisor"] in anima_names:
                connectors.append((p["supervisor"], p["name"]))

        connectors.sort()
        assert len(connectors) == 4, (
            f"Expected 4 connector lines, got {len(connectors)}: {connectors}"
        )
        assert ("kotoha", "chatwork_checker") in connectors
        assert ("rin", "aoi") in connectors
        assert ("sakura", "kotoha") in connectors
        assert ("sakura", "rin") in connectors


# ── Bug 2: Supervisor data repair via org-sync ────────────────


@pytest.mark.e2e
class TestSupervisorDataRepair:
    """E2E: Verify org-sync can repair supervisor for old-flow animas."""

    def test_sync_repairs_rin_aoi_scenario(self, tmp_path: Path) -> None:
        """Simulate the exact rin/aoi scenario: identity.md has 上司 row
        added as data repair, and org-sync picks it up correctly.
        """
        from core.config.models import (
            AnimaWorksConfig,
            AnimaModelConfig,
            invalidate_cache,
            load_config,
            save_config,
        )
        from core.org_sync import sync_org_structure

        data_dir = tmp_path / "animaworks"
        data_dir.mkdir()
        animas_dir = data_dir / "animas"
        animas_dir.mkdir()
        config_path = data_dir / "config.json"

        # Initial buggy state: rin and aoi have null supervisors
        cfg = AnimaWorksConfig(
            setup_complete=True,
            animas={
                "sakura": AnimaModelConfig(supervisor=None),
                "kotoha": AnimaModelConfig(supervisor="sakura"),
                "chatwork_checker": AnimaModelConfig(supervisor="kotoha"),
                "rin": AnimaModelConfig(supervisor=None),
                "aoi": AnimaModelConfig(supervisor=None),
            },
        )
        save_config(cfg, config_path)
        invalidate_cache()

        # Create anima directories with identity.md
        # rin: now has the repaired 上司 row
        rin_dir = animas_dir / "rin"
        rin_dir.mkdir(parents=True)
        (rin_dir / "identity.md").write_text(
            "# Identity: rin\n\n"
            "## 基本プロフィール\n\n"
            "| 項目 | 設定 |\n"
            "|------|------|\n"
            "| 上司 | 桜庭咲良（sakura） |\n"
            "| 誕生日 | 2月14日 |\n",
            encoding="utf-8",
        )

        # aoi: now has the repaired 上司 row
        aoi_dir = animas_dir / "aoi"
        aoi_dir.mkdir(parents=True)
        (aoi_dir / "identity.md").write_text(
            "# Identity: aoi\n\n"
            "## 基本プロフィール\n\n"
            "| 項目 | 設定 |\n"
            "|------|------|\n"
            "| 上司 | 凛堂凛（rin） |\n"
            "| 誕生日 | 7月20日 |\n",
            encoding="utf-8",
        )

        # Other animas (minimal)
        for name in ("sakura", "kotoha", "chatwork_checker"):
            d = animas_dir / name
            d.mkdir(parents=True)
            (d / "identity.md").write_text(f"# {name}\n", encoding="utf-8")

        # Run org-sync
        result = sync_org_structure(animas_dir, config_path)

        # rin's supervisor should now be resolved to "sakura"
        assert result["rin"] == "sakura", (
            f"Expected rin supervisor='sakura', got '{result['rin']}'"
        )
        # aoi's supervisor should now be resolved to "rin"
        assert result["aoi"] == "rin", (
            f"Expected aoi supervisor='rin', got '{result['aoi']}'"
        )

        # Verify config.json was updated
        invalidate_cache()
        updated_cfg = load_config(config_path)
        assert updated_cfg.animas["rin"].supervisor == "sakura"
        assert updated_cfg.animas["aoi"].supervisor == "rin"
