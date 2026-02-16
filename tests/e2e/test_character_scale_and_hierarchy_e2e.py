"""E2E tests for workspace character scale and hierarchy bug fixes.

Bug 1: Validates character.js bone-based bounding box via static file serving.
Bug 2: Validates /api/persons returns correct 3-level hierarchy with the
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
    person_names: list[str],
    config_data: dict,
):
    """Build a real FastAPI app with a concrete config.json on disk."""
    persons_dir = tmp_path / "persons"
    persons_dir.mkdir(parents=True, exist_ok=True)
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
    ):
        cfg = MagicMock()
        cfg.setup_complete = True
        mock_app_cfg.return_value = cfg

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

        app = create_app(persons_dir, shared_dir)

    app.state.person_names = person_names

    from core.config.models import invalidate_cache, load_config as real_load_config

    invalidate_cache()
    real_config = real_load_config(config_path)

    import server.routes.persons as persons_module
    persons_module.load_config = lambda *a, **kw: real_config

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


# ── Bug 2: Organization hierarchy ────────────────────────────


@pytest.mark.e2e
class TestOrganizationHierarchyFix:
    """E2E: Verify /api/persons returns correct 3-level hierarchy."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        from core.config.models import invalidate_cache
        invalidate_cache()
        import server.routes.persons as persons_module
        from core.config.models import load_config
        persons_module.load_config = load_config

    async def test_full_animaworks_hierarchy(self, tmp_path: Path) -> None:
        """Verify the complete AnimaWorks hierarchy with all 5 persons.

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
            "persons": {
                "sakura": {},
                "kotoha": {"supervisor": "sakura"},
                "chatwork_checker": {"supervisor": "kotoha"},
                "rin": {"supervisor": "sakura"},
                "aoi": {"supervisor": "rin"},
            },
        }

        app = _create_app_with_config(
            tmp_path,
            person_names=["sakura", "kotoha", "chatwork_checker", "rin", "aoi"],
            config_data=config_data,
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons")

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
            "persons": {
                "sakura": {},
                "kotoha": {"supervisor": "sakura"},
                "chatwork_checker": {"supervisor": "kotoha"},
                "rin": {"supervisor": "sakura"},
                "aoi": {"supervisor": "rin"},
            },
        }

        app = _create_app_with_config(
            tmp_path,
            person_names=["sakura", "kotoha", "chatwork_checker", "rin", "aoi"],
            config_data=config_data,
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons")

        persons = resp.json()

        # Simulate buildOrgTree from office3d.js
        node_map = {}
        for p in persons:
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

        Verify that 4 connector lines are drawn for the 5-person hierarchy.
        """
        config_data = {
            "version": 1,
            "setup_complete": True,
            "persons": {
                "sakura": {},
                "kotoha": {"supervisor": "sakura"},
                "chatwork_checker": {"supervisor": "kotoha"},
                "rin": {"supervisor": "sakura"},
                "aoi": {"supervisor": "rin"},
            },
        }

        app = _create_app_with_config(
            tmp_path,
            person_names=["sakura", "kotoha", "chatwork_checker", "rin", "aoi"],
            config_data=config_data,
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons")

        persons = resp.json()

        # Simulate buildConnectors: draw line for each person with supervisor
        person_names = {p["name"] for p in persons}
        connectors = []
        for p in persons:
            if p.get("supervisor") and p["supervisor"] in person_names:
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
    """E2E: Verify org-sync can repair supervisor for old-flow persons."""

    def test_sync_repairs_rin_aoi_scenario(self, tmp_path: Path) -> None:
        """Simulate the exact rin/aoi scenario: identity.md has 上司 row
        added as data repair, and org-sync picks it up correctly.
        """
        from core.config.models import (
            AnimaWorksConfig,
            PersonModelConfig,
            invalidate_cache,
            load_config,
            save_config,
        )
        from core.org_sync import sync_org_structure

        data_dir = tmp_path / "animaworks"
        data_dir.mkdir()
        persons_dir = data_dir / "persons"
        persons_dir.mkdir()
        config_path = data_dir / "config.json"

        # Initial buggy state: rin and aoi have null supervisors
        cfg = AnimaWorksConfig(
            setup_complete=True,
            persons={
                "sakura": PersonModelConfig(supervisor=None),
                "kotoha": PersonModelConfig(supervisor="sakura"),
                "chatwork_checker": PersonModelConfig(supervisor="kotoha"),
                "rin": PersonModelConfig(supervisor=None),
                "aoi": PersonModelConfig(supervisor=None),
            },
        )
        save_config(cfg, config_path)
        invalidate_cache()

        # Create person directories with identity.md
        # rin: now has the repaired 上司 row
        rin_dir = persons_dir / "rin"
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
        aoi_dir = persons_dir / "aoi"
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

        # Other persons (minimal)
        for name in ("sakura", "kotoha", "chatwork_checker"):
            d = persons_dir / name
            d.mkdir(parents=True)
            (d / "identity.md").write_text(f"# {name}\n", encoding="utf-8")

        # Run org-sync
        result = sync_org_structure(persons_dir, config_path)

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
        assert updated_cfg.persons["rin"].supervisor == "sakura"
        assert updated_cfg.persons["aoi"].supervisor == "rin"
