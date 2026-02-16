"""Unit tests for supervisor field in GET /api/persons response.

Validates that the list_persons endpoint includes the ``supervisor`` field
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
    person_names: list[str] | None = None,
    config_persons: dict | None = None,
):
    """Build a real FastAPI app with mocked externals.

    Parameters
    ----------
    config_persons:
        Dict of person name → PersonModelConfig-like dict to inject
        into the config.  Keys not listed default to empty config.
    """
    persons_dir = tmp_path / "persons"
    persons_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch("server.app.ProcessSupervisor") as mock_sup_cls,
        patch("server.app.load_config") as mock_cfg,
        patch("server.app.WebSocketManager") as mock_ws_cls,
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

        app = create_app(persons_dir, shared_dir)

    if person_names is not None:
        app.state.person_names = person_names

    # Patch load_config used inside the persons route
    if config_persons is not None:
        _patch_route_config(app, config_persons)

    return app


def _patch_route_config(app, config_persons: dict):
    """Patch ``core.config.models.load_config`` to return a config with
    the given person overrides.

    This patches the module-level import used inside the route handler.
    """
    from core.config.models import AnimaWorksConfig, PersonModelConfig

    persons = {}
    for name, overrides in config_persons.items():
        persons[name] = PersonModelConfig(**overrides)

    config = AnimaWorksConfig(persons=persons)

    # Patch load_config in the module where it was imported by persons.py
    import server.routes.persons as persons_module
    app._test_original_load_config = persons_module.load_config
    persons_module.load_config = lambda *a, **kw: config


# ── Tests ──────────────────────────────────────────────


class TestListPersonsSupervisorField:
    """Verify that GET /api/persons includes the supervisor field."""

    @pytest.fixture(autouse=True)
    def _cleanup_patch(self):
        """Restore original load_config after each test."""
        yield
        import server.routes.persons as persons_module
        if hasattr(persons_module, "_test_original_load_config"):
            persons_module.load_config = persons_module._test_original_load_config

    async def test_supervisor_included_in_response(
        self, tmp_path: Path,
    ) -> None:
        """Each person in the response should have a 'supervisor' key."""
        app = _create_app(
            tmp_path,
            person_names=["sakura", "kotoha"],
            config_persons={
                "sakura": {},
                "kotoha": {"supervisor": "sakura"},
            },
        )

        from httpx import ASGITransport, AsyncClient
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

        for person in data:
            assert "supervisor" in person, (
                f"Person '{person['name']}' missing 'supervisor' field"
            )

    async def test_supervisor_value_from_config(
        self, tmp_path: Path,
    ) -> None:
        """supervisor field should reflect the config.json value."""
        app = _create_app(
            tmp_path,
            person_names=["sakura", "kotoha"],
            config_persons={
                "sakura": {},
                "kotoha": {"supervisor": "sakura"},
            },
        )

        from httpx import ASGITransport, AsyncClient
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons")

        data = resp.json()
        by_name = {p["name"]: p for p in data}

        assert by_name["sakura"]["supervisor"] is None
        assert by_name["kotoha"]["supervisor"] == "sakura"

    async def test_supervisor_none_when_not_in_config(
        self, tmp_path: Path,
    ) -> None:
        """Persons not listed in config.json should have supervisor=null."""
        app = _create_app(
            tmp_path,
            person_names=["unknown_person"],
            config_persons={},  # empty config
        )

        from httpx import ASGITransport, AsyncClient
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons")

        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "unknown_person"
        assert data[0]["supervisor"] is None

    async def test_multiple_hierarchy_levels(
        self, tmp_path: Path,
    ) -> None:
        """Three-level hierarchy should be correctly represented."""
        app = _create_app(
            tmp_path,
            person_names=["boss", "manager", "worker"],
            config_persons={
                "boss": {},
                "manager": {"supervisor": "boss"},
                "worker": {"supervisor": "manager"},
            },
        )

        from httpx import ASGITransport, AsyncClient
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons")

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
            person_names=["sakura"],
            config_persons={"sakura": {}},
        )

        from httpx import ASGITransport, AsyncClient
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/persons")

        data = resp.json()
        person = data[0]

        # All pre-existing fields must still be present
        expected_fields = {"name", "status", "bootstrapping", "pid", "uptime_sec", "appearance", "supervisor"}
        assert set(person.keys()) == expected_fields
