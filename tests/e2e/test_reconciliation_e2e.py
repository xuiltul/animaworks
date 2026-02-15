"""E2E tests for Supervisor person reconciliation.

Tests the complete flow: status.json on disk -> API endpoints -> process lifecycle.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient


def _make_app(persons_dir: Path, shared_dir: Path, person_names: list[str]):
    """Create a minimal FastAPI app with both persons and system routers."""
    from fastapi import FastAPI
    from server.routes.persons import create_persons_router
    from server.routes.system import create_system_router

    app = FastAPI()
    app.state.persons_dir = persons_dir
    app.state.shared_dir = shared_dir
    app.state.person_names = list(person_names)

    supervisor = MagicMock()
    supervisor.processes = {name: MagicMock() for name in person_names}
    supervisor.get_process_status.return_value = {"status": "running", "pid": 1234}
    supervisor.get_all_status.return_value = {}

    # Wire up start/stop to mutate supervisor.processes so route guards work
    async def _start(name: str) -> None:
        supervisor.processes[name] = MagicMock()

    async def _stop(name: str) -> None:
        supervisor.processes.pop(name, None)

    supervisor.start_person = AsyncMock(side_effect=_start)
    supervisor.stop_person = AsyncMock(side_effect=_stop)
    supervisor.restart_person = AsyncMock()
    app.state.supervisor = supervisor

    ws_manager = MagicMock()
    ws_manager.active_connections = []
    app.state.ws_manager = ws_manager

    app.include_router(create_persons_router(), prefix="/api")
    app.include_router(create_system_router(), prefix="/api")
    return app


class TestReconciliationE2E:
    """End-to-end tests for the reconciliation workflow."""

    async def test_full_enable_disable_cycle(self, tmp_path):
        """Test: create person -> disable -> enable -> verify status.json states."""
        persons_dir = tmp_path / "persons"
        shared_dir = tmp_path / "shared"
        persons_dir.mkdir()

        # Create person on disk
        alice_dir = persons_dir / "alice"
        alice_dir.mkdir()
        (alice_dir / "identity.md").write_text("# Alice")

        app = _make_app(persons_dir, shared_dir, person_names=["alice"])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Disable
            resp = await client.post("/api/persons/alice/disable")
            assert resp.status_code == 200
            data = resp.json()
            assert data["enabled"] is False

            # Verify status.json on disk
            status_data = json.loads((alice_dir / "status.json").read_text())
            assert status_data["enabled"] is False

            # Verify person removed from person_names
            assert "alice" not in app.state.person_names

            # Enable
            resp = await client.post("/api/persons/alice/enable")
            assert resp.status_code == 200
            data = resp.json()
            assert data["enabled"] is True

            # Verify status.json on disk
            status_data = json.loads((alice_dir / "status.json").read_text())
            assert status_data["enabled"] is True

            # Verify person back in person_names
            assert "alice" in app.state.person_names

    async def test_reload_respects_disabled_status(self, tmp_path):
        """Test: disabled person is not started during reload."""
        persons_dir = tmp_path / "persons"
        shared_dir = tmp_path / "shared"
        persons_dir.mkdir()

        # Create enabled person
        alice_dir = persons_dir / "alice"
        alice_dir.mkdir()
        (alice_dir / "identity.md").write_text("# Alice")
        (alice_dir / "status.json").write_text('{"enabled": true}')

        # Create disabled person
        bob_dir = persons_dir / "bob"
        bob_dir.mkdir()
        (bob_dir / "identity.md").write_text("# Bob")
        (bob_dir / "status.json").write_text('{"enabled": false}')

        app = _make_app(persons_dir, shared_dir, person_names=[])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/system/reload")
            data = resp.json()

            # alice should be added (enabled), bob should not
            assert "alice" in data["added"]
            assert "bob" not in data["added"]
            assert data["total"] == 1

    async def test_disable_then_reload_keeps_disabled(self, tmp_path):
        """Test: disabling via API persists through reload."""
        persons_dir = tmp_path / "persons"
        shared_dir = tmp_path / "shared"
        persons_dir.mkdir()

        alice_dir = persons_dir / "alice"
        alice_dir.mkdir()
        (alice_dir / "identity.md").write_text("# Alice")

        app = _make_app(persons_dir, shared_dir, person_names=["alice"])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Disable alice via API
            resp = await client.post("/api/persons/alice/disable")
            assert resp.status_code == 200

            # Reload -- alice should NOT be re-added because she's disabled
            resp = await client.post("/api/system/reload")
            data = resp.json()
            assert "alice" not in data.get("added", [])
            assert "alice" not in data.get("refreshed", [])

    async def test_person_without_status_json_is_loaded_on_reload(self, tmp_path):
        """Test: backward compatibility -- no status.json means person is enabled."""
        persons_dir = tmp_path / "persons"
        shared_dir = tmp_path / "shared"
        persons_dir.mkdir()

        alice_dir = persons_dir / "alice"
        alice_dir.mkdir()
        (alice_dir / "identity.md").write_text("# Alice")
        # No status.json -- should be treated as enabled

        app = _make_app(persons_dir, shared_dir, person_names=[])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/system/reload")
            data = resp.json()
            assert "alice" in data["added"]
