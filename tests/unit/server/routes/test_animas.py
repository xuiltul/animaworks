"""Unit tests for server/routes/animas.py — Anima API endpoints."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import ASGITransport, AsyncClient

from server.routes.animas import _read_appearance

# ── Helper to build a minimal FastAPI app with animas router ──


def _make_test_app(
    animas_dir: Path | None = None,
    anima_names: list[str] | None = None,
):
    from fastapi import FastAPI

    from server.routes.animas import create_animas_router

    app = FastAPI()
    app.state.animas_dir = animas_dir or Path("/tmp/fake/animas")
    app.state.anima_names = anima_names or []

    # Mock supervisor
    supervisor = MagicMock()
    supervisor.get_process_status.return_value = {"status": "running", "pid": 1234}
    supervisor.send_request = AsyncMock(return_value={"action": "skip", "summary": "ok"})
    app.state.supervisor = supervisor

    router = create_animas_router()
    app.include_router(router, prefix="/api")
    return app


# ── _read_appearance ─────────────────────────────────────


class TestReadAppearance:
    def test_no_file(self, tmp_path):
        result = _read_appearance(tmp_path)
        assert result is None

    def test_valid_json(self, tmp_path):
        (tmp_path / "appearance.json").write_text(json.dumps({"hair": "black"}), encoding="utf-8")
        result = _read_appearance(tmp_path)
        assert result == {"hair": "black"}

    def test_empty_json(self, tmp_path):
        (tmp_path / "appearance.json").write_text("{}", encoding="utf-8")
        result = _read_appearance(tmp_path)
        assert result is None

    def test_invalid_json(self, tmp_path):
        (tmp_path / "appearance.json").write_text("not json", encoding="utf-8")
        result = _read_appearance(tmp_path)
        assert result is None


# ── GET /animas ─────────────────────────────────────────


class TestListAnimas:
    async def test_list_empty(self):
        app = _make_test_app(anima_names=[])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_list_with_animas(self, tmp_path):
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "IDENTITY.md").write_text("# Alice", encoding="utf-8")

        app = _make_test_app(
            animas_dir=animas_dir,
            anima_names=["alice"],
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "alice"
        assert data[0]["status"] == "running"
        assert data[0]["enabled"] is True

    async def test_list_reports_disabled_from_status_json(self, tmp_path):
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "IDENTITY.md").write_text("# Alice", encoding="utf-8")
        (alice_dir / "status.json").write_text(json.dumps({"enabled": False}), encoding="utf-8")

        app = _make_test_app(animas_dir=animas_dir, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas")
        assert resp.json()[0]["enabled"] is False


# ── GET /animas/{name} ─────────────────────────────────


class TestGetAnima:
    async def test_found(self, tmp_path):
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)

        with patch("core.memory.manager.MemoryManager") as MockMM:
            mock_mm = MagicMock()
            mock_mm.read_identity.return_value = "# Identity"
            mock_mm.read_injection.return_value = ""
            mock_mm.read_current_state.return_value = "idle"
            mock_mm.list_knowledge_files.return_value = ["topic1.md"]
            mock_mm.list_episode_files.return_value = ["2026-01-01.md"]
            mock_mm.list_procedure_files.return_value = []
            MockMM.return_value = mock_mm

            app = _make_test_app(
                animas_dir=animas_dir,
                anima_names=["alice"],
            )
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/animas/alice")
        data = resp.json()
        assert "status" in data
        assert "identity" in data
        assert data["knowledge_files"] == ["topic1.md"]

    async def test_not_found(self):
        app = _make_test_app(anima_names=[])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/nonexistent")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Anima not found: nonexistent"


class TestGetAnimaSummary:
    async def test_summary_returns_only_lightweight_fields(self, tmp_path):
        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        (anima_dir / "status.json").write_text(
            json.dumps({"display_name": "Alice", "model": "test-model", "role": "assistant", "enabled": False}),
            encoding="utf-8",
        )
        app = _make_test_app(animas_dir=tmp_path)

        with (
            patch("core.memory.manager.MemoryManager") as memory_manager,
            patch("server.routes.animas._read_busy_sidecar", return_value={"is_busy": True}),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/api/animas/alice/summary")

        assert response.status_code == 200
        assert len(response.content) < 10 * 1024
        data = response.json()
        assert set(data) == {"name", "display_name", "status", "busy", "model", "role", "enabled"}
        assert data["name"] == "alice"
        assert data["display_name"] == "Alice"
        assert data["model"] == "test-model"
        assert data["role"] == "assistant"
        assert data["enabled"] is False
        assert data["busy"] == {"is_busy": True}
        assert data["status"]["status"] == "running"
        assert "identity" not in data and "knowledge_files" not in data
        memory_manager.assert_not_called()

    async def test_summary_anima_not_found(self, tmp_path):
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/animas/alice/summary")
        assert response.status_code == 404

    async def test_state_endpoint_reads_current_state_without_listing_memory_files(self, tmp_path):
        (tmp_path / "alice").mkdir()
        with patch("core.memory.manager.MemoryManager") as memory_manager:
            memory = MagicMock()
            memory.read_current_state.return_value = "Idle"
            memory_manager.return_value = memory
            app = _make_test_app(animas_dir=tmp_path)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/api/animas/alice/state")

        assert response.status_code == 200
        assert response.json() == {"state": "Idle"}
        memory.read_current_state.assert_called_once_with()
        memory.read_identity.assert_not_called()
        memory.read_injection.assert_not_called()
        memory.list_knowledge_files.assert_not_called()
        memory.list_episode_files.assert_not_called()
        memory.list_procedure_files.assert_not_called()


# ── POST /animas/{name}/trigger ─────────────────────────


class TestTriggerHeartbeat:
    async def test_success(self, tmp_path):
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)

        app = _make_test_app(
            animas_dir=animas_dir,
            anima_names=["alice"],
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/trigger")
        data = resp.json()
        assert data["action"] == "skip"

    async def test_not_found(self):
        app = _make_test_app(anima_names=[])
        # Make supervisor raise KeyError for unknown anima
        app.state.supervisor.send_request = AsyncMock(side_effect=KeyError("nobody"))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/nobody/trigger")
        assert resp.status_code == 404
        assert "Anima not found" in resp.json()["detail"]


# ── GET /animas/{name}/config ───────────────────────────


class TestGetAnimaConfig:
    async def test_anima_not_found(self):
        app = _make_test_app(anima_names=[])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/nobody/config")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Anima not found: nobody"

    async def test_success(self, tmp_path):
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)

        mock_resolved = MagicMock()
        mock_resolved.model = "claude-sonnet-4-6"
        mock_resolved.execution_mode = "a1"
        mock_resolved.model_dump.return_value = {
            "model": "claude-sonnet-4-6",
            "execution_mode": "a1",
        }

        with (
            patch(
                "core.config.models.load_config",
                return_value=MagicMock(),
            ),
            patch(
                "core.config.models.resolve_anima_config",
                return_value=(mock_resolved, "anthropic"),
            ),
        ):
            app = _make_test_app(
                animas_dir=animas_dir,
                anima_names=["alice"],
            )
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/animas/alice/config")

        assert resp.status_code == 200
        data = resp.json()
        assert data["anima"] == "alice"
        assert data["model"] == "claude-sonnet-4-6"
        assert data["execution_mode"] == "a1"
        assert "config" in data


# ── PUT /animas/{name}/model ───────────────────────────


class TestUpdateAnimaModel:
    async def test_updates_status_json_with_execution_mode(self, tmp_path):
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "IDENTITY.md").write_text("# Alice", encoding="utf-8")
        (alice_dir / "status.json").write_text(
            json.dumps({"model": "claude-sonnet-4-6", "credential": "anthropic", "mode_s_auth": "max"}),
            encoding="utf-8",
        )

        app = _make_test_app(animas_dir=animas_dir, anima_names=["alice"])
        transport = ASGITransport(app=app)
        with (
            patch("core.config.io.load_config", return_value=MagicMock()),
            patch(
                "core.config.model_mode.resolve_execution_mode",
                return_value="C",
            ),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.put(
                    "/api/animas/alice/model",
                    json={"model": "codex/gpt-5.4-mini", "credential": "openai"},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["execution_mode"] == "C"
        assert data["credential"] == "openai"
        assert data["mode_s_auth"] is None

        status = json.loads((alice_dir / "status.json").read_text(encoding="utf-8"))
        assert status["model"] == "codex/gpt-5.4-mini"
        assert status["credential"] == "openai"
        assert status["execution_mode"] == "C"
        assert "mode_s_auth" not in status

    async def test_sets_mode_s_auth_for_claude_subscription(self, tmp_path):
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "IDENTITY.md").write_text("# Alice", encoding="utf-8")
        (alice_dir / "status.json").write_text(
            json.dumps({"model": "gpt-4", "credential": "openai"}),
            encoding="utf-8",
        )

        config = MagicMock()
        config.credentials = {"anthropic": MagicMock(type="claude_code_login")}
        config.anima_defaults = MagicMock(mode_s_auth=None)

        app = _make_test_app(animas_dir=animas_dir, anima_names=["alice"])
        transport = ASGITransport(app=app)
        with (
            patch("core.config.io.load_config", return_value=config),
            patch(
                "core.config.model_mode.resolve_execution_mode",
                return_value="S",
            ),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.put(
                    "/api/animas/alice/model",
                    json={"model": "claude-sonnet-4-6", "credential": "anthropic"},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["execution_mode"] == "S"
        assert data["mode_s_auth"] == "max"

        status = json.loads((alice_dir / "status.json").read_text(encoding="utf-8"))
        assert status["execution_mode"] == "S"
        assert status["mode_s_auth"] == "max"

    async def test_sets_mode_s_auth_api_for_claude_api_key(self, tmp_path):
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "IDENTITY.md").write_text("# Alice", encoding="utf-8")
        (alice_dir / "status.json").write_text(
            json.dumps({"model": "gpt-4", "credential": "openai"}),
            encoding="utf-8",
        )

        config = MagicMock()
        config.credentials = {"anthropic": MagicMock(type="api_key")}
        config.anima_defaults = MagicMock(mode_s_auth="max")

        app = _make_test_app(animas_dir=animas_dir, anima_names=["alice"])
        transport = ASGITransport(app=app)
        with (
            patch("core.config.io.load_config", return_value=config),
            patch(
                "core.config.model_mode.resolve_execution_mode",
                return_value="S",
            ),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.put(
                    "/api/animas/alice/model",
                    json={"model": "claude-sonnet-4-6", "credential": "anthropic"},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["execution_mode"] == "S"
        assert data["mode_s_auth"] == "api"

        status = json.loads((alice_dir / "status.json").read_text(encoding="utf-8"))
        assert status["mode_s_auth"] == "api"


# ── POST /animas/{name}/enable ─────────────────────────


class TestEnableAnima:
    async def test_enable_success(self, tmp_path):
        """Enable an anima: status.json written, process started, name added."""
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "IDENTITY.md").write_text("# Alice", encoding="utf-8")

        app = _make_test_app(animas_dir=animas_dir, anima_names=[])
        supervisor = app.state.supervisor
        supervisor.processes = {}
        supervisor.start_anima = AsyncMock()
        supervisor.stop_anima = AsyncMock()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/enable")

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "alice"
        assert data["enabled"] is True

        # Verify status.json was written
        status_file = alice_dir / "status.json"
        assert status_file.exists()
        status_data = json.loads(status_file.read_text(encoding="utf-8"))
        assert status_data["enabled"] is True

        # Process was started and name was added
        supervisor.start_anima.assert_awaited_once_with("alice")
        assert "alice" in app.state.anima_names

    async def test_enable_already_running(self, tmp_path):
        """When anima is already running, status.json is written but start_anima is NOT called."""
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "IDENTITY.md").write_text("# Alice", encoding="utf-8")

        app = _make_test_app(animas_dir=animas_dir, anima_names=["alice"])
        supervisor = app.state.supervisor
        supervisor.processes = {"alice": MagicMock()}
        supervisor.start_anima = AsyncMock()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/enable")

        assert resp.status_code == 200

        # status.json still written
        status_file = alice_dir / "status.json"
        assert status_file.exists()
        status_data = json.loads(status_file.read_text(encoding="utf-8"))
        assert status_data["enabled"] is True

        # start_anima NOT called because already running
        supervisor.start_anima.assert_not_awaited()

    async def test_enable_rejects_corrupt_status_without_overwriting(self, tmp_path):
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "IDENTITY.md").write_text("# Alice", encoding="utf-8")
        status_file = alice_dir / "status.json"
        original = b"{broken status"
        status_file.write_bytes(original)

        app = _make_test_app(animas_dir=animas_dir, anima_names=[])
        supervisor = app.state.supervisor
        supervisor.processes = {}
        supervisor.start_anima = AsyncMock()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/enable")

        assert resp.status_code == 409
        assert status_file.read_bytes() == original
        supervisor.start_anima.assert_not_awaited()

    async def test_enable_not_found(self):
        """Enable a nonexistent anima returns 404."""
        app = _make_test_app(anima_names=[])
        supervisor = app.state.supervisor
        supervisor.processes = {}

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/nonexistent/enable")

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    async def test_enable_no_identity(self, tmp_path):
        """Directory exists but no IDENTITY.md returns 404."""
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        # No IDENTITY.md created

        app = _make_test_app(animas_dir=animas_dir, anima_names=[])
        supervisor = app.state.supervisor
        supervisor.processes = {}

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/enable")

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]


# ── POST /animas/{name}/disable ────────────────────────


class TestDisableAnima:
    async def test_disable_success(self, tmp_path):
        """Disable a running anima: status.json written, process stopped, name removed."""
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "IDENTITY.md").write_text("# Alice", encoding="utf-8")

        app = _make_test_app(animas_dir=animas_dir, anima_names=["alice"])
        supervisor = app.state.supervisor
        supervisor.processes = {"alice": MagicMock()}
        supervisor.start_anima = AsyncMock()
        supervisor.stop_anima = AsyncMock()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/disable")

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "alice"
        assert data["enabled"] is False

        # Verify status.json was written
        status_file = alice_dir / "status.json"
        assert status_file.exists()
        status_data = json.loads(status_file.read_text(encoding="utf-8"))
        assert status_data["enabled"] is False

        # Process was stopped and name was removed
        supervisor.stop_anima.assert_awaited_once_with("alice")
        assert "alice" not in app.state.anima_names

    async def test_disable_not_running(self, tmp_path):
        """When anima is not running, status.json is written and stop_anima is still called (no-op safe)."""
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "IDENTITY.md").write_text("# Alice", encoding="utf-8")

        app = _make_test_app(animas_dir=animas_dir, anima_names=[])
        supervisor = app.state.supervisor
        supervisor.processes = {}
        supervisor.stop_anima = AsyncMock()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/disable")

        assert resp.status_code == 200

        # status.json still written
        status_file = alice_dir / "status.json"
        assert status_file.exists()
        status_data = json.loads(status_file.read_text(encoding="utf-8"))
        assert status_data["enabled"] is False

        # Always call stop_anima so in-flight start races are covered
        supervisor.stop_anima.assert_awaited_once_with("alice")

    async def test_disable_rejects_corrupt_status_without_overwriting(self, tmp_path):
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "IDENTITY.md").write_text("# Alice", encoding="utf-8")
        status_file = alice_dir / "status.json"
        original = b"{broken status"
        status_file.write_bytes(original)

        app = _make_test_app(animas_dir=animas_dir, anima_names=["alice"])
        supervisor = app.state.supervisor
        supervisor.processes = {"alice": MagicMock()}
        supervisor.stop_anima = AsyncMock()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/disable")

        assert resp.status_code == 409
        assert status_file.read_bytes() == original
        supervisor.stop_anima.assert_not_awaited()

    async def test_disable_not_found(self):
        """Disable a nonexistent anima returns 404."""
        app = _make_test_app(anima_names=[])
        supervisor = app.state.supervisor
        supervisor.processes = {}

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/nonexistent/disable")

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]


# ── POST /animas/{name}/start ──────────────────────────


class TestStartAnimaEndpoint:
    async def test_start_disabled_returns_409(self, tmp_path):
        """Disabled anima on disk (not in anima_names): 409, not 404."""
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "IDENTITY.md").write_text("# Alice", encoding="utf-8")
        (alice_dir / "status.json").write_text(
            json.dumps({"enabled": False}),
            encoding="utf-8",
        )

        # Do not inject into anima_names — existence is disk-based.
        app = _make_test_app(animas_dir=animas_dir, anima_names=[])
        supervisor = app.state.supervisor
        supervisor.processes = {}
        supervisor.get_process_status.return_value = {"status": "stopped"}
        supervisor.start_anima = AsyncMock()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/start")

        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert "disabled" in detail.lower()
        assert "enable" in detail.lower()
        supervisor.start_anima.assert_not_awaited()

    async def test_start_not_found_returns_404(self, tmp_path):
        """Missing anima directory / IDENTITY.md → 404."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir(parents=True)

        app = _make_test_app(animas_dir=animas_dir, anima_names=[])
        supervisor = app.state.supervisor
        supervisor.start_anima = AsyncMock()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/ghost/start")

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()
        supervisor.start_anima.assert_not_awaited()

    async def test_start_enabled_starts_process(self, tmp_path):
        """Enabled anima: POST /start proceeds and calls start_anima."""
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "IDENTITY.md").write_text("# Alice", encoding="utf-8")
        (alice_dir / "status.json").write_text(
            json.dumps({"enabled": True}),
            encoding="utf-8",
        )

        app = _make_test_app(animas_dir=animas_dir, anima_names=["alice"])
        supervisor = app.state.supervisor
        supervisor.processes = {}
        supervisor.get_process_status.return_value = {"status": "stopped"}

        async def _start(name: str) -> None:
            supervisor.processes[name] = MagicMock()

        supervisor.start_anima = AsyncMock(side_effect=_start)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/start")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "started"
        assert data["name"] == "alice"
        supervisor.start_anima.assert_awaited_once_with("alice")

    async def test_start_enabled_appends_to_anima_names(self, tmp_path):
        """Start success adds name to anima_names when missing (enable→start path)."""
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "IDENTITY.md").write_text("# Alice", encoding="utf-8")
        (alice_dir / "status.json").write_text(
            json.dumps({"enabled": True}),
            encoding="utf-8",
        )

        app = _make_test_app(animas_dir=animas_dir, anima_names=[])
        supervisor = app.state.supervisor
        supervisor.processes = {}
        supervisor.get_process_status.return_value = {"status": "stopped"}

        async def _start(name: str) -> None:
            supervisor.processes[name] = MagicMock()

        supervisor.start_anima = AsyncMock(side_effect=_start)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/start")

        assert resp.status_code == 200
        assert "alice" in app.state.anima_names
        supervisor.start_anima.assert_awaited_once_with("alice")

    async def test_start_refused_returns_409(self, tmp_path):
        """If start_anima runs but does not register a process (race with disable), return 409."""
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "IDENTITY.md").write_text("# Alice", encoding="utf-8")
        (alice_dir / "status.json").write_text(
            json.dumps({"enabled": True}),
            encoding="utf-8",
        )

        app = _make_test_app(animas_dir=animas_dir, anima_names=[])
        supervisor = app.state.supervisor
        supervisor.processes = {}
        supervisor.get_process_status.return_value = {"status": "stopped"}
        # start_anima no-ops (e.g. disabled mid-flight) — processes stays empty
        supervisor.start_anima = AsyncMock()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/start")

        assert resp.status_code == 409
        detail = resp.json()["detail"].lower()
        assert "disabled" in detail or "refused" in detail
        assert "alice" not in app.state.anima_names
        supervisor.start_anima.assert_awaited_once_with("alice")


class TestDeleteAnima:
    async def test_running_anima_that_cannot_be_stopped_returns_409_without_deleting(self, tmp_path):
        data_dir = tmp_path / "runtime"
        animas_dir = data_dir / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "IDENTITY.md").write_text("# Alice", encoding="utf-8")
        (alice_dir / "status.json").write_text(json.dumps({"enabled": True}), encoding="utf-8")
        (data_dir / "config.json").write_text(
            json.dumps({"version": 1, "animas": {"alice": {}}}),
            encoding="utf-8",
        )

        app = _make_test_app(animas_dir=animas_dir, anima_names=["alice"])
        supervisor = app.state.supervisor
        handle = MagicMock()
        handle.process.poll.return_value = None
        supervisor.processes = {"alice": handle}
        supervisor.stop_anima = AsyncMock()

        with patch("core.anima.roster.refresh_anima_roster"):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.delete("/api/animas/alice")

        assert resp.status_code == 409
        assert alice_dir.is_dir()
        assert (alice_dir / "IDENTITY.md").is_file()
        assert not (data_dir / "archive").exists()
        assert "alice" in app.state.anima_names
        supervisor.stop_anima.assert_awaited_once_with("alice")

    async def test_success_archives_unregisters_and_reports_supervisor_warning(self, tmp_path):
        data_dir = tmp_path / "runtime"
        animas_dir = data_dir / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "IDENTITY.md").write_text("# Alice", encoding="utf-8")
        (alice_dir / "status.json").write_text(json.dumps({"enabled": True}), encoding="utf-8")
        bob_dir = animas_dir / "bob"
        bob_dir.mkdir()
        (bob_dir / "status.json").write_text(json.dumps({"supervisor": "alice"}), encoding="utf-8")
        config_path = data_dir / "config.json"
        config_path.write_text(json.dumps({"version": 1, "animas": {"alice": {}}}), encoding="utf-8")

        app = _make_test_app(animas_dir=animas_dir, anima_names=["alice", "bob"])
        supervisor = app.state.supervisor
        handle = MagicMock()
        handle.process.poll.return_value = None
        supervisor.processes = {"alice": handle}

        async def _stop(name: str) -> None:
            supervisor.processes.pop(name, None)

        supervisor.stop_anima = AsyncMock(side_effect=_stop)

        with patch("core.anima.roster.refresh_anima_roster"):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.delete("/api/animas/alice")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deleted"
        assert data["archive_path"]
        assert Path(data["archive_path"]).is_file()
        assert data["supervisor_warnings"] == ["Anima 'bob' has deleted anima 'alice' as supervisor"]
        assert not alice_dir.exists()
        assert app.state.anima_names == ["bob"]
        assert "alice" not in json.loads(config_path.read_text(encoding="utf-8"))["animas"]
        supervisor.stop_anima.assert_awaited_once_with("alice")


class TestRootOwnedPromptSettings:
    async def test_background_model_update_reloads_running_anima(self, tmp_path):
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "IDENTITY.md").write_text("Alice\n", encoding="utf-8")
        (anima_dir / "status.json").write_text(json.dumps({"enabled": True}), encoding="utf-8")
        app = _make_test_app(animas_dir=tmp_path / "animas")
        app.state.supervisor.processes = {"alice": MagicMock()}

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.put(
                "/api/animas/alice/background-model",
                json={"model": "claude-haiku-4-5"},
            )

        assert response.status_code == 200
        assert json.loads((anima_dir / "status.json").read_text(encoding="utf-8"))["background_model"] == (
            "claude-haiku-4-5"
        )
        app.state.supervisor.send_request.assert_awaited_once_with("alice", "reload_config", {}, timeout=10.0)

    async def test_identity_injection_and_permissions_use_atomic_settings_writers(self, tmp_path):
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "IDENTITY.md").write_text("Old identity\n", encoding="utf-8")
        (anima_dir / "AGENTS.md").write_text("Old injection\n", encoding="utf-8")
        (anima_dir / "permissions.json").write_text("{}\n", encoding="utf-8")
        app = _make_test_app(animas_dir=tmp_path / "animas")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            identity = await client.put("/api/animas/alice/identity", json={"content": "New identity"})
            injection = await client.put("/api/animas/alice/injection", json={"content": "New injection"})
            permissions = await client.put(
                "/api/animas/alice/permissions",
                json={"version": 1, "file_roots": ["/tmp/workspace"]},
            )

        assert identity.status_code == injection.status_code == permissions.status_code == 200
        assert (anima_dir / "IDENTITY.md").read_text(encoding="utf-8") == "New identity"
        assert (anima_dir / "AGENTS.md").read_text(encoding="utf-8") == "New injection"
        assert json.loads((anima_dir / "permissions.json").read_text(encoding="utf-8"))["file_roots"] == [
            "/tmp/workspace"
        ]

    async def test_running_server_rename_moves_settings_on_root(self, tmp_path, monkeypatch):
        from core.config.models import AnimaModelConfig, AnimaWorksConfig, load_config, save_config

        data_dir = tmp_path / "runtime"
        animas_dir = data_dir / "animas"
        old_dir = animas_dir / "alice"
        old_dir.mkdir(parents=True)
        (old_dir / "IDENTITY.md").write_text("Alice\n", encoding="utf-8")
        (old_dir / "status.json").write_text(json.dumps({"enabled": True, "role": "general"}), encoding="utf-8")
        (old_dir / "permissions.json").write_text(json.dumps({"version": 1, "file_roots": []}), encoding="utf-8")
        save_config(AnimaWorksConfig(animas={"alice": AnimaModelConfig(supervisor=None)}), data_dir / "config.json")
        monkeypatch.setattr("core.paths.get_data_dir", lambda: data_dir)
        app = _make_test_app(animas_dir=animas_dir, anima_names=["alice"])
        app.state.supervisor.stop_anima = AsyncMock()
        app.state.supervisor.start_anima = AsyncMock()

        with patch("core.anima.admin.cleanup_rag_collections", return_value=False):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/api/animas/alice/rename", json={"new_name": "alicia"})

        assert response.status_code == 200
        assert response.json()["new_name"] == "alicia"
        assert not old_dir.exists()
        assert (animas_dir / "alicia" / "IDENTITY.md").is_file()
        assert "alicia" in load_config(data_dir / "config.json").animas
        app.state.supervisor.stop_anima.assert_awaited_once_with("alice")
        app.state.supervisor.start_anima.assert_awaited_once_with("alicia")
        assert app.state.anima_names == ["alicia"]
