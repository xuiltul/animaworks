"""Unit tests for server/app.py — FastAPI app factory and lifespan."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.stream_registry import StreamRegistry

# ── create_app ───────────────────────────────────────────


class TestCreateApp:
    """Tests for create_app factory."""

    @pytest.fixture(autouse=True)
    def _completed_setup_config(self, data_dir_at_tmp_path: Path) -> None:
        config_path = data_dir_at_tmp_path / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["setup_complete"] = True
        config_path.write_text(json.dumps(config), encoding="utf-8")
        from core.config import invalidate_cache

        invalidate_cache()

    @patch("server.app.ProcessSupervisor")
    @patch("server.app.WebSocketManager")
    def test_create_app_no_animas_dir(self, mock_ws_cls, mock_sup_cls, tmp_path):
        from server.app import create_app

        animas_dir = tmp_path / "animas"
        shared_dir = tmp_path / "shared"
        # animas_dir does not exist

        mock_ws_cls.return_value = MagicMock()
        mock_sup_cls.return_value = MagicMock()

        app = create_app(animas_dir, shared_dir)

        assert app.state.anima_names == []
        assert app.state.animas_dir == animas_dir
        assert app.state.shared_dir == shared_dir

    @patch("server.app.ProcessSupervisor")
    @patch("server.app.WebSocketManager")
    def test_create_app_with_animas(self, mock_ws_cls, mock_sup_cls, tmp_path):
        from server.app import create_app

        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        shared_dir = tmp_path / "shared"

        # Create a fake anima directory with IDENTITY.md
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir()
        (alice_dir / "IDENTITY.md").write_text("# Alice", encoding="utf-8")

        mock_ws_cls.return_value = MagicMock()
        mock_sup_cls.return_value = MagicMock()

        app = create_app(animas_dir, shared_dir)

        assert "alice" in app.state.anima_names

    @patch("server.app.ProcessSupervisor")
    @patch("server.app.WebSocketManager")
    def test_create_app_skips_dirs_without_identity(self, mock_ws_cls, mock_sup_cls, tmp_path):
        from server.app import create_app

        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        shared_dir = tmp_path / "shared"

        # Create dir without IDENTITY.md
        (animas_dir / "invalid").mkdir()

        mock_ws_cls.return_value = MagicMock()
        mock_sup_cls.return_value = MagicMock()

        app = create_app(animas_dir, shared_dir)

        assert app.state.anima_names == []

    @patch("server.app.ProcessSupervisor")
    @patch("server.app.WebSocketManager")
    def test_create_app_skips_files_in_animas_dir(self, mock_ws_cls, mock_sup_cls, tmp_path):
        from server.app import create_app

        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        shared_dir = tmp_path / "shared"

        # Create a file (not a directory)
        (animas_dir / "not_a_dir.txt").write_text("hello", encoding="utf-8")

        mock_ws_cls.return_value = MagicMock()
        mock_sup_cls.return_value = MagicMock()

        app = create_app(animas_dir, shared_dir)

        assert app.state.anima_names == []

    @patch("server.app.ProcessSupervisor")
    @patch("server.app.WebSocketManager")
    def test_create_app_skips_disabled_anima(self, mock_ws_cls, mock_sup_cls, tmp_path):
        """Anima with status.json enabled:false is excluded from anima_names."""
        from server.app import create_app

        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        shared_dir = tmp_path / "shared"

        # Enabled anima
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir()
        (alice_dir / "IDENTITY.md").write_text("# Alice", encoding="utf-8")
        (alice_dir / "status.json").write_text(json.dumps({"enabled": True}), encoding="utf-8")

        # Disabled anima
        bob_dir = animas_dir / "bob"
        bob_dir.mkdir()
        (bob_dir / "IDENTITY.md").write_text("# Bob", encoding="utf-8")
        (bob_dir / "status.json").write_text(json.dumps({"enabled": False}), encoding="utf-8")

        mock_ws_cls.return_value = MagicMock()
        mock_sup_cls.return_value = MagicMock()

        app = create_app(animas_dir, shared_dir)

        assert "alice" in app.state.anima_names
        assert "bob" not in app.state.anima_names


# ── lifespan ─────────────────────────────────────────────


class TestLifespan:
    """Tests for the lifespan context manager."""

    @pytest.mark.asyncio
    @patch("server.app.AsyncIOScheduler")
    async def test_lifespan_start_and_stop(self, mock_scheduler_cls):
        from server.app import lifespan

        mock_app = MagicMock()
        mock_supervisor = AsyncMock()
        mock_ws_manager = AsyncMock()
        mock_app.state.setup_complete = True
        mock_app.state.supervisor = mock_supervisor
        mock_app.state.ws_manager = mock_ws_manager
        mock_app.state.anima_names = ["alice"]
        mock_app.state.animas_dir = MagicMock()
        mock_app.state.shared_dir = MagicMock()
        mock_app.state.stream_registry = StreamRegistry()
        mock_app.state.slack_socket_manager = None  # prevent await on auto-generated MagicMock
        mock_app.state.discord_gateway_manager = None
        mock_app.state.zoom_gateway_manager = None
        mock_app.state.github_gateway_manager = None
        mock_app.state.startup_preflight_runner = lambda: None

        mock_scheduler = MagicMock()
        mock_scheduler_cls.return_value = mock_scheduler
        mock_supervisor.send_request.return_value = {"collections": []}

        async def fake_startup_animas(app, *, suppress_errors=True):
            await app.state.supervisor.start_all(app.state.anima_names)

        with (
            patch("core.tasks.board.tasks.ensure_task_store_schema"),
            patch("core.org.org_sync.sync_org_structure"),
            patch("core.org.org_sync.detect_orphan_animas"),
            patch("server.app._reconcile_assets_at_startup", new_callable=AsyncMock),
            patch("server.app._prepare_child_env_urls"),
            patch("server.app._startup_animas_background", new=fake_startup_animas),
            patch("core.config.global_permissions.GlobalPermissionsCache.get") as mock_gp,
            patch("core.config.helper_models.validate_helper_model_credentials") as validate_helper_credentials,
        ):
            mock_gp.return_value = MagicMock(loaded=True, check_integrity=MagicMock(return_value=True))
            async with lifespan(mock_app):
                await asyncio.wait_for(mock_app.state._anima_startup_task, timeout=1.0)
                mock_supervisor.start_all.assert_awaited_once_with(["alice"])
                validate_helper_credentials.assert_called_once_with(mock_app.state.animas_dir)
                scheduled_ids = {call.kwargs.get("id") for call in mock_scheduler.add_job.call_args_list}
                assert "asset_reconciliation" not in scheduled_ids

                from core.memory.rag.vector_client import VectorClient
                from core.memory.rag.vector_registry import get_vector_store

                store = get_vector_store("alice")
                assert isinstance(store, VectorClient)
                assert await asyncio.to_thread(store.list_collections) == []
                mock_supervisor.send_request.assert_awaited_once_with(
                    "alice",
                    "memory",
                    {"method": "memory.list_collections", "params": {}},
                )

        mock_supervisor.shutdown_all.assert_awaited_once()

    def test_startup_asset_reconciliation_is_scheduled_once(self, tmp_path):
        from types import SimpleNamespace

        from server.app import _schedule_startup_asset_reconciliation

        app = SimpleNamespace(state=SimpleNamespace(animas_dir=tmp_path))
        with patch("server.app.spawn") as mock_spawn:
            _schedule_startup_asset_reconciliation(app)
            _schedule_startup_asset_reconciliation(app)

        mock_spawn.assert_called_once()
        mock_spawn.assert_called_once_with(
            mock_spawn.call_args.args[0],
            name="startup-asset-reconciliation",
        )
        mock_spawn.call_args.args[0].close()
        assert app.state._asset_reconciliation_scheduled is True


# ── Public icon path (auth bypass regex) ────────────


class TestPublicIconAssetPath:
    """_PUBLIC_ICON_ASSET_PATH allows unauthenticated GET/HEAD for embed."""

    def test_matches_anime_and_realistic_filenames(self):
        from server.app import _PUBLIC_ICON_ASSET_PATH

        assert _PUBLIC_ICON_ASSET_PATH.match("/api/animas/rabbit/assets/icon.png")
        assert _PUBLIC_ICON_ASSET_PATH.match("/api/animas/foo-bar/assets/icon_realistic.png")

    def test_rejects_other_assets_or_traversal(self):
        from server.app import _PUBLIC_ICON_ASSET_PATH

        assert _PUBLIC_ICON_ASSET_PATH.match("/api/animas/rabbit/assets/avatar_bustup.png") is None
        assert _PUBLIC_ICON_ASSET_PATH.match("/api/animas/rabbit/assets/icon.jpg") is None
        assert _PUBLIC_ICON_ASSET_PATH.match("/api/animas/../evil/assets/icon.png") is None
