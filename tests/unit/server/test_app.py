"""Unit tests for server/app.py — FastAPI app factory and lifespan."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.stream_registry import StreamRegistry


# ── create_app ───────────────────────────────────────────


class TestCreateApp:
    """Tests for create_app factory."""

    @patch("core.paths.get_data_dir")
    @patch("server.app.load_config")
    @patch("server.app.ProcessSupervisor")
    @patch("server.app.WebSocketManager")
    def test_create_app_no_animas_dir(
        self, mock_ws_cls, mock_sup_cls, mock_load_config, mock_get_data_dir, tmp_path
    ):
        from server.app import create_app

        animas_dir = tmp_path / "animas"
        shared_dir = tmp_path / "shared"
        # animas_dir does not exist

        mock_ws_cls.return_value = MagicMock()
        mock_sup_cls.return_value = MagicMock()
        mock_load_config.return_value = MagicMock(setup_complete=True)
        mock_get_data_dir.return_value = tmp_path

        app = create_app(animas_dir, shared_dir)

        assert app.state.anima_names == []
        assert app.state.animas_dir == animas_dir
        assert app.state.shared_dir == shared_dir

    @patch("core.paths.get_data_dir")
    @patch("server.app.load_config")
    @patch("server.app.ProcessSupervisor")
    @patch("server.app.WebSocketManager")
    def test_create_app_with_animas(
        self, mock_ws_cls, mock_sup_cls, mock_load_config, mock_get_data_dir, tmp_path
    ):
        from server.app import create_app

        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        shared_dir = tmp_path / "shared"

        # Create a fake anima directory with identity.md
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir()
        (alice_dir / "identity.md").write_text("# Alice", encoding="utf-8")

        mock_ws_cls.return_value = MagicMock()
        mock_sup_cls.return_value = MagicMock()
        mock_load_config.return_value = MagicMock(setup_complete=True)
        mock_get_data_dir.return_value = tmp_path

        app = create_app(animas_dir, shared_dir)

        assert "alice" in app.state.anima_names

    @patch("core.paths.get_data_dir")
    @patch("server.app.load_config")
    @patch("server.app.ProcessSupervisor")
    @patch("server.app.WebSocketManager")
    def test_create_app_skips_dirs_without_identity(
        self, mock_ws_cls, mock_sup_cls, mock_load_config, mock_get_data_dir, tmp_path
    ):
        from server.app import create_app

        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        shared_dir = tmp_path / "shared"

        # Create dir without identity.md
        (animas_dir / "invalid").mkdir()

        mock_ws_cls.return_value = MagicMock()
        mock_sup_cls.return_value = MagicMock()
        mock_load_config.return_value = MagicMock(setup_complete=True)
        mock_get_data_dir.return_value = tmp_path

        app = create_app(animas_dir, shared_dir)

        assert app.state.anima_names == []

    @patch("core.paths.get_data_dir")
    @patch("server.app.load_config")
    @patch("server.app.ProcessSupervisor")
    @patch("server.app.WebSocketManager")
    def test_create_app_skips_files_in_animas_dir(
        self, mock_ws_cls, mock_sup_cls, mock_load_config, mock_get_data_dir, tmp_path
    ):
        from server.app import create_app

        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        shared_dir = tmp_path / "shared"

        # Create a file (not a directory)
        (animas_dir / "not_a_dir.txt").write_text("hello", encoding="utf-8")

        mock_ws_cls.return_value = MagicMock()
        mock_sup_cls.return_value = MagicMock()
        mock_load_config.return_value = MagicMock(setup_complete=True)
        mock_get_data_dir.return_value = tmp_path

        app = create_app(animas_dir, shared_dir)

        assert app.state.anima_names == []

    @patch("core.paths.get_data_dir")
    @patch("server.app.load_config")
    @patch("server.app.ProcessSupervisor")
    @patch("server.app.WebSocketManager")
    def test_create_app_skips_disabled_anima(
        self, mock_ws_cls, mock_sup_cls, mock_load_config, mock_get_data_dir, tmp_path
    ):
        """Anima with status.json enabled:false is excluded from anima_names."""
        import json
        from server.app import create_app

        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        shared_dir = tmp_path / "shared"

        # Enabled anima
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir()
        (alice_dir / "identity.md").write_text("# Alice", encoding="utf-8")
        (alice_dir / "status.json").write_text(
            json.dumps({"enabled": True}), encoding="utf-8"
        )

        # Disabled anima
        bob_dir = animas_dir / "bob"
        bob_dir.mkdir()
        (bob_dir / "identity.md").write_text("# Bob", encoding="utf-8")
        (bob_dir / "status.json").write_text(
            json.dumps({"enabled": False}), encoding="utf-8"
        )

        mock_ws_cls.return_value = MagicMock()
        mock_sup_cls.return_value = MagicMock()
        mock_load_config.return_value = MagicMock(setup_complete=True)
        mock_get_data_dir.return_value = tmp_path

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

        mock_scheduler = MagicMock()
        mock_scheduler_cls.return_value = mock_scheduler

        with patch("core.org_sync.sync_org_structure"), \
             patch("core.org_sync.detect_orphan_animas"), \
             patch("server.app._reconcile_assets_at_startup", new_callable=AsyncMock):
            async with lifespan(mock_app):
                mock_supervisor.start_all.assert_awaited_once_with(["alice"])

        mock_supervisor.shutdown_all.assert_awaited_once()
