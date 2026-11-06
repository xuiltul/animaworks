"""Unit tests for channel POST authentication hardening."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture(autouse=True)
def _sakura_status(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "sakura"
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "status.json").write_text("{}", encoding="utf-8")


def _make_test_app(shared_dir: Path):
    from fastapi import FastAPI

    from server.routes.channels import create_channels_router

    anima_dir = shared_dir.parent / "animas" / "sakura"
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "status.json").write_text("{}", encoding="utf-8")
    app = FastAPI()
    app.state.shared_dir = shared_dir
    app.state.ws_manager = MagicMock()
    app.state.ws_manager.broadcast = AsyncMock()
    router = create_channels_router()
    app.include_router(router, prefix="/api")
    return app


def _write_channel(shared_dir: Path, name: str, entries: list[dict]) -> None:
    channels_dir = shared_dir / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    filepath = channels_dir / f"{name}.jsonl"
    lines = [json.dumps(e, ensure_ascii=False) for e in entries]
    filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── max_length validation ────────────────────────────────


class TestChannelPostMaxLength:
    async def test_10000_chars_accepted(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        _write_channel(shared_dir, "general", [])

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        text = "a" * 10000
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/channels/general",
                json={"text": text},
            )
        assert resp.status_code == 200

    async def test_10001_chars_rejected_422(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        _write_channel(shared_dir, "general", [])

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        text = "a" * 10001
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/channels/general",
                json={"text": text},
            )
        assert resp.status_code == 422


# ── from_name min_length validation ──────────────────────


class TestChannelPostFromNameMinLength:
    async def test_empty_from_name_rejected_422(self, tmp_path: Path):
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        _write_channel(shared_dir, "general", [])

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/channels/general",
                json={"text": "Hello", "from_name": ""},
            )
        assert resp.status_code == 422


# ── Unauthenticated from_name forced to "human" ─────────


class TestChannelPostUnauthenticatedFromName:
    async def test_unauthenticated_from_name_forced_to_human(self, tmp_path: Path):
        """Without auth, from_name should be forced to 'human' regardless of request."""
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        _write_channel(shared_dir, "general", [])

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/channels/general",
                json={"text": "Trying to impersonate", "from_name": "sakura"},
            )
        assert resp.status_code == 200

        entry = json.loads((shared_dir / "channels" / "general.jsonl").read_text(encoding="utf-8"))
        assert entry["from"] == "human"

    async def test_unauthenticated_default_from_name_is_human(self, tmp_path: Path):
        """Default from_name (no from_name in request) is 'human' when unauthenticated."""
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        _write_channel(shared_dir, "general", [])

        app = _make_test_app(shared_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/channels/general",
                json={"text": "Default name test"},
            )
        assert resp.status_code == 200

        entry = json.loads((shared_dir / "channels" / "general.jsonl").read_text(encoding="utf-8"))
        assert entry["from"] == "human"


# ── post_channel() from_name validation ──────────────────


class TestPostChannelFromNameValidation:
    @patch("core.config.models.load_config")
    def test_known_anima_accepted(self, mock_load: MagicMock, tmp_path: Path):
        cfg = MagicMock()
        cfg.animas = {"sakura": MagicMock(), "mio": MagicMock()}
        mock_load.return_value = cfg

        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        (shared_dir / "channels").mkdir()
        (shared_dir / "channels" / "general.jsonl").touch()

        from core.messaging.messenger import Messenger

        m = Messenger(shared_dir, "sakura")
        m.post_channel("general", "Hello", from_name="sakura")

        lines = (shared_dir / "channels" / "general.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["from"] == "sakura"

    @patch("core.config.models.load_config")
    def test_human_accepted(self, mock_load: MagicMock, tmp_path: Path):
        cfg = MagicMock()
        cfg.animas = {"sakura": MagicMock()}
        mock_load.return_value = cfg

        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        (shared_dir / "channels").mkdir()
        (shared_dir / "channels" / "general.jsonl").touch()

        from core.messaging.messenger import Messenger

        m = Messenger(shared_dir, "sakura")
        m.post_channel("general", "From human", from_name="human")

        lines = (shared_dir / "channels" / "general.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1

    @patch("core.config.models.load_config")
    def test_unknown_from_name_rejected(self, mock_load: MagicMock, tmp_path: Path):
        cfg = MagicMock()
        cfg.animas = {"sakura": MagicMock()}
        mock_load.return_value = cfg

        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        (shared_dir / "channels").mkdir()
        (shared_dir / "channels" / "general.jsonl").touch()

        from core.messaging.messenger import Messenger

        m = Messenger(shared_dir, "sakura")
        with pytest.raises(ValueError):
            m.post_channel("general", "Spoofed post", from_name="hacker")

        content = (shared_dir / "channels" / "general.jsonl").read_text().strip()
        assert content == "", "Unknown from_name should result in rejected post"

    @patch("core.config.models.load_config")
    def test_unknown_from_name_warning_logged(
        self,
        mock_load: MagicMock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        import logging

        cfg = MagicMock()
        cfg.animas = {"sakura": MagicMock()}
        mock_load.return_value = cfg

        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        (shared_dir / "channels").mkdir()
        (shared_dir / "channels" / "general.jsonl").touch()

        from core.messaging.messenger import Messenger

        m = Messenger(shared_dir, "sakura")
        with caplog.at_level(logging.WARNING, logger="animaworks.messenger"), pytest.raises(ValueError):
            m.post_channel("general", "Spoofed", from_name="hacker")

        assert any("unknown from_name" in r.message for r in caplog.records)

    @patch("core.config.models.load_config", side_effect=RuntimeError("config unavailable"))
    def test_config_load_failure_rejects_post(self, mock_load: MagicMock, tmp_path: Path, caplog):
        import logging

        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        (shared_dir / "channels").mkdir()
        channel_file = shared_dir / "channels" / "general.jsonl"
        channel_file.touch()

        from core.messaging.messenger import Messenger

        messenger = Messenger(shared_dir, "sakura")
        with caplog.at_level(logging.WARNING, logger="animaworks.messenger"), pytest.raises(ValueError):
            messenger.post_channel("general", "Should not be posted", from_name="sakura")

        assert channel_file.read_text(encoding="utf-8") == ""
        assert any("Failed to validate channel from_name" in record.message for record in caplog.records)
        assert mock_load.called
