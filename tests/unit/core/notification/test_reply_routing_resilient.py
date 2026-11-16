from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for sandbox-resilient notification-mapping persistence.

Sandboxed ``animaworks-tool call_human`` runs cannot write
``{data_dir}/run/`` (EROFS); the resilient saver must fall back to the
server internal API so Slack thread replies keep routing back.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def routing_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(
        "core.notification.reply_routing.get_data_dir",
        lambda: tmp_path,
    )
    return tmp_path


class TestSaveReturnsBool:
    def test_success_returns_true(self, routing_dir: Path) -> None:
        from core.notification.reply_routing import save_notification_mapping

        assert save_notification_mapping("1.0", "C1", "sakura") is True
        data = json.loads((routing_dir / "run" / "notification_map.json").read_text())
        assert data["1.0"]["anima"] == "sakura"

    def test_oserror_returns_false(self, routing_dir: Path) -> None:
        from core.notification.reply_routing import save_notification_mapping

        with patch(
            "pathlib.Path.open",
            side_effect=OSError(30, "Read-only file system"),
        ):
            assert save_notification_mapping("1.0", "C1", "sakura") is False


class TestResilientFallback:
    def test_direct_write_success_skips_api(self, routing_dir: Path) -> None:
        from core.notification.reply_routing import save_notification_mapping_resilient

        with patch("httpx.post") as mock_post:
            assert save_notification_mapping_resilient("1.0", "C1", "sakura") is True
            mock_post.assert_not_called()

    def test_falls_back_to_api_on_write_failure(self, routing_dir: Path) -> None:
        from core.notification import reply_routing

        resp = MagicMock()
        resp.json.return_value = {"ok": True}
        resp.raise_for_status.return_value = None
        with (
            patch.object(reply_routing, "save_notification_mapping", return_value=False),
            patch("core.host_api.host_api.post", return_value=resp) as mock_post,
        ):
            assert (
                reply_routing.save_notification_mapping_resilient(
                    "2.0",
                    "C1",
                    "sakura",
                    notification_text="subject\nbody",
                    callback_id="cb1",
                    notification_key="key-1",
                )
                is True
            )
        path = mock_post.call_args.args[0]
        assert path == "/api/internal/notification-mapping"
        payload = mock_post.call_args.kwargs["json"]
        assert payload["ts"] == "2.0"
        assert payload["anima_name"] == "sakura"
        assert payload["callback_id"] == "cb1"
        assert payload["notification_key"] == "key-1"

    def test_api_failure_returns_false(self, routing_dir: Path) -> None:
        from core.notification import reply_routing

        with (
            patch.object(reply_routing, "save_notification_mapping", return_value=False),
            patch("core.host_api.host_api.post", side_effect=ConnectionError("refused")),
        ):
            assert reply_routing.save_notification_mapping_resilient("3.0", "C1", "mei") is False

    def test_server_url_env_override(self, routing_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from core.notification import reply_routing

        monkeypatch.setenv("ANIMAWORKS_SERVER_URL", "http://127.0.0.1:9999/")
        resp = MagicMock()
        resp.json.return_value = {"ok": True}
        resp.raise_for_status.return_value = None
        with (
            patch.object(reply_routing, "save_notification_mapping", return_value=False),
            patch("core.host_api.host_api.post", return_value=resp) as mock_post,
        ):
            reply_routing.save_notification_mapping_resilient("4.0", "C1", "aoi")
        assert mock_post.call_args.args[0] == "/api/internal/notification-mapping"


class TestInternalEndpoint:
    def test_notification_mapping_endpoint(self, routing_dir: Path) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from server.routes.internal import create_internal_router

        app = FastAPI()
        app.include_router(create_internal_router(), prefix="/api")
        client = TestClient(app)
        resp = client.post(
            "/api/internal/notification-mapping",
            json={
                "ts": "9.9",
                "channel": "C9",
                "anima_name": "ritsu",
                "notification_text": "hello",
                "notification_key": "key-9",
            },
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        data = json.loads((routing_dir / "run" / "notification_map.json").read_text())
        assert data["9.9"]["anima"] == "ritsu"
        assert data["9.9"]["notification_key"] == "key-9"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
