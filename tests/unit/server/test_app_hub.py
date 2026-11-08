from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from core.config import invalidate_cache
from core.hub.members import HubMember, save_member, token_digest


def _enable_hub(data_dir: Path, *, enabled: bool = True) -> None:
    config_path = data_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["setup_complete"] = True
    config["hub"] = {
        "enabled": enabled,
        "timeline_enabled": True,
        "company": "a",
        "board": "hub",
        "public_hosts": ["hub.example.com"],
        "max_posts_per_hour": 20,
        "max_round_trips": 4,
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    invalidate_cache()


def _create_app(data_dir: Path):
    from server.app import create_app

    with patch("server.app.ProcessSupervisor") as supervisor_cls, patch("server.app.WebSocketManager") as ws_cls:
        supervisor_cls.return_value = MagicMock()
        ws_cls.return_value = MagicMock()
        return create_app(data_dir / "animas", data_dir / "shared")


def test_hub_mcp_is_mounted_and_session_manager_is_available(data_dir_at_tmp_path: Path) -> None:
    _enable_hub(data_dir_at_tmp_path)
    app = _create_app(data_dir_at_tmp_path)

    assert app.state.hub_mcp_session_manager is not None
    mounted = [route for route in app.routes if getattr(route, "name", None) == "hub_mcp"]
    assert mounted and mounted[0].path == "/hub"


def test_hub_endpoint_is_404_when_disabled(data_dir_at_tmp_path: Path) -> None:
    _enable_hub(data_dir_at_tmp_path, enabled=False)
    app = _create_app(data_dir_at_tmp_path)

    response = TestClient(app).post("/hub/mcp")
    assert response.status_code == 404
    assert response.json() == {"error": "not_found"}


def test_hub_endpoint_requires_bearer_token(data_dir_at_tmp_path: Path) -> None:
    _enable_hub(data_dir_at_tmp_path)
    save_member(
        data_dir_at_tmp_path / "shared",
        HubMember(
            id="grok",
            display_name="Grok",
            company="a",
            token_sha256=token_digest("token"),
            message_route={"type": "poll"},
        ),
    )
    app = _create_app(data_dir_at_tmp_path)
    client = TestClient(app, base_url="http://localhost:8000")

    assert client.post("/hub/mcp").status_code == 401
    assert client.post("/hub/mcp", headers={"Authorization": "Bearer wrong"}).status_code == 401

    save_member(
        data_dir_at_tmp_path / "shared",
        HubMember(
            id="disabled",
            display_name="Disabled",
            company="a",
            enabled=False,
            token_sha256=token_digest("disabled-token"),
            message_route={"type": "poll"},
        ),
    )
    assert client.post("/hub/mcp", headers={"Authorization": "Bearer disabled-token"}).status_code == 401


def test_lifespan_starts_and_stops_hub_silence_task(data_dir_at_tmp_path: Path) -> None:
    _enable_hub(data_dir_at_tmp_path)
    config_path = data_dir_at_tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["setup_complete"] = False
    config_path.write_text(json.dumps(config), encoding="utf-8")
    invalidate_cache()
    app = _create_app(data_dir_at_tmp_path)

    with TestClient(app, base_url="http://localhost:8000"):
        task = app.state.hub_silence_task
        assert task is not None
        assert not task.done()

    assert app.state.hub_silence_task.done()
