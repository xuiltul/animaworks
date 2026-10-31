from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from core.tasks.queue import TaskQueueManager

pytestmark = pytest.mark.e2e


def _create_app(tmp_path: Path, anima_names: list[str]):
    os.environ["ANIMAWORKS_DATA_DIR"] = str(tmp_path)
    animas_dir = tmp_path / "animas"
    shared_dir = tmp_path / "shared"
    for name in anima_names:
        anima_dir = animas_dir / name
        (anima_dir / "state").mkdir(parents=True, exist_ok=True)
        (anima_dir / "IDENTITY.md").write_text(f"# {name}\n", encoding="utf-8")
    (shared_dir / "channels").mkdir(parents=True, exist_ok=True)

    with (
        patch("server.app.ProcessSupervisor") as mock_sup_cls,
        patch("server.app.load_config") as mock_cfg,
        patch("server.app.WebSocketManager") as mock_ws_cls,
        patch("server.app.load_auth") as mock_auth,
    ):
        cfg = MagicMock()
        cfg.setup_complete = True
        mock_cfg.return_value = cfg

        auth_cfg = MagicMock()
        auth_cfg.auth_mode = "local_trust"
        mock_auth.return_value = auth_cfg

        supervisor = MagicMock()
        supervisor.get_all_status.return_value = {}
        supervisor.get_process_status.return_value = {"status": "stopped", "pid": None}
        supervisor.is_scheduler_running.return_value = False
        supervisor.scheduler = None
        mock_sup_cls.return_value = supervisor

        ws_manager = MagicMock()
        ws_manager.active_connections = []
        mock_ws_cls.return_value = ws_manager

        from server.app import create_app

        app = create_app(animas_dir, shared_dir)

    import server.app as server_app

    local_auth = MagicMock()
    local_auth.auth_mode = "local_trust"
    server_app.load_auth = lambda: local_auth
    return app


async def test_taskboard_operations_do_not_modify_board_channel_jsonl(tmp_path: Path) -> None:
    app = _create_app(tmp_path, ["alice"])
    channel_path = app.state.shared_dir / "channels" / "general.jsonl"
    channel_entry = {
        "ts": "2026-05-14T09:00:00+09:00",
        "from": "alice",
        "text": "board conversation remains separate",
    }
    channel_path.write_text(json.dumps(channel_entry, ensure_ascii=False) + "\n", encoding="utf-8")
    before = channel_path.read_bytes()
    before_channel_files = sorted(path.name for path in (app.state.shared_dir / "channels").glob("*.jsonl"))

    queue = TaskQueueManager(app.state.animas_dir / "alice")
    task = queue.add_task(
        source="human",
        original_instruction="remove stale board-independent task",
        assignee="alice",
        summary="remove stale board-independent task",
        task_id="task-stale",
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        cancel_resp = await client.post(
            f"/api/task-board/alice/{task.task_id}/cancel",
            json={"reason": "deadline passed"},
        )
        channels_resp = await client.get("/api/channels")

    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["result"]["status"] == "cancelled"
    assert channel_path.read_bytes() == before
    assert sorted(path.name for path in (app.state.shared_dir / "channels").glob("*.jsonl")) == before_channel_files
    assert (app.state.shared_dir / "taskboard.sqlite3").exists()

    assert channels_resp.status_code == 200
    assert channels_resp.json()[0]["name"] == "general"
    assert channels_resp.json()[0]["message_count"] == 1
