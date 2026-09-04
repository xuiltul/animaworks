from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from core.memory.task_queue import TaskQueueManager
from core.taskboard.store import TaskBoardStore

pytestmark = pytest.mark.e2e


def _create_app(tmp_path: Path, anima_names: list[str]):
    animas_dir = tmp_path / "animas"
    shared_dir = tmp_path / "shared"
    for name in anima_names:
        anima_dir = animas_dir / name
        (anima_dir / "state").mkdir(parents=True, exist_ok=True)
        (anima_dir / "identity.md").write_text(f"# {name}\n", encoding="utf-8")
    shared_dir.mkdir(parents=True, exist_ok=True)

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


async def test_taskboard_api_lists_patches_and_summarizes_through_full_app(tmp_path: Path) -> None:
    app = _create_app(tmp_path, ["alice", "bob"])
    alice_queue = TaskQueueManager(app.state.animas_dir / "alice")
    bob_queue = TaskQueueManager(app.state.animas_dir / "bob")
    alice_task = alice_queue.add_task(
        source="human",
        original_instruction="draft launch checklist",
        assignee="alice",
        summary="draft launch checklist",
        task_id="task-launch",
    )
    bob_task = bob_queue.add_task(
        source="human",
        original_instruction="review partner response",
        assignee="bob",
        summary="review partner response",
        task_id="task-review",
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        list_resp = await client.get("/api/task-board", params={"assignee": "alice"})
        patch_resp = await client.patch(
            f"/api/task-board/bob/{bob_task.task_id}",
            json={
                "visibility": "snoozed",
                "snoozed_until": "2026-05-15T10:00:00+09:00",
                "actor": "planner",
            },
        )
        summary_resp = await client.get("/api/task-board/summary")
        legacy_resp = await client.get("/api/tasks/summary")

    assert list_resp.status_code == 200
    assert [task["task_id"] for task in list_resp.json()["tasks"]] == [alice_task.task_id]

    assert patch_resp.status_code == 200
    assert patch_resp.json()["task"]["visibility"] == "snoozed"
    assert bob_queue.get_task_by_id(bob_task.task_id).status == "pending"

    summary = summary_resp.json()
    assert summary["pending"] == 1
    assert summary["blocked"] == 0
    assert summary["snoozed"] == 1
    assert summary["total_active"] == 1
    assert legacy_resp.json() == {"pending": 1, "in_progress": 0, "total_active": 1}

    metadata = TaskBoardStore(app.state.shared_dir / "taskboard.sqlite3").get_metadata("bob", bob_task.task_id)
    assert metadata is not None
    assert metadata.snoozed_until == "2026-05-15T10:00:00+09:00"
