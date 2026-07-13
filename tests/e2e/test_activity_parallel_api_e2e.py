"""E2E coverage for parallel activity timeline backend data."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from httpx import ASGITransport, AsyncClient

from core.memory.task_queue import TaskQueueManager


def _create_app(tmp_path: Path):
    animas_dir = tmp_path / "animas"
    shared_dir = tmp_path / "shared"
    (animas_dir / "alice").mkdir(parents=True)
    shared_dir.mkdir(parents=True)

    with (
        patch("server.app.ProcessSupervisor") as supervisor_cls,
        patch("server.app.load_config") as load_config,
        patch("server.app.WebSocketManager") as ws_manager_cls,
        patch("server.app.load_auth") as load_auth,
    ):
        config = MagicMock()
        config.setup_complete = True
        load_config.return_value = config
        auth = MagicMock()
        auth.auth_mode = "local_trust"
        load_auth.return_value = auth
        supervisor = MagicMock()
        supervisor.get_all_status.return_value = {}
        supervisor.get_process_status.return_value = {"status": "stopped", "pid": None}
        supervisor.is_scheduler_running.return_value = False
        supervisor.scheduler = None
        supervisor_cls.return_value = supervisor
        ws_manager = MagicMock()
        ws_manager.active_connections = []
        ws_manager_cls.return_value = ws_manager

        from server.app import create_app

        app = create_app(animas_dir, shared_dir)

    import server.app as server_app

    runtime_auth = MagicMock()
    runtime_auth.auth_mode = "local_trust"
    server_app.load_auth = lambda: runtime_auth
    app.state.anima_names = ["alice"]
    return app


async def test_parallel_activity_data_through_application_router(tmp_path: Path) -> None:
    app = _create_app(tmp_path)
    anima_dir = tmp_path / "animas" / "alice"
    TaskQueueManager(anima_dir).add_task(
        source="anima",
        original_instruction="Synthetic E2E instruction",
        assignee="alice",
        summary="Synthetic E2E task",
        task_id="e2e-task",
        status="in_progress",
        meta={"executor": "taskexec"},
    )
    sidecar_dir = tmp_path / "run" / "animas"
    sidecar_dir.mkdir(parents=True)
    sidecar_dir.joinpath("alice.busy.json").write_text(
        json.dumps(
            {
                "is_busy": True,
                "busy_since": "2026-07-13T11:00:00+09:00",
                "lanes": ["background-worker:2:e2e-task"],
            }
        ),
        encoding="utf-8",
    )

    now = datetime.now(UTC).isoformat()
    activity_dir = anima_dir / "activity_log"
    activity_dir.mkdir()
    activity_dir.joinpath(f"{now[:10]}.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": now,
                        "type": "tool_use",
                        "summary": "E2E context event",
                        "ctx": "task:e2e-task",
                    }
                ),
                json.dumps(
                    {
                        "ts": now,
                        "type": "response_sent",
                        "summary": "E2E legacy event",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        running = await client.get("/api/activity/running-tasks?anima=alice")
        recent = await client.get("/api/activity/recent?hours=1&anima=alice")

    assert running.status_code == 200
    assert running.json() == {
        "animas": [
            {
                "name": "alice",
                "tasks": [
                    {
                        "slot_id": 2,
                        "task_id": "e2e-task",
                        "title": "Synthetic E2E task",
                        "started_at": "2026-07-13T11:00:00+09:00",
                    }
                ],
            }
        ],
        "total": 1,
    }
    assert recent.status_code == 200
    events = recent.json()["events"]
    assert next(e for e in events if e["summary"] == "E2E context event")["ctx"] == "task:e2e-task"
    assert next(e for e in events if e["summary"] == "E2E legacy event").get("ctx", "") == ""
