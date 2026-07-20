"""E2E coverage for parallel activity timeline backend data."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from httpx import ASGITransport, AsyncClient

from core.memory.task_queue import TaskQueueManager
from core.time_utils import now_local


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

    # Match ActivityLogger partitioning / hours cutoff (app-local TZ) and use
    # two genuinely overlapping task lifecycles. Partition by each event date
    # so this remains valid when the test happens to run across midnight.
    now = now_local()
    activity_events = [
        {
            "ts": (now - timedelta(seconds=40)).isoformat(),
            "type": "task_exec_start",
            "summary": "Parallel task A started",
            "ctx": "task:parallel-a",
            "meta": {"task_id": "parallel-a", "title": "Parallel task A"},
        },
        {
            "ts": (now - timedelta(seconds=30)).isoformat(),
            "type": "task_exec_start",
            "summary": "Parallel task B started",
            "ctx": "task:parallel-b",
            "meta": {"task_id": "parallel-b", "title": "Parallel task B"},
        },
        {
            "ts": (now - timedelta(seconds=20)).isoformat(),
            "type": "tool_use",
            "summary": "Parallel task A tool",
            "ctx": "task:parallel-a",
        },
        {
            "ts": (now - timedelta(seconds=15)).isoformat(),
            "type": "tool_use",
            "summary": "Parallel task B tool",
            "ctx": "task:parallel-b",
        },
        {
            "ts": (now - timedelta(seconds=10)).isoformat(),
            "type": "task_exec_end",
            "summary": "Parallel task B ended",
            "ctx": "task:parallel-b",
            "meta": {"task_id": "parallel-b", "title": "Parallel task B"},
        },
        {
            "ts": (now - timedelta(seconds=5)).isoformat(),
            "type": "task_exec_end",
            "summary": "Parallel task A ended",
            "ctx": "task:parallel-a",
            "meta": {"task_id": "parallel-a", "title": "Parallel task A"},
        },
        {
            "ts": now.isoformat(),
            "type": "response_sent",
            "summary": "E2E legacy event",
        },
    ]
    activity_dir = anima_dir / "activity_log"
    activity_dir.mkdir()
    events_by_date: dict[str, list[dict[str, object]]] = {}
    for event in activity_events:
        events_by_date.setdefault(str(event["ts"])[:10], []).append(event)
    for date, date_events in events_by_date.items():
        activity_dir.joinpath(f"{date}.jsonl").write_text(
            "\n".join(json.dumps(event) for event in date_events) + "\n",
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
    parallel_events = [e for e in events if e.get("ctx", "").startswith("task:parallel-")]
    assert len(parallel_events) == 6
    assert {e["ctx"] for e in parallel_events} == {"task:parallel-a", "task:parallel-b"}
    for task_ctx in ("task:parallel-a", "task:parallel-b"):
        assert {e["type"] for e in parallel_events if e["ctx"] == task_ctx} == {
            "task_exec_start",
            "tool_use",
            "task_exec_end",
        }
    starts = {e["ctx"]: e for e in parallel_events if e["type"] == "task_exec_start"}
    assert starts["task:parallel-a"]["meta"]["title"] == "Parallel task A"
    assert starts["task:parallel-b"]["meta"]["title"] == "Parallel task B"
    assert next(e for e in events if e["summary"] == "E2E legacy event").get("ctx", "") == ""
