"""Tests for activity execution context and running TaskExec worker APIs."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from core.memory.task_queue import TaskQueueManager
from server.routes.system import create_system_router


def _app(tmp_path: Path, names: list[str]) -> FastAPI:
    animas_dir = tmp_path / "animas"
    shared_dir = tmp_path / "shared"
    animas_dir.mkdir(parents=True, exist_ok=True)
    shared_dir.mkdir(parents=True, exist_ok=True)
    app = FastAPI()
    app.state.animas_dir = animas_dir
    app.state.shared_dir = shared_dir
    app.state.anima_names = names
    app.state.supervisor = MagicMock()
    app.state.stream_registry = MagicMock()
    app.state.ws_manager = MagicMock()
    app.include_router(create_system_router(), prefix="/api")
    return app


def _add_task(anima_dir: Path, task_id: str, title: str) -> None:
    TaskQueueManager(anima_dir).add_task(
        source="anima",
        original_instruction=f"Synthetic instruction for {task_id}",
        assignee=anima_dir.name,
        summary=title,
        task_id=task_id,
        status="in_progress",
        meta={"executor": "taskexec"},
    )


def _write_sidecar(tmp_path: Path, name: str, payload: dict) -> None:
    sidecar_dir = tmp_path / "run" / "animas"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    (sidecar_dir / f"{name}.busy.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _write_activity(anima_dir: Path, entries: list[dict]) -> None:
    log_dir = anima_dir / "activity_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        path = log_dir / f"{entry['ts'][:10]}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")


class TestRunningActivityTasks:
    async def test_joins_worker_slots_with_task_titles(self, tmp_path: Path) -> None:
        app = _app(tmp_path, ["alice", "bob"])
        alice_dir = app.state.animas_dir / "alice"
        alice_dir.mkdir()
        (app.state.animas_dir / "bob").mkdir()
        _add_task(alice_dir, "task-a", "Prepare synthetic report")
        _add_task(alice_dir, "task-b", "Check synthetic output")
        # Queue completion can race slightly ahead of worker sidecar release.
        TaskQueueManager(alice_dir).update_status("task-b", "done")
        started_at = "2026-07-13T10:15:30+09:00"
        _write_sidecar(
            tmp_path,
            "alice",
            {
                "is_busy": True,
                "busy_since": started_at,
                "lanes": [
                    "background-worker:3:task-b",
                    "conversation:chat:user",
                    "background-worker:1:task-a",
                ],
            },
        )

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/activity/running-tasks")

        assert response.status_code == 200
        assert response.json() == {
            "animas": [
                {
                    "name": "alice",
                    "tasks": [
                        {
                            "slot_id": 1,
                            "task_id": "task-a",
                            "title": "Prepare synthetic report",
                            "started_at": started_at,
                        },
                        {
                            "slot_id": 3,
                            "task_id": "task-b",
                            "title": "Check synthetic output",
                            "started_at": started_at,
                        },
                    ],
                },
                {"name": "bob", "tasks": []},
            ],
            "total": 2,
        }

    async def test_anima_filter_and_malformed_sidecar_are_safe(self, tmp_path: Path) -> None:
        app = _app(tmp_path, ["alice", "bob"])
        (app.state.animas_dir / "alice").mkdir()
        (app.state.animas_dir / "bob").mkdir()
        sidecar_dir = tmp_path / "run" / "animas"
        sidecar_dir.mkdir(parents=True)
        (sidecar_dir / "alice.busy.json").write_text("{not-json", encoding="utf-8")

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/activity/running-tasks?anima=alice")

        assert response.status_code == 200
        assert response.json() == {
            "animas": [{"name": "alice", "tasks": []}],
            "total": 0,
        }


class TestRecentActivityContextCompatibility:
    async def test_flat_and_grouped_routes_preserve_mixed_context(self, tmp_path: Path) -> None:
        app = _app(tmp_path, ["alice"])
        anima_dir = app.state.animas_dir / "alice"
        anima_dir.mkdir()
        now = datetime.now(UTC)
        _write_activity(
            anima_dir,
            [
                {
                    "ts": (now - timedelta(seconds=1)).isoformat(),
                    "type": "response_sent",
                    "summary": "Context-free legacy event",
                    "content": "",
                },
                {
                    "ts": now.isoformat(),
                    "type": "tool_use",
                    "summary": "Context-aware event",
                    "content": "",
                    "tool": "send_message",
                    "ctx": "task:task-a",
                },
            ],
        )

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            flat = await client.get("/api/activity/recent?hours=1")
            grouped = await client.get("/api/activity/recent?hours=1&grouped=true")
            semantic = await client.get(
                "/api/activity/recent?hours=1&grouped=true&replay=true&semantic=true"
            )

        assert flat.status_code == 200
        events = flat.json()["events"]
        assert next(e for e in events if e["summary"] == "Context-aware event")["ctx"] == "task:task-a"
        legacy = next(e for e in events if e["summary"] == "Context-free legacy event")
        assert legacy.get("ctx", "") == ""

        assert grouped.status_code == 200
        grouped_events = [
            event
            for group in grouped.json()["groups"]
            for event in group["events"]
        ]
        assert next(e for e in grouped_events if e["summary"] == "Context-aware event")["ctx"] == "task:task-a"
        legacy = next(e for e in grouped_events if e["summary"] == "Context-free legacy event")
        assert legacy.get("ctx", "") == ""

        assert semantic.status_code == 200
        semantic_events = semantic.json()["events"]
        assert any(e.get("ctx") == "task:task-a" for e in semantic_events)
        assert any(not e.get("ctx") for e in semantic_events)
