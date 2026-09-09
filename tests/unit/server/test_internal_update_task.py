from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from core.memory.task_queue import TaskQueueManager


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_internal_update_task_persists_meta_and_status(tmp_path, monkeypatch) -> None:
    from fastapi import FastAPI

    from server.routes.internal import create_internal_router

    animas_dir = tmp_path / "animas"
    anima_dir = animas_dir / "rin"
    (anima_dir / "state").mkdir(parents=True)
    manager = TaskQueueManager(anima_dir)
    entry = manager.add_task(
        source="anima",
        original_instruction="work",
        assignee="rin",
        summary="work",
    )
    monkeypatch.setattr("core.paths.get_animas_dir", lambda: animas_dir)
    app = FastAPI()
    app.state.ws_manager = MagicMock()
    app.include_router(create_internal_router(), prefix="/api")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/internal/update-task",
            json={
                "anima_name": "rin",
                "task_id": entry.task_id,
                "status": "done",
                "summary": "verified",
                "meta": {"completed_by": "agent_declaration", "result_note": "verified"},
            },
        )

    assert response.status_code == 200
    updated = manager.get_task_by_id(entry.task_id)
    assert updated is not None
    assert updated.status == "done"
    assert updated.summary == "verified"
    assert updated.meta["completed_by"] == "agent_declaration"


@pytest.mark.anyio
async def test_host_updates_are_fenced_inside_executor_thread(tmp_path, monkeypatch):
    from fastapi import FastAPI

    from core.tasks_dispatch import publish_tasks
    from server.routes.internal import create_internal_router

    anima_dir = tmp_path / "animas" / "rin"
    anima_dir.mkdir(parents=True)
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("core.paths.get_animas_dir", lambda: anima_dir.parent)
    publish_tasks(anima_dir, [{"task_id": "owned", "title": "Work", "description": "Work safely"}])
    store = TaskQueueManager(anima_dir).store
    first = store.claim("rin", "owned", {"pid": 1})
    store.finish(first["_attempt_token"], status="pending", stop_kind="interrupted")
    publish_tasks(anima_dir, [{"task_id": "owned", "resume": True}])
    second = store.claim("rin", "owned", {"pid": 1})
    app = FastAPI()
    app.include_router(create_internal_router(), prefix="/api")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        body = {
            "anima_name": "rin",
            "task_id": "owned",
            "status": "done",
            "meta": {"result_note": "stale"},
            "attempt_identity": {"anima": "rin", "task_id": "owned", "token": first["_attempt_token"]},
        }
        stale = await client.post("/api/internal/update-task", json=body)
        assert stale.status_code == 500
        entry = store.read("rin")["owned"]
        assert entry.status == "in_progress"
        assert "result_note" not in entry.meta
        body["attempt_identity"]["token"] = second["_attempt_token"]
        body["meta"]["result_note"] = "verified"
        current = await client.post("/api/internal/update-task", json=body)
        assert current.status_code == 200
        assert store.read("rin")["owned"].meta["result_note"] == "verified"


@pytest.mark.anyio
async def test_host_submit_and_read_resume_saved_input(tmp_path, monkeypatch):
    from fastapi import FastAPI

    from server.routes.internal import create_internal_router

    anima_dir = tmp_path / "animas" / "rin"
    anima_dir.mkdir(parents=True)
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("core.paths.get_animas_dir", lambda: anima_dir.parent)
    app = FastAPI()
    app.include_router(create_internal_router(), prefix="/api")
    task = {"task_id": "owned", "title": "Work", "description": "Original instruction", "constraints": ["Keep scope"]}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/internal/submit-tasks", json={"anima_name": "rin", "tasks": [task]})
        assert response.status_code == 200
        store = TaskQueueManager(anima_dir).store
        attempt = store.claim("rin", "owned", {"pid": 1})
        store.finish(attempt["_attempt_token"], status="pending", stop_kind="interrupted")
        resumed = await client.post(
            "/api/internal/submit-tasks", json={"anima_name": "rin", "tasks": [{"task_id": "owned", "resume": True}]}
        )
        assert resumed.status_code == 200
        assert store.pending("rin") == [{**task, "task_type": "llm"}]
        snapshot = await client.get("/api/internal/tasks", params={"anima_name": "rin"})
        assert snapshot.status_code == 200
        assert snapshot.json()["tasks"][0]["original_instruction"] == "Original instruction"
        single = await client.get("/api/internal/tasks", params={"anima_name": "rin", "task_id": "owned"})
        assert single.status_code == 200
        assert [row["task_id"] for row in single.json()["tasks"]] == ["owned"]
        missing = await client.get("/api/internal/tasks", params={"anima_name": "rin", "task_id": "missing"})
        assert missing.status_code == 200
        assert missing.json()["tasks"] == []
