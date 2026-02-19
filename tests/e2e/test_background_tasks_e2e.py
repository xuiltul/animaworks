# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for background task workflow.

Validates the full lifecycle of background task execution:
  - BackgroundTaskManager submit/complete/fail flow
  - ToolHandler dispatch to background execution path
  - REST API endpoints for listing/querying background tasks
  - WebSocket broadcast callback on task completion
  - IPC timeout configuration resolution
  - Old task cleanup
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.background import BackgroundTask, BackgroundTaskManager, TaskStatus
from core.tooling.handler import ToolHandler

pytestmark = pytest.mark.e2e


# ── Helpers ──────────────────────────────────────────────────


def _slow_handler(name: str, args: dict[str, Any]) -> str:
    """Mock tool handler that takes a short time to complete."""
    time.sleep(0.1)
    return f"result-from-{name}"


def _failing_handler(name: str, args: dict[str, Any]) -> str:
    """Mock tool handler that raises an exception."""
    raise RuntimeError("tool execution exploded")


def _write_task_json(
    bg_dir: Path,
    task_id: str,
    *,
    status: str = "completed",
    tool_name: str = "image_generation",
    anima_name: str = "test-anima",
    completed_at: float | None = None,
) -> Path:
    """Write a background task JSON file to disk for API endpoint tests."""
    bg_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "task_id": task_id,
        "anima_name": anima_name,
        "tool_name": tool_name,
        "tool_args": {"prompt": "test"},
        "status": status,
        "created_at": time.time() - 3600,
        "completed_at": completed_at or time.time(),
        "result": f"result-{task_id}" if status == "completed" else None,
        "error": "boom" if status == "failed" else None,
    }
    path = bg_dir / f"{task_id}.json"
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


# ── 1. BackgroundTaskManager lifecycle ───────────────────────


class TestBackgroundTaskManagerLifecycle:
    """Test submit -> RUNNING -> COMPLETED flow with disk persistence."""

    async def test_submit_transitions_to_completed(self, tmp_path: Path) -> None:
        """Submit a tool call with a mock handler; verify RUNNING -> COMPLETED."""
        anima_dir = tmp_path / "animas" / "bg-test"
        anima_dir.mkdir(parents=True)
        (anima_dir / "state").mkdir()

        callback_called = asyncio.Event()
        callback_task: list[BackgroundTask] = []

        async def on_complete(task: BackgroundTask) -> None:
            callback_task.append(task)
            callback_called.set()

        mgr = BackgroundTaskManager(
            anima_dir, anima_name="bg-test",
            eligible_tools={"slow_tool": 5},
        )
        mgr.on_complete = on_complete

        # Submit
        task_id = mgr.submit("slow_tool", {"key": "val"}, _slow_handler)

        # Task should be in-memory and RUNNING immediately
        task = mgr.get_task(task_id)
        assert task is not None
        assert task.status == TaskStatus.RUNNING

        # Wait for completion
        await asyncio.wait_for(callback_called.wait(), timeout=5.0)

        # Verify final state
        task = mgr.get_task(task_id)
        assert task is not None
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "result-from-slow_tool"
        assert task.completed_at is not None
        assert task.error is None

        # Verify disk persistence
        disk_path = anima_dir / "state" / "background_tasks" / f"{task_id}.json"
        assert disk_path.exists()
        disk_data = json.loads(disk_path.read_text(encoding="utf-8"))
        assert disk_data["status"] == "completed"
        assert disk_data["result"] == "result-from-slow_tool"

        # Verify on_complete callback was fired
        assert len(callback_task) == 1
        assert callback_task[0].task_id == task_id
        assert callback_task[0].status == TaskStatus.COMPLETED


# ── 2. BackgroundTaskManager failure path ────────────────────


class TestBackgroundTaskManagerFailurePath:
    """Test submit -> RUNNING -> FAILED when handler raises."""

    async def test_failing_handler_transitions_to_failed(
        self, tmp_path: Path,
    ) -> None:
        """Submit a task with a failing handler; verify FAILED status."""
        anima_dir = tmp_path / "animas" / "bg-fail"
        anima_dir.mkdir(parents=True)
        (anima_dir / "state").mkdir()

        callback_called = asyncio.Event()
        callback_task: list[BackgroundTask] = []

        async def on_complete(task: BackgroundTask) -> None:
            callback_task.append(task)
            callback_called.set()

        mgr = BackgroundTaskManager(
            anima_dir, anima_name="bg-fail",
            eligible_tools={"bad_tool": 5},
        )
        mgr.on_complete = on_complete

        task_id = mgr.submit("bad_tool", {}, _failing_handler)

        # Wait for completion (will fail internally)
        await asyncio.wait_for(callback_called.wait(), timeout=5.0)

        task = mgr.get_task(task_id)
        assert task is not None
        assert task.status == TaskStatus.FAILED
        assert task.error is not None
        assert "RuntimeError" in task.error
        assert "tool execution exploded" in task.error
        assert task.completed_at is not None

        # Verify disk persistence records the failure
        disk_path = anima_dir / "state" / "background_tasks" / f"{task_id}.json"
        assert disk_path.exists()
        disk_data = json.loads(disk_path.read_text(encoding="utf-8"))
        assert disk_data["status"] == "failed"
        assert "RuntimeError" in disk_data["error"]

        # on_complete should fire even on failure
        assert len(callback_task) == 1
        assert callback_task[0].status == TaskStatus.FAILED


# ── 3. ToolHandler background dispatch ───────────────────────


class TestToolHandlerBackgroundDispatch:
    """Test ToolHandler.handle() routes eligible tools to background."""

    async def test_eligible_tool_returns_background_json(
        self, tmp_path: Path,
    ) -> None:
        """handle() for an eligible tool should return JSON with status=background."""
        anima_dir = tmp_path / "animas" / "handler-test"
        anima_dir.mkdir(parents=True)
        (anima_dir / "state").mkdir()
        (anima_dir / "permissions.md").write_text("", encoding="utf-8")

        memory = MagicMock()
        memory.read_permissions.return_value = ""

        bg_mgr = BackgroundTaskManager(
            anima_dir, anima_name="handler-test",
            eligible_tools={"image_generation": 30},
        )

        handler = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
            background_manager=bg_mgr,
        )

        result = handler.handle("image_generation", {"prompt": "cat photo"})

        parsed = json.loads(result)
        assert parsed["status"] == "background"
        assert "task_id" in parsed
        assert len(parsed["task_id"]) == 12  # uuid hex[:12]
        assert "message" in parsed

        # Verify task was actually created in the manager
        task = bg_mgr.get_task(parsed["task_id"])
        assert task is not None
        assert task.tool_name == "image_generation"

    async def test_non_eligible_tool_not_dispatched_to_background(
        self, tmp_path: Path,
    ) -> None:
        """handle() for a non-eligible tool should NOT go to background."""
        anima_dir = tmp_path / "animas" / "handler-test2"
        anima_dir.mkdir(parents=True)
        (anima_dir / "state").mkdir()

        memory = MagicMock()
        memory.read_permissions.return_value = ""

        bg_mgr = BackgroundTaskManager(
            anima_dir, anima_name="handler-test2",
            eligible_tools={"image_generation": 30},
        )

        # Mock the external dispatcher to return a result
        mock_external = MagicMock()
        mock_external.dispatch.return_value = "direct-result"

        handler = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
            background_manager=bg_mgr,
        )
        handler._external = mock_external

        result = handler.handle("web_search", {"query": "hello"})

        # Should go through external dispatch, not background
        assert result == "direct-result"
        mock_external.dispatch.assert_called_once()


# ── 4. Background task API endpoint ──────────────────────────


class TestBackgroundTaskAPIEndpoint:
    """Test /animas/{name}/background-tasks REST endpoints."""

    async def test_list_background_tasks_endpoint(
        self, tmp_path: Path,
    ) -> None:
        """GET /animas/{name}/background-tasks returns task list from disk."""
        animas_dir = tmp_path / "animas"
        anima_dir = animas_dir / "api-anima"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("# API Anima", encoding="utf-8")

        bg_dir = anima_dir / "state" / "background_tasks"
        _write_task_json(bg_dir, "task001", status="completed")
        _write_task_json(bg_dir, "task002", status="failed")
        _write_task_json(bg_dir, "task003", status="running")

        app = _create_test_app(tmp_path, anima_names=["api-anima"])

        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/api-anima/background-tasks")

        assert resp.status_code == 200
        data = resp.json()
        assert "tasks" in data
        assert len(data["tasks"]) == 3

        task_ids = {t["task_id"] for t in data["tasks"]}
        assert task_ids == {"task001", "task002", "task003"}

    async def test_get_single_background_task(
        self, tmp_path: Path,
    ) -> None:
        """GET /animas/{name}/background-tasks/{task_id} returns single task."""
        animas_dir = tmp_path / "animas"
        anima_dir = animas_dir / "api-anima"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("# API Anima", encoding="utf-8")

        bg_dir = anima_dir / "state" / "background_tasks"
        _write_task_json(bg_dir, "single001", status="completed")

        app = _create_test_app(tmp_path, anima_names=["api-anima"])

        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/animas/api-anima/background-tasks/single001",
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == "single001"
        assert data["status"] == "completed"

    async def test_get_nonexistent_task_returns_404(
        self, tmp_path: Path,
    ) -> None:
        """GET for a missing task_id returns 404."""
        animas_dir = tmp_path / "animas"
        anima_dir = animas_dir / "api-anima"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("# API Anima", encoding="utf-8")

        app = _create_test_app(tmp_path, anima_names=["api-anima"])

        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/animas/api-anima/background-tasks/nonexistent",
            )

        assert resp.status_code == 404

    async def test_empty_background_tasks_dir(
        self, tmp_path: Path,
    ) -> None:
        """GET returns empty list when no background_tasks dir exists."""
        animas_dir = tmp_path / "animas"
        anima_dir = animas_dir / "api-anima"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("# API Anima", encoding="utf-8")

        app = _create_test_app(tmp_path, anima_names=["api-anima"])

        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/api-anima/background-tasks")

        assert resp.status_code == 200
        data = resp.json()
        assert data["tasks"] == []


def _create_test_app(
    tmp_path: Path,
    anima_names: list[str] | None = None,
) -> Any:
    """Build a FastAPI app with mocked externals for API endpoint testing."""
    animas_dir = tmp_path / "animas"
    animas_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = tmp_path / "shared"
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
        supervisor.get_process_status.return_value = {
            "status": "stopped",
            "pid": None,
        }
        supervisor.is_scheduler_running.return_value = False
        supervisor.scheduler = None
        mock_sup_cls.return_value = supervisor

        ws_manager = MagicMock()
        ws_manager.active_connections = []
        mock_ws_cls.return_value = ws_manager

        from server.app import create_app

        app = create_app(animas_dir, shared_dir)

    # Persist auth mock beyond the with-block for request-time middleware
    import server.app as _sa
    _auth = MagicMock()
    _auth.auth_mode = "local_trust"
    _sa.load_auth = lambda: _auth

    if anima_names is not None:
        app.state.anima_names = anima_names

    return app


# ── 5. WebSocket notification on background task completion ──


class TestBackgroundTaskWSNotification:
    """Test that task completion triggers WebSocket broadcast."""

    async def test_ws_broadcast_called_on_completion(
        self, tmp_path: Path,
    ) -> None:
        """When a background task completes, _ws_broadcast is invoked."""
        anima_dir = tmp_path / "animas" / "ws-test"
        anima_dir.mkdir(parents=True)
        (anima_dir / "state").mkdir()
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir(parents=True)

        broadcast_payloads: list[dict] = []

        async def mock_broadcast(payload: dict) -> None:
            broadcast_payloads.append(payload)

        # Build DigitalAnima with mocked dependencies
        with (
            patch("core.agent.AgentCore._init_tool_registry", return_value=[]),
            patch("core.agent.AgentCore._discover_personal_tools", return_value={}),
            patch("core.agent.AgentCore._check_sdk", return_value=False),
            patch("core.agent.AgentCore._build_human_notifier", return_value=None),
            patch(
                "core.agent.AgentCore._build_background_manager",
                return_value=BackgroundTaskManager(
                    anima_dir, anima_name="ws-test",
                    eligible_tools={"slow_tool": 5},
                ),
            ),
            patch("core.agent.AgentCore._create_executor"),
        ):
            from core.anima import DigitalAnima

            anima = DigitalAnima(anima_dir, shared_dir)
            anima.set_ws_broadcast(mock_broadcast)

        # The on_complete callback should be wired
        mgr = anima.agent.background_manager
        assert mgr is not None
        assert mgr.on_complete is not None

        # Submit a task
        callback_done = asyncio.Event()
        original_on_complete = mgr.on_complete

        async def tracked_on_complete(task: BackgroundTask) -> None:
            await original_on_complete(task)
            callback_done.set()

        mgr.on_complete = tracked_on_complete

        mgr.submit("slow_tool", {"key": "val"}, _slow_handler)
        await asyncio.wait_for(callback_done.wait(), timeout=5.0)

        # Verify WebSocket broadcast was called
        assert len(broadcast_payloads) == 1
        payload = broadcast_payloads[0]
        assert payload["type"] == "background_task.done"
        assert payload["data"]["anima"] == "ws-test"
        assert payload["data"]["tool_name"] == "slow_tool"
        assert payload["data"]["status"] == "completed"
        assert "result_summary" in payload["data"]

    async def test_ws_broadcast_called_on_failure(
        self, tmp_path: Path,
    ) -> None:
        """WebSocket broadcast fires even when the task fails."""
        anima_dir = tmp_path / "animas" / "ws-fail"
        anima_dir.mkdir(parents=True)
        (anima_dir / "state").mkdir()
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir(parents=True)

        broadcast_payloads: list[dict] = []

        async def mock_broadcast(payload: dict) -> None:
            broadcast_payloads.append(payload)

        with (
            patch("core.agent.AgentCore._init_tool_registry", return_value=[]),
            patch("core.agent.AgentCore._discover_personal_tools", return_value={}),
            patch("core.agent.AgentCore._check_sdk", return_value=False),
            patch("core.agent.AgentCore._build_human_notifier", return_value=None),
            patch(
                "core.agent.AgentCore._build_background_manager",
                return_value=BackgroundTaskManager(
                    anima_dir, anima_name="ws-fail",
                    eligible_tools={"bad_tool": 5},
                ),
            ),
            patch("core.agent.AgentCore._create_executor"),
        ):
            from core.anima import DigitalAnima

            anima = DigitalAnima(anima_dir, shared_dir)
            anima.set_ws_broadcast(mock_broadcast)

        mgr = anima.agent.background_manager
        assert mgr is not None

        callback_done = asyncio.Event()
        original_on_complete = mgr.on_complete

        async def tracked_on_complete(task: BackgroundTask) -> None:
            await original_on_complete(task)
            callback_done.set()

        mgr.on_complete = tracked_on_complete

        mgr.submit("bad_tool", {}, _failing_handler)
        await asyncio.wait_for(callback_done.wait(), timeout=5.0)

        assert len(broadcast_payloads) == 1
        payload = broadcast_payloads[0]
        assert payload["type"] == "background_task.done"
        assert payload["data"]["status"] == "failed"
        assert "failed" in payload["data"]["result_summary"]


# ── 6. IPC timeout configuration ────────────────────────────


class TestIPCTimeoutConfigurable:
    """Test IPCClient._resolve_ipc_timeout reads from config."""

    def test_returns_configured_value(self, data_dir: Path) -> None:
        """_resolve_ipc_timeout returns config.server.ipc_stream_timeout."""
        # Update config.json with a custom ipc_stream_timeout
        config_path = data_dir / "config.json"
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        config_data["server"] = {"ipc_stream_timeout": 600}
        config_path.write_text(
            json.dumps(config_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        from core.config import invalidate_cache
        invalidate_cache()

        from core.supervisor.ipc import IPCClient
        timeout = IPCClient._resolve_ipc_timeout()
        assert timeout == 600.0

    def test_returns_default_when_config_unavailable(self) -> None:
        """_resolve_ipc_timeout falls back to 60.0 when config loading fails."""
        from core.supervisor.ipc import IPCClient

        # load_config is imported inside _resolve_ipc_timeout, so patch at source
        with patch(
            "core.config.load_config",
            side_effect=RuntimeError("no config"),
        ):
            timeout = IPCClient._resolve_ipc_timeout()

        assert timeout == 60.0

    def test_returns_default_server_value(self, data_dir: Path) -> None:
        """Without explicit server config, the default ipc_stream_timeout is 60."""
        from core.config import invalidate_cache
        invalidate_cache()

        from core.supervisor.ipc import IPCClient
        timeout = IPCClient._resolve_ipc_timeout()
        assert timeout == 60.0


# ── 7. Background task cleanup ───────────────────────────────


class TestBackgroundTaskCleanup:
    """Test BackgroundTaskManager.cleanup_old_tasks removes expired tasks."""

    def test_removes_expired_completed_tasks(self, tmp_path: Path) -> None:
        """Completed tasks older than max_age_hours are removed from disk."""
        anima_dir = tmp_path / "animas" / "cleanup-test"
        anima_dir.mkdir(parents=True)
        (anima_dir / "state").mkdir()

        mgr = BackgroundTaskManager(
            anima_dir, anima_name="cleanup-test",
            eligible_tools={"some_tool": 5},
        )

        bg_dir = anima_dir / "state" / "background_tasks"

        # Write an old completed task (48 hours ago)
        _write_task_json(
            bg_dir, "old-task",
            status="completed",
            completed_at=time.time() - 48 * 3600,
        )

        # Write a recent completed task (1 hour ago)
        _write_task_json(
            bg_dir, "recent-task",
            status="completed",
            completed_at=time.time() - 3600,
        )

        # Write an old running task (should NOT be removed)
        _write_task_json(
            bg_dir, "running-task",
            status="running",
            completed_at=None,
        )

        assert (bg_dir / "old-task.json").exists()
        assert (bg_dir / "recent-task.json").exists()
        assert (bg_dir / "running-task.json").exists()

        removed = mgr.cleanup_old_tasks(max_age_hours=24)

        assert removed == 1
        assert not (bg_dir / "old-task.json").exists()
        assert (bg_dir / "recent-task.json").exists()
        assert (bg_dir / "running-task.json").exists()

    def test_removes_expired_failed_tasks(self, tmp_path: Path) -> None:
        """Failed tasks older than max_age_hours are also removed."""
        anima_dir = tmp_path / "animas" / "cleanup-fail"
        anima_dir.mkdir(parents=True)
        (anima_dir / "state").mkdir()

        mgr = BackgroundTaskManager(
            anima_dir, anima_name="cleanup-fail",
            eligible_tools={"some_tool": 5},
        )

        bg_dir = anima_dir / "state" / "background_tasks"

        _write_task_json(
            bg_dir, "old-fail",
            status="failed",
            completed_at=time.time() - 48 * 3600,
        )

        removed = mgr.cleanup_old_tasks(max_age_hours=24)
        assert removed == 1
        assert not (bg_dir / "old-fail.json").exists()

    def test_cleanup_returns_zero_when_nothing_expired(
        self, tmp_path: Path,
    ) -> None:
        """cleanup_old_tasks returns 0 when no tasks are expired."""
        anima_dir = tmp_path / "animas" / "cleanup-none"
        anima_dir.mkdir(parents=True)
        (anima_dir / "state").mkdir()

        mgr = BackgroundTaskManager(
            anima_dir, anima_name="cleanup-none",
            eligible_tools={"some_tool": 5},
        )

        bg_dir = anima_dir / "state" / "background_tasks"
        _write_task_json(
            bg_dir, "fresh-task",
            status="completed",
            completed_at=time.time() - 60,  # 1 minute ago
        )

        removed = mgr.cleanup_old_tasks(max_age_hours=24)
        assert removed == 0
        assert (bg_dir / "fresh-task.json").exists()

    def test_cleanup_evicts_from_in_memory_cache(
        self, tmp_path: Path,
    ) -> None:
        """cleanup_old_tasks also removes the task from the in-memory dict."""
        anima_dir = tmp_path / "animas" / "cleanup-mem"
        anima_dir.mkdir(parents=True)
        (anima_dir / "state").mkdir()

        mgr = BackgroundTaskManager(
            anima_dir, anima_name="cleanup-mem",
            eligible_tools={"some_tool": 5},
        )

        bg_dir = anima_dir / "state" / "background_tasks"
        task_id = "mem-task"
        _write_task_json(
            bg_dir, task_id,
            status="completed",
            completed_at=time.time() - 48 * 3600,
        )

        # Pre-populate in-memory cache
        mgr._tasks[task_id] = BackgroundTask(
            task_id=task_id,
            anima_name="cleanup-mem",
            tool_name="some_tool",
            tool_args={},
            status=TaskStatus.COMPLETED,
            completed_at=time.time() - 48 * 3600,
        )

        assert task_id in mgr._tasks
        mgr.cleanup_old_tasks(max_age_hours=24)
        assert task_id not in mgr._tasks
