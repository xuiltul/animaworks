"""phase3-fixed: execution routing is independent of process model."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.runtime.runner import AnimaRunner
from core.runtime.scheduler_manager import SchedulerManager
from core.runtime.streaming_handler import StreamingIPCHandler
from core.runtime.task_runner_supervisor import TaskRunnerSupervisor
from core.schemas import CronTask
from core.tasks.pending_executor import PendingTaskExecutor


def _mk_anima_dir(tmp_path: Path, *, mode: str | None = None) -> Path:
    anima_dir = tmp_path / "animas" / "sakura"
    anima_dir.mkdir(parents=True)
    if mode is not None:
        (anima_dir / "status.json").write_text(
            json.dumps({"process_model": mode}),
            encoding="utf-8",
        )
    return anima_dir


def _mk_manager(tmp_path: Path, *, mode: str | None = None) -> tuple[SchedulerManager, MagicMock]:
    anima_dir = _mk_anima_dir(tmp_path, mode=mode)
    (tmp_path / "shared").mkdir()
    anima = MagicMock()
    anima.shared_dir = tmp_path / "shared"
    anima.run_heartbeat = AsyncMock()
    anima.run_cron_task = AsyncMock()
    anima.run_cron_command = AsyncMock()
    emit = MagicMock()
    manager = SchedulerManager(anima, "sakura", anima_dir, emit)
    return manager, anima


@pytest.mark.parametrize("mode", [None, "legacy", "phase2", "phase3"])
def test_scheduler_always_builds_supervisor_with_root_memory(tmp_path: Path, mode: str | None) -> None:
    manager, _ = _mk_manager(tmp_path, mode=mode)
    assert manager._task_runner_supervisor is not None
    assert isinstance(manager._task_runner_supervisor, TaskRunnerSupervisor)
    assert manager._task_runner_supervisor._memory_service is not None


@pytest.mark.parametrize("mode", [None, "legacy"])
def test_scheduler_without_process_model_or_legacy_still_supervisor(tmp_path: Path, mode: str | None) -> None:
    manager, _ = _mk_manager(tmp_path, mode=mode)
    assert manager._task_runner_supervisor is not None
    assert manager._task_runner_supervisor._memory_service is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [None, "legacy"])
async def test_heartbeat_tick_delegates_even_with_legacy_status(tmp_path: Path, mode: str | None) -> None:
    manager, anima = _mk_manager(tmp_path, mode=mode)
    manager._task_runner_supervisor.run_heartbeat = AsyncMock(
        return_value={
            "task_type": "heartbeat",
            "result": {"action": "completed", "summary": "child"},
            "success": True,
        }
    )

    await manager.heartbeat_tick()

    anima.run_heartbeat.assert_not_awaited()
    manager._task_runner_supervisor.run_heartbeat.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_llm_cron_uses_runner_but_shell_command_runs_in_root(tmp_path: Path) -> None:
    manager, anima = _mk_manager(tmp_path)
    manager._task_runner_supervisor.run_cron = AsyncMock(
        return_value={"result": {"action": "completed", "summary": "ok"}, "success": True}
    )
    manager._task_runner_supervisor.run_cron_followup = AsyncMock()
    anima.run_cron_command = AsyncMock(
        return_value={"task": "cmd", "exit_code": 0, "stdout": "", "stderr": "", "duration_ms": 1}
    )

    llm_task = CronTask(name="llm", schedule="0 9 * * *", description="d", type="llm")
    cmd_task = CronTask(
        name="cmd",
        schedule="0 9 * * *",
        type="command",
        command="echo hi",
        trigger_heartbeat=False,
    )
    await manager._run_cron_task(llm_task)
    await manager._run_cron_task(cmd_task)

    anima.run_cron_task.assert_not_awaited()
    anima.run_cron_command.assert_awaited_once()
    manager._task_runner_supervisor.run_cron.assert_awaited_once_with(llm_task)
    manager._task_runner_supervisor.run_cron_followup.assert_not_awaited()


def test_runner_has_no_run_cron_task_handler(tmp_path: Path) -> None:
    runner = AnimaRunner("sakura", tmp_path / "a.sock", tmp_path / "animas", tmp_path / "shared")
    assert runner._get_handler("run_cron_task") is None
    assert not hasattr(runner, "_handle_run_cron_task")


@pytest.mark.asyncio
async def test_stream_handler_routes_to_supervisor_when_present(tmp_path: Path) -> None:
    class _Sup:
        async def run_chat_stream(self, payload):
            yield {"stream": True, "chunk": "chunk"}
            yield {"done": True, "result": {"response": "child"}}

    handler = StreamingIPCHandler(
        MagicMock(),
        "sakura",
        tmp_path,
        task_runner_supervisor=_Sup(),
    )
    responses = [
        r
        async for r in handler.handle_stream(
            type("Req", (), {"id": "1", "method": "process_message", "params": {"message": "hi"}})()
        )
    ]
    assert responses[0].chunk == "chunk"
    assert responses[-1].done is True
    assert responses[-1].result == {"response": "child"}


@pytest.mark.asyncio
async def test_stream_handler_direct_path_when_no_supervisor(tmp_path: Path) -> None:
    anima = MagicMock()

    async def _stream(*args, **kwargs):
        yield {"type": "cycle_done", "cycle_result": {"summary": "direct"}}

    anima.process_message_stream = _stream
    handler = StreamingIPCHandler(anima, "sakura", tmp_path, task_runner_supervisor=None)
    responses = [
        r
        async for r in handler.handle_stream(
            type("Req", (), {"id": "1", "method": "process_message", "params": {"message": "hi"}})()
        )
    ]
    assert responses[-1].result["response"] == "direct"


@pytest.mark.asyncio
async def test_pending_executor_without_supervisor_uses_run_llm_task_directly(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "sakura"
    anima_dir.mkdir(parents=True)
    anima = MagicMock()
    anima.shared_dir = tmp_path / "shared"
    anima._background_worker_pool_size = 1
    anima._background_lock = asyncio.Lock()
    anima._mark_busy_start = MagicMock()
    anima._clear_busy_status_sidecar_if_idle = MagicMock()
    anima._status_slots = {"background": "idle"}
    anima._task_slots = {"background": ""}
    anima._keepalive_while_busy = None
    executor = PendingTaskExecutor(
        anima,
        "sakura",
        anima_dir,
        asyncio.Event(),
        task_runner_supervisor=None,
    )
    assert executor._task_isolated is False
    assert executor._background_isolated is False
    assert executor._task_runner_supervisor is None

    with (
        patch.object(executor, "_run_llm_task", new=AsyncMock(return_value="done")) as run_llm,
        patch.object(executor, "_sync_task_queue"),
    ):
        await executor._execute_llm_task({"task_id": "t", "title": "t", "description": "w", "task_type": "llm"})

    run_llm.assert_awaited_once()
    assert executor._task_runner_supervisor is None
