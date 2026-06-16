from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.supervisor.pending_executor import PendingTaskExecutor


class _FakeAnima:
    def __init__(self) -> None:
        self._background_lock = asyncio.Lock()
        self._status_slots = {"background": "idle"}
        self._task_slots = {"background": ""}
        self._mark_busy_start = MagicMock()
        self._clear_busy_status_sidecar_if_idle = MagicMock()
        self.keepalive_started = asyncio.Event()
        self.keepalive_cancelled = asyncio.Event()

    async def _keepalive_while_busy(self) -> None:
        self.keepalive_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.keepalive_cancelled.set()


@pytest.mark.asyncio
async def test_llm_task_runs_keepalive_while_background_lock_is_held(tmp_path: Path) -> None:
    anima = _FakeAnima()
    executor = PendingTaskExecutor(
        anima=anima,  # type: ignore[arg-type]
        anima_name="alice",
        anima_dir=tmp_path,
        shutdown_event=asyncio.Event(),
    )
    executor._sync_task_queue = MagicMock()  # type: ignore[method-assign]
    executor._handle_goal_completion = AsyncMock()  # type: ignore[method-assign]

    async def _run_task(_task_desc):
        await anima.keepalive_started.wait()
        return "done"

    executor._run_llm_task = AsyncMock(side_effect=_run_task)  # type: ignore[method-assign]

    await executor._execute_llm_task({"task_id": "task-1", "title": "Task"})

    anima._mark_busy_start.assert_called_once()
    anima._clear_busy_status_sidecar_if_idle.assert_called_once()
    assert anima.keepalive_started.is_set()
    assert anima.keepalive_cancelled.is_set()
    assert anima._status_slots["background"] == "idle"
    assert anima._task_slots["background"] == ""
