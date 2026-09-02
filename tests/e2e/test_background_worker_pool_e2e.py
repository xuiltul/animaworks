from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.config.models import invalidate_cache
from core.supervisor.pending_executor import PendingTaskExecutor
from tests.helpers.filesystem import create_anima_dir, create_test_data_dir


class _OverlapRecorder:
    """Coordinate three mocked streams and retain their execution intervals."""

    def __init__(self, expected: int) -> None:
        self.expected = expected
        self.started: dict[str, float] = {}
        self.ended: dict[str, float] = {}
        self.all_started = asyncio.Event()
        self.all_ended = asyncio.Event()

    async def stream(self, thread_id: str) -> AsyncGenerator[dict, None]:
        self.started[thread_id] = time.monotonic()
        if len(self.started) == self.expected:
            self.all_started.set()

        # This barrier makes accidental serialization fail deterministically:
        # every stream must start before any of them is allowed to finish.
        await asyncio.wait_for(self.all_started.wait(), timeout=2.0)
        await asyncio.sleep(0.05)
        yield {"type": "text_delta", "text": f"completed {thread_id}"}
        yield {
            "type": "cycle_done",
            "cycle_result": {
                "action": "complete",
                "summary": f"completed {thread_id}",
            },
        }
        self.ended[thread_id] = time.monotonic()
        if len(self.ended) == self.expected:
            self.all_ended.set()


class _MockExecutor:
    """No-service executor used by each isolated DigitalAnima worker slot."""

    def __init__(self, recorder: _OverlapRecorder) -> None:
        self._recorder = recorder

    def execute_streaming(self, thread_id: str) -> AsyncGenerator[dict, None]:
        return self._recorder.stream(thread_id)


class _MockAgentCore:
    """Small AgentCore stand-in that preserves the run_cycle_streaming boundary."""

    def __init__(self, recorder: _OverlapRecorder) -> None:
        self.background_manager = None
        self._tool_handler = MagicMock()
        self._executor = _MockExecutor(recorder)
        self._progress_callback = None
        self.set_interrupt_event = MagicMock()
        self.set_task_cwd = MagicMock()
        self.reset_reply_tracking = MagicMock()
        self.reset_read_paths = MagicMock()

    async def run_cycle_streaming(
        self,
        _prompt: str,
        *,
        trigger: str,
        thread_id: str,
        **_kwargs: object,
    ) -> AsyncGenerator[dict, None]:
        del trigger
        async for chunk in self._executor.execute_streaming(thread_id):
            yield chunk


@pytest.mark.e2e
async def test_three_single_pending_llm_tasks_overlap_across_worker_slots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """watcher_loop dispatches three standalone TaskExec files concurrently."""
    data_dir = create_test_data_dir(tmp_path)
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(data_dir))
    invalidate_cache()

    anima_dir = create_anima_dir(data_dir, "pool-e2e")
    status_path = anima_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["background_worker_pool_size"] = 3
    status_path.write_text(json.dumps(status), encoding="utf-8")

    workspaces = [tmp_path / f"workspace-{index}" for index in range(3)]
    for workspace in workspaces:
        workspace.mkdir()

    recorder = _OverlapRecorder(expected=3)
    created_agents: list[_MockAgentCore] = []

    def _create_mock_agent(*_args, **_kwargs) -> _MockAgentCore:
        agent = _MockAgentCore(recorder)
        created_agents.append(agent)
        return agent

    try:
        with patch("core.anima.AgentCore", side_effect=_create_mock_agent):
            from core.anima import DigitalAnima

            anima = DigitalAnima(anima_dir, data_dir / "shared")

        assert anima._background_worker_pool_size == 3
        assert len(anima._background_worker_slots) == 3
        assert len({id(slot.agent) for slot in anima._background_worker_slots}) == 3

        pending_dir = anima_dir / "state" / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        task_ids = [f"pool-e2e-{index}" for index in range(3)]
        for task_id, workspace in zip(task_ids, workspaces, strict=True):
            descriptor = {
                "task_type": "llm",
                "task_id": task_id,
                "title": f"Worker pool task {task_id}",
                "description": "Prove standalone TaskExec concurrency",
                "working_directory": str(workspace),
            }
            (pending_dir / f"{task_id}.json").write_text(
                json.dumps(descriptor),
                encoding="utf-8",
            )

        shutdown_event = asyncio.Event()
        executor = PendingTaskExecutor(
            anima=anima,
            anima_name=anima.name,
            anima_dir=anima_dir,
            shutdown_event=shutdown_event,
        )

        watcher = asyncio.create_task(executor.watcher_loop())
        await asyncio.wait_for(recorder.all_ended.wait(), timeout=5.0)
        # Let each detached coordinator task finish its processing-file
        # cleanup before asking watcher_loop to shut down.
        async with asyncio.timeout(2.0):
            while executor._active_dispatch_tasks:
                await asyncio.sleep(0.01)
        shutdown_event.set()
        executor.wake()
        await asyncio.wait_for(watcher, timeout=2.0)

        assert set(recorder.started) == set(task_ids)
        assert set(recorder.ended) == set(task_ids)
        for left_index, left_id in enumerate(task_ids):
            for right_id in task_ids[left_index + 1 :]:
                assert recorder.started[left_id] < recorder.ended[right_id]
                assert recorder.started[right_id] < recorder.ended[left_id]

        processing_dir = pending_dir / "processing"
        assert not list(pending_dir.glob("*.json"))
        assert not list(processing_dir.glob("*.json"))
        assert not list(processing_dir.glob("*.lease"))
        assert not executor._active_task_ids
        assert not anima._active_background_workers
        assert anima._background_worker_queue.qsize() == 3
        for slot in anima._background_worker_slots:
            slot.agent.set_task_cwd.assert_any_call(workspaces[slot.slot_id])
            slot.agent.set_task_cwd.assert_any_call(None)
    finally:
        invalidate_cache()
