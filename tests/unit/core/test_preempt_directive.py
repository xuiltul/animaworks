from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory.task_queue import TaskQueueManager
from core.supervisor.pending_executor import (
    _SENTINEL_CONTINUED,
    PendingTaskExecutor,
)
from core.tasks_dispatch import dispatch_direct_task, request_preemption


class _LaneAnima:
    """Minimal anima stub exposing a real background interrupt event."""

    def __init__(self) -> None:
        self.agent = MagicMock(name="chat_agent")
        self.background_agent = MagicMock(name="background_agent")
        self.messenger = MagicMock()
        self._events: dict[str, asyncio.Event] = {}
        self._background_lock = asyncio.Lock()

    def _agent_for_lane(self, lane: str):
        return self.background_agent if lane == "background" else self.agent

    def _agent_session_context(self, lane: str):
        return self._background_lock if lane == "background" else asyncio.Lock()

    def _get_interrupt_event(self, name: str) -> asyncio.Event:
        self._events.setdefault(name, asyncio.Event())
        return self._events[name]


def _make_executor(tmp_path: Path) -> PendingTaskExecutor:
    anima_dir = tmp_path / "animas" / "lane-anima"
    anima_dir.mkdir(parents=True, exist_ok=True)
    return PendingTaskExecutor(
        anima=_LaneAnima(),
        anima_name="lane-anima",
        anima_dir=anima_dir,
        shutdown_event=asyncio.Event(),
    )


def _queue_task(executor: PendingTaskExecutor, task_id: str, *, meta: dict | None = None) -> TaskQueueManager:
    manager = TaskQueueManager(executor._anima_dir)
    manager.add_task(
        source="anima",
        original_instruction="finish the task",
        assignee="lane-anima",
        summary="preempt test",
        task_id=task_id,
        meta=meta,
    )
    return manager


def _task(task_id: str, **overrides) -> dict:
    task = {
        "task_type": "llm",
        "task_id": task_id,
        "title": "Preempt test",
        "description": "finish the task",
        "context": "original context",
        "reply_to": None,
        "submitted_at": "",
    }
    task.update(overrides)
    return task


# ── request_preemption (task 1) ────────────────────────────────────────────


def test_request_preemption_marks_only_in_progress_holder(tmp_path: Path) -> None:
    target_dir = tmp_path / "natsume"
    (target_dir / "state" / "pending").mkdir(parents=True)
    manager = TaskQueueManager(target_dir)
    manager.add_task(
        source="anima",
        original_instruction="running PR work",
        assignee="natsume",
        summary="running PR work",
        task_id="holder-running",
        meta={"exclusive_key": "pr-5008"},
    )
    manager.update_status("holder-running", "in_progress", summary="running")
    manager.add_task(
        source="anima",
        original_instruction="queued PR work",
        assignee="natsume",
        summary="queued PR work",
        task_id="holder-pending",
        meta={"exclusive_key": "pr-5008"},
    )

    holder_id = request_preemption(target_dir, "pr-5008", "gh-cmd-101")
    assert holder_id == "holder-running"
    running = manager.get_task_by_id("holder-running")
    assert running is not None
    assert running.meta["preempt_requested_by"] == "gh-cmd-101"
    assert "preempt_requested_at" in running.meta
    pending = manager.get_task_by_id("holder-pending")
    assert pending is not None
    assert "preempt_requested_by" not in pending.meta


def test_request_preemption_returns_none_when_no_in_progress_holder(tmp_path: Path) -> None:
    target_dir = tmp_path / "natsume"
    (target_dir / "state" / "pending").mkdir(parents=True)
    manager = TaskQueueManager(target_dir)
    manager.add_task(
        source="anima",
        original_instruction="queued PR work",
        assignee="natsume",
        summary="queued PR work",
        task_id="holder-pending",
        meta={"exclusive_key": "pr-5008"},
    )

    assert request_preemption(target_dir, "pr-5008", "gh-cmd-101") is None
    pending = manager.get_task_by_id("holder-pending")
    assert pending is not None
    assert "preempt_requested_by" not in pending.meta


def test_request_preemption_no_match_returns_none(tmp_path: Path) -> None:
    target_dir = tmp_path / "natsume"
    (target_dir / "state" / "pending").mkdir(parents=True)
    assert request_preemption(target_dir, "pr-9999", "gh-cmd-101") is None


# ── dispatch_direct_task preemption trigger (task 1-2) ────────────────────


def _dispatch(tmp_path: Path, task_id: str, meta: dict) -> bool:
    target_dir = tmp_path / "natsume"
    (target_dir / "state" / "pending").mkdir(parents=True, exist_ok=True)
    return dispatch_direct_task(
        target="natsume",
        task_id=task_id,
        summary="directive",
        instruction="do it",
        meta=meta,
        animas_dir=tmp_path,
    )


def test_dispatch_directive_requests_preemption_when_key_held(tmp_path: Path, monkeypatch) -> None:
    target_dir = tmp_path / "natsume"
    (target_dir / "state" / "pending").mkdir(parents=True)
    manager = TaskQueueManager(target_dir)
    manager.add_task(
        source="anima",
        original_instruction="in-flight PR work",
        assignee="natsume",
        summary="in-flight PR work",
        task_id="holder-1",
        meta={"exclusive_key": "pr-5008"},
    )
    manager.update_status("holder-1", "in_progress", summary="running")

    calls: list = []

    def _fake_preempt(t_dir, key, preemptor):
        calls.append((t_dir, key, preemptor))
        return "holder-1"

    monkeypatch.setattr("core.tasks_dispatch.request_preemption", _fake_preempt)
    assert _dispatch(
        tmp_path,
        task_id="gh-cmd-101",
        meta={"repo": "o/r", "number": 5008, "priority_class": "directive"},
    )
    assert len(calls) == 1
    t_dir, key, preemptor = calls[0]
    assert key == "pr-5008"
    assert preemptor == "gh-cmd-101"


def test_dispatch_directive_no_preemption_without_key(tmp_path: Path, monkeypatch) -> None:
    calls: list = []
    monkeypatch.setattr(
        "core.tasks_dispatch.request_preemption",
        lambda *a, **k: calls.append(a),
    )
    assert _dispatch(
        tmp_path,
        task_id="gh-cmd-101",
        meta={"repo": "o/r", "priority_class": "directive"},
    )
    assert calls == []


def test_dispatch_non_directive_no_preemption_even_with_key(tmp_path: Path, monkeypatch) -> None:
    calls: list = []
    monkeypatch.setattr(
        "core.tasks_dispatch.request_preemption",
        lambda *a, **k: calls.append(a),
    )
    assert _dispatch(
        tmp_path,
        task_id="gh-ci-o-r#5008-aaaa",
        meta={"repo": "o/r", "number": 5008, "sha": "aaaa"},
    )
    assert calls == []


# ── claim ordering (task 3-3) ──────────────────────────────────────────────


def test_order_pending_claims_directive_first(tmp_path: Path) -> None:
    executor = _make_executor(tmp_path)
    pending_dir = executor._anima_dir / "state" / "pending"
    pending_dir.mkdir(parents=True)
    normal = pending_dir / "normal.json"
    directive = pending_dir / "directive.json"
    normal.write_text(json.dumps({"task_id": "normal", "priority_class": "ci"}))
    directive.write_text(json.dumps({"task_id": "directive", "priority_class": "directive"}))
    other = pending_dir / "other.json"
    other.write_text(json.dumps({"task_id": "other"}))

    ordered = executor._order_pending_claims(sorted(pending_dir.glob("*.json")))
    assert ordered[0] == directive
    assert set(ordered[1:]) == {normal, other}


def test_order_pending_claims_keeps_unreadable_for_existing_error_handling(tmp_path: Path) -> None:
    executor = _make_executor(tmp_path)
    pending_dir = executor._anima_dir / "state" / "pending"
    pending_dir.mkdir(parents=True)
    bad = pending_dir / "bad.json"
    bad.write_text("not json")
    directive = pending_dir / "directive.json"
    directive.write_text(json.dumps({"task_id": "directive", "priority_class": "directive"}))

    ordered = executor._order_pending_claims(sorted(pending_dir.glob("*.json")))
    assert ordered[0] == directive
    assert ordered[1] == bad


def test_yield_to_defers_claim_until_preemptor_terminal(tmp_path: Path) -> None:
    executor = _make_executor(tmp_path)
    pending_dir = executor._anima_dir / "state" / "pending"
    pending_dir.mkdir(parents=True)
    processing_dir = pending_dir / "processing"
    processing_dir.mkdir()
    manager = _queue_task(executor, "preemptor")
    manager.update_status("preemptor", "in_progress", summary="running")

    task_desc = _task("yielded", yield_to="preemptor")
    path = pending_dir / "yielded.json"
    path.write_text(json.dumps(task_desc))

    # Preemptor still in_progress → defer.
    assert executor._should_defer_claim(path, task_desc, processing_dir) is True

    manager.update_status("preemptor", "done", summary="finished")
    # Preemptor terminal → claim allowed (no continuation_not_before / no active id).
    assert executor._should_defer_claim(path, task_desc, processing_dir) is False


def test_yield_to_defers_while_preemptor_pending(tmp_path: Path) -> None:
    executor = _make_executor(tmp_path)
    pending_dir = executor._anima_dir / "state" / "pending"
    pending_dir.mkdir(parents=True)
    processing_dir = pending_dir / "processing"
    processing_dir.mkdir()
    _queue_task(executor, "preemptor")
    # Default status pending.

    task_desc = _task("yielded", yield_to="preemptor")
    path = pending_dir / "yielded.json"
    path.write_text(json.dumps(task_desc))
    assert executor._should_defer_claim(path, task_desc, processing_dir) is True


def test_yield_to_missing_target_allows_claim(tmp_path: Path) -> None:
    executor = _make_executor(tmp_path)
    pending_dir = executor._anima_dir / "state" / "pending"
    pending_dir.mkdir(parents=True)
    processing_dir = pending_dir / "processing"
    processing_dir.mkdir()

    task_desc = _task("yielded", yield_to="no-such-task")
    path = pending_dir / "yielded.json"
    path.write_text(json.dumps(task_desc))
    assert executor._should_defer_claim(path, task_desc, processing_dir) is False


# ── executor polling: preempt detection (task 3-1) ────────────────────────


def _patch_streaming(executor: PendingTaskExecutor, stop_kind: str):
    async def stream(*_args, **_kwargs):
        yield {"type": "text_delta", "text": "output"}
        yield {
            "type": "cycle_done",
            "cycle_result": {
                "summary": "result",
                "action": "responded",
                "stop_kind": stop_kind,
                "tool_call_records": [],
            },
        }

    executor._anima.background_agent.run_cycle_streaming = MagicMock(side_effect=lambda *a, **k: stream(*a, **k))
    executor._anima.background_agent.reset_reply_tracking = MagicMock()
    executor._anima.background_agent.reset_read_paths = MagicMock()
    executor._anima.background_agent.set_interrupt_event = MagicMock()
    executor._anima.background_agent.set_task_cwd = MagicMock()


@pytest.mark.asyncio
async def test_polling_preempt_request_sets_interrupt_and_reenqueues(tmp_path: Path) -> None:
    executor = _make_executor(tmp_path)
    _patch_streaming(executor, "interrupted")
    manager = _queue_task(executor, "preempted-task")
    manager.update_meta(
        "preempted-task",
        {"preempt_requested_by": "gh-cmd-101", "preempt_requested_at": "2026-09-01T00:00:00Z"},
    )
    bg_event = executor._anima._get_interrupt_event("_background")

    with (
        patch("core.supervisor.pending_executor._CANCEL_POLL_SECONDS", 0.0),
        patch("core.supervisor.pending_executor._completion_declaration_required", return_value=False),
        patch("core.paths.load_prompt", return_value="prompt"),
        patch("core.memory.activity.ActivityLogger"),
    ):
        result = await executor._run_llm_task(_task("preempted-task"))

    assert result == _SENTINEL_CONTINUED
    assert bg_event.is_set() is True

    entry = manager.get_task_by_id("preempted-task")
    assert entry is not None
    assert entry.meta.get("preempt_requested_by") is None
    assert entry.meta.get("preempt_requested_at") is None

    pending = json.loads(
        (executor._anima_dir / "state" / "pending" / "preempted-task.json").read_text(encoding="utf-8")
    )
    assert pending["yield_to"] == "gh-cmd-101"
    assert pending["preempt_reenqueue_count"] == 1
    # continuation_count is untouched by preemption (was never set on the desc).
    assert pending.get("continuation_count", 0) == 0
    assert pending["continuation_not_before"] > time.time()
    assert "preempted" in pending["context"]


def test_reenqueue_with_checkpoint_preempt_sets_yield_and_not_before(tmp_path: Path) -> None:
    executor = _make_executor(tmp_path)
    pending_dir = executor._anima_dir / "state" / "pending"
    pending_dir.mkdir(parents=True)
    task_desc = _task("preempt-task", continuation_count=1)
    executor._reenqueue_with_checkpoint(
        task_desc,
        accumulated_text="progress output",
        tool_call_records=[],
        preempted_by="gh-cmd-101",
    )
    pending = json.loads((pending_dir / "preempt-task.json").read_text(encoding="utf-8"))
    assert pending["yield_to"] == "gh-cmd-101"
    assert pending["preempt_reenqueue_count"] == 1
    assert pending["continuation_count"] == 1
    assert pending["continuation_not_before"] > time.time()
    assert "preempted" in pending["context"]


# ── preemption eviction details (task 3-2) ────────────────────────────────


@pytest.mark.asyncio
async def test_preempt_eviction_does_not_consume_continuation(tmp_path: Path) -> None:
    """Continuation_count stays untouched; preempt_reenqueue_count increments."""
    executor = _make_executor(tmp_path)
    _patch_streaming(executor, "interrupted")
    manager = _queue_task(executor, "preempt-task")
    manager.update_meta("preempt-task", {"preempt_requested_by": "gh-cmd-101"})

    with (
        patch("core.supervisor.pending_executor._CANCEL_POLL_SECONDS", 0.0),
        patch("core.supervisor.pending_executor._completion_declaration_required", return_value=False),
        patch("core.paths.load_prompt", return_value="prompt"),
        patch("core.memory.activity.ActivityLogger"),
    ):
        await executor._run_llm_task(_task("preempt-task", continuation_count=2))

    pending = json.loads((executor._anima_dir / "state" / "pending" / "preempt-task.json").read_text(encoding="utf-8"))
    assert pending["continuation_count"] == 2
    assert pending["preempt_reenqueue_count"] == 1
    assert pending["yield_to"] == "gh-cmd-101"


@pytest.mark.asyncio
async def test_runaway_halt_still_continuation_reenqueued(tmp_path: Path) -> None:
    """Regression guard: runaway_halt must still be continuation re-enqueued.

    The preemption handling is only for ``stop_kind == "interrupted"``; the
    traditional continuation path must still cover ``runaway_halt`` (along
    with empty_response / hard_timeout), consuming a continuation.
    """
    executor = _make_executor(tmp_path)
    _patch_streaming(executor, "runaway_halt")
    _queue_task(executor, "runaway-task")

    with (
        patch("core.supervisor.pending_executor._CANCEL_POLL_SECONDS", 0.0),
        patch("core.supervisor.pending_executor._completion_declaration_required", return_value=False),
        patch("core.paths.load_prompt", return_value="prompt"),
        patch("core.memory.activity.ActivityLogger"),
    ):
        result = await executor._run_llm_task(_task("runaway-task"))

    assert result == _SENTINEL_CONTINUED
    pending = json.loads((executor._anima_dir / "state" / "pending" / "runaway-task.json").read_text(encoding="utf-8"))
    assert pending["continuation_count"] == 1
    assert "yield_to" not in pending
    assert "preempt_reenqueue_count" not in pending
    assert "runaway" in pending["context"]


@pytest.mark.asyncio
async def test_polling_preempt_limit_reached_clears_meta_without_interrupt(tmp_path: Path) -> None:
    executor = _make_executor(tmp_path)
    _patch_streaming(executor, "normal")
    manager = _queue_task(executor, "limited-task")
    manager.update_meta(
        "limited-task",
        {"preempt_requested_by": "gh-cmd-101", "preempt_requested_at": "2026-09-01T00:00:00Z"},
    )
    bg_event = executor._anima._get_interrupt_event("_background")

    with (
        patch("core.supervisor.pending_executor._CANCEL_POLL_SECONDS", 0.0),
        patch("core.supervisor.pending_executor._completion_declaration_required", return_value=False),
        patch("core.paths.load_prompt", return_value="prompt"),
        patch("core.memory.activity.ActivityLogger"),
    ):
        await executor._run_llm_task(_task("limited-task", preempt_reenqueue_count=5))

    assert bg_event.is_set() is False
    entry = manager.get_task_by_id("limited-task")
    assert entry is not None
    assert entry.meta.get("preempt_requested_by") is None
    assert entry.meta.get("preempt_requested_at") is None


@pytest.mark.asyncio
async def test_polling_preempt_ignored_when_interrupt_event_is_none(tmp_path: Path) -> None:
    """interrupt_event is None → preemption polling is skipped (legacy behavior)."""
    anima_dir = tmp_path / "animas" / "no-lane"
    anima_dir.mkdir(parents=True)
    anima = MagicMock(name="no-lane")
    anima._background_lock = asyncio.Lock()

    async def stream(*_args, **_kwargs):
        yield {"type": "text_delta", "text": "output"}
        yield {
            "type": "cycle_done",
            "cycle_result": {"summary": "result", "action": "responded", "stop_kind": "normal"},
        }

    anima.agent.run_cycle_streaming = stream
    anima.agent.reset_reply_tracking = MagicMock()
    anima.agent.reset_read_paths = MagicMock()
    anima.agent.set_task_cwd = MagicMock()
    executor = PendingTaskExecutor(
        anima=anima,
        anima_name="no-lane",
        anima_dir=anima_dir,
        shutdown_event=asyncio.Event(),
    )
    manager = _queue_task(executor, "no-event-task")
    manager.update_meta("no-event-task", {"preempt_requested_by": "gh-cmd-101"})

    with (
        patch("core.supervisor.pending_executor._CANCEL_POLL_SECONDS", 0.0),
        patch("core.supervisor.pending_executor._completion_declaration_required", return_value=False),
        patch("core.paths.load_prompt", return_value="prompt"),
        patch("core.memory.activity.ActivityLogger"),
    ):
        await executor._run_llm_task(_task("no-event-task"))

    # Preempt request not honored; task completes normally.
    entry = manager.get_task_by_id("no-event-task")
    assert entry is not None
    assert entry.meta.get("preempt_requested_by") == "gh-cmd-101"
