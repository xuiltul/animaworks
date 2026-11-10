from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.messaging.inbox_quiet_window import is_deferrable_inbox_message
from core.messaging.messenger import Messenger
from core.runtime.inbox_rate_limiter import InboxRateLimiter
from core.schemas import Message


def _make_limiter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    quiet_seconds: float = 0.15,
    max_wait_seconds: float = 0.6,
) -> tuple[InboxRateLimiter, Messenger, Messenger, list[int], asyncio.Event]:
    anima_dir = tmp_path / "animas" / "alice"
    anima_dir.mkdir(parents=True)
    (anima_dir / "status.json").write_text(json.dumps({"enabled": True}), encoding="utf-8")

    shared_dir = tmp_path / "shared"
    receiver = Messenger(shared_dir, "alice")
    sender = Messenger(shared_dir, "bob")
    run_batches: list[int] = []
    inbox_ran = asyncio.Event()

    async def run_inbox() -> dict[str, object]:
        run_batches.append(receiver.unread_count())
        receiver.archive_all()
        inbox_ran.set()
        return {
            "task_type": "inbox",
            "result": {"action": "responded", "reason": "", "summary": "ok"},
            "success": True,
        }

    supervisor = MagicMock()
    supervisor.run_inbox = AsyncMock(side_effect=run_inbox)
    scheduler_mgr = MagicMock()
    scheduler_mgr.heartbeat_running = False
    scheduler_mgr._task_runner_supervisor = supervisor
    anima = MagicMock()
    anima.anima_dir = anima_dir
    anima.messenger = receiver

    monkeypatch.setattr(
        "core.runtime.inbox_rate_limiter.load_config",
        lambda: SimpleNamespace(
            inbox=SimpleNamespace(
                inbox_quiet_seconds=quiet_seconds,
                inbox_quiet_max_wait_seconds=max_wait_seconds,
            )
        ),
    )
    limiter = InboxRateLimiter(
        anima=anima,
        anima_name="alice",
        shutdown_event=asyncio.Event(),
        scheduler_mgr=scheduler_mgr,
    )
    return limiter, receiver, sender, run_batches, inbox_ran


async def _start_watcher(limiter: InboxRateLimiter) -> asyncio.Task[None]:
    task = asyncio.create_task(limiter.inbox_watcher_loop())
    await asyncio.sleep(0.03)
    return task


async def _stop_watcher(limiter: InboxRateLimiter, task: asyncio.Task[None]) -> None:
    limiter._shutdown_event.set()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def _send_and_wake(
    limiter: InboxRateLimiter,
    sender: Messenger,
    *,
    content: str = "notification",
    **kwargs: object,
) -> None:
    sender.send("alice", content, **kwargs)
    limiter._inbox_wake_event.set()


@pytest.mark.parametrize(
    ("source", "intent", "meta", "expected"),
    [
        ("anima", "report", {}, True),
        ("anima", "", {"event": "task_completed"}, True),
        ("anima", "question", {}, False),
    ],
)
def test_deferrable_message_categories(source: str, intent: str, meta: dict[str, str], expected: bool) -> None:
    msg = Message(from_person="bob", to_person="alice", content="message", source=source, intent=intent, meta=meta)
    assert is_deferrable_inbox_message(msg) is expected


@pytest.mark.asyncio
async def test_system_notifications_trail_into_one_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    limiter, _receiver, sender, run_batches, inbox_ran = _make_limiter(tmp_path, monkeypatch)
    task = await _start_watcher(limiter)
    try:
        for index in range(3):
            _send_and_wake(limiter, sender, source="system", content=f"system {index}")
            if index < 2:
                await asyncio.sleep(0.04)

        await asyncio.sleep(0.06)
        assert run_batches == []
        await asyncio.wait_for(inbox_ran.wait(), timeout=0.4)
        assert run_batches == [3]
    finally:
        await _stop_watcher(limiter, task)


@pytest.mark.asyncio
async def test_system_notifications_trigger_at_max_wait(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Every arrival extends the quiet window, so only max_wait can end it. The
    # two deadlines are far apart: the four sends fit well inside max_wait even
    # when each disk write is slow, and the run still comes long before the
    # quiet window could have ended.
    limiter, _receiver, sender, run_batches, inbox_ran = _make_limiter(
        tmp_path,
        monkeypatch,
        quiet_seconds=3.0,
        max_wait_seconds=0.8,
    )
    task = await _start_watcher(limiter)
    try:
        started_at = time.monotonic()
        for index in range(4):
            _send_and_wake(limiter, sender, source="system", content=f"system {index}")
            if index < 3:
                await asyncio.sleep(0.05)

        await asyncio.wait_for(inbox_ran.wait(), timeout=2.0)
        assert time.monotonic() - started_at < 2.0
        assert run_batches == [4]
    finally:
        await _stop_watcher(limiter, task)


@pytest.mark.asyncio
async def test_safety_rescans_do_not_extend_quiet_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    limiter, _receiver, sender, run_batches, inbox_ran = _make_limiter(
        tmp_path,
        monkeypatch,
        quiet_seconds=0.15,
        max_wait_seconds=1.0,
    )
    monkeypatch.setattr("core.runtime.inbox_rate_limiter._INBOX_RECHECK_INTERVAL_SEC", 0.03)
    task = await _start_watcher(limiter)
    try:
        started_at = time.monotonic()
        _send_and_wake(limiter, sender, source="system")
        await asyncio.wait_for(inbox_ran.wait(), timeout=0.35)
        assert time.monotonic() - started_at < 0.3
        assert run_batches == [1]
    finally:
        await _stop_watcher(limiter, task)


@pytest.mark.asyncio
async def test_question_arriving_during_quiet_window_triggers_immediately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limiter, _receiver, sender, run_batches, inbox_ran = _make_limiter(
        tmp_path,
        monkeypatch,
        quiet_seconds=0.4,
        max_wait_seconds=1.0,
    )
    task = await _start_watcher(limiter)
    try:
        _send_and_wake(limiter, sender, source="system")
        await asyncio.sleep(0.05)
        started_at = time.monotonic()
        _send_and_wake(limiter, sender, source="anima", intent="question")
        await asyncio.wait_for(inbox_ran.wait(), timeout=0.25)
        assert time.monotonic() - started_at < 0.25
        assert run_batches == [2]
    finally:
        await _stop_watcher(limiter, task)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["human", "web", "slack"])
async def test_human_and_external_messages_trigger_immediately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    limiter, _receiver, sender, run_batches, inbox_ran = _make_limiter(
        tmp_path,
        monkeypatch,
        quiet_seconds=0.5,
        max_wait_seconds=1.0,
    )
    task = await _start_watcher(limiter)
    try:
        _send_and_wake(limiter, sender, source=source)
        await asyncio.wait_for(inbox_ran.wait(), timeout=0.25)
        assert run_batches == [1]
    finally:
        await _stop_watcher(limiter, task)


@pytest.mark.asyncio
async def test_zero_quiet_seconds_triggers_immediately(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    limiter, _receiver, sender, run_batches, inbox_ran = _make_limiter(
        tmp_path,
        monkeypatch,
        quiet_seconds=0,
    )
    task = await _start_watcher(limiter)
    try:
        _send_and_wake(limiter, sender, source="system")
        await asyncio.wait_for(inbox_ran.wait(), timeout=0.25)
        assert run_batches == [1]
    finally:
        await _stop_watcher(limiter, task)


@pytest.mark.asyncio
async def test_malformed_inbox_json_triggers_immediately(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    limiter, receiver, _sender, run_batches, inbox_ran = _make_limiter(
        tmp_path,
        monkeypatch,
        quiet_seconds=0.5,
        max_wait_seconds=1.0,
    )
    task = await _start_watcher(limiter)
    try:
        (receiver.inbox_dir / "broken.json").write_text("{broken", encoding="utf-8")
        limiter._inbox_wake_event.set()
        await asyncio.wait_for(inbox_ran.wait(), timeout=0.25)
        assert run_batches == [1]
    finally:
        await _stop_watcher(limiter, task)
