"""Chat steer queuing: interrupt concurrency, injection expiry, and keepalives."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core.runtime import task_runner, task_runner_supervisor
from core.runtime.ipc_v2 import IPCV2Envelope, IPCV2Identity
from core.runtime.task_runner_supervisor import TaskRunnerJob, TaskRunnerSupervisor


class _LoopConnection:
    """IPC v2 connection whose frames are consumed only by the control loop."""

    def __init__(self, identity: IPCV2Identity) -> None:
        self.identity = identity
        self.frames: asyncio.Queue[IPCV2Envelope] = asyncio.Queue()
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.responses: list[tuple[str, dict[str, Any] | None, dict[str, Any] | None]] = []
        self.closed = False
        self.inject_results: list[dict[str, Any]] = []
        self.inject_result = asyncio.Event()

    async def send_event(self, event: str, data: dict[str, Any] | None = None) -> int:
        payload = data or {}
        self.events.append((event, payload))
        if event == "inject_result":
            self.inject_results.append(payload)
            self.inject_result.set()
        return len(self.events)

    async def send_request(self, _request_id: str, _method: str, _params: dict[str, Any]) -> int:
        return 1

    async def receive(self) -> IPCV2Envelope:
        return await self.frames.get()

    async def send_response(
        self,
        request_id: str,
        *,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> int:
        self.responses.append((request_id, result, error))
        return 1

    async def wait_for_ack(self, _seq: int) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


def _identity(job_id: str) -> IPCV2Identity:
    return IPCV2Identity(
        job_id=job_id,
        root_epoch=str(uuid.uuid4()),
        attempt=1,
        lane="chat",
        display_lane="chat",
    )


def _chat_contract(identity: IPCV2Identity) -> IPCV2Envelope:
    return IPCV2Envelope(
        kind="request",
        identity=identity,
        seq=2,
        body={
            "request_id": "run-chat-1",
            "method": "run",
            "params": {
                "kind": "message",
                "payload": {"message": "first", "stream": True, "thread_id": "default"},
                "environment": {
                    "urls": {
                        "ANIMAWORKS_EMBED_URL": "http://embed.test",
                        "ANIMAWORKS_VECTOR_URL": "http://vector.test",
                        "ANIMAWORKS_RERANK_URL": "http://rerank.test",
                    }
                },
            },
        },
    )


def _inject_frame(identity: IPCV2Identity, injection_id: str, message: str, **extra: Any) -> IPCV2Envelope:
    return IPCV2Envelope(
        kind="event",
        identity=identity,
        seq=3,
        body={
            "event": "inject_message",
            "data": {
                "injection_id": injection_id,
                "message": message,
                "from_person": "human",
                "thread_id": "default",
                **extra,
            },
        },
    )


def _event_frame(identity: IPCV2Identity, event: str, data: dict[str, Any]) -> IPCV2Envelope:
    return IPCV2Envelope(kind="event", identity=identity, seq=3, body={"event": event, "data": data})


def _set_runner_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANIMAWORKS_EMBED_URL", "http://embed.test")
    monkeypatch.setenv("ANIMAWORKS_VECTOR_URL", "http://vector.test")
    monkeypatch.setenv("ANIMAWORKS_RERANK_URL", "http://rerank.test")


class _BlockingAnima:
    """Anima whose injection blocks until the test releases it."""

    def __init__(self) -> None:
        self.agent = SimpleNamespace(_executor=SimpleNamespace(supports_message_injection=True))
        self.inject_started = asyncio.Event()
        self.release_inject = asyncio.Event()
        self.inject_done = asyncio.Event()
        self.interrupted = asyncio.Event()
        self.injected: list[str] = []

    async def inject_message(self, message: str, **_kwargs: Any) -> bool:
        self.injected.append(message)
        self.inject_started.set()
        await self.release_inject.wait()
        self.inject_done.set()
        return True

    async def interrupt(self, thread_id: str | None = None) -> None:
        self.interrupted.set()


@pytest.mark.asyncio
async def test_interrupt_event_is_handled_while_injection_is_in_flight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = _identity("chat-stream-interrupt")
    args = argparse.Namespace(anima="sakura", lane="chat", job=identity.job_id)
    _set_runner_urls(monkeypatch)
    connection = _LoopConnection(identity)
    anima = _BlockingAnima()
    connection.frames.put_nowait(_inject_frame(identity, "inject-1", "steer"))
    connection.frames.put_nowait(_event_frame(identity, "interrupt", {"thread_id": "default"}))

    async def _execute_chat(
        _anima: Any,
        *,
        kind: str,
        payload: dict[str, Any],
        send_stream_event: Any,
    ) -> dict[str, Any]:
        await asyncio.wait_for(connection.inject_result.wait(), timeout=4.0)
        return {"response": "steered", "cycle_result": {"summary": "steered"}}

    monkeypatch.setattr(task_runner, "_connect", AsyncMock(return_value=(connection, _chat_contract(identity))))
    monkeypatch.setattr(task_runner, "DigitalAnima", lambda **_kwargs: anima)
    monkeypatch.setattr(task_runner, "execute_chat_contract", _execute_chat)

    run = asyncio.create_task(task_runner.run_task(args, tmp_path / "task.sock", identity))
    try:
        await asyncio.wait_for(anima.inject_started.wait(), timeout=2.0)
        await asyncio.wait_for(anima.interrupted.wait(), timeout=2.0)
        # The interrupt was delivered before the injection finished.
        assert not anima.inject_done.is_set()
    finally:
        anima.release_inject.set()

    exit_code = await asyncio.wait_for(run, timeout=5.0)

    assert exit_code == 0
    assert anima.injected == ["steer"]
    assert connection.inject_results == [{"injection_id": "inject-1", "accepted": True, "error": ""}]


class _CountingAnima:
    def __init__(self) -> None:
        self.agent = SimpleNamespace(_executor=SimpleNamespace(supports_message_injection=True))
        self.injected: list[str] = []

    async def inject_message(self, message: str, **_kwargs: Any) -> bool:
        self.injected.append(message)
        return True


@pytest.mark.asyncio
async def test_expired_injection_skips_persistence_and_reports_expired(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = _identity("chat-stream-expiry")
    args = argparse.Namespace(anima="sakura", lane="chat", job=identity.job_id)
    _set_runner_urls(monkeypatch)
    connection = _LoopConnection(identity)
    anima = _CountingAnima()
    connection.frames.put_nowait(_inject_frame(identity, "inject-expired", "late", expires_at=time.time() - 1.0))
    connection.frames.put_nowait(_inject_frame(identity, "inject-legacy", "alive"))

    async def _execute_chat(
        _anima: Any,
        *,
        kind: str,
        payload: dict[str, Any],
        send_stream_event: Any,
    ) -> dict[str, Any]:
        async with asyncio.timeout(4.0):
            while len(connection.inject_results) < 2:
                await asyncio.sleep(0.01)
        return {"response": "steered", "cycle_result": {"summary": "steered"}}

    monkeypatch.setattr(task_runner, "_connect", AsyncMock(return_value=(connection, _chat_contract(identity))))
    monkeypatch.setattr(task_runner, "DigitalAnima", lambda **_kwargs: anima)
    monkeypatch.setattr(task_runner, "execute_chat_contract", _execute_chat)

    exit_code = await asyncio.wait_for(task_runner.run_task(args, tmp_path / "task.sock", identity), timeout=5.0)

    assert exit_code == 0
    assert anima.injected == ["alive"]
    assert connection.inject_results == [
        {"injection_id": "inject-expired", "accepted": False, "error": "expired"},
        {"injection_id": "inject-legacy", "accepted": True, "error": ""},
    ]


def _active_job(supervisor: TaskRunnerSupervisor) -> TaskRunnerJob:
    identity = _identity("chat-stream-active")
    job = TaskRunnerJob(
        identity=identity,
        request_id="run-active",
        params={"payload": {"stream": True, "thread_id": "default"}},
        result=asyncio.get_running_loop().create_future(),
        peer_state=task_runner_supervisor.IPCV2ConnectionState(identity),
        pid=1234,
        capabilities={"reconnect": True, "steer": True},
    )
    supervisor.jobs[identity.job_id] = job
    return job


@pytest.mark.asyncio
async def test_steer_event_carries_expiry_five_seconds_before_the_wait_budget(tmp_path: Path) -> None:
    supervisor = TaskRunnerSupervisor("sakura", tmp_path / "animas" / "sakura", tmp_path / "shared")
    job = _active_job(supervisor)
    connection = AsyncMock()
    captured: dict[str, Any] = {}

    async def _send(event: str, data: dict[str, Any]) -> None:
        captured.update(data)
        job.inject_waiters[data["injection_id"]].set_result(
            {"injection_id": data["injection_id"], "accepted": True, "error": ""}
        )

    connection.send_event.side_effect = _send
    job.connection = connection

    before = time.time()
    found, active, _queue = await supervisor._try_inject_active_chat({"message": "later", "thread_id": "default"})
    after = time.time()

    assert found is True
    assert active is job
    assert task_runner_supervisor._CHAT_STEER_TIMEOUT == 30.0
    expires_at = captured["expires_at"]
    assert before + 25.0 <= expires_at <= after + 25.0


@pytest.mark.asyncio
async def test_queued_stream_yields_keepalives_and_then_the_normal_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = TaskRunnerSupervisor("sakura", tmp_path / "animas" / "sakura", tmp_path / "shared")
    supervisor._run_isolated_job = AsyncMock(return_value={"response": "queued done"})
    monkeypatch.setattr(task_runner_supervisor, "_chat_keepalive_interval", lambda: 0.05)
    await supervisor._chat_semaphore.acquire()

    events: list[dict[str, Any]] = []

    async def _collect() -> None:
        async for event in supervisor.run_chat_stream({"message": "later", "thread_id": "default"}):
            events.append(event)

    stream = asyncio.create_task(_collect())
    try:
        async with asyncio.timeout(2.0):
            while sum(1 for event in events if _is_keepalive(event)) < 2:
                await asyncio.sleep(0.01)
    finally:
        supervisor._chat_semaphore.release()

    await asyncio.wait_for(stream, timeout=3.0)

    keepalives = [json.loads(event["chunk"]) for event in events if _is_keepalive(event)]
    assert all(payload["queued"] is True for payload in keepalives)
    assert all(payload["elapsed_s"] >= 0 for payload in keepalives)
    assert events[-1] == {"done": True, "result": {"response": "queued done"}}
    supervisor._run_isolated_job.assert_awaited_once()
    assert "default" not in supervisor._chat_thread_locks


def _is_keepalive(event: dict[str, Any]) -> bool:
    chunk = event.get("chunk")
    return isinstance(chunk, str) and json.loads(chunk).get("type") == "keepalive"


@pytest.mark.asyncio
async def test_closing_while_queued_releases_thread_lock_and_semaphore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = TaskRunnerSupervisor("sakura", tmp_path / "animas" / "sakura", tmp_path / "shared")
    supervisor._run_isolated_job = AsyncMock(return_value={"response": "never"})
    monkeypatch.setattr(task_runner_supervisor, "_chat_keepalive_interval", lambda: 0.05)
    await supervisor._chat_semaphore.acquire()

    stream = supervisor.run_chat_stream({"message": "later", "thread_id": "default"})
    try:
        first = await asyncio.wait_for(anext(stream), timeout=2.0)
        assert _is_keepalive(first)
        await stream.aclose()
    finally:
        supervisor._chat_semaphore.release()

    async with asyncio.timeout(2.0):
        async with supervisor._chat_slot("default"):
            pass

    supervisor._run_isolated_job.assert_not_awaited()
    assert "default" not in supervisor._chat_thread_locks
