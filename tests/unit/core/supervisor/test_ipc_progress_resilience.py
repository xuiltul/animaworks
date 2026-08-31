"""Regression tests for the false hang-kill mechanism (2026-08-31).

A broken control socket used to permanently silence a healthy task runner:
the child cancelled its progress loop on receiver loss, the progress loop
died on its first send failure, and the root severed connections after a 15s
receive timeout. 900s later the hang watchdog killed runners that were still
working. These tests pin the healing behaviour on all three layers.
"""

from __future__ import annotations

import asyncio
import socket
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.supervisor import task_runner, task_runner_supervisor
from core.supervisor.ipc import IPCServer
from core.supervisor.ipc_v2 import (
    IPC_V2_MAX_FRAME_BYTES,
    IPCV2Connection,
    IPCV2ConnectionError,
    IPCV2ConnectionState,
    IPCV2Identity,
)
from core.supervisor.task_runner import _progress_loop, _RootLink
from core.supervisor.task_runner_supervisor import TaskRunnerJob, TaskRunnerSupervisor

# ── child: _progress_loop never dies ───────────────────────────────────


class _FlakyConnection:
    def __init__(self, fail_first: int = 1) -> None:
        self.fail_remaining = fail_first
        self.sent: list[str] = []

    async def send_event(self, event: str, data=None) -> int:
        if self.fail_remaining > 0:
            self.fail_remaining -= 1
            raise IPCV2ConnectionError("connection is closed")
        self.sent.append(event)
        return len(self.sent)


class _StubLink:
    def __init__(self, first: _FlakyConnection, healed: _FlakyConnection) -> None:
        self.connection = first
        self._healed = healed
        self.reconnect_calls = 0

    async def reconnect(self, broken):
        self.reconnect_calls += 1
        self.connection = self._healed
        return self.connection


@pytest.mark.asyncio
async def test_progress_loop_survives_send_failure_and_heals(monkeypatch) -> None:
    monkeypatch.setattr(task_runner, "_PROGRESS_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(task_runner, "_RECONNECT_RETRY_SECONDS", 0.01)
    identity = IPCV2Identity(
        job_id="job-p", root_epoch="epoch", attempt=1, lane="task", display_lane="background"
    )
    broken = _FlakyConnection(fail_first=10_000)  # never recovers on its own
    healed = _FlakyConnection(fail_first=0)
    link = _StubLink(broken, healed)

    loop_task = asyncio.create_task(_progress_loop(link, identity))
    try:
        for _ in range(200):
            if len(healed.sent) >= 2:
                break
            await asyncio.sleep(0.01)
        assert len(healed.sent) >= 2, "progress loop died instead of healing the link"
        assert link.reconnect_calls >= 1
    finally:
        loop_task.cancel()
        await asyncio.gather(loop_task, return_exceptions=True)


# ── child: _RootLink.reconnect is single-flight and validates contract ─


@pytest.mark.asyncio
async def test_rootlink_reconnect_single_flight(monkeypatch, tmp_path) -> None:
    identity = IPCV2Identity(
        job_id="job-l", root_epoch="epoch", attempt=1, lane="task", display_lane="background"
    )
    state = IPCV2ConnectionState(identity)
    broken = SimpleNamespace(close=AsyncMock())
    new_conn = SimpleNamespace(close=AsyncMock())
    connect_mock = AsyncMock(return_value=(new_conn, SimpleNamespace(body={"request_id": "req-1"})))
    monkeypatch.setattr(task_runner, "_connect", connect_mock)

    memory_client = SimpleNamespace(connection=broken)
    link = _RootLink(broken, tmp_path / "sock", state, "req-1", memory_client)

    results = await asyncio.gather(link.reconnect(broken), link.reconnect(broken))
    assert results == [new_conn, new_conn]
    assert connect_mock.await_count == 1, "reconnect must be single-flight"
    assert link.connection is new_conn
    assert memory_client.connection is new_conn


@pytest.mark.asyncio
async def test_rootlink_reconnect_rejects_foreign_run_contract(monkeypatch, tmp_path) -> None:
    identity = IPCV2Identity(
        job_id="job-x", root_epoch="epoch", attempt=1, lane="task", display_lane="background"
    )
    broken = SimpleNamespace(close=AsyncMock())
    new_conn = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(
        task_runner,
        "_connect",
        AsyncMock(return_value=(new_conn, SimpleNamespace(body={"request_id": "other"}))),
    )
    link = _RootLink(broken, tmp_path / "sock", IPCV2ConnectionState(identity), "req-1", None)
    with pytest.raises(IPCV2ConnectionError):
        await link.reconnect(broken)
    new_conn.close.assert_awaited()


# ── root: a quiet receive interval must not sever the connection ───────


async def _stream_pair():
    left, right = socket.socketpair()
    r1, w1 = await asyncio.open_connection(sock=left, limit=IPC_V2_MAX_FRAME_BYTES + 1)
    r2, w2 = await asyncio.open_connection(sock=right, limit=IPC_V2_MAX_FRAME_BYTES + 1)
    return (r1, w1), (r2, w2)


@pytest.mark.asyncio
async def test_supervisor_receive_timeout_does_not_sever(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(task_runner_supervisor, "_RECEIVE_POLL_TIMEOUT_SEC", 0.1)
    sup = TaskRunnerSupervisor("t-anima", tmp_path / "anima", tmp_path / "shared")
    identity = IPCV2Identity(
        job_id="job-quiet",
        root_epoch=sup.root_epoch,
        attempt=1,
        lane="task",
        display_lane="background",
    )
    job = TaskRunnerJob(
        identity=identity,
        request_id=f"run-{uuid.uuid4()}",
        params={},
        result=asyncio.get_running_loop().create_future(),
        peer_state=IPCV2ConnectionState(identity),
        process=None,
        pid=999_999,
        pgid=999_999,
        last_progress_at=asyncio.get_running_loop().time(),
    )
    sup._jobs[identity.job_id] = job

    (server_r, server_w), (client_r, client_w) = await _stream_pair()
    handler = asyncio.create_task(sup._handle_connection(server_r, server_w))
    client = IPCV2Connection(client_r, client_w, IPCV2ConnectionState(identity))

    inbound: list = []

    async def _drain_client() -> None:
        while True:
            inbound.append(await client.receive())

    drain = asyncio.create_task(_drain_client())
    try:
        await client.send_event(
            "hello",
            {"capabilities": {"reconnect": True, "steer": False}, "last_received_seq": 0, "last_acked_seq": 0},
        )
        for _ in range(100):  # wait for the replayed run request
            if any(e.kind == "request" for e in inbound):
                break
            await asyncio.sleep(0.01)
        assert any(e.kind == "request" for e in inbound), "handshake did not complete"

        # Stay silent for several receive-poll intervals.
        await asyncio.sleep(0.5)
        assert not handler.done(), "quiet interval severed the connection"

        before = job.last_progress_at
        await client.send_event(
            "progress",
            {
                "pid": 1,
                "pgid": 1,
                "lane": identity.lane,
                "job_id": identity.job_id,
                "display_lane": identity.display_lane,
                "progress_at": 0.0,
            },
        )
        for _ in range(100):
            if job.last_progress_at > before:
                break
            await asyncio.sleep(0.01)
        assert job.last_progress_at > before, "progress after a quiet interval was not recorded"
    finally:
        for task in (drain, handler):
            task.cancel()
        await asyncio.gather(drain, handler, return_exceptions=True)
        for w in (client_w, server_w):
            w.close()


# ── v1 IPC: connection handler must never propagate (client_connected_cb) ──


class _BrokenWriter:
    def __init__(self) -> None:
        self.closed = False

    def get_extra_info(self, *args, **kwargs):
        return "test-peer"

    def write(self, data) -> None:
        raise BrokenPipeError(32, "Broken pipe")

    async def drain(self) -> None:
        raise BrokenPipeError(32, "Broken pipe")

    def close(self) -> None:
        self.closed = True
        raise BrokenPipeError(32, "Broken pipe")

    async def wait_closed(self) -> None:
        raise BrokenPipeError(32, "Broken pipe")


@pytest.mark.asyncio
async def test_v1_handler_survives_broken_pipe(tmp_path) -> None:
    server = IPCServer(tmp_path / "v1.sock", AsyncMock(side_effect=ConnectionResetError("Connection lost")))
    reader = asyncio.StreamReader()
    reader.feed_data(b'{"id": "1", "method": "ping", "params": {}}\n')
    reader.feed_eof()
    writer = _BrokenWriter()
    # Must not raise — an escape here surfaces as
    # "Unhandled exception in client_connected_cb" and severs sibling state.
    await server._handle_connection(reader, writer)
    assert writer.closed
