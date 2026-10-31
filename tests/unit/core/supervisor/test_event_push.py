from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from core.activity.logger import ActivityLogger, set_live_event_sink
from core.runtime.event_bus import RootEventBus
from core.runtime.ipc import IPCRequest, IPCResponse
from core.runtime.runner import AnimaRunner
from server.supervisor.manager import HealthConfig, ProcessSupervisor
from server.supervisor.process_handle import ProcessHandle, ProcessState


async def _next_or_none(events):
    try:
        return await anext(events)
    except StopAsyncIteration:
        return None


def _make_supervisor(tmp_path: Path, *, ws_manager=None) -> ProcessSupervisor:
    return ProcessSupervisor(
        animas_dir=tmp_path / "animas",
        shared_dir=tmp_path / "shared",
        run_dir=tmp_path / "run",
        ws_manager=ws_manager,
        health_config=HealthConfig(ping_interval_sec=0.001),
    )


@pytest.mark.asyncio
async def test_root_event_bus_replays_buffered_events_in_order() -> None:
    bus = RootEventBus()
    bus.bind_loop()
    bus.publish({"index": 1})
    bus.publish({"index": 2})

    events = bus.subscribe()
    try:
        assert await anext(events) == {"index": 1}
        assert await anext(events) == {"index": 2}
    finally:
        await events.aclose()


@pytest.mark.asyncio
async def test_root_event_bus_drops_oldest_and_marks_loss() -> None:
    bus = RootEventBus(max_events=2)
    bus.bind_loop()
    for index in range(3):
        bus.publish({"index": index})

    events = bus.subscribe()
    try:
        assert await anext(events) == {"index": 1}
        assert await anext(events) == {"index": 2, "dropped": 1}
    finally:
        await events.aclose()


@pytest.mark.asyncio
async def test_root_event_bus_accepts_publish_from_another_thread() -> None:
    bus = RootEventBus()
    bus.bind_loop()
    thread = threading.Thread(target=bus.publish, args=({"source": "thread"},))
    thread.start()
    await asyncio.to_thread(thread.join)

    events = bus.subscribe()
    try:
        assert await asyncio.wait_for(anext(events), timeout=1) == {"source": "thread"}
    finally:
        await events.aclose()


@pytest.mark.asyncio
async def test_new_root_event_subscriber_ends_previous_subscriber() -> None:
    bus = RootEventBus()
    bus.bind_loop()
    first = bus.subscribe()
    first_waiter = asyncio.create_task(_next_or_none(first))
    await asyncio.sleep(0)

    second = bus.subscribe()
    second_waiter = asyncio.create_task(_next_or_none(second))
    assert await asyncio.wait_for(first_waiter, timeout=1) is None

    bus.publish({"subscriber": "new"})
    try:
        assert await asyncio.wait_for(second_waiter, timeout=1) == {"subscriber": "new"}
    finally:
        await first.aclose()
        await second.aclose()


@pytest.mark.asyncio
async def test_runner_subscribe_events_streams_events_and_shutdown_done() -> None:
    runner = AnimaRunner.__new__(AnimaRunner)
    runner._event_bus = RootEventBus()
    runner._event_bus.bind_loop()
    runner._event_keepalive_interval = 0.05
    runner.shutdown_event = asyncio.Event()
    runner._streaming_handler = None

    stream = await runner._handle_request(IPCRequest(id="subscribe-1", method="subscribe_events"))
    assert hasattr(stream, "__aiter__")
    runner._emit_event("anima.heartbeat", {"name": "alice"})
    event_response = await asyncio.wait_for(anext(stream), timeout=1)
    assert json.loads(event_response.chunk) == {"event": "anima.heartbeat", "data": {"name": "alice"}}

    runner.shutdown_event.set()
    done_response = await asyncio.wait_for(anext(stream), timeout=1)
    assert done_response.done is True
    await stream.aclose()


@pytest.mark.asyncio
async def test_runner_event_stream_emits_keepalive_when_idle() -> None:
    runner = AnimaRunner.__new__(AnimaRunner)
    runner._event_bus = RootEventBus()
    runner._event_bus.bind_loop()
    runner._event_keepalive_interval = 0.001
    runner.shutdown_event = asyncio.Event()
    runner._streaming_handler = None

    stream = await runner._handle_request(IPCRequest(id="subscribe-2", method="subscribe_events"))
    keepalive = await asyncio.wait_for(anext(stream), timeout=1)
    assert json.loads(keepalive.chunk) == {"keepalive": True}

    runner.shutdown_event.set()
    done = await asyncio.wait_for(anext(stream), timeout=1)
    assert done.done is True
    await stream.aclose()


@pytest.mark.asyncio
async def test_process_handle_event_stream_does_not_mark_chat_streaming(tmp_path: Path) -> None:
    handle = ProcessHandle(
        anima_name="alice",
        socket_path=tmp_path / "alice.sock",
        animas_dir=tmp_path / "animas",
        shared_dir=tmp_path / "shared",
    )
    handle.state = ProcessState.RUNNING
    requests: list[tuple[IPCRequest, float | None]] = []

    class FakeIPCClient:
        async def send_request_stream(self, request: IPCRequest, timeout: float | None = None):
            requests.append((request, timeout))
            yield IPCResponse(id=request.id, stream=True, chunk=json.dumps({"keepalive": True}))
            yield IPCResponse(
                id=request.id,
                stream=True,
                chunk=json.dumps({"event": "anima.cron", "data": {"name": "alice"}}),
            )
            yield IPCResponse(id=request.id, stream=True, done=True)

    handle.ipc_client = FakeIPCClient()
    events = [event async for event in handle.open_event_stream()]

    assert events == [{"event": "anima.cron", "data": {"name": "alice"}}]
    assert requests[0][0].method == "subscribe_events"
    assert requests[0][1] == 60.0
    assert handle.is_streaming is False


@pytest.mark.asyncio
async def test_supervisor_consumes_events_and_resubscribes_after_disconnect(tmp_path: Path) -> None:
    broadcasted = asyncio.Event()
    ws_manager = SimpleNamespace(broadcast=AsyncMock(side_effect=lambda _event: broadcasted.set()))
    supervisor = _make_supervisor(tmp_path, ws_manager=ws_manager)
    handle = SimpleNamespace(state=ProcessState.RUNNING)
    supervisor.processes["alice"] = handle
    subscription_count = 0

    async def event_stream():
        nonlocal subscription_count
        subscription_count += 1
        if subscription_count == 1:
            yield {"event": "anima.interaction", "data": {"from_person": "alice"}}
            raise ConnectionError("stream disconnected")
        await asyncio.Event().wait()
        yield {}

    handle.open_event_stream = event_stream
    task = asyncio.create_task(supervisor._consume_events("alice", handle))
    try:
        await asyncio.wait_for(broadcasted.wait(), timeout=1)
        for _ in range(100):
            if subscription_count >= 2:
                break
            await asyncio.sleep(0.01)
        assert subscription_count >= 2
        ws_manager.broadcast.assert_awaited_once_with({"type": "anima.interaction", "data": {"from_person": "alice"}})
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_activity_logger_uses_sink_without_writing_fallback_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ActivityLogger._live_rate_limiter.reset()
    anima_dir = tmp_path / "animas" / "alice"
    anima_dir.mkdir(parents=True)
    activity = ActivityLogger(anima_dir)
    received: list[dict] = []
    monkeypatch.setattr("core.activity.logger._LIVE_EVENT_SINK", None)
    set_live_event_sink(received.append)
    try:
        with patch.object(ActivityLogger, "_export_event"):
            activity.log("tool_use", tool="Read", summary="source")
    finally:
        set_live_event_sink(None)

    assert len(received) == 1
    assert received[0]["event"] == "anima.tool_activity"
    assert received[0]["data"]["tool"] == "Read"
    assert not (tmp_path / "run" / "events" / "alice").exists()


@pytest.mark.asyncio
async def test_activity_logger_falls_back_to_file_when_sink_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ActivityLogger._live_rate_limiter.reset()
    anima_dir = tmp_path / "animas" / "alice"
    anima_dir.mkdir(parents=True)
    activity = ActivityLogger(anima_dir)
    monkeypatch.setattr("core.activity.logger._LIVE_EVENT_SINK", None)

    def failing_sink(_event: dict) -> None:
        raise RuntimeError("sink unavailable")

    set_live_event_sink(failing_sink)
    try:
        with (
            patch.object(ActivityLogger, "_export_event"),
            patch("core.activity.logger.get_data_dir", return_value=tmp_path),
        ):
            activity.log("tool_use", tool="Read", summary="fallback")
    finally:
        set_live_event_sink(None)

    event_files = list((tmp_path / "run" / "events" / "alice").glob("ta_*.json"))
    assert len(event_files) == 1
    assert json.loads(event_files[0].read_text(encoding="utf-8"))["data"]["summary"] == "fallback"


@pytest.mark.asyncio
async def test_health_loop_does_not_poll_event_files(tmp_path: Path) -> None:
    supervisor = _make_supervisor(tmp_path)
    supervisor._poll_anima_events = AsyncMock()
    supervisor._poll_requested_rag_repairs = AsyncMock(side_effect=lambda: setattr(supervisor, "_shutdown", True))

    await supervisor._health_check_loop()

    supervisor._poll_anima_events.assert_not_awaited()


@pytest.mark.asyncio
async def test_event_file_drain_loop_polls_fallback_files(tmp_path: Path) -> None:
    supervisor = _make_supervisor(tmp_path)
    polled = asyncio.Event()

    async def poll_events() -> None:
        polled.set()

    supervisor._poll_anima_events = poll_events
    task = asyncio.create_task(supervisor._event_file_drain_loop())
    try:
        await asyncio.wait_for(polled.wait(), timeout=1)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
