"""Phase 3 chat steering through the existing IPC v2 stream."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.anima.messaging import _inject_chat_message
from core.execution.base import BaseExecutor, ExecutionResult
from core.execution.engines.claude.executor import AgentSDKExecutor
from core.memory.conversation.memory import ConversationMemory
from core.runtime import task_runner
from core.runtime.ipc import IPCRequest
from core.runtime.ipc_v2 import IPCV2ConnectionState, IPCV2Identity
from core.runtime.runner import AnimaRunner
from core.runtime.streaming_handler import StreamingIPCHandler
from core.runtime.task_runner_supervisor import TaskRunnerJob, TaskRunnerSupervisor
from core.schemas import ModelConfig


class _UnsupportedExecutor(BaseExecutor):
    async def execute(self, *args, **kwargs) -> ExecutionResult:
        return ExecutionResult(text="")


class _StreamSupervisor:
    async def run_chat_stream(self, payload: dict):
        assert payload["message"] == "hello"
        yield {"chunk": json.dumps({"type": "text_delta", "text": "child"})}
        yield {"done": True, "result": {"response": "child", "cycle_result": {"summary": "child"}}}


def _active_job(supervisor: TaskRunnerSupervisor, *, steer: bool) -> TaskRunnerJob:
    identity = IPCV2Identity(
        job_id="chat-stream-stable",
        root_epoch=supervisor.root_epoch,
        attempt=1,
        lane="chat",
        display_lane="chat",
    )
    job = TaskRunnerJob(
        identity=identity,
        request_id="run-stable",
        params={"payload": {"stream": True, "thread_id": "default"}},
        result=asyncio.get_running_loop().create_future(),
        peer_state=IPCV2ConnectionState(identity),
        pid=1234,
        capabilities={"reconnect": True, "steer": steer},
    )
    supervisor.jobs[identity.job_id] = job
    return job


@pytest.mark.asyncio
async def test_only_mode_s_adapter_supports_message_injection(tmp_path: Path) -> None:
    unsupported = _UnsupportedExecutor(ModelConfig(), tmp_path)
    assert unsupported.supports_message_injection is False
    assert await unsupported.inject_message("later") is False

    executor = AgentSDKExecutor(ModelConfig(model="claude-sonnet-4-6"), tmp_path)
    client = AsyncMock()
    executor._active_client = client

    assert executor.supports_message_injection is True
    assert await executor.inject_message("later") is True
    client.query.assert_awaited_once_with("later")


@pytest.mark.asyncio
async def test_injected_user_turn_is_persisted_once(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "sakura"
    conversation = ConversationMemory(anima_dir, ModelConfig(model="claude-sonnet-4-6"))
    conversation.append_turn("human", "first")
    conversation.save()
    agent = SimpleNamespace(supports_message_injection=True, inject_message=AsyncMock(return_value=True))
    owner = SimpleNamespace(
        agent=agent,
        _active_chat_conversations={"default": conversation},
        _log_human_conversation=MagicMock(),
        _activity=SimpleNamespace(alog=AsyncMock()),
    )

    assert await _inject_chat_message(owner, "steer", from_person="human", thread_id="default") is True

    turns = ConversationMemory(anima_dir, ModelConfig()).load().turns
    assert [turn.content for turn in turns if turn.role == "human"] == ["first", "steer"]
    agent.inject_message.assert_awaited_once_with("steer")


@pytest.mark.asyncio
async def test_declined_injection_rolls_back_user_turn(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "sakura"
    conversation = ConversationMemory(anima_dir, ModelConfig())
    conversation.append_turn("human", "first")
    conversation.save()
    owner = SimpleNamespace(
        agent=SimpleNamespace(supports_message_injection=True, inject_message=AsyncMock(return_value=False)),
        _active_chat_conversations={"default": conversation},
        _log_human_conversation=MagicMock(),
        _activity=SimpleNamespace(alog=AsyncMock()),
    )

    assert await _inject_chat_message(owner, "not sent", from_person="human", thread_id="default") is False
    turns = ConversationMemory(anima_dir, ModelConfig()).load().turns
    assert [turn.content for turn in turns] == ["first"]


@pytest.mark.asyncio
async def test_injection_round_trip_reuses_job_pid_and_session(tmp_path: Path) -> None:
    supervisor = TaskRunnerSupervisor("sakura", tmp_path / "animas" / "sakura", tmp_path / "shared")
    job = _active_job(supervisor, steer=True)
    connection = AsyncMock()
    job.connection = connection

    async def _send(event: str, data: dict) -> None:
        assert event == "inject_message"
        job.inject_waiters[data["injection_id"]].set_result(
            {"injection_id": data["injection_id"], "accepted": True, "error": ""}
        )
        for queue in job.stream_events:
            queue.put_nowait({"chunk": json.dumps({"type": "text_delta", "text": "steered"})})
            queue.put_nowait(None)
        job.result.set_result(
            {
                "result": {
                    "response": "steered",
                    "cycle_result": {"summary": "steered", "session_id": "session-stable"},
                }
            }
        )

    connection.send_event.side_effect = _send
    supervisor._run_isolated_job = AsyncMock(side_effect=AssertionError("must reuse active job"))

    events = [event async for event in supervisor.run_chat_stream({"message": "later", "thread_id": "default"})]

    assert job.identity.job_id == "chat-stream-stable"
    assert job.pid == 1234
    assert json.loads(events[0]["chunk"])["text"] == "steered"
    assert events[-1]["result"]["cycle_result"]["session_id"] == "session-stable"
    supervisor._run_isolated_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_unsupported_engine_falls_back_to_interrupt_then_lock(tmp_path: Path) -> None:
    supervisor = TaskRunnerSupervisor("sakura", tmp_path / "animas" / "sakura", tmp_path / "shared")
    job = _active_job(supervisor, steer=False)
    job.connection = AsyncMock()
    supervisor._run_isolated_job = AsyncMock(
        return_value={"response": "fallback", "cycle_result": {"summary": "fallback"}}
    )

    events = [event async for event in supervisor.run_chat_stream({"message": "later", "thread_id": "default"})]

    job.connection.send_event.assert_awaited_once_with("interrupt", {"thread_id": "default"})
    supervisor._run_isolated_job.assert_awaited_once()
    assert events[-1]["result"]["response"] == "fallback"


@pytest.mark.asyncio
async def test_rejected_injection_falls_back_without_dropping_message(tmp_path: Path) -> None:
    supervisor = TaskRunnerSupervisor("sakura", tmp_path / "animas" / "sakura", tmp_path / "shared")
    job = _active_job(supervisor, steer=True)
    connection = AsyncMock()
    job.connection = connection

    async def _send(event: str, data: dict) -> None:
        if event == "inject_message":
            job.inject_waiters[data["injection_id"]].set_result(
                {"injection_id": data["injection_id"], "accepted": False, "error": "not ready"}
            )

    connection.send_event.side_effect = _send
    supervisor._run_isolated_job = AsyncMock(return_value={"response": "fallback"})

    events = [event async for event in supervisor.run_chat_stream({"message": "later", "thread_id": "default"})]

    assert [call.args[0] for call in connection.send_event.await_args_list] == ["inject_message", "interrupt"]
    assert events[-1] == {"done": True, "result": {"response": "fallback"}}


@pytest.mark.asyncio
async def test_phase3_stream_relays_child_chunks_without_root_llm(tmp_path: Path) -> None:
    anima = MagicMock(needs_bootstrap=False)
    anima.process_message_stream = MagicMock(side_effect=AssertionError("root LLM must not run"))
    handler = StreamingIPCHandler(
        anima,
        "sakura",
        tmp_path,
        task_runner_supervisor=_StreamSupervisor(),
    )

    responses = [
        response
        async for response in handler.handle_stream(
            IPCRequest(id="req-1", method="process_message", params={"message": "hello", "stream": True})
        )
    ]

    assert json.loads(responses[0].chunk or "{}")["text"] == "child"
    assert responses[-1].done is True
    anima.process_message_stream.assert_not_called()


@pytest.mark.asyncio
async def test_phase3_nonstream_chat_uses_child(tmp_path: Path) -> None:
    runner = AnimaRunner("sakura", tmp_path / "a.sock", tmp_path / "animas", tmp_path / "shared")
    runner.anima = MagicMock()
    supervisor = MagicMock()
    supervisor.run_chat = AsyncMock(return_value={"response": "child"})
    runner._scheduler_mgr = SimpleNamespace(_task_runner_supervisor=supervisor)

    result = await runner._handle_process_message({"message": "hello"})

    assert result == {"response": "child"}
    supervisor.run_chat.assert_awaited_once_with(kind="message", payload={"message": "hello"})
    runner.anima.process_message.assert_not_called()


@pytest.mark.asyncio
async def test_child_stream_contract_preserves_existing_wire_format(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "sakura"
    anima_dir.mkdir(parents=True)
    anima = MagicMock(name="sakura", anima_dir=anima_dir, needs_bootstrap=False)
    anima.name = "sakura"
    anima.anima_dir = anima_dir

    async def _stream(*args, **kwargs):
        yield {"type": "text_delta", "text": "answer"}
        yield {"type": "cycle_done", "cycle_result": {"summary": "answer"}}

    anima.process_message_stream = _stream
    events: list[tuple[str, dict]] = []

    async def _send(event: str, data: dict) -> None:
        events.append((event, data))

    with patch("core.config.load_config") as config:
        config.return_value.server.keepalive_interval = 30
        result = await task_runner.execute_chat_contract(
            anima,
            kind="message",
            payload={"message": "hello", "stream": True},
            send_stream_event=_send,
        )

    assert result["response"] == "answer"
    assert json.loads(events[0][1]["chunk"]) == {"type": "text_delta", "text": "answer"}
