"""Phase 3 chat-lane routing, streaming, serialization, and recovery."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.execution._sdk_session import _load_session_id, _save_session_id
from core.execution.codex_sdk import _load_thread_id, _save_thread_id
from core.memory.conversation import ConversationMemory
from core.memory.streaming_journal import StreamingJournal
from core.schemas import ModelConfig
from core.supervisor import task_runner
from core.supervisor.ipc import IPCRequest
from core.supervisor.ipc_v2 import IPCV2ConnectionState, IPCV2Identity
from core.supervisor.runner import AnimaRunner
from core.supervisor.streaming_handler import StreamingIPCHandler
from core.supervisor.task_runner_supervisor import TaskRunnerJob, TaskRunnerSupervisor


class _StreamSupervisor:
    async def run_chat_stream(self, payload: dict):
        assert payload["message"] == "hello"
        yield {"stream": True, "chunk": json.dumps({"type": "text_delta", "text": "child"})}
        yield {"done": True, "result": {"response": "child", "cycle_result": {"summary": "child"}}}


class _CrashedStreamSupervisor:
    async def run_chat_stream(self, payload: dict):
        yield {"stream": True, "chunk": json.dumps({"type": "text_delta", "text": "partial"})}
        raise RuntimeError("task runner exited before returning a result (exit=-9)")


@pytest.mark.asyncio
async def test_phase3_stream_relays_child_chunks_without_root_llm(tmp_path: Path) -> None:
    anima = MagicMock(needs_bootstrap=False)
    anima.process_message_stream = MagicMock(side_effect=AssertionError("root LLM must not run"))
    handler = StreamingIPCHandler(
        anima,
        "sakura",
        tmp_path,
        task_runner_supervisor=_StreamSupervisor(),
        chat_isolated=True,
    )

    responses = [
        response
        async for response in handler.handle_stream(
            IPCRequest(id="req-1", method="process_message", params={"message": "hello", "stream": True})
        )
    ]

    assert json.loads(responses[0].chunk or "{}") == {"type": "text_delta", "text": "child"}
    assert responses[-1].done is True
    assert responses[-1].result == {"response": "child", "cycle_result": {"summary": "child"}}
    anima.process_message_stream.assert_not_called()


@pytest.mark.asyncio
async def test_closing_isolated_stream_cancels_producer_and_releases_lock(tmp_path: Path) -> None:
    supervisor = TaskRunnerSupervisor("sakura", tmp_path / "animas" / "sakura", tmp_path / "shared")
    producer_cancelled = asyncio.Event()

    async def _run(**kwargs):
        await kwargs["stream_events"][0].put({"chunk": "partial"})
        try:
            await asyncio.Event().wait()
        finally:
            producer_cancelled.set()

    supervisor._run_isolated_job = AsyncMock(side_effect=_run)
    handler = StreamingIPCHandler(
        MagicMock(needs_bootstrap=False),
        "sakura",
        tmp_path,
        task_runner_supervisor=supervisor,
        chat_isolated=True,
    )
    stream = handler.handle_stream(
        IPCRequest(id="req-close", method="process_message", params={"message": "hello", "stream": True})
    )

    assert (await anext(stream)).chunk == "partial"
    assert supervisor._chat_lock.locked()

    await stream.aclose()

    assert producer_cancelled.is_set()
    assert not supervisor._chat_lock.locked()


@pytest.mark.asyncio
async def test_chat_child_sigkill_becomes_stream_error_and_root_stays_usable(tmp_path: Path) -> None:
    anima = MagicMock(needs_bootstrap=False)
    handler = StreamingIPCHandler(
        anima,
        "sakura",
        tmp_path,
        task_runner_supervisor=_CrashedStreamSupervisor(),
        chat_isolated=True,
    )

    responses = [
        response
        async for response in handler.handle_stream(
            IPCRequest(id="req-1", method="process_message", params={"message": "hello", "stream": True})
        )
    ]

    assert json.loads(responses[0].chunk or "{}")["text"] == "partial"
    assert responses[-1].error == {
        "code": "CHAT_RUNNER_ERROR",
        "message": "task runner exited before returning a result (exit=-9)",
    }
    assert handler._anima is anima


@pytest.mark.asyncio
async def test_legacy_stream_does_not_spawn_chat_runner(tmp_path: Path) -> None:
    anima = MagicMock(needs_bootstrap=False)

    async def _stream(*args, **kwargs):
        yield {"type": "cycle_done", "cycle_result": {"summary": "legacy"}}

    anima.process_message_stream = _stream
    isolated = MagicMock()
    handler = StreamingIPCHandler(
        anima,
        "sakura",
        tmp_path,
        task_runner_supervisor=isolated,
        chat_isolated=False,
    )

    responses = [
        response
        async for response in handler.handle_stream(
            IPCRequest(id="req-1", method="process_message", params={"message": "hello", "stream": True})
        )
    ]

    assert responses[-1].result == {
        "response": "legacy",
        "replied_to": [],
        "cycle_result": {"summary": "legacy"},
    }
    isolated.run_chat_stream.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,kind",
    [("_handle_process_message", "message"), ("_handle_greet", "greet"), ("_handle_run_bootstrap", "bootstrap")],
)
async def test_phase3_nonstream_chat_contracts_use_child(
    tmp_path: Path,
    method: str,
    kind: str,
) -> None:
    runner = AnimaRunner("sakura", tmp_path / "a.sock", tmp_path / "animas", tmp_path / "shared")
    runner.anima = MagicMock()
    supervisor = MagicMock()
    supervisor.run_chat = AsyncMock(return_value={"response": "child"})
    runner._scheduler_mgr = SimpleNamespace(_chat_isolated=True, _task_runner_supervisor=supervisor)

    result = await getattr(runner, method)({"message": "hello"})

    assert result == {"response": "child"}
    supervisor.run_chat.assert_awaited_once_with(kind=kind, payload={"message": "hello"})
    runner.anima.process_message.assert_not_called()
    runner.anima.process_greet.assert_not_called()
    runner.anima.run_bootstrap.assert_not_called()


@pytest.mark.asyncio
async def test_second_chat_interrupts_then_waits_for_lane_lock(tmp_path: Path) -> None:
    supervisor = TaskRunnerSupervisor("sakura", tmp_path / "animas" / "sakura", tmp_path / "shared")
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    connection = AsyncMock()
    calls: list[str] = []

    async def _run(**kwargs):
        calls.append(kwargs["job_prefix"])
        if len(calls) == 1:
            identity = IPCV2Identity(
                job_id="active-chat",
                root_epoch=supervisor.root_epoch,
                attempt=1,
                lane="chat",
                display_lane="chat",
            )
            supervisor.jobs[identity.job_id] = TaskRunnerJob(
                identity=identity,
                request_id="run-active",
                params={},
                result=asyncio.get_running_loop().create_future(),
                peer_state=IPCV2ConnectionState(identity),
                connection=connection,
            )
            first_started.set()
            await release_first.wait()
            supervisor.jobs.pop(identity.job_id)
        return {"response": "ok"}

    supervisor._run_isolated_job = AsyncMock(side_effect=_run)
    first = asyncio.create_task(supervisor.run_chat(kind="message", payload={"thread_id": "default"}))
    await first_started.wait()
    second = asyncio.create_task(supervisor.run_chat(kind="message", payload={"thread_id": "default"}))
    await asyncio.sleep(0)

    connection.send_event.assert_awaited_once_with("interrupt", {"thread_id": "default"})
    assert calls == ["chat-message"]
    assert not second.done()

    release_first.set()
    await asyncio.gather(first, second)
    assert calls == ["chat-message", "chat-message"]


@pytest.mark.asyncio
async def test_child_stream_contract_keeps_existing_double_json(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "sakura"
    anima_dir.mkdir(parents=True)
    anima = MagicMock(name="sakura", anima_dir=anima_dir, needs_bootstrap=False)
    anima.name = "sakura"
    anima.anima_dir = anima_dir

    async def _stream(*args, **kwargs):
        yield {"type": "thinking_delta", "text": "hmm"}
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
    assert [json.loads(data["chunk"])["type"] for event, data in events if event == "stream"] == [
        "thinking_delta",
        "text_delta",
    ]


async def test_chat_journal_recovery_persists_once_after_child_crash(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "sakura"
    anima_dir.mkdir(parents=True)
    journal = StreamingJournal(anima_dir, thread_id="default")
    journal.open(trigger="message:human", from_person="human")
    journal.write_text("partial response" * 40)
    journal.close()
    owner = SimpleNamespace(model_config=ModelConfig(model="claude-sonnet-4-6", max_tokens=4096))
    supervisor = TaskRunnerSupervisor("sakura", anima_dir, tmp_path / "shared", busy_status_owner=owner)

    await supervisor._recover_task_journals(("chat",))
    await supervisor._recover_task_journals(("chat",))

    turns = ConversationMemory(anima_dir, owner.model_config).load().turns
    assistant_turns = [turn for turn in turns if turn.role == "assistant"]
    assert len(assistant_turns) == 1
    assert assistant_turns[0].content.startswith("partial response")
    assert not StreamingJournal.has_orphan(anima_dir, "chat")


def test_chat_exit_fsyncs_conversation_and_engine_resume_files(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "sakura"
    conversation = anima_dir / "state" / "conversation.json"
    conversation.parent.mkdir(parents=True, exist_ok=True)
    conversation.write_text("state", encoding="utf-8")
    _save_session_id(anima_dir, "claude-session", "chat")
    _save_thread_id(anima_dir, "codex-thread", "chat")

    with patch("core.supervisor.task_runner.os.fsync") as fsync:
        task_runner._fsync_chat_state(anima_dir)

    assert fsync.call_count == 3
    assert _load_session_id(anima_dir, "chat") == "claude-session"
    assert _load_thread_id(anima_dir, "chat") == "codex-thread"
