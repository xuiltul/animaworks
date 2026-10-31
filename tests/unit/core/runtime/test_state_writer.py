"""Tests for Anima-main and IPC state-writer equivalence and chunking."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from core.runtime.ipc_v2 import (
    IPC_V2_MAX_FRAME_BYTES,
    IPCV2ConnectionState,
    IPCV2Identity,
)
from core.runtime.state_writer import IpcStateWriter, LocalStateWriter, StateWriterError


class _LocalWriterLink:
    """In-process test peer implementing the state_write frame protocol."""

    def __init__(self, writer: LocalStateWriter) -> None:
        self.writer = writer
        self.pending: dict[str, dict[str, Any]] = {}
        self.sent_frames: list[dict[str, Any]] = []
        self.response_count = 0

    @staticmethod
    def _check_frame(params: dict[str, Any]) -> None:
        wire_size = len(json.dumps(params, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        assert wire_size < IPC_V2_MAX_FRAME_BYTES

    async def send_unanswered_request(self, method: str, params: dict[str, Any]) -> None:
        assert method == "state_write"
        self._check_frame(params)
        self.sent_frames.append(params)
        phase = params["phase"]
        transaction_id = params["transaction_id"]
        if phase == "begin":
            self.pending[transaction_id] = {
                "operation": params["operation"],
                "byte_length": params["byte_length"],
                "sha256": params["sha256"],
                "chunks": [],
            }
        else:
            transaction = self.pending[transaction_id]
            transaction["chunks"].append(base64.b64decode(params["data"], validate=True))

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert method == "state_write"
        self._check_frame(params)
        self.response_count += 1
        if params.get("phase") == "commit":
            transaction = self.pending.pop(params["transaction_id"])
            payload_bytes = b"".join(transaction["chunks"])
            assert len(payload_bytes) == transaction["byte_length"]
            assert hashlib.sha256(payload_bytes).hexdigest() == transaction["sha256"]
            body = json.loads(payload_bytes)
            assert body["operation"] == transaction["operation"]
            return await self.writer.execute_operation(body["operation"], body["payload"])

        return await self.writer.execute_operation(params["operation"], params["payload"])


def _relative_files(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


async def _exercise_semantic_operations(
    writer: LocalStateWriter | IpcStateWriter,
    anima_dir: Path,
) -> dict[str, Any]:
    await writer.save_conversation("default", {"anima_name": "alice", "turns": [{"content": "hello"}]})
    await writer.append_transcript({"ts": "2026-10-01T10:00:00+09:00", "role": "human", "content": "hello"})
    await writer.save_shortterm(
        "chat",
        "default",
        {"session_id": "session-1", "turn_count": 2},
        "# Short-term state\n",
    )
    await writer.save_stream_checkpoint(
        "chat",
        "default",
        {"timestamp": "2026-10-01T10:00:00+09:00", "retry_count": 1},
    )
    await writer.save_session_record("agent_sdk", "chat", "default", {"session_id": "sdk-1", "timestamp": "now"})
    await writer.save_session_record("cursor", "chat", "thread-1", {"session_id": "cursor-1", "turn_count": 3})
    await writer.log_token_usage({"ts": "2026-10-01T10:00:00+09:00", "total_tokens": 12})
    await writer.save_token_usage_rollup({"2026-09-30": 20})
    await writer.log_prompt({"ts": "2026-10-01T10:00:00+09:00", "type": "request_start", "user_message": "x"})
    await writer.log_prompt_end({"ts": "2026-10-01T10:00:00+09:00", "type": "request_end"})
    await writer.write_heartbeat_checkpoint({"ts": "2026-10-01T10:00:00+09:00", "trigger": "heartbeat"})
    await writer.write_recovery_note("recover me")
    await writer.update_inbox_read_counts({"message.json": 2})
    await writer.save_task_result("task-1", "attempt-1", "task result")
    await writer.write_inbox_overflow_file(
        {"from_person": "sender", "ts": "2026-10-01T10:00:00+09:00", "content": "overflow"}
    )
    await writer.write_token_budget_notification(
        "2026-10",
        {"budget": 10, "consumed": 10, "trigger": "chat"},
    )
    await writer.mark_cron_rejected_notice("digest-1")
    await writer.write_background_notification("task-2", "background result")
    await writer.write_bootstrap_state({"version": 1, "state": "running"})
    await writer.save_background_task("bg-1", {"task_id": "bg-1", "status": "running"})
    await writer.set_consolidation_mode(True)
    await writer.set_consolidation_mode(False)
    await writer.rotate_prompt_logs()

    (anima_dir / "HEARTBEAT.md").write_text("# Heartbeat\n", encoding="utf-8")
    (anima_dir / "BOOTSTRAP.md").write_text("# Bootstrap\n", encoding="utf-8")
    archived_heartbeat = await writer.archive_heartbeat_md_snapshot()
    archived_bootstrap = await writer.archive_bootstrap_file("BOOTSTRAP.md")

    await writer.archive_shortterm("chat", "default")
    legacy_dir = anima_dir / "shortterm"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    (legacy_dir / "session_state.json").write_text('{"session_id":"legacy"}', encoding="utf-8")
    await writer.migrate_legacy_shortterm("chat", "default")

    recovery_note = await writer.consume_recovery_note()
    notifications = await writer.consume_background_notifications("all")
    await writer.clear_stream_checkpoint("chat", "default")
    await writer.clear_heartbeat_checkpoint()
    await writer.clear_session("cursor", "chat", "thread-1")
    await writer.clear_background_task("bg-1")
    await writer.clear_conversation("thread-2")
    await writer.cleanup_inbox_overflow()
    return {
        "archived_heartbeat": archived_heartbeat,
        "archived_bootstrap": str(archived_bootstrap.relative_to(anima_dir)),
        "recovery_note": recovery_note,
        "notifications": notifications,
    }


@pytest.mark.asyncio
async def test_local_and_ipc_state_writers_produce_identical_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.platform.state_writer as state_writer_module

    monkeypatch.setenv("ANIMAWORKS_PROCESS_ROLE", "anima")
    fixed = datetime(2026, 10, 1, 10, 0, 0).astimezone()
    monkeypatch.setattr(state_writer_module, "now_local", lambda: fixed)
    monkeypatch.setattr(state_writer_module, "now_iso", lambda: "2026-10-01T10:00:00+09:00")
    local_dir = tmp_path / "local" / "alice"
    ipc_dir = tmp_path / "ipc" / "alice"
    local_dir.mkdir(parents=True)
    ipc_dir.mkdir(parents=True)

    local_results = await _exercise_semantic_operations(LocalStateWriter(local_dir), local_dir)
    ipc_link = _LocalWriterLink(LocalStateWriter(ipc_dir))
    ipc_results = await _exercise_semantic_operations(IpcStateWriter(ipc_link), ipc_dir)

    assert local_results == ipc_results
    assert _relative_files(local_dir) == _relative_files(ipc_dir)
    assert not ipc_link.pending
    assert ipc_link.response_count > 0


@pytest.mark.asyncio
async def test_local_state_writer_rejects_task_runner_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANIMAWORKS_PROCESS_ROLE", "task_runner")
    with pytest.raises(StateWriterError, match="cannot be used by a task_runner"):
        await LocalStateWriter(tmp_path / "alice").save_conversation("default", {"turns": []})


@pytest.mark.asyncio
async def test_real_ipc_v2_chunks_five_mib_payload_without_backpressure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANIMAWORKS_PROCESS_ROLE", "anima")
    from core.runtime.task_runner import _connect, _RootLink
    from core.runtime.task_runner_supervisor import TaskRunnerJob, TaskRunnerSupervisor

    identity = IPCV2Identity(
        job_id="chunked-state-write",
        root_epoch=str(uuid.uuid4()),
        attempt=1,
        lane="chat",
        display_lane="chat",
    )
    loop = asyncio.get_running_loop()
    anima_dir = tmp_path / "alice"
    shared_dir = tmp_path / "shared"
    supervisor = TaskRunnerSupervisor("alice", anima_dir, shared_dir)
    job = TaskRunnerJob(
        identity=identity,
        request_id="run-contract",
        params={},
        result=loop.create_future(),
        peer_state=IPCV2ConnectionState(identity),
    )
    supervisor.jobs[identity.job_id] = job
    socket_path = tmp_path / "chunked-state.sock"
    server = await asyncio.start_unix_server(
        supervisor._handle_connection,
        path=str(socket_path),
        limit=IPC_V2_MAX_FRAME_BYTES + 1,
    )
    child_state = IPCV2ConnectionState(identity)
    connection, _run = await _connect(socket_path, child_state)
    link = _RootLink(connection, socket_path, child_state, "run-contract")
    receiver_errors: list[Exception] = []

    async def _receive_responses() -> None:
        while True:
            try:
                envelope = await link.connection.receive()
            except Exception as exc:
                receiver_errors.append(exc)
                return
            if envelope.kind == "response":
                link.receive_response(envelope)

    receiver_task = asyncio.create_task(_receive_responses())
    payload_text = "x" * (5 * 1024 * 1024)
    state = {"anima_name": "alice", "turns": [{"content": payload_text}]}
    try:
        await asyncio.wait_for(IpcStateWriter(link).save_conversation("default", state), timeout=20.0)
        actual = json.loads((anima_dir / "state" / "conversation.json").read_text(encoding="utf-8"))
        assert actual == state
        assert not receiver_errors

        ack_seq = await connection.send_request(
            "write-no-response",
            "state_write",
            {
                "operation": "write_background_notification",
                "payload": {"task_id": "orphan-write", "content": "already accepted"},
            },
        )
        await connection.wait_for_ack(ack_seq)
        for _ in range(100):
            if "write-no-response" in job.state_write_pending:
                break
            await asyncio.sleep(0.01)
        assert "write-no-response" in job.state_write_pending
        await connection.close()
        for _ in range(200):
            notification_path = anima_dir / "state" / "background_notifications" / "orphan-write.md"
            if notification_path.exists():
                break
            await asyncio.sleep(0.01)
        assert notification_path.read_text(encoding="utf-8") == "already accepted"
    finally:
        receiver_task.cancel()
        await asyncio.gather(receiver_task, return_exceptions=True)
        await connection.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_ipc_state_writer_chunks_five_mib_payload_and_commits_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload_text = "x" * (5 * 1024 * 1024)
    state = {"anima_name": "alice", "turns": [{"content": payload_text}]}
    local_dir = tmp_path / "local" / "alice"
    ipc_dir = tmp_path / "ipc" / "alice"
    local_dir.mkdir(parents=True)
    ipc_dir.mkdir(parents=True)

    monkeypatch.setenv("ANIMAWORKS_PROCESS_ROLE", "anima")
    await LocalStateWriter(local_dir).save_conversation("default", state)
    link = _LocalWriterLink(LocalStateWriter(ipc_dir))
    await IpcStateWriter(link).save_conversation("default", state)

    assert (local_dir / "state" / "conversation.json").read_bytes() == (
        ipc_dir / "state" / "conversation.json"
    ).read_bytes()
    assert len(link.sent_frames) > 1
    assert link.response_count == 1
    assert all(frame["phase"] == "chunk" for frame in link.sent_frames[1:])
