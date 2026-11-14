"""Regression: a chat steer must not deadlock the task runner control loop.

Before the fix the control loop awaited ``anima.inject_message`` inline, so the
state-write request issued inside it could never receive its response frame
(only the same loop's ``connection.receive()`` delivers that frame) and the
injection hung until the 300 s IPC timeout.
"""

from __future__ import annotations

import argparse
import asyncio
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core.platform.state_writer import get_state_writer
from core.runtime import task_runner
from core.runtime.ipc_v2 import IPCV2Envelope, IPCV2Identity


class _LoopConnection:
    """IPC v2 connection whose frames are consumed only by the control loop."""

    def __init__(self, identity: IPCV2Identity) -> None:
        self.identity = identity
        self.frames: asyncio.Queue[IPCV2Envelope] = asyncio.Queue()
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.requests: list[tuple[str, str, dict[str, Any]]] = []
        self.responses: list[tuple[str, dict[str, Any] | None, dict[str, Any] | None]] = []
        self.closed = False
        self.inject_result = asyncio.Event()

    async def send_event(self, event: str, data: dict[str, Any] | None = None) -> int:
        payload = data or {}
        self.events.append((event, payload))
        if event == "inject_result":
            self.inject_result.set()
        return len(self.events)

    async def send_request(self, request_id: str, method: str, params: dict[str, Any]) -> int:
        self.requests.append((request_id, method, params))
        # The response is only delivered by the control loop's receive().
        self.frames.put_nowait(
            IPCV2Envelope(
                kind="response",
                identity=self.identity,
                seq=1,
                body={"request_id": request_id, "result": {"ok": True}, "error": None},
            )
        )
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


class _InjectingAnima:
    """Minimal chat anima whose injection performs an IPC state write."""

    def __init__(self) -> None:
        self.injected: list[str] = []
        self.agent = SimpleNamespace(_executor=SimpleNamespace(supports_message_injection=True))

    async def inject_message(self, message: str, **_kwargs: Any) -> bool:
        self.injected.append(message)
        # Same path as _persist_conversation in the real anima.
        await get_state_writer().save_conversation("default", {"turns": []})
        return True


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


@pytest.mark.asyncio
async def test_inject_message_completes_while_control_loop_keeps_receiving(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = IPCV2Identity(
        job_id="chat-stream-deadlock",
        root_epoch=str(uuid.uuid4()),
        attempt=1,
        lane="chat",
        display_lane="chat",
    )
    args = argparse.Namespace(anima="sakura", lane="chat", job=identity.job_id)
    monkeypatch.setenv("ANIMAWORKS_EMBED_URL", "http://embed.test")
    monkeypatch.setenv("ANIMAWORKS_VECTOR_URL", "http://vector.test")
    monkeypatch.setenv("ANIMAWORKS_RERANK_URL", "http://rerank.test")

    connection = _LoopConnection(identity)
    anima = _InjectingAnima()
    connection.frames.put_nowait(
        IPCV2Envelope(
            kind="event",
            identity=identity,
            seq=3,
            body={
                "event": "inject_message",
                "data": {
                    "injection_id": "inject-1",
                    "message": "steer",
                    "from_person": "human",
                    "thread_id": "default",
                },
            },
        )
    )

    async def _execute_chat(
        _anima: Any,
        *,
        kind: str,
        payload: dict[str, Any],
        send_stream_event: Any,
    ) -> dict[str, Any]:
        # Keep the execution alive until the steer round-trip has finished.
        await asyncio.wait_for(connection.inject_result.wait(), timeout=4.0)
        return {"response": "steered", "cycle_result": {"summary": "steered"}}

    monkeypatch.setattr(task_runner, "_connect", AsyncMock(return_value=(connection, _chat_contract(identity))))
    monkeypatch.setattr(task_runner, "DigitalAnima", lambda **_kwargs: anima)
    monkeypatch.setattr(task_runner, "execute_chat_contract", _execute_chat)

    exit_code = await asyncio.wait_for(task_runner.run_task(args, tmp_path / "task.sock", identity), timeout=5.0)

    assert exit_code == 0
    assert anima.injected == ["steer"]
    assert len(connection.requests) == 1
    inject_results = [data for event, data in connection.events if event == "inject_result"]
    assert inject_results == [{"injection_id": "inject-1", "accepted": True, "error": ""}]
    assert connection.responses[0][0] == "run-chat-1"
    assert connection.responses[0][2] is None
    assert connection.closed
