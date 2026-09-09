# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest

from cli.tui.app import AnimaChatApp, decide_reattach
from cli.tui.client import AnimaWorksClientError
from cli.tui.session import new_session, save_session
from cli.tui.sse import SseEvent


class FakeClient:
    def __init__(self, *, animas=None, history=None):
        self.animas = animas or [{"name": "sora", "status": "idle"}]
        self.history = history or {"sessions": [], "has_more": False, "next_before": None}
        self.active_stream = {"active": False}
        self.chat_queue: asyncio.Queue = asyncio.Queue()
        self.stream_calls: list = []
        self.history_before: list = []
        self._ws = asyncio.Event()

    async def list_animas(self):
        return self.animas

    async def get_history(self, anima, *, thread_id="default", limit=50, before=None):
        self.history_before.append(before)
        return self.history

    async def get_active_stream(self, anima, *, thread_id="default"):
        return self.active_stream

    async def list_skills(self, anima, thread_id="default"):
        return {"anima": anima, "skills": []}

    async def chat_stream(self, anima, message, *, thread_id="default", resume=None, last_event_id=None, model=None):
        self.stream_calls.append((message, resume, last_event_id, model))
        while True:
            ev = await self.chat_queue.get()
            if ev is None:
                return
            yield ev

    async def ws_events(self):
        await self._ws.wait()
        yield {"type": "_ws_status", "data": {"connected": False}}


class FailingThenOkClient(FakeClient):
    """chat_stream raises on the first call, then streams remaining events."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.first_call = True

    async def chat_stream(self, anima, message, *, thread_id="default", resume=None, last_event_id=None, model=None):
        self.stream_calls.append((message, resume, last_event_id, model))
        if self.first_call:
            self.first_call = False
            raise AnimaWorksClientError("boom")
        while True:
            ev = await self.chat_queue.get()
            if ev is None:
                return
            yield ev


async def _pump(n=60):
    for _ in range(n):
        await asyncio.sleep(0)


def _session(anima="sora", **kw):
    s = new_session(anima=anima)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


# ── decide_reattach (pure) ──────────────────────────────


def test_decision_streaming_reattach():
    assert decide_reattach({"active": True, "status": "streaming"}, True) == "reattach"
    assert decide_reattach({"active": True, "status": "streaming"}, False) == "reattach"


def test_decision_complete():
    assert decide_reattach({"active": True, "status": "complete"}, True) == "render_final"
    assert decide_reattach({"active": True, "status": "complete"}, False) == "nothing"


def test_decision_none():
    assert decide_reattach({"active": False}, True) == "lost"
    assert decide_reattach({"active": False}, False) == "nothing"
    assert decide_reattach(None, False) == "nothing"
    assert decide_reattach(None, True) == "lost"


# ── Reattach pilot ─────────────────────────────────────


@pytest.mark.asyncio
async def test_reattach_renders_full_and_resumes_stream():
    client = FakeClient()
    client.active_stream = {
        "active": True,
        "status": "streaming",
        "response_id": "r1",
        "last_event_id": "r1:5",
        "full_text": "partial reply",
        "tool_history": [{"tool_name": "Bash", "tool_id": "t1", "input_summary": "in", "result_summary": "ok"}],
    }
    app = AnimaChatApp(
        client=client,
        anima_name="sora",
        session=_session(in_flight=True),
    )
    async with app.run_test() as _pt:
        await _pump()
        await _pump()

        # stream worker should have been started with resume info
        assert any((msg, r, le) == ("", "r1", "r1:5") for msg, r, le, _m in client.stream_calls), client.stream_calls
        assert app.current is not None
        assert "partial reply" in app.current._body

        # finish it
        await client.chat_queue.put(SseEvent("done", {"summary": "final"}))
        await client.chat_queue.put(None)
        await _pump()


@pytest.mark.asyncio
async def test_react_tach_noop_when_not_streaming():
    client = FakeClient()
    client.active_stream = {"active": True, "status": "complete", "full_text": "done body"}
    app = AnimaChatApp(client=client, anima_name="sora", session=_session(in_flight=True))
    async with app.run_test() as _pt:
        await _pump()
        await _pump()
        # complete + in_flight => render_final, no stream worker
        assert client.stream_calls == []
        assert app.current is not None
        assert "done body" in app.current._body
        assert app.session.in_flight is False


# ── History lazy-load ──────────────────────────────────


@pytest.mark.asyncio
async def test_load_older_history_uses_cursor():
    client = FakeClient()
    client.history = {
        "sessions": [{"messages": [{"role": "human", "content": "old"}]}],
        "has_more": False,
        "next_before": None,
    }
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as _pt:
        await _pump()
        # bootstrap already ran; force a cursor to simulate we have older pages
        app._history_cursor = "cur123"
        app._history_end = False
        await app.load_older_history()
        assert client.history_before[-1] == "cur123"


# ── Session save ───────────────────────────────────────


def test_session_saved_on_exit(tmp_path):
    from pathlib import Path

    client = FakeClient()
    sess = _session(in_flight=True)
    app = AnimaChatApp(
        client=client,
        anima_name="sora",
        session=sess,
        session_dir=Path(tmp_path),
    )
    app._write_session()
    f = Path(tmp_path) / f"{sess.session_id}.json"
    assert f.exists()


def test_session_dir_respected(tmp_path):
    from pathlib import Path

    client = FakeClient()
    sess = new_session(anima="sora")
    d = Path(tmp_path) / "nested"
    app = AnimaChatApp(client=client, anima_name="sora", session=sess, session_dir=d)
    save_session(sess, base_dir=d)
    app._write_session()
    assert (d / f"{sess.session_id}.json").exists()


# ── SSE reconnect (task 4) ───────────────────────────


@pytest.mark.asyncio
async def test_stream_reconnects_after_error_and_keeps_last_event_id():
    client = FailingThenOkClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    app._last_response_id = "r1"
    app._last_event_id = "r1:3"
    async with app.run_test():
        task = asyncio.create_task(app._chat_worker("hello"))
        await _pump()
        await client.chat_queue.put(SseEvent("done", {"summary": "ok"}))
        await client.chat_queue.put(None)
        await task
    # first call (failed) + one reconnect using the saved resume info
    assert len(client.stream_calls) == 2, client.stream_calls
    assert client.stream_calls[0][1:3] == (None, None)
    assert client.stream_calls[1][1:3] == ("r1", "r1:3")


# ── /keys command ──────────────────────────────────────


@pytest.mark.asyncio
async def test_keys_command_shows_bindings():
    client = FakeClient()
    seen: list[str] = []
    app = AnimaChatApp(client=client, anima_name="sora")
    orig = app.show_transient
    app.show_transient = lambda text: seen.append(str(text)) or orig(text)
    async with app.run_test() as _pt:
        await _pump()
        await app._handle_message("/keys")
        await _pump()
        combined = "\n".join(seen)
        assert "quit" in combined
        assert "interrupt" in combined
