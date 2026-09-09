# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest

from cli.tui.app import AnimaChatApp
from cli.tui.client import AnimaWorksClientError
from cli.tui.sse import SseEvent
from cli.tui.widgets import ToolCard


class FakeClient:
    """A stub AnimaWorksClient with the same async interface."""

    def __init__(self, *, animas=None, history=None, ws_events=None):
        self.animas = animas or [{"name": "sora", "status": "idle"}]
        self.history = history or {"sessions": []}
        self._ws_events = ws_events or []
        self.interrupt_calls = 0
        self.messages = []
        self.active_stream = {"active": False}
        self.chat_error: AnimaWorksClientError | None = None
        self.chat_queue: asyncio.Queue = asyncio.Queue()
        self.reattach_calls: list = []

    async def list_animas(self):
        return self.animas

    async def get_history(self, anima, *, thread_id="default", limit=50, before=None):
        return self.history

    async def get_active_stream(self, anima, *, thread_id="default"):
        return self.active_stream

    async def interrupt(self, anima, *, thread_id):
        self.interrupt_calls += 1
        return {"status": "interrupted"}

    async def chat_stream(self, anima, message, *, thread_id="default", resume=None, last_event_id=None, model=None):
        self.messages.append(message)
        self.reattach_calls.append((resume, last_event_id))
        if self.chat_error is not None:
            raise self.chat_error
        while True:
            ev = await self.chat_queue.get()
            if ev is None:
                return
            yield ev

    async def ws_events(self):
        for ev in self._ws_events:
            yield ev
        await asyncio.Event().wait()


def _history_with(*messages):
    msgs = [{"ts": "1", "role": "human", "content": content} for content in messages]
    return {"sessions": [{"messages": msgs}]}


async def _pump(n=40):
    for _ in range(n):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_startup_shows_history():
    client = FakeClient(history=_history_with("Hello from history"))
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as _:
        await _pump()
        from cli.tui.widgets import HumanTurn

        turns = list(app.query(HumanTurn))
        assert len(turns) >= 1
        assert any("Hello from history" in t._text for t in turns)


@pytest.mark.asyncio
async def test_streaming_renders_text_and_tool_card():
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _pump()
        app.input_container.focus_input()
        await _pump()
        await pilot.press("h", "e", "l", "l", "o")
        await pilot.press("enter")
        await _pump()

        assert client.messages == ["hello"]

        # Feed a full streaming sequence.
        await client.chat_queue.put(SseEvent("text_delta", {"text": "part1 "}))
        await client.chat_queue.put(SseEvent("text_delta", {"text": "part2"}))
        await client.chat_queue.put(SseEvent("tool_start", {"tool_name": "Bash", "tool_id": "t1"}))
        await client.chat_queue.put(
            SseEvent("tool_end", {"tool_id": "t1", "tool_name": "Bash", "result_summary": "ok"})
        )
        await client.chat_queue.put(SseEvent("done", {"summary": "final summary"}))
        await client.chat_queue.put(None)
        await _pump()

        assert app.current is not None
        assert "final summary" in app.current._body
        assert len(list(app.query(ToolCard))) == 1


@pytest.mark.asyncio
async def test_tool_card_stays_between_surrounding_text():
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _pump()
        app.input_container.focus_input()
        await pilot.press("h", "i", "enter")
        await _pump()

        await client.chat_queue.put(SseEvent("text_delta", {"text": "before"}))
        await client.chat_queue.put(SseEvent("tool_start", {"tool_name": "Read", "tool_id": "t1"}))
        await client.chat_queue.put(SseEvent("tool_end", {"tool_id": "t1", "tool_name": "Read"}))
        await client.chat_queue.put(SseEvent("text_delta", {"text": "after"}))
        await client.chat_queue.put(SseEvent("done", {"summary": "beforeafter"}))
        await client.chat_queue.put(None)
        await _pump()

        assert app.current is not None
        children = list(app.current.timeline.children)
        assert len(children) == 3
        assert children[0].has_class("assistant-text")
        assert isinstance(children[1], ToolCard)
        assert children[2].has_class("assistant-text")
        assert children[0].content.plain == "before"
        assert children[2].content.plain == "after"


@pytest.mark.asyncio
async def test_escape_calls_interrupt_when_busy():
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as _:
        await _pump()
        app.busy = True
        app.action_maybe_interrupt()
        await _pump()
        assert client.interrupt_calls == 1
        assert app.busy is False


@pytest.mark.asyncio
async def test_escape_does_nothing_when_idle():
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as _:
        await _pump()
        app.busy = False
        app.action_maybe_interrupt()
        await _pump()
        assert client.interrupt_calls == 0


@pytest.mark.asyncio
async def test_thinking_toggle():
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as _:
        await _pump()
        assert app.show_thinking is False
        await app._handle_message("/thinking")
        assert app.show_thinking is True
        await app._handle_message("/thinking")
        assert app.show_thinking is False


@pytest.mark.asyncio
async def test_unknown_command_not_sent_to_server():
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as _:
        await _pump()
        await app._handle_message("/definitely-not-a-command")
        await _pump()
        assert client.messages == []


# ── Phase 1 review fixes ─────────────────────────────────


def _rich_plain(widget) -> str:
    return getattr(widget.text.renderable, "plain", str(widget.text.renderable))


@pytest.mark.asyncio
async def test_widgets_have_height_after_history_and_stream():
    client = FakeClient(history=_history_with("Line one of history\nLine two longer"))
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _pump()
        await pilot.pause()

        from cli.tui.widgets import HumanTurn

        turns = list(app.query(HumanTurn))
        assert turns, "no human turn rendered"
        assert turns[0].region.height >= 2, turns[0].region

        # stream a reply (multi-line) then finish
        app.input_container.focus_input()
        await pilot.press("h", "i")
        await pilot.press("enter")
        await _pump()
        await client.chat_queue.put(SseEvent("text_delta", {"text": "reply first line\nreply second line"}))
        await client.chat_queue.put(SseEvent("done", {"summary": "final first\nfinal second"}))
        await client.chat_queue.put(None)
        await _pump()
        await pilot.pause()

        assert app.current is not None
        assert app.current.region.height >= 2, app.current.region
        # A two-line reply with no tool cards must not balloon (regression: 1fr tools container)
        assert app.current.region.height <= 4, app.current.region


@pytest.mark.asyncio
async def test_input_grows_with_newlines_and_shrinks_on_submit():
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _pump()
        await pilot.pause()
        app.input_container.focus_input()
        await pilot.pause()

        h0 = app.input_container.input.region.height
        for _ in range(3):
            await pilot.press("shift+enter")
        await pilot.pause()
        h_grown = app.input_container.input.region.height
        assert h_grown > h0, (h0, h_grown)

        await pilot.press("h", "i")
        await pilot.pause()
        await pilot.press("enter")
        await _pump()
        await pilot.pause()
        assert app.input_container.input.region.height == h0
        await client.chat_queue.put(None)


@pytest.mark.asyncio
async def test_status_bar_is_below_input():
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _pump()
        await pilot.pause()
        status = app.query_one("#status")
        inp = app.input_container.input
        assert status.region.y > inp.region.y, (status.region, inp.region)


@pytest.mark.asyncio
async def test_system_history_rows_not_rendered():
    history = {
        "sessions": [
            {
                "messages": [
                    {"role": "system", "content": "定期巡回開始"},
                    {"role": "system", "content": "記憶統合エラー"},
                    {"role": "human", "content": "hello there"},
                ],
            }
        ]
    }
    client = FakeClient(history=history)
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _pump()
        await pilot.pause()
        from cli.tui.widgets import HumanTurn

        assert len(list(app.query(HumanTurn))) == 1
        assert list(app.query(".transient")) == []


@pytest.mark.asyncio
async def test_emotion_comment_stripped_from_stream():
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _pump()
        app.input_container.focus_input()
        await pilot.press("h", "i")
        await pilot.press("enter")
        await _pump()
        await client.chat_queue.put(SseEvent("text_delta", {"text": "hello <!-- emotion:"}))
        await client.chat_queue.put(SseEvent("text_delta", {"text": ' {"emotion": "smile"} --> there'}))
        await client.chat_queue.put(None)
        await _pump()

        assert app.current is not None
        plain = app.current._display_body()
        assert "hello" in plain
        assert "there" in plain
        assert "<!--" not in plain
        assert "emotion" not in plain


@pytest.mark.asyncio
async def test_busy_not_cleared_by_followup_idle_ws():
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as _:
        await _pump()
        app.busy = True
        app.handle_ws({"type": "anima.status", "data": {"name": "sora", "status": "idle"}})
        assert app.busy is True


@pytest.mark.asyncio
async def test_busy_cleared_when_stream_ends_with_exception():
    client = FakeClient()
    client.chat_error = AnimaWorksClientError("boom")
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _pump()
        app.input_container.focus_input()
        await pilot.press("h", "i")
        await pilot.press("enter")
        await _pump()
        assert app.busy is False


@pytest.mark.asyncio
async def test_thinking_buffered_hidden_then_toggle_shows():
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _pump()
        app.input_container.focus_input()
        await pilot.press("h", "i")
        await pilot.press("enter")
        await _pump()
        await client.chat_queue.put(SseEvent("thinking_delta", {"text": "secret chain of thought"}))
        await client.chat_queue.put(None)
        await _pump()
        await pilot.pause()

        assert app.current is not None
        tb = app.current.thinking
        assert tb is not None, "thinking block should be buffered even when hidden"
        assert tb.display is False
        assert "secret chain of thought" in tb._body

        await app._handle_message("/thinking")
        await pilot.pause()
        assert app.show_thinking is True
        assert tb.display is True


@pytest.mark.asyncio
async def test_history_assistant_body_is_rendered_and_emotion_stripped():
    history = {
        "sessions": [
            {
                "messages": [
                    {"ts": "1", "role": "human", "content": "hi"},
                    {
                        "ts": "2",
                        "role": "assistant",
                        "content": 'Reply body.\n\n<!-- emotion: {"emotion": "smile"} -->',
                    },
                ]
            }
        ]
    }
    client = FakeClient(history=history)
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _pump()
        await pilot.pause()
        from cli.tui.widgets.transcript import AssistantBlock

        blocks = list(app.query(AssistantBlock))
        assert len(blocks) == 1
        rendered = "".join(
            segment.text
            for y in range(blocks[0].text.size.height)
            for segment in blocks[0].text.render_line(y)
        )
        assert "Reply body." in rendered
        assert "emotion" not in rendered
        assert blocks[0].region.height <= 3, blocks[0].region
