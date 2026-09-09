# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Input newline alternatives, Tab-completion caret, palette highlight,
the blinking thinking marker, the wrapping input box and the model on the
status line (2026-09-07 TUI feedback)."""

from __future__ import annotations

import asyncio

import pytest

from cli.tui.app import AnimaChatApp
from cli.tui.sse import SseEvent
from cli.tui.widgets.thinking import ThinkingBlock
from tests.unit.cli.tui.test_app import FakeClient, _pump


async def _ready(app, pilot):
    await _pump()
    await pilot.pause()
    app.input_container.focus_input()
    await pilot.pause()


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["shift+enter", "ctrl+j"])
async def test_newline_keys_insert_newline_without_submitting(key):
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await pilot.press("a", key, "b")
        await pilot.pause()
        assert app.input_container.input.text == "a\nb"
        assert client.messages == []


@pytest.mark.asyncio
async def test_backslash_enter_continues_line():
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await pilot.press("a", "backslash", "enter", "b")
        await pilot.pause()
        assert app.input_container.input.text == "a\nb"
        assert client.messages == []
        # A backslash not directly before the caret is ordinary text.
        await pilot.press("enter")
        await _pump()
        assert client.messages == ["a\nb"]
        await client.chat_queue.put(None)


@pytest.mark.asyncio
async def test_tab_completion_leaves_caret_at_end():
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await pilot.press("/", "t", "h", "r", "e", "a", "d")
        await pilot.pause()
        assert app.palette.is_open
        await pilot.press("tab")
        await pilot.pause()
        widget = app.input_container.input
        assert widget.text.startswith("/thread")
        assert widget.cursor_location == widget.document.end
        assert not app.palette.is_open


@pytest.mark.asyncio
async def test_palette_highlight_is_reverse_video():
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await pilot.press("/")
        await pilot.pause()
        assert app.palette.is_open

        def reversed_rows():
            return [
                y
                for y in range(min(app.palette.size.height, len(app.palette._items)))
                if all(
                    seg.style is not None and seg.style.reverse
                    for seg in app.palette.render_line(y)
                    if seg.text.strip()
                )
            ]

        assert reversed_rows() == [0]
        await pilot.press("down")
        await pilot.pause()
        assert app.palette.highlighted == 1
        assert reversed_rows() == [1]


@pytest.mark.asyncio
async def test_thinking_marker_blinks_only_while_thinking():
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        app.show_thinking = True
        await pilot.press("h", "i", "enter")
        await _pump()
        await pilot.pause()
        await app.handle_sse(SseEvent(event="thinking_start", data={}))
        await app.handle_sse(SseEvent(event="thinking_delta", data={"text": "hmm"}))
        await pilot.pause()
        block = app.query_one(ThinkingBlock)
        assert block.is_active and block._timer is not None
        marks = set()
        for _ in range(6):
            await asyncio.sleep(ThinkingBlock.BLINK_INTERVAL / 2)
            marks.add(str(block.header.content)[0])
        assert marks == {"▸", " "}, marks
        await app.handle_sse(SseEvent(event="thinking_end", data={}))
        await pilot.pause()
        assert not block.is_active and block._timer is None
        assert str(block.header.content).startswith("▸")
        await client.chat_queue.put(None)


@pytest.mark.asyncio
async def test_input_grows_with_wrapped_rows_not_document_lines():
    """A long single line wraps into visible rows instead of scrolling away."""
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test(size=(100, 32)) as pilot:
        await _ready(app, pilot)
        chat_input = app.input_container.input
        chat_input.insert("これは折り返されるべき長い一行の入力です。" * 3)
        await pilot.pause()
        assert chat_input.document.line_count == 1
        rows = chat_input.wrapped_document.height
        assert rows > 1, "the sample text is meant to wrap"
        assert chat_input.size.height == rows

        # Never past `max_lines`; from there the box scrolls.
        chat_input.insert("さらに続く長い文章。" * 30)
        await pilot.pause()
        assert chat_input.size.height == chat_input.max_lines

        # A narrower terminal wraps harder, so the box grows again.
        chat_input.clear()
        chat_input.insert("あ" * 40)
        await pilot.pause()
        wide = chat_input.size.height
        await pilot.resize_terminal(60, 32)
        await pilot.pause()
        assert chat_input.size.height > wide

        # Submitting empties it back to one row.
        chat_input.clear()
        await pilot.pause()
        assert chat_input.size.height == 1


@pytest.mark.asyncio
async def test_status_line_names_the_model_in_use():
    """The model replaces `ws: connected`, which never said anything."""
    client = FakeClient(animas=[{"name": "sora", "status": "idle", "model": "claude-opus-4-6"}])
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        app.status_bar.set_state(connected=True)
        await pilot.pause()

        # No override: the anima's own model, marked as the default.
        line = app.status_bar.render().plain
        assert "model:claude-opus-4-6 (default)" in line
        assert "ws:" not in line

        app.set_chat_model("codex/gpt-5.6-sol")
        await pilot.pause()
        assert "model:codex/gpt-5.6-sol" in app.status_bar.render().plain

        app.set_chat_model("")
        await pilot.pause()
        assert "model:claude-opus-4-6 (default)" in app.status_bar.render().plain

        # A dropped websocket is the one connection state worth a slot.
        app.status_bar.set_state(connected=False)
        await pilot.pause()
        assert "ws: disconnected" in app.status_bar.render().plain
