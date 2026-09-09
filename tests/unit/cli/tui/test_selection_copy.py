# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Enter copies the mouse selection (tmux copy-mode style) and only
submits when nothing is selected."""

from __future__ import annotations

import asyncio

import pytest
from textual.geometry import Offset
from textual.selection import Selection
from textual.widgets import Static

from cli.tui.app import AnimaChatApp
from tests.unit.cli.tui.test_app import FakeClient, _pump


async def _ready(app, pilot):
    await _pump()
    await pilot.pause()
    app.input_container.focus_input()
    await pilot.pause()


async def _select(app, pilot, text: str, start: int = 0, end: int | None = None):
    """Mount a transcript line and select ``text[start:end]`` in it."""
    widget = Static(text, classes="assistant-text")
    await app.transcript.mount(widget)
    await pilot.pause()
    app.screen.selections = {
        widget: Selection(Offset(start, 0), Offset(len(text) if end is None else end, 0))
    }
    await pilot.pause()
    return widget


@pytest.mark.asyncio
async def test_enter_copies_the_selection_instead_of_submitting():
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await pilot.press("h", "i")
        await _select(app, pilot, "copy me please", 5, 13)

        await pilot.press("enter")
        await pilot.pause()

        assert app.clipboard == "me pleas"
        # Nothing was sent and the draft is still in the input box.
        assert client.messages == []
        assert app.input_container.input.text == "hi"


@pytest.mark.asyncio
async def test_selection_is_cleared_after_copying():
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await _select(app, pilot, "one line")

        await pilot.press("enter")
        await pilot.pause()

        assert app.clipboard == "one line"
        assert not app.screen.selections


@pytest.mark.asyncio
async def test_enter_submits_when_nothing_is_selected():
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await pilot.press("h", "e", "y")
        await pilot.press("enter")
        await _pump()

        assert client.messages == ["hey"]


@pytest.mark.asyncio
async def test_second_enter_submits_after_the_selection_is_consumed():
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await pilot.press("y", "o")
        await _select(app, pilot, "picked")

        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("enter")
        await _pump()

        assert app.clipboard == "picked"
        assert client.messages == ["yo"]


@pytest.mark.asyncio
async def test_input_selection_does_not_hijack_enter():
    """A selection inside the input box is the TextArea's own; Enter sends."""
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await pilot.press("d", "r", "a", "f", "t")
        chat_input = app.input_container.input
        app.screen.selections = {chat_input: Selection(Offset(0, 0), Offset(5, 0))}
        await pilot.pause()

        await pilot.press("enter")
        await _pump()

        assert client.messages == ["draft"]


@pytest.mark.asyncio
async def test_copy_reports_on_the_status_bar():
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await _select(app, pilot, "abcdef")

        await pilot.press("enter")
        await pilot.pause()

        assert app.status_bar.right_hint == "Copied 6 chars"


@pytest.mark.asyncio
async def test_focus_returns_to_the_input_after_copying():
    """Starting a selection moves focus to the scrollable transcript."""
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await _select(app, pilot, "grab this")
        app.screen.set_focus(app.transcript)
        await pilot.pause()

        await pilot.press("enter")
        await pilot.pause()

        assert app.clipboard == "grab this"
        assert app.focused is app.input_container.input


@pytest.mark.asyncio
async def test_enter_after_a_click_still_submits():
    """A click in the transcript takes focus away; Enter must still send."""
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await pilot.press("h", "i")
        app.screen.set_focus(app.transcript)
        await pilot.pause()

        await pilot.press("enter")
        await _pump()

        assert client.messages == ["hi"]


@pytest.mark.asyncio
async def test_multiline_selection_is_copied_whole():
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        widget = Static("first\nsecond\nthird", classes="assistant-text")
        await app.transcript.mount(widget)
        await pilot.pause()
        app.screen.selections = {widget: Selection(Offset(0, 0), Offset(5, 2))}
        await pilot.pause()

        await pilot.press("enter")
        await pilot.pause()

        assert app.clipboard == "first\nsecond\nthird"
        assert app.status_bar.right_hint == "Copied 3 lines"


@pytest.mark.asyncio
async def test_status_hint_is_restored_after_the_flash():
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        app.status_bar.set_state(right_hint="Esc: interrupt")
        app.status_bar.flash("Copied 3 chars", duration=0.05)
        assert app.status_bar.right_hint == "Copied 3 chars"

        await asyncio.sleep(0.2)
        await pilot.pause()
        assert app.status_bar.right_hint == "Esc: interrupt"


@pytest.mark.asyncio
async def test_escape_still_closes_the_palette():
    """Escape is a priority binding now; the palette must keep it first."""
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await pilot.press("/")
        await pilot.pause()
        assert app.palette_is_open()

        await pilot.press("escape")
        await pilot.pause()

        assert not app.palette_is_open()
        assert client.interrupt_calls == 0


@pytest.mark.asyncio
async def test_escape_clears_the_selection_before_interrupting():
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        app.busy = True
        await _select(app, pilot, "still selecting")

        await pilot.press("escape")
        await pilot.pause()

        assert not app.screen.selections
        assert client.interrupt_calls == 0
        assert app.busy is True

        await pilot.press("escape")
        await _pump()
        assert client.interrupt_calls == 1
