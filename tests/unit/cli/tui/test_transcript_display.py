# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Who is speaking, and staying on the newest line while they speak."""

from __future__ import annotations

import asyncio

import pytest

from cli.tui.app import AnimaChatApp
from cli.tui.sse import SseEvent
from cli.tui.widgets import AssistantBlock, HumanTurn, SystemNote
from tests.unit.cli.tui.test_app import FakeClient


async def _pump(n=40):
    for _ in range(n):
        await asyncio.sleep(0)


async def _settle(pilot, rounds=8):
    """Let deferred layout and the history-fill worker run to a stop."""
    for _ in range(rounds):
        await pilot.pause()
        await asyncio.sleep(0.02)


def _rendered_plain(widget) -> str:
    return "".join(
        segment.text
        for y in range(widget.size.height)
        for segment in widget.render_line(y)
    )


def _history(*pairs):
    return {"sessions": [{"messages": [{"ts": "1", "role": role, "content": text} for role, text in pairs]}]}


@pytest.mark.asyncio
async def test_assistant_name_gets_its_own_row():
    client = FakeClient(history=_history(("assistant", "本文です。")))
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _pump()
        await pilot.pause()

        block = next(iter(app.query(AssistantBlock)))
        assert block.label_widget.render().plain == "sora:"
        # The name must not be buried at the start of the body any more.
        assert "sora:" not in _rendered_plain(block.text)


@pytest.mark.asyncio
async def test_human_turn_is_painted_so_it_stands_apart():
    client = FakeClient(history=_history(("human", "こんにちは")))
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _pump()
        await pilot.pause()

        turn = next(iter(app.query(HumanTurn)))
        assert turn.styles.background != app.screen.styles.background


@pytest.mark.asyncio
async def test_assistant_markdown_is_rendered_not_shown_as_markup():
    client = FakeClient(history=_history(("assistant", "結論は**はい**です。\n\n- 一つ目\n- 二つ目")))
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _pump()
        await pilot.pause()

        block = next(iter(app.query(AssistantBlock)))
        plain = _rendered_plain(block.text)
        assert "**" not in plain and "- 一つ目" not in plain
        assert "はい" in plain and "• 一つ目" in plain


@pytest.mark.asyncio
async def test_streaming_keeps_the_newest_line_in_view():
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test(size=(80, 20)) as pilot:
        await _pump()
        app.input_container.focus_input()
        await pilot.press("h", "i")
        await pilot.press("enter")
        await _pump()

        for i in range(60):
            await client.chat_queue.put(SseEvent("text_delta", {"text": f"行{i}\n"}))
        await client.chat_queue.put(None)
        await _pump()
        await pilot.pause()
        await pilot.pause()

        transcript = app.transcript
        assert transcript.max_scroll_y > 0, "the reply did not overflow the pane"
        assert transcript.scroll_y == transcript.max_scroll_y


@pytest.mark.asyncio
async def test_scrolling_up_stops_the_transcript_from_jumping_back():
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test(size=(80, 20)) as pilot:
        await _pump()
        app.input_container.focus_input()
        await pilot.press("h", "i")
        await pilot.press("enter")
        await _pump()

        for i in range(60):
            await client.chat_queue.put(SseEvent("text_delta", {"text": f"行{i}\n"}))
        await _pump()
        await pilot.pause()

        app.transcript.scroll_to(y=5, animate=False, force=True)
        await pilot.pause()
        assert app.transcript.scroll_y == 5

        for i in range(60, 90):
            await client.chat_queue.put(SseEvent("text_delta", {"text": f"行{i}\n"}))
        await client.chat_queue.put(None)
        await _pump()
        await pilot.pause()

        assert app.transcript.scroll_y == 5, "the reader was dragged back to the bottom"


# ── system-role entries ───────────────────────────────────
class PagingClient(FakeClient):
    """A client whose history is served in pages, oldest last."""

    def __init__(self, pages):
        super().__init__()
        self.pages = pages
        self.requested: list[str | None] = []

    async def get_history(self, anima, *, thread_id="default", limit=50, before=None):
        self.requested.append(before)
        index = 0 if before is None else int(before)
        page = self.pages[index]
        has_more = index + 1 < len(self.pages)
        return {
            "sessions": [{"messages": page}],
            "has_more": has_more,
            "next_before": str(index + 1) if has_more else None,
        }


def _system(content, **extra):
    return {"ts": "1", "role": "system", "content": content, **extra}


def _colours(widget) -> set[int | None]:
    """The ANSI slot every visible run of a widget is drawn in (8 = grey).

    Read off the rendered lines rather than the renderable: that is where
    styles are resolved, and it is what the terminal is handed.
    """
    seen: set[int | None] = set()
    for y in range(widget.size.height):
        for segment in widget.render_line(y):
            if not segment.text.strip():
                continue
            color = None if segment.style is None else segment.style.color
            seen.add(None if color is None else color.number)
    return seen


@pytest.mark.asyncio
async def test_system_entries_are_shown_faintly_with_their_source():
    client = FakeClient(
        history={
            "sessions": [
                {
                    "messages": [
                        _system("ハートビートを完了しました。", source_key="heartbeat"),
                        _system("確認をお願いします。", source_key="call_human", subject="systemd failedの増加"),
                    ]
                }
            ]
        }
    )
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _pump()
        await pilot.pause()

        notes = list(app.query(SystemNote))
        assert len(notes) == 2, "system entries were dropped from the transcript"
        assert notes[0].label_widget.render().plain == "system · heartbeat"
        assert notes[1].label_widget.render().plain == "system · call_human — systemd failedの増加"

        assert "ハートビートを完了しました。" in _rendered_plain(notes[0].message)
        # Grey, not the `dim` attribute: terminals are free to ignore SGR
        # 2, and on the ansi themes nothing turns it into a colour. The
        # label counts too — `dim bold` came out as plain bold, brighter
        # rather than fainter, wherever bold wins over faint.
        assert _colours(notes[0].message) == {8}
        assert _colours(notes[0].label_widget) == {8}


@pytest.mark.asyncio
async def test_a_thread_of_only_system_entries_is_not_blank():
    client = FakeClient(history={"sessions": [{"messages": [_system("cron: 巡回", source_key="cron")]}]})
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        await _pump()
        await pilot.pause()

        assert list(app.query(SystemNote)), "the transcript opened empty"


@pytest.mark.asyncio
async def test_older_pages_load_until_the_transcript_can_scroll():
    # The newest page is one short line: without a scrollbar there is no
    # way to reach the conversation buried behind it.
    pages = [
        [_system("cron: 巡回", source_key="cron")],
        [{"ts": "1", "role": "human", "content": f"古い発言 {i}"} for i in range(40)],
    ]
    client = PagingClient(pages)
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test(size=(80, 20)) as pilot:
        await _pump()
        await _settle(pilot)

        assert client.requested == [None, "1"], client.requested
        assert app.transcript.max_scroll_y > 0, "still no scrollbar, so history stays unreachable"
        assert list(app.query(HumanTurn)), "the older conversation never arrived"


@pytest.mark.asyncio
async def test_filling_stops_when_the_history_runs_out():
    client = PagingClient([[_system("cron: 巡回", source_key="cron")]])
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test(size=(80, 20)) as pilot:
        await _pump()
        await _settle(pilot)

        assert client.requested == [None], client.requested
