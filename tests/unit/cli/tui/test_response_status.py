# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest

from cli.tui.app import AnimaChatApp
from cli.tui.sse import SseEvent
from cli.tui.widgets.response_status import ResponseStatus, format_elapsed
from cli.tui.widgets.transcript import AssistantBlock
from tests.unit.cli.tui.test_app import FakeClient


def test_elapsed_time_keeps_minutes() -> None:
    assert format_elapsed(0) == "0m 0s"
    assert format_elapsed(125.9) == "2m 5s"


@pytest.mark.asyncio
async def test_response_indicator_runs_until_done() -> None:
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        for _ in range(40):
            await asyncio.sleep(0)
        await app.send_message("hello")
        await pilot.pause()

        indicator = app.query_one(ResponseStatus)
        assert indicator.is_active
        assert indicator.display
        assert "Thinking... (0m 0s)" in indicator.render().plain
        assert indicator.render().plain.startswith("●")
        indicator._tick()
        assert indicator.render().plain.startswith(" ")
        assert app.input_container.styles.border.top[0] == "solid"
        assert app.input_container.styles.border.bottom[0] == "solid"
        assert indicator.region.bottom == app.input_container.region.y

        await app.handle_sse(SseEvent("done", {"summary": "finished"}))
        await pilot.pause()
        assert not indicator.is_active
        assert not indicator.display
        await client.chat_queue.put(None)


@pytest.mark.asyncio
async def test_context_usage_stays_in_status() -> None:
    client = FakeClient()
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test() as pilot:
        for _ in range(40):
            await asyncio.sleep(0)
        await app.handle_sse(
            SseEvent(
                "context_update",
                {"context_usage_ratio": 0.234, "input_tokens": 29952, "context_window": 128000},
            )
        )
        await pilot.pause()
        assert "ctx:23%" in app.status_bar.render().plain


@pytest.mark.asyncio
async def test_mixed_japanese_text_does_not_split_a_latin_word() -> None:
    source = "実測が出ました。sumireの遅さの構造を確認して、すぐ手を打ちます。"
    client = FakeClient(history={"sessions": [{"messages": [{"role": "assistant", "content": source}]}]})
    app = AnimaChatApp(client=client, anima_name="sora")
    async with app.run_test(size=(47, 30)) as pilot:
        for _ in range(40):
            await asyncio.sleep(0)
        await pilot.pause()
        block = app.query_one(AssistantBlock)
        lines = ["".join(segment.text for segment in block.text.render_line(y)) for y in range(block.text.size.height)]
        for first, second in zip(lines, lines[1:], strict=False):
            assert not (first.rstrip().endswith("su") and second.lstrip().startswith("mire"))
