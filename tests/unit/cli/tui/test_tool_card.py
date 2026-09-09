# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the tool card header: the dimmed one-line argument preview."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from cli.tui.widgets.tool_card import ToolCard, format_input_summary


class _Harness(App):
    def __init__(self, card: ToolCard) -> None:
        super().__init__()
        self._card = card

    def compose(self) -> ComposeResult:
        yield self._card


def _header_text(card: ToolCard) -> str:
    return card.header.content.plain


class TestFormatInputSummary:

    def test_dict_repr_becomes_key_value_line(self):
        raw = "{'command': 'ls -la', 'description': 'list'}"
        assert format_input_summary(raw) == "command=ls -la, description=list"

    def test_truncated_repr_falls_back_to_raw(self):
        raw = "{'command': 'ls -la"
        assert format_input_summary(raw) == "{'command': 'ls -la"

    def test_collapses_newlines(self):
        assert format_input_summary("a\n   b") == "a b"

    def test_clips_to_preview_limit(self):
        assert len(format_input_summary("x" * 500)) == 200


@pytest.mark.asyncio
class TestToolCardHeader:

    async def test_detail_shows_on_header_while_collapsed(self):
        card = ToolCard("Bash", "t1")
        async with _Harness(card).run_test():
            card.add_detail("ls -la /tmp")
            assert _header_text(card) == "▸ Bash  …  ls -la /tmp"
            assert card.expanded is False

    async def test_result_summary_follows_the_preview(self):
        card = ToolCard("Bash", "t1")
        async with _Harness(card).run_test():
            card.add_detail("ls -la")
            card.finish(result_summary="12 lines")
            assert _header_text(card) == "▸ Bash  ✓  ls -la  12 lines"

    async def test_input_summary_used_when_no_detail_arrived(self):
        card = ToolCard("Grep", "t2")
        async with _Harness(card).run_test():
            card.finish(result_summary="3 hits", input_summary="{'pattern': 'foo'}")
            assert _header_text(card) == "▸ Grep  ✓  pattern=foo  3 hits"

    async def test_detail_wins_over_input_summary(self):
        card = ToolCard("Grep", "t3")
        async with _Harness(card).run_test():
            card.add_detail("foo in /src")
            card.finish(input_summary="{'pattern': 'foo', 'path': '/src'}")
            assert _header_text(card) == "▸ Grep  ✓  foo in /src"

    async def test_multiline_detail_collapses_to_one_line(self):
        card = ToolCard("Bash", "t4")
        async with _Harness(card).run_test():
            card.add_detail("echo one\necho two")
            assert _header_text(card) == "▸ Bash  …  echo one echo two"

    async def test_long_preview_is_ellipsised_to_the_card_width(self):
        card = ToolCard("Bash", "t7")
        async with _Harness(card).run_test(size=(40, 8)):
            card.add_detail("git log --oneline --graph --decorate --all")
            card.finish(result_summary="42 lines")
            rendered = card.header.render_line(0).text
            assert len(rendered) == 40
            assert rendered.startswith("▸ Bash  ✓  git log ")
            assert rendered.endswith("…")
            # The card is a single row: nothing wrapped onto a second one.
            assert card.header.render_line(1).text.strip() == ""

    async def test_error_marker_keeps_preview(self):
        card = ToolCard("Bash", "t5")
        async with _Harness(card).run_test():
            card.add_detail("false")
            card.finish(is_error=True)
            assert _header_text(card) == "▸ Bash  ✗  false error"

    async def test_toggle_shows_full_detail(self):
        card = ToolCard("Bash", "t6")
        async with _Harness(card).run_test():
            card.add_detail("line1\nline2")
            card.toggle()
            assert card.expanded is True
            assert card.detail.content.plain == "line1\nline2"
