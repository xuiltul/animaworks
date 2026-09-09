# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""The transcript renders markdown instead of showing its markup."""

from __future__ import annotations

import re

from rich.console import Console
from rich.table import Table
from rich.text import Text

from cli.tui.markdown import render_inline, render_markdown

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(source: str, width: int = 60) -> str:
    console = Console(width=width, no_color=True, legacy_windows=False)
    with console.capture() as capture:
        console.print(render_markdown(source))
    return _ANSI_RE.sub("", capture.get())


def _styles(source: str) -> dict[str, str]:
    """Map each styled run of an inline render to its style."""
    text = render_inline(source)
    return {text.plain[span.start : span.end]: str(span.style) for span in text.spans}


# ── inline ────────────────────────────────────────────────
def test_bold_markup_is_not_shown_literally():
    text = render_inline("結論から言うと**はい、いい面談でした**")
    assert "**" not in text.plain
    assert "はい、いい面談でした" in text.plain
    assert any("bold" in str(span.style) for span in text.spans)


def test_italic_applies_after_japanese_characters():
    # `*` may emphasise inside a word, and Japanese has no spaces to lean on.
    styles = _styles("抵当権1.4億で*マイナス評価*の可能性")
    assert "italic" in styles.get("マイナス評価", "")


def test_underscores_inside_an_identifier_are_left_alone():
    text = render_inline("`snake_case_name` と some_var_here")
    assert "snake_case_name" in text.plain
    assert "some_var_here" in text.plain


def test_inline_code_keeps_markup_inside_it_literal():
    styles = _styles("値は `a ** b` です")
    assert "a ** b" in styles


def test_links_render_the_label_and_keep_the_url():
    text = render_inline("詳細は [ドキュメント](https://example.com/doc) を参照")
    assert "ドキュメント" in text.plain
    assert "https://example.com/doc" not in text.plain
    assert any(span.style.link == "https://example.com/doc" for span in text.spans)


# ── blocks ────────────────────────────────────────────────
def test_headings_lose_their_hashes():
    out = _plain("## ①国外転出時課税の評価方法\n本文です。")
    assert "#" not in out
    assert "①国外転出時課税の評価方法" in out


def test_bullets_become_glyphs_and_nest():
    out = _plain("- 一つ目\n- 二つ目\n  - ネスト")
    assert "• 一つ目" in out
    assert "◦ ネスト" in out
    assert "- 一つ目" not in out


def test_ordered_lists_keep_their_numbers():
    out = _plain("1. 初鹿さんへの指示\n2. CUCへの連絡文面")
    assert "1. 初鹿さんへの指示" in out
    assert "2. CUCへの連絡文面" in out


def test_tables_render_as_a_table_not_pipes():
    source = "| 誰の宿題 | 内容 |\n|---|---|\n| 森さん | 評価詳細 |"
    blocks = render_markdown(source)
    tables = [r for r in getattr(blocks, "renderables", [blocks]) if isinstance(r, Table)]
    assert tables, "table row was left as raw markdown"
    out = _plain(source)
    assert "|---|" not in out
    assert "誰の宿題" in out and "森さん" in out


def test_code_fence_is_shown_without_its_backticks():
    out = _plain('```python\ndef hello():\n    return "hi"\n```')
    assert "```" not in out
    assert "def hello():" in out
    assert "│" in out


def test_unclosed_code_fence_still_renders_while_streaming():
    out = _plain("前置き\n\n```python\ndef hello():")
    assert "```" not in out
    assert "def hello():" in out


def test_blockquotes_get_a_gutter():
    out = _plain("> 引用のテスト")
    assert ">" not in out
    assert "▌" in out and "引用のテスト" in out


def test_plain_text_survives_untouched():
    assert render_markdown("ふつうの文章です。").plain == "ふつうの文章です。"


def test_latin_word_next_to_japanese_moves_whole_to_the_next_line():
    source = "実測が出ました。sumireの遅さを確認します。"
    rendered = render_markdown(source)
    assert isinstance(rendered, Text)
    lines = rendered.wrap(Console(width=18), 18)
    out = "\n".join(line.plain for line in lines)
    assert "su\nmire" not in out
    assert "sum\nire" not in out
    assert "sumire" in out


def test_blank_lines_between_paragraphs_are_kept_but_collapsed():
    rendered = render_markdown("一段落目。\n\n\n\n二段落目。")
    assert isinstance(rendered, Text)
    assert rendered.plain == "一段落目。\n\n二段落目。"


def test_muted_rendering_drops_the_accent_colours():
    normal = render_inline("値は `code` です")
    muted = render_inline("値は `code` です", muted=True)
    assert any("magenta" in str(span.style) for span in normal.spans)
    assert not any("magenta" in str(span.style) for span in muted.spans)
    assert all("bright_black" in str(span.style) for span in muted.spans)
    assert "bright_black" in str(muted.style)


def _colours(renderable, width: int = 60) -> set[int | None]:
    """The ANSI slot every visible run of a renderable is drawn in (8 = grey)."""
    console = Console(width=width, legacy_windows=False)
    seen: set[int | None] = set()
    for segment in console.render(renderable):
        if not segment.text.strip():
            continue
        color = None if segment.style is None else segment.style.color
        seen.add(None if color is None else color.number)
    return seen


def test_muted_text_is_grey_rather_than_leaning_on_the_faint_attribute():
    # `dim` is only a request: terminals are free to ignore SGR 2, and
    # this app runs on the ansi themes, so nothing turns it into a
    # colour. Bold used to win outright and pull the run back to full
    # contrast; emphasis has to stay grey too.
    muted = render_markdown("見出し\n\n**強い**注意と `code` と [link](https://x)", muted=True)
    assert _colours(muted) == {8}
