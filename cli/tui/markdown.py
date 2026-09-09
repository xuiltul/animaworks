# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Markdown → Rich renderables, tuned for a terminal transcript.

Rich ships its own ``Markdown`` renderable, but it centres headings in
panels and boxes every code fence, which is far too loud for a chat log.
This renders the subset an anima actually emits the way a CLI assistant
does: bold headings, ``•`` bullets, a gutter for code, simple tables.

Only text styles and the sixteen ANSI colour names are used, so the
transcript keeps being drawn with the terminal's own palette. Note the
colour names here are *Rich* names (``blue``), not the Textual CSS ones
(``ansi_blue``) used in the app stylesheet.

Pass ``muted=True`` for background traffic — system notes — which is
drawn in the terminal's own grey throughout, with the accent colours
dropped so it recedes behind the conversation instead of competing with
it.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from rich import box
from rich.cells import cell_len
from rich.console import Group, RenderableType
from rich.style import Style
from rich.table import Table
from rich.text import DEFAULT_JUSTIFY, DEFAULT_OVERFLOW, Lines, Text, pick_bool


class Palette(NamedTuple):
    """The styles a rendered document is drawn with."""

    base: str
    code: str
    link: str
    heading: str
    heading_top: str
    quote: str
    gutter: str
    bullet: str


NORMAL = Palette(
    base="",
    code="magenta",
    link="underline blue",
    heading="bold",
    heading_top="bold underline",
    quote="italic",
    gutter="dim",
    bullet="bold",
)
# Faint text is a colour here, not the ANSI *faint* attribute (SGR 2).
# Whether `dim` is drawn any fainter is entirely up to the terminal —
# plenty of them ignore it, and this app runs on the ansi themes, so
# Textual's own dim-to-colour filters are switched off and nothing else
# resolves it. `bright_black` is the terminal's own grey, the same slot
# the scrollbar thumb uses, so a system note is visibly secondary
# everywhere. Bold and italic still ride on top of it: they change the
# weight of the grey rather than pulling the text back to full contrast,
# which `dim bold` did on every terminal that lets bold win.
MUTED_STYLE = "bright_black"
MUTED = Palette(
    base=MUTED_STYLE,
    code=MUTED_STYLE,
    link=f"{MUTED_STYLE} underline",
    heading=f"{MUTED_STYLE} bold",
    heading_top=f"{MUTED_STYLE} bold",
    quote=f"{MUTED_STYLE} italic",
    gutter=MUTED_STYLE,
    bullet=MUTED_STYLE,
)

BULLETS = ("•", "◦", "▪")
_MAX_INLINE_DEPTH = 4

_FENCE_RE = re.compile(r"^\s*(```+|~~~+)\s*([\w+#-]*)\s*$")
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_RULE_RE = re.compile(r"^\s{0,3}([-*_])[ \t]*(?:\1[ \t]*){2,}$")
_QUOTE_RE = re.compile(r"^\s{0,3}>\s?(.*)$")
_ULIST_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_OLIST_RE = re.compile(r"^(\s*)(\d{1,9})[.)]\s+(.*)$")
_TABLE_DELIM_RE = re.compile(r"^\s*\|?(\s*:?-+:?\s*\|)+\s*:?-*:?\s*\|?\s*$")
_WRAP_TOKEN_RE = re.compile(r"[!-~]+|[ \t]+|.", re.S)

# Ordered so the longest fences win: ``***x***`` before ``**x**`` before ``*x*``.
_INLINE_RE = re.compile(
    r"(?P<tick>`+)(?P<code>.+?)(?P=tick)"
    r"|\*\*\*(?P<bolditalic>\S(?:.*?\S)?)\*\*\*"
    r"|\*\*(?P<bold>\S(?:.*?\S)?)\*\*"
    r"|__(?P<bold_us>\S(?:.*?\S)?)__"
    r"|~~(?P<strike>\S(?:.*?\S)?)~~"
    # ``*`` may emphasise inside a word (CommonMark allows it, and Japanese
    # text has no spaces to lean on); ``_`` may not, or `snake_case_name`
    # would come out italic.
    r"|\*(?P<italic>[^\s*](?:[^*\n]*[^\s*])?)\*"
    r"|(?<![\w_])_(?P<italic_us>[^_\n]+?)_(?![\w_])"
    r"|\[(?P<link_text>[^\]\n]*)\]\((?P<link_url>[^)\s]+)(?:\s+\"[^\"]*\")?\)",
    re.S,
)

# Emphasis groups of `_INLINE_RE`, each parsed again so they can nest.
_EMPHASIS = (
    ("bolditalic", "bold italic"),
    ("bold", "bold"),
    ("bold_us", "bold"),
    ("strike", "strike"),
    ("italic", "italic"),
    ("italic_us", "italic"),
)


# ── Inline ────────────────────────────────────────────────
def _mixed_script_breaks(source: str, width: int) -> list[int]:
    """Return line breaks that keep ASCII words intact in Japanese prose."""
    if width <= 0:
        return []
    breaks: list[int] = []
    used = 0
    for match in _WRAP_TOKEN_RE.finditer(source):
        token = match.group(0)
        token_width = cell_len(token)
        if token.isspace():
            used += token_width
            continue
        if token.isascii() and token_width <= width:
            if used and used + token_width > width:
                breaks.append(match.start())
                used = token_width
            else:
                used += token_width
            continue
        if token.isascii():
            if used and match.start():
                breaks.append(match.start())
                used = 0
            for offset in range(width, len(token), width):
                breaks.append(match.start() + offset)
            used = token_width % width
            continue
        if used and used + token_width > width:
            breaks.append(match.start())
            used = token_width
        else:
            used += token_width
    return breaks


class _TranscriptText(Text):
    """Rich text with CJK-aware wrapping that does not split ASCII words."""

    def blank_copy(self, plain: str = "") -> _TranscriptText:
        return _TranscriptText(
            plain,
            style=self.style,
            justify=self.justify,
            overflow=self.overflow,
            no_wrap=self.no_wrap,
            end=self.end,
            tab_size=self.tab_size,
        )

    def wrap(
        self,
        console,
        width: int,
        *,
        justify=None,
        overflow=None,
        tab_size: int = 8,
        no_wrap=None,
    ) -> Lines:
        wrap_justify = justify or self.justify or DEFAULT_JUSTIFY
        wrap_overflow = overflow or self.overflow or DEFAULT_OVERFLOW
        no_wrap = pick_bool(no_wrap, self.no_wrap, False) or overflow == "ignore"
        lines = Lines()
        for source_line in self.split(allow_blank=True):
            if "\t" in source_line:
                source_line.expand_tabs(tab_size)
            if no_wrap:
                new_lines = Lines([source_line])
            else:
                new_lines = source_line.divide(_mixed_script_breaks(str(source_line), width))
                for wrapped_line in new_lines:
                    wrapped_line.rstrip_end(width)
            if wrap_justify:
                new_lines.justify(console, width, justify=wrap_justify, overflow=wrap_overflow)
            for wrapped_line in new_lines:
                wrapped_line.truncate(width, overflow=wrap_overflow)
            lines.extend(new_lines)
        return lines


class _TranscriptRenderable:
    """Keep Textual from flattening :class:`_TranscriptText` to plain Text."""

    def __init__(self, renderable: RenderableType) -> None:
        self.renderable = renderable

    @property
    def plain(self) -> str:
        return getattr(self.renderable, "plain", "")

    def __rich_console__(self, console, options):
        yield from console.render(self.renderable, options)


def transcript_renderable(renderable: RenderableType) -> RenderableType:
    """Preserve transcript-specific wrapping through Textual's render path."""
    return _TranscriptRenderable(renderable)


def plain_text(source: str, *, style: str | Style = "") -> Text:
    """Create unparsed transcript text with mixed-script-safe wrapping."""
    return _TranscriptText(source, style=style, overflow="fold")


def render_inline(source: str, *, base: str | Style = "", muted: bool = False) -> Text:
    """Render a single run of markdown text (no block structure)."""
    palette = MUTED if muted else NORMAL
    text = _TranscriptText(style=_style(palette.base) + _style(base), overflow="fold")
    _append_inline(text, source, Style.null(), 0, palette)
    return text


def _style(spec: str | Style) -> Style:
    return spec if isinstance(spec, Style) else Style.parse(spec)


def _append_inline(text: Text, source: str, inherited: Style, depth: int, palette: Palette) -> None:
    if depth >= _MAX_INLINE_DEPTH:
        text.append(source, style=inherited)
        return
    pos = 0
    for match in _INLINE_RE.finditer(source):
        if match.start() > pos:
            text.append(source[pos : match.start()], style=inherited)
        pos = match.end()
        if (body := match.group("code")) is not None:
            # Code spans are literal: never look for markup inside them.
            text.append(body, style=inherited + _style(palette.code))
            continue
        for group, style in _EMPHASIS:
            if (body := match.group(group)) is not None:
                _append_inline(text, body, inherited + _style(style), depth + 1, palette)
                break
        else:
            url = match.group("link_url")
            label = match.group("link_text") or url
            link = inherited + _style(palette.link) + Style(link=url)
            _append_inline(text, label, link, depth + 1, palette)
    if pos < len(source):
        text.append(source[pos:], style=inherited)


# ── Blocks ────────────────────────────────────────────────
def render_markdown(source: str, *, muted: bool = False) -> RenderableType:
    """Render a markdown document as a Rich renderable."""
    blocks = _parse_blocks(source, MUTED if muted else NORMAL)
    if not blocks:
        return Text("")
    if len(blocks) == 1:
        return blocks[0]
    return Group(*blocks)


def _parse_blocks(source: str, palette: Palette) -> list[RenderableType]:
    lines = source.split("\n")
    out: list[RenderableType] = []
    pending: list[Text] = []

    def flush() -> None:
        if pending:
            out.append(_TranscriptText("\n", overflow="fold").join(pending))
            pending.clear()

    def blank() -> None:
        # Collapse runs of blank lines, and never open the document with one.
        if pending:
            if pending[-1].plain:
                pending.append(Text(""))
        elif out:
            # A table was just flushed out; keep it off the next paragraph.
            pending.append(Text(""))

    i = 0
    total = len(lines)
    while i < total:
        line = lines[i]

        fence = _FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)[0] * 3
            i += 1
            code: list[str] = []
            while i < total and not lines[i].strip().startswith(marker):
                code.append(lines[i])
                i += 1
            i += 1  # step over the closing fence (or past the end while streaming)
            blank()
            pending.append(_code_block(code, palette))
            pending.append(Text(""))
            continue

        if line.strip() and "|" in line and i + 1 < total and _TABLE_DELIM_RE.match(lines[i + 1]):
            table, next_i = _parse_table(lines, i, palette)
            if table is not None:
                flush()
                out.append(table)
                i = next_i
                continue

        heading = _HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            blank()
            style = palette.heading_top if level <= 2 else palette.heading
            pending.append(_inline(heading.group(2), palette, base=style))
            i += 1
            continue

        if _RULE_RE.match(line):
            blank()
            pending.append(Text("─" * 24, style=palette.gutter))
            i += 1
            continue

        quote = _QUOTE_RE.match(line)
        if quote:
            quoted: list[str] = []
            while i < total and (m := _QUOTE_RE.match(lines[i])):
                quoted.append(m.group(1))
                i += 1
            pending.append(_quote_block(quoted, palette))
            continue

        if not line.strip():
            blank()
            i += 1
            continue

        pending.append(_list_item(line, palette) or _inline(line.rstrip(), palette))
        i += 1

    while pending and not pending[-1].plain:
        pending.pop()
    flush()
    return out


def _inline(source: str, palette: Palette, *, base: str | Style = "") -> Text:
    return render_inline(source, base=base, muted=palette is MUTED)


def _list_item(line: str, palette: Palette) -> Text | None:
    unordered = _ULIST_RE.match(line)
    ordered = _OLIST_RE.match(line) if unordered is None else None
    if unordered is None and ordered is None:
        return None
    match = unordered or ordered
    assert match is not None
    level = min(len(match.group(1).expandtabs(4)) // 2, len(BULLETS) - 1)
    marker = BULLETS[level] if unordered else f"{ordered.group(2)}."  # type: ignore[union-attr]
    body = (unordered.group(2) if unordered else ordered.group(3)).rstrip()  # type: ignore[union-attr]
    item = _TranscriptText("  " * level, overflow="fold")
    item.append(f"{marker} ", style=palette.bullet)
    item.append_text(_inline(body, palette))
    return item


def _code_block(lines: list[str], palette: Palette) -> Text:
    while lines and not lines[-1].strip():
        lines.pop()
    body = _TranscriptText(style=palette.base, overflow="fold")
    for index, raw in enumerate(lines):
        if index:
            body.append("\n")
        body.append("│ ", style=palette.gutter)
        body.append(raw)
    return body


def _quote_block(lines: list[str], palette: Palette) -> Text:
    body = _TranscriptText(overflow="fold")
    for index, raw in enumerate(lines):
        if index:
            body.append("\n")
        body.append("▌ ", style=palette.gutter)
        body.append_text(_inline(raw.rstrip(), palette, base=palette.quote))
    return body


def _split_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _justify(spec: str) -> str:
    spec = spec.strip()
    if spec.startswith(":") and spec.endswith(":"):
        return "center"
    if spec.endswith(":"):
        return "right"
    return "left"


def _parse_table(lines: list[str], start: int, palette: Palette) -> tuple[Table | None, int]:
    header = _split_row(lines[start])
    aligns = [_justify(spec) for spec in _split_row(lines[start + 1])]
    if len(header) != len(aligns) or not header:
        return None, start
    i = start + 2
    rows: list[list[str]] = []
    while i < len(lines) and lines[i].strip() and "|" in lines[i]:
        rows.append(_split_row(lines[i]))
        i += 1
    table = Table(
        box=box.SIMPLE_HEAD,
        show_edge=False,
        pad_edge=False,
        header_style=palette.heading,
        border_style=palette.gutter,
        expand=False,
    )
    for title, align in zip(header, aligns, strict=True):
        table.add_column(_inline(title, palette), justify=align, overflow="fold")
    for row in rows:
        cells = (row + [""] * len(header))[: len(header)]
        table.add_row(*[_inline(cell, palette) for cell in cells])
    return table, i
