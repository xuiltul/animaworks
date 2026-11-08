"""The desktop chat pane keeps its input form in the last grid row.

The pane is a grid of header, messages and form on wide screens. When a row
is removed from the markup, the row template has to shrink with it: otherwise
the messages area lands in an ``auto`` row that grows with the conversation and
the form lands in the flexible row, which collapses to 0px once the
conversation is taller than the screen (the input box disappears).
"""

from __future__ import annotations

import re
from pathlib import Path

STYLES = Path(__file__).resolve().parents[3] / "server/static/styles"
# Load order in index.html: chat.css is overridden by chat-presence.css.
DESKTOP_FILES = ("chat.css", "chat-presence.css")
DESKTOP_MEDIA = "@media (min-width: 769px)"


def _media_blocks(css: str, media: str) -> list[str]:
    blocks: list[str] = []
    start = css.find(media)
    while start != -1:
        open_at = css.index("{", start)
        depth = 0
        for pos in range(open_at, len(css)):
            if css[pos] == "{":
                depth += 1
            elif css[pos] == "}":
                depth -= 1
                if depth == 0:
                    blocks.append(css[open_at + 1 : pos])
                    break
        start = css.find(media, open_at)
    return blocks


def _declarations(block: str, selector: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for prelude, body in re.findall(r"([^{}]+)\{([^{}]*)\}", block):
        selectors = [s.strip() for s in re.sub(r"/\*.*?\*/", "", prelude, flags=re.S).split(",")]
        if selector not in selectors:
            continue
        for decl in body.split(";"):
            if ":" in decl:
                name, value = decl.split(":", 1)
                found[name.strip()] = value.strip()
    return found


def _desktop_value(selector: str, prop: str) -> str:
    value = ""
    for name in DESKTOP_FILES:
        css = (STYLES / name).read_text(encoding="utf-8")
        for block in _media_blocks(css, DESKTOP_MEDIA):
            value = _declarations(block, selector).get(prop, value)
    assert value, f"{selector} has no {prop} in the desktop chat styles"
    return value


def _tracks(template: str) -> list[str]:
    tracks: list[str] = []
    depth = 0
    current = ""
    for ch in template:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch.isspace() and depth == 0:
            if current:
                tracks.append(current)
            current = ""
        else:
            current += ch
    if current:
        tracks.append(current)
    return tracks


def test_messages_get_the_flexible_row_and_form_the_last_row() -> None:
    tracks = _tracks(_desktop_value(".chat-page-main", "grid-template-rows"))
    messages_row = int(_desktop_value(".chat-messages-area", "grid-row"))
    form_row = int(_desktop_value('[data-chat-id="chatPageForm"]', "grid-row"))

    assert form_row == len(tracks), f"form is in row {form_row} of {len(tracks)} rows: {tracks}"
    assert "1fr" in tracks[messages_row - 1], f"messages row {messages_row} is not flexible: {tracks}"
    assert "fr" not in tracks[form_row - 1], f"form row {form_row} is flexible: {tracks}"
