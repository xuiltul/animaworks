from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
STATIC = ROOT / "server" / "static"

JS_FILES = [
    "pages/hub.js",
    "pages/hub/index.js",
    "pages/hub/members.js",
    "pages/board/index.js",
    "pages/board/composer.js",
    "pages/board/timeline.js",
    "pages/board/topbar.js",
    "modules/router.js",
]

I18N_KEYS = [
    "nav.hub",
    "hub.title",
    "hub.member_count",
    "hub.members_label",
    "hub.tab_chat",
    "hub.tab_timeline",
    "hub.last_seen",
    "hub.never_seen",
    "hub.route_webhook",
    "hub.route_poll",
    "hub.every_m",
    "hub.every_h",
    "hub.every_d",
    "hub.disabled_chip",
    "hub.overdue",
    "hub.call",
    "hub.no_members",
    "hub.disabled_title",
    "hub.disabled_body",
    "hub.timeline_filter_all",
    "hub.timeline_filter_unread",
    "hub.timeline_no_unread",
]


def _read(relative: str) -> str:
    return (STATIC / relative).read_text()


def test_router_registers_hub_page() -> None:
    router = _read("modules/router.js")

    assert 'routes["/hub"]' in router
    assert "../pages/hub.js" in router


def test_hub_javascript_is_valid_es_modules() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")

    for relative in JS_FILES:
        source = _read(relative)
        result = subprocess.run(
            [node, "--input-type=module", "--check"],
            input=source,
            text=True,
            capture_output=True,
            cwd=ROOT,
            check=False,
        )
        assert result.returncode == 0, f"{relative}: {result.stderr}"


def test_hub_i18n_keys_exist_in_all_languages() -> None:
    for language in ("ja", "en", "ko"):
        translations = json.loads(_read(f"i18n/{language}.json"))
        missing = [key for key in I18N_KEYS if key not in translations]
        assert not missing, f"{language} is missing {missing}"


def test_board_supports_pinned_channel_and_all_timeline() -> None:
    assert "pinnedChannel" in _read("pages/board/index.js")
    assert 'mode === "all"' in _read("pages/board/timeline.js")


def test_members_escapes_html() -> None:
    members = _read("pages/hub/members.js")

    assert "import { escapeAttr, escapeHtml }" in members


def test_roster_is_a_dropdown_in_the_board_header() -> None:
    board = _read("pages/board/index.js")
    header = _read("pages/board/header.js")
    hub = _read("pages/hub/index.js")
    assert "options.pinnedTitle" in board
    assert "titleHtml" in header
    assert "pinnedTitle: switchMarkup" in hub
    assert 'data-hub-roster role="dialog"' in hub
    assert 'aria-haspopup="dialog"' in hub
    assert "export function renderAvatarStack" in _read("pages/hub/members.js")
