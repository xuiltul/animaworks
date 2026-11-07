# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for the redesigned board page.

The test uses a tiny static server for the real frontend assets and intercepts
all ``/api`` requests with deterministic, generic fixture data. This keeps the
test independent of a running AnimaWorks server or real workspace data.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import date, datetime, timedelta
from datetime import time as dt_time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import pytest

try:
    from playwright.sync_api import Browser, Page, sync_playwright

    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False


STATIC_DIR = Path(__file__).resolve().parents[2] / "server" / "static"


def _find_chromium() -> str | None:
    """Find a Playwright Chromium binary, respecting custom HOME setups."""
    roots: list[Path] = []
    env_root = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if env_root:
        roots.append(Path(env_root))
    roots.append(Path.home() / ".cache" / "ms-playwright")
    try:
        roots.extend(Path("/home").glob("*/.cache/ms-playwright"))
    except OSError:
        pass

    patterns = (
        "chromium-*/chrome-linux64/chrome",
        "chromium-*/chrome-linux/chrome",
        "chromium_headless_shell-*/chrome-linux/headless_shell",
    )
    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        for pattern in patterns:
            candidates = sorted(resolved.glob(pattern))
            if candidates:
                return str(candidates[-1])
    return None


_CHROMIUM = _find_chromium() if HAS_PLAYWRIGHT else None

pytestmark = pytest.mark.skipif(
    _CHROMIUM is None,
    reason="playwright or a chromium browser binary is not installed",
)


# ── Fixture data ─────────────────────────────────────────────


def _local_today(hour: int, minute: int) -> datetime:
    tzinfo = datetime.now().astimezone().tzinfo
    return datetime.combine(date.today(), dt_time(hour, minute), tzinfo=tzinfo)


def _iso(value: datetime) -> str:
    return value.isoformat()


TODAY_AM = _local_today(9, 0)
YESTERDAY_PM = TODAY_AM - timedelta(days=1, minutes=1)
LONG_TEXT = "# 今日の共有\n\n" + "\n\n".join(
    f"項目{i}: チームの進捗を共有します。次の作業に取りかかります。" for i in range(90)
)


class _BoardApi:
    """Deterministic API fixture used by Playwright route interception."""

    def __init__(self) -> None:
        self.posts: list[dict[str, object]] = []
        self.messages: list[dict[str, object]] = [
            {
                "id": "m5",
                "ts": _iso(YESTERDAY_PM),
                "from": "bob",
                "text": "昨日の投稿です。",
                "source": "anima",
                "kind": "anima",
            },
            {
                "id": "m1",
                "ts": _iso(TODAY_AM),
                "from": "alice",
                "text": LONG_TEXT,
                "source": "anima",
                "kind": "anima",
            },
            {
                "id": "m1b",
                "ts": _iso(TODAY_AM + timedelta(minutes=5)),
                "from": "alice",
                "text": "続けての短い投稿です。",
                "source": "anima",
                "kind": "anima",
            },
            {
                "id": "m2",
                "ts": _iso(TODAY_AM + timedelta(minutes=10)),
                "from": "human",
                "text": "人の投稿です。",
                "source": "human",
                "kind": "human",
            },
            {
                "id": "m3",
                "ts": _iso(TODAY_AM + timedelta(minutes=12)),
                "from": "slackbot",
                "text": "Slack bot からの通知です。",
                "source": "slack",
                "kind": "external_bot",
            },
            {
                "id": "m4",
                "ts": _iso(TODAY_AM + timedelta(minutes=14)),
                "from": "system",
                "text": "システム通知です。",
                "source": "system",
                "kind": "operator",
            },
        ]
        self.dm_messages: list[dict[str, object]] = [
            {
                "id": "d1",
                "ts": _iso(TODAY_AM + timedelta(minutes=20)),
                "from": "alice",
                "text": "bob へ共有します。",
                "source": "anima",
                "kind": "anima",
            },
            {
                "id": "d2",
                "ts": _iso(TODAY_AM + timedelta(minutes=21)),
                "from": "bob",
                "text": "確認しました。",
                "source": "anima",
                "kind": "anima",
            },
        ]
        self.animas = [
            {"name": "alice", "status": "running", "enabled": True, "role": "進行役"},
            {"name": "bob", "status": "idle", "enabled": True, "role": "調査担当"},
            {"name": "system", "status": "idle", "enabled": True, "role": "システム"},
            {"name": "slackbot", "status": "idle", "enabled": True, "role": "Slack bot"},
        ]
        activity = [1, 0, 3, 2, 0, 4, 6, 2, 1, 0, 5, 3, 2, 4]
        self.channels = [
            {
                "name": "general",
                "message_count": 6,
                "last_post_ts": _iso(TODAY_AM + timedelta(minutes=14)),
                "last_from": "system",
                "access": "open",
                "writable": True,
                "company": "team-a",
                "members": ["alice", "bob", "human", "system"],
                "description": "チームの共有チャンネル",
                "slack": {"inbound": True, "outbound": False},
                "activity_14d": activity,
            },
            {
                "name": "ops",
                "message_count": 2,
                "last_post_ts": _iso(TODAY_AM + timedelta(minutes=8)),
                "last_from": "bob",
                "access": "members",
                "writable": False,
                "company": "team-a",
                "members": ["alice", "bob"],
                "description": "運用の連絡",
                "slack": {"inbound": False, "outbound": True},
                "activity_14d": [0, 1, 1, 2, 0, 0, 3, 1, 0, 2, 1, 1, 0, 2],
            },
            {
                "name": "random",
                "message_count": 1,
                "last_post_ts": _iso(YESTERDAY_PM),
                "last_from": "alice",
                "access": "open",
                "writable": True,
                "company": "",
                "members": ["alice"],
                "description": "",
                "slack": {"inbound": False, "outbound": False},
                "activity_14d": [0] * 14,
            },
            {
                "name": "archive",
                "message_count": 3,
                "last_post_ts": _iso(YESTERDAY_PM - timedelta(hours=2)),
                "last_from": "bob",
                "access": "closed",
                "writable": False,
                "company": "team-a",
                "members": ["bob"],
                "description": "終了したチャンネル",
                "slack": {"inbound": False, "outbound": False},
                "activity_14d": [0] * 14,
            },
        ]
        self.dm_pairs = [
            {
                "pair": "alice-bob",
                "participants": ["alice", "bob"],
                "message_count": 2,
                "last_message_ts": _iso(TODAY_AM + timedelta(minutes=21)),
                "last_from": "bob",
            }
        ]

    def _json(self, route, payload: object, status: int = 200) -> None:
        route.fulfill(
            status=status,
            content_type="application/json",
            body=json.dumps(payload, ensure_ascii=False),
        )

    def handle(self, route) -> None:
        request = route.request
        path = urlparse(request.url).path
        method = request.method.upper()

        if path == "/api/animas":
            self._json(route, self.animas)
            return
        if path == "/api/channels":
            self._json(route, self.channels)
            return
        if path == "/api/dm":
            self._json(route, self.dm_pairs)
            return
        if path.startswith("/api/animas/") and "/assets/" in path:
            self._json(route, {"error": "not found"}, status=404)
            return

        if path.startswith("/api/channels/"):
            name = unquote(path[len("/api/channels/") :])
            if method == "POST":
                body = json.loads(request.post_data or "{}")
                self.posts.append(body)
                self._json(route, {"post": {"id": f"posted-{len(self.posts)}"}})
                return
            query = parse_qs(urlparse(request.url).query)
            limit = int(query.get("limit", ["50"])[0])
            offset = int(query.get("offset", ["0"])[0])
            since = query.get("since", [""])[0]
            if since:
                rows = [row for row in self.messages if str(row["ts"]) > since]
                self._json(
                    route,
                    {
                        "channel": name,
                        "messages": rows,
                        "total": len(self.messages),
                        "has_more": False,
                    },
                )
                return
            rows = self.messages[offset : offset + limit]
            self._json(
                route,
                {
                    "channel": name,
                    "messages": rows,
                    "total": len(self.messages),
                    "has_more": offset + limit < len(self.messages),
                },
            )
            return

        if path.startswith("/api/dm/"):
            pair = unquote(path[len("/api/dm/") :])
            self._json(
                route,
                {
                    "pair": pair,
                    "messages": self.dm_messages,
                    "total": len(self.dm_messages),
                    "has_more": False,
                },
            )
            return

        self._json(route, {"error": "not found"}, status=404)


TEST_HTML = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>board e2e</title>
<link rel="stylesheet" href="/styles/tokens.css">
<link rel="stylesheet" href="/styles/board.css">
<style>
  body { margin: 0; font-family: system-ui, sans-serif; }
  #boardApp { height: 100vh; padding: 10px; box-sizing: border-box; }
</style>
</head>
<body class="theme-business">
<div id="boardApp"></div>
<script>
  window.marked = {
    use() {},
    Renderer: class {
      link(token) { return '<a href="' + token.href + '">' + token.text + '</a>'; }
      image(token) { return '<img src="' + token.href + '" alt="' + token.text + '">'; }
      code(token) { return '<pre><code>' + token.text + '</code></pre>'; }
    },
    parse(text) {
      const esc = (value) => String(value)
        .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
      const lines = String(text).split('\\n');
      let html = '';
      let code = [];
      let inCode = false;
      for (const line of lines) {
        if (line.startsWith('```')) {
          if (inCode) { html += '<pre><code>' + esc(code.join('\\n')) + '</code></pre>'; code = []; }
          inCode = !inCode;
          continue;
        }
        if (inCode) { code.push(line); continue; }
        if (!line.trim()) continue;
        const heading = /^(#{1,3})\\s+(.*)$/.exec(line);
        if (heading) {
          const level = heading[1].length;
          html += '<h' + level + '>' + esc(heading[2]) + '</h' + level + '>';
          continue;
        }
        html += '<p>' + esc(line) + '</p>';
      }
      if (inCode) html += '<pre><code>' + esc(code.join('\\n')) + '</code></pre>';
      return html;
    }
  };
</script>
<script type="module">
  import { loadTranslations } from '/shared/i18n.js';
  await loadTranslations('ja');
  const mod = await import('/pages/board.js');
  mod.render(document.getElementById('boardApp'));
  window.__boardReady = true;
</script>
</body>
</html>
"""


class _StaticHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, *args):  # keep pytest output clean
        pass


@pytest.fixture(scope="module")
def static_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StaticHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def browser_instance():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            executable_path=_CHROMIUM,
        )
        try:
            yield browser
        finally:
            browser.close()


@pytest.fixture
def board_page(static_server: str, browser_instance: Browser):
    api = _BoardApi()
    context = browser_instance.new_context(viewport={"width": 1100, "height": 900})
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.route("**/api/**", api.handle)
    page.route(
        "**/board-test.html",
        lambda route: route.fulfill(status=200, content_type="text/html", body=TEST_HTML),
    )
    page.goto(f"{static_server}/board-test.html", wait_until="load")
    try:
        page.wait_for_function("window.__boardReady === true", timeout=7000)
    except Exception:
        context.close()
        raise AssertionError(f"board page failed to initialise: {errors}") from None
    page.locator('[data-board-mode="board"]').wait_for(timeout=5000)
    page.locator('[data-board-type="channel"][data-board-name="general"]').wait_for(timeout=5000)
    page.api = api  # type: ignore[attr-defined]
    try:
        yield page
    finally:
        page.close()
        context.close()


def _open_channel(page: Page, name: str) -> None:
    page.locator(f'[data-board-type="channel"][data-board-name="{name}"]').click()
    page.locator(".board-header-name").filter(has_text=name).wait_for(timeout=5000)


def _wait_for_messages(page: Page) -> None:
    page.locator('[data-board-id="m1"]').wait_for(timeout=5000)


# ── Tests ───────────────────────────────────────────────────


def test_same_id_ws_and_refetch_is_rendered_once(board_page: Page) -> None:
    api = board_page.api  # type: ignore[attr-defined]
    _open_channel(board_page, "general")
    _wait_for_messages(board_page)
    assert board_page.locator('[data-board-id="m1"]').count() == 1

    new_message = {
        "id": "m9",
        "ts": _iso(TODAY_AM + timedelta(minutes=30)),
        "from": "bob",
        "text": "WS で届いた新しい投稿です。",
        "source": "anima",
        "kind": "anima",
        "channel": "general",
    }
    api.messages.append(new_message)
    board_page.evaluate(
        "(message) => window.__boardWsHandler(message)",
        new_message,
    )
    board_page.locator('[data-board-id="m9"]').wait_for(timeout=5000)
    assert board_page.locator('[data-board-id="m9"]').count() == 1

    duplicate = dict(new_message, text="同じ ID の再送です。")
    board_page.evaluate(
        "(message) => window.__boardWsHandler(message)",
        duplicate,
    )
    board_page.wait_for_timeout(150)
    assert board_page.locator('[data-board-id="m9"]').count() == 1

    # Re-fetching the channel must reconcile by key, not append a duplicate.
    _open_channel(board_page, "general")
    board_page.locator('[data-board-id="m9"]').wait_for(timeout=5000)
    assert board_page.locator('[data-board-id="m9"]').count() == 1
    assert board_page.locator('[data-board-id="m1"]').count() == 1


def test_composing_enter_does_not_send(board_page: Page) -> None:
    api = board_page.api  # type: ignore[attr-defined]
    _open_channel(board_page, "general")
    textarea = board_page.locator("#boardComposerInput")
    textarea.fill("hello")
    textarea.evaluate(
        """element => {
          const event = new KeyboardEvent('keydown', {
            key: 'Enter', bubbles: true, cancelable: true, isComposing: true
          });
          Object.defineProperty(event, 'isComposing', { value: true });
          element.dispatchEvent(event);
        }"""
    )
    board_page.wait_for_timeout(300)
    assert api.posts == []

    textarea.evaluate(
        """element => {
          element.dispatchEvent(new KeyboardEvent('keydown', {
            key: 'Enter', bubbles: true, cancelable: true
          }));
        }"""
    )
    board_page.wait_for_function(
        "document.querySelector('#boardComposerInput').value === ''",
        timeout=5000,
    )
    assert len(api.posts) == 1
    assert api.posts[0]["text"] == "hello"


def test_readonly_channel_hides_composer(board_page: Page) -> None:
    _open_channel(board_page, "ops")
    assert board_page.locator(".board-header-name").inner_text() == "ops"
    assert board_page.locator("#boardComposer").is_hidden()
    readonly = board_page.locator("#boardReadonly")
    assert readonly.is_visible()
    assert "このボードは閉じています" in readonly.inner_text()
    assert board_page.locator(".board-badge-access-members").count() == 1


def test_kind_shapes_and_badges(board_page: Page) -> None:
    _open_channel(board_page, "general")
    _wait_for_messages(board_page)

    anima_post = board_page.locator('[data-board-id="m1"]')
    assert anima_post.locator(".board-avatar-anima").count() == 1

    human_post = board_page.locator('[data-board-id="m2"]')
    assert human_post.locator(".board-avatar-human .board-avatar-icon").count() == 1
    assert "人" in human_post.locator(".board-kind-badge-human").inner_text()

    external_post = board_page.locator('[data-board-id="m3"]')
    assert external_post.locator(".board-avatar-external").count() == 1
    assert "Slack bot" in external_post.locator(".board-kind-badge-external").inner_text()

    operator_post = board_page.locator('[data-board-id="m4"]')
    assert operator_post.locator(".board-avatar-operator .board-avatar-icon").count() == 1
    assert "システム" in operator_post.locator(".board-kind-badge-operator").inner_text()


def test_long_post_folds_and_expands(board_page: Page) -> None:
    _open_channel(board_page, "general")
    _wait_for_messages(board_page)
    post = board_page.locator('[data-board-id="m1"]')
    text = post.locator(".board-post-text")
    assert "board-md-folded" in (text.get_attribute("class") or "")
    folded_height = text.evaluate("element => element.getBoundingClientRect().height")
    assert folded_height <= 240

    more = post.locator(".board-more")
    assert more.is_visible()
    assert "続きを読む" in more.inner_text()
    more.click()
    assert "board-md-folded" not in (text.get_attribute("class") or "")
    expanded_height = text.evaluate("element => element.getBoundingClientRect().height")
    assert expanded_height > folded_height
    assert more.get_attribute("aria-expanded") == "true"
    assert "たたむ" in more.inner_text()


def test_day_separators(board_page: Page) -> None:
    _open_channel(board_page, "general")
    _wait_for_messages(board_page)
    labels = board_page.locator(".board-day-label").all_text_contents()
    assert len(labels) >= 2
    assert any(label.startswith("今日") for label in labels)
    assert any(label.startswith("昨日") for label in labels)


def test_archive_is_collapsed(board_page: Page) -> None:
    board_page.locator("details.board-rail-archive").wait_for(timeout=5000)
    details = board_page.locator("details.board-rail-archive")
    assert details.evaluate("element => element.open") is False
    assert "アーカイブ" in details.locator("summary").inner_text()
    assert details.locator('[data-board-name="archive"]').is_hidden()


def test_dm_mode_lists_pairs_and_is_readonly(board_page: Page) -> None:
    board_page.locator('[data-board-mode="dm"]').click()
    dm_row = board_page.locator('[data-board-type="dm"][data-board-name="alice-bob"]')
    dm_row.wait_for(timeout=5000)
    assert dm_row.is_visible()
    dm_row.click()

    board_page.locator(".board-header-title").filter(has_text="alice").wait_for(timeout=5000)
    title = board_page.locator(".board-header-title").inner_text()
    assert "alice" in title
    assert "bob" in title
    assert board_page.locator("#boardComposer").is_hidden()
    readonly = board_page.locator("#boardReadonly")
    assert readonly.is_visible()
    assert "DM は読み取り専用です" in readonly.inner_text()
    board_page.locator('[data-board-id="d1"]').wait_for(timeout=5000)
    assert board_page.locator('[data-board-id="d1"]').count() == 1
    assert board_page.locator('[data-board-id="d2"]').count() == 1
