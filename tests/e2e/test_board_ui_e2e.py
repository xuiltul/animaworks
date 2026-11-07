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
import re
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
        self.settings_calls: list[dict[str, object]] = []
        self.settings_error: tuple[int, str] | None = None
        self.read_calls: list[dict[str, object]] = []
        self.reads = {
            "readers": {
                "anima:alice": {
                    "upto_id": "m2",
                    "upto_ts": _iso(TODAY_AM + timedelta(minutes=10)),
                    "updated_at": _iso(TODAY_AM + timedelta(minutes=11)),
                },
                "anima:bob": {
                    "upto_id": "m2",
                    "upto_ts": _iso(TODAY_AM + timedelta(minutes=10)),
                    "updated_at": _iso(TODAY_AM + timedelta(minutes=11)),
                },
                "anima:carol": {
                    "upto_id": "m2",
                    "upto_ts": _iso(TODAY_AM + timedelta(minutes=10)),
                    "updated_at": _iso(TODAY_AM + timedelta(minutes=11)),
                },
                "human:human": {
                    "upto_id": "m1b",
                    "upto_ts": _iso(TODAY_AM + timedelta(minutes=5)),
                    "updated_at": _iso(TODAY_AM + timedelta(minutes=6)),
                },
            },
            "picked": {},
        }
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
                "reply_to": "m1",
            },
            {
                "id": "m2",
                "ts": _iso(TODAY_AM + timedelta(minutes=10)),
                "from": "human",
                "text": "人の投稿です。",
                "source": "human",
                "kind": "human",
                "delivery": {
                    "notified": ["alice"],
                    "not_notified": {"bob": "メンバーではないので届きません"},
                },
                "hooks": {"slack_outbound": "failed"},
            },
            {
                "id": "m2r",
                "ts": _iso(TODAY_AM + timedelta(minutes=11)),
                "from": "alice",
                "text": "人の投稿への返事です。",
                "source": "anima",
                "kind": "anima",
                "reply_to": "m2",
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
                "reply_to": "d1",
            },
        ]
        self.animas = [
            {"name": "alice", "status": "running", "enabled": True, "role": "進行役", "company": "team-a"},
            {"name": "bob", "status": "idle", "enabled": True, "role": "調査担当", "company": "team-a"},
            {"name": "carol", "status": "stopped", "enabled": False, "role": "補助", "company": "team-a"},
            {"name": "system", "status": "idle", "enabled": True, "role": "システム", "company": "team-a"},
            {"name": "slackbot", "status": "idle", "enabled": True, "role": "Slack bot", "company": "team-b"},
        ]
        activity = [1, 0, 3, 2, 0, 4, 6, 2, 1, 0, 5, 3, 2, 4]
        self.channels = [
            {
                "name": "general",
                "message_count": 7,
                "unread": 2,
                "last_post_ts": _iso(TODAY_AM + timedelta(minutes=14)),
                "last_from": "system",
                "access": "open",
                "writable": True,
                "company": "team-a",
                "members": ["alice", "bob", "human", "system"],
                "description": "チームの共有チャンネル",
                "slack": {"inbound": True, "outbound": True},
                "activity_14d": activity,
            },
            {
                "name": "team",
                "message_count": 0,
                "last_post_ts": _iso(TODAY_AM + timedelta(minutes=6)),
                "last_from": "alice",
                "access": "members",
                "writable": True,
                "company": "team-a",
                "members": ["alice", "bob"],
                "description": "メンバーだけで使うチャンネル",
                "slack": {"inbound": False, "outbound": False},
                "activity_14d": [0] * 14,
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

        if path.startswith("/api/channels/") and path.endswith("/reads"):
            name = unquote(path[len("/api/channels/") : -len("/reads")])
            if method != "GET":
                self._json(route, {"error": "method not allowed"}, status=405)
                return
            self._json(route, {"channel": name, **self.reads})
            return

        if path.startswith("/api/channels/") and path.endswith("/read"):
            name = unquote(path[len("/api/channels/") : -len("/read")])
            if method != "POST":
                self._json(route, {"error": "method not allowed"}, status=405)
                return
            body = json.loads(request.post_data or "{}")
            upto_id = str(body.get("upto_id", ""))
            self.read_calls.append({"channel": name, **body})
            message = next((row for row in self.messages if str(row.get("id")) == upto_id), None)
            upto_ts = str(message.get("ts")) if message else ""
            self.reads["readers"]["human:human"] = {
                "upto_id": upto_id,
                "upto_ts": upto_ts,
                "updated_at": _iso(TODAY_AM + timedelta(minutes=40)),
            }
            channel = next((row for row in self.channels if row["name"] == name), None)
            if channel is not None:
                channel["unread"] = 0
            self._json(
                route,
                {
                    "channel": name,
                    "reader": "human:human",
                    "upto_id": upto_id,
                    "upto_ts": upto_ts,
                },
            )
            return

        if path.startswith("/api/channels/") and path.endswith("/settings"):
            name = unquote(path[len("/api/channels/") : -len("/settings")])
            if method != "PUT":
                self._json(route, {"error": "method not allowed"}, status=405)
                return
            body = json.loads(request.post_data or "{}")
            self.settings_calls.append({"name": name, **body})
            if self.settings_error is not None:
                status, detail = self.settings_error
                self._json(route, {"detail": detail}, status=status)
                return
            channel = next((row for row in self.channels if row["name"] == name), None)
            if channel is None:
                self._json(route, {"detail": "channel not found"}, status=404)
                return
            for field in ("description", "access", "members"):
                if field in body:
                    channel[field] = body[field]
            self._json(route, channel)
            return

        if path.startswith("/api/channels/"):
            name = unquote(path[len("/api/channels/") :])
            if method == "POST":
                body = json.loads(request.post_data or "{}")
                self.posts.append(body)
                channel = next((row for row in self.channels if row["name"] == name), None)
                notified: list[str] = []
                not_notified: dict[str, str] = {}
                seen: set[str] = set()
                for match in re.finditer(r"@([A-Za-z][A-Za-z0-9_-]{0,30})", str(body.get("text", ""))):
                    mention = match.group(1)
                    if mention.lower() in seen:
                        continue
                    seen.add(mention.lower())
                    anima = next((row for row in self.animas if row["name"] == mention), None)
                    if anima is None:
                        not_notified[mention] = "存在しない anima です"
                    elif anima.get("enabled") is False:
                        not_notified[mention] = "無効にしてあるため届きません"
                    elif channel and channel.get("access") == "members" and mention not in channel.get("members", []):
                        not_notified[mention] = "メンバーではないので届きません"
                    elif channel and channel.get("company") and anima.get("company") != channel.get("company"):
                        not_notified[mention] = "別の会社なので届きません"
                    else:
                        notified.append(mention)
                hooks: dict[str, str] = {}
                if channel and (channel.get("slack") or {}).get("outbound"):
                    hooks["slack_outbound"] = "sent"
                self._json(
                    route,
                    {
                        "post": {
                            "id": f"posted-{len(self.posts)}",
                            "notified": notified,
                            "not_notified": not_notified,
                            "hooks": hooks,
                        }
                    },
                )
                return
            query = parse_qs(urlparse(request.url).query)
            limit = int(query.get("limit", ["50"])[0])
            offset = int(query.get("offset", ["0"])[0])
            since = query.get("since", [""])[0]
            around = query.get("around", [""])[0]
            if around:
                index = next(
                    (i for i, row in enumerate(self.messages) if str(row.get("id")) == around),
                    None,
                )
                if index is None:
                    self._json(route, {"detail": "post not found"}, status=404)
                    return
                start = max(0, index - 2)
                rows = self.messages[start : start + limit]
                self._json(
                    route,
                    {
                        "channel": name,
                        "messages": rows,
                        "total": len(self.messages),
                        "offset": start,
                        "limit": limit,
                        "has_more": start > 0,
                        "has_newer": start + len(rows) < len(self.messages),
                    },
                )
                return
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
            total = len(self.messages)
            start = max(0, total - offset - limit)
            end = max(0, total - offset)
            rows = self.messages[start:end]
            self._json(
                route,
                {
                    "channel": name,
                    "messages": rows,
                    "total": total,
                    "offset": offset,
                    "limit": limit,
                    "has_more": start > 0,
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
    assert anima_post.locator(".board-post-gutter .board-avatar-anima").count() == 1

    human_post = board_page.locator('[data-board-id="m2"]')
    assert human_post.locator(".board-post-gutter .board-avatar-human .board-avatar-icon").count() == 1
    assert "人" in human_post.locator(".board-kind-badge-human").inner_text()

    external_post = board_page.locator('[data-board-id="m3"]')
    assert external_post.locator(".board-post-gutter .board-avatar-external").count() == 1
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


def test_mention_suggestions_and_target_reasons(board_page: Page) -> None:
    _open_channel(board_page, "team")
    textarea = board_page.locator("#boardComposerInput")
    textarea.type("hi @ali")

    suggestion = board_page.locator(".board-suggest-item").filter(has_text="@alice")
    suggestion.wait_for(timeout=5000)
    assert "このボードのメンバー" in board_page.locator(".board-suggest-head").inner_text()
    assert "作業中" in suggestion.inner_text()

    textarea.press("Enter")
    assert textarea.input_value() == "hi @alice "

    textarea.fill("hi @alice @system")
    target = board_page.locator(".board-target-ng").filter(has_text="system")
    target.wait_for(timeout=5000)
    assert "メンバーではないので届きません" in target.inner_text()
    ok_target = board_page.locator(".board-target:not(.board-target-ng)").filter(has_text="alice")
    assert ok_target.count() == 1


def test_quote_reply_and_delivery_footer(board_page: Page) -> None:
    api = board_page.api  # type: ignore[attr-defined]
    _open_channel(board_page, "general")
    _wait_for_messages(board_page)

    post = board_page.locator('[data-board-id="m1"]')
    post.hover()
    post.locator('[data-board-action="quote"]').click()
    textarea = board_page.locator("#boardComposerInput")
    value = textarea.input_value()
    assert value.startswith("> ")
    assert "@alice " in value

    textarea.press("Enter")
    posted = board_page.locator('[data-board-id="posted-1"]')
    posted.wait_for(timeout=5000)
    assert api.posts[-1]["reply_to"] == "m1"
    assert "> " in str(api.posts[-1]["text"])

    delivery = posted.locator(".board-delivery")
    delivery.wait_for(timeout=5000)
    text = delivery.inner_text()
    assert "届いた相手" in text
    assert "alice" in text
    assert "Slack に送りました" in text


def test_row_delivery_is_rendered(board_page: Page) -> None:
    _open_channel(board_page, "general")
    _wait_for_messages(board_page)
    delivery = board_page.locator('[data-board-id="m2"] .board-delivery')
    delivery.wait_for(timeout=5000)
    text = delivery.inner_text()
    assert "届いた相手" in text
    assert "alice" in text
    assert "届かなかった相手" in text
    assert "bob" in text
    assert "メンバーではないので届きません" in text
    assert "Slack への送信に失敗しました" in text


def test_draft_is_restored_when_reopening_board(board_page: Page) -> None:
    _open_channel(board_page, "general")
    textarea = board_page.locator("#boardComposerInput")
    textarea.fill("下書きテスト")

    _open_channel(board_page, "random")
    assert board_page.locator("#boardComposerInput").input_value() == ""

    _open_channel(board_page, "general")
    restored = board_page.locator("#boardComposerInput")
    restored.wait_for(timeout=5000)
    assert restored.input_value() == "下書きテスト"
    assert "下書きを戻しました" in board_page.locator("#boardComposerDraftNote").inner_text()


def test_markdown_preview_toggle(board_page: Page) -> None:
    _open_channel(board_page, "general")
    textarea = board_page.locator("#boardComposerInput")
    textarea.fill("# 見出し")
    board_page.locator("#boardComposerPreviewToggle").click()

    preview = board_page.locator("#boardComposerPreview")
    preview.wait_for(timeout=5000)
    assert preview.is_visible()
    assert textarea.is_hidden()
    assert preview.locator("h1").inner_text() == "見出し"

    board_page.locator("#boardComposerPreviewToggle").click()
    assert textarea.is_visible()


def test_settings_drawer_saves_and_closed_needs_two_clicks(board_page: Page) -> None:
    api = board_page.api  # type: ignore[attr-defined]
    _open_channel(board_page, "general")
    board_page.locator("[data-board-info]").click()
    drawer = board_page.locator("#boardDrawer")
    drawer.wait_for(timeout=5000)
    assert "このボードについて" in drawer.inner_text()

    drawer.locator("[data-board-drawer-edit]").click()
    form = drawer.locator("#boardSettingsForm")
    form.wait_for(timeout=5000)
    form.locator("#boardSetDescription").fill("新しい説明")
    form.locator(".board-drawer-save").click()
    drawer.locator(".board-drawer-flash").wait_for(timeout=5000)
    assert api.settings_calls[-1]["description"] == "新しい説明"
    assert api.settings_calls[-1]["access"] == "open"

    calls_before = len(api.settings_calls)
    drawer.locator("[data-board-drawer-edit]").click()
    form = drawer.locator("#boardSettingsForm")
    form.wait_for(timeout=5000)
    form.locator('input[name="boardSetAccess"][value="closed"]').click()
    form.locator(".board-drawer-save").click()
    warning = drawer.locator(".board-drawer-warning")
    warning.wait_for(timeout=5000)
    assert "もう一度押すと保存します" in warning.inner_text()
    assert len(api.settings_calls) == calls_before

    drawer.locator(".board-drawer-save").click()
    drawer.locator(".board-drawer-flash").wait_for(timeout=5000)
    assert len(api.settings_calls) == calls_before + 1
    assert api.settings_calls[-1]["access"] == "closed"
    assert "閉鎖" in drawer.locator(".board-drawer-kv").inner_text()


def test_settings_permission_error_is_shown(board_page: Page) -> None:
    api = board_page.api  # type: ignore[attr-defined]
    api.settings_error = (403, "owner または admin だけが変更できます")
    _open_channel(board_page, "general")
    board_page.locator("[data-board-info]").click()
    drawer = board_page.locator("#boardDrawer")
    drawer.locator("[data-board-drawer-edit]").click()
    drawer.locator("#boardSetDescription").fill("変更できない説明")
    drawer.locator(".board-drawer-save").click()

    error = drawer.locator(".board-drawer-error")
    error.wait_for(timeout=5000)
    assert "owner または admin" in error.inner_text()
    assert drawer.locator("#boardSettingsForm").is_visible()


def test_settings_member_chips_can_remove_and_add(board_page: Page) -> None:
    api = board_page.api  # type: ignore[attr-defined]
    _open_channel(board_page, "general")
    board_page.locator("[data-board-info]").click()
    drawer = board_page.locator("#boardDrawer")
    drawer.locator("[data-board-drawer-edit]").click()
    form = drawer.locator("#boardSettingsForm")
    form.locator('input[name="boardSetAccess"][value="members"]').click()

    chips = drawer.locator(".board-drawer-chip")
    assert chips.count() == 4
    drawer.locator('[data-board-member-remove="human"]').click()
    assert drawer.locator(".board-drawer-chip").count() == 3

    drawer.locator("#boardSetAdd").select_option("carol")
    assert drawer.locator('[data-board-member-remove="carol"]').count() == 1

    drawer.locator(".board-drawer-save").click()
    drawer.locator(".board-drawer-flash").wait_for(timeout=5000)
    assert api.settings_calls[-1]["access"] == "members"
    assert set(api.settings_calls[-1]["members"]) == {"alice", "bob", "system", "carol"}


def test_new_post_shows_pill_without_moving_scroll(board_page: Page) -> None:
    api = board_page.api  # type: ignore[attr-defined]
    _open_channel(board_page, "general")
    _wait_for_messages(board_page)

    messages = board_page.locator("#boardMessages")
    messages.evaluate("element => element.scrollTo({ top: 0, behavior: 'instant' })")
    board_page.wait_for_function(
        "() => document.getElementById('boardMessages').scrollTop < 10",
        timeout=5000,
    )

    new_message = {
        "id": "m9",
        "ts": _iso(TODAY_AM + timedelta(minutes=30)),
        "from": "bob",
        "text": "新着の投稿です。",
        "source": "anima",
        "kind": "anima",
        "channel": "general",
    }
    api.messages.append(new_message)
    board_page.evaluate("(message) => window.__boardWsHandler(message)", new_message)

    post = board_page.locator('[data-board-id="m9"]')
    post.wait_for(timeout=5000)
    assert "board-post-arrive" in (post.get_attribute("class") or "")

    pill = board_page.locator("[data-board-new-pill]")
    pill.wait_for(timeout=5000)
    assert pill.is_visible()
    assert "新着 1 件" in pill.inner_text()

    assert messages.evaluate("element => element.scrollTop") < 100

    pill.click()
    board_page.wait_for_timeout(150)
    assert not pill.is_visible()
    distance = messages.evaluate("element => element.scrollHeight - element.scrollTop - element.clientHeight")
    assert distance < 80


def test_delivery_stages_and_reply_jumps(board_page: Page) -> None:
    _open_channel(board_page, "general")
    _wait_for_messages(board_page)

    post = board_page.locator('[data-board-id="m2"]')
    call = post.locator(".board-delivery-call").filter(has_text="alice")
    assert call.count() == 1
    call_text = call.inner_text()
    assert "届いた" in call_text
    assert "返事" in call_text

    replied = post.locator(".board-delivery-replied")
    assert replied.count() == 1
    assert replied.get_attribute("data-board-jump") == "m2r"
    replied.click()
    board_page.locator('[data-board-id="m2r"].board-post-focus').wait_for(timeout=5000)

    stage_message = {
        "id": "m10",
        "ts": _iso(TODAY_AM + timedelta(minutes=31)),
        "from": "human",
        "text": "配信段階の確認です。",
        "source": "human",
        "kind": "human",
        "channel": "general",
        "notified": ["bob"],
        "not_notified": {},
    }
    api = board_page.api  # type: ignore[attr-defined]
    api.messages.append(stage_message)
    board_page.evaluate("(message) => window.__boardWsHandler(message)", stage_message)
    stage_post = board_page.locator('[data-board-id="m10"]')
    stage_post.wait_for(timeout=5000)
    stage_call = stage_post.locator(".board-delivery-call").filter(has_text="bob")
    assert "未読" in stage_call.inner_text()

    board_page.evaluate(
        """(status) => window.__boardWsHandler(status)""",
        {"name": "bob", "status": "busy"},
    )
    board_page.wait_for_function(
        """() => {
          const post = document.querySelector('[data-board-id="m10"]');
          return post && post.textContent.includes('考え中');
        }""",
        timeout=5000,
    )

    board_page.evaluate(
        """(data) => window.__boardWsHandler(data)""",
        {
            "channel": "general",
            "reader": "anima:bob",
            "upto_id": "m10",
            "upto_ts": _iso(TODAY_AM + timedelta(minutes=31)),
            "picked_id": "m10",
        },
    )
    board_page.wait_for_function(
        """() => {
          const post = document.querySelector('[data-board-id="m10"]');
          return post && post.textContent.includes('読んだ') && post.textContent.includes('返事待ち');
        }""",
        timeout=5000,
    )

    reply_link = board_page.locator('[data-board-id="m1b"] .board-reply-link')
    assert reply_link.count() == 1
    assert "投稿への返事" in reply_link.inner_text()
    reply_link.click()
    board_page.locator('[data-board-id="m1"].board-post-focus').wait_for(timeout=5000)


def test_read_faces_count_names_and_live_drop(board_page: Page) -> None:
    _open_channel(board_page, "general")
    _wait_for_messages(board_page)

    seen = board_page.locator('[data-board-id="m1"] .board-seen')
    seen.wait_for(timeout=5000)
    assert seen.locator(".board-seen-face").count() == 2
    assert "全員が読んだ" in seen.inner_text()

    seen.hover()
    names = seen.locator(".board-seen-names")
    names.wait_for(timeout=5000)
    assert names.is_visible()
    names_text = names.inner_text()
    assert "bob" in names_text
    assert "carol" in names_text

    board_page.evaluate(
        """(data) => window.__boardReadWsHandler(data)""",
        {
            "channel": "general",
            "reader": "anima:alice",
            "upto_id": "m4",
            "upto_ts": _iso(TODAY_AM + timedelta(minutes=14)),
            "picked_id": "m4",
        },
    )
    seen_m4 = board_page.locator('[data-board-id="m4"] .board-seen')
    seen_m4.wait_for(timeout=5000)
    assert seen_m4.locator(".board-seen-face").count() == 1
    assert seen_m4.locator(".board-seen-face.board-is-new").count() == 1
    assert "1人が読んだ" in seen_m4.inner_text()


def test_read_line_stays_and_read_advances(board_page: Page) -> None:
    api = board_page.api  # type: ignore[attr-defined]
    _open_channel(board_page, "general")
    _wait_for_messages(board_page)

    line = board_page.locator('[data-board-id="m1b"] + .board-newline')
    line.wait_for(timeout=5000)
    text = line.inner_text()
    assert "ここから新着" in text
    assert "前回は alice" in text
    assert "投稿まで読みました" in text

    board_page.wait_for_timeout(1400)
    assert api.read_calls
    assert api.read_calls[-1]["upto_id"] == "m4"
    assert api.reads["readers"]["human:human"]["upto_id"] == "m4"
    # The line is fixed for this visit even though the read position moved.
    assert line.count() == 1
    assert "ここから新着" in line.inner_text()
    assert board_page.locator(".board-rail-unread").count() == 0


def test_read_line_fetches_old_position_with_around(board_page: Page) -> None:
    api = board_page.api  # type: ignore[attr-defined]
    base = TODAY_AM + timedelta(minutes=40)
    for index in range(55):
        api.messages.append(
            {
                "id": f"y{index}",
                "ts": _iso(base + timedelta(minutes=index)),
                "from": "bob",
                "text": f"追加の投稿 {index}",
                "source": "anima",
                "kind": "anima",
            }
        )
    # m2 is well outside the newest-50 window.
    api.reads["readers"]["human:human"] = {
        "upto_id": "m2",
        "upto_ts": api.messages[2]["ts"],
        "updated_at": api.messages[2]["ts"],
    }

    _open_channel(board_page, "general")
    _wait_for_messages(board_page)
    line = board_page.locator('[data-board-newline="m2"]')
    line.wait_for(timeout=5000)
    assert "ここから新着" in line.inner_text()
    assert "human" in line.inner_text()


def test_deep_link_opens_channel_and_focuses_post(board_page: Page) -> None:
    board_page.evaluate(
        """async () => {
          const mod = await import('/pages/board.js');
          mod.render(document.getElementById('boardApp'), { subPath: 'general/m5' });
        }"""
    )
    board_page.locator(".board-header-name").filter(has_text="general").wait_for(timeout=5000)
    board_page.locator('[data-board-id="m5"].board-post-focus').wait_for(timeout=5000)


def test_reply_to_unloaded_post_fetches_around(board_page: Page) -> None:
    api = board_page.api  # type: ignore[attr-defined]
    base = TODAY_AM + timedelta(minutes=40)
    for index in range(55):
        api.messages.append(
            {
                "id": f"x{index}",
                "ts": _iso(base + timedelta(minutes=index)),
                "from": "bob",
                "text": f"追加の投稿 {index}",
                "source": "anima",
                "kind": "anima",
            }
        )
    last_ts = base + timedelta(minutes=54)
    api.messages.append(
        {
            "id": "x55",
            "ts": _iso(base + timedelta(minutes=60)),
            "from": "alice",
            "text": "古い投稿への返事です。",
            "source": "anima",
            "kind": "anima",
            "reply_to": "m5",
        }
    )
    api.reads["readers"]["human:human"] = {
        "upto_id": "x54",
        "upto_ts": _iso(last_ts),
        "updated_at": _iso(last_ts),
    }

    _open_channel(board_page, "general")
    board_page.locator('[data-board-id="x55"]').wait_for(timeout=5000)
    assert board_page.locator('[data-board-id="m5"]').count() == 0

    reply = board_page.locator('[data-board-id="x55"] .board-reply-link')
    reply.wait_for(timeout=5000)
    # The target is not loaded yet, so the label falls back to a generic one.
    assert "元の投稿への返事" in reply.inner_text()
    reply.click()

    board_page.locator('[data-board-id="m5"].board-post-focus').wait_for(timeout=5000)


def test_dm_reply_link_jumps_to_original_message(board_page: Page) -> None:
    board_page.locator('[data-board-mode="dm"]').click()
    dm = board_page.locator('[data-board-type="dm"][data-board-name="alice-bob"]')
    dm.wait_for(timeout=5000)
    dm.click()

    reply = board_page.locator('[data-board-id="d2"] .board-reply-link')
    reply.wait_for(timeout=5000)
    text = reply.inner_text()
    assert "alice" in text
    assert "メッセージへの返事" in text

    reply.click()
    board_page.locator('[data-board-id="d1"].board-post-focus').wait_for(timeout=5000)
