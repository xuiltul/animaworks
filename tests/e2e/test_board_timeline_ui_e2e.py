# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for the timeline column on the board home.

A real FastAPI app (uvicorn on an ephemeral port) serves the frontend assets and
the production timeline router backed by an isolated temporary data directory.
The rest of the board API is fulfilled by the deterministic fixture used by
``test_board_ui_e2e.py`` so the tests do not touch the real runtime.
"""

from __future__ import annotations

import socket
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlparse

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from core.config.schemas import HubConfig
from core.hub.timeline import append_item
from tests.e2e.test_board_ui_e2e import TEST_HTML, _BoardApi, _find_chromium

try:
    from playwright.sync_api import Browser, Page, sync_playwright
except ImportError:  # pragma: no cover - exercised only without playwright
    Browser = Page = object  # type: ignore[assignment,misc]

STATIC_DIR = Path(__file__).resolve().parents[2] / "server" / "static"
_CHROMIUM = _find_chromium()

pytestmark = pytest.mark.skipif(
    _CHROMIUM is None,
    reason="playwright or a chromium browser binary is not installed",
)


def _create_app(shared_dir: Path) -> FastAPI:
    from server.routes.timeline import create_timeline_router

    app = FastAPI()
    app.state.shared_dir = shared_dir
    app.state.animas_dir = shared_dir.parent / "animas"

    @app.middleware("http")
    async def set_test_user(request, call_next):  # type: ignore[no-untyped-def]
        request.state.user = SimpleNamespace(username="human")
        return await call_next(request)

    @app.get("/board-test.html")
    def board_test() -> HTMLResponse:
        return HTMLResponse(TEST_HTML)

    app.include_router(create_timeline_router(), prefix="/api")
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


@pytest.fixture(scope="module")
def browser_instance():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=_CHROMIUM)
        try:
            yield browser
        finally:
            browser.close()


@pytest.fixture
def timeline_case(tmp_path: Path, monkeypatch):
    """An isolated timeline server with three unread items (one update)."""
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True)
    (tmp_path / "animas").mkdir(parents=True)
    hub = HubConfig(timeline_enabled=True, company="team-a")
    monkeypatch.setattr(
        "core.config.models.load_config",
        lambda: SimpleNamespace(hub=hub),
    )

    now = datetime.now().astimezone()
    with patch("core.hub.timeline.now_iso", return_value=(now - timedelta(minutes=3)).isoformat()):
        original = append_item(
            shared_dir,
            from_name="alice",
            company=hub.company,
            kind="notified",
            text="方針を確認してください。",
            source="anima",
            delivered=False,
        )
    with patch("core.hub.timeline.now_iso", return_value=(now - timedelta(minutes=2)).isoformat()):
        update = append_item(
            shared_dir,
            from_name="alice",
            company=hub.company,
            kind="done",
            text="方針を確認して進めました。",
            source="anima",
            about=original["id"],
        )
    with patch("core.hub.timeline.now_iso", return_value=(now - timedelta(minutes=1)).isoformat()):
        newest = append_item(
            shared_dir,
            from_name="bob",
            company=hub.company,
            kind="decided",
            text="来週の進め方が決まりました。",
            source="anima",
            urgent=True,
        )

    app = _create_app(shared_dir)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started, "timeline test server failed to start"

    case = SimpleNamespace(
        base_url=f"http://127.0.0.1:{port}",
        shared_dir=shared_dir,
        hub=hub,
        original=original,
        update=update,
        newest=newest,
    )
    try:
        yield case
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()


@pytest.fixture
def make_timeline_page(browser_instance: Browser, timeline_case):
    """Factory for pages whose timeline API hits the real isolated server."""
    opened: list[tuple[object, Page]] = []

    def _make(*, enabled: bool = True, viewport: dict[str, int] | None = None) -> Page:
        timeline_case.hub.timeline_enabled = enabled
        api = _BoardApi()
        context = browser_instance.new_context(viewport=viewport or {"width": 1100, "height": 900})
        page = context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))

        def handle(route) -> None:  # type: ignore[no-untyped-def]
            path = urlparse(route.request.url).path
            if path.startswith("/api/timeline"):
                route.continue_()
            else:
                api.handle(route)

        page.route("**/api/**", handle)
        page.goto(f"{timeline_case.base_url}/board-test.html", wait_until="load")
        page.wait_for_function("window.__boardReady === true", timeout=7000)
        page.locator('[data-board-mode="board"]').wait_for(timeout=5000)
        page.locator('[data-board-type="channel"][data-board-name="general"]').wait_for(timeout=5000)
        page.api = api  # type: ignore[attr-defined]
        page.errors = errors  # type: ignore[attr-defined]
        opened.append((context, page))
        return page

    try:
        yield _make
    finally:
        for context, page in opened:
            errors = getattr(page, "errors", [])
            page.close()
            context.close()
            assert not errors


def _timeline_item(page: Page, item_id: str):
    return page.locator(f'[data-timeline-item="{item_id}"]')


def _wait_timeline(page: Page) -> None:
    page.locator("[data-board-timeline]").wait_for(state="visible", timeout=7000)
    page.locator("[data-timeline-title]").wait_for(timeout=7000)


def test_unread_marks_then_read_advances(make_timeline_page, timeline_case) -> None:
    page = make_timeline_page()
    _wait_timeline(page)
    _timeline_item(page, timeline_case.newest["id"]).wait_for(timeout=7000)

    assert page.locator("[data-timeline-unread-mark]").count() == 3
    unread = page.locator("[data-timeline-unread]")
    assert unread.count() == 1
    assert "未読 3件" in unread.inner_text()

    page.wait_for_function(
        "() => document.querySelectorAll('[data-timeline-unread-mark]').length === 0",
        timeout=7000,
    )
    assert page.locator("[data-timeline-unread]").count() == 0

    response = httpx.get(f"{timeline_case.base_url}/api/timeline", params={"limit": 50}, timeout=10)
    assert response.status_code == 200
    data = response.json()
    assert data["unread"] == 0
    assert data["reader"]["upto_id"] == timeline_case.newest["id"]
    assert all(item["read"] for item in data["items"])


def test_update_is_grouped_under_original(make_timeline_page, timeline_case) -> None:
    page = make_timeline_page()
    _wait_timeline(page)
    original = _timeline_item(page, timeline_case.original["id"])
    original.wait_for(timeout=7000)

    update = original.locator(f'[data-timeline-update="{timeline_case.update["id"]}"]')
    update.wait_for(timeout=7000)
    assert "方針を確認して進めました。" in update.inner_text()
    assert page.locator(f'[data-timeline-item="{timeline_case.update["id"]}"]').count() == 0
    assert original.locator("[data-timeline-updates]").count() == 1


def test_refetch_and_ws_add_new_item_once(make_timeline_page, timeline_case) -> None:
    page = make_timeline_page()
    _wait_timeline(page)

    with patch("core.hub.timeline.now_iso", return_value=datetime.now().astimezone().isoformat()):
        fresh = append_item(
            timeline_case.shared_dir,
            from_name="carol",
            company=timeline_case.hub.company,
            kind="done",
            text="再取得で届いた項目です。",
            source="anima",
        )
    page.evaluate("() => window.dispatchEvent(new Event('focus'))")
    row = _timeline_item(page, fresh["id"])
    row.wait_for(timeout=7000)
    first = page.locator("[data-timeline-list] > [data-timeline-item]").first
    assert first.get_attribute("data-timeline-item") == fresh["id"]
    assert page.locator(f'[data-timeline-item="{fresh["id"]}"]').count() == 1

    # A second refetch must not duplicate the same id.
    page.evaluate("() => window.dispatchEvent(new Event('focus'))")
    page.wait_for_timeout(300)
    assert page.locator(f'[data-timeline-item="{fresh["id"]}"]').count() == 1

    live = {
        "id": "tl_0123456789abcdef",
        "ts": datetime.now().astimezone().isoformat(),
        "from": "alice",
        "kind": "decided",
        "text": "WS で即時に届いた項目です。",
        "about": None,
        "urgent": False,
        "ref": None,
        "priority": "normal",
        "delivered": True,
        "read": False,
        "changed": False,
        "updates": [],
    }
    page.evaluate("(item) => window.__boardTimelineWsHandler(item)", live)
    page.evaluate("(item) => window.__boardWsHandler(item)", live)
    live_row = _timeline_item(page, live["id"])
    live_row.wait_for(timeout=7000)
    assert page.locator(f'[data-timeline-item="{live["id"]}"]').count() == 1
    first = page.locator("[data-timeline-list] > [data-timeline-item]").first
    assert first.get_attribute("data-timeline-item") == live["id"]


def test_timeline_hidden_when_disabled(make_timeline_page) -> None:
    page = make_timeline_page(enabled=False)
    page.wait_for_timeout(300)
    assert not page.locator("[data-board-timeline]").is_visible()
    assert page.locator("[data-timeline-title]").count() == 0
    assert page.locator("[data-timeline-item]").count() == 0
    # The ordinary board home is still there.
    assert page.locator(".board-home-card").count() > 0


def test_timeline_does_not_break_at_400px(make_timeline_page, timeline_case) -> None:
    page = make_timeline_page(enabled=True, viewport={"width": 400, "height": 700})
    _wait_timeline(page)
    _timeline_item(page, timeline_case.newest["id"]).wait_for(timeout=7000)
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    width = page.locator("[data-board-timeline]").evaluate("(element) => element.getBoundingClientRect().width")
    assert width <= 400
    text = page.evaluate(
        "() => document.querySelector('[data-board-timeline]').textContent.includes('会社のタイムライン')"
    )
    assert text is True
    # The floating rail button must not cover the timeline header on phones.
    assert page.evaluate(
        """() => {
          const timeline = document.querySelector('[data-board-timeline]').getBoundingClientRect();
          const toggle = document.querySelector('#boardMobileToggle').getBoundingClientRect();
          return timeline.top >= toggle.bottom;
        }"""
    )
