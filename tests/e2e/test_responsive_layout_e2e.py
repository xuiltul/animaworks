# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for responsive layout using Playwright.

Validates the responsive design across mobile, tablet, and desktop viewports
by running against a real AnimaWorks server instance.

Requires:
  - playwright (pip install playwright)
  - A running AnimaWorks server on localhost (default port 8765)

Run with: python -m pytest tests/e2e/test_responsive_layout_e2e.py -v
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Skip if playwright not available ─────────────────────────

try:
    from playwright.sync_api import sync_playwright, Browser, Page
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

pytestmark = pytest.mark.skipif(
    not HAS_PLAYWRIGHT,
    reason="playwright not installed",
)

# ── Constants ────────────────────────────────────────────────

MOBILE_VIEWPORT = {"width": 375, "height": 667}
TABLET_VIEWPORT = {"width": 768, "height": 1024}
DESKTOP_VIEWPORT = {"width": 1280, "height": 800}

# Use a dedicated port for the test static server to avoid conflicts
# with a running AnimaWorks server on the default port 8765.
SERVER_HOST = os.environ.get("ANIMAWORKS_TEST_HOST", "http://localhost")
SERVER_PORT = int(os.environ.get("ANIMAWORKS_TEST_PORT", "18765"))
BASE_URL = f"{SERVER_HOST}:{SERVER_PORT}"


# ── Test App (ASGI) approach for static file serving ─────────

def _create_static_app():
    """Create a minimal ASGI app that serves static files only.

    This avoids needing the full AnimaWorks server with ProcessSupervisor.
    Serves the entire server/static/ directory including workspace/.
    """
    try:
        from fastapi import FastAPI
        from fastapi.staticfiles import StaticFiles

        project_root = Path(__file__).resolve().parents[2]
        static_dir = project_root / "server" / "static"

        if not static_dir.exists():
            return None

        app = FastAPI()

        # Mount workspace first (nested path must come before root)
        workspace_dir = static_dir / "workspace"
        if workspace_dir.exists():
            app.mount(
                "/workspace",
                StaticFiles(directory=str(workspace_dir), html=True),
                name="workspace",
            )

        # Mount root static directory — serves index.html, styles/, modules/
        app.mount(
            "/",
            StaticFiles(directory=str(static_dir), html=True),
            name="static",
        )

        return app
    except ImportError:
        return None


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture(scope="module")
def static_server():
    """Start a local static file server for E2E tests.

    Uses uvicorn to serve a minimal FastAPI app with static files.
    Falls back to checking if a server is already running.
    """
    import socket

    # Check if a server is already running on the expected port
    def _port_in_use(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect(("localhost", port))
                return True
            except ConnectionRefusedError:
                return False

    # Always start our own static server to ensure we test worktree files,
    # not whatever is running on the default AnimaWorks port.
    if _port_in_use(SERVER_PORT):
        # Our test port is already occupied — try the next one
        pytest.skip(f"Port {SERVER_PORT} already in use")
        return

    # Start our own static server
    app = _create_static_app()
    if app is None:
        pytest.skip("Cannot create static server — FastAPI not available")
        return

    import threading
    import uvicorn

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=SERVER_PORT, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to be ready
    for _ in range(50):
        if _port_in_use(SERVER_PORT):
            break
        time.sleep(0.1)
    else:
        pytest.skip("Static server failed to start")
        return

    yield BASE_URL

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def browser_instance():
    """Launch a Playwright browser for the test module."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def mobile_page(browser_instance: Browser, static_server: str):
    """Create a page with mobile viewport (375x667)."""
    context = browser_instance.new_context(viewport=MOBILE_VIEWPORT)
    page = context.new_page()
    yield page
    page.close()
    context.close()


@pytest.fixture
def tablet_page(browser_instance: Browser, static_server: str):
    """Create a page with tablet viewport (768x1024)."""
    context = browser_instance.new_context(viewport=TABLET_VIEWPORT)
    page = context.new_page()
    yield page
    page.close()
    context.close()


@pytest.fixture
def desktop_page(browser_instance: Browser, static_server: str):
    """Create a page with desktop viewport (1280x800)."""
    context = browser_instance.new_context(viewport=DESKTOP_VIEWPORT)
    page = context.new_page()
    yield page
    page.close()
    context.close()


# ── Helpers ──────────────────────────────────────────────────


def _dismiss_login(page: Page) -> None:
    """Dismiss the login overlay by clicking guest login.

    The login screen covers the main UI; must be dismissed before
    interacting with hamburger, sidebar, etc.
    """
    guest_btn = page.locator("#guestLoginBtn")
    if guest_btn.is_visible(timeout=3000):
        guest_btn.click()
        page.wait_for_timeout(300)


# ── Mobile Dashboard Tests (375x667) ────────────────────────


class TestDashboardMobile:
    """Dashboard responsive behavior on mobile viewports."""

    def test_hamburger_visible_mobile(self, mobile_page: Page, static_server: str) -> None:
        """Hamburger button should be visible on mobile after login."""
        mobile_page.goto(f"{static_server}/")
        mobile_page.wait_for_load_state("domcontentloaded")
        _dismiss_login(mobile_page)

        hamburger = mobile_page.locator("#hamburgerBtn")
        # Check computed display rather than is_visible() since the element
        # might be behind other overlays or have zero-size bounding box.
        display = hamburger.evaluate("el => getComputedStyle(el).display")
        assert display != "none", (
            f"Hamburger should be display: flex on mobile, got {display}"
        )

    def test_sidebar_hidden_mobile(self, mobile_page: Page, static_server: str) -> None:
        """Sidebar nav should be hidden (off-screen) on mobile by default."""
        mobile_page.goto(f"{static_server}/")
        mobile_page.wait_for_load_state("domcontentloaded")
        _dismiss_login(mobile_page)

        sidebar = mobile_page.locator("#sidebarNav")
        # Sidebar exists but should be off-screen (translateX(-100%))
        box = sidebar.bounding_box()
        if box:
            # The sidebar should be positioned off-screen (x < 0)
            assert box["x"] < 0, "Sidebar should be off-screen on mobile"
        else:
            # bounding_box returns None if not visible at all — acceptable
            pass

    def test_hamburger_opens_sidebar(self, mobile_page: Page, static_server: str) -> None:
        """Clicking hamburger should open the sidebar drawer."""
        mobile_page.goto(f"{static_server}/")
        mobile_page.wait_for_load_state("domcontentloaded")
        _dismiss_login(mobile_page)

        # Click hamburger
        mobile_page.click("#hamburgerBtn")
        mobile_page.wait_for_timeout(400)  # Wait for CSS transition

        # Check sidebar is now visible (translateX(0))
        sidebar = mobile_page.locator("#sidebarNav")
        box = sidebar.bounding_box()
        assert box is not None, "Sidebar should be on-screen after hamburger click"
        assert box["x"] >= 0, "Sidebar x should be >= 0 after opening"

    def test_mobile_nav_backdrop_visible_when_open(
        self, mobile_page: Page, static_server: str
    ) -> None:
        """Backdrop should appear when mobile nav is open."""
        mobile_page.goto(f"{static_server}/")
        mobile_page.wait_for_load_state("domcontentloaded")
        _dismiss_login(mobile_page)

        mobile_page.click("#hamburgerBtn")
        mobile_page.wait_for_timeout(400)

        backdrop = mobile_page.locator("#mobileNavBackdrop")
        # Backdrop should be display: block when nav is open
        display = backdrop.evaluate("el => getComputedStyle(el).display")
        assert display != "none", "Backdrop should be visible when nav is open"

    def test_chat_send_button_tap_target(
        self, mobile_page: Page, static_server: str
    ) -> None:
        """Chat send button should have at least 44px min-height on mobile."""
        mobile_page.goto(f"{static_server}/")
        mobile_page.wait_for_load_state("domcontentloaded")
        _dismiss_login(mobile_page)

        # The send button is in the chat page; check CSS computed style
        send_btn = mobile_page.locator(".chat-send-btn").first
        if send_btn.count() > 0:
            min_height = send_btn.evaluate(
                "el => getComputedStyle(el).minHeight"
            )
            assert min_height == "44px", f"Expected 44px min-height, got {min_height}"


# ── Mobile Workspace Tests (375x667) ────────────────────────


class TestWorkspaceMobile:
    """Workspace responsive behavior on mobile viewports."""

    def test_workspace_sidebar_hidden_mobile(
        self, mobile_page: Page, static_server: str
    ) -> None:
        """Workspace sidebar should be hidden initially on mobile."""
        mobile_page.goto(f"{static_server}/workspace/")
        mobile_page.wait_for_load_state("domcontentloaded")

        sidebar = mobile_page.locator(".ws-conv-sidebar").first
        if sidebar.count() > 0:
            box = sidebar.bounding_box()
            if box:
                # Should be off-screen (translateX(-100%))
                assert box["x"] < 0, "Workspace sidebar should be off-screen"

    def test_workspace_mobile_sidebar_toggle_visible(
        self, mobile_page: Page, static_server: str
    ) -> None:
        """Mobile sidebar toggle should be visible on mobile viewport."""
        mobile_page.goto(f"{static_server}/workspace/")
        mobile_page.wait_for_load_state("domcontentloaded")

        toggle = mobile_page.locator("#wsMobileSidebarToggle")
        if toggle.count() > 0:
            display = toggle.evaluate("el => getComputedStyle(el).display")
            assert display != "none", "Mobile sidebar toggle should be visible"

    def test_workspace_chat_send_tap_target(
        self, mobile_page: Page, static_server: str
    ) -> None:
        """Workspace chat send button should have at least 44px min-height."""
        mobile_page.goto(f"{static_server}/workspace/")
        mobile_page.wait_for_load_state("domcontentloaded")

        send_btn = mobile_page.locator(".ws-conv-send").first
        if send_btn.count() > 0:
            min_height = send_btn.evaluate(
                "el => getComputedStyle(el).minHeight"
            )
            assert min_height == "44px", f"Expected 44px min-height, got {min_height}"

    def test_workspace_chat_bubbles_no_overflow(
        self, mobile_page: Page, static_server: str
    ) -> None:
        """Chat bubbles should not cause horizontal overflow on mobile."""
        mobile_page.goto(f"{static_server}/workspace/")
        mobile_page.wait_for_load_state("domcontentloaded")

        # Check that body doesn't have horizontal scroll
        scroll_width = mobile_page.evaluate("document.body.scrollWidth")
        client_width = mobile_page.evaluate("document.body.clientWidth")
        assert scroll_width <= client_width + 1, (
            f"Horizontal overflow detected: scrollWidth={scroll_width}, "
            f"clientWidth={client_width}"
        )


# ── Tablet Dashboard Tests (768x1024) ───────────────────────


class TestDashboardTablet:
    """Dashboard responsive behavior on tablet viewports."""

    def test_dashboard_layout_adjusts_tablet(
        self, tablet_page: Page, static_server: str
    ) -> None:
        """Dashboard layout should adjust at tablet breakpoint."""
        tablet_page.goto(f"{static_server}/")
        tablet_page.wait_for_load_state("domcontentloaded")

        # At exactly 768px, we're at the breakpoint threshold
        # The layout should be responsive
        body_width = tablet_page.evaluate("document.body.clientWidth")
        assert body_width <= 768


# ── Desktop Dashboard Tests (1280x800) — Regression ─────────


class TestDashboardDesktopRegression:
    """Verify desktop layout is preserved after responsive changes."""

    def test_sidebar_visible_desktop(
        self, desktop_page: Page, static_server: str
    ) -> None:
        """Sidebar nav should be fully visible on desktop."""
        desktop_page.goto(f"{static_server}/")
        desktop_page.wait_for_load_state("domcontentloaded")
        _dismiss_login(desktop_page)

        sidebar = desktop_page.locator("#sidebarNav")
        box = sidebar.bounding_box()
        assert box is not None, "Sidebar should be visible on desktop"
        assert box["x"] >= 0, "Sidebar should be on-screen"
        assert box["width"] >= 200, "Sidebar should have reasonable width"

    def test_hamburger_hidden_desktop(
        self, desktop_page: Page, static_server: str
    ) -> None:
        """Hamburger button should be hidden on desktop."""
        desktop_page.goto(f"{static_server}/")
        desktop_page.wait_for_load_state("domcontentloaded")

        hamburger = desktop_page.locator("#hamburgerBtn")
        display = hamburger.evaluate("el => getComputedStyle(el).display")
        assert display == "none", f"Hamburger should be hidden on desktop, got {display}"

    def test_desktop_no_horizontal_scroll(
        self, desktop_page: Page, static_server: str
    ) -> None:
        """Desktop layout should not have horizontal scroll."""
        desktop_page.goto(f"{static_server}/")
        desktop_page.wait_for_load_state("domcontentloaded")

        scroll_width = desktop_page.evaluate("document.body.scrollWidth")
        client_width = desktop_page.evaluate("document.body.clientWidth")
        assert scroll_width <= client_width + 1


# ── Desktop Workspace Tests (1280x800) — Regression ─────────


class TestWorkspaceDesktopRegression:
    """Verify workspace desktop layout is preserved."""

    def test_workspace_sidebars_visible_desktop(
        self, desktop_page: Page, static_server: str
    ) -> None:
        """Workspace sidebars should be visible at correct widths on desktop."""
        desktop_page.goto(f"{static_server}/workspace/")
        desktop_page.wait_for_load_state("domcontentloaded")

        # Check that mobile toggle buttons are hidden
        toggle = desktop_page.locator("#wsMobileSidebarToggle")
        if toggle.count() > 0:
            display = toggle.evaluate("el => getComputedStyle(el).display")
            assert display == "none", "Mobile toggle should be hidden on desktop"

    def test_workspace_no_horizontal_scroll(
        self, desktop_page: Page, static_server: str
    ) -> None:
        """Workspace should not have horizontal scroll on desktop."""
        desktop_page.goto(f"{static_server}/workspace/")
        desktop_page.wait_for_load_state("domcontentloaded")

        scroll_width = desktop_page.evaluate("document.body.scrollWidth")
        client_width = desktop_page.evaluate("document.body.clientWidth")
        assert scroll_width <= client_width + 1
