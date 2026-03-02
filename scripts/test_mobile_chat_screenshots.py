#!/usr/bin/env python3
"""Capture mobile chat UI screenshots: unified header, thread dropdown, user menu."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from playwright.sync_api import sync_playwright

PORTS = [18500, 51325, 8000]
MOBILE_VIEWPORT = {"width": 375, "height": 812}


def main() -> None:
    with sync_playwright() as p:
        # Start with mobile viewport so initRightPaneVisibility sees isMobile and hides sidebar
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport=MOBILE_VIEWPORT)
        # Clear localStorage to avoid stale sidebar state
        context.add_init_script("""
            window.addEventListener('load', () => {
                if (window.matchMedia('(max-width: 768px)').matches) {
                    localStorage.removeItem('aw-chat-right-pane-visible');
                }
            });
        """)
        page = context.new_page()

        url = None
        for port in PORTS:
            for base in (f"http://localhost:{port}/#/chat", f"http://127.0.0.1:{port}/#/chat"):
                try:
                    page.goto(base, wait_until="domcontentloaded", timeout=8000)
                    url = base
                    break
                except Exception:
                    continue
            if url:
                break

        if not url:
            print("ERROR: Could not connect")
            browser.close()
            sys.exit(1)

        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2000)

        # Login if needed
        login_form = page.locator("#loginForm, .login-form").first
        if login_form.count() > 0 and login_form.is_visible():
            user = os.environ.get("ANIMAWORKS_USER", "taka")
            pw = os.environ.get("ANIMAWORKS_PASS", "")
            page.fill("#loginUsername, input[placeholder*='ユーザー'], input[name='username']", user)
            page.fill("#loginPassword, input[placeholder*='パスワード'], input[type='password']", pw)
            page.click("button.btn-login, button[type='submit']")
            page.wait_for_timeout(3000)
            if "/chat" not in page.url:
                page.goto(url, wait_until="networkidle")
                page.wait_for_timeout(2000)

        try:
            page.wait_for_selector(".chat-page-layout, .chat-anima-tabs-header", timeout=15000)
        except Exception:
            pass
        page.wait_for_timeout(2000)

        # Select Anima if needed
        add_btn = page.locator("#chatAddConversationBtn").first
        if add_btn.count() > 0 and add_btn.is_visible():
            add_btn.click()
            page.wait_for_timeout(1200)
            item = page.locator(".chat-add-conversation-item[data-anima]:not(.disabled)").first
            if item.count() > 0:
                item.click()
                page.wait_for_timeout(2000)

        # Force sidebar hidden on mobile (in case localStorage had it open)
        page.evaluate("""
            () => {
                const layout = document.getElementById('chatPageLayout');
                const sidebar = document.querySelector('.chat-page-sidebar');
                if (layout && sidebar && window.innerWidth <= 768) {
                    layout.classList.add('sidebar-hidden');
                    sidebar.classList.add('mobile-hidden');
                }
            }
        """)
        page.wait_for_timeout(500)

        # Step 1: Snapshot / screenshot of current state (main chat with unified header)
        out1 = Path("/tmp/animaworks-mobile-unified-header.png")
        page.screenshot(path=out1, full_page=False)
        print(f"Step 1: Main chat with unified header -> {out1}")

        # Step 2: Click thread dropdown
        dd_btn = page.locator("#chatThreadDropdownBtn")
        if dd_btn.count() > 0:
            dd_btn.click()
            page.wait_for_timeout(400)
            out2 = Path("/tmp/animaworks-mobile-thread-dropdown-open.png")
            page.screenshot(path=out2, full_page=False)
            print(f"Step 2: Thread dropdown open -> {out2}")
            # Close dropdown (click outside or press Escape)
            page.keyboard.press("Escape")
            page.wait_for_timeout(300)
        else:
            print("Step 2: chatThreadDropdownBtn not found")

        # Step 3: Click user icon
        user_btn = page.locator("#chatUnifiedUserBtn")
        if user_btn.count() > 0:
            user_btn.click()
            page.wait_for_timeout(400)
            out3 = Path("/tmp/animaworks-mobile-user-menu-open.png")
            page.screenshot(path=out3, full_page=False)
            print(f"Step 3: User menu open -> {out3}")
            page.keyboard.press("Escape")
            page.wait_for_timeout(300)
        else:
            print("Step 3: chatUnifiedUserBtn not found")

        browser.close()

    print("\nDone. Screenshots saved to /tmp/")


if __name__ == "__main__":
    main()
