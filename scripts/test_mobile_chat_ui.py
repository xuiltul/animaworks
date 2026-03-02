#!/usr/bin/env python3
"""Test mobile responsive chat UI: unified header, layout, scroll-to-bottom."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from playwright.sync_api import sync_playwright

PORTS = [18500, 51325, 8000]  # 18500 default; 51325 if user runs with --port 51325
OUTPUT_DESKTOP = Path("/tmp/animaworks-chat-desktop.png")
OUTPUT_MOBILE = Path("/tmp/animaworks-chat-mobile.png")


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
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
            print("ERROR: Could not connect (tried ports 51325, 18500, 8000)")
            browser.close()
            sys.exit(1)

        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2000)

        # Login if form visible
        login_form = page.locator("#loginForm, .login-form").first
        if login_form.count() > 0 and login_form.is_visible():
            user = os.environ.get("ANIMAWORKS_USER", "taka")
            pw = os.environ.get("ANIMAWORKS_PASS", "")
            if not pw:
                print("WARN: ANIMAWORKS_PASS not set - login may fail.")
            page.fill("#loginUsername, input[placeholder*='ユーザー'], input[name='username']", user)
            page.fill("#loginPassword, input[placeholder*='パスワード'], input[type='password']", pw)
            page.click("button.btn-login, button[type='submit']")
            page.wait_for_timeout(3000)
            if "/chat" not in page.url:
                page.goto(url, wait_until="networkidle")
                page.wait_for_timeout(2000)

        try:
            page.wait_for_selector(".chat-page-layout, #chatPageForm, .chat-anima-tabs-header", timeout=15000)
        except Exception:
            pass
        page.wait_for_timeout(2500)

        # Select Anima if needed
        add_btn = page.locator("#chatAddConversationBtn").first
        if add_btn.count() > 0 and add_btn.is_visible():
            add_btn.click()
            page.wait_for_timeout(1200)
            item = page.locator(".chat-add-conversation-item[data-anima]:not(.disabled)").first
            if item.count() > 0:
                item.click()
                page.wait_for_timeout(2000)

        page.wait_for_timeout(800)

        # 1. Desktop screenshot
        page.screenshot(path=OUTPUT_DESKTOP, full_page=False)
        print(f"Desktop screenshot: {OUTPUT_DESKTOP}")

        # 2. Resize to mobile (iPhone-like)
        page.set_viewport_size({"width": 375, "height": 812})
        page.wait_for_timeout(1500)

        # 3. Mobile screenshot
        page.screenshot(path=OUTPUT_MOBILE, full_page=False)
        print(f"Mobile screenshot: {OUTPUT_MOBILE}")

        # 4. DOM inspection for mobile layout
        result = page.evaluate(
            """() => {
            const out = {
                unifiedHeader: {},
                appHeader: {},
                threadTabs: {},
                scrollToBottom: {},
                viewport: { width: window.innerWidth, height: window.innerHeight }
            };

            // Unified header (hamburger + avatars + thread dropdown + user icon)
            const unified = document.querySelector('.chat-anima-tabs-header');
            if (unified) {
                const rect = unified.getBoundingClientRect();
                out.unifiedHeader.found = true;
                out.unifiedHeader.visible = rect.width > 0 && rect.height > 0;
                out.unifiedHeader.hasHamburger = !!unified.querySelector('.chat-unified-hamburger');
                out.unifiedHeader.hasAnimaTabs = !!unified.querySelector('.anima-tabs');
                out.unifiedHeader.hasThreadDropdown = !!unified.querySelector('.chat-thread-dropdown');
                out.unifiedHeader.hasUserBtn = !!unified.querySelector('.chat-unified-user-btn');
                out.unifiedHeader.hasInfoBtn = !!unified.querySelector('.chat-unified-info-btn');
                out.unifiedHeader.childrenCount = unified.children.length;
            } else {
                out.unifiedHeader.found = false;
            }

            // Original app-header (should be hidden on chat page)
            const appHeader = document.querySelector('.app-header');
            if (appHeader) {
                const style = getComputedStyle(appHeader);
                out.appHeader.found = true;
                out.appHeader.display = style.display;
                out.appHeader.visibility = style.visibility;
                out.appHeader.hidden = style.display === 'none' || style.visibility === 'hidden';
            } else {
                out.appHeader.found = false;
            }

            // Thread tabs row (should be hidden on mobile)
            const threadTabs = document.querySelector('.thread-tabs');
            if (threadTabs) {
                const style = getComputedStyle(threadTabs);
                out.threadTabs.found = true;
                out.threadTabs.display = style.display;
                out.threadTabs.hidden = style.display === 'none';
            } else {
                out.threadTabs.found = false;
            }

            // Scroll-to-bottom button
            const scrollBtn = document.getElementById('chatScrollToBottom');
            if (scrollBtn) {
                const rect = scrollBtn.getBoundingClientRect();
                const form = document.getElementById('chatPageForm');
                const formRect = form ? form.getBoundingClientRect() : null;
                out.scrollToBottom.found = true;
                out.scrollToBottom.visible = scrollBtn.classList.contains('visible');
                out.scrollToBottom.inDOM = true;
                out.scrollToBottom.position = { top: rect.top, bottom: rect.bottom, left: rect.left, right: rect.right };
                out.scrollToBottom.aboveInput = formRect ? rect.bottom <= formRect.top + 5 : null;
            } else {
                out.scrollToBottom.found = false;
            }

            return out;
        }"""
        )

        browser.close()

    # Report
    print("\n" + "=" * 60)
    print("MOBILE CHAT UI TEST REPORT (375x812)")
    print("=" * 60)
    print(f"\nViewport: {result.get('viewport', {})}")

    uh = result.get("unifiedHeader", {})
    print("\n--- Unified Header (hamburger + avatars + thread dropdown + user icon) ---")
    if uh.get("found"):
        print(f"  Found: yes, visible: {uh.get('visible')}")
        print(f"  Hamburger: {uh.get('hasHamburger')}")
        print(f"  Anima avatars: {uh.get('hasAnimaTabs')}")
        print(f"  Thread dropdown: {uh.get('hasThreadDropdown')}")
        print(f"  User icon: {uh.get('hasUserBtn')}")
        print(f"  Info button: {uh.get('hasInfoBtn')}")
        print(f"  Children count: {uh.get('childrenCount')}")
    else:
        print("  NOT FOUND")

    ah = result.get("appHeader", {})
    print("\n--- Original app-header (should be hidden) ---")
    if ah.get("found"):
        print(f"  Display: {ah.get('display')}, hidden: {ah.get('hidden')}")
    else:
        print("  Not found (OK if chat uses different structure)")

    tt = result.get("threadTabs", {})
    print("\n--- Thread tabs row (should be hidden on mobile) ---")
    if tt.get("found"):
        print(f"  Display: {tt.get('display')}, hidden: {tt.get('hidden')}")
    else:
        print("  Not found")

    sb = result.get("scrollToBottom", {})
    print("\n--- Scroll-to-bottom button ---")
    if sb.get("found"):
        print(f"  In DOM: yes, visible (has .visible): {sb.get('visible')}")
        print(f"  Above input: {sb.get('aboveInput')}")
    else:
        print("  NOT FOUND")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
