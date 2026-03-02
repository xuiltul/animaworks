#!/usr/bin/env python3
"""Verify desktop chat layout: app-header visible, anima tabs normal, thread tabs visible, mobile elements hidden."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from playwright.sync_api import sync_playwright

PORTS = [18500, 51325, 8000]
OUTPUT = Path("/tmp/animaworks-desktop-chat.png")
DESKTOP_VIEWPORT = {"width": 1280, "height": 900}


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport=DESKTOP_VIEWPORT)
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
            page.wait_for_selector(".chat-page-layout, .app-header", timeout=15000)
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

        # Screenshot
        page.screenshot(path=OUTPUT, full_page=False)
        print(f"Screenshot: {OUTPUT}")

        # DOM verification
        result = page.evaluate(
            """() => {
            const out = {
                viewport: { width: window.innerWidth, height: window.innerHeight },
                appHeader: {},
                animaTabs: {},
                threadTabs: {},
                mobileElements: {}
            };

            // App header (should be visible on desktop)
            const appHeader = document.querySelector('.app-header');
            if (appHeader) {
                const style = getComputedStyle(appHeader);
                out.appHeader.found = true;
                out.appHeader.display = style.display;
                out.appHeader.visible = style.display !== 'none';
            } else {
                out.appHeader.found = false;
            }

            // Anima tabs - should show avatars + names
            const animaTabs = document.querySelector('.chat-anima-tabs-header .anima-tabs');
            if (animaTabs) {
                const style = getComputedStyle(animaTabs);
                const names = animaTabs.querySelectorAll('.anima-tab-name');
                out.animaTabs.found = true;
                out.animaTabs.nameDisplay = names.length ? getComputedStyle(names[0]).display : 'N/A';
                out.animaTabs.tabCount = animaTabs.querySelectorAll('.anima-tab').length;
            } else {
                out.animaTabs.found = false;
            }

            // Thread tabs row (should be visible on desktop)
            const threadTabs = document.querySelector('.thread-tabs');
            if (threadTabs) {
                const style = getComputedStyle(threadTabs);
                out.threadTabs.found = true;
                out.threadTabs.display = style.display;
                out.threadTabs.visible = style.display !== 'none';
            } else {
                out.threadTabs.found = false;
            }

            // Mobile-only elements (should be hidden on desktop)
            const hamburger = document.querySelector('.chat-unified-hamburger');
            const threadDropdown = document.querySelector('.chat-thread-dropdown');
            const infoBtn = document.querySelector('.chat-unified-info-btn');
            const userBtn = document.querySelector('.chat-unified-user-btn');
            out.mobileElements.hamburgerHidden = hamburger ? getComputedStyle(hamburger).display === 'none' : true;
            out.mobileElements.threadDropdownHidden = threadDropdown ? getComputedStyle(threadDropdown).display === 'none' : true;
            out.mobileElements.infoBtnHidden = infoBtn ? getComputedStyle(infoBtn).display === 'none' : true;
            out.mobileElements.userBtnHidden = userBtn ? getComputedStyle(userBtn).display === 'none' : true;
            out.mobileElements.allHidden = out.mobileElements.hamburgerHidden && out.mobileElements.threadDropdownHidden
                && out.mobileElements.infoBtnHidden && out.mobileElements.userBtnHidden;

            return out;
        }"""
        )

        browser.close()

    # Report
    print("\n" + "=" * 60)
    print("DESKTOP CHAT LAYOUT VERIFICATION (1280x900)")
    print("=" * 60)
    print(f"\nViewport: {result.get('viewport')}")

    ah = result.get("appHeader", {})
    print("\n--- App header (should be visible) ---")
    if ah.get("found"):
        print(f"  Display: {ah.get('display')}, visible: {ah.get('visible')}")
    else:
        print("  NOT FOUND")

    at = result.get("animaTabs", {})
    print("\n--- Anima tabs (avatars + names) ---")
    if at.get("found"):
        print(f"  Tab count: {at.get('tabCount')}, name display: {at.get('nameDisplay')}")
    else:
        print("  NOT FOUND")

    tt = result.get("threadTabs", {})
    print("\n--- Thread tabs row (should be visible) ---")
    if tt.get("found"):
        print(f"  Display: {tt.get('display')}, visible: {tt.get('visible')}")
    else:
        print("  NOT FOUND")

    me = result.get("mobileElements", {})
    print("\n--- Mobile-only elements (should be HIDDEN) ---")
    print(f"  Hamburger hidden: {me.get('hamburgerHidden')}")
    print(f"  Thread dropdown hidden: {me.get('threadDropdownHidden')}")
    print(f"  Info button hidden: {me.get('infoBtnHidden')}")
    print(f"  User button hidden: {me.get('userBtnHidden')}")
    print(f"  All mobile elements hidden: {me.get('allHidden')}")

    all_ok = (
        ah.get("visible")
        and tt.get("visible")
        and me.get("allHidden")
    )
    print("\n" + ("PASS: Desktop layout correct" if all_ok else "FAIL: Some checks failed"))
    print("=" * 60)


if __name__ == "__main__":
    main()
