#!/usr/bin/env python3
"""Capture chat page screenshot to check right sidebar for whitespace."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:18500"
OUTPUT = Path("/tmp/animaworks-chat-sidebar.png")


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()

        page.goto(BASE_URL + "/#/chat", wait_until="networkidle")
        page.wait_for_load_state("domcontentloaded")
        page.wait_for_timeout(2500)  # 2–3 sec load

        # If login form visible, try login with env credentials
        login_form = page.locator("#loginForm, .login-form").first
        if login_form.count() > 0 and login_form.is_visible():
            user = os.environ.get("ANIMAWORKS_USER", "taka")
            pw = os.environ.get("ANIMAWORKS_PASS", "")
            page.fill("#loginUsername, input[placeholder*='ユーザー名']", user)
            page.fill("#loginPassword, input[placeholder*='パスワード']", pw)
            page.click("button.btn-login, button[type='submit']")
            page.wait_for_timeout(2000)
            if page.locator("#loginForm, .login-form").first.is_visible():
                print("Login failed. Set ANIMAWORKS_USER and ANIMAWORKS_PASS env vars.")
                browser.close()
                return

        # Try to select first Anima if dropdown exists
        anima_select = page.locator("#chatPageAnimaSelect, select.anima-dropdown").first
        if anima_select.count() > 0:
            options = anima_select.locator("option")
            opts = [o.get_attribute("value") for o in options.all() if o.get_attribute("value")]
            if opts:
                anima_select.select_option(value=opts[0])
                page.wait_for_timeout(1000)

        # Full page screenshot
        page.screenshot(path=OUTPUT, full_page=False)

        # Optional: element screenshot of right sidebar only
        sidebar = page.locator(".chat-page-sidebar").first
        if sidebar.count() > 0:
            sidebar.screenshot(path=str(OUTPUT).replace(".png", "-sidebar-only.png"))

        browser.close()

    print(f"Screenshot saved: {OUTPUT}")
    if OUTPUT.with_name(OUTPUT.stem + "-sidebar-only.png").exists():
        print(f"Sidebar-only: {OUTPUT.with_name(OUTPUT.stem + '-sidebar-only.png')}")


if __name__ == "__main__":
    main()
