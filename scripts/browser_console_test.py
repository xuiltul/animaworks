#!/usr/bin/env python3
"""Check local AnimaWorks UI for console errors and test Business toggle."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root for playwright
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:18500"
OUTPUT_DIR = Path("/tmp/animaworks-browser-test")


def main() -> None:
    console_logs: list[dict] = []

    def on_console(msg):
        console_logs.append({
            "type": msg.type,
            "text": msg.text,
            "location": msg.location,
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Desktop viewport so sidebar is visible (not hidden on mobile)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()
        page.on("console", on_console)

        # 1. Navigate to homepage
        report.append("=== Step 1: Homepage ===")
        page.goto(BASE_URL + "/", wait_until="networkidle")
        page.wait_for_load_state("domcontentloaded")
        page.wait_for_timeout(1500)

        page.screenshot(path=OUTPUT_DIR / "01-homepage.png", full_page=True)
        errors_home = [m for m in console_logs if m["type"] == "error"]
        if errors_home:
            report.append(f"Console errors on homepage: {len(errors_home)}")
            for e in errors_home:
                report.append(f"  - {e['text'][:200]}")
        else:
            report.append("No console errors on homepage")

        # 2. Navigate to chat
        report.append("\n=== Step 2: Chat page ===")
        page.goto(BASE_URL + "/#/chat", wait_until="networkidle")
        page.wait_for_load_state("domcontentloaded")
        page.wait_for_timeout(1500)

        page.screenshot(path=OUTPUT_DIR / "02-chat.png", full_page=True)
        errors_chat = [m for m in console_logs if m["type"] == "error"]
        report.append(f"Total console errors so far: {len(errors_chat)}")
        new_errors = [e for e in errors_chat if e not in errors_home] if errors_home else errors_chat
        if new_errors:
            for e in new_errors[-5:]:  # last 5
                report.append(f"  - {e['text'][:200]}")

        # 3. Anima dropdown
        report.append("\n=== Step 3: Anima selection ===")
        anima_select = page.locator('select[id*="anima"], select.anima-select, [data-anima-select]').first
        if anima_select.count() > 0:
            options = anima_select.locator("option")
            opts = options.all_inner_texts()
            if len(opts) > 1:
                anima_select.select_option(index=1)
                page.wait_for_timeout(800)
                page.screenshot(path=OUTPUT_DIR / "03-anima-selected.png", full_page=True)
                report.append(f"Selected Anima option: {opts[1] if len(opts) > 1 else opts[0]}")
            else:
                report.append("Anima dropdown has no selectable options")
        else:
            # Try alternate selectors
            sel = page.locator("select").first
            if sel.count() > 0:
                opts = sel.locator("option").all_inner_texts()
                if opts:
                    sel.select_option(index=min(1, len(opts) - 1))
                    page.wait_for_timeout(800)
                    page.screenshot(path=OUTPUT_DIR / "03-anima-selected.png", full_page=True)
                    report.append(f"Selected from first select: {opts}")
            report.append("No Anima dropdown found with common selectors")

        errors_after_anima = [m for m in console_logs if m["type"] == "error"]
        report.append(f"Console errors after Anima selection: {len(errors_after_anima)}")

        # 4. Business toggle (input has opacity:0 for custom styling; click label/slider)
        report.append("\n=== Step 4: Business toggle ===")
        hamburger = page.locator("#hamburgerBtn")
        if hamburger.count() > 0 and hamburger.is_visible():
            hamburger.click()
            page.wait_for_timeout(400)

        # Click the visible label/slider (input is opacity:0, so use force or parent)
        theme_toggle = page.locator("#themeToggle, .theme-toggle, label.theme-switch").first
        if theme_toggle.count() > 0:
            theme_toggle.click(timeout=5000)
            report.append("Clicked Business toggle area")
        else:
            page.evaluate("document.getElementById('themeSwitch')?.click()")
            report.append("Clicked themeSwitch via JS")

        page.wait_for_timeout(800)
        page.screenshot(path=OUTPUT_DIR / "04-business-toggled.png", full_page=True)

        has_business_class = page.evaluate("document.body.classList.contains('theme-business')")
        report.append(f"Theme business active: {has_business_class}")

        # Check for emoji in sidebar
        sidebar = page.locator(".sidebar, .app-sidebar, nav, [class*='sidebar']").first
        if sidebar.count() > 0:
            sidebar_text = sidebar.inner_text()
            report.append(f"Sidebar contains emoji (sample): {('😀' in sidebar_text or '🎯' in sidebar_text or '📋' in sidebar_text)}")
        report.append(f"Screenshots saved to {OUTPUT_DIR}")

        browser.close()

    # Final summary
    report.append("\n=== All console errors ===")
    all_errors = [m for m in console_logs if m["type"] == "error"]
    for i, e in enumerate(all_errors):
        report.append(f"{i+1}. {e['text'][:300]}")

    report.append("\n=== Console log types seen ===")
    from collections import Counter
    types = Counter(m["type"] for m in console_logs)
    for t, c in types.most_common():
        report.append(f"  {t}: {c}")

    text = "\n".join(report)
    print(text)
    (OUTPUT_DIR / "report.txt").write_text(text, encoding="utf-8")
    print(f"\nReport also saved to {OUTPUT_DIR / 'report.txt'}")


if __name__ == "__main__":
    main()
