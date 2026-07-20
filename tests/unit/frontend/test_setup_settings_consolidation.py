# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Frontend structure checks for SPA #/setup → #/settings consolidation."""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATIC = PROJECT_ROOT / "server" / "static"
SETTINGS_JS = STATIC / "pages" / "settings.js"
SETUP_JS = STATIC / "pages" / "setup.js"
ROUTER_JS = STATIC / "modules" / "router.js"
INDEX_HTML = STATIC / "index.html"
I18N_DIR = STATIC / "i18n"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_spa_setup_page_removed() -> None:
    assert not SETUP_JS.exists()


def test_router_has_no_setup_route_registration() -> None:
    js = _read(ROUTER_JS)
    # No dynamic import of pages/setup.js / no routes["/setup"] registration
    assert 'routes["/setup"]' not in js
    assert "pages/setup.js" not in js
    # Legacy hash access redirects to settings
    assert 'window.location.hash = "#/settings"' in js
    assert 'path === "/setup"' in js


def test_nav_has_settings_only_no_setup() -> None:
    html = _read(INDEX_HTML)
    assert 'data-route="/setup"' not in html
    assert 'data-i18n="nav.setup"' not in html
    assert 'href="#/setup"' not in html
    assert 'data-route="/settings"' in html
    assert 'data-i18n="nav.settings"' in html
    # Exactly one settings nav entry
    assert html.count('data-route="/settings"') == 1


def test_settings_page_has_api_auth_section() -> None:
    js = _read(SETTINGS_JS)
    assert 'settings.api_auth.title' in js
    assert 'id="settingsApiAuthSection"' in js
    assert 'id="settingsApiKeys"' in js
    assert 'id="anthropicAuthSettings"' in js
    assert 'id="openaiAuthSettings"' in js
    assert 'id="cliToolsAuth"' in js
    assert 'id="authSettings"' in js
    # Auth save / status APIs used by settings UI
    assert "/api/settings/anthropic-auth" in js
    assert "/api/settings/openai-auth" in js
    assert "/api/setup/environment" in js
    assert "/api/system/config" in js
    # Checklist discarded
    assert "setupChecklist" not in js
    assert "_loadChecklist" not in js


def test_existing_settings_sections_still_present() -> None:
    js = _read(SETTINGS_JS)
    for key in (
        "settings.activity_level.title",
        "settings.mode.title",
        "settings.theme.title",
        "settings.font_size.title",
        "settings.input.title",
    ):
        assert key in js


def test_i18n_auth_keys_under_settings_no_setup_keys() -> None:
    required = [
        "settings.api_auth.title",
        "settings.api_auth.api_keys",
        "settings.api_auth.anthropic_title",
        "settings.api_auth.openai_title",
        "settings.api_auth.cli_title",
        "settings.api_auth.auth_settings",
        "settings.api_auth.network_error",
        "nav.settings",
        "app.auth_banner",
    ]
    forbidden_prefixes = ("setup.",)
    for locale in ("en", "ja", "ko"):
        data = json.loads((I18N_DIR / f"{locale}.json").read_text(encoding="utf-8"))
        for key in required:
            assert key in data, f"missing {key} in {locale}"
            assert data[key], f"empty {key} in {locale}"
        for key in data:
            for prefix in forbidden_prefixes:
                assert not key.startswith(prefix), f"leftover {key} in {locale}"
        assert "nav.setup" not in data
        # Password banner points at settings, not setup
        assert "#/settings" in data["app.auth_banner"]
        assert "#/setup" not in data["app.auth_banner"]


def test_router_setup_redirect_pattern() -> None:
    """Ensure redirect is for SPA hash /setup only (not bare string matches elsewhere)."""
    js = _read(ROUTER_JS)
    # Redirect block exists before route lookup
    m = re.search(
        r'if \(path === "/setup" \|\| path\.startsWith\("/setup/"\)\)\s*\{[^}]*#/settings',
        js,
        re.DOTALL,
    )
    assert m is not None
