# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Settings "Secrets" tab (settings-secrets.js).

Validates that:
- settings.js registers the secrets tab between api and users and loads it
  with the same dynamic-import shape as the users tab
- settings-secrets.js exists and exports render/destroy
- every t("...") key used by settings-secrets.js exists in ja/en/ko
- the secrets UI does not use window.confirm/alert and never interpolates
  unescaped values into HTML (only ${t(...)}, ${escapeHtml(...)}, and a small
  allow-list of internally-built HTML snippets)
- settings.css braces are balanced
- the API tab links to #/settings/secrets
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_STATIC_DIR = Path(__file__).resolve().parents[4] / "server" / "static"
_SETTINGS_JS = _STATIC_DIR / "pages" / "settings.js"
_SECRETS_JS = _STATIC_DIR / "pages" / "settings-secrets.js"
_SETTINGS_CSS = _STATIC_DIR / "styles" / "settings.css"
_I18N_DIR = _STATIC_DIR / "i18n"
_LOCALES = ("ja", "en", "ko")

# Variables that are themselves assembled from escaped/i18n fragments and are
# safe to interpolate verbatim into the template strings.
_ALLOWED_INTERPOLATIONS = {
    "writeWarningHtml",
    "onepasswordHtml",
    "syncPanelHtml",
    "entriesHtml",
    "addHtml",
    "animaHtml",
    "hubHtml",
    "plaintextHtml",
    "planRowsHtml",
    "rowsHtml",
    "actionsHtml",
    "ownerHtml",
    "webhookHtml",
    "urlHtml",
    "keyHtml",
    "enabledHtml",
    "errorText",
    "passwordFieldHtml",
    "editFormHtml",
    "deleteFormHtml",
    "rollbackFormHtml",
    "reissueFormHtml",
    "tokenNoticeHtml",
    "disabledAttr",
}


def _read(path: Path) -> str:
    assert path.exists(), f"missing file: {path}"
    return path.read_text(encoding="utf-8")


def _translations(locale: str) -> dict[str, str]:
    return json.loads(_read(_I18N_DIR / f"{locale}.json"))


class TestSettingsTabRegistration:
    """settings.js must host the secrets tab like the users tab."""

    @pytest.fixture(autouse=True)
    def _load_source(self) -> None:
        self.source = _read(_SETTINGS_JS)

    def test_secrets_tab_defined(self) -> None:
        assert '{ id: "secrets", labelKey: "settings.tab_secrets" }' in self.source

    def test_secrets_tab_between_api_and_users(self) -> None:
        api = self.source.index('{ id: "api", labelKey: "settings.tab_api" }')
        secrets = self.source.index('{ id: "secrets", labelKey: "settings.tab_secrets" }')
        users = self.source.index('{ id: "users", labelKey: "settings.tab_users" }')
        assert api < secrets < users

    def test_secrets_module_loaded_dynamically(self) -> None:
        assert 'import("./settings-secrets.js")' in self.source
        assert 'typeof mod.render === "function"' in self.source
        assert 'typeof _activeTabModule.destroy === "function"' in self.source

    def test_secrets_tab_is_routed(self) -> None:
        assert 'if (_activeTab === "secrets")' in self.source

    def test_api_tab_links_to_secrets_tab(self) -> None:
        assert 'href="#/settings/secrets"' in self.source
        assert 't("settings.api_auth.secrets_link")' in self.source


class TestSecretsModuleExports:
    """settings-secrets.js must expose render/destroy."""

    @pytest.fixture(autouse=True)
    def _load_source(self) -> None:
        self.source = _read(_SECRETS_JS)

    def test_exports_render(self) -> None:
        assert re.search(r"export function render\(", self.source)

    def test_exports_destroy(self) -> None:
        assert re.search(r"export function destroy\(", self.source)

    def test_imports_escape_html(self) -> None:
        assert 'import { escapeHtml } from "../modules/state.js";' in self.source

    def test_imports_i18n_t(self) -> None:
        assert 'import { t } from "/shared/i18n.js";' in self.source

    def test_uses_api_helper(self) -> None:
        assert 'import { api } from "../modules/api.js";' in self.source

    def test_no_window_confirm(self) -> None:
        assert "window.confirm" not in self.source
        assert "confirm(" not in self.source

    def test_no_alert(self) -> None:
        assert "alert(" not in self.source

    def test_key_name_pattern_validated(self) -> None:
        assert r"/^[A-Za-z_][A-Za-z0-9_]{0,127}$/" in self.source

    def test_token_once_only(self) -> None:
        assert "navigator.clipboard.writeText" in self.source
        assert "secrets.hub.token_once" in self.source
        assert "close-token" in self.source


class TestSecretsInterpolations:
    """No unescaped value may be interpolated into HTML."""

    @pytest.fixture(autouse=True)
    def _load_source(self) -> None:
        self.source = _read(_SECRETS_JS)

    def test_all_interpolations_escaped_or_allowed(self) -> None:
        offenders = []
        for match in re.finditer(r"\$\{", self.source):
            rest = self.source[match.end() :]
            if rest.startswith("t(") or rest.startswith("escapeHtml("):
                continue
            if any(rest.startswith(f"{name}}}") for name in _ALLOWED_INTERPOLATIONS):
                continue
            line_no = self.source[: match.start()].count("\n") + 1
            offenders.append((line_no, rest[:60]))
        assert not offenders, f"unescaped interpolations: {offenders}"

    def test_escaped_key_interpolations(self) -> None:
        # Row identifiers and code cells must always go through escapeHtml.
        assert "escapeHtml(key)" in self.source
        assert "escapeHtml(id)" in self.source


class TestSecretsI18n:
    """Every key used by the secrets tab exists in all three dictionaries."""

    @pytest.fixture(autouse=True)
    def _load_source(self) -> None:
        self.source = _read(_SECRETS_JS)

    def test_used_keys_exist_in_all_locales(self) -> None:
        used = set(re.findall(r"\bt\(\s*\"([^\"]+)\"", self.source))
        assert used, "settings-secrets.js must use t() for its wording"
        for locale in _LOCALES:
            dict_keys = set(_translations(locale))
            missing = sorted(used - dict_keys)
            assert not missing, f"{locale}.json is missing keys: {missing}"

    def test_secrets_key_sets_match(self) -> None:
        ja = {k for k in _translations("ja") if k.startswith("secrets.")}
        assert "settings.tab_secrets" in _translations("ja")
        for locale in _LOCALES:
            other = {k for k in _translations(locale) if k.startswith("secrets.")}
            assert ja == other, f"{locale}.json secrets.* keys differ from ja.json"
            assert "settings.tab_secrets" in _translations(locale)
            assert "settings.api_auth.secrets_link" in _translations(locale)


class TestSecretsCss:
    """settings.css must stay balanced and contain the secrets block."""

    @pytest.fixture(autouse=True)
    def _load_source(self) -> None:
        self.source = _read(_SETTINGS_CSS)

    def test_braces_balanced(self) -> None:
        assert self.source.count("{") == self.source.count("}")

    def test_secrets_classes_present(self) -> None:
        for selector in (".secrets-table-wrap", ".secrets-badge-op", ".secrets-status-set"):
            assert selector in self.source

    def test_table_wrapper_scrolls_horizontally(self) -> None:
        assert "overflow-x: auto" in self.source


@pytest.mark.parametrize("path", [_SETTINGS_JS, _SECRETS_JS], ids=lambda p: p.name)
def test_page_modules_parse_as_es_modules(path: Path) -> None:
    """``node --check <file>`` does not parse these files as modules; feed stdin instead."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    result = subprocess.run(
        [node, "--input-type=module", "--check"],
        input=path.read_text(encoding="utf-8"),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
